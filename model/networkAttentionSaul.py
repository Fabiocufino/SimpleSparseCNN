import torch
import torch.nn as nn
from torch.optim import SGD
import MinkowskiEngine as ME
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch.nn.utils.rnn import pad_sequence


from .utils import (
    Block,
    LayerNorm,
    MinkowskiLayerNorm,
    MinkowskiGRN,
    MinkowskiDropPath
)

from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiConvolutionTranspose,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
    MinkowskiGlobalMaxPooling,
    MinkowskiReLU,
    MinkowskiGELU,
    MinkowskiBatchNorm
)


# Custom weight initialization function
def _init_weights(m):
    if isinstance(m, MinkowskiConvolution):
        trunc_normal_(m.kernel, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, MinkowskiConvolutionTranspose):
        trunc_normal_(m.kernel, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, MinkowskiDepthwiseConvolution):
        trunc_normal_(m.kernel, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, MinkowskiLinear):
        trunc_normal_(m.linear.weight, std=.02)
        if m.linear.bias is not None:
            nn.init.constant_(m.linear.bias, 0)
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


'''MODEL'''
class RelPosSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead = nhead
        self.scale = (d_model // nhead) ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model)
        
        self.coord_dim = 2  # Match the dimension of your coordinates
        self.bias_mlp = nn.Sequential(
            nn.Linear(self.coord_dim, nhead*4),
            nn.GELU(), 
            nn.Linear(nhead*4, nhead)
        )
        
        # For debugging
        self.attn_weights = None
    
    def forward(self, x, coords, src_key_padding_mask=None):
        B, N, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, self.nhead, -1).transpose(1, 2) for t in qkv]
        
        # coords: [B,N,3] as (x,y,z)
        ci, cj = coords.unsqueeze(2), coords.unsqueeze(1)  # [B,N,1,3], [B,1,N,3]
        dpos = ci - cj  # [B,N,N,3]
        bias = self.bias_mlp(dpos).permute(0, 3, 1, 2)  # [B,heads,N,N]
        
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        logits = dots + bias
        
        if src_key_padding_mask is not None:
            key_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            logits = logits.masked_fill(key_mask, float('-inf'))
        
        attn = torch.softmax(logits, dim=-1)
        self.attn_weights = attn  # Store for debugging
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out(out)


class RelPosEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = RelPosSelfAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, d_model*4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model*4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) 
        self.act = nn.ReLU()
    
    def forward(self, src, coords, src_key_padding_mask=None):
        src2 = self.self_attn(src, coords, src_key_padding_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout(src2)
        return self.norm2(src)


class SparseRelPosAttentionAdapter(nn.Module):
    """
    Adapter class that converts SparseTensor to dense tensors for RelPosSelfAttention
    and converts the result back to SparseTensor
    """
    def __init__(self, d_model, nhead=1, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([                     
            RelPosEncoderLayer(d_model, nhead, dropout),
            # RelPosEncoderLayer(d_model, nhead, dropout),
        ])
    
    def forward(self, x: ME.SparseTensor):
        features = x.F
        coordinates = x.C  # Get coordinates including batch index
        batch_indices = coordinates[:, 0]  # Extract batch indices
        
        # Group features by batch
        unique_batches = batch_indices.unique(sorted=True)
        grouped_features = [features[batch_indices == b] for b in unique_batches]
        grouped_coords = [coordinates[batch_indices == b, 1:].float() for b in unique_batches]  # Spatial coords only
        
        # Get batch sizes
        batch_sizes = [f.shape[0] for f in grouped_features]
        
        # Create padding mask for attention
        max_len = max(batch_sizes)
        B = len(unique_batches)
        key_padding_mask = torch.zeros((B, max_len), dtype=torch.bool, device=features.device)
        for i, size in enumerate(batch_sizes):
            key_padding_mask[i, size:] = True
        
        # Pad sequences for batch processing
        padded_features = pad_sequence(grouped_features, batch_first=True)  # [B, N_max, D]
        padded_coords = pad_sequence(grouped_coords, batch_first=True)  # [B, N_max, 3]
        
        # Process through the layers
        out = padded_features
        for layer in self.layers:
            out = layer(out, padded_coords, key_padding_mask)
        
        # Unpad to recover sparse structure
        unpadded = torch.cat([out[i, :size] for i, size in enumerate(batch_sizes)], dim=0)
        
        # Return as SparseTensor with original coordinate structure
        return ME.SparseTensor(
            features=unpadded,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )


class MinkEncClsConvNeXtV2(nn.Module):
    def __init__(self, in_channels, out_channels, D=2, args=None): # D=2 important (2D images here)
        nn.Module.__init__(self)
        self.is_v5 = True if 'v5' in args.dataset_path else False 

        """Encoder"""
        depths=[3, 3, 9, 3]
        dims=[96, 192, 384, 768]     
        kernel_size = 5
        drop_path_rate=0.

        assert len(depths) == len(dims)

        self.nb_elayers = len(dims)

        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # stem
        self.stem = MinkowskiConvolution(in_channels, dims[0], kernel_size=1, stride=1, dimension=D)
        self.stem_ln = MinkowskiLayerNorm(dims[0], eps=1e-6)

        # Linear transformation for global features
        self.global_mlp = nn.Sequential(
            nn.Linear(1 + 1 + 1 + 1, dims[0]), #NO MODULES, easier
        ) 
 
        for i in range(self.nb_elayers):
            encoder_layer = nn.Sequential(
                *[Block(dim=dims[i], kernel_size=kernel_size, drop_path=dp_rates[cur + j], D=D) for j in range(depths[i])]
            )
            self.encoder_layers.append(encoder_layer)
            cur += depths[i]

            if i < self.nb_elayers - 1:  
                downsample_layer = nn.Sequential(
                    MinkowskiLayerNorm(dims[i], eps=1e-6),                
                    MinkowskiConvolution(dims[i], dims[i+1], kernel_size=2, stride=2, bias=True, dimension=D),
                )
                self.downsample_layers.append(downsample_layer)
        
        """NEW: Layer Normalization"""
        self.layer_norm_ba = MinkowskiLayerNorm(dims[-1], eps=1e-6)

        """Offset Attention module"""
        self.offset_attn = SparseRelPosAttentionAdapter(d_model=dims[-1])


        """Classification/regression layers"""
        self.global_pool = MinkowskiGlobalMaxPooling()
        self.flavour_layer = nn.Sequential(
            MinkowskiLinear(dims[-1], dims[-1]),
            MinkowskiGELU(),
            MinkowskiLinear(dims[-1], 4)
        ) 

        """ Initialise weights """
        self.apply(_init_weights)

    def forward(self, x, x_glob):
        """Encoder"""
        # Ensure x_glob is on the same device as x
        x_glob = x_glob.to(x.device)  

        # stem
        x = self.stem(x)
        
        x_glob = self.global_mlp(x_glob)
        
        # add global to voxel features
        batch_indices = x.C[:, 0].long()  # batch idx
        x_glob_expanded = x_glob[batch_indices]
        new_feats = x.F + x_glob_expanded
        x = ME.SparseTensor(features=new_feats, coordinates=x.C)
        
        x = self.stem_ln(x)

        # encoder layers
        x_enc = []
        for i in range(self.nb_elayers):
            x = self.encoder_layers[i](x)
            if i < self.nb_elayers - 1:
                x_enc.append(x)
                x = self.downsample_layers[i](x)

        # Layer Nomr
        x = self.layer_norm_ba(x)
        
        # offset attention
        x = self.offset_attn(x)
        
        # event predictions
        x_pooled = self.global_pool(x)
        out_flavour = self.flavour_layer(x_pooled)
        output = {"out_flavour": out_flavour.F}
        
        return output

    def replace_depthwise_with_channelwise(self):
        for name, module in self.named_modules():
            if isinstance(module, ME.MinkowskiDepthwiseConvolution):
                # Get the parameters of the current depthwise convolution
                in_channels = module.in_channels
                kernel_size = module.kernel_generator.kernel_size
                stride = module.kernel_generator.kernel_stride
                dilation = module.kernel_generator.kernel_dilation
                bias = module.bias is not None
                dimension = module.dimension
                
                # Create a new MinkowskiChannelwiseConvolution with the same parameters
                new_conv = ME.MinkowskiChannelwiseConvolution(
                    in_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    bias=bias,
                    dimension=dimension
                )
                new_conv.kernel = module.kernel
                if bias:
                    new_conv.bias = module.bias
                
                # Replace the old depthwise convolution with the new channelwise convolution
                parent_module, attr_name = self._get_parent_module(name)
                setattr(parent_module, attr_name, new_conv)
        
        return
    
    def _get_parent_module(self, layer_name):
        components = layer_name.split('.')
        parent = self
        for comp in components[:-1]:
            parent = getattr(parent, comp)
        return parent, components[-1]