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
class OffsetAttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        self.layer_nm = MinkowskiLayerNorm(embed_dim, eps=1e-6)
        self.relu = MinkowskiReLU()

        # Simpler positional encoding: 1 linear layer from (x, z) → embed_dim
        self.pos_encoder = nn.Linear(2, embed_dim)

    def forward(self, x):
        features = x.F  # [N, C]
        coords = x.C[:, 1:].float()  # [N, 2] for (x, z)

        # Positional encoding
        pos_enc = self.pos_encoder(coords)  # [N, embed_dim]
        features = features + pos_enc
        residual = features.clone()

        # Group features by batch
        batch_ids = x.C[:, 0]
        unique_batches = batch_ids.unique()
        grouped_features = [features[batch_ids == b] for b in unique_batches]

        # Pad sequences
        padded = pad_sequence(grouped_features, batch_first=True)  # [B, max_len, C]
        lengths = torch.tensor([f.shape[0] for f in grouped_features], device=features.device)
        key_padding_mask = torch.arange(padded.size(1), device=features.device)[None, :] >= lengths[:, None]

        # Self-attention
        attn_out, _ = self.attn(padded, padded, padded, key_padding_mask=key_padding_mask)

        # Remove padding
        new_features = []
        for i, f in enumerate(grouped_features):
            length = f.shape[0]
            new_features.append(attn_out[i, :length])

        new_features = torch.cat(new_features, dim=0)

        # Build new sparse tensor
        new_features = self.layer_nm(ME.SparseTensor(features=new_features,
                                                     coordinate_map_key=x.coordinate_map_key,
                                                     coordinate_manager=x.coordinate_manager))
        new_features = self.relu(new_features + residual)
        return new_features


class OffsetAttentionModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.layer1 = OffsetAttentionLayer(embed_dim)
        self.layer2 = OffsetAttentionLayer(embed_dim)
        self.layer3 = OffsetAttentionLayer(embed_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class MinkEncClsConvNeXtV2(nn.Module):
    def __init__(self, in_channels, out_channels, D=2, args=None): #D=2 important
        nn.Module.__init__(self)
        self.is_v5 = True if 'v5' in args.dataset_path else False 

        """Encoder"""
        #depths=[2, 4, 4, 8, 8, 8]
        #dims = (16, 32, 64, 128, 256, 512)
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

        # Linear transformation for glibal features
        self.global_mlp = nn.Sequential(
            nn.Linear(1 + 1 + 1 + 1, dims[0]),  # here chagned
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


        """Offset Attention module"""
        self.offset_attn = OffsetAttentionModule(embed_dim=dims[-1])

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

        # Check the shape of x_glob before passing it to the linear layer
        
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




