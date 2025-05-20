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

import math    

'''MODEL'''
class OffsetAttentionLayer(nn.Module):
    def __init__(self, embed_dim, max_seq_len=200):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        # Initialize with truncated normal distribution (ViT paper)
        trunc_normal_(self.pos_embed.weight, std=0.02)
        
        self.norm = nn.BatchNorm1d(embed_dim)
        self.relu = nn.ReLU()
        
        # for interpretability/debugging
        self.attn_weights = None  
        self.input_tokens = None
        self.pos_embeddings = None

    def get_positional_embeddings(self, batch_indices):
        """
        Returns learned positional embeddings using ViT-style embedding approach.
        
        Args:
            batch_indices: Tensor of shape [N_total] containing batch/event indices.
            
        Returns:
            Tensor of shape [N_total, embed_dim] with positional embeddings.
        """
        device = batch_indices.device
        
        # Get unique batch indices and count points per batch
        unique_batches, counts = torch.unique(batch_indices, return_counts=True)
        
        # Create position IDs for each point 
        position_ids = []
        for b, count in zip(unique_batches, counts):
            position_ids.append(torch.arange(count, device=device))
        
        # Concatenate to match original tensor shape
        position_ids = torch.cat(position_ids)
        
        # Look up the embeddings from the embedding table (nn.Embedding())
        pos_embeddings = self.pos_embed(position_ids)
        self.pos_embeddings = pos_embeddings
        
        return pos_embeddings  # shape: [N_total, embed_dim]


    def forward(self, x: ME.SparseTensor):
        features = x.F
        batch_indices = x.C[:, 0]

        # Add learned positional encoding based on hit order
        pos_enc = self.get_positional_embeddings(batch_indices)
        
        # Store original features for residual connection
        residual = features.clone()
        
        # Add positional encoding after normalization
        features = features + pos_enc

        # Group by batch
        unique_batches = batch_indices.unique(sorted=True)
        grouped = [features[batch_indices == b] for b in unique_batches]

        # Pad to create [B, N_max, D] for attention
        padded = pad_sequence(grouped, batch_first=True)

        lengths = torch.tensor([g.shape[0] for g in grouped], device=features.device)
        key_padding_mask = torch.arange(padded.size(1), device=features.device).unsqueeze(0) >= lengths.unsqueeze(1)

        # Apply attention
        attn_out, self.attn_weights = self.attn(
            padded, padded, padded,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        self.input_tokens = padded.detach().cpu()

        # Remove padding and restore [N_total, D]
        unpadded = torch.cat([attn_out[i, :l] for i, l in enumerate(lengths)], dim=0)
        
        # Apply BatchNorm and Relu
        att = self.norm(unpadded + residual)
        out = self.relu(att)
        
        # Return new Sparse Tensor with updated features and residual connection
        return ME.SparseTensor(
            features=out,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )


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
    def __init__(self, in_channels, out_channels, D=2, args=None): #D=2 important (2D images here)
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
        
        """NEW: Batch Normalization"""
        self.batch_norm = MinkowskiBatchNorm(dims[-1], eps=1e-6)

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

        # Barch Nomr
        x = self.batch_norm(x)
        
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