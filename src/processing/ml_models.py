
"""@package ml_models.py
Classes which represent machine learning models.
"""
import numpy as np

from src.utl.logging import Log as log

import src.transunet.vit_seg_configs as vit_config
from src.transunet.vit_seg_modeling import VisionTransformer, Transformer, Conv2dReLU




import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
from torchinfo import summary


from operator import sub, floordiv 

class ModelBase(nn.Module):
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose

    def _get_padding(self, kernel_size):
        padding = tuple(map(sub, kernel_size, (1,1)))
        padding = tuple(map(floordiv, padding, (2,2)))
        return padding

    def _compute_transpose_params_1d(self, kernel_size, stride=2, dilation=1):
        """
        Compute padding and output_padding for ConvTranspose2d so that
        output size = stride * input size (undo pooling).
        
        Args:
            kernel_size (int): Kernel size (or tuple for H/W).
            stride (int): Stride (default=2).
            dilation (int): Dilation (default=1).
        
        Returns:
            (padding, output_padding)
        """
        # Try output_padding values from 0 to stride-1
        for op in range(stride):
            numerator = dilation * (kernel_size - 1) + op + 1 - stride
            if numerator % 2 == 0:
                padding = numerator // 2
                if padding >= 0:
                    return padding, op

        raise ValueError("No valid padding found. Check parameters.")

    def _compute_transpose_params(self, kernel_size, stride=(2,2), dilation=(1,1) ):
        """
        Returns:
            (padding, output_padding)
        """
        # Handle tuple kernel sizes
        assert isinstance(kernel_size, tuple), f"kernel_size must be a tuple"
        assert isinstance(stride, tuple), f"stride must be a tuple"
        assert isinstance(dilation, tuple), f"dilation must be a tuple"

        n = len(kernel_size)
        assert n == len(stride), f"kernel_size and stride must be the same size"
        assert n == len(dilation), f"kernel_size and dilation must be the same size"

        p = list()
        op = list()
        for k,s,d in zip(kernel_size, stride, dilation):
            _p, _op = self._compute_transpose_params_1d(k, s, d)
            p.append(_p)
            op.append(_op)

        return tuple(p), tuple(op)

    
    def _double_convolution(self, in_channels, out_channels, kernel_size=(3,3), dilation=(1,1) ):
        """
        In the original paper implementation, the convolution operations were
        not padded but we are padding them here. This is because, we need the 
        output result size to be same as input size.
        """
        #padding = self._get_padding(kernel_size)
        #pm='zeros'

        padding = 'same'
        #pm='reflect'  # padding_mode
        pm='zeros'  # necessary for reproducibility

        #if self.verbose: 
        #    log.debug(f"padding = {padding}")

        conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode=pm, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode=pm, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return conv_op
    
    def summary(self, input_size=None):
        return summary(self,input_size=input_size)
        #print(self)


class EncoderDecoder(ModelBase):
    def __init__(self, in_channels=1, out_channels=1, n_filters_init=16, 
                 conv_kernel_size=(3,3), pool_kernel_size=(2,2), pool_stride=(2,2), conv_dilation_size=(1,1),
                 batch_normalization=False, dropout=False, verbose=False):
        super().__init__(verbose=verbose)
        self.verbose = verbose
        self.conv_kernel_size = conv_kernel_size
        self.conv_dilation_size = conv_dilation_size
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters_init = n_filters_init
        self.model = self.build_model() 
        locals_ = locals().copy() 
        locals_.pop("self")
        locals_.pop("__class__")
        self.saved_parms = {**locals_}
        self.print_params()

    def print_params(self):
        log.printMessage(f"Encoder-Decoder Parameters")
        log.info(f"\t{self.saved_parms}")
    
    def _contracting_path(self):
        """
        """
        return nn.Sequential(
            self._double_convolution(self.in_channels, 64, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size),
            nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride),

            self._double_convolution(64, 128, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size),
            nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride),

            self._double_convolution(128, 256, kernel_size=self.conv_kernel_size,dilation=self.conv_dilation_size),
            nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride),

            self._double_convolution(256, 512, kernel_size=self.conv_kernel_size,dilation=self.conv_dilation_size),
            nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride),

            self._double_convolution(512, 1024, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size),
        )

    def _expanding_path(self):
        """
        """
        transpose_stride = (2,2)
        return nn.Sequential(
            nn.ConvTranspose2d( in_channels=1024, out_channels=512, kernel_size=transpose_stride, stride=transpose_stride),
            self._double_convolution(512, 256, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size),

            nn.ConvTranspose2d( in_channels=256, out_channels=128, kernel_size=transpose_stride, stride=transpose_stride),
            self._double_convolution(128, 64, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size),

            nn.ConvTranspose2d( in_channels=64, out_channels=32, kernel_size=transpose_stride, stride=transpose_stride),
            self._double_convolution(32, 16, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size),

            nn.ConvTranspose2d( in_channels=16, out_channels=8, kernel_size=transpose_stride, stride=transpose_stride),
            self._double_convolution(8, 4, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)
            )

    def build_model(self):

        return nn.Sequential(
            self._contracting_path(),
            self._expanding_path(),
            nn.Conv2d( in_channels=4, out_channels=self.out_channels, kernel_size=1)
            )

    def forward(self, x):
        return self.model.forward(x)


class UNet(ModelBase):
    def __init__(self, in_channels=1, out_channels=1, n_filters_init=16, 
                 conv_kernel_size=(3,3), pool_kernel_size=(2,2), pool_stride=(2,2), 
                 transposed_conv_kernel_size=(2,2), conv_dilation_size=(1,1),
                 batch_normalization=False, dropout=0.2,
                 verbose=False):
        super().__init__(verbose=verbose)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters_init = n_filters_init
        self.conv_kernel_size = conv_kernel_size
        self.conv_dilation_size = conv_dilation_size
        self.transposed_conv_kernel_size = transposed_conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.batch_normalization = batch_normalization
        self.dropout = dropout

        # Unique max_pool2d instance    
        self.max_pool2d = nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)

        nf = self.n_filters_init

        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution_1 = self._double_convolution(self.in_channels, nf*1, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)
        self.down_dropout_1 = nn.Dropout(dropout)

        self.down_convolution_2 = self._double_convolution(nf*1, nf*2, kernel_size=self.conv_kernel_size, dilationm=self.conv_dilation_size)
        self.down_dropout_2 = nn.Dropout(dropout)

        self.down_convolution_3 = self._double_convolution(nf*2, nf*4, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)
        self.down_dropout_3 = nn.Dropout(dropout)

        self.down_convolution_4 = self._double_convolution(nf*4, nf*8, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)
        self.down_dropout_4 = nn.Dropout(dropout)

        self.down_convolution_5 = self._double_convolution(nf*8, nf*16, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)
        self.down_dropout_5 = nn.Dropout(dropout)

        self.down_convolution_51 = self._double_convolution(nf*16, nf*32, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)

        # Expanding path.
        transpose_stride = (2,2)

        p, op = self._compute_transpose_params(self.transposed_conv_kernel_size, stride=transpose_stride, dilation=(1,1))

        self.up_transpose_5 = nn.ConvTranspose2d( in_channels=nf*32, out_channels=nf*16, kernel_size=self.transposed_conv_kernel_size,
                                                  stride=transpose_stride, padding=p, output_padding=op)
        self.up_dropout_5 = nn.Dropout(dropout)
        self.up_convolution_5 = self._double_convolution(nf*32, nf*16, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)

        self.up_transpose_4 = nn.ConvTranspose2d( in_channels=nf*16, out_channels=nf*8, kernel_size=self.transposed_conv_kernel_size,
                                                  stride=transpose_stride, padding=p, output_padding=op)
        self.up_dropout_4 = nn.Dropout(dropout)
        self.up_convolution_4 = self._double_convolution(nf*16, nf*8, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)

        self.up_transpose_3 = nn.ConvTranspose2d( in_channels=nf*8, out_channels=nf*4, kernel_size=self.transposed_conv_kernel_size,
                                                  stride=transpose_stride, padding=p, output_padding=op)
        self.up_dropout_3 = nn.Dropout(dropout)
        self.up_convolution_3 = self._double_convolution(nf*8, nf*4, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)

        self.up_transpose_2 = nn.ConvTranspose2d( in_channels=nf*4, out_channels=nf*2, kernel_size=self.transposed_conv_kernel_size,
                                                  stride=transpose_stride, padding=p, output_padding=op)
        self.up_dropout_2 = nn.Dropout(dropout)
        self.up_convolution_2 = self._double_convolution(nf*4, nf*2, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)

        self.up_transpose_1 = nn.ConvTranspose2d( in_channels=nf*2, out_channels=nf*1, kernel_size=self.transposed_conv_kernel_size,
                                                  stride=transpose_stride, padding=p, output_padding=op)
        self.up_dropout_1 = nn.Dropout(dropout)
        self.up_convolution_1 = self._double_convolution(nf*2, nf*1, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)

        self.out_conv = nn.Conv2d( nf*1, out_channels=self.out_channels, kernel_size=(1,1))



    def forward(self, x):
        # encoding
        down_1 = self.down_convolution_1(x)
        maxpool_1 = self.max_pool2d(down_1)
        dropout_1 = self.down_dropout_1(maxpool_1)

        down_2 = self.down_convolution_2(dropout_1)
        maxpool_2 = self.max_pool2d(down_2)
        dropout_2 = self.down_dropout_2(maxpool_2)

        down_3 = self.down_convolution_3(dropout_2)
        maxpool_3 = self.max_pool2d(down_3)
        dropout_3 = self.down_dropout_3(maxpool_3)

        down_4 = self.down_convolution_4(dropout_3)
        maxpool_4 = self.max_pool2d(down_4)
        dropout_4 = self.down_dropout_4(maxpool_4)

        down_5 = self.down_convolution_5(dropout_4)
        maxpool_5 = self.max_pool2d(down_5)
        dropout_5 = self.down_dropout_5(maxpool_5)

        down_51 = self.down_convolution_51(dropout_5)

        # decoding
        up_5 = self.up_transpose_5(down_51)
        x = torch.cat([up_5, down_5], 1)
        x = self.up_dropout_5(x)
        x = self.up_convolution_5(x)

        up_4 = self.up_transpose_4(x)
        x = torch.cat([up_4, down_4], 1)
        x = self.up_dropout_4(x)
        x = self.up_convolution_4(x)

        up_3 = self.up_transpose_3(x)
        x = torch.cat([up_3, down_3], 1)
        x = self.up_dropout_3(x)
        x = self.up_convolution_3(x)

        up_2 = self.up_transpose_2(x)
        x = torch.cat([up_2, down_2], 1)
        x = self.up_dropout_2(x)
        x = self.up_convolution_2(x)

        up_1 = self.up_transpose_1(x)
        x = torch.cat([up_1, down_1], 1)
        x = self.up_dropout_1(x)
        x = self.up_convolution_1(x)

        out = self.out_conv(x)
        return out


class TransUNet_paper(VisionTransformer):
    """
      vis - Seems like a debug parameter for visualization of weights
      img_size - 
      num_classes - Nmb of classes in the classification use case. For regression must be 1.
      zero_head
      config
           - hidden_size - is used  
    """
    def __init__(self, config=None, img_size=None, num_classes=1, zero_head=False, vis=False, pretrain=False, verbose=False):
        super().__init__(config, img_size=img_size, num_classes=num_classes, zero_head=zero_head, vis=vis)
        self.config['pretrain'] = pretrain
        self.verbose = verbose

    def summary(self, input_size=None):
        return summary(self,input_size=input_size)


class TransUNet(ModelBase):
    def __init__(self, in_channels=1, out_channels=1, n_filters_init=16, 
                 conv_kernel_size=(3,3), pool_kernel_size=(2,2), pool_stride=(2,2), 
                 transposed_conv_kernel_size=(2,2), conv_dilation_size=(1,1),
                 batch_normalization=False, dropout=0.2,
                 use_transformers=True, unembed_type='paper',
                 attention_config=None, attention_in_channels=512, attention_img_size=None, vis=False,
                 use_skip_connections=True,
                 verbose=False):
        super().__init__(verbose=verbose)

        assert unembed_type == 'none' or unembed_type == "paper" or unembed_type == "linear", \
        f"unembed_type = {unembed_type} is invalid. Must be 'none' or 'paper' or 'linear'"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters_init = n_filters_init
        self.conv_kernel_size = conv_kernel_size
        self.conv_dilation_size = conv_dilation_size
        self.transposed_conv_kernel_size = transposed_conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.config = attention_config
        self.use_transformers=use_transformers
        self.unembed_type = unembed_type
        self.attention_in_channels = attention_in_channels
        self.attention_img_size = attention_img_size
        self.vis = vis
        self.use_skip_connections = use_skip_connections


        # Unique max_pool2d instance    
        self.max_pool2d = nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)

        nf = self.n_filters_init

        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution_1 = self._double_convolution(self.in_channels, nf*1, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)
        self.down_dropout_1 = nn.Dropout(dropout)

        self.down_convolution_2 = self._double_convolution(nf*1, nf*2, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)
        self.down_dropout_2 = nn.Dropout(dropout)

        self.down_convolution_3 = self._double_convolution(nf*2, nf*4, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)
        self.down_dropout_3 = nn.Dropout(dropout)

        self.down_convolution_4 = self._double_convolution(nf*4, nf*8, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)
        self.down_dropout_4 = nn.Dropout(dropout)

        self.down_convolution_5 = self._double_convolution(nf*8, nf*16, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)
        self.down_dropout_5 = nn.Dropout(dropout)

        self.down_convolution_51 = self._double_convolution(nf*16, nf*32, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)



        if self.use_transformers:
            # Attention mechanism
            self.attention = Transformer(
                self.config, 
                self.attention_img_size, 
                self.attention_in_channels, 
                self.vis
            )

            if self.unembed_type == 'linear':
                self.unembedding = nn.Conv2d(
                    in_channels=self.config.hidden_size,
                    out_channels=self.attention_in_channels,
                    kernel_size=1,
                    stride=1
                )
            elif self.unembed_type == 'paper':
                self.unembedding = Conv2dReLU(
                    self.config.hidden_size,
                    self.attention_in_channels,
                    kernel_size=3,
                    padding=1,
                    use_batchnorm=True,
                )
            else:
                self.unembedding = None

            log.info(f"TRANSUNET: Using transformers!")
        else:
            self.attention = None
            self.unembedding = None
            log.info(f"TRANSUNET: NOT using transformers!")


        # Expanding path.
        transpose_stride = (2,2)

        p, op = self._compute_transpose_params(self.transposed_conv_kernel_size, stride=transpose_stride, dilation=(1,1))

        self.up_transpose_5 = nn.ConvTranspose2d( in_channels=nf*32, out_channels=nf*16, kernel_size=self.transposed_conv_kernel_size, 
                                                  stride=transpose_stride, padding=p, output_padding=op)
        self.up_dropout_5 = nn.Dropout(dropout)
        self.up_convolution_5 = self._double_convolution(nf*32, nf*16, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)

        self.up_transpose_4 = nn.ConvTranspose2d( in_channels=nf*16, out_channels=nf*8, kernel_size=self.transposed_conv_kernel_size, 
                                                  stride=transpose_stride, padding=p, output_padding=op)
        self.up_dropout_4 = nn.Dropout(dropout)
        self.up_convolution_4 = self._double_convolution(nf*16, nf*8, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)

        self.up_transpose_3 = nn.ConvTranspose2d( in_channels=nf*8, out_channels=nf*4, kernel_size=self.transposed_conv_kernel_size,
                                                  stride=transpose_stride, padding=p, output_padding=op)
        self.up_dropout_3 = nn.Dropout(dropout)
        self.up_convolution_3 = self._double_convolution(nf*8, nf*4, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)

        self.up_transpose_2 = nn.ConvTranspose2d( in_channels=nf*4, out_channels=nf*2, kernel_size=self.transposed_conv_kernel_size,
                                                  stride=transpose_stride, padding=p, output_padding=op)
        self.up_dropout_2 = nn.Dropout(dropout)
        self.up_convolution_2 = self._double_convolution(nf*4, nf*2, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)

        self.up_transpose_1 = nn.ConvTranspose2d( in_channels=nf*2, out_channels=nf*1, kernel_size=self.transposed_conv_kernel_size,
                                                  stride=transpose_stride, padding=p, output_padding=op)
        self.up_dropout_1 = nn.Dropout(dropout)
        self.up_convolution_1 = self._double_convolution(nf*2, nf*1, kernel_size=self.conv_kernel_size, dilation=self.conv_dilation_size)

        self.out_conv = nn.Conv2d( nf*1, out_channels=self.out_channels, kernel_size=(1,1))


        # DICT FOR EASE ACCESS
        self.encoder = nn.ModuleDict({
            'down_conv_1': self.down_convolution_1,
            'down_dropout_1': self.down_dropout_1,
            'down_conv_2': self.down_convolution_2,
            'down_dropout_2': self.down_dropout_2,
            'down_conv_3': self.down_convolution_3,
            'down_dropout_3': self.down_dropout_3,
            'down_conv_4': self.down_convolution_4,
            'down_dropout_4': self.down_dropout_4,
            'down_conv_5': self.down_convolution_5,
            'down_dropout_5': self.down_dropout_5,
            'down_conv_51': self.down_convolution_51,
        })

        self.latent = nn.ModuleDict({
            'attention': self.attention,
        })

        self.decoder = nn.ModuleDict({
            'unembedding': self.unembedding,
            'up_transpose_5': self.up_transpose_5,
            'up_dropout_5': self.up_dropout_5,
            'up_convolution_5': self.up_convolution_5,
            'up_transpose_4': self.up_transpose_4,
            'up_dropout_4': self.up_dropout_4,
            'up_convolution_4': self.up_convolution_4,
            'up_transpose_3': self.up_transpose_3,
            'up_dropout_3': self.up_dropout_3,
            'up_convolution_3': self.up_convolution_3,
            'up_transpose_2': self.up_transpose_2,
            'up_dropout_2': self.up_dropout_2,
            'up_convolution_2': self.up_convolution_2,
            'up_transpose_1': self.up_transpose_1,
            'up_dropout_1': self.up_dropout_1,
            'up_convolution_1': self.up_convolution_1,
            'out_conv' : self.out_conv,

        })

    def forward(self, x):
        # encoding
        #print(f"x.shape = {x.shape}")
        down_1 = self.down_convolution_1(x)
        maxpool_1 = self.max_pool2d(down_1)
        dropout_1 = self.down_dropout_1(maxpool_1)
        #print(f"down_1.shape = {down_1.shape}")

        down_2 = self.down_convolution_2(dropout_1)
        maxpool_2 = self.max_pool2d(down_2)
        dropout_2 = self.down_dropout_2(maxpool_2)
        #print(f"down_2.shape = {down_2.shape}")

        down_3 = self.down_convolution_3(dropout_2)
        maxpool_3 = self.max_pool2d(down_3)
        dropout_3 = self.down_dropout_3(maxpool_3)
        #print(f"down_3.shape = {down_3.shape}")

        down_4 = self.down_convolution_4(dropout_3)
        maxpool_4 = self.max_pool2d(down_4)
        dropout_4 = self.down_dropout_4(maxpool_4)
        #print(f"down_4.shape = {down_4.shape}")

        down_5 = self.down_convolution_5(dropout_4)
        maxpool_5 = self.max_pool2d(down_5)
        dropout_5 = self.down_dropout_5(maxpool_5)
        #print(f"down_5.shape = {down_5.shape}")

        down_51 = self.down_convolution_51(dropout_5)
        #print(f"down_51.shape = {down_51.shape}")

        if self.use_transformers:
            att_output, att_weights, att_features = self.attention(down_51)

            # Undo the reshape operations made by attention internally
            B, n_patch, hidden = att_output.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)

            # <<<<< NOTE >>>>>> The line above works just for patches.size = (1,1)
            h, w = self.attention_img_size #int(np.sqrt(n_patch)), int(np.sqrt(n_patch)) 
            x = att_output.permute(0, 2, 1)
            att_output = x.contiguous().view(B, hidden, h, w)

            # Undo the embedding transformation
            if self.unembedding is not None:
                att_output = self.unembedding(att_output)

            init_decoding = att_output
        else:
            init_decoding = down_51

        # decoding
        up_5 = self.up_transpose_5(init_decoding)
        #print(f"up_5.shape = {up_5.shape}")
        #print(f"down_5.shape = {down_5.shape}")
        x = torch.cat([up_5, down_5 if self.use_skip_connections else torch.zeros_like(down_5)], dim=1)
        x = self.up_dropout_5(x)
        x = self.up_convolution_5(x)
        
        up_4 = self.up_transpose_4(x)
        x = torch.cat([up_4, down_4 if self.use_skip_connections else torch.zeros_like(down_4)], dim=1)
        x = self.up_dropout_4(x)
        x = self.up_convolution_4(x)

        up_3 = self.up_transpose_3(x)
        x = torch.cat([up_3, down_3 if self.use_skip_connections else torch.zeros_like(down_3)], dim=1)
        x = self.up_dropout_3(x)
        x = self.up_convolution_3(x)

        up_2 = self.up_transpose_2(x)
        x = torch.cat([up_2, down_2 if self.use_skip_connections else torch.zeros_like(down_2)], dim=1)
        x = self.up_dropout_2(x)
        x = self.up_convolution_2(x)

        up_1 = self.up_transpose_1(x)
        x = torch.cat([up_1, down_1 if self.use_skip_connections else torch.zeros_like(down_1)], dim=1)
        x = self.up_dropout_1(x)
        x = self.up_convolution_1(x)

        out = self.out_conv(x)
        return out


class DeepTomographyUnets:
    def __init__(self, unet_highres:ModelBase=None, unet_mediumres:ModelBase=None, unet_lowres:ModelBase=None, verbose=False):
        self.verbose = verbose
        self.unet_highres = unet_highres
        self.unet_mediumres = unet_mediumres
        self.unet_lowres = unet_lowres
    
    def get_highres_net(self):
        return self.unet_highres
    
    def get_mediumres_net(self):
        return self.unet_mediumres
    
    def get_lowres_net(self):
        return self.unet_lowres

