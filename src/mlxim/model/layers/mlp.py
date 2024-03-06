# """ MLP module w/ dropout and configurable activation layer

# Hacked together by / Copyright 2020 Ross Wightman
# """
# from functools import partial

# import mlx.nn as nn

# from .utils import to_2tuple


# class Mlp(nn.Module):
#     """MLP as used in Vision Transformer, MLP-Mixer and related networks

#     Args:
#         in_features (int): number of input features
#         hidden_features (int, optional): number of features in the hidden layer, defaults to in_features * 4
#         out_features (int, optional): number of output features, defaults to in_features
#         act_layer (nn.Module, optional): activation layer, defaults to GELU
#         norm_layer (nn.Module, optional): normalization layer, defaults to None
#         bias (bool, optional): whether to include bias in the linear layers, defaults to True
#         drop (float | tuple, optional): dropout rate, defaults to 0.
#         use_conv (bool, optional): whether to use a 1x1 convolution in place of a linear layer, defaults to False
#     """

#     def __init__(
#         self,
#         in_features,
#         hidden_features=None,
#         out_features=None,
#         act_layer=nn.GELU,
#         norm_layer=None,
#         bias=True,
#         drop=0.0,
#         use_conv=False,
#     ):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         bias = to_2tuple(bias)
#         drop_probs = to_2tuple(drop)
#         linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

#         self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
#         self.act = act_layer()
#         self.drop1 = nn.Dropout(drop_probs[0])
#         self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
#         self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
#         self.drop2 = nn.Dropout(drop_probs[1])

#     def __call__(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop1(x)
#         x = self.norm(x)
#         x = self.fc2(x)
#         x = self.drop2(x)
#         return x
