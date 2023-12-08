import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Linear(nn.Module):
    def __init__(self, input_feature=256, num_classes=10):
        super().__init__()
        self.linear_1 = nn.Linear(input_feature, 128)
        self.dropout_1 = nn.Dropout(p=0.5)
        self.linear_2 = nn.Linear(128, 128)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.linear_3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out1 = F.relu(self.dropout_1(self.linear_1(x)))
        out2 = F.relu(self.dropout_2(self.linear_2(out1)))
        out = self.linear_3(out2)
        return out


class OriginModel(nn.Module):
    def __init__(self, last_layer_feature=256):
        super().__init__()
        self.linear_1 = nn.Linear(last_layer_feature, 128)
        self.linear_2 = nn.Linear(128, 128)
        self.linear_3 = nn.Linear(128, 10)

    def forward(self, x):
        out = F.relu(self.linear_1(x))
        out = F.relu(self.linear_2(out))
        out = self.linear_3(out)
        return out


class ConvModel(nn.Module):
    def __init__(
        self,
        channel,
        n_random_features,
        net_width=64,
        net_depth=3,
        net_act="relu",
        net_norm="batchnorm",
        net_pooling="avgpooling",
        im_size=(32, 32),
    ):
        super().__init__()
        # print('Building Conv Model')
        self.features, shape_feat = self._make_layers(
            channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size
        )
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = GaussianLinear(num_feat, n_random_features)

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _get_activation(self, net_act):
        if net_act == "sigmoid":
            return nn.Sigmoid()
        elif net_act == "relu":
            return nn.ReLU(inplace=True)
        elif net_act == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == "gelu":
            return nn.SiLU()
        else:
            exit("unknown activation function: %s" % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == "maxpooling":
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == "avgpooling":
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == "none":
            return None
        else:
            exit("unknown net_pooling: %s" % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == "batchnorm":
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == "layernorm":
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == "instancenorm":
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == "groupnorm":
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == "none":
            return None
        else:
            exit("unknown net_norm: %s" % net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        # if im_size[0] == 28:
        #     im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            # print(shape_feat)
            layers += [Conv2d_gaussian(in_channels, net_width, kernel_size=3, padding=1)]
            # layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding='same')]
            shape_feat[0] = net_width
            if net_norm != "none":
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != "none":
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


class Conv2d_gaussian(torch.nn.Conv2d):
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        # torch.nn.init.kaiming_normal_(self.weight, a= math.sqrt(5))
        # W has shape out, in, h, w
        torch.nn.init.normal_(
            self.weight, 0, np.sqrt(2) / np.sqrt(self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3])
        )
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            # print(fan_in)
            if fan_in != 0:
                # bound = 0 * 1 / math.sqrt(fan_in)
                # torch.nn.init.uniform_(self.bias, -bound, bound)
                # torch.nn.init.uniform_(self.bias, -bound, bound)
                torch.nn.init.normal_(self.bias, 0, 0.1)


class GaussianLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, funny=False
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(GaussianLinear, self).__init__()
        self.funny = funny
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # torch.nn.init.kaiming_normal_(self.weight, a=1 * np.sqrt(5))
        torch.nn.init.normal_(self.weight, 0, np.sqrt(2) / np.sqrt(self.in_features))
        # torch.nn.init.normal_(self.weight, 0, 3/np.sqrt(self.in_features))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            # torch.nn.init.uniform_(self.bias, -bound, bound)
            torch.nn.init.normal_(self.bias, 0, 0.1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
