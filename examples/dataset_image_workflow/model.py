from torch import nn


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
        self.features, shape_feat = self._make_layers(
            channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size
        )
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, n_random_features)

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
            raise Exception("unknown activation function: %s" % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == "maxpooling":
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == "avgpooling":
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == "none":
            return None
        else:
            raise Exception("unknown net_pooling: %s" % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
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
            raise Exception("unknown net_norm: %s" % net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding="same")]

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
