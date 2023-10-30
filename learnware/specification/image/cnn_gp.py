import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


__all__ = ("NNGPKernel", "Conv2d", "ReLU", "Sequential", "ConvKP", "NonlinKP")
"""
With this package, we are able to accurately and efficiently compute the kernel matrix corresponding to the NNGP during the search phase.

Github Repository: https://github.com/cambridge-mlg/cnn-gp 

References: [1] A. Garriga-Alonso, L. Aitchison, and C. E. Rasmussen. Deep Convolutional Networks as shallow Gaussian Processes. In: International Conference on Learning Representations (ICLR'19), 2019.
"""


class NNGPKernel(nn.Module):
    """
    Transforms one kernel matrix into another.
    [N1, N2, W, H] -> [N1, N2, W, H]
    """

    def forward(self, x, y=None, same=None, diag=False):
        """
        Either takes one minibatch (x), or takes two minibatches (x and y), and
        a boolean indicating whether they're the same.
        """
        if y is None:
            assert same is None
            y = x
            same = True

        assert not diag or len(x) == len(y), "diagonal kernels must operate with data of equal length"

        assert 4 == len(x.size())
        assert 4 == len(y.size())
        assert x.size(1) == y.size(1)
        assert x.size(2) == y.size(2)
        assert x.size(3) == y.size(3)

        N1 = x.size(0)
        N2 = y.size(0)
        C = x.size(1)
        W = x.size(2)
        H = x.size(3)

        # [N1, C, W, H], [N2, C, W, H] -> [N1 N2, 1, W, H]
        if diag:
            xy = (x * y).mean(1, keepdim=True)
        else:
            xy = (x.unsqueeze(1) * y).mean(2).view(N1 * N2, 1, W, H)
        xx = (x**2).mean(1, keepdim=True)
        yy = (y**2).mean(1, keepdim=True)

        initial_kp = ConvKP(same, diag, xy, xx, yy)
        final_kp = self.propagate(initial_kp)
        r = NonlinKP(final_kp).xy
        if diag:
            return r.view(N1)
        else:
            return r.view(N1, N2)


class Conv2d(NNGPKernel):
    def __init__(
        self,
        kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        var_weight=1.0,
        var_bias=0.0,
        in_channel_multiplier=1,
        out_channel_multiplier=1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.var_weight = var_weight
        self.var_bias = var_bias
        self.kernel_has_row_of_zeros = False
        if padding == "same":
            self.padding = dilation * (kernel_size // 2)
            if kernel_size % 2 == 0:
                self.kernel_has_row_of_zeros = True
        else:
            self.padding = padding

        if self.kernel_has_row_of_zeros:
            # We need to pad one side larger than the other. We just make a
            # kernel that is slightly too large and make its last column and
            # row zeros.
            kernel = t.ones(1, 1, self.kernel_size + 1, self.kernel_size + 1)
            kernel[:, :, 0, :] = 0.0
            kernel[:, :, :, 0] = 0.0
        else:
            kernel = t.ones(1, 1, self.kernel_size, self.kernel_size)
        self.register_buffer("kernel", kernel * (self.var_weight / self.kernel_size**2))
        self.in_channel_multiplier, self.out_channel_multiplier = (in_channel_multiplier, out_channel_multiplier)

    def propagate(self, kp):
        kp = ConvKP(kp)

        def f(patch):
            return (
                F.conv2d(patch, self.kernel, stride=self.stride, padding=self.padding, dilation=self.dilation)
                + self.var_bias
            )

        return ConvKP(kp.same, kp.diag, f(kp.xy), f(kp.xx), f(kp.yy))

    def nn(self, channels, in_channels=None, out_channels=None):
        if in_channels is None:
            in_channels = channels
        if out_channels is None:
            out_channels = channels
        conv2d = nn.Conv2d(
            in_channels=in_channels * self.in_channel_multiplier,
            out_channels=out_channels * self.out_channel_multiplier,
            kernel_size=self.kernel_size + (1 if self.kernel_has_row_of_zeros else 0),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=(self.var_bias > 0.0),
        )
        conv2d.weight.data.normal_(0, math.sqrt(self.var_weight / conv2d.in_channels) / self.kernel_size)
        if self.kernel_has_row_of_zeros:
            conv2d.weight.data[:, :, 0, :] = 0
            conv2d.weight.data[:, :, :, 0] = 0
        if self.var_bias > 0.0:
            conv2d.bias.data.normal_(0, math.sqrt(self.var_bias))
        return conv2d

    def layers(self):
        return 1


class ReLU(NNGPKernel):
    """
    A ReLU nonlinearity, the covariance is numerically stabilised by clamping
    values.
    """

    f32_tiny = np.finfo(np.float32).tiny

    def propagate(self, kp):
        kp = NonlinKP(kp)
        """
        We need to calculate (xy, xx, yy == c, v₁, v₂):
                      ⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤
        √(v₁v₂) / 2π ⎷1 - c²/v₁v₂ + (π - θ)c / √(v₁v₂)

        which is equivalent to:
        1/2π ( √(v₁v₂ - c²) + (π - θ)c )

        # NOTE we divide by 2 to avoid multiplying the ReLU by sqrt(2)
        """
        xx_yy = kp.xx * kp.yy + self.f32_tiny

        # Clamp these so the outputs are not NaN
        cos_theta = (kp.xy * xx_yy.rsqrt()).clamp(-1, 1)
        sin_theta = t.sqrt((xx_yy - kp.xy**2).clamp(min=0))
        theta = t.acos(cos_theta)
        xy = (sin_theta + (math.pi - theta) * kp.xy) / (2 * math.pi)

        xx = kp.xx / 2.0
        if kp.same:
            yy = xx
            if kp.diag:
                xy = xx
            else:
                # Make sure the diagonal agrees with `xx`
                eye = t.eye(xy.size()[0]).unsqueeze(-1).unsqueeze(-1).to(kp.xy.device)
                xy = (1 - eye) * xy + eye * xx
        else:
            yy = kp.yy / 2.0
        return NonlinKP(kp.same, kp.diag, xy, xx, yy)

    def nn(self, channels, in_channels=None, out_channels=None):
        assert in_channels is None
        assert out_channels is None
        return nn.ReLU()

    def layers(self):
        return 0


class Sequential(NNGPKernel):
    def __init__(self, *mods):
        super().__init__()
        self.mods = mods
        for idx, mod in enumerate(mods):
            self.add_module(str(idx), mod)

    def propagate(self, kp):
        for mod in self.mods:
            kp = mod.propagate(kp)
        return kp

    def nn(self, channels, in_channels=None, out_channels=None):
        if len(self.mods) == 0:
            return nn.Sequential()
        elif len(self.mods) == 1:
            return self.mods[0].nn(channels, in_channels=in_channels, out_channels=out_channels)
        else:
            return nn.Sequential(
                self.mods[0].nn(channels, in_channels=in_channels),
                *[mod.nn(channels) for mod in self.mods[1:-1]],
                self.mods[-1].nn(channels, out_channels=out_channels)
            )

    def layers(self):
        return sum(mod.layers() for mod in self.mods)


class KernelPatch:
    """
    Represents a block of the kernel matrix.
    Critically, we need the variances of the rows and columns, even if the
    diagonal isn't part of the block, and this introduces considerable
    complexity.
    In particular, we also need to know whether the
    rows and columns of the matrix correspond, in which case, we need to do
    something different when we add IID noise.
    """

    def __init__(self, same_or_kp, diag=False, xy=None, xx=None, yy=None):
        if isinstance(same_or_kp, KernelPatch):
            same = same_or_kp.same
            diag = same_or_kp.diag
            xy = same_or_kp.xy
            xx = same_or_kp.xx
            yy = same_or_kp.yy
        else:
            same = same_or_kp

        self.Nx = xx.size(0)
        self.Ny = yy.size(0)
        self.W = xy.size(-2)
        self.H = xy.size(-1)

        self.init(same, diag, xy, xx, yy)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        return self._do_elementwise(other, "__add__")

    def __mul__(self, other):
        return self._do_elementwise(other, "__mul__")

    def _do_elementwise(self, other, op):
        KP = type(self)
        if isinstance(other, KernelPatch):
            other = KP(other)
            assert self.same == other.same
            assert self.diag == other.diag
            return KP(
                self.same,
                self.diag,
                getattr(self.xy, op)(other.xy),
                getattr(self.xx, op)(other.xx),
                getattr(self.yy, op)(other.yy),
            )
        else:
            return KP(
                self.same,
                self.diag,
                getattr(self.xy, op)(other),
                getattr(self.xx, op)(other),
                getattr(self.yy, op)(other),
            )


class ConvKP(KernelPatch):
    def init(self, same, diag, xy, xx, yy):
        self.same = same
        self.diag = diag
        if diag:
            self.xy = xy.view(self.Nx, 1, self.W, self.H)
        else:
            self.xy = xy.view(self.Nx * self.Ny, 1, self.W, self.H)
        self.xx = xx.view(self.Nx, 1, self.W, self.H)
        self.yy = yy.view(self.Ny, 1, self.W, self.H)


class NonlinKP(KernelPatch):
    def init(self, same, diag, xy, xx, yy):
        self.same = same
        self.diag = diag
        if diag:
            self.xy = xy.view(self.Nx, 1, self.W, self.H)
            self.xx = xx.view(self.Nx, 1, self.W, self.H)
            self.yy = yy.view(self.Ny, 1, self.W, self.H)
        else:
            self.xy = xy.view(self.Nx, self.Ny, self.W, self.H)
            self.xx = xx.view(self.Nx, 1, self.W, self.H)
            self.yy = yy.view(self.Ny, self.W, self.H)
