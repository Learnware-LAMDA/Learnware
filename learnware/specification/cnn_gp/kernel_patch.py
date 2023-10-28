__all__ = ('ConvKP', 'NonlinKP')


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
        return self._do_elementwise(other, '__add__')

    def __mul__(self, other):
        return self._do_elementwise(other, '__mul__')

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
                getattr(self.yy, op)(other.yy)
            )
        else:
            return KP(
                self.same,
                self.diag,
                getattr(self.xy, op)(other),
                getattr(self.xx, op)(other),
                getattr(self.yy, op)(other)
            )


class ConvKP(KernelPatch):
    def init(self, same, diag, xy, xx, yy):
        self.same = same
        self.diag = diag
        if diag:
            self.xy = xy.view(self.Nx,         1, self.W, self.H)
        else:
            self.xy = xy.view(self.Nx*self.Ny, 1, self.W, self.H)
        self.xx = xx.view(self.Nx,             1, self.W, self.H)
        self.yy = yy.view(self.Ny,             1, self.W, self.H)


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
            self.xx = xx.view(self.Nx,       1, self.W, self.H)
            self.yy =         yy.view( self.Ny, self.W, self.H)
