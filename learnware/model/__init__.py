from .base import BaseModel
from ..utils import is_torch_available

if not is_torch_available(verbose=False):
    TorchModel = None
else:
    from .torch_model import TorchModel

__all__ = ["BaseModel", "TorchModel"]
