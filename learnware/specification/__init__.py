from .base import Specification, BaseStatSpecification
from .regular import (
    RegularStatsSpecification,
    RKMEStatSpecification,
    RKMETableSpecification,
    RKMEImageSpecification,
    RKMETextSpecification,
)
from ..utils import is_torch_avaliable

if not is_torch_avaliable(verbose=False):
    generate_stat_spec = None
    generate_rkme_spec = None
    generate_rkme_image_spec = None
    generate_rkme_text_spec = None
else:
    from .module import generate_stat_spec, generate_rkme_spec, generate_rkme_image_spec, generate_rkme_text_spec
