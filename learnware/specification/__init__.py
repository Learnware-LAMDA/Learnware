from .base import Specification, BaseStatSpecification
from .regular import (
    RegularStatSpecification,
    RKMEStatSpecification,
    RKMETableSpecification,
    RKMEImageSpecification,
    RKMETextSpecification,
    rkme_solve_qp,
)

from .system import HeteroMapTableSpecification

from ..utils import is_torch_available

if not is_torch_available(verbose=False):
    generate_stat_spec = None
    generate_rkme_table_spec = None
    generate_rkme_image_spec = None
    generate_rkme_text_spec = None
else:
    from .module import generate_stat_spec, generate_rkme_table_spec, generate_rkme_image_spec, generate_rkme_text_spec
