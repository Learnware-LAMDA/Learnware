from .base import RegularStatSpecification
from ...utils import is_torch_available

from .text import RKMETextSpecification
from .table import RKMETableSpecification, RKMEStatSpecification, rkme_solve_qp
from .image import RKMEImageSpecification
