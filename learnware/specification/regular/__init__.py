from .base import RegularStatsSpecification
from ...utils import is_torch_avaliable

from .text import RKMETextSpecification
from .table import RKMETableSpecification, RKMEStatSpecification, rkme_solve_qp
from .image import RKMEImageSpecification
