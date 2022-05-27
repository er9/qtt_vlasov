from typing import Union, Optional, Sequence, Any, Iterable
from setup.enums import *

import numpy as np
import matplotlib.pyplot as plt
import quimb.tensor as qtn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axis import Axis


Numeric = Union[int, float, complex, np.number]

MPSType = Optional[qtn.MatrixProductState]
MPOType = Optional[qtn.MatrixProductOperator]
TN1Type = Optional[qtn.TensorNetwork1D]
TNType = Union[Optional[qtn.TensorNetwork], TN1Type, MPOType, MPSType]
Iter = list or tuple # Union[list, tuple]

CUTOFF = 1.0e-20
CUTOFF_MODE = 'rsum2'
MAXBOND = None

DEFAULT_BC = BCType.ZEROGRADIENT
DEFAULT_ORDER = 1
DEFAULT_FDTYPE = FDType.CENTER
