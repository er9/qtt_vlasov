from typing import Union, Optional, Sequence, Any, Iterable, Type, Callable
from setup.enums import *
import sys, getopt

import numpy as np

import matplotlib
# matplotlib.use('TkAgg')     ## Qt... doesn't work after updating packages...
import matplotlib.pyplot as plt

import quimb.tensor as qtn
from setup.quimb_TN1D import MatrixProductTensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axis import Axis


Numeric = Union[int, float, complex, np.number]

MPSType = Optional[qtn.MatrixProductState]
MPOType = Optional[qtn.MatrixProductOperator]
MPTType = Optional[MatrixProductTensor]
TN1Type = Optional[qtn.TensorNetwork1D]
TNType = Union[Optional[qtn.TensorNetwork], TN1Type, MPOType, MPSType]
Iter = list or tuple # Union[list, tuple]

CUTOFF = 1.0e-20
CUTOFF_MODE = 'rsum2'
MAXBOND = None

DEFAULT_BC = BCType.SYMMETRIC   # ZEROGRADIENT
DEFAULT_ORDER = 1
DEFAULT_FDTYPE = FDType.CENTER

DEFAULT_SL_ORDER = 3

DEEP_GRID_CHECK = False     ## True for debugging


#### parameters that can be set by command line flags
flags = {}
def _set_args(argv):
    opts, args = getopt.getopt(argv, "hP:",["use_mp=",] )
    print('in func')
    print('opts', opts, args)
    for opt, arg in opts:
        print('opt', opt, 'arg', arg)
        if opt == '-h':
            print('help')
            exit()
        elif opt in ('-P', '--use_mp'):
            flags['USE_MP'] = arg

    return

init_args = sys.argv
print('init args', init_args)
if len(init_args) > 1:
    _set_args(init_args[1:])

USE_MP = flags.get('USE_MP', False)

