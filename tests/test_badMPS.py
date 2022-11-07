import sys
sys.path.append('../')

from setup.configs import *
import helper_quimb as helper
# from enums import *

import numpy as np
import matplotlib.pyplot as plt
import findiff

from coord.coord_sys import Coordinate
from axis import Axis

"""
what does data look like if we have rainbow MPS state?
"""

q = 2
L = 6
npts = q ** L

# TEST_LAYOUT = LayoutType.PARALLEL_GROUP
TEST_LAYOUT = LayoutType.COMB
TEST_FD = FDType.CENTER


### in 1D
tot_mps = None
for i in range(L//2):

    ## Bell state (00 + 11)/sqrt(2)
    b1 = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    b2 = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    iden2 = np.array([[[1.0, 0.0],[0.0,1.0]],[[1.0, 0.0],[0.0,1.0]]])
    iden1 = np.array([[[1.0, 1.0]]])

    print('b1', b1, b1.shape)
    print('b2', b2, b2.shape)
    # print('i2', iden2, iden2.shape)
    # print('i1', iden1, iden1.shape)
    # exit()

    tens_list = [iden1] * i + [b1] + [iden2] * (L-(2*i+2)) + [b2] + [iden1] * i
    tens_list[0] = tens_list[0][0,:,:]
    tens_list[-1] = tens_list[-1][:,0, :]

    mps = qtn.MatrixProductState(tens_list, shape='lrp')
    print('mps', mps)

    if tot_mps is not None:
        helper.add_MPS(tot_mps, mps, inplace=True)
    else:
        tot_mps = mps

print('tot mps', tot_mps)

print('tot mps', tot_mps.max_bond())
helper.compress(tot_mps)
print('tot mps', tot_mps.max_bond())

X = Coordinate('X')
axis1 = Axis(L, q, coordinate=X)

mps_data = axis1.map_mps_to_state(tot_mps)

plt.figure()
plt.plot(mps_data)
plt.show()
