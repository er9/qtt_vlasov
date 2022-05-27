from enum import Enum, IntEnum

class DataType(Enum):
    MPS = 'MPS'
    MPX = 'MPX'
    MPO = 'MPO'
    TN3 = 'TN3'

class LayoutType(Enum):
    PARALLEL_GROUP = 'PG'  # 'parallelG'
    PARALLEL = 'PF'   # 'parallelF'
    SEQUENTIAL = 'SF' # 'sequentialF'
    COMB = 'COMB'

class AxisMapType(Enum):
    BINARY = '0'
    FLIPPED = '1'
    SIGNED = '2'

class BCType(Enum):
    ANTIPERIODIC = -1
    PERIODIC = 1
    OPEN = 0
    ZEROGRADIENT = 11
    ZEROVALUE = 12
    ABSORBING = 21
    REFLECTING = 22

class FDType(Enum):
    CENTER = 'center'
    FORWARD = 'forward'
    BACKWARD = 'backward'
    # UPWIND = 'upwind'

class BasisType(Enum):
    SPATIAL = 'RealSpace'
    FOURIER = 'Fourier'
    HERMITE = 'Hermite'

class CoordinateSystemType(Enum):
    CARTESIAN = 'cartesian'
    CYLINDRICAL = 'cylindrical'
    SPHERICAL = 'spherical'

class CoordinateType(IntEnum):    # is recognized as an int
    X = 0
    Y = 1
    Z = 2
    R = 0
    THETA = 1
    PHI = 2
    SCALAR = -1

class CollisionType(Enum):
    LB = 'Lenard-Bernstein'
    H6 = 'hyper6'


def parse_BCType(bc: BCType or str) -> BCType:
    if isinstance(bc, BCType):
        return bc
    else:
        if bc in ['zg', 'ZG', 'zero gradient']:
            return BCType.ZEROGRADIENT
        if bc in ['zero value', 'zero', 'z', 'Z']:
            return BCType.ZEROVALUE
        if bc in ['open','o','O']:
            return BCType.OPEN
        if bc in ['p','periodic','P']:
            return BCType.PERIODIC
        if bc in ['ap','antiperiodic','AP']:
            return BCType.ANTIPERIODIC
        if bc in ['r','reflecting','R']:
            return BCType.REFLECTING
        if bc in ['a','absorbing','A']:
            return BCType.ABSORBING

def parse_FDType(fd_type: FDType or str) -> FDType:
    if isinstance(fd_type, FDType):
        return fd_type
    else:
        if fd_type in ['center', 'C', 'c']:
            return FDType.CENTER
        if fd_type in ['forward','F','f']:
            return FDType.FORWARD
        if fd_type in ['backward','B','b']:
            return FDType.BACKWARD
        # if fd_type in ['upwind','up','UP','U','u']:
        #     return FDType.UPWIND

# def get_axis_map(map_type: AxisMapType) -> 'AxisMap':
#     if map_type is AxisMapType.BINARY:
#         return BinaryMap()
#     elif map_type is AxisMapType.FLIPPED:
#         return FlipBinaryMap()
#     elif map_type is AxisMapType.SIGNED:
#         return MirrorMap()

