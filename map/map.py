import numpy as np
import quimb.tensor as qtn

from map import TNMap
import helper_quimb as helper

''' mapping of grid points along 1 dimension Parent class
'''

class TNMap():

    @classmethod
    def array_reshape(cls,ndarray,*shape):
        return ndarray.reshape(*shape)

    @classmethod
    def get_position_inds(cls,L,q,idx):
        ''' returns physical bond indices (0,1) of MPS that corresponds to vector position idx
        '''
        raise NotImplementedError

    @classmethod
    def get_inds_position(cls,L,q,inds_list):
        ''' returns index in vector coresponding to physical bond indices specified in inds_list
        '''
        raise NotImplementedError

    #################
    ## derivatives ##
    #################

    @classmethod
    def _mpo_firstderivative_9pt(cls,L,q,boundary_condition,order=1,**compress_opts):
        raise NotImplementedError

    @classmethod
    def _mpo_firstderivative_forward(cls,L,q,boundary_condition,order=1,**compress_opts):
        raise NotImplementedError

    @classmethod
    def _mpo_firstderivative_backward(cls,L,q,boundary_condition,order=1,**compress_opts):
        raise NotImplementedError

    @classmethod
    def _mpo_secondderivative_9pt(cls,L,q,boundary_condition,order=1,**compress_opts):
        raise NotImplementedError

    @classmethod
    def _mpo_heaviside(cls,L,q,zero_ind,get='right'):
        raise NotImplementedError





