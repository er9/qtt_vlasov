import numpy as np
import quimb.tensor as qtn

from setup.configs import *
import helper_quimb as helper

""" mapping of grid points along 1 dimension Parent class
"""

class AxisMap:

    @classmethod
    def array_reshape(cls,ndarray,*shape):
        return ndarray.reshape(*shape)

    @classmethod
    def get_position_inds(cls,L,q,idx):
        """ returns physical bond indices (0,1) of MPS that corresponds to vector position idx
        """
        raise NotImplementedError

    @classmethod
    def get_inds_position(cls,q,inds_list):
        """ returns index in vector coresponding to physical bond indices specified in inds_list
        """
        raise NotImplementedError

    @classmethod
    def get_tens_order(cls, L):
        return range(L)

    @classmethod
    def transform_vector(cls, ndarray, axis=0):
        return ndarray

    @classmethod
    def inverse_transform_vector(cls, ndarray, axis=0):
        return ndarray

    @classmethod
    def transform_operator(cls, ndarray, axis1=0, axis2=1):
        return ndarray

    @classmethod
    def inverse_transform_operator(cls, ndarray, axis1=0, axis2=1):
        return ndarray

    @classmethod
    def transform_mps(cls, mps: 'MPSType', inplace=False):
        return mps

    @classmethod
    def transform_mpo(cls, mpo: 'MPOType', inplace=False):
        return mpo

    @classmethod
    def transform_tn1d(cls, tn1d: 'MPTType', inplace=False):
        return tn1d

    @classmethod
    def inverse_transform_mps(cls, mps: 'MPSType', inplace=False):
        return cls.transform_mps(mps, inplace=inplace)

    @classmethod
    def inverse_transform_mpo(cls, mpo: 'MPOType', inplace=False):
        return cls.transform_mpo(mpo, inplace=inplace)

    @classmethod
    def inverse_transform_tn1d(cls, tn1d: 'TN1Type', *site_ind_ids, inplace=False):
        return cls.transform_tn1d(tn1d, inplace=inplace)







