import helper_quimb as helper
from setup.configs import *
from axis_map.map import AxisMap

""" mapping of grid points along 1 dimension to binary TN ordering
    reverse tensor ordering of map_binary
    MPS left to right:  fine to coarse grid
"""

## binary TN class
class FlipBinaryMap(AxisMap):

    @classmethod
    def get_position_inds(cls,L,q,idx):
        """ returns physical bond indices (0,1) of MPS that corresponds to vector position idx
        """
        if L == 1:
            return [idx]

        assert (q==2), 'atm method only implemented for q=2'

        if q==2:   str_b = bin(idx)[2:]
        else:
            if idx == 0:  str_b = [0]
            else:
                str_b = []
                while idx:
                    str_b.append(int(idx % q))
                    idx //= q
                str_b = str_b[::-1]

        if len(str_b) > L:
            print('ind not in binaryTN of len L',L)
            raise ValueError
        elif len(str_b) < L:
            str_b = '0'*(L-len(str_b)) + str_b

        ind_list = [int(s) for s in str_b][::-1]
        # print('flip axis_map', idx, ind_list)
        return ind_list

    @classmethod
    def get_inds_position(cls, q, inds_list):
        """ returns index in vector corresponding to physical bond indices specified in inds_list
        """
        bstring = ''
        for b in inds_list:   bstring += str(b)
        i = bstring[::-1]
        idx = int(i,q)
        return idx

    @classmethod
    def get_tens_order(cls, L):
        return range(L-1, -1, -1)


    ### transform data and 1D TNs ###

    @classmethod
    def transform_mps(cls, mps: 'MPSType', inplace=False):
        return helper.mps_flip_lr(mps, inplace=inplace)

    @classmethod
    def transform_mpo(cls, mpo: 'MPOType', inplace=False):
        return helper.mpo_flip_lr(mpo, inplace=inplace)

    @classmethod
    def transform_tn1d(cls, tn1d: 'MPTType', inplace=False):
        tn1d = tn1d if inplace else tn1d.copy()
        L = tn1d.L
        site_ind_ids = (tn1d.upper_ind_id, tn1d.lower_ind_id) + tn1d.extra_ind_ids
        for site_ind_id in site_ind_ids:
            ind_map = {site_ind_id.format(i): site_ind_id.format(L - 1 - i) for i in range(L)}
            tn1d.reindex(ind_map, inplace=True)
        tag_map = {tn1d.site_tag_id.format(i): tn1d.site_tag_id.format(L - 1 - i) for i in range(L)}
        tn1d.retag(tag_map, inplace=True)
        return tn1d
