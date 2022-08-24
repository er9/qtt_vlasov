from setup.configs import *
import helper_quimb as helper
from basis.basis import Basis
import basis.findiff_coeffs as fd_coeff

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axis import Axis


class SpatialBasis(Basis):

    def __init__(self):
        self.type = BasisType.SPATIAL

    def get_realspace_1D(self, data, ax_ind, **kwargs):
        """ ie. coeffs of delta fcts """
        return data

    def get_realspace_nD(self, data: 'np.ndarray', ax_inds: list[int], x0: Numeric=0.0, xL: Numeric=1.0,
                         npts: Numeric=128):
        """ ie. coeffs of delta fcts """
        return data

    # @classmethod
    # def elemental_multiply(cls, gtn1, gtn2, inplace=False, **compress_opts):
    #     gtn1 = gtn1 if inplace else gtn1.copy()
    #     gtn1.elemental_multiply(gtn2, **compress_opts)
    #     return gtn1


    def get_ones_mps(self, ax: 'Axis', site_ind_id='i({})', site_tag_id='X({})') -> 'MPSType':
        q, L = ax.q, ax.L
        ones = np.ones((q,))
        mps_0 = None
        if L >= 2:
            mps_0 = qtn.MatrixProductState(
                [np.array([ones])] + [np.array([[ones]])] * (L - 2) + [np.array([ones])],
                shape='lrp', site_tag_id=site_tag_id, site_ind_id=site_ind_id)
        elif L == 1:
            mps_0 = qtn.TensorNetwork([qtn.Tensor(ones, inds=('i(0)',), tags=('X(0)',))])
            mps_0.view_as(qtn.MatrixProductState, inplace=True, L=1, cyclic=False,
                          site_tag_id=site_tag_id, site_ind_id=site_ind_id)
        elif L == 0:
            mps_0 = qtn.TensorNetwork([])
            mps_0.view_as(qtn.MatrixProductState, inplace=True, L=0, cyclic=False,
                          site_tag_id=site_tag_id, site_ind_id=site_ind_id)
        return mps_0


    def build_elemental_multiply_tn(self, ax, in1_ind_id, in2_ind_id, out_ind_id, site_tag_id):
        """ d_ijk TN
        """
        L, q = ax.L, ax.q
        out = qtn.TensorNetwork([])
        for i in range(L):
            d_ijk = qtn.tensor_core.COPY_tensor(q,(in1_ind_id.format(i), in2_ind_id.format(i), out_ind_id.format(i)),
                                                tags=(site_tag_id.format(i),))
            out.add_tensor(d_ijk)
        # out.view_as(qtn.TensorNetwork1D, L=L, site_tag_id=site_tag_id, inplace=True)
        out.view_as(MatrixProductTensor, inplace=True, L=L, cyclic=False, site_tag_id=site_tag_id,
                    upper_ind_id=out_ind_id, lower_ind_id=in1_ind_id, extra_ind_ids=(in2_ind_id,))
        return out


    def build_xmultiply_mpo(self, ax: 'Axis', x_power=1, offset=0.0, scale=1.0, split_opts: dict = None):
        """ (a(x-b))**c * f(x)
            ax: Axis object
        """
        mps = ax.map_state_to_mps( (ax.xpts * scale + offset)**x_power, split_opts=split_opts)
        mpo = helper.mps_to_diag_mpo(mps)
        return mpo


    def build_firstderivative_mpo(self, ax, left_bc=DEFAULT_BC, right_bc=DEFAULT_BC, order=DEFAULT_ORDER,
                                  fd_type=DEFAULT_FDTYPE, compress_opts=None):
        """ get first derivative along specified axis of tn_grid (MPO along 1D)
        """
        q, L = ax.q, ax.L

        # print('build deriv mpo', order, left_bc, right_bc, fd_type)

        if fd_type == FDType.FORWARD:
            mpo = self._mpo_firstderivative_forward(L, q, left_bc, right_bc, order=order, compress_opts=compress_opts)
        elif fd_type == FDType.BACKWARD:
            mpo = self._mpo_firstderivative_backward(L, q, left_bc, right_bc, order=order, compress_opts=compress_opts)
        elif fd_type == FDType.CENTER:
            mpo = self._mpo_firstderivative_center(L, q, left_bc, right_bc, order=order, compress_opts=compress_opts)
        else:
            raise NotImplementedError

        # mpo_data = ax.map_mpo_to_operator(mpo)
        # print('first deriv', order, fd_type)
        # print(mpo_data[:10, :10])

        helper.scalar_multiply(mpo, 1. / ax.dx, inplace=True)
        # if ax.is_flipped:
        #     helper.mpo_flip_lr(mpo, inplace=True)
        mpo = ax.map.transform_mpo(mpo)
        return mpo


    def build_secondderivative_mpo(self, ax, left_bc=DEFAULT_BC, right_bc=DEFAULT_BC, order=DEFAULT_ORDER,
                                   fd_type=DEFAULT_FDTYPE, compress_opts=None):
        """ get first derivative along specified axis
        """
        q, L = ax.q, ax.L
        fd_type = FDType.CENTER

        if fd_type == FDType.CENTER:
            mpo = self._mpo_secondderivative_center(L, q, left_bc, right_bc, order=order)
        else:
            raise NotImplementedError

        helper.scalar_multiply(mpo, 1. / ax.dx**2, inplace=True)
        # if ax.is_flipped:
        #     helper.mpo_flip_lr(mpo, inplace=True)
        mpo = ax.map.transform_mpo(mpo)
        return mpo


    def get_integral_weight(self, ax):
        """ normalization for integral_mps
        """
        return ax.dx


    def build_integral_mps(self, ax, is_sqrt=False, site_ind_id='i({})', site_tag_id='T({})'):
        """ get MPS that integrates out axis
        """
        if ax.is_even:
            # if is_sqrt:
            #     raise TypeError('spatial basis does not have sqrt form of integration mps')
            #     # return elemental multiply MPO?
            # else:
            #     integ_mps = ax.get_ones_mps()
            #     helper.scalar_multiply(integ_mps, ax.dx, inplace=True)
            #
            integ_mps = ax.get_ones_mps()
            helper.scalar_multiply(integ_mps, ax.dx, inplace=True)
            integ_mps = ax.map.transform_mps(integ_mps)
            # if ax.is_flipped:
            #     helper.mps_flip_lr(integ_mps, inplace=True)
        else:
            raise NotImplementedError
            ## something like
            # integ_mps = ax.map_state_to_mps(ax.dx)
        return integ_mps


    def build_indefinite_integral_mps(self, ax, order=1):
        """ get MPS that takes indefinite integral along axis
        """
        raise NotImplementedError


    def _mpo_firstderivative_center(self, L, q, left_bc=DEFAULT_BC, right_bc=DEFAULT_BC, order=DEFAULT_ORDER,
                                    compress_opts=None):
        """ S+|x> = |x+1> , S-|x> = |x-1>
            df/dx = \sum_i (S- - S+)|x_i>a
            is for a 1D system so dim does not need to be specified
            scale by 1/dt later
            boundary_condition:  boundary condition to use when taking derivative

            FD coeffs found using python package FinDiff
        """

        if q != 2:
            raise NotImplementedError('check q=2 implementation for binary mapping, esp if not pbc')

        if order == 1:
            c1, c2, c3, c4 = (1. / 2, 0, 0, 0)
        elif order == 2:
            c1, c2, c3, c4 = (2. / 3, -1. / 12, 0, 0)
        elif order == 3:
            c1, c2, c3, c4 = (3. / 4, -3. / 20, 1. / 60, 0)
        elif order == 4:
            c1, c2, c3, c4 = (4. / 5, -1. / 5, 4. / 105, -1. / 280)
        else:
            return self._mpo_higher_order_derivative(L, q, 1, left_bc, right_bc, order=order,
                                                    compress_opts=compress_opts)

        #### build n-ary +/- operator ####
        ## define operator such that operating on the desired spatial dimension
        sp = np.diag([1, ] * (q - 1), k=-1)
        sm = np.diag([1, ] * (q - 1), k=1)

        ## build MPO
        iden = np.eye(q)
        zero = np.zeros((q, q))

        # print('center', left_bc, right_bc)

        if left_bc == BCType.PERIODIC:
            mpo_tens = [np.array([iden, sm + sp, sp + sm])]  ## ignore dummy size 1 bond
            for i in range(1, L - 2):
                mpo_tens += [np.array([[iden, sm, sp],
                                       [zero, sp, zero],
                                       [zero, zero, sm]])]
            mpo_tens += [np.array([[iden, sm, sp, zero, zero],
                                   [zero, sp, zero, iden, zero],
                                   [zero, zero, sm, zero, iden]])]
            mpo_tens += [np.array([c1 * (sm - sp), (c1 * sp + c2 * iden + c3 * sm), -(c1 * sm + c2 * iden + c3 * sp),
                                   (c4 * iden + c3 * sp), -(c4 * iden + c3 * sm)])]
            ## ignore dummy size 1 bond
            mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                            upper_ind_id='o({})', lower_ind_id='i({})')

        elif left_bc == BCType.ANTIPERIODIC:
            mpo_tens = [np.array([iden, sm - sp, sp - sm])]  ## ignore dummy size 1 bond
            for i in range(1, L - 2):
                mpo_tens += [np.array([[iden, sm, sp],
                                       [zero, sp, zero],
                                       [zero, zero, sm]])]
            mpo_tens += [np.array([[iden, sm, sp, zero, zero],
                                   [zero, sp, zero, iden, zero],
                                   [zero, zero, sm, zero, iden]])]
            mpo_tens += [np.array([c1 * (sm - sp), (c1 * sp + c2 * iden + c3 * sm), -(c1 * sm + c2 * iden + c3 * sp),
                                   (c4 * iden + c3 * sp), -(c4 * iden + c3 * sm)])]
            ## ignore dummy size 1 bond
            mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                            upper_ind_id='o({})', lower_ind_id='i({})')

        else:
            m0 = np.diag([1.] + [0.] * (q - 1))
            m1 = np.diag([0.] * (q - 1) + [1.])

            nb = 4  # int(np.ceil(np.log2(order*2+1)))   ## number of bits to encode boundary condition
            bc0 = np.zeros((q ** nb, q ** nb))
            bc1 = np.zeros((q ** nb, q ** nb))

            import findiff

            def get_bulk_boundary_stencil():
                """ function that returns bulk coefficientss at the boundaries
                """
                ## edge boundary conditions: use the same number of points in stencil
                if order == 1:  # 3-pt stencil
                    fd_mat_0 = np.array([[0, c1, c2]])
                    # fd_mat_L = fd_mat_0[:, ::-1] * -1

                elif order == 2:  # 5-pt stencil
                    ## 000.. points
                    v0 = np.array([0., c1, c2, c3, c4])
                    v1 = np.array([-c1, 0., c1, c2, c3])
                    fd_mat_0 = np.array([v0, v1])

                    ## 111.. points
                    # fd_mat_L = np.array([v1[::-1], v0[::-1]]) * -1

                elif order == 3:  # 7-pt stencil
                    v0 = np.array([0., c1, c2, c3, c4, 0., 0.])
                    v1 = np.array([-c1, 0., c1, c2, c3, c4, 0.])
                    v2 = np.array([-c2, -c1, 0., c1, c2, c3, c4])
                    fd_mat_0 = np.array([v0, v1, v2])
                    # fd_mat_L = np.array([v2[::-1], v1[::-1], v0[::-1]]) * -1

                elif order == 4:  # order == 4
                    v0 = np.array([0., c1, c2, c3, c4, 0., 0., 0., 0.])
                    v1 = np.array([-c1, 0., c1, c2, c3, c4, 0., 0., 0.])
                    v2 = np.array([-c2, -c1, 0., c1, c2, c3, c4, 0., 0.])
                    v3 = np.array([-c3, -c2, -c1, 0., c1, c2, c3, c4, 0.])
                    fd_mat_0 = np.array([v0, v1, v2, v3])
                    # fd_mat_L = np.array([v3[::-1], v2[::-1], v1[::-1], v0[::-1]]) * -1

                else:
                    raise ValueError(f'order must be 1-4, not {order}')

                fd_mat_L = fd_mat_0[::-1,::-1] * -1

                return fd_mat_0, fd_mat_L


            def get_gen_boundary_stencil(num_stencil=2*order+1, offset=0):
                """ function that returns open boundary condition stencils in matrix form
                """
                coeffs = []
                for x0 in range(offset, offset+order):
                    stencil = findiff.coefficients(1,offsets=list(range(-x0, num_stencil-x0)))
                    coeffs += [stencil['coefficients']]
                    # print('stencil', stencil)

                end_bc_mat_0 = np.array(coeffs)
                end_bc_mat_L = end_bc_mat_0[::-1,::-1] * -1
                return end_bc_mat_0, end_bc_mat_L


            # def get_open_boundary_stencil():
            #     """ function that returns open boundary condition stencils in matrix form
            #     """
            #     ## edge boundary conditions: use the same number of points in stencil
            #     if order == 1:  # 3-pt stencil
            #         end_bc_mat_0 = np.array([[-3. / 2, 2. - c1, -1. / 2 - c2]])
            #         end_bc_mat_L = end_bc_mat_0[:, ::-1] * -1
            #
            #     elif order == 2:  # 5-pt stencil
            #         fwd0 = np.array([-25. / 12, 4., -3., 4. / 3, -1. / 4])  # forward FD at 0
            #         gen1 = np.array([-1. / 4, -5. / 6, 3. / 2, -1. / 2, 1. / 12])  # FD stencil at -1,0,1,2,3
            #
            #         ## 000.. points
            #         v0 = fwd0 - np.array([0., c1, c2, c3, c4])
            #         v1 = gen1 - np.array([-c1, 0., c1, c2, c3])
            #         end_bc_mat_0 = np.array([v0, v1])
            #
            #         ## 111.. points
            #         end_bc_mat_L = np.array([v1[::-1], v0[::-1]]) * -1
            #
            #     elif order == 3:  # 7-pt stencil
            #         fwd0 = np.array([-49. / 20, 6., -15. / 2, 20. / 3, -15. / 4, 6. / 5, -1. / 6])  # forward at 0
            #         gen1 = np.array([-1 / 6, -77 / 60, 5 / 2, -5 / 3, 5 / 6, -1 / 4, 1 / 30])  # stencil -1, ..., 5
            #         gen2 = np.array([1 / 30, -2 / 5, -7 / 12, 4 / 3, -1 / 2, 2 / 15, -1 / 60])  # stencil -2, ..., 4
            #
            #         v0 = fwd0 - np.array([0., c1, c2, c3, c4, 0., 0.])
            #         v1 = gen1 - np.array([-c1, 0., c1, c2, c3, c4, 0.])
            #         v2 = gen2 - np.array([-c2, -c1, 0., c1, c2, c3, c4])
            #         end_bc_mat_0 = np.array([v0, v1, v2])
            #         end_bc_mat_L = np.array([v2[::-1], v1[::-1], v0[::-1]]) * -1
            #
            #     elif order == 4:  # order == 4
            #         fwd0 = np.array([-761. / 280, 8., -14., 56. / 3, -35. / 2, 56. / 5, -14. / 3, 8. / 7, -1. / 8])  # 0
            #         gen1 = np.array([-1 / 8, -223 / 140, 7 / 2, -7 / 2, 35 / 12, -7 / 4, 7 / 10, -1 / 6, 1 / 56])  # 1
            #         gen2 = np.array([1 / 56, -2 / 7, -19 / 20, 2, -5 / 4, 2 / 3, -1 / 4, 2 / 35, -1 / 168])  # 2
            #         gen3 = np.array([-1 / 168, 1 / 14, -1 / 2, -9 / 20, 5 / 4, -1 / 2, 1 / 6, -1 / 28, 1 / 280])  # 3
            #
            #         v0 = fwd0 - np.array([0., c1, c2, c3, c4, 0., 0., 0., 0.])
            #         v1 = gen1 - np.array([-c1, 0., c1, c2, c3, c4, 0., 0., 0.])
            #         v2 = gen2 - np.array([-c2, -c1, 0., c1, c2, c3, c4, 0., 0.])
            #         v3 = gen3 - np.array([-c3, -c2, -c1, 0., c1, c2, c3, c4, 0.])
            #         end_bc_mat_0 = np.array([v0, v1, v2, v3])
            #         end_bc_mat_L = np.array([v3[::-1], v2[::-1], v1[::-1], v0[::-1]]) * -1
            #
            #     else:
            #         raise ValueError(f'order must be 1-4, not {order}')
            #
            #     return end_bc_mat_0, end_bc_mat_L

            # print('bc',left_bc, right_bc)
            # for order in range(1,5):
            #     print('order', order)
            #     left_bulk, right_bulk = get_boundary_stencil()
            #     left_open, right_open = get_gen_boundary_stencil(offset=0)
            #     left_open, right_open = get_gen_boundary_stencil(offset=1)
            # exit()

            left_bulk, right_bulk = get_bulk_boundary_stencil()
            # print('left bulk', left_bulk)
            # print('right bulk', right_bulk)

            ## left-hand side boundaries
            if left_bc == BCType.ZEROVALUE or left_bc == BCType.ABSORBING:
                # bc = np.array([[c1, c2, c3, c4], [c2, c3, c4, 0.], [c3, c4, 0., 0.], [c4, 0., 0., 0.]])
                # bc0[0:4, 1:5] = bc

                end_bc_mat_0 = -left_bulk

                ## if one ghost cell x[-1]= -x[1], x[0] = 0
                num_ghost = 0       # 1 ghost cell works best?
                left_bc_mat = get_gen_boundary_stencil(offset=num_ghost)[0]
                left_bc_mat[:, num_ghost] = 0  # x[0] = 0
                if left_bc == BCType.ABSORBING:
                    left_bc_mat[0, :] = 0  # d/dx [0] = 0
                if num_ghost == 0:
                    end_bc_mat_0 += left_bc_mat
                else:
                    end_bc_mat_0[:,:-num_ghost] += left_bc_mat[:,num_ghost:]
                    end_bc_mat_0[:,1:num_ghost+1] -= left_bc_mat[:,num_ghost-1::-1]


            elif left_bc == BCType.ZEROGRADIENT:
                # bc = np.array([[c1, c2, c3, c4], [c2, c3, c4, 0.], [c3, c4, 0., 0.], [c4, 0., 0., 0.]]) * -1
                # bc0[0:4, 1:5] = bc

                end_bc_mat_0 = -left_bulk

                ## if one ghost cell x[-1]= x[1]
                num_ghost = 0
                left_bc_mat = get_gen_boundary_stencil(offset=num_ghost)[0]
                left_bc_mat[0, :] = 0  # d/dx [0] = 0, so that it doesn't leave 0
                if num_ghost == 0:
                    end_bc_mat_0 += left_bc_mat
                else:
                    end_bc_mat_0[:, :-num_ghost] += left_bc_mat[:, num_ghost:]
                    end_bc_mat_0[:, 1:num_ghost + 1] += left_bc_mat[:, num_ghost-1::-1]


            # elif left_bc == BCType.ABSORBING:
            #     bc = np.array([[c1, c2, c3, c4], [c2, c3, c4, 0.], [c3, c4, 0., 0.], [c4, 0., 0., 0.]])
            #     bc0[0:4, 1:5] = bc
            #     # bc0[1:5,0] = np.array([-c1,-c2,-c3,-c4])*-1   # pin element at ind 0 to 0
            #     # print(c1,c2,c3,c4)
            #     # print(bc0[:5,:5])

            elif left_bc == BCType.OPEN:
                end_bc_mat_0 = -left_bulk
                end_bc_mat_0 += get_gen_boundary_stencil(offset=0)[0]

            else:
                raise ValueError(f'{left_bc} not a valid BCType')


            ## right-hand side boundaries
            if right_bc == BCType.ZEROVALUE or right_bc == BCType.ABSORBING:
                # bc = np.array([[0., 0., 0., c4], [0., 0., c4, c3], [0., c4, c3, c2], [c4, c3, c2, c1]]) * -1
                # bc1[-4:, -5:-1] = bc

                end_bc_mat_1 = -right_bulk

                ## if one ghost cell x[L+1]= -x[L-1], x[L] = 0
                num_ghost = 0       # 1 ghost cell works best?
                right_bc_mat = get_gen_boundary_stencil(offset=num_ghost)[1]
                right_bc_mat[:, -num_ghost-1] = 0  # x[L] = 0
                if right_bc == BCType.ABSORBING:
                    right_bc_mat[-1, :] = 0  # d/dx [L] = 0, so that it doesn't leave 0
                if num_ghost == 0:
                    end_bc_mat_1 += right_bc_mat
                else:
                    end_bc_mat_1[:, num_ghost:] += right_bc_mat[:, :-num_ghost]
                    end_bc_mat_1[:, -num_ghost - 1:-1] -= right_bc_mat[:, -1:-num_ghost-1:-1]

            elif right_bc == BCType.ZEROGRADIENT:
                # bc = np.array([[0., 0., 0., c4], [0., 0., c4, c3], [0., c4, c3, c2], [c4, c3, c2, c1]])
                # bc1[-4:, -5:-1] = bc

                end_bc_mat_1 = -right_bulk

                ## if one ghost cell x[L+1]= -x[L-1], x[L] = 0
                num_ghost = 0
                right_bc_mat = get_gen_boundary_stencil(offset=num_ghost)[1]
                right_bc_mat[-1, :] = 0  # d/dx [L] = 0
                if num_ghost == 0:
                    end_bc_mat_1 += right_bc_mat
                else:
                    end_bc_mat_1[:, num_ghost:] += right_bc_mat[:, :-num_ghost]
                    end_bc_mat_1[:, -num_ghost - 1:-1] += right_bc_mat[:, -1:-num_ghost-1:-1]
                    # end_bc_mat_1[:, -num_ghost - 1:-1] += right_bc[:, -num_ghost:][:,::-1]

            # elif right_bc == BCType.ABSORBING: # ['a', 'A', 'absorbing']:
            #     bc = np.array([[0., 0., 0., c4], [0., 0., c4, c3], [0., c4, c3, c2], [c4, c3, c2, c1]]) * -1
            #     bc1[-4:, -5:-1] = bc
            #     # bc1[-5:-1,-1] = np.array([c4,c3,c2,c1])*-1   # pin element at ind L to 0
            #     # print(c1,c2,c3,c4)
            #     # print(bc1[-5:,-5:])

            elif right_bc == BCType.OPEN:
                end_bc_mat_1 = -right_bulk
                end_bc_mat_1 += get_gen_boundary_stencil(offset=0)[1]

            else:
                raise ValueError(f'{right_bc} not a valid BCType')

            nx, ny = end_bc_mat_0.shape
            bc0[:nx, :ny] = end_bc_mat_0
            bc1[-nx:, -ny:] = end_bc_mat_1


            ## 0 side
            bc0 = np.reshape(bc0, (q,) * nb * 2)
            bc0_tens = qtn.Tensor(bc0, inds=tuple([f'o({i})' for i in range(nb)] + \
                                                  [f'i({i})' for i in range(nb)]))
            bc0_mpo = helper.mpx_from_dense(bc0_tens, nb, ('o({})', 'i({})'), site_tag_id='X({})')

            ## L-1 side
            bc1 = np.reshape(bc1, (q,) * nb * 2)
            bc1_tens = qtn.Tensor(bc1, inds=tuple([f'o({i})' for i in range(nb)] + \
                                                  [f'i({i})' for i in range(nb)]))
            bc1_mpo = helper.mpx_from_dense(bc1_tens, nb, ('o({})', 'i({})'), site_tag_id='X({})')

            ## leading 00... or 1... strings
            if L - nb >= 2:
                mpo_0 = qtn.MatrixProductOperator(
                    [np.array([m0])] + [np.array([[m0]])] * (L - nb - 2) + [np.array([m0])],
                    shape='lrud', site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
                mpo_1 = qtn.MatrixProductOperator(
                    [np.array([m1])] + [np.array([[m1]])] * (L - nb - 2) + [np.array([m1])],
                    shape='lrud', site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
            elif L - nb == 1:
                mpo_0 = qtn.TensorNetwork([qtn.Tensor(m0, inds=('o(0)', 'i(0)'), tags=('X(0)',))])
                mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
                              site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
                mpo_1 = qtn.TensorNetwork([qtn.Tensor(m1, inds=('o(0)', 'i(0)'), tags=('X(0)',))])
                mpo_1.view_as(qtn.MatrixProductOperator, like=mpo_0, inplace=True)
            elif L - nb == 0:
                mpo_0 = qtn.TensorNetwork([])
                mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
                              site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
                mpo_1 = qtn.TensorNetwork([])
                mpo_1.view_as(qtn.MatrixProductOperator, like=mpo_0, inplace=True)
            else:
                raise ValueError('FD order too large for current grid size')

            helper.append_mpx(mpo_0, bc0_mpo, inplace=True)
            helper.append_mpx(mpo_1, bc1_mpo, inplace=True)

            ### the bulk mpo
            mpo_tens = [np.array([iden, sm, sp])]  ## ignore dummy size 1 bond
            for i in range(1, L - 2):
                mpo_tens += [np.array([[iden, sm, sp],
                                       [zero, sp, zero],
                                       [zero, zero, sm]])]
            mpo_tens += [np.array([[iden, sm, sp, zero, zero],
                                   [zero, sp, zero, iden, zero],
                                   [zero, zero, sm, zero, iden]])]
            mpo_tens += [np.array([c1 * (sm - sp), (c1 * sp + c2 * iden + c3 * sm), -(c1 * sm + c2 * iden + c3 * sp),
                                   (c4 * iden + c3 * sp), -(c4 * iden + c3 * sm)])]  ## ignore size 1 bond

            mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                            upper_ind_id='o({})', lower_ind_id='i({})')
            helper.add_MPO(mpo, mpo_0, inplace=True)
            helper.add_MPO(mpo, mpo_1, inplace=True)

            helper.compress(mpo, compress_opts=compress_opts)

        return mpo


    @classmethod
    def _mpo_firstderivative_forward(self, L, q, left_bc=DEFAULT_BC, right_bc=DEFAULT_BC, order=DEFAULT_ORDER,
                                     compress_opts=None):
        """ S+|x> = |x+1> , S-|x> = |x-1>
            df/dx = \sum_i (S- - S+)|x_i>
            is for a 1D system so dim does not need to be specified
            scale by 1/dt later
            forward diff or backward diff
            boundary_condition:  boundary condition to use when taking derivative

            FD coeffs found using python package FinDiff
        """
        print('forward FD mpo')
        print('config', left_bc, right_bc, order)

        if q != 2:
            raise NotImplementedError('check q=2 implementation for binary mapping, esp if not pbc')

        coeffs_fd = fd_coeff.coefficients(deriv=1, acc=2 * order)
        c = coeffs_fd['forward']['coefficients']
        # print('FD coeffs', c)

        nc = len(c)
        nb = int(np.floor(np.log(nc - 1) / np.log(q)) + 1)  ## number of sites at the end

        c = np.append(c, np.zeros(q ** nb - nc))
        # c *= -1

        #### build n-ary +/- operator ####
        ## define operator such that operating on the desired spatial dimension
        sm = np.diag([1, ] * (q - 1), k=1)
        sp = np.diag([1, ] * (q - 1), k=-1)

        ## build MPO
        iden = np.eye(q)
        zero = np.zeros((q, q))

        sign = right_bc.value  # -1, 1 if AP or P; else some other int
        if abs(sign) != 1:
            sign = 0


        #### build finite difference MPO for PBC/APBC ####
        if nb == 2:
            if abs(sign) == 1:
                mpo_tens = [np.array([iden, sm, sign * sp])]  ## ignore dummy size 1 bond
                for i in range(1, L - nb):
                    mpo_tens += [np.array([[iden, sm, zero],
                                           [zero, sp, zero],
                                           [zero, zero, sp]])]
                mpo_tens += [np.array([[iden, sm, zero],
                                       [zero, sp, zero],
                                       [zero, sp, iden]])]
                mpo_tens += [np.array([c[0] * iden + c[1] * sm,
                                       c[1] * sp + c[2] * iden + c[3] * sm,
                                       c[3] * sp])]
            else:

                mpo_tens = [np.array([iden, sm])]  ## ignore dummy size 1 bond
                for i in range(1, L - nb):
                    mpo_tens += [np.array([[iden, sm],
                                           [zero, sp]])]
                mpo_tens += [np.array([[iden, sm],
                                       [zero, sp]])]
                mpo_tens += [np.array([c[0] * iden + c[1] * sm,
                                       c[1] * sp + c[2] * iden + c[3] * sm])]

            mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                            upper_ind_id='o({})', lower_ind_id='i({})')
            # print('op', axis.map_mpo_to_operator(mpo))
        else:
            npts = q ** L
            op = np.zeros((npts,) * 2)
            for i in range(npts):
                for j in range(nc):
                    if i + j < npts:
                        op[i, i + j] = c[j]
                    elif sign:
                        op[i, (i + j) % npts] = c[j] * sign

            op_tensor = np.reshape(op, (q,) * (2 * L))
            op_tensor = qtn.Tensor(op_tensor, inds=[f'o({i})' for i in range(L)] + \
                                                   [f'i({i})' for i in range(L)])
            mpo = helper.mpx_from_dense(op_tensor, L, ('o({})', 'i({})'), site_tag_id='B{}')


        #### get boundary conditions (subtract existing coeffs used above) ####
        #### bc forward diff, only right bc matters ####

        if abs(sign) != 1:  # not periodic or antiperiodic, apply bc's

            num_stencil = 2*order+1
            right_bulk = np.zeros((num_stencil-1, num_stencil))
            for i in range(num_stencil-1):
                for j in range(nc):
                    try:
                        right_bulk[i, i + j + 1] = c[j]
                    except IndexError:
                        continue

            # print('right bulk', right_bulk)


            def get_gen_boundary_stencil(num_stencil=num_stencil, offset=0):
                """ function that returns open boundary condition stencils in matrix form
                    offset must be less than num_stencil
                """
                coeffs = np.zeros((num_stencil-1, num_stencil+offset))
                for x0 in range(0, offset-1):
                    stencil = fd_coeff.coefficients(deriv=1, offsets=list(range(num_stencil)))
                    coeffs[x0, x0+1:x0+1+num_stencil] = stencil['coefficients']
                    # print('stencil', x0, stencil)
                for x0 in range(offset-1, num_stencil-1):
                    i0 = (offset-1)-x0
                    stencil = fd_coeff.coefficients(deriv=1, offsets=list(range(i0, i0 + num_stencil)))
                    # print('stencil', x0, stencil)
                    coeffs[x0, -num_stencil:] = stencil['coefficients']

                end_bc_mat_L = np.array(coeffs)
                # print('gen bc', end_bc_mat_L)
                return end_bc_mat_L


            ## added on top of bulk mpo
            ## right-hand side boundaries
            if right_bc == BCType.ZEROVALUE or right_bc == BCType.ABSORBING:
                # for i in range(1, nc):    bc1[-nc + i, -i - 1:-1] = c[nc - i:nc][::-1] * -1

                end_bc_mat_1 = -right_bulk

                ## ghost cells:  x[L+1]= -x[L-1], x[L] = 0
                num_ghost = 0     # 0 works the best
                right_bc_mat = get_gen_boundary_stencil(offset=num_ghost)
                right_bc_mat[:, -num_ghost - 1] = 0  # x[L] = 0
                if right_bc == BCType.ABSORBING:
                    right_bc[-1, :] = 0  # d/dx [L] = 0, so that it doesn't leave 0
                if num_ghost == 0:
                    end_bc_mat_1[:, :] += right_bc_mat
                else:
                    end_bc_mat_1[:, :] += right_bc_mat[:, :-num_ghost]
                    end_bc_mat_1[:, -num_ghost - 1:-1] -= right_bc_mat[:, -1:-num_ghost - 1:-1]

            # elif right_bc == BCType.ABSORBING:
            #
            #     op = np.zeros((q ** nb,) * 2)
            #     for i in range(q ** nb):  ## assume only 1 ghost cell (absorbing bc set to 0)
            #         if i < q ** nb - 2:  ## avoid putting stencil on ghost cell
            #             space = q ** nb - i - 1
            #             # offsets = np.append(np.arange((space-nc),0),np.arange(0,min(nc,space)))
            #             offsets = list(np.arange(min(space - nc, 0), min(nc, space)))
            #             coeffs = fd_coeff.coefficients(deriv=1, offsets=offsets)
            #             # print(i, offsets)
            #             op[i, i + offsets[0]:i + offsets[-1] + 1] = coeffs['coefficients']
            #         elif i == q ** nb - 2:  ## connects to ghost cell set to 0
            #             offsets = list(np.arange(-nc + 2, 2))
            #             # print(i, offsets)
            #             coeffs = fd_coeff.coefficients(deriv=1, offsets=offsets)
            #             op[i, i + offsets[0]:i + offsets[-1] + 1] = coeffs['coefficients']
            #         elif i == q ** nb - 1:  ## the ghost cell -> set derivative to 0
            #             pass
            #             # offsets = np.arange(-nc+1,1)
            #             # print(i,offsets)
            #             # coeffs = fd_coeff.coefficients(deriv=1,offsets=offsets)
            #             # op[i,i+offsets[0]:i+offsets[-1]+1] = coeffs['coefficients']
            #
            #     # op = np.zeros((q**nb,)*2)
            #     # print('right op', op[-nc:, -nc:])
            #     op -= right_bulk
            #     bc1 = op

            elif right_bc == BCType.ZEROGRADIENT:
                # for i in range(1, nc):    bc1[-nc + i, -i - 1:-1] = c[nc - i:nc][::-1]

                end_bc_mat_1 = -right_bulk

                ## ghost cells: x[L+1]= -x[L-1], x[L] = 0
                num_ghost = 0
                right_bc_mat = get_gen_boundary_stencil(offset=num_ghost)
                right_bc_mat[-1, :] = 0  # d/dx [L] = 0
                if num_ghost == 0:
                    end_bc_mat_1[:, :] += right_bc_mat
                else:
                    end_bc_mat_1[:, :] += right_bc_mat[:, :-num_ghost]
                    end_bc_mat_1[:, -num_ghost - 1:-1] += right_bc_mat[:, -1:-num_ghost - 1:-1]

            elif right_bc == BCType.OPEN:

                end_bc_mat_1 = -right_bulk
                end_bc_mat_1 += get_gen_boundary_stencil(offset=0)

            else:
                raise ValueError(f'{right_bc} not a valid BCType')

            bc1 = np.zeros((q ** nb, q ** nb))
            nx, ny = end_bc_mat_1.shape
            bc1[-nx:, -ny:] = end_bc_mat_1

            m0 = np.diag([1.] + [0.] * (q - 1))
            m1 = np.diag([0.] * (q - 1) + [1.])

            ## L-1 side
            bc1 = np.reshape(bc1, (q,) * nb * 2)
            bc1_tens = qtn.Tensor(bc1, inds=tuple([f'o({i})' for i in range(nb)] + \
                                                  [f'i({i})' for i in range(nb)]))
            bc1_mpo = helper.mpx_from_dense(bc1_tens, nb, ('o({})', 'i({})'), site_tag_id='X({})')

            ## leading 00... or 1... strings
            if L - nb >= 2:
                mpo_1 = qtn.MatrixProductOperator(
                    [np.array([m1])] + [np.array([[m1]])] * (L - nb - 2) + [np.array([m1])],
                    shape='lrud', site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
            elif L - nb == 1:
                mpo_1 = qtn.TensorNetwork([qtn.Tensor(m1, inds=('o(0)', 'i(0)'), tags=('X(0)',))])
                mpo_1.view_as(qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
                              site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
            elif L - nb == 0:
                mpo_1 = qtn.TensorNetwork([])
                mpo_1.view_as(qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
                              site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
            else:
                raise ValueError('FD order too large for current grid size')

            helper.append_mpx(mpo_1, bc1_mpo, inplace=True)
            helper.add_MPO(mpo, mpo_1, inplace=True)

            helper.compress(mpo, compress_opts=compress_opts)

        return mpo


    def _mpo_firstderivative_backward(self, L, q, left_bc=DEFAULT_BC, right_bc=DEFAULT_BC, order=DEFAULT_ORDER,
                                      compress_opts=None):
        """ S+|x> = |x+1> , S-|x> = |x-1>
            df/dx = \sum_i (S- - S+)|x_i>
            is for a 1D system so dim does not need to be specified
            scale by 1/dt later
            forward diff or backward diff
            boundary_condition:  boundary condition to use when taking derivative

            FD coeffs found using python package FinDiff
        """
        if q != 2:
            raise NotImplementedError('check q=2 implementation for binary mapping, esp if not pbc')

        print('backward FD mpo')

        coeffs_fd = fd_coeff.coefficients(deriv=1, acc=2 * order)
        c = coeffs_fd['backward']['coefficients'][::-1]

        nc = len(c)
        nb = int(np.floor(np.log(nc - 1) / np.log(q)) + 1)  ## number of sites at the end
        c = np.append(c, np.zeros(q ** nb - nc))

        #### build n-ary +/- operator ####
        ## define operator such that operating on the desired spatial dimension
        sm = np.diag([1, ] * (q - 1), k=1)
        sp = np.diag([1, ] * (q - 1), k=-1)

        ## build MPO
        iden = np.eye(q)
        zero = np.zeros((q, q))

        sign = left_bc.value    # 1, -1 if P, AP; else some other number
        if abs(sign) != 1:
            sign = 0

        #### build finite difference MPO for PBC/APBC ####
        if nb == 2:
            if abs(sign) == 1:
                mpo_tens = [np.array([iden, sp, sign * sm])]  ## ignore dummy size 1 bond
                for i in range(1, L - nb):
                    mpo_tens += [np.array([[iden, sp, zero],
                                           [zero, sm, zero],
                                           [zero, zero, sm]])]

                mpo_tens += [np.array([[iden, sp, zero],
                                       [zero, sm, zero],
                                       [zero, sm, iden]])]
                mpo_tens += [np.array([c[0] * iden + c[1] * sp,
                                       c[1] * sm + c[2] * iden + c[3] * sp,
                                       c[3] * sm])]

            else:
                mpo_tens = [np.array([iden, sp])]  ## ignore dummy size 1 bond
                for i in range(1, L - nb):
                    mpo_tens += [np.array([[iden, sp],
                                           [zero, sm]])]

                mpo_tens += [np.array([[iden, sp],
                                       [zero, sm]])]
                mpo_tens += [np.array([c[0] * iden + c[1] * sp,
                                       c[1] * sm + c[2] * iden + c[3] * sp])]

            mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                            upper_ind_id='o({})', lower_ind_id='i({})')
        else:
            npts = q ** L
            op = np.zeros((npts,) * 2)
            for i in range(npts):
                for j in range(nc):
                    if i - j >= 0:
                        op[i, i - j] = c[j]
                    elif sign:
                        op[i, i - j] = c[j] * sign

            op_tensor = np.reshape(op, (q,) * (2 * L))
            op_tensor = qtn.Tensor(op_tensor, inds=[f'o({i})' for i in range(L)] + \
                                                   [f'i({i})' for i in range(L)])
            mpo = helper.mpx_from_dense(op_tensor, L, ('o({})', 'i({})'), site_tag_id='B{}')

        #### get boundary conditions (subtract existing coeffs used above) ####
        #### bc forward diff, only right bc matters ####

        if abs(sign) != 1:  # not periodic or antiperiodic, apply bc's

            num_stencil = 2 * order + 1
            left_bulk = np.zeros((num_stencil - 1, num_stencil))
            for i in range(num_stencil - 1):
                for j in range(nc):
                    if i - j < 0:
                        continue
                    else:
                        left_bulk[i, i - j] = c[j]

            # coeffs_mat = np.zeros((q ** nb,) * 2)
            # for i in range(q ** nb):
            #     for j in range(nc):
            #         if i - j < 0:
            #             continue
            #         else:
            #             coeffs_mat[i, i - j] = c[j]

            # print('left bulk', left_bulk)

            def get_gen_boundary_stencil(num_stencil=num_stencil, offset=0):
                """ function that returns open boundary condition stencils in matrix form
                    offset must be less than num_stencil
                """
                coeffs = np.zeros((num_stencil-1, num_stencil+offset))
                for x0 in range(num_stencil-offset-1):
                    i0 = -x0-offset
                    stencil = fd_coeff.coefficients(deriv=1, offsets=list(range(i0, i0 + num_stencil)))
                    coeffs[x0,:num_stencil] = stencil['coefficients']

                for x0 in range(num_stencil-offset-1, num_stencil-1):
                    stencil = fd_coeff.coefficients(deriv=1, offsets=list(range(-num_stencil + 1,1)))
                    coeffs[x0,x0-(num_stencil-offset)+1:x0+offset+1] = stencil['coefficients']

                end_bc_mat_0 = coeffs
                # print('gen bc', end_bc_mat_0)

                return end_bc_mat_0


            ## added on top of bulk mpo
            ## left-hand side boundaries
            if left_bc == BCType.ZEROVALUE or left_bc == BCType.ABSORBING:
                # for i in range(nc):     bc0[i, 1:nc - i] = c[i + 1:nc] * -1

                end_bc_mat_0 = -left_bulk

                num_ghost = 0       # 1 works the best
                ## assume one ghost cell x[-1]= -x[1], x[0] = 0
                left_bc_mat = get_gen_boundary_stencil(offset=num_ghost)
                left_bc_mat[:, num_ghost] = 0  # x[0] = 0
                if left_bc == BCType.ABSORBING:
                    left_bc[0, :] = 0  # d/dx [0] = 0
                end_bc_mat_0[:, :] += left_bc_mat[:, num_ghost:]
                if num_ghost > 0:
                    end_bc_mat_0[:, 1:num_ghost + 1] -= left_bc_mat[:, num_ghost - 1::-1]

            # elif left_bc == BCType.ABSORBING:
            #
            #     op = np.zeros((q ** nb,) * 2)
            #     for i in range(q ** nb):  ## assume only 1 ghost cell (absorbing bc set to 0)
            #         if i >= 2:  ## avoid putting stencil on ghost cell
            #             offsets = list(np.arange(-min(i - 1, nc - 1), max(nc - i + 1, 1)))
            #             # print(i, offsets)
            #             coeffs = fd_coeff.coefficients(deriv=1, offsets=offsets)
            #             op[i, i + offsets[0]:i + offsets[-1] + 1] = coeffs['coefficients']
            #         elif i == 1:  ## connects to ghost cell set to 0
            #             offsets = list(np.arange(-1, nc - 1))
            #             # print(i, offsets)
            #             coeffs = fd_coeff.coefficients(deriv=1, offsets=offsets)
            #             op[i, i + offsets[0]:i + offsets[-1] + 1] = coeffs['coefficients']
            #         elif i == 0:  ## the ghost cell -> set derivative to 0
            #             pass
            #             # offsets = np.arange(0,nc)
            #             # print(i,offsets)
            #             # coeffs = fd_coeff.coefficients(deriv=1,offsets=offsets)
            #             # op[i,i+offsets[0]:i+offsets[-1]+1] = coeffs['coefficients']
            #
            #     op -= coeffs_mat
            #     bc0 = op

            elif left_bc == BCType.ZEROGRADIENT:
                # for i in range(nc):     bc0[i, 1:nc - i] = c[i + 1:nc]  # skip 0 offset

                end_bc_mat_0 = -left_bulk

                ## assume one ghost cell x[-1]= -x[1], x[0] = 0
                num_ghost = 0
                left_bc_mat = get_gen_boundary_stencil(offset=num_ghost)
                left_bc_mat[0, :] = 0  # d/dx [0] = 0
                end_bc_mat_0[:, :] += left_bc_mat[:, num_ghost:]
                if num_ghost > 0:
                    end_bc_mat_0[:, 1:num_ghost + 1] += left_bc_mat[:, num_ghost - 1::-1]

            elif left_bc == BCType.OPEN:

                end_bc_mat_0 = -left_bulk
                end_bc_mat_0 += get_gen_boundary_stencil(offset=0)

            else:
                raise ValueError(f'{left_bc} not a valid BCType')

            bc0 = np.zeros((q ** nb, q ** nb))
            nx, ny = end_bc_mat_0.shape
            bc0[:nx,:ny] = end_bc_mat_0

            m0 = np.diag([1.] + [0.] * (q - 1))
            m1 = np.diag([0.] * (q - 1) + [1.])

            ## 0 side
            bc0 = np.reshape(bc0, (q,) * nb * 2)
            bc0_tens = qtn.Tensor(bc0, inds=tuple([f'o({i})' for i in range(nb)] + \
                                                  [f'i({i})' for i in range(nb)]))
            bc0_mpo = helper.mpx_from_dense(bc0_tens, nb, ('o({})', 'i({})'), site_tag_id='X({})')

            ## leading 00... or 1... strings
            if L - nb >= 2:
                mpo_0 = qtn.MatrixProductOperator(
                    [np.array([m0])] + [np.array([[m0]])] * (L - nb - 2) + [np.array([m0])],
                    shape='lrud', site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
            elif L - nb == 1:
                mpo_0 = qtn.TensorNetwork([qtn.Tensor(m0, inds=('o(0)', 'i(0)'), tags=('X(0)',))])
                mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
                              site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
            elif L - nb == 0:
                mpo_0 = qtn.TensorNetwork([])
                mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
                              site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
            else:
                raise ValueError('FD order too large for current grid size')

            helper.append_mpx(mpo_0, bc0_mpo, inplace=True)
            helper.add_MPO(mpo, mpo_0, inplace=True)

            helper.compress(mpo, compress_opts=compress_opts)

        return mpo



    def _mpo_secondderivative_center(self, L, q, left_bc=DEFAULT_BC, right_bc=DEFAULT_BC, order=DEFAULT_ORDER,
                                     compress_opts=None):
        """ for dim1=dim2=dim=0, since this is 1D
            S+|x> = |x+1> , S-|x> = |x-1>
            d2f/dx2 = \sum_i 1/2*(S- + S+ - 2I)|x_i>
            dim specifies which dimension in which shift is taken, indexed starting at 0
            scale by 1/dt later
            FD coeffs found using python package FinDiff
        """
        if q != 2:
            raise NotImplementedError('check q=2 implementation for binary mapping, esp if not pbc')

        if order == 1:
            c0, c1, c2, c3, c4 = (-2., 1., 0., 0., 0.)
        elif order == 2:
            c0, c1, c2, c3, c4 = (-5. / 2, 4. / 3, -1. / 12, 0, 0)
        elif order == 3:
            c0, c1, c2, c3, c4 = (-49. / 18, 3. / 2, -3 / 20, 1 / 90, 0)
        elif order == 4:
            c0, c1, c2, c3, c4 = (-205. / 72, 8. / 5, -1 / 5, 8 / 315, -1 / 560)
        else:
            return self._mpo_higher_order_derivative(L, q, 2, left_bc, right_bc, order=order,
                                                     compress_opts=compress_opts)
        print('coeffs', c0, c1, c2, c3, c4)

        #### build n-ary +/- operator ####
        ## define operator such that operating on the desired spatial dimension
        sm = np.diag([1, ] * (q - 1), k=1)
        sp = np.diag([1, ] * (q - 1), k=-1)

        ## build MPO
        iden = np.eye(q)
        zero = np.zeros((q, q))

        if left_bc == BCType.PERIODIC:  # 0000 <--> 1111
            mpo_tens = [np.array([iden, sp + sm, sm + sp])]  ## ignore dummy size 1 bond
            for i in range(1, L - 2):
                mpo_tens += [np.array([[iden, sp, sm],
                                       [zero, sm, zero],
                                       [zero, zero, sp]])]
            mpo_tens += [np.array([[iden, sp, sm, zero, zero],
                                   [zero, sm, zero, iden, zero],
                                   [zero, zero, sp, zero, iden]])]
            mpo_tens += [
                np.array([c1 * (sm + sp) + c0 * iden, (c1 * sm + c2 * iden + c3 * sp), (c1 * sp + c2 * iden + c3 * sm),
                          (c4 * iden + c3 * sm), (c4 * iden + c3 * sp)])]
            ## ignore dummy size 1 bond
            mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                            upper_ind_id='o({})', lower_ind_id='i({})')

        elif left_bc == BCType.ANTIPERIODIC:
            mpo_tens = [np.array([iden, sp - sm, sm - sp])]  ## ignore dummy size 1 bond
            for i in range(1, L - 2):
                mpo_tens += [np.array([[iden, sp, sm],
                                       [zero, sm, zero],
                                       [zero, zero, sp]])]
            mpo_tens += [np.array([[iden, sp, sm, zero, zero],
                                   [zero, sm, zero, iden, zero],
                                   [zero, zero, sp, zero, iden]])]
            mpo_tens += [
                np.array([c1 * (sm + sp) + c0 * iden, (c1 * sm + c2 * iden + c3 * sp), (c1 * sp + c2 * iden + c3 * sm),
                          (c4 * iden + c3 * sm), (c4 * iden + c3 * sp)])]
            ## ignore dummy size 1 bond
            mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                            upper_ind_id='o({})', lower_ind_id='i({})')

        else:  # forward/backward diff, not center diff for 000, 111
            m0 = np.diag([1.] + [0.] * (q - 1))
            m1 = np.diag([0.] * (q - 1) + [1.])

            nb = 4  # int(np.ceil(np.log2(order*2+1)))   ## number of bits to encode boundary condition
            bc0 = np.zeros((q ** nb, q ** nb))
            bc1 = np.zeros((q ** nb, q ** nb))


            num_stencil = 2*order + 2

            def get_bulk_boundary_stencil():
                """ function that returns bulk coefficients at the boundaries
                """
                ## edge boundary conditions: use the same number of points in stencil
                if order == 1:  # 3-pt stencil
                    fd_mat_0 = np.array([[c0, c1, c2, c3]])

                elif order == 2:  # 5-pt stencil
                    ## 000.. points
                    v0 = np.array([c0, c1, c2, c3, c4, 0.])
                    v1 = np.array([c1, c0, c1, c2, c3, c4])
                    fd_mat_0 = np.array([v0, v1])

                elif order == 3:  # 7-pt stencil
                    v0 = np.array([c0, c1, c2, c3, c4, 0., 0., 0.])
                    v1 = np.array([c1, c0, c1, c2, c3, c4, 0., 0.])
                    v2 = np.array([c2, c1, c0, c1, c2, c3, c4, 0.])
                    fd_mat_0 = np.array([v0, v1, v2])

                elif order == 4:  # order == 4
                    v0 = np.array([c0, c1, c2, c3, c4, 0., 0., 0., 0., 0.])
                    v1 = np.array([c1, c0, c1, c2, c3, c4, 0., 0., 0., 0.])
                    v2 = np.array([c2, c1, c0, c1, c2, c3, c4, 0., 0., 0.])
                    v3 = np.array([c3, c2, c1, c0, c1, c2, c3, c4, 0., 0.])
                    fd_mat_0 = np.array([v0, v1, v2, v3])

                else:
                    raise ValueError(f'order must be 1-4, not {order}')

                fd_mat_L = fd_mat_0[::-1,::-1]

                return fd_mat_0, fd_mat_L


            def get_gen_boundary_stencil(num_stencil=num_stencil, offset=0):
                """ function that returns open boundary condition stencils in matrix form
                """
                coeffs = []
                for x0 in range(offset, offset+order):
                    stencil = fd_coeff.coefficients(2, offsets=list(range(-x0, num_stencil-x0)))
                    # print('stencil', stencil)
                    coeffs += [stencil['coefficients']]

                end_bc_mat_0 = np.array(coeffs)
                end_bc_mat_L = end_bc_mat_0[::-1,::-1]
                # print('gen bc', end_bc_mat_0)
                return end_bc_mat_0, end_bc_mat_L

            left_bulk, right_bulk = get_bulk_boundary_stencil()
            # print('left bulk', left_bulk)
            # print('right bulk', right_bulk)


            ## left-hnad side boundaries
            if left_bc == BCType.ZEROVALUE:
                # bc = np.array([[c1, c2, c3, c4], [c2, c3, c4, 0.], [c3, c4, 0., 0.], [c4, 0., 0., 0.]]) * -1
                # bc0[0:4, 1:5] = bc

                end_bc_mat_0 = -left_bulk

                ## if one ghost cell x[-1]= -x[1], x[0] = 0
                num_ghost = 0  # 1 ghost cell works best?
                left_bc_mat = get_gen_boundary_stencil(offset=num_ghost)[0]
                left_bc_mat[:, num_ghost] = 0  # x[0] = 0
                if num_ghost == 0:
                    end_bc_mat_0 += left_bc_mat
                else:
                    end_bc_mat_0[:, :-num_ghost] += left_bc_mat[:, num_ghost:]
                    end_bc_mat_0[:, 1:num_ghost + 1] -= left_bc_mat[:, num_ghost - 1::-1]

            elif left_bc == BCType.ABSORBING:
                raise NotImplementedError
                pass

            elif left_bc == BCType.ZEROGRADIENT:
                # bc = np.array([[c1, c2, c3, c4], [c2, c3, c4, 0.], [c3, c4, 0., 0.], [c4, 0., 0., 0.]])
                # bc0[0:4, 1:5] = bc

                end_bc_mat_0 = -left_bulk

                ## if one ghost cell x[-1]= -x[1], x[0] = 0
                num_ghost = 0
                left_bc_mat = get_gen_boundary_stencil(offset=num_ghost)[0]
                if num_ghost == 0:
                    end_bc_mat_0 += left_bc_mat
                else:
                    end_bc_mat_0[:, :-num_ghost] += left_bc_mat[:, num_ghost:]
                    end_bc_mat_0[:, 1:num_ghost + 1] += left_bc_mat[:, num_ghost - 1::-1]

            elif left_bc == BCType.OPEN:
                end_bc_mat_0 = -left_bulk
                end_bc_mat_0 += get_gen_boundary_stencil(offset=0)[0]

            else:
                raise ValueError(f'{left_bc} not a valid BCType')

            ## right-hand side boundaries
            if right_bc == BCType.ZEROVALUE:
                # bc = np.array([[0., 0., 0., c4], [0., 0., c4, c3], [0., c4, c3, c2], [c4, c3, c2, c1]]) * -1
                # bc1[-4:, -5:-1] = bc

                end_bc_mat_1 = -right_bulk

                ## if one ghost cell x[L+1]= -x[L-1], x[L] = 0
                num_ghost = 0  # 0 ghost cell works best
                right_bc_mat = get_gen_boundary_stencil(offset=num_ghost)[1]
                right_bc_mat[:, -num_ghost - 1] = 0  # x[L] = 0
                if num_ghost == 0:
                    end_bc_mat_1 += right_bc_mat
                if num_ghost > 0:
                    end_bc_mat_1[:, num_ghost:] += right_bc_mat[:, :-num_ghost]
                    end_bc_mat_1[:, -num_ghost - 1:-1] -= right_bc_mat[:, -1:-num_ghost - 1:-1]

            elif right_bc == BCType.ABSORBING:
                raise NotImplementedError
                pass

            elif right_bc == BCType.ZEROGRADIENT:
                # bc = np.array([[0., 0., 0., c4], [0., 0., c4, c3], [0., c4, c3, c2], [c4, c3, c2, c1]])
                # bc1[-4:, -5:-1] = bc

                end_bc_mat_1 = -right_bulk

                ## if one ghost cell x[L+1]= -x[L-1], x[L] = 0
                num_ghost = 0  # 0 ghost cells works best
                right_bc_mat = get_gen_boundary_stencil(offset=num_ghost)[1]
                if num_ghost == 0:
                    end_bc_mat_1 += right_bc_mat
                else:
                    end_bc_mat_1[:, num_ghost:] += right_bc_mat[:, :-num_ghost]
                    end_bc_mat_1[:, -num_ghost - 1:-1] += right_bc_mat[:, -1:-num_ghost - 1:-1]

            elif right_bc == BCType.OPEN:
                end_bc_mat_1 = -right_bulk
                end_bc_mat_1 += get_gen_boundary_stencil(offset=0)[1]

            else:
                raise ValueError(f'{right_bc} not a valid BCType')


            nx, ny = end_bc_mat_0.shape
            bc0[:nx, :ny] = end_bc_mat_0
            bc1[-nx:, -ny:] = end_bc_mat_1


            ## 0 side
            bc0 = np.reshape(bc0, (q,) * nb * 2)
            bc0_tens = qtn.Tensor(bc0, inds=tuple([f'o({i})' for i in range(nb)] + \
                                                  [f'i({i})' for i in range(nb)]))
            bc0_mpo = helper.mpx_from_dense(bc0_tens, nb, ('o({})', 'i({})'), site_tag_id='X({})')

            ## L-1 side
            bc1 = np.reshape(bc1, (q,) * nb * 2)
            bc1_tens = qtn.Tensor(bc1, inds=tuple([f'o({i})' for i in range(nb)] + \
                                                  [f'i({i})' for i in range(nb)]))
            bc1_mpo = helper.mpx_from_dense(bc1_tens, nb, ('o({})', 'i({})'), site_tag_id='X({})')

            ## leading 00... or 1... strings
            if L - nb >= 2:
                mpo_0 = qtn.MatrixProductOperator(
                    [np.array([m0])] + [np.array([[m0]])] * (L - nb - 2) + [np.array([m0])],
                    shape='lrud', site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
                mpo_1 = qtn.MatrixProductOperator(
                    [np.array([m1])] + [np.array([[m1]])] * (L - nb - 2) + [np.array([m1])],
                    shape='lrud', site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
            elif L - nb == 1:
                mpo_0 = qtn.TensorNetwork([qtn.Tensor(m0, inds=('o(0)', 'i(0)'), tags=('X(0)',))])
                mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
                              site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
                mpo_1 = qtn.TensorNetwork([qtn.Tensor(m1, inds=('o(0)', 'i(0)'), tags=('X(0)',))])
                mpo_1.view_as(qtn.MatrixProductOperator, like=mpo_0, inplace=True)
            elif L - nb == 0:
                mpo_0 = qtn.TensorNetwork([])
                mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
                              site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
                mpo_1 = qtn.TensorNetwork([])
                mpo_1.view_as(qtn.MatrixProductOperator, like=mpo_0, inplace=True)
            else:
                raise ValueError('FD order too large for current grid size')

            helper.append_mpx(mpo_0, bc0_mpo, inplace=True)
            helper.append_mpx(mpo_1, bc1_mpo, inplace=True)

            ### the bulk mpo
            mpo_tens = [np.array([iden, sp, sm])]  ## ignore dummy size 1 bond
            for i in range(1, L - 2):
                mpo_tens += [np.array([[iden, sp, sm],
                                       [zero, sm, zero],
                                       [zero, zero, sp]])]
            mpo_tens += [np.array([[iden, sp, sm, zero, zero],
                                   [zero, sm, zero, iden, zero],
                                   [zero, zero, sp, zero, iden]])]
            mpo_tens += [
                np.array([c1 * (sm + sp) + c0 * iden, (c1 * sm + c2 * iden + c3 * sp), (c1 * sp + c2 * iden + c3 * sm),
                          (c4 * iden + c3 * sm), (c4 * iden + c3 * sp)])]  ## ignore size 1 bond

            mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                            upper_ind_id='o({})', lower_ind_id='i({})')
            helper.add_MPO(mpo, mpo_0, inplace=True)
            helper.add_MPO(mpo, mpo_1, inplace=True)

            helper.compress(mpo, compress_opts=compress_opts)

        return mpo


    def _mpo_higher_order_derivative(self, L, q, ndim, deriv_order, left_bc=DEFAULT_BC, right_bc=DEFAULT_BC,
                                     order=DEFAULT_ORDER, compress_opts=None):
        """ derivs:  tuples of the form (axis, step size, derivative degree)
            open boundary conditions
            used for central difference with order > 4
        """
        print('warning: boundary conditions not applied')

        import findiff

        derivative = findiff.FinDiff(0, 1, deriv_order, acc=order * 2)
        op = derivative.matrix((q ** L,))  # scipy sparse matrix
        op = np.reshape(op.toarray(), (q,) * (2 * L))

        ## guess:  initial shape is (o10, ..., oL0) (o11,..., oL1) x (i10, ...,iL0) (i11, ..., iL1)
        ##         where i(a,b) corresponds to ath input leg along bth dimension
        op_tensor = qtn.Tensor(op, inds=[f'o({i})' for i in range(L)] + \
                                        [f'i({i})' for i in range(L)])

        mpo = helper.mpx_from_dense(op_tensor, L * ndim, ('o({})', 'i({})'), site_tag_id='B{}',
                                    split_opts=compress_opts)

        return mpo

