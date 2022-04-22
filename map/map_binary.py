import numpy as np
import quimb.tensor as qtn

from map import TNMap
import helper_quimb as helper

''' mapping of grid points along 1 dimension to binary TN ordering
'''

## binary TN class
class BinaryMap(TNMap):

    @classmethod
    def array_reshape(cls,ndarray,*shape):
        return ndarray.reshape(*shape)


    @classmethod
    def get_position_inds(cls,L,q,idx):
        ''' returns physical bond indices (0,1) of MPS that corresponds to vector position idx
        '''
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
            raise(ValueError)
        elif len(str_b) < L:
            str_b = '0'*(L-len(str_b)) + str_b

        ind_list = [int(s) for s in str_b]
        return ind_list


    @classmethod
    def get_inds_position(cls,L,q,inds_list):
        ''' returns index in vector coresponding to physical bond indices specified in inds_list
        '''
        bstring = ''
        for b in inds_list:   bstring += str(b)
        i = bstring
        idx = int(i,q)
        return idx


    #################
    ## derivatives ##
    #################

    @classmethod
    def _mpo_firstderivative_9pt(cls,L,q,boundary_condition,order=1,**compress_opts):
        ''' S+|x> = |x+1> , S-|x> = |x-1>
            df/dx = \sum_i (S- - S+)|x_i>a
            is for a 1D system so dim does not need to be specified
            scale by 1/dt later
            boundary_condition:  boundary condition to use when taking derivative

            FD coeffs found using python package FinDiff
        '''
        if q != 2:
            raise NotImplementedError('check q=2 implementation for binary mapping, esp if not pbc')

        if   order == 1:     c1,c2,c3,c4 = (1./2, 0, 0, 0)
        elif order == 2:     c1,c2,c3,c4 = (2./3, -1./12, 0, 0)
        elif order == 3:     c1,c2,c3,c4 = (3./4, -3./20, 1./60, 0)
        elif order == 4:     c1,c2,c3,c4 = (4./5, -1./5, 4./105, -1./280)

        #### build n-ary +/- operator ####
        ## define operator such that operating on the desired spatial dimension
        sm = np.diag([1,]*(q-1), k=-1)
        sp = np.diag([1,]*(q-1), k= 1)
        ### these are swapped, but was easier to change here than below

        ## build MPO
        iden = np.eye(q)
        zero = np.zeros((q,q))

        if boundary_condition in ['p','P','periodic']:
            mpo_tens = [ np.array( [ iden, sp+sm, sm+sp ] ) ]    ## ignore dummy size 1 bond
            for i in range(1,L-2):
                mpo_tens += [ np.array( [[ iden, sp,   sm   ],
                                         [ zero, sm,   zero ],
                                         [ zero, zero, sp   ]] ) ]
            mpo_tens += [ np.array( [[ iden, sp,   sm  , zero, zero ],
                                     [ zero, sm,   zero, iden, zero ],
                                     [ zero, zero, sp  , zero, iden ]] ) ]
            mpo_tens += [ np.array( [ c1*(sp-sm), (c1*sm+c2*iden+c3*sp), -(c1*sp+c2*iden+c3*sm),\
                                     (c4*iden+c3*sm), -(c4*iden+c3*sp )] ) ]
                        ## ignore dummy size 1 bond
            mpo = qtn.MatrixProductOperator( mpo_tens, shape='lrud', site_tag_id='B({})', 
                                                 upper_ind_id='o({})', lower_ind_id='i({})' )

        elif boundary_condition in ['ap','AP','antiperiodic']:
            mpo_tens = [ np.array( [ iden, sp-sm, sm-sp ] ) ]    ## ignore dummy size 1 bond
            for i in range(1,L-2):
                mpo_tens += [ np.array( [[ iden, sp,   sm   ],
                                         [ zero, sm,   zero ],
                                         [ zero, zero, sp   ]] ) ]
            mpo_tens += [ np.array( [[ iden, sp,   sm  , zero, zero ],
                                     [ zero, sm,   zero, iden, zero ],
                                     [ zero, zero, sp  , zero, iden ]] ) ]
            mpo_tens += [ np.array( [ c1*(sp-sm), (c1*sm+c2*iden+c3*sp), -(c1*sp+c2*iden+c3*sm),\
                                     (c4*iden+c3*sm), -(c4*iden+c3*sp )] ) ]
                        ## ignore dummy size 1 bond
            mpo = qtn.MatrixProductOperator( mpo_tens, shape='lrud', site_tag_id='B({})', 
                                                 upper_ind_id='o({})', lower_ind_id='i({})' )

        else:
            m0 = np.diag([1.] + [0.]*(q-1))
            m1 = np.diag([0.]*(q-1) + [1.])

            nb = 4 # int(np.ceil(np.log2(order*2+1)))   ## number of bits req to encode boundary condition
            bc0 = np.zeros((q**nb,q**nb))
            bc1 = np.zeros((q**nb,q**nb))

            def get_open_boundary_stencil(order):
                ''' function that returns open boundary condition stencils in matrix form 
                '''
                ## edge boundary conditions: use the same number of points in stencil
                if   order == 1:  # 3-pt stencil
                    end_bc_mat_0 = np.array([[-3./2, 2.-c1, -1./2-c2]])
                    end_bc_mat_L = end_bc_mat_0[:,::-1]*-1

                elif order == 2:  # 5-pt stencil
                    fwd0 = np.array([-25./12, 4., -3., 4./3, -1./4])     # forward finite difference at 0
                    gen1 = np.array([-1./4, -5./6, 3./2, -1./2, 1./12])  # FD stencil at -1,0,1,2,3

                    ## 000.. points
                    v0 = fwd0 - np.array([0., c1, c2, c3, c4])
                    v1 = gen1 - np.array([-c1, 0., c1, c2, c3])
                    end_bc_mat_0 = np.array([v0,v1])

                    # ## 111.. points
                    # v0  = -1*fwd0[::-1] + np.array([c4, c3, c2, c1, 0.])  # backward FD, -1*(c_array)
                    # v11 = -1*fwd1[::-1] + np.array([c3, c2, c1, 0, -c1])
                    # v12 = -1*cen1[::-1] + np.array([c3, c2, c1, 0., -c1])  
                    #        # center FD: transpose, then flip sign so appears shifted

                    ## 111.. points
                    end_bc_mat_L = np.array([v1[::-1],v0[::-1]])*-1

                elif order == 3:  # 7-pt stencil
                    fwd0 = np.array([-49./20, 6., -15./2, 20./3, -15./4, 6./5, -1./6]) # forward at 0
                    gen1 = np.array([-1/6, -77/60, 5/2, -5/3, 5/6, -1/4, 1/30])    # FD stencil -1, ..., 5
                    gen2 = np.array([1/30, -2/5, -7/12, 4/3, -1/2, 2/15, -1/60])   # FD stencil -2, ..., 4

                    v0 = fwd0 - np.array([0., c1, c2, c3, c4, 0., 0.])
                    v1 = gen1 - np.array([-c1, 0., c1, c2, c3, c4, 0.])
                    v2 = gen2 - np.array([-c2, -c1, 0., c1, c2, c3, c4])
                    end_bc_mat_0 = np.array([v0,v1,v2])
                    end_bc_mat_L = np.array([v2[::-1],v1[::-1],v0[::-1]])*-1

                elif order == 4:
                    fwd0 = np.array([-761./280, 8., -14., 56./3, -35./2, 56./5, -14./3, 8./7, -1./8]) # 0
                    gen1 = np.array([-1/8, -223/140, 7/2, -7/2, 35/12, -7/4, 7/10, -1/6, 1/56])       # 1
                    gen2 = np.array([1/56, -2/7, -19/20, 2, -5/4, 2/3, -1/4, 2/35, -1/168])           # 2
                    gen3 = np.array([-1/168, 1/14, -1/2, -9/20, 5/4, -1/2, 1/6, -1/28, 1/280])        # 3

                    v0 = fwd0 - np.array([0., c1, c2, c3, c4, 0., 0. ,0., 0.])
                    v1 = gen1 - np.array([-c1, 0., c1, c2, c3, c4, 0., 0., 0.])
                    v2 = gen2 - np.array([-c2, -c1, 0., c1, c2, c3, c4, 0., 0.])
                    v3 = gen3 - np.array([-c3, -c2, -c1, 0., c1, c2, c3, c4, 0.])
                    end_bc_mat_0 = np.array([v0, v1, v2, v3])
                    end_bc_mat_L = np.array([v3[::-1], v2[::-1], v1[::-1], v0[::-1]])*-1

                return end_bc_mat_0, end_bc_mat_L


            if isinstance(boundary_condition, tuple):    left_bc , right_bc = boundary_condition
            else:                                        left_bc = right_bc = boundary_condition

            ## left-hand side boundaries
            if left_bc in ['r','R','reflecting']:
                bc = np.array([[c1,c2,c3,c4],[c2,c3,c4,0.],[c3,c4,0.,0.],[c4,0.,0.,0.]])
                bc0[0:4,1:5] = bc

            elif left_bc in ['zero gradient','zg','ZG']:
                bc = np.array([[c1,c2,c3,c4],[c2,c3,c4,0.],[c3,c4,0.,0.],[c4,0.,0.,0.]])*-1
                bc0[0:4,1:5] = bc

            elif left_bc in ['a','A','absorbing']:
                bc = np.array([[c1,c2,c3,c4],[c2,c3,c4,0.],[c3,c4,0.,0.],[c4,0.,0.,0.]])
                bc0[0:4,1:5] = bc
                # bc0[1:5,0] = np.array([-c1,-c2,-c3,-c4])*-1   # pin element at ind 0 to 0
                # print(c1,c2,c3,c4)
                # print(bc0[:5,:5])

            elif left_bc in ['open','o','O']:
                end_bc_mat_0 = get_open_boundary_stencil(order)[0]
                nx,ny = end_bc_mat_0.shape
                bc0[:nx,:ny] = end_bc_mat_0

            ## right-hand side boundaries
            if right_bc in ['r','R','reflect','reflecting']:
                bc = np.array([[0.,0.,0.,c4],[0.,0.,c4,c3],[0.,c4,c3,c2],[c4,c3,c2,c1]]) * -1
                bc1[-4:,-5:-1] = bc

            elif right_bc in ['zero gradient','zg','ZG']:
                bc = np.array([[0.,0.,0.,c4],[0.,0.,c4,c3],[0.,c4,c3,c2],[c4,c3,c2,c1]]) 
                bc1[-4:,-5:-1] = bc

            elif right_bc in ['a','A','absorbing']:
                bc = np.array([[0.,0.,0.,c4],[0.,0.,c4,c3],[0.,c4,c3,c2],[c4,c3,c2,c1]]) * -1
                bc1[-4:,-5:-1] = bc
                # bc1[-5:-1,-1] = np.array([c4,c3,c2,c1])*-1   # pin element at ind L to 0
                # print(c1,c2,c3,c4)
                # print(bc1[-5:,-5:])

            elif right_bc in ['open','o','O']: 
                end_bc_mat_1 = get_open_boundary_stencil(order)[1]
                nx,ny = end_bc_mat_1.shape
                bc1[-nx:,-ny:] = end_bc_mat_1

            ## 0 side
            bc0 = BinaryMap.array_reshape(bc0, (q,)*nb*2)
            bc0_tens = qtn.Tensor( bc0, inds=tuple([f'o({i})' for i in range(nb)] + \
                                                   [f'i({i})' for i in range(nb)]) )
            bc0_mpo = helper.mpx_from_dense(bc0_tens,nb,('o({})','i({})'),site_tag_id='X({})')

            ## L-1 side
            bc1 = BinaryMap.array_reshape(bc1, (q,)*nb*2)
            bc1_tens = qtn.Tensor( bc1, inds=tuple([f'o({i})' for i in range(nb)] + \
                                                   [f'i({i})' for i in range(nb)]) )
            bc1_mpo = helper.mpx_from_dense(bc1_tens,nb,('o({})','i({})'),site_tag_id='X({})')

            ## leading 00... or 1... strings
            if L-nb >= 2:
                mpo_0 = qtn.MatrixProductOperator( 
                          [np.array([m0])] + [np.array([[m0]])]*(L-nb-2) + [np.array([m0])], 
                          shape='lrud', site_tag_id='X({})',upper_ind_id='o({})',lower_ind_id='i({})')
                mpo_1 = qtn.MatrixProductOperator( 
                          [np.array([m1])] + [np.array([[m1]])]*(L-nb-2) + [np.array([m1])], 
                          shape='lrud', site_tag_id='X({})',upper_ind_id='o({})',lower_ind_id='i({})')
            elif L-nb == 1:
                mpo_0 = qtn.TensorNetwork( [qtn.Tensor(m0,inds=('o(0)','i(0)'),tags=('X(0)',))] )
                mpo_0.view_as( qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
                               site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})' )
                mpo_1 = qtn.TensorNetwork( [qtn.Tensor(m1,inds=('o(0)','i(0)'),tags=('X(0)',))] )
                mpo_1.view_as( qtn.MatrixProductOperator, like=mpo_0, inplace=True)
            elif L-nb == 0:
                mpo_0 = qtn.TensorNetwork( [] )
                mpo_0.view_as( qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
                               site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})' )
                mpo_1 = qtn.TensorNetwork( [] )
                mpo_1.view_as( qtn.MatrixProductOperator, like=mpo_0, inplace=True)
            else:  
                raise ValueError('FD order too large for current grid size')

            helper.append_mpx(mpo_0,bc0_mpo,inplace=True)
            helper.append_mpx(mpo_1,bc1_mpo,inplace=True)

            ### the bulk mpo
            mpo_tens = [ np.array([ iden, sp, sm ]) ]  ## ignore dummy size 1 bond
            for i in range(1,L-2):
                mpo_tens += [ np.array([[ iden, sp,   sm   ],
                                        [ zero, sm,   zero ],
                                        [ zero, zero, sp   ]]) ]
            mpo_tens += [ np.array([[ iden, sp,   sm,   zero, zero ],
                                    [ zero, sm,   zero, iden, zero ],
                                    [ zero, zero, sp,   zero, iden ]]) ]
            mpo_tens += [ np.array([ c1*(sp-sm), (c1*sm+c2*iden+c3*sp), -(c1*sp+c2*iden+c3*sm), \
                                     (c4*iden+c3*sm), -(c4*iden+c3*sp) ]) ]    ## ignore size 1 bond

            mpo = qtn.MatrixProductOperator( mpo_tens, shape='lrud', site_tag_id='B({})', 
                                             upper_ind_id='o({})', lower_ind_id='i({})' )
            helper.add_MPO(mpo, mpo_0, inplace=True)
            helper.add_MPO(mpo, mpo_1, inplace=True)
            helper.compress(mpo,**compress_opts)

        return mpo


    @classmethod
    def _mpo_firstderivative_forward(cls,L,q,boundary_condition,order=1,**compress_opts):
        ''' S+|x> = |x+1> , S-|x> = |x-1>
            df/dx = \sum_i (S- - S+)|x_i>
            is for a 1D system so dim does not need to be specified
            scale by 1/dt later
            forward diff or backward diff
            boundary_condition:  boundary condition to use when taking derivative

            FD coeffs found using python package FinDiff
        '''
        print('forward FD mpo')

        if q != 2:
            raise NotImplementedError('check q=2 implementation for binary mapping, esp if not pbc')

        coeffs_fd = findiff.coefficients(deriv=1, acc=2*order) 
        c = coeffs_fd['forward']['coefficients']

        nc = len(c)
        nb = int(np.floor(np.log(nc-1)/np.log(q))+1)  ## number of sites at the end

        c = np.append(c, np.zeros(q**nb-nc))
        # c *= -1

        #### build n-ary +/- operator ####
        ## define operator such that operating on the desired spatial dimension
        sm = np.diag([1,]*(q-1), k= 1)
        sp = np.diag([1,]*(q-1), k=-1)

        ## build MPO
        iden = np.eye(q)
        zero = np.zeros((q,q))

        if   boundary_condition in ['p','P','periodic']:        sign = 1
        elif boundary_condition in ['ap','AP','antiperiodic']:  sign = -1
        else:                                                   sign = 0

        #### build finite difference MPO for PBC/APBC ####
        if nb == 2:
            if sign != 0:
                # mpo_tens = [ np.array( [ -iden, -sp, -sign*sm ] ) ]    ## ignore dummy size 1 bond
                # for i in range(1,L-nb):
                #     mpo_tens += [ np.array( [[ iden, sp,   zero ],
                #                              [ zero, sm,   zero ],
                #                              [ zero, zero, sm   ]] ) ]

                # mpo_tens += [ np.array( [[ iden, sp, zero ],
                #                          [ zero, sm, zero ],
                #                          [ zero, sm, iden ]] ) ]
                # mpo_tens += [ np.array( [ c[0]*iden + c[1]*sp,
                #                           c[1]*sm + c[2]*iden + c[3]*sp,
                #                           c[3]*sm ] ) ]

                mpo_tens = [ np.array( [ iden, sm, sign*sp ] ) ]    ## ignore dummy size 1 bond
                for i in range(1,L-nb):
                    mpo_tens += [ np.array( [[ iden, sm,   zero ],
                                             [ zero, sp,   zero ],
                                             [ zero, zero, sp   ]] ) ]

                mpo_tens += [ np.array( [[ iden, sm, zero ],
                                         [ zero, sp, zero ],
                                         [ zero, sp, iden ]] ) ]
                mpo_tens += [ np.array( [ c[0]*iden + c[1]*sm,
                                          c[1]*sp + c[2]*iden + c[3]*sm,
                                          c[3]*sp ] ) ]
            else:
                # mpo_tens = [ np.array( [ -iden, -sp ] ) ]    ## ignore dummy size 1 bond
                # for i in range(1,L-nb):
                #     mpo_tens += [ np.array( [[ iden, sp ],
                #                              [ zero, sm ]] ) ]

                # mpo_tens += [ np.array( [[ iden, sp ],
                #                          [ zero, sm ]] ) ]
                # mpo_tens += [ np.array( [ c[0]*iden + c[1]*sp,
                #                           c[1]*sm + c[2]*iden + c[3]*sp ] ) ]

                mpo_tens = [ np.array( [ iden, sm ] ) ]    ## ignore dummy size 1 bond
                for i in range(1,L-nb):
                    mpo_tens += [ np.array( [[ iden, sm ],
                                             [ zero, sp ]] ) ]

                mpo_tens += [ np.array( [[ iden, sm ],
                                         [ zero, sp ]] ) ]
                mpo_tens += [ np.array( [ c[0]*iden + c[1]*sm,
                                          c[1]*sp + c[2]*iden + c[3]*sm ] ) ]

            mpo = qtn.MatrixProductOperator( mpo_tens, shape='lrud', site_tag_id='B({})', 
                                                 upper_ind_id='o({})', lower_ind_id='i({})' )
        else:
            npts = q**L
            op = np.zeros((npts,)*2)
            for i in range(npts):
                for j in range(nc):
                    if i+j < npts:  op[i,i+j] = c[j]
                    elif sign:      op[i,(i+j)%npts] = c[j]*sign

            op_tensor = BinaryMap.array_reshape(op, (q,)*(2*L) )
            op_tensor = qtn.Tensor(op_tensor, inds=[f'o({i})' for i in range(L)] + \
                                                   [f'i({i})' for i in range(L)])
            mpo = helper.mpx_from_dense(op_tensor,L,('o({})','i({})'),site_tag_id='B{}')


        #### get boundary conditions (subtract existing coeffs used above) ####
        #### bc forward diff, only right bc matters ####

        if sign == 0:   # not periodic or antiperiodic, apply bc's

            coeffs_mat = np.zeros((q**nb,)*2)
            for i in range(q**nb):
                for j in range(nc):
                    try:   coeffs_mat[i,i+j] = c[j]
                    except(IndexError):  continue

            m0 = np.diag([1.] + [0.]*(q-1))
            m1 = np.diag([0.]*(q-1) + [1.])

            bc1 = np.zeros((q**nb,q**nb))

            def get_open_boundary_stencil_right(order):
                ''' function that returns open boundary condition stencils in matrix form 
                '''
                op = np.zeros((q**nb,)*2)
                for i in range(q**nb):
                    space = q**nb-i
                    offsets = np.append(np.arange((space-nc),0),np.arange(0,min(nc,space)))
                    coeffs = findiff.coefficients(deriv=1,offsets=offsets)
                    op[i,i+offsets[0]:i+offsets[-1]+1] = coeffs['coefficients']

                print('right op',op)
                print('coeffs',coeffs)

                op -= coeffs_mat

                # derivative = findiff.FinDiff(0,1.,1,acc=2*order)   # axis, dx, derivative, accuracy
                # try:
                #     op = derivative.matrix( (q**nb,) )  # scipy sparse matrix
                # except(IndexError):
                #     op = derivative.matrix( (q**nb+1,) )  # scipy sparse matrix
                #     print(op.toarray())
                #     exit()
                # op = op.toarray()
                # op -= coeffs_mat

                # op = BinaryMap.array_reshape(op, (q,)*(2*nb) )
                # op_tensor = qtn.Tensor(op, inds=[f'o({i})' for i in range(nb)] + \
                #                                 [f'i({i})' for i in range(nb)])
                # end_bc_mat_L  = helper.mpx_from_dense(op_tensor,nb,('o({})','i({})'),site_tag_id='B{}')

                return op # end_bc_mat_L


            if isinstance(boundary_condition, tuple):    left_bc , right_bc = boundary_condition
            else:                                        left_bc = right_bc = boundary_condition

            ## added on top of bulk mpo
            ## right-hand side boundaries
            if right_bc in ['r','R','reflect','reflecting']:
                for i in range(1,nc):    bc1[-nc+i,-i-1:-1] = c[nc-i:nc][::-1]*-1
                # for i in range(nc):      bc1[-1-i,-1] = -c[i]
                # bc = np.array([[0.,0.,0.,c4],[0.,0.,c4,c3],[0.,c4,c3,c2],[c4,c3,c2,c1]]) * -1

            elif right_bc in ['a','A','absorbing']:
                # for i in range(1,nc):
                #     bc1[-nc+i,-i-1:-1] = c[nc-i:nc][::-1]*-1
                #     print(i,c[nc-i:nc])
                # # for i in range(nc):      bc1[-1-i,-1] = c[i]

                op = np.zeros((q**nb,)*2)
                for i in range(q**nb):  ## assume only 1 ghost cell (absorbing bc set to 0)
                    if i < q**nb-2:     ## avoid putting stencil on ghost cell
                        space = q**nb-i-1
                        # offsets = np.append(np.arange((space-nc),0),np.arange(0,min(nc,space)))
                        offsets = np.arange(min(space-nc,0),min(nc,space))
                        coeffs = findiff.coefficients(deriv=1,offsets=offsets)
                        print(i,offsets)
                        op[i,i+offsets[0]:i+offsets[-1]+1] = coeffs['coefficients']
                    elif i == q**nb-2:  ## connects to ghost cell set to 0
                        offsets = np.arange(-nc+2,2)
                        print(i,offsets)
                        coeffs = findiff.coefficients(deriv=1,offsets=offsets)
                        op[i,i+offsets[0]:i+offsets[-1]+1] = coeffs['coefficients']
                    elif i == q**nb-1:  ## the ghost cell -> set derivative to 0
                        pass
                        # offsets = np.arange(-nc+1,1)
                        # print(i,offsets)
                        # coeffs = findiff.coefficients(deriv=1,offsets=offsets)
                        # op[i,i+offsets[0]:i+offsets[-1]+1] = coeffs['coefficients']

                # op = np.zeros((q**nb,)*2)
                print('right op',op[-nc:,-nc:])
                op -= coeffs_mat
                bc1 = op

                # bc1 = get_open_boundary_stencil_right(order)

                # print('right op',bc1[-nc:,-nc:])
                # print('c',c)


            elif right_bc in ['zero gradient','zg','ZG']:
                for i in range(1,nc):    bc1[-nc+i,-i-1:-1] = c[nc-i:nc][::-1]
                # bc = np.array([[0.,0.,0.,c4],[0.,0.,c4,c3],[0.,c4,c3,c2],[c4,c3,c2,c1]]) 
                # bc1[-4:,-5:-1] = bc

            elif right_bc in ['open','o','O']: 
                bc1 = get_open_boundary_stencil_right(order)
                # end_bc_mat_1 = get_open_boundary_stencil(order)
                # print(end_bc_mat_1.shape)
                # nx,ny = end_bc_mat_1.shape
                # bc1[-nx:,-ny:] = end_bc_mat_1


            ## L-1 side
            bc1 = BinaryMap.array_reshape(bc1, (q,)*nb*2)
            bc1_tens = qtn.Tensor( bc1, inds=tuple([f'o({i})' for i in range(nb)] + \
                                                   [f'i({i})' for i in range(nb)]) )
            bc1_mpo = helper.mpx_from_dense(bc1_tens,nb,('o({})','i({})'),site_tag_id='X({})')

            ## leading 00... or 1... strings
            if L-nb >= 2:
                mpo_1 = qtn.MatrixProductOperator( 
                          [np.array([m1])] + [np.array([[m1]])]*(L-nb-2) + [np.array([m1])], 
                          shape='lrud', site_tag_id='X({})',upper_ind_id='o({})',lower_ind_id='i({})')
            elif L-nb == 1:
                mpo_1 = qtn.TensorNetwork( [qtn.Tensor(m1,inds=('o(0)','i(0)'),tags=('X(0)',))] )
                mpo_1.view_as( qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
                               site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})' )
            elif L-nb == 0:
                mpo_1 = qtn.TensorNetwork( [] )
                mpo_1.view_as( qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
                               site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})' )
            else:  
                raise ValueError('FD order too large for current grid size')

            helper.append_mpx(mpo_1,bc1_mpo,inplace=True)
            helper.add_MPO(mpo, mpo_1, inplace=True)
            helper.compress(mpo,**compress_opts)

        return mpo


    @classmethod
    def _mpo_firstderivative_backward(cls,L,q,boundary_condition,order=1,**compress_opts):
        ''' S+|x> = |x+1> , S-|x> = |x-1>
            df/dx = \sum_i (S- - S+)|x_i>
            is for a 1D system so dim does not need to be specified
            scale by 1/dt later
            forward diff or backward diff
            boundary_condition:  boundary condition to use when taking derivative

            FD coeffs found using python package FinDiff
        '''
        if q != 2:
            raise NotImplementedError('check q=2 implementation for binary mapping, esp if not pbc')

        print('backward FD mpo')

        coeffs_fd = findiff.coefficients(deriv=1, acc=2*order) 
        c = coeffs_fd['backward']['coefficients'][::-1]

        nc = len(c)
        nb = int(np.floor(np.log(nc-1)/np.log(q))+1)  ## number of sites at the end
        c = np.append(c, np.zeros(q**nb-nc))

        #### build n-ary +/- operator ####
        ## define operator such that operating on the desired spatial dimension
        sm = np.diag([1,]*(q-1), k= 1)
        sp = np.diag([1,]*(q-1), k=-1)

        ## build MPO
        iden = np.eye(q)
        zero = np.zeros((q,q))

        if   boundary_condition in ['p','P','periodic']:        sign = 1
        elif boundary_condition in ['ap','AP','antiperiodic']:  sign = -1
        else:                                                   sign = 0

        #### build finite difference MPO for PBC/APBC ####
        if nb == 2:
            if sign != 0:
                mpo_tens = [ np.array( [ iden, sp, sign*sm ] ) ]    ## ignore dummy size 1 bond
                for i in range(1,L-nb):
                    mpo_tens += [ np.array( [[ iden, sp,   zero ],
                                             [ zero, sm,   zero ],
                                             [ zero, zero, sm   ]] ) ]

                mpo_tens += [ np.array( [[ iden, sp, zero ],
                                         [ zero, sm, zero ],
                                         [ zero, sm, iden ]] ) ]
                mpo_tens += [ np.array( [ c[0]*iden + c[1]*sp,
                                          c[1]*sm + c[2]*iden + c[3]*sp,
                                          c[3]*sm ] ) ]

            else:
                mpo_tens = [ np.array( [ iden, sp ] ) ]    ## ignore dummy size 1 bond
                for i in range(1,L-nb):
                    mpo_tens += [ np.array( [[ iden, sp ],
                                             [ zero, sm ]] ) ]

                mpo_tens += [ np.array( [[ iden, sp ],
                                         [ zero, sm ]] ) ]
                mpo_tens += [ np.array( [ c[0]*iden + c[1]*sp,
                                          c[1]*sm + c[2]*iden + c[3]*sp ] ) ]

            mpo = qtn.MatrixProductOperator( mpo_tens, shape='lrud', site_tag_id='B({})', 
                                                 upper_ind_id='o({})', lower_ind_id='i({})' )
            # mpo_tens = mpo.contract()
            # mpo_tens.transpose(*([f'o({x})' for x in range(L)] + [f'i({x})' for x in range(L)]),
            #                    inplace=True)
            # mpo_data = mpo_tens.data.reshape(q**L,q**L)
            # print(mpo_data[:q**nb,:q**nb])

        else:
            npts = q**L
            op = np.zeros((npts,)*2)
            for i in range(npts):
                for j in range(nc):
                    if i-j >= 0:     op[i,i-j] = c[j]
                    elif sign:       op[i,i-j] = c[j]*sign

            op_tensor = BinaryMap.array_reshape(op, (q,)*(2*L) )
            op_tensor = qtn.Tensor(op_tensor, inds=[f'o({i})' for i in range(L)] + \
                                                   [f'i({i})' for i in range(L)])
            mpo = helper.mpx_from_dense(op_tensor,L,('o({})','i({})'),site_tag_id='B{}')


        #### get boundary conditions (subtract existing coeffs used above) ####
        #### bc forward diff, only right bc matters ####

        if sign == 0:   # not periodic or antiperiodic, apply bc's

            coeffs_mat = np.zeros((q**nb,)*2)
            for i in range(q**nb):
                for j in range(nc):
                    if i-j < 0:   continue
                    else:         coeffs_mat[i,i-j] = c[j]
            # print('ceoffs mat', coeffs_mat)

            m0 = np.diag([1.] + [0.]*(q-1))
            m1 = np.diag([0.]*(q-1) + [1.])

            bc0 = np.zeros((q**nb,q**nb))

            def get_open_boundary_stencil_left(order):
                ''' function that returns open boundary condition stencils in matrix form 
                    left boundary
                '''
                op = np.zeros((q**nb,)*2)
                for i in range(q**nb):
                    offsets = np.append(np.arange(-min(i,nc-1),1),np.arange(1,nc-i))
                    # print(offsets)

                    coeffs = findiff.coefficients(deriv=1,offsets=offsets)
                    # print('coeffs',coeffs)
                    op[i,i+offsets[0]:i+offsets[-1]+1] = coeffs['coefficients']

                # print('op',op)

                op -= coeffs_mat

                # op = self.map.array_reshape(op, (q,)*(nb) )
                # op_tensor = qtn.Tensor(op, inds=[f'o({i})' for i in range(nb)] + \
                #                                 [f'i({i})' for i in range(nb)])
                # end_bc_mat_0  = helper.mpx_from_dense(op_tensor,nb,('o({})','i({})'),site_tag_id='B{}')

                return op # end_bc_mat_0


            if isinstance(boundary_condition, tuple):    left_bc , right_bc = boundary_condition
            else:                                        left_bc = right_bc = boundary_condition

            ## added on top of bulk mpo
            ## left-hand side boundaries
            if left_bc in ['r','R','reflecting']: # , 'a','A','absorbing']:
                for i in range(nc):     bc0[i,1:nc-i] = c[i+1:nc]*-1
                # bc = np.array([[c1,c2,c3,c4],[c2,c3,c4,0.],[c3,c4,0.,0.],[c4,0.,0.,0.]]) 
                ## (center diff right side coeffs)

            elif left_bc in ['a','A','absorbing']:
                # print(coeffs_mat)
                # for i in range(nc):
                #     bc0[i,1:nc-i] = c[i+1:nc]*-1
                #     print(i,c[i+1:nc])
                # print(bc0)
                # for i in range(nc):     bc0[i,0] = -c[i]
                # print(c)
                # print(bc0)
                # exit()

                op = np.zeros((q**nb,)*2)
                for i in range(q**nb):  ## assume only 1 ghost cell (absorbing bc set to 0)
                    if i >= 2:     ## avoid putting stencil on ghost cell
                        offsets = np.arange(-min(i-1,nc-1),max(nc-i+1,1))
                        print(i,offsets)
                        coeffs = findiff.coefficients(deriv=1,offsets=offsets)
                        op[i,i+offsets[0]:i+offsets[-1]+1] = coeffs['coefficients']
                    elif i == 1:  ## connects to ghost cell set to 0
                        offsets = np.arange(-1,nc-1)
                        print(i,offsets)
                        coeffs = findiff.coefficients(deriv=1,offsets=offsets)
                        op[i,i+offsets[0]:i+offsets[-1]+1] = coeffs['coefficients']
                    elif i == 0:  ## the ghost cell -> set derivative to 0
                        pass
                        # offsets = np.arange(0,nc)
                        # print(i,offsets)
                        # coeffs = findiff.coefficients(deriv=1,offsets=offsets)
                        # op[i,i+offsets[0]:i+offsets[-1]+1] = coeffs['coefficients']

                # print('left op',op[:nc,:nc])
                op -= coeffs_mat
                bc0 = op

                # bc0 = get_open_boundary_stencil_left(order)

                # print('left op',bc0[:nc,:nc])
                # print('c back',c)

            elif left_bc in ['zero gradient','zg','ZG']:
                for i in range(nc):     bc0[i,1:nc-i] = c[i+1:nc]  # skip 0 offset
                # bc = np.array([[c1,c2,c3,c4],[c2,c3,c4,0.],[c3,c4,0.,0.],[c4,0.,0.,0.]])*-1
                # bc0[0:4,1:5] = bc

            elif left_bc in ['open','o','O','a','A','absorbing']:
                # pass
                bc0 = get_open_boundary_stencil_left(order)
                # end_bc_mat_0 = get_open_boundary_stencil(order)
                # nx,ny = end_bc_mat_0.shape
                # bc0[:nx,:ny] = end_bc_mat_0


            ## 0 side
            bc0 = BinaryMap.array_reshape(bc0, (q,)*nb*2)
            bc0_tens = qtn.Tensor( bc0, inds=tuple([f'o({i})' for i in range(nb)] + \
                                                   [f'i({i})' for i in range(nb)]) )
            bc0_mpo = helper.mpx_from_dense(bc0_tens,nb,('o({})','i({})'),site_tag_id='X({})')


            ## leading 00... or 1... strings
            if L-nb >= 2:
                mpo_0 = qtn.MatrixProductOperator( 
                          [np.array([m0])] + [np.array([[m0]])]*(L-nb-2) + [np.array([m0])], 
                          shape='lrud', site_tag_id='X({})',upper_ind_id='o({})',lower_ind_id='i({})')
            elif L-nb == 1:
                mpo_0 = qtn.TensorNetwork( [qtn.Tensor(m0,inds=('o(0)','i(0)'),tags=('X(0)',))] )
                mpo_0.view_as( qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
                               site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})' )
            elif L-nb == 0:
                mpo_0 = qtn.TensorNetwork( [] )
                mpo_0.view_as( qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
                               site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})' )
            else:  
                raise ValueError('FD order too large for current grid size')

            helper.append_mpx(mpo_0,bc0_mpo,inplace=True)
            helper.add_MPO(mpo, mpo_0, inplace=True)
            helper.compress(mpo,**compress_opts)

        return mpo


    @classmethod
    def _mpo_secondderivative_9pt(cls,L,q,boundary_condition,order=1,**compress_opts):
        ''' for dim1=dim2=dim=0, since this is 1D
            S+|x> = |x+1> , S-|x> = |x-1>
            d2f/dx2 = \sum_i 1/2*(S- + S+ - 2I)|x_i>
            dim specifies which dimension in which shift is taken, indexed starting at 0
            scale by 1/dt later
            FD coeffs found using python package FinDiff
        '''
        if q != 2:
            raise NotImplementedError('check q=2 implementation for binary mapping, esp if not pbc')

        if   order == 1:     c0,c1,c2,c3,c4 = (-2, 1, 0, 0, 0)
        elif order == 2:     c0,c1,c2,c3,c4 = (-5./2, 4./3, -1./12, 0, 0)
        elif order == 3:     c0,c1,c2,c3,c4 = (-49/18, 3/2, -3/20, 1/90, 0)
        elif order == 4:     c0,c1,c2,c3,c4 = (-205/72, 8/5, -1/5, 8/315, -1/560)

        #### build n-ary +/- operator ####
        ## define operator such that operating on the desired spatial dimension
        sm = np.diag([1,]*(q-1), k= 1)
        sp = np.diag([1,]*(q-1), k=-1)

        ## build MPO
        iden = np.eye(q)
        zero = np.zeros((q,q))

        if boundary_condition in ['p','P','periodic']:    # 0000 <--> 1111
            mpo_tens = [ np.array( [ iden, sp+sm, sm+sp ] ) ]    ## ignore dummy size 1 bond
            for i in range(1,L-2):
                mpo_tens += [ np.array( [[ iden, sp,   sm   ],
                                         [ zero, sm,   zero ],
                                         [ zero, zero, sp   ]] ) ]
            mpo_tens += [ np.array( [[ iden, sp,   sm  , zero, zero ],
                                     [ zero, sm,   zero, iden, zero ],
                                     [ zero, zero, sp  , zero, iden ]] ) ]
            mpo_tens += [ np.array( [ c1*(sm+sp)+c0*iden, (c1*sm+c2*iden+c3*sp), (c1*sp+c2*iden+c3*sm),\
                                      (c4*iden+c3*sm), (c4*iden+c3*sp)] ) ] 
                        ## ignore dummy size 1 bond
            mpo = qtn.MatrixProductOperator( mpo_tens, shape='lrud', site_tag_id='B({})', 
                                                 upper_ind_id='o({})', lower_ind_id='i({})' )

        elif boundary_condition in ['ap','AP','antiperiodic']:
            mpo_tens = [ np.array( [ iden, sp-sm, sm-sp ] ) ]    ## ignore dummy size 1 bond
            for i in range(1,L-2):
                mpo_tens += [ np.array( [[ iden, sp,   sm   ],
                                         [ zero, sm,   zero ],
                                         [ zero, zero, sp   ]] ) ]
            mpo_tens += [ np.array( [[ iden, sp,   sm  , zero, zero ],
                                     [ zero, sm,   zero, iden, zero ],
                                     [ zero, zero, sp  , zero, iden ]] ) ]
            mpo_tens += [ np.array( [ c1*(sm+sp)+c0*iden, (c1*sm+c2*iden+c3*sp), (c1*sp+c2*iden+c3*sm),\
                                      (c4*iden+c3*sm), (c4*iden+c3*sp)] ) ] 
                        ## ignore dummy size 1 bond
            mpo = qtn.MatrixProductOperator( mpo_tens, shape='lrud', site_tag_id='B({})', 
                                                 upper_ind_id='o({})', lower_ind_id='i({})' )

        else:       # forward/backward diff, not center diff for 000, 111
            m0 = np.diag([1.] + [0.]*(q-1))
            m1 = np.diag([0.]*(q-1) + [1.])

            nb = 4 # int(np.ceil(np.log2(order*2+1)))   ## number of bits req to encode boundary condition
            bc0 = np.zeros((q**nb,q**nb))
            bc1 = np.zeros((q**nb,q**nb))

            def get_open_boundary_stencil(order):
                ''' function that returns open boundary condition stencils in matrix form 
                '''
                ## edge boundary conditions: use the same number of points in stencil
                if   order == 1:  # 3-pt stencil
                    end_bc_mat_0 = np.array([[2-c0, -5.-c1, 4.-c2, -1.-c3]])  
                    end_bc_mat_L = end_bc_mat_0[:,::-1]

                elif order == 2:  # 5-pt stencil
                    fwd0 = np.array([15/4, -77/6, 107/6, -13, 61/12, -5/6])  # forward FD at 0
                    gen1 = np.array([5/6, -5/4, -1/3, 7/6, -1/2, 1/12])      # FD stencil at -1,0,1,2,3,4

                    ## 000.. points
                    v0 = fwd0 - np.array([c0, c1, c2, c3, c4, 0.])
                    v1 = gen1 - np.array([c1, c0, c1, c2, c3, c4])
                    end_bc_mat_0 = np.array([v0,v1])

                    ## 111.. points
                    end_bc_mat_L = np.array([v1[::-1],v0[::-1]])

                elif order == 3:  # 7-pt stencil
                    fwd0 = np.array([469/90, -223/10, 879/20, -949/18, 41, -201/10, 1019/180, -7/10])  # 0
                    gen1 = np.array([7/10, -7/18, -27/10, 19/4, -67/18, 9/5, -1/2, 11/180])      # FD at 1
                    gen2 = np.array([-11/180, 107/90, -21/10, 13/18, 17/36, -3/10, 4/45, -1/90]) # FD at 2

                    v0 = fwd0 - np.array([c0, c1, c2, c3, c4, 0., 0., 0.])
                    v1 = gen1 - np.array([c1, c0, c1, c2, c3, c4, 0., 0.])
                    v2 = gen2 - np.array([c2, c1, c0, c1, c2, c3, c4, 0.])
                    end_bc_mat_0 = np.array([v0,v1,v2])
                    end_bc_mat_L = np.array([v2[::-1],v1[::-1],v0[::-1]])

                elif order == 4:
                    fwd0 = np.array([6515/1008, -4609/140, 5869/70, -6289/45, 6499/40, -265/2, 6709/90,\
                                     -967/35, 3407/560, -761/1260])   # forward FD at 0
                    gen1 = np.array([761/1260, 61/144, -201/35, 341/30, -1163/90, 411/40, -17/3,\
                                     1303/630, -9/20, 223/5040])           # 1
                    gen2 = np.array([-223/5040, 293/280, -395/252, -13/30, 83/40, -319/180, 59/60,\
                                     -5/14, 389/5040, -19/2520])
                    gen3 = np.array([19/2520, -67/560, 97/70, -89/36, 23/20, 7/40, -17/90, 11/140,\
                                     -1/56, 1/560])

                    v0 = fwd0 - np.array([c0, c1, c2, c3, c4, 0., 0. ,0., 0., 0.])
                    v1 = gen1 - np.array([c1, c0, c1, c2, c3, c4, 0., 0., 0., 0.])
                    v2 = gen2 - np.array([c2, c1, c0, c1, c2, c3, c4, 0., 0., 0.])
                    v3 = gen3 - np.array([c3, c2, c1, c0, c1, c2, c3, c4, 0., 0.])
                    end_bc_mat_0 = np.array([v0, v1, v2, v3])
                    end_bc_mat_L = np.array([v3[::-1], v2[::-1], v1[::-1], v0[::-1]])

                return end_bc_mat_0, end_bc_mat_L


            if isinstance(boundary_condition,tuple):    left_bc , right_bc = boundary_condition
            else:                                       left_bc = right_bc = boundary_condition

            ## left-hnad side boundaries
            if left_bc in ['r','R','reflecting']:
                bc = np.array([[c1,c2,c3,c4],[c2,c3,c4,0.],[c3,c4,0.,0.],[c4,0.,0.,0.]])*-1
                bc0[0:4,1:5] = bc

            elif left_bc in ['a','A','absorbing']:
                raise(NotImplementedError)
                pass

            elif left_bc in ['zero gradient','zg','ZG']:
                bc = np.array([[c1,c2,c3,c4],[c2,c3,c4,0.],[c3,c4,0.,0.],[c4,0.,0.,0.]])
                bc0[0:4,1:5] = bc

            elif left_bc in ['open','o','O']:
                end_bc_mat_0 = get_open_boundary_stencil(order)[0]
                nx,ny = end_bc_mat_0.shape
                bc0[:nx,:ny] = end_bc_mat_0

            ## right-hand side boundaries
            if right_bc in ['r','R','reflect','reflecting']:
                bc = np.array([[0.,0.,0.,c4],[0.,0.,c4,c3],[0.,c4,c3,c2],[c4,c3,c2,c1]])*-1
                bc1[-4:,-5:-1] = bc

            elif right_bc in ['a','A','absorbing']:
                raise(NotImplementedError)
                pass

            elif right_bc in ['zero gradient','zg','ZG']:
                bc = np.array([[0.,0.,0.,c4],[0.,0.,c4,c3],[0.,c4,c3,c2],[c4,c3,c2,c1]]) 
                bc1[-4:,-5:-1] = bc

            elif right_bc in ['open','o','O']: 
                end_bc_mat_1 = get_open_boundary_stencil(order)[1]
                nx,ny = end_bc_mat_1.shape
                bc1[-nx:,-ny:] = end_bc_mat_1

            ## 0 side
            bc0 = BinaryMap.array_reshape(bc0, (q,)*nb*2)
            bc0_tens = qtn.Tensor( bc0, inds=tuple([f'o({i})' for i in range(nb)] + \
                                                   [f'i({i})' for i in range(nb)]) )
            bc0_mpo = helper.mpx_from_dense(bc0_tens,nb,('o({})','i({})'),site_tag_id='X({})')

            ## L-1 side
            bc1 = BinaryMap.array_reshape(bc1, (q,)*nb*2)
            bc1_tens = qtn.Tensor( bc1, inds=tuple([f'o({i})' for i in range(nb)] + \
                                                   [f'i({i})' for i in range(nb)]) )
            bc1_mpo = helper.mpx_from_dense(bc1_tens,nb,('o({})','i({})'),site_tag_id='X({})')

            ## leading 00... or 1... strings
            if L-nb >= 2:
                mpo_0 = qtn.MatrixProductOperator( 
                          [np.array([m0])] + [np.array([[m0]])]*(L-nb-2) + [np.array([m0])], 
                          shape='lrud', site_tag_id='X({})',upper_ind_id='o({})',lower_ind_id='i({})')
                mpo_1 = qtn.MatrixProductOperator( 
                          [np.array([m1])] + [np.array([[m1]])]*(L-nb-2) + [np.array([m1])], 
                          shape='lrud', site_tag_id='X({})',upper_ind_id='o({})',lower_ind_id='i({})')
            elif L-nb == 1:
                mpo_0 = qtn.TensorNetwork( [qtn.Tensor(m0,inds=('o(0)','i(0)'),tags=('X(0)',))] )
                mpo_0.view_as( qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
                               site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})' )
                mpo_1 = qtn.TensorNetwork( [qtn.Tensor(m1,inds=('o(0)','i(0)'),tags=('X(0)',))] )
                mpo_1.view_as( qtn.MatrixProductOperator, like=mpo_0, inplace=True)
            elif L-nb == 0:
                mpo_0 = qtn.TensorNetwork( [] )
                mpo_0.view_as( qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
                               site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})' )
                mpo_1 = qtn.TensorNetwork( [] )
                mpo_1.view_as( qtn.MatrixProductOperator, like=mpo_0, inplace=True)
            else:  
                raise ValueError('FD order too large for current grid size')

            helper.append_mpx(mpo_0,bc0_mpo,inplace=True)
            helper.append_mpx(mpo_1,bc1_mpo,inplace=True)

            ### the bulk mpo
            mpo_tens = [ np.array([ iden, sp, sm ]) ]  ## ignore dummy size 1 bond
            for i in range(1,L-2):
                mpo_tens += [ np.array([[ iden, sp,   sm   ],
                                        [ zero, sm,   zero ],
                                        [ zero, zero, sp   ]]) ]
            mpo_tens += [ np.array([[ iden, sp,   sm,   zero, zero ],
                                    [ zero, sm,   zero, iden, zero ],
                                    [ zero, zero, sp,   zero, iden ]]) ]
            mpo_tens += [ np.array([ c1*(sm+sp)+c0*iden, (c1*sm+c2*iden+c3*sp), (c1*sp+c2*iden+c3*sm),\
                                     (c4*iden+c3*sm), (c4*iden+c3*sp) ]) ]    ## ignore size 1 bond

            mpo = qtn.MatrixProductOperator( mpo_tens, shape='lrud', site_tag_id='B({})', 
                                             upper_ind_id='o({})', lower_ind_id='i({})' )
            helper.add_MPO(mpo, mpo_0, inplace=True)
            helper.add_MPO(mpo, mpo_1, inplace=True)
            helper.compress(mpo,**compress_opts)

        return mpo


    @classmethod
    def _mpo_heaviside(cls,L,q,zero_ind,get='right'):
        ''' get mpo representing heaviside mpo function
            zero_ind specifies index at which the step occurs
            get: 'right','xright','left','xleft'
               right: inds >= zero_ind = 1 else 0
               left:  inds <= zero_ind = 1 else 0
               x:  excludes zero_ind
        '''
        npts = q**L

        if   get=='right':    vec = np.append(np.zeros((zero_ind,)),np.ones((npts-zero_ind,)))
        elif get=='xright':   vec = np.append(np.zeros((zero_ind+1,)),np.ones((npts-zero_ind-1,)))
        elif get=='left':     vec = np.append(np.ones((zero_ind+1,)),np.zeros((npts-zero_ind-1,)))
        elif get=='xleft':    vec = np.append(np.ones((zero_ind,)),np.zeros((npts-zero_ind,)))

        vec = BinaryMap.array_reshape(vec,(q,)*L)
        vec_tensor = qtn.Tensor(data=vec,inds=[f'i({x})' for x in range(L)])
        heaviside_mps = helper.mpx_from_dense(vec_tensor,L,('i({})',),site_tag_id='X({})')
        heaviside_mpo = helper.mps_to_diag_mpo(heaviside_mps, upper_ind_id='o({})', lower_ind_id='i({})')

        return heaviside_mpo


