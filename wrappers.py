import edrixs
import numpy as np
from math import sqrt
from solvers import ed_siam_petsc, rixs_siam_petsc

loc_axis = np.array([[sqrt(1/6), sqrt(1/6), -sqrt(2/3)],
                     [-sqrt(1/2), sqrt(1/2), 0],
                     [sqrt(1/3), sqrt(1/3), sqrt(1/3)]])

NiPS3_ed = dict(
shell_name = ('d', 'p'),
nd = 8,
norb_d = 10,
norb_bath = 10,
nbath = 1,
num_gs=1,   # with magnetic field break ground state 

F0_dd = 7.88,
F2_dd = 10.68,
F4_dd = 6.68,

F0_dp = 7.45,
F2_dp = 6.56,
G1_dp = 4.92,
G3_dp = 2.80,

# 10x10 orbital only matrix from DFT 5 d orbital 5 L orbitals
# in real harmonic basis with EDRIXS orbital order. 
H_MO = np.loadtxt('EDRIXS_cf_matrix.csv', delimiter=','),

mod_10dq = 0.0, # Change 10dq from DFT-based starting point

Delta = -0.7, # Force d-L splitting overriding DFT.

ext_B = np.array([0.028, 0, 0]),

c_level = -674.0,

zeta_d_i = 0.083,
zeta_d_n = 0.102,
c_soc = 11.4,

maxiter=1000,
eigval_tol=1e-8
)

NiPS3_rixs = dict(
shell_name=('d', 'p'),
nbath=1,
ominc = np.arange(850.5, 856, .25),
eloss = np.arange(-0.25, 2, 0.01),
gamma_f = 0.025,
gamma_c = 0.6,
v_noccu=18,
num_gs=3,
temperature=40,
loc_axis=loc_axis,
pol_type = [('linear', 0, 'linear', 0), ('linear', 0, 'linear', np.pi/2)],
thin = np.deg2rad(22.6),
thout = np.deg2rad(150 - 22.6),
phi = np.deg2rad(90.0),
nkryl=200,
linsys_max=1000,
linsys_tol=1e-10,
)



def ed_wrapper(comm,
    fortran = True,
    shell_name=None,
    nd=None,
    norb_d=None,
    norb_bath=None,
    nbath=None,
    num_gs=None,
    F0_dd=None,
    F2_dd=None,
    F4_dd=None,
    F0_dp=None,
    F2_dp=None,
    G1_dp=None,
    G3_dp=None,
    H_MO=None,
    Delta=None,
    mod_10dq=None,
    ext_B=None,
    c_level=None,
    zeta_d_i=None,
    zeta_d_n=None,
    c_soc=None,
    maxiter=None,
    eigval_tol=None
    ):
    """
    Wrapper function for RIXS calculation. Here we generate just the poles.
    This can be passed to 

    See:
    edrixs.solvers.ed_siam_fort
    and
    edrixs.solvers.rixs_siam_fort

    for parameter defintions and
    https://edrixs.github.io/edrixs/auto_examples/example_3_AIM_XAS.html#sphx-glr-auto-examples-example-3-aim-xas-py
    for explanation.
    """
        
    ## Electrons and active shells
    v_noccu  = nd + nbath*norb_d
    
    ## Coulomb interactions
    slater = ([F0_dd, F2_dd, F4_dd],  # initial
              [F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp])  # with core hole
    
    ## Energies for different shells
    U_dd = F0_dd - edrixs.get_F0('d', F2_dd, F4_dd)
    U_dp = F0_dp - edrixs.get_F0('dp', G1_dp, G3_dp)
    E_d, E_L = edrixs.CT_imp_bath(U_dd, Delta, nd)
    E_dc, E_Lc, E_p = edrixs.CT_imp_bath_core_hole(U_dd, U_dp, Delta, nd)
    
    # Set up the two fermion matrices for the initial and intermdiate states
    hopping_i = np.zeros((20,20), dtype=complex)
    hopping_n = np.zeros((20,20), dtype=complex)
    
    # Crystal field parameter
    inc = mod_10dq*3/5
    dec = -mod_10dq*2/5
    hopping_i_temp = H_MO + np.diag([inc, dec, dec, inc, dec, 0, 0, 0, 0, 0, ])
    hopping_n_temp = H_MO + np.diag([inc, dec, dec, inc, dec, 0, 0, 0, 0, 0, ])
    
    # Offsets to the ligand and d levels taking into account U and Delta
    E_d, E_L = edrixs.CT_imp_bath(U_dd, Delta, nd)
    E_dc, E_Lc, E_p = edrixs.CT_imp_bath_core_hole(U_dd, U_dp, Delta, nd)
    
    ## d onsite
    hopping_i_temp[:5,:5] += np.diag([E_d]*5)
    ## d onsite with a core hole
    hopping_n_temp[:5,:5] += np.diag([E_dc]*5)
    ## L onsite
    hopping_i_temp[5:,5:] += np.diag([E_L]*5)
    ## L onsite with a core hole
    hopping_n_temp[5:,5:] += np.diag([E_Lc]*5)
    
    # assign to matrices with spin 
    hopping_i[0:20:2,0:20:2] += hopping_i_temp
    hopping_i[1:20:2,1:20:2] += hopping_i_temp
    hopping_n[0:20:2,0:20:2] += hopping_n_temp
    hopping_n[1:20:2,1:20:2] += hopping_n_temp
    
    ## Add SOC
    hopping_i[:10,:10] += edrixs.cb_op(edrixs.atom_hsoc('d', zeta_d_i), edrixs.tmat_c2r('d', True))
    hopping_n[:10,:10] += edrixs.cb_op(edrixs.atom_hsoc('d', zeta_d_n), edrixs.tmat_c2r('d', True))
    
    siam_type = 1
    do_ed = 1
    ed_solver = 2 # 2
    neval = 3
    nvector = 3
    ncv = 100
    idump = True
    on_which = 'spin'
    trans_c2n = edrixs.tmat_c2r('d', True)

    if fortran == True:
        eval_i, denmat, noccu_gs = edrixs.ed_siam_fort(
            comm, shell_name, nbath, siam_type=siam_type,
            hopping=hopping_i, hopping_n=hopping_n, 
            c_level=c_level, c_soc=c_soc, slater=slater, ext_B=ext_B,
            on_which=on_which, trans_c2n=trans_c2n, v_noccu=v_noccu, do_ed=do_ed,
            ed_solver=ed_solver, neval=neval, nvector=nvector, ncv=ncv, idump=idump,
            maxiter=maxiter, eigval_tol=eigval_tol) 
    
        return eval_i, denmat, noccu_gs
    else:
        out = ed_siam_petsc(
            comm, shell_name, nbath, siam_type=siam_type,
            hopping=hopping_i, hopping_n=hopping_n, 
            c_level=c_level, c_soc=c_soc, slater=slater, ext_B=ext_B,
            on_which=on_which, trans_c2n=trans_c2n, v_noccu=v_noccu, do_ed=do_ed,
            ed_solver=ed_solver, neval=neval, nvector=nvector, ncv=ncv, idump=idump,
            maxiter=maxiter, eigval_tol=eigval_tol
        )
        eval_i, evec_i, emat_i, emat_n, umat_i, umat_n, = out
        return eval_i, evec_i, emat_i, emat_n, umat_i, umat_n


def rixs_wrapper(comm,
                 fortran=True,
                 shell_name=None,
                 nbath=None,
                 ominc=None,
                 eloss=None,
                 gamma_c=None,
                 gamma_f=None,
                 v_noccu=None,
                 thin=None,
                 thout=None,
                 phi=None,
                 pol_type=None,
                 num_gs=None,
                 temperature=None,
                 loc_axis=None,
                 nkryl=None,
                 linsys_max=None,
                 linsys_tol=None,
                 scatter_axis=None,
                 # passes for non-Fortran
                 # i.e. stuff passed not-very-transparently
                 # via disk before
                 eval_i=None,
                 evec_i=None,
                 emat_i=None,
                 umat_i=None,
                 emat_n=None,
                 umat_n=None,
                ):

    if fortran:
        rixs, poles = edrixs.rixs_siam_fort(
            comm, shell_name, nbath,
            ominc, eloss,
            gamma_c=gamma_c, gamma_f=gamma_f,
            v_noccu=v_noccu,
            thin=thin, thout=thout, phi=phi,
            pol_type=pol_type,
            num_gs=num_gs,
            temperature=temperature,
            loc_axis=loc_axis,
            nkryl=nkryl,
            linsys_max=linsys_max,
            linsys_tol=linsys_tol, 
            scatter_axis=scatter_axis
            )
    else:
        rixs, poles = rixs_siam_petsc(
            comm,
            eval_i, evec_i,
            emat_i, umat_i,
            emat_n, umat_n,
            shell_name, nbath, ominc, eloss, gamma_c=gamma_c, gamma_f=gamma_f,
            v_noccu=v_noccu, thin=thin, thout=thout, phi=phi, pol_type=pol_type, num_gs=num_gs,
            temperature=temperature, loc_axis=loc_axis, 
            nkryl=nkryl, linsys_max=linsys_max, linsys_tol=linsys_tol, 
            scatter_axis=scatter_axis)
        
    
    return rixs, poles

# rixs_petsc_wrapper to do
