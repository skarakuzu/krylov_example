import numpy as np
#import matplotlib.pyplot as plt
from wrappers import ed_wrapper, rixs_wrapper, NiPS3_ed, NiPS3_rixs
from datetime import datetime
import time
from mpi4py import MPI
#from petsc4py import PETSc


comm = MPI.COMM_WORLD
#petsc_comm = PETSc.COMM_WORLD

rank = comm.rank

# four cores on mark's m2 macbook via docker. 
#nd = 5 ED 21 s RIXS 72
#nd = 4 ED 44 s RIXS 223
#nd = 3 ED 80 s RIXS 600 s
nd = 3
#nd = 8
if rank == 0:
    print(f"Start nd={nd}")
NiPS3_ed['nd'] = nd
NiPS3_rixs['v_noccu'] = NiPS3_ed['nd'] + NiPS3_ed['nbath']*NiPS3_ed['norb_d']
NiPS3_rixs['ominc'] = np.arange(850.5, 856, 2)
NiPS3_rixs['pol_type'] = [('linear', 0, 'linear', 0)]


def log(message, start_time=time.time()):
    """
    Print message with:
      - Current wall-clock time
      - Elapsed time since script start
    Only prints on MPI rank 1.
    """
    if rank == 0:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - start_time
        print(f"### {message}: [{now}] (+{elapsed:8.3f}s)", flush=True)
        return elapsed

#t0 = log("ED fortran start")
print("ED fortran starts...", flush=True)
t0 = time.perf_counter()
eval_i_F, denmat, noccu_gs = ed_wrapper(comm, **NiPS3_ed)
t1 = time.perf_counter()
duration = t1 - t0
print(f"ED fortran took about {duration:.6f} seconds", flush=True)
#t1 = log("ED fortran finish")

#t2 = log("ED python start")
print("ED python starts...", flush=True)
t2 = time.perf_counter()
out = ed_wrapper(comm, fortran=False, **NiPS3_ed)
eval_i, evec_i, emat_i, emat_n, umat_i, umat_n = out
#t3 = log("ED python finish")
t3 = time.perf_counter()
duration = t3 - t2
print(f"ED python took about {duration:.6f} seconds", flush=True)

#t4 = log("RIXS fortran start")
print("RIXS fortran start")
t4 = time.perf_counter()
rixs_F, poles_F = rixs_wrapper(comm, fortran=True, **NiPS3_rixs)
print("RIXS fortran finish")
t5 = time.perf_counter()
#t5 = log("RIXS fortran finish")

#t6 = log("RIXS python start")
print("RIXS python start")
t6 = time.perf_counter()
rixs, poles = rixs_wrapper(comm, fortran=False, **NiPS3_rixs,
             eval_i=eval_i,
             evec_i=evec_i,
             emat_i=emat_i,
             umat_i=umat_i,
             emat_n=emat_n,
             umat_n=umat_n,
            )
print("RIXS python finish")
t7 = time.perf_counter()
#t7 = log("RIXS python finish")

#print("Trying to print results now, python: ")
#print(rixs[0])
#print("Trying to print results now, fortran: ")
#print(rixs_F[0])
print("Trying to print sum of difference now: ")
print(np.sum(abs(rixs_F[0] - rixs[0])))


np.testing.assert_allclose(rixs, rixs_F, atol=1e-4)

if rank == 0:
    print(f"Doing nd={nd}")
    print(f"ED \t F={(t1-t0):8.3f}  s \t  P={(t3-t2):8.3} s")
    print(f"RIXS \t F={(t5-t4):8.3f} s \t P={(t7-t6):8.3} af")
