import pyscf
import numpy as np
from tups import T_UPS
from pyscf.tools import cubegen

# create mol object
mol = pyscf.gto.Mole()
mol.build(
    atom="h6.xyz",
    basis="sto-3g",
)

# RHF MO coefficients may differ!!!

# run RHF for Cmo if pyscf is used
mf = pyscf.scf.RHF(mol)
mf.kernel()
Cmo = mf.mo_coeff

# OR obtain Cmo from file if mo coefficients obtained somewhere else
Cmo = np.fromfile(f"mo_coeff") # flattened file from ndarray.flatten(order='F').tofile("mo_coeff")
Cmo = Cmo.reshape((6,6),order='F')

# create tUPS object
tUPS = T_UPS(mol, Cmo, include_dmat=True, layers=2, oo=False, pp=True, include_first_singles=False)

coord = "./coords2"
x = np.genfromtxt(coord)
tUPS.take_step(x) # apply the rotation angles

rdm1 = tUPS.spat_rdm1_ao()

path = "./density.cube" # path for density.cube

cubegen.density(mol, path, rdm1)