import pyscf
import numpy as np
from tups import T_UPS
from pyscf.tools import cubegen

# create mol object
mol = pyscf.gto.Mole()
mol.build(
    atom="h6_triangle.xyz",
    basis="sto-3g",
)

# RHF MO coefficients may differ!!!

# run RHF for Cmo if pyscf is used
mf = pyscf.scf.RHF(mol)
mf.kernel()
Cmo = mf.mo_coeff
v_nuc = mol.energy_nuc()
print(f"Nuclear energy: {v_nuc}")
# OR obtain Cmo from file if mo coefficients obtained somewhere else
# Cmo = np.fromfile(f"mo_coeff") # flattened file from ndarray.flatten(order='F').tofile("mo_coeff")
# Cmo = Cmo.reshape((6,6),order='F')

# create tUPS object
tUPS = T_UPS(mol, Cmo, include_dmat=True, layers=1, oo=True, pp=True, include_first_singles=True)

coord = "./coords"
x = np.genfromtxt(coord)
tUPS.take_step(x) # apply the rotation angles
print(f"Total E: {tUPS.energy}")
print(f"Electronic E: {tUPS.energy - v_nuc}")
rdm1 = tUPS.spat_rdm1_ao()
print("1-electron RDM:")
print(rdm1)
path = "./density.cube" # path for density.cube

cubegen.density(mol, path, rdm1)
