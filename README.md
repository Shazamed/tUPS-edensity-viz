# tUPS-edensity-viz
Generate .cube files of the electron density of the a tUPS obtained wave function for visualisation using vmd.
## Description
The script tups.py generates the wave function using the parameters from the 'coords' file and outputs a one-electron reduced density matrix for the pyscf cubegen function.

The molecular orbital coefficients must be given either from pyscf RHF calculations or provided as 'mo_coeff' file.
## Usage
Navigate to the directory where the repository is cloned.

Copy the parameters from the converged tUPS calculation (eg. extractedmin from PATHSAMPLE) into a file named 'coords'. The number of parameters must be equal to the number of dimensions in the tUPS circuit.

Copy an .xyz file for the molecular geometry and change the path of the mol.build atom parameter to match the file in gencube.py

Copy mo_coeff into the same directory and comment out the pyscf RHF lines if MO coefficients are obtained elsewhere.

Run gencube.py with python to produce a density.cube file:
```
python gencube.py
```

Run VMD using the following and add the isosurface representation to visualise the electron density of the molecule:
```
vmd density.cube
```

or simply run the following to view from a saved visualisation state named view.vmd, which takes in density.cube:
```
vmd -e view.vmd
```


