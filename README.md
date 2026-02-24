# tUPS-edensity-viz
Generate .cube files of the electron density of the a tUPS obtained wave function for visualisation using vmd.
## Description
The script tups.py generates the wave function using the parameters from the 'coords' file and outputs a one-electron reduced density matrix for the pyscf cubegen function.

The molecular orbital coefficients must be given either from pyscf RHF calculations or provided as 'mo_coeff' file.
## Usage
Navigate to the directory where the repository is cloned.

Copy the parameters from the converged tUPS calculation (eg. extractedmin from PATHSAMPLE) into a file named 'coords'.

Copy an .xyz file for the molecular geometry and change the path of the mol.build atom parameter to match the file in gencube.py

Copy mo_coeff into the same directory and comment out the pyscf RHF lines. 

Run gencube.py with python:
```
python gencube.py
```
