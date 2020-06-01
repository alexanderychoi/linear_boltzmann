import pandas
import os
import numpy as np
import re

f = open('problem_parameters.txt')

alltext = f.read()
scratchLoc = re.findall(r"\s*scratchLoc\s*=\s*'(.+)'\n", alltext)
T = float(re.findall(r"\s*Temperature\s*=\s*(.+)\n", alltext)[0])
mu = float(re.findall(r"\s*FermiLevel\s*=\s*(.+)\n", alltext)[0])
b = float(re.findall(r"\s*GaussianBroadening\s*=\s*(.+)\n", alltext)[0])
kgrid = float(re.findall(r"\s*GridDensity\s*=\s*(\d+)\n", alltext)[0])
cutoff = float(re.findall(r"\s*EnergyWindow\s*=\s*(.+)\n", alltext)[0])

if scratchLoc:
	scratchLoc = scratchLoc[0]
	print('Scratch location is \'' + scratchLoc + '\'')

# Problem setup. Fields, frequencies, valleys.
fieldVector = np.array([1e2, 1e3, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5,2e5,2.5e5,3e5,3.5e5,4e5,4.5e5])
freqGHz = 5
getX = False
inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/2_Problem_PERT/0_Data/'
outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/2_Problem_PERT/1_Pipeline/Output/'

print('\nData for this run loaded from "{:s}"\n'.format(inputLoc))
print('Lattice temperature is {:.1f} K'.format(T))
print('Fermi Level is {:.5f} eV'.format(mu))
print('Gaussian broadening is {:.1e} eV'.format(b))
print('Grid density is {:.1f} cubed'.format(kgrid))
if scratchLoc:
	scratchLoc = scratchLoc[0]
	print('Scratch location is \'' + scratchLoc + '\'')

# Is the matrix in the inputLoc canonical or simple? False = Canonical linearization.
simpleBool = True
# Should we apply the scmVal factor to the scattering rates as a correction? False = Don't apply factor.
scmBool = False
scmVal = (2*np.pi)**2*2
# Convergence parameters for the GMRES solver
relConvergence = 5e-3
absConvergence = 1e-60
# Should we calculate and store the residual error?
verboseError = False
# What's the name of the matrix?
scmName = 'scattering_matrix.mmap'
# String for title of plots
title_str = 'Grid={:.0f}^3, mu={:.4f} eV, {:.1f} K, Emax={:.3f} eV'.format(kgrid, mu, T, cutoff)

print('\nIs matrix is presumed to be in simple linearization? {:s}'.format(str(simpleBool)))
print('Will an arbitrary correction factor be applied to the matrix? {:s}'.format(str(scmBool)))
print('The name of the scattering matrix to be loaded is "{:s}"\n'.format(scmName))
