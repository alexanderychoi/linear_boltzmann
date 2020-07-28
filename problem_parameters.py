import pandas
import os
import numpy as np
import re

# Problem setup. Fields, frequencies, valleys.
# fieldVector = np.array([1e2, 1e3, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5,2e5,2.5e5,3e5,3.5e5,4e5,4.5e5])
# fieldVector = np.array([1e-1,1e0,1e1,1e2,1e3,1e4])
fieldVector = np.geomspace(1e-1,1e4,25)

# fieldVector = np.array([1e2, 1e3, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4])

# fieldVector = np.array([2e5,2.5e5,3e5])

# fieldVector = np.array([2e5])
# fieldVector = np.array([1e2,1e3,1e4,1e5,2e5])
# fieldVector = np.array([5e4])
# fieldVector = np.array([1e2, 1e3, 1e4, 2e4, 3e4, 4e4])
freqGHz = 1
getX = False
derL = True

# inputLoc = '/home/peishi/calculations/BoltzmannGreenFunctionNoise/GaAs/9_Problem_f_steady_matrix/0_Data/'
# outputLoc = '/home/peishi/calculations/BoltzmannGreenFunctionNoise/GaAs/5_Problem_f_steady_matrix/1_Pipeline/Output/'

inputLoc = '/home/peishi/calculations/BoltzmannGreenFunctionNoise/GaAs/10_Problem_Validation/160_kpts_0.3eV_window/0_Data/'
outputLoc = '/home/peishi/calculations/BoltzmannGreenFunctionNoise/GaAs/10_Problem_Validation/160_kpts_0.3eV_window/1_Pipeline/Output/'
# inputLoc = '/home/peishi/calculations/BoltzmannGreenFunctionNoise/4_Problem_160kpts_0.45eV/0_Data/'
# outputLoc = '/home/peishi/calculations/BoltzmannGreenFunctionNoise/4_Problem_160kpts_0.45eV/1_Pipeline/Output/'

# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/2_Problem_PERT/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/2_Problem_PERT/1_Pipeline/Output/'

# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/2_Problem_PERT_Lder/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/2_Problem_PERT_Lder/1_Pipeline/Output/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/2_Problem_PERT_Lder/1_Pipeline/Output_hFDM/'

# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/2_Problem/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/2_Problem/1_Pipeline/Output/'

# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/6_Problem/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/6_Problem/1_Pipeline/Output_V1/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/6_Problem/1_Pipeline/Output_V2/'

# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/8_Problem/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/8_Problem/1_Pipeline/Output_V1/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/8_Problem/1_Pipeline/Output_V2/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/8_Problem/1_Pipeline/Output_V3/'

# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/6_Problem_Si/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/6_Problem_Si/1_Pipeline/Output_V1/'
# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/2_Problem/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/2_Problem/1_Pipeline/Output/'

f = open(inputLoc+'problem_parameters.txt')
alltext = f.read()
scratchLoc = re.findall(r"\s*scratchLoc\s*=\s*'(.+)'\n", alltext)
T = float(re.findall(r"\s*Temperature\s*=\s*(.+)\n", alltext)[0])
mu = float(re.findall(r"\s*FermiLevel\s*=\s*(.+)\n", alltext)[0])
b = float(re.findall(r"\s*GaussianBroadening\s*=\s*(.+)\n", alltext)[0])
kgrid = float(re.findall(r"\s*GridDensity\s*=\s*(\d+)\n", alltext)[0])
cutoff = float(re.findall(r"\s*EnergyWindow\s*=\s*(.+)\n", alltext)[0])
prefix = re.findall(r"\s*Prefix\s*=\s*'(.+)'", alltext)[0]
print('\nData for this run loaded from "{:s}"\n'.format(inputLoc))
print('Material prefix is {:s}'.format(prefix))
print('Lattice temperature is {:.1f} K'.format(T))
print('Fermi Level is {:.5f} eV'.format(mu))
print('Gaussian broadening is {:.1e} eV'.format(b))
print('Grid density is {:.1f} cubed'.format(kgrid))
if scratchLoc:
	scratchLoc = scratchLoc[0]
	print('Scratch location is \'' + scratchLoc + '\'')
# Do we want to use the new hybrid FDM scheme?
hybridFDM = False
# Is the matrix in the inputLoc canonical or simple? False = Canonical linearization.
simpleBool = True
# Should we apply the scmVal factor to the scattering rates as a correction? False = Don't apply factor.
scmBool = False
scmVal = (2*np.pi)**2/1.5*1.2298  # Problem 2
# Convergence parameters for the GMRES solver
relConvergence = 1e-3
absConvergence = 1e-60
# Should we calculate and store the residual error?
verboseError = True
# What's the name of the matrix?
# scmName = 'scatt_mat_pert.mmap'  # Problem 2 Pert
# scmName = 'scattering_matrix_simple_2.mmap'  # Problem 2
scmName = 'scatt_mat_pert.mmap'  # Problem 2

# String for title of plots
title_str = 'Case 8: Grid={:.0f}^3, mu={:.4f} eV, {:.1f} K, Emax={:.3f} eV'.format(kgrid, mu, T, cutoff)

print('\nIs matrix is presumed to be in simple linearization? {:s}'.format(str(simpleBool)))
print('Will an arbitrary correction factor be applied to the matrix? {:s}'.format(str(scmBool)))
print('The name of the scattering matrix to be loaded is "{:s}"\n'.format(scmName))
