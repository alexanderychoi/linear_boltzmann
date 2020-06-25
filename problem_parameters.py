import numpy as np
import re

# Input locations: Where is the data needed to run the calculations?
# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/6_Problem_Si/0_Data/'
inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/Si/2_Problem_0.2eV/0_Data/'

# inputLoc = '/home/peishi/calculations/BoltzmannGreenFunctionNoise/GaAs/5_Problem_080kpts_0.4eV/0_Data/'
# outputLoc = '/home/peishi/calculations/BoltzmannGreenFunctionNoise/GaAs/5_Problem_080kpts_0.4eV/1_Pipeline/Output/'

inputLoc = '/home/peishi/calculations/BoltzmannGreenFunctionNoise/Si/2_Problem_0.2eV/0_Data/'
outputLoc = '/home/peishi/calculations/BoltzmannGreenFunctionNoise/Si/2_Problem_0.2eV/1_Pipeline/Output/'

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
# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/6_Problem/0_Data/'
# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/7_Problem/0_Data/'
# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/4_Problem_160kpts_0.45eV/0_Data/'
# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/6_Problem/0_Data/'

# Output locations: Where should the data generated by the calculations be stored?
outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/6_Problem_Si/1_Pipeline/Output_V1/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/8_Problem/1_Pipeline/Output_V2/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/6_Problem/1_Pipeline/Output_V2/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/6_Problem/1_Pipeline/Output_V3/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/6_Problem/1_Pipeline/Output_V4/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/6_Problem/1_Pipeline/Output_V5/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/7_Problem/1_Pipeline/Output_V1/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/4_Problem_160kpts_0.45eV/1_Pipeline/Output_V3/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/6_Problem/1_Pipeline/Test/'


# Physical problem parameters: Fields and Frequencies
fieldVector = np.array([1e2, 1e3, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5,2e5,2.5e5,3e5,3.5e5,4e5,4.5e5])
fieldVector = np.array([1e2, 1e3, 1e4, 5e4, 1e5,2e5,2.5e5, 4.5e5])
fieldVector = np.array([2.5e5, 4.5e5])
# fieldVector = np.array([1e2,4.5e5])
# fieldVector = np.geomspace(1e-1,1e4,25)  # Electric field magnitude in V/m
# fieldVector = np.array([1e2,1e3,1e4,1e5,2e5])
# fieldVector = np.array([1e-2,1e-1,1,1e1,1e2])
freqVector = np.array([1])  # Frequencies in GHz
fieldDirection = np.array([1,0,0])  # Crystal direction of field orientation (x,y,z)
freqGHz = 1
# Simulation parameters
relConvergence = 1e-4  # Convergence parameters for the GMRES solver
absConvergence = 1e-60  # Convergence parameters for the GMRES solver
verboseError = True  # Do we want to calculate and store the residual error of GMRES? (Increases expense)

# Matrix parameters
scmName = 'scatt_mat_pert.mmap'  # Name of the scattering matrix
scmBool = False  # Should we apply an arbitrary correction factor to the scattering rates?
scmVal = (2*np.pi)**2/1.5*1.2298  # The value of the linear correction factor
simpleBool = True  # Is the matrix in the simple linearization (on-diagonal RTs)
fdmName = 'Column Preserving Central Difference'  # What stencil are we using for the finite difference scheme?
# fdmName = 'Hybrid Difference'
# fdmName = 'Backwards Difference'

# Text file Problem Parameters
f = open(inputLoc+'problem_parameters.txt')
alltext = f.read()
scratchLoc = re.findall(r"\s*scratchLoc\s*=\s*'(.+)'\n", alltext)
T = float(re.findall(r"\s*Temperature\s*=\s*(.+)\n", alltext)[0])
mu = float(re.findall(r"\s*FermiLevel\s*=\s*(.+)\n", alltext)[0])
mu = 6.4
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
f.close()

# String for title of plots
title_str = 'Case 6_Si: Grid={:.0f}^3, mu={:.4f} eV, {:.1f} K, Emax={:.3f} eV'.format(kgrid, mu, T, cutoff)

# GaAs-specific Parameters
if prefix == 'gaas':
	getX = True  # Do we have X valleys?
	derL = True  # Should we apply the central difference scheme in the L valleys?

# Si-specific Parameters
if prefix == 'si':
	pass

print('\nIs matrix is presumed to be in simple linearization? {:s}'.format(str(simpleBool)))
print('Will an arbitrary correction factor be applied to the matrix? {:s}'.format(str(scmBool)))
print('The name of the scattering matrix to be loaded is "{:s}"\n'.format(scmName))