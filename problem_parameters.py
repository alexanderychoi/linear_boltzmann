import numpy as np
import re

# BLOCK 1: DATA DIRECTORIES
# Input locations: Where is the data needed to run the calculations?
# Output locations: Where should we store the results of the calculations?

# For the same input data, there may be multiple simulations with different toggles (convergence parameters, FDM scheme, etc.)
# Each output is split into multiple subproblems with a text file detailing the relevant toggles

# Output structure:
# #_Problem
	# -> 1_Pipeline
		# -> 1_Subproblem
			# Toggle.txt
			# Steady Distributions
			# Transient Distributions
			# Effective Distributions
			# Small Signal Distributions
		# -> 2_Subproblem
		# -> ...
	# -> 2_Output
		# -> 1_Subproblem
			# Paper figures
		# -> 2_Subproblem
		# -> ...

# Peishi Local directories
subproblemVer = '1_Subproblem/'

parentdir = '/home/peishi/calculations/BoltzmannGreenFunctionNoise/'
# inputLoc = parentdir+'GaAs/14_Problem_Mode_Noise/0_Data/'
# outputLoc = parentdir+'GaAs/14_Problem_Mode_Noise/1_Pipeline/'+subproblemVer
# inputLoc = parentdir+'GaAs/13_Problem_Paper/0_Data/'
# outputLoc = parentdir+'GaAs/13_Problem_Paper/1_Pipeline/'+subproblemVer
# inputLoc = parentdir+'GaAs/11_Problem_Temperature/400K/0_Data/'
# outputLoc = parentdir+'GaAs/11_Problem_Temperature/400K/1_Pipeline/'+subproblemVer
inputLoc = parentdir+'GaAs/10_Problem_Validation/200_kpts_0.3eV_window/10_mev_smear/0_Data/'
outputLoc = parentdir+'GaAs/10_Problem_Validation/200_kpts_0.3eV_window/10_mev_smear/1_Pipeline/'+subproblemVer
# inputLoc = '/home/peishi/calculations/BoltzmannGreenFunctionNoise/Si/2_Problem_0.2eV/0_Data/'
# outputLoc = '/home/peishi/calculations/BoltzmannGreenFunctionNoise/Si/2_Problem_0.2eV/1_Pipeline/Output/'
figureLoc = parentdir+'GaAs/10_Problem_Validation/200_kpts_0.3eV_window/10_mev_smear/2_Output/'+subproblemVer+'PaperFigures/'


# Alex Dropbox directories
# subproblemVer = '1_Subproblem/'

# Grid used for plots in the paper as of 8/10/20:
# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/12_Problem_Paper/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/12_Problem_Paper/1_Pipeline/'+subproblemVer
# figureLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/12_Problem_Paper/2_Output/'+subproblemVer+'PaperFigures/'

# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/13_Problem_Paper/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/13_Problem_Paper/1_Pipeline/'+subproblemVer
# figureLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/13_Problem_Paper/2_Output/'+subproblemVer+'PaperFigures/'

# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/10_Problem_Validation/200_kpts_0.35eV_window/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/10_Problem_Validation/200_kpts_0.35eV_window/1_Pipeline/'+subproblemVer
# figureLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/10_Problem_Validation/200_kpts_0.35eV_window/2_Output/'+subproblemVer+'PaperFigures/'

# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/10_Problem_Validation/160_kpts_0.35eV_window/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/10_Problem_Validation/160_kpts_0.35eV_window/1_Pipeline/'+subproblemVer
# figureLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/10_Problem_Validation/160_kpts_0.35eV_window/2_Output/'+subproblemVer+'PaperFigures/'
# subproblemVer = '4_Subproblem/'

# Grid used for plots in the paper as of 7/27/20:
# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/6_Problem/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/6_Problem/1_Pipeline/'+subproblemVer
# figureLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/6_Problem/2_Output/'+subproblemVer+'PaperFigures/'


# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/5_Problem_080kpts_0.4eV/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/5_Problem_080kpts_0.4eV/1_Pipeline/'+subproblemVer
# figureLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/5_Problem_080kpts_0.4eV/2_Output/'+subproblemVer+'PaperFigures/'

# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/2_Problem_PERT_Lder/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/2_Problem_PERT_Lder/1_Pipeline/'+subproblemVer
# figureLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/2_Problem_PERT_Lder/2_Output/'+subproblemVer+'PaperFigures/'

# Grids used to test convergence:
# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/10_Problem_Validation/160_kpts_0.2eV_window/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/10_Problem_Validation/160_kpts_0.2eV_window/1_Pipeline/'+subproblemVer
# figureLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/10_Problem_Validation/160_kpts_0.2eV_window/2_Output/'+subproblemVer+'PaperFigures/'

# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/10_Problem_Validation/160_kpts_0.3eV_window/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/10_Problem_Validation/160_kpts_0.3eV_window/1_Pipeline/'+subproblemVer
# figureLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/10_Problem_Validation/160_kpts_0.3eV_window/2_Output/'+subproblemVer+'PaperFigures/'

#
# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/11_Problem_Temperature/400K/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/11_Problem_Temperature/400K/1_Pipeline/'+subproblemVer
# figureLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/11_Problem_Temperature/400K/2_Output/'+subproblemVer+'PaperFigures/'

# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/11_Problem_Temperature/200K/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/11_Problem_Temperature/200K/1_Pipeline/'+subproblemVer
# figureLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/11_Problem_Temperature/200K/2_Output/'+subproblemVer+'PaperFigures/'


# Old grids:
# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/7_Problem/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/7_Problem/1_Pipeline/Output_V2/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/GaAs/7_Problem/1_Pipeline/Output_V3/'

# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/Si/2_Problem_0.2eV/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/Si/2_Problem_0.2eV/1_Pipeline/Output_V1/'


# BLOCK 2: PHYSICAL PROBLEM PARAMETERS: FIELDS AND FREQUENCIES
# fieldVector = np.array([1e-3,1e4,4e4])							# GaAs fields for noise plots
# fieldVector =np.geomspace(1e2,4e4,20)  								# GaAs fields for moment plots
moment_fields = np.geomspace(1e2, 5e4, 20)
# small_signal_fields = np.array([1e-3, 1e4, 4e4])
small_signal_fields = np.array([1e-3,1e4,5e4])
# fieldVector = np.unique(np.concatenate((moment_fields,small_signal_fields)))
# fieldVector = np.array([1e-3, 1e4, 4e4])							# GaAs fields for noise plots
# fieldVector =np.geomspace(1e2,4e4,20)  							# GaAs fields for moment plots
fieldVector = np.array([1e4, 5e4])										# Highest field for paper, convergence tests
# fieldVector =np.geomspace(1e2,4e4,20)  								# GaAs fields for moment plots
moment_fields = np.linspace(1e2, 5e4, 20)
# moment_fields = np.geomspace(1e2,4e4,20)
small_signal_fields = np.array([1e-3, 1e4, 5e4])
# fieldVector = np.unique(np.concatenate((moment_fields,small_signal_fields)))

# fieldVector =np.array([1e2,1e4,4e4])  							# GaAs fields for small signal freq plot
# fieldVector = np.array([1e2, 1e3, 1e4, 2e4, 4e4, 6e4,1e5,1.5e5,2e5,3e5]) 	# Si Fields

# Field direction (for now only implemented for Si, not GaAs)
fieldDirection = np.array([1, 1, 1])  								# Crystal direction of field orientation (x,y,z)

freqVector = np.geomspace(0.1,10000,30)  							# GaAs freqs for small signal freq plot
# freqVector = np.geomspace(0.1,10000,15)
# freqVector = np.array(freqVector[0:3])
# freqVector = np.array([0.1])										# Low freq for testing
# freqVector = np.array([1,5,10,50,100]) 							# Si frqs
freqGHz = freqVector[0]


# BLOCK 3: SIMULATION PARAMETERS
relConvergence = 1e-3  									# Convergence parameters for the GMRES solver
absConvergence = 1e-60  								# Convergence parameters for the GMRES solver
verboseError = False  									# Do we want to calculate and store the residual error of GMRES?


# BLOCK 4: MATRIX PARAMETERS
scmName = 'scatt_mat_pert.mmap'  # Name of the scattering matrix
scmBool = False  # Should we apply an arbitrary correction factor to the scattering rates?
scmVal = (2*np.pi)**2/1.5*1.2298  # The value of the linear correction factor
simpleBool = True  # Is the matrix in the simple linearization (on-diagonal RTs)
fdmName = 'Column Preserving Central Difference'  # What stencil are we using for the finite difference scheme?
# fdmName = 'Hybrid Difference'
# fdmName = 'Backwards Difference'


# BLOCK 5: TEXT FILE PROBLEM PARAMETERS
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
f.close()

# BLOCK 6: MATERIAL SPECIFIC PARAMETERS
# GaAs-specific Parameters
if prefix == 'gaas':
	getX = False  # Do we have X valleys?
	derL = False  # Should we apply the central difference scheme in the L valleys?
# Si-specific Parameters
if prefix == 'si':
	pass

# BLOCK 7: STRING FOR PLOT TITLES (NOT PAPER FIGURES)
title_str = '{:s} Grid={:.0f}^3, mu={:.4f} eV, {:.1f} K, Emax={:.3f} eV'.format(prefix, kgrid, mu, T, cutoff)

print('\nIs matrix is presumed to be in simple linearization? {:s}'.format(str(simpleBool)))
print('Will an arbitrary correction factor be applied to the matrix? {:s}'.format(str(scmBool)))
print('The name of the scattering matrix to be loaded is "{:s}"\n'.format(scmName))