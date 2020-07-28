import pandas
import numpy as np

# Change this directory to point at the problem-specific input folder (containing 'ProblemParameters.txt')
# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/5_Problem_080kpts_0.4eV/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/5_Problem_080kpts_0.4eV/1_Pipeline/'

# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/4_Problem_160kpts_0.45eV/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/4_Problem_160kpts_0.45eV/1_Pipeline/Output/'


# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/3_Problem/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/3_Problem/1_Pipeline/Output/'


inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/2_Problem_PERT/0_Data/'
outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/2_Problem_PERT/1_Pipeline/Output/'

# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/2_Problem/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/2_Problem/1_Pipeline/Output/'

# inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/1_Problem/0_Data/'
# outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/1_Problem/1_Pipeline/Output/'


# Is the matrix in the inputLoc canonical or simple? False = Canonical linearization.
simpleBool = True

# Should we apply the (2*Pi)**2 factor to the scattering rates as a correction? False = Don't apply factor.
scmBool = False
verboseError = False

# What's the name of the matrix?
# scmName = 'scattering_matrix.mmap'
# scmName = 'scattering_matrix_simple.mmap'
# scmName = 'scattering_matrix_hdd.mmap'
# scmName = 'scattering_matrix_5.87_simple.mmap'
# scmName = 'scattering_matrix_simple_2.mmap'
scmName = 'scatt_mat_pert.mmap'

# scmVal = (2*np.pi)**2*2
# scmVal = (2*np.pi)**2*1.545  # Problem 4
# scmVal = (2*np.pi)**2 * 0.01      # Problem 3
# scmVal = (2*np.pi)**2/1.5*1.2298  # Problem 2
# scmVal = (2*np.pi)**2/1.5    # Problem 1

relConvergence = 5e-3
absConvergence = 1e-60
# fieldVector = np.array([1e2, 1e3, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5,2e5,2.5e5,3e5,3.5e5,4e5,4.5e5])
# fieldVector = np.array([5e5,6e5,7e5,8e5,9e5,1e6,2e6,3e6])
# fieldVector = np.array([1e2, 1e3, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5,2e5,2.5e5,3e5,3.5e5,4e5,4.5e5,5e5,6e5,7e5,8e5,9e5,1e6,2e6,3e6])
# fieldVector = np.array([1e2, 1e3, 1e4, 1e5,2e5,3e5,4e5])
# fieldVector = np.array([1e2, 1e3])
# fieldVector = np.array([1e2, 1e3, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5,2e5,2.5e5,3e5,3.5e5,4e5,4.5e5])
fieldVector = np.array([1e-2,1e-1,1,1e1,1e2])
# fieldVector = np.array([6e4, 7e4, 8e4, 9e4, 1e5,2e5,2.5e5,3e5,3.5e5,4e5,4.5e5])
# fieldVector = np.array([2e5])
# fieldVector = np.array([1e2, 1e3, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5,2e5])

freqGHz = 10
getX = False


# Load the problem parameters (these change depending on which calc we're doing)
df = pandas.read_csv(inputLoc+'ProblemParameters.txt')
T = df.iloc[0]['LatticeT(K)']
mu = df.iloc[0]['FermiLevel(eV)']
b = df.iloc[0]['GaussianBroadening(eV)']
gD = df.iloc[0]['GridDensity(k)']
cutoff = 0.4

title_str = 'Grid {:.0f}'.format(gD) + '^3,' + ' {:3f}'.format(mu) + 'eV,' + ' {:.3f}'.format(T) + ' K,' + ' {:.3f}'.format(cutoff) + ' eV'
print(title_str)