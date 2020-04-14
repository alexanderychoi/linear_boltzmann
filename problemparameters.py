import pandas

# Change this directory to point at the problem-specific input folder (containing 'ProblemParameters.txt')
inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/#2_Problem/0_Data/'
outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/#2_Problem/1_Pipeline/Output/'

# Is the matrix in the inputLoc canonical or simple? False = Canonical linearization.
simpleBool = False

# Should we apply the (2*Pi)**2 factor to the scattering rates as a correction? False = Don't apply factor.
scmBool = True

# What's the name of the matrix?
scmName = 'scattering_matrix_canonical.mmap'

# Load the problem parameters (these change depending on which calc we're doing)
df = pandas.read_csv(inputLoc+'ProblemParameters.txt')
T = df.iloc[0]['LatticeT(K)']
mu = df.iloc[0]['FermiLevel(eV)']
b = df.iloc[0]['GaussianBroadening(eV)']
gD = df.iloc[0]['GridDensity(k)']