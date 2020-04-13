import pandas

# Change this directory to point at the problem-specific input folder (containing 'ProblemParameters.txt')
inputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/#1_Problem/0_Data/'
outputLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/#1_Problem/1_Pipeline/Output/'

# Load the problem parameters (these change depending on which calc we're doing)
df = pandas.read_csv(inputLoc+'ProblemParameters.txt')
T = df.iloc[0]['LatticeT(K)']
mu = df.iloc[0]['FermiLevel(eV)']
b = df.iloc[0]['GaussianBroadening(eV)']
gD = df.iloc[0]['GridDensity(k)']