import numpy as np
import pandas

# Change this directory to point at the problem-specific input folder (containing 'ProblemParameters.txt')
inLoc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/#1_Problem/0_Data/'

# Load the problem parameters (these change depending on which calc we're doing)
df = pandas.read_csv(inLoc+'ProblemParameters.txt')
T = df.iloc[0]['LatticeT(K)']
mu = df.iloc[0]['FermiLevel(eV)']
b = df.iloc[0]['GaussianBroadening(eV)']
gD = df.iloc[0]['GridDensity(k)']


# Physical parameters (these will typically not change)
a = 5.5563606  # Lattice constant for GaAs [Angstrom]
e = 1.602 * 10 ** (-19)  # Fundamental electronic charge [C]
kb_joule = 1.38064852 * 10 ** (-23)  # Boltzmann constant in SI [m^2 kg s^-2 K^-1]
kb_ev = 8.617333 * (10 ** -5)  # Boltzmann constant in eV/K
hbar_joule = 1.054571817 * 10 ** (-34)  # Reduced Planck's constant [J/s]
hbar_ev = 6.582119569 * 10 ** (-16)  # Reduced Planck's constant [eV/s]

# Lattice vectors
a1 = np.array([-2.7781803, 0.0000000, 2.7781803])
a2 = np.array([+0.0000000, 2.7781803, 2.7781803])
a3 = np.array([-2.7781803, 2.7781803, 0.0000000])
b1 = np.array([-1.1308095, -1.1308095, +1.1308095])
b2 = np.array([+1.1308095, +1.1308095, +1.1308095])
b3 = np.array([-1.1308095, +1.1308095, -1.1308095])
Vuc = np.dot(np.cross(b1, b2), b3) * 1E-30  # unit cell volume in m^3