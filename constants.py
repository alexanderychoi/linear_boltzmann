import numpy as np
import problem_parameters as pp

# Physical parameters (these will typically not change)
e = 1.602 * 10 ** (-19)  # Fundamental electronic charge [C]
kb_joule = 1.38064852 * 10 ** (-23)  # Boltzmann constant in SI [m^2 kg s^-2 K^-1]
kb_ev = 8.617333 * (10 ** -5)  # Boltzmann constant in eV/K
hbar_joule = 1.054571817 * 10 ** (-34)  # Reduced Planck's constant [J/s]
hbar_ev = 6.582119569 * 10 ** (-16)  # Reduced Planck's constant [eV/s]
ryd2ev = 13.605693122994  # CODATA value. Different from perturbo at 8th sigfig

# Lattice vectors
if pp.prefix == 'gaas':
    alat = 5.556360705  # lattice parameter in Angstrom consistent with what perturbo uses
    a1 = alat * np.array([-0.5, 0.0, 0.5])
    a2 = alat * np.array([+0.0, 0.5, 0.5])
    a3 = alat * np.array([-0.5, 0.5, 0.0])

    Vuc = np.dot(np.cross(a1, a2), a3)  # unit cell volume in angstrom^3
    b1 = 2 * np.pi * np.cross(a2, a3) / Vuc
    b2 = 2 * np.pi * np.cross(a3, a1) / Vuc
    b3 = 2 * np.pi * np.cross(a1, a2) / Vuc
    Vuc = Vuc * 1E-30  # unit cell volume in m^3

elif pp.prefix == 'si':
    alat = 5.431474883  # lattice parameter in Angstrom consistent with what perturbo uses
    a1 = alat * np.array([-0.5, 0.0, 0.5])
    a2 = alat * np.array([+0.0, 0.5, 0.5])
    a3 = alat * np.array([-0.5, 0.5, 0.0])

    Vuc = np.dot(np.cross(a1, a2), a3)  # unit cell volume in angstrom^3
    b1 = 2 * np.pi * np.cross(a2, a3) / Vuc
    b2 = 2 * np.pi * np.cross(a3, a1) / Vuc
    b3 = 2 * np.pi * np.cross(a1, a2) / Vuc
    Vuc = Vuc * 1E-30  # unit cell volume in m^3

else:
    exit('Could not identify material, so could not set lattice parameters. Please set prefix in problem_parameters.txt')
