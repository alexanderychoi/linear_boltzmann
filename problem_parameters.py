import pandas
import os
import numpy
import re

f = open('ProblemParameters.txt')

alltext = f.read()

inputLoc = re.findall(r"\s*inputLoc\s*=\s*'(.+)'\n", alltext)[0]
outputLoc = re.findall(r"\s*outputLoc\s*=\s*'(.+)'\n", alltext)[0]
scratchLoc = re.findall(r"\s*scratchLoc\s*=\s*'(.+)'\n", alltext)
T = float(re.findall(r"\s*Temperature\s*=\s*(.+)\n", alltext)[0])
mu = float(re.findall(r"\s*FermiLevel\s*=\s*(.+)\n", alltext)[0])
b = float(re.findall(r"\s*GaussianBroadening\s*=\s*(.+)\n", alltext)[0])
gD = float(re.findall(r"\s*GridDensity\s*=\s*(\d+)\n", alltext)[0])

print('\nData for this run loaded from \'' + inputLoc + '\'')
print('Lattice temperature is {:.1f} K'.format(T))
print('Fermi Level is {:.5f} eV'.format(mu))
print('Gaussian broadening is {:.1e} eV'.format(b))
print('Grid density is {:.1f} cubed'.format(gD))
if scratchLoc:
	scratchLoc = scratchLoc[0]
	print('Scratch location is \'' + scratchLoc + '\'')

# Is the matrix in the inputLoc canonical or simple? False = Canonical linearization.
simpleBool = True
# Should we apply the (2*Pi)**2 factor to the scattering rates as a correction? False = Don't apply factor.
scmBool = False
# What's the name of the matrix?
scmName = 'scattering_matrix.mmap'

print('\nIs matrix is presumed to be in simple linearization? {:s}'.format(str(simpleBool)))
print('Will an arbitrary correction factor be applied to the matrix? {:s}'.format(str(scmBool)))
print('The name of the scattering matrix to be loaded is {:s}'.format(scmName))
