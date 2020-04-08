import numpy as np
import constants as c
import utilities


# The following set of functions calculate the low-frequency PSD based on solutions to effective Boltzmann equation
def lowfreq_diffusion(g, df):
    """Calculate the low-frequency non-eq diffusion coefficent as per Wu Li PRB 92, 2015. Effective distribution g must
    be passed in as chi.
    Parameters:
        g (nparray): Effective distribution function passed as chi.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        D (double): The value of the non-eq diffusion coefficient in m^2/s.
    """
    n = utilities.calculate_density(df)
    Nuc = len(df)
    D = 1 / Nuc / c.Vuc * np.sum(g*df['vx [m/s]']) / n
    return D


def noiseT(inLoc,D,mobility,df):
    """Calculates the noise temperature using the Price relationship.
    Parameters:
        inLoc (str): String containing the location of the directory containing the scattering matrix, assumed simple
        linearization by default.
        D (double): The value of the non-eq diffusion coefficient in m^2/s.
        mobility (df): The value of the mobility in m^2/V-s.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        None. Just writes the chi solutions to file. chi_#_EField. #1 corresponds to low-field RTA, #2 corresponds to
        low-field iterative, #3 corresponds to full finite-difference iterative.
    """
    nkpts = len(df)
    scm = np.memmap(inLoc + 'scattering_matrix_5.87_simple.mmap', dtype='float64', mode='r', shape=(nkpts, nkpts))
    scmfac = (2*np.pi)**2
    invdiag = (np.diag(scm) * scmfac) ** (-1)
    f0 = df['k_FD'].values
    tau = -np.sum(f0 * invdiag) / np.sum(f0)
    n = utilities.calculate_density(df)
    D_eq = c.kb_joule*c.T*tau/(0.063*9.11e-31)

    con_eq = c.e*tau/(0.063*9.11e-31) * c.e * n *2
    con_neq = mobility * c.e * n
    Tn = D/D_eq*c.T*con_eq/con_neq
    return Tn


if __name__ == '__main__':
    out_Loc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/#1_Problem/1_Pipeline/Output/'
    in_Loc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/#1_Problem/0_Data/'
    utilities.read_problem_params(in_Loc)
