import numpy as np
import problemparameters as pp
import constants as c


def read_problem_params(inLoc):
    """Reads the problem parameters that are loaded from the constants.py module
    Parameters:
        inLoc (str): String containing the location of the directory containing the input text file.
    Returns:
        None. Just prints the values of the problem parameters.
    """
    print('Physical constants loaded from' + inLoc)
    print('Temperature is {:.1e} K'.format(pp.T))
    print('Fermi Level is {:.1e} eV'.format(pp.mu))
    print('Gaussian broadening is {:.1e} eV'.format(pp.b))
    print('Grid density is {:.1e} cubed'.format(pp.gD))


# The following set of functions calculate quantities based on the kpt DataFrame
def fermi_distribution(df, testboltzmann=False):
    """Given an electron DataFrame, a Fermi Level, and a temperature, calculate the Fermi-Dirac distribution and add ...
    as a column to the DataFrame. Flagged option to add another column with the Boltzmann distribution.

    Parameters:
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        testboltzmann (bool): Boolean for whether to write the Maxwell-Boltzmann distribution as a column.

    Returns:
        df (dataframe): Edited electron DataFrame containing the new columns with equilibrium distribution functions.
    """

    df['k_FD'] = (np.exp((df['energy'].values * c.e - pp.mu * c.e) / (c.kb_joule * pp.T)) + 1) ** (-1)
    if testboltzmann:
        boltzdist = (np.exp((df['energy'].values * c.e - pp.mu * c.e) / (c.kb_joule * pp.T))) ** (-1)
        partfunc = np.sum(boltzdist)
        df['k_MB'] = boltzdist/partfunc
    return df


def calculate_density(df):
    """Function that calculates the carrier density by a sum of the equilibrium distribution function over the BZ.
    Parameters:
        df (dataframe): Electron DataFrame indexed by kpt containing the eq. dist associated with each state in eV.

    Returns:
        n (double): The value of the carrier density specified by the equilibrium FD distribution.
    """
    f0 = df['k_FD'].values
    Nuc = len(df)
    n = 2 / Nuc / c.Vuc * np.sum(f0)
    return n


# The following set of functions calculate quantities based on steady state chi solutions
def drift_velocity(chi, df):
    """Function that calculates the drift velocity given a Chi solution through moment of group velocity in BZ. Note that
    this makes use of the equilibrium carrier density.
    Parameters:
        chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in chi form.
        df (dataframe): Electron DataFrame indexed by kpt containing the group velocity associated with each state in eV.

    Returns:
        vd (double): The value of the drift velocity for a given steady-state solution of the BTE in m/s.
    """
    f0 = df['k_FD'].values
    f = chi + f0
    n = calculate_density(df)
    vd = np.sum(f * df['vx [m/s]']) / np.sum(f)
    print('Carrier density (including chi) is {:.10E}'.format(n * 1E-6) + ' per cm^{-3}')
    print('Drift velocity is {:.10E} [m/s]'.format(vd))
    return vd


def calculate_noneq_density(chi, df):
    """Function that calculates the carrier density by a sum of the steady non-eq distribution function over the BZ.
    Parameters:
        chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in chi form.
        df (dataframe): Electron DataFrame indexed by kpt containing the eq. dist associated with each state in eV.

    Returns:
        n (double): The value of the carrier density specified by the noneq distribution.
    """
    f0 = df['k_FD'].values
    f = chi + f0
    Nuc = len(df)
    n = 2 / Nuc / c.Vuc * np.sum(f)
    print('Non-eq carrier density (including chi) is {:.10E}'.format(n * 1E-6) + ' per cm^{-3}')
    return n


def mean_energy(chi, df):
    """Function that calculates the mean energy by a sum of the electron energy over the BZ. Note that this makes use of
    the equilbrium carrier density.
    Parameters:
        chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in chi form.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        meanE (double): The value of the mean carrier energy in eV.
    """
    f0 = df['k_FD'].values
    f = chi + df['k_FD'].values
    n = calculate_density(df)
    meanE = np.sum(f * df['energy']) / np.sum(f0)
    print('Carrier density (including chi) is {:.10E}'.format(n * 1E-6) + ' per cm^{-3}')
    print('Mean carrier energy is {:.10E} [eV]'.format(meanE))
    return meanE


def calc_mobility(F, df):
    """Calculate mobility as per Wu Li PRB 92, 2015. Solution must be fed in as F with no field provided.
    Parameters:
        F (nparray): Numpy array containing a solution of the steady Boltzmann equation in F form or as psi
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        mobility (double): The value of the mobility carrier energy in m^2/V-s.
    """
    Nuc = len(df)
    print('Field not specified. Mobility calculated using linear in E formula.')
    prefactor = 2 * c.e ** 2 / (c.Vuc * c.kb_joule * pp.T * Nuc)
    conductivity = prefactor * np.sum(df['k_FD'] * (1 - df['k_FD']) * df['vx [m/s]'] * F)
    n = calculate_density(df)
    mobility = conductivity / c.e / n
    print('Carrier density is {:.8E}'.format(n * 1E-6) + ' per cm^{-3}')
    print('Mobility is {:.10E} (cm^2 / V / s)'.format(mobility * 1E4))
    return mobility


def calc_diff_mobility(chi, df,field):
    """Calculate differential mobility as per general definition of conductivity. Solution must be fed in as chi.
    I'm not sure that this formula is the right formula for differential mobility. I just used Wu Li's formula and modified
    to acept chi as an input, which I don't think is the same thing.
    Parameters:
        chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in F form or as psi
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        field (dbl): Value of the electric field in V/m.
    Returns:
        mobility (double): The value of the mobility carrier energy in m^2/V-s.
    """
    Nuc = len(df)
    print('Field specified. Mobility calculated using general definition of conductivity')
    n = calculate_density(df)
    prefactor = 2 *c.e / c.Vuc / Nuc /field
    conductivity = prefactor * np.sum(df['vx [m/s]'] * chi)
    mobility = conductivity / c.e / n
    print('Carrier density is {:.8E}'.format(n * 1E-6) + ' per cm^{-3}')
    print('Mobility is {:.10E} (cm^2 / V / s)'.format(mobility * 1E4))
    return mobility


def calc_L_Gamma_pop(chi, df):
    """Function that calculates the carrrier populations in the Gamma and L valleys given a Chi solution.
    Parameters:
        chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in chi form.
        df (dataframe): Electron DataFrame indexed by kpt containing the group velocity associated with each state in eV.

    Returns:
        ng (double): The value of the gamma carrier population in m^-3.
        nl (double): The value of the upper carrier population in m^-3.
    """
    f0 = df['k_FD'].values
    f = chi + f0
    df['kpt_mag'] = np.sqrt(df['kx [1/A]'].values**2 + df['ky [1/A]'].values**2 +
                                 df['kz [1/A]'].values**2)
    df['ingamma'] = df['kpt_mag'] < 0.3  # Boolean. In gamma if kpoint magnitude less than some amount
    g_inds = df.loc[df['ingamma'] == 1].index - 1
    l_inds = df.loc[df['ingamma'] == 0].index - 1
    Nuc = len(df)
    n_g = 2 / Nuc / c.Vuc * np.sum(f[g_inds])
    n_l = 2 / Nuc / c.Vuc * np.sum(f[l_inds])
    return n_g, n_l


def f2chi(f, df, field):
    """Convert F_k from low field approximation iterative scheme into chi which is easy to plot. Since the solution we obtain from cg and from iterative scheme is F_k where chi_k = eE/kT * f0(1-f0) * F_k
    then we need to bring these factors back in to get the right units
    Parameters:
        f (nparray): Numpy array containing a solution of the steady Boltzmann equation in f form.
        df (dataframe): Electron DataFrame indexed by kpt containing the group velocity associated with each state in eV.
        field (dbl): Value of the electric field in V/m.
    Returns:
        chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in chi form.
    """
    f0 = np.squeeze(df['k_FD'].values)
    prefactor = field * c.e / c.kb_joule / pp.T * f0 * (1 - f0)
    chi = np.squeeze(f) * np.squeeze(prefactor)
    return chi


if __name__ == '__main__':
    out_Loc = pp.outputLoc
    in_Loc = pp.inputLoc
    read_problem_params(in_Loc)