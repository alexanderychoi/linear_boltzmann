import numpy as np
import constants as c
import os
import pandas as pd
import problem_parameters as pp
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import material_plotter

# PURPOSE: THIS MODULE CONTAINS A VARIETY OF FUNCTIONS THAT ARE USED THROUGHOUT OTHER MODULES TO PERFORM MORE COMPLEX
# CALCULATIONS. FOR EXAMPLE, THE CALCULATION OF THE DRIFT VELOCITY IS REQUIRED IN MULTIPLE MODULES: TO PLOT THE STEADY
# MOMENT AS A FUNCTION OF FIELD, TO CALCULATE THE EFFECTIVE DISTRIBUTION, ETC. UTILITIES.PY IS A CENTRALIZED MODULE THAT
# DEFINES THESE OFTEN-USED FUNCTIONS.

# ORDER: THIS MODULE IS NOT TYPICALLY RUN DIRECTLY, BUT CALLED BY OTHER MODULES.

# OUTPUT: THIS MODULE DOES NOT PRODUCE OUTPUTS.

def load_el_ph_data(inputLoc):
    if not (os.path.isfile(inputLoc + pp.prefix + '_full_electron_data.parquet') and
            os.path.isfile(inputLoc + pp.prefix + '_enq.parquet')):
        exit('Electron or phonon dataframes could not be found. ' +
             'You can create it using preprocessing.create_el_ph_dataframes.')
    else:
        el_df = pd.read_parquet(inputLoc + pp.prefix + '_full_electron_data.parquet')
        ph_df = pd.read_parquet(inputLoc + pp.prefix + '_enq.parquet')
    return el_df, ph_df


# The following set of functions performs calculations on the scattering or finite difference matrices
def calc_sparsity(matrix):
    """Count the number of non-zero elements in the matrix and return sparsity, normalized to 1.
    Parameters:
        matrix (nparray): Matrix on which sparsity is to be calculated.
    Returns:
        sparsity (dbl): Value of the sparsity, normalized to unity.
    """
    nkpts = len(matrix)
    sparsity = 1 - (np.count_nonzero(matrix) / nkpts**2)
    return sparsity


def matrix_check_colsum(matrix, nkpts):
    """Calculates the column sums of the given matrix.
    Parameters:
        matrix (array): NxN matrix on which colsums are to be calculated.
        nkpts (int): The size of the matrix (N)
    Returns:
        colsum (array): NX1 vector containing the column sums.
    """
    colsum = np.zeros(nkpts)
    for k in range(nkpts):
        colsum[k] = np.sum(matrix[:, k])
    return colsum


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_matrix_properties(matrix):
    cs = matrix_check_colsum(matrix, len(matrix))
    print('The average absolute value of column sum is {:E}'.format(np.average(np.abs(cs))))
    print('The largest column sum is {:E}'.format(cs.max()))
    # print('The matrix is symmetric: {0!s}'.format(check_symmetric(matrix)))
    # print('The average absolute value of element is {:E}'.format(np.average(np.abs(matrix))))
    print('The average value of on-diagonal element is {:E}'.format(np.average(np.diag(matrix))))
    print('Matrix sparsity is {:E}'.format(calc_sparsity(matrix)))


# The following set of functions calculate quantities based on the kpt DataFrame
def cartesian_projection(crystalVector):
    """Given a set of coordinates indicating the field direction, return the unit projections of the vector along the
    Cartesian crystallographic axes.
    Parameters:
        cystalVector (nparray, len = 3) : Numpy array containing the three coordinates specifying the direction of the
        applied electric field. Does not have to be normalized to 1.
    Returns:
        unitProjection (nparray, len = 3): Numpy array containing the three coordinates specifying the direction of the
        field normalized such that unitProjection is a unit vector (2-norm = 1).
    """
    unitProjection = np.zeros(len(crystalVector))
    mag = np.sqrt(crystalVector[0]**2 + crystalVector[1]**2 + crystalVector[2]**2)
    unitProjection[0] = crystalVector[0]/mag
    unitProjection[1] = crystalVector[1]/mag
    unitProjection[2] = crystalVector[2]/mag

    return unitProjection


def gaas_split_valleys(df, plot_Valleys = True):
    """Hardcoded for GaAs, obtains the indices for Gamma valley, the 8 L valleys, and the 6 X valleys and returns these
    as zero-indexed vectors.
    Parameters:
        df (DataFrame): Dataframe that has coordinates that have already been shifted back into the FBZ.
        plot_Valleys (Bool): Boolean signifying whether to generate plots of the three distinct valley types.
    Returns:
        valley_key_G (BooleanList): Containing the zero-indexed Gamma valley indices.
        valley_key_L (BooleanList): Containing the zero-indexed L valley indices.
        valley_key_X (BooleanList): Containing the zero-indexed X valley inds. If get_X = False, return an empty
    """
    kmag = np.sqrt(df['kx [1/A]'].values ** 2 + df['ky [1/A]'].values ** 2 + df['kz [1/A]'].values ** 2)
    kx = df['kx [1/A]'].values
    valley_key_L = np.array(kmag > 0.3) & np.array(abs(kx) > 0.25) & np.array(abs(kx) < 0.75)
    valley_key_G = np.array(kmag < 0.3)

    if pp.getX:
        valley_key_X = np.invert(valley_key_L) & np.invert(valley_key_G)
        x_df = df.loc[valley_key_X]
    else:
        valley_key_X = np.empty([1])

    g_df = df.loc[valley_key_G]
    l_df = df.loc[valley_key_L]
    print('There are {:d} kpoints in the Gamma valley'.format(np.count_nonzero(valley_key_G)))
    print('There are {:d} kpoints in the L valley'.format(np.count_nonzero(valley_key_L)))

    if plot_Valleys:
        material_plotter.bz_3dscatter(g_df, True, False)
        material_plotter.bz_3dscatter(l_df, True, False)
        if pp.getX:
            material_plotter.bz_3dscatter(x_df, True, False)
            print('There are {:d} kpoints in the X valley'.format(np.count_nonzero(valley_key_X)))

    return valley_key_G, valley_key_L, valley_key_X


def split_L_valleys(df, plot_Valleys = True):
    """Hardcoded for GaAs, obtains the indices for the 8 L valleys and returns these as zero-indexed vectors. Vectors
    are indexed with respect to the full valley df, not the L df.
    Parameters:
        df (DataFrame): Dataframe that has coordinates that have already been shifted back into the FBZ.
        plot_Valleys (Bool): Boolean signifying whether to generate plots of the eight distinct L valleys.
    Returns:
        valley_key_L1 (nparray): Numpy array containing the zero-indexed L valley indices. (+,+,+) k-coords
        valley_key_L2 (nparray): Numpy array containing the zero-indexed L valley indices. (+,+,-) k-coords
        valley_key_L3 (nparray): Numpy array containing the zero-indexed L valley indices. (+,-,+) k-coords
        valley_key_L4 (nparray): Numpy array containing the zero-indexed L valley indices. (+,-,-) k-coords
        valley_key_L5 (nparray): Numpy array containing the zero-indexed L valley indices. (-,+,+) k-coords
        valley_key_L6 (nparray): Numpy array containing the zero-indexed L valley indices. (-,+,-) k-coords
        valley_key_L7 (nparray): Numpy array containing the zero-indexed L valley indices. (-,-,+) k-coords
        valley_key_L8 (nparray): Numpy array containing the zero-indexed L valley indices. (-,-,-) k-coords
    """
    _, L_inds, _ = gaas_split_valleys(df, False)
    print('There are {:d} kpoints in the L valley'.format(np.count_nonzero(L_inds)))
    kx = df['kx [1/A]'].values
    ky = df['ky [1/A]'].values
    kz = df['kz [1/A]'].values

    valley_key_L1 = np.array(kx>0) & np.array(ky>0) & np.array(kz>0) & np.array(L_inds)
    valley_key_L2 = np.array(kx>0) & np.array(ky>0) & np.array(kz<0) & np.array(L_inds)
    valley_key_L3 = np.array(kx>0) & np.array(ky<0) & np.array(kz>0) & np.array(L_inds)
    valley_key_L4 = np.array(kx>0) & np.array(ky<0) & np.array(kz<0) & np.array(L_inds)
    valley_key_L5 = np.array(kx<0) & np.array(ky>0) & np.array(kz>0) & np.array(L_inds)
    valley_key_L6 = np.array(kx<0) & np.array(ky>0) & np.array(kz<0) & np.array(L_inds)
    valley_key_L7 = np.array(kx<0) & np.array(ky<0) & np.array(kz>0) & np.array(L_inds)
    valley_key_L8 = np.array(kx<0) & np.array(ky<0) & np.array(kz<0) & np.array(L_inds)

    if plot_Valleys:
        x = df['kx [1/A]'].values / (2 * np.pi / c.a)
        y = df['ky [1/A]'].values / (2 * np.pi / c.a)
        z = df['kz [1/A]'].values / (2 * np.pi / c.a)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[valley_key_L1], y[valley_key_L1], z[valley_key_L1],label='L1',s=0.02)
        ax.scatter(x[valley_key_L2], y[valley_key_L2], z[valley_key_L2],label='L2',s=0.02)
        ax.scatter(x[valley_key_L3], y[valley_key_L3], z[valley_key_L3],label='L3',s=0.02)
        ax.scatter(x[valley_key_L4], y[valley_key_L4], z[valley_key_L4],label='L4',s=0.02)
        ax.scatter(x[valley_key_L5], y[valley_key_L5], z[valley_key_L5],label='L5',s=0.02)
        ax.scatter(x[valley_key_L6], y[valley_key_L6], z[valley_key_L6],label='L6',s=0.02)
        ax.scatter(x[valley_key_L7], y[valley_key_L7], z[valley_key_L7],label='L7',s=0.02)
        ax.scatter(x[valley_key_L8], y[valley_key_L8], z[valley_key_L8],label='L8',s=0.02)
        ax.set_xlabel(r'$kx/2\pi a$')
        ax.set_ylabel(r'$ky/2\pi a$')
        ax.set_zlabel(r'$kz/2\pi a$')
        fontP = FontProperties()
        fontP.set_size('small')
        plt.legend(prop=fontP,loc='center left',ncol=1)
        plt.title('L valleys')
    return valley_key_L1, valley_key_L2, valley_key_L3, valley_key_L4, valley_key_L5, valley_key_L6, valley_key_L7, valley_key_L8


def fermi_distribution(df, testboltzmann=False):
    """Given an electron DataFrame, a Fermi Level, and a temperature, calculate the Fermi-Dirac distribution and add ...
    as a column to the DataFrame. Flagged option to add another column with the Boltzmann distribution.

    Parameters:
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        testboltzmann (bool): Boolean for whether to write the Maxwell-Boltzmann distribution as a column.

    Returns:
        df (dataframe): Edited electron DataFrame containing the new columns with equilibrium distribution functions.
    """
    df['k_FD'] = (np.exp((df['energy [eV]'].values * c.e - pp.mu * c.e) / (c.kb_joule * pp.T)) + 1) ** (-1)
    if testboltzmann:
        boltzdist = (np.exp((df['energy [eV]'].values * c.e - pp.mu * c.e) / (c.kb_joule * pp.T))) ** (-1)
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
    Nuc = pp.kgrid ** 3
    normalization = 1 / Nuc / c.Vuc
    n = 2 * np.sum(f0) * normalization
    return n


# The following set of functions calculate quantities based on steady state chi solutions
def mean_velocity(chi, df):
    """Function that calculates the mean velocity given a Chi solution through moment of group velocity in BZ. Note that
    this makes use of the equilibrium carrier density. Hardcoded for field applied in the (1 0 0)
    Parameters:
        chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in chi form.
        df (dataframe): Electron DataFrame indexed by kpt containing the group velocity associated with each state in eV.

    Returns:
        v_ave (double): The value of the average velocity for a given steady-state solution of the BTE in m/s.
    """
    f0 = df['k_FD'].values
    f = chi + f0
    v_ave = np.sum(f * df['vx [m/s]']) / np.sum(f)
    return v_ave


def mean_xvelocity_mag(chi, df):
    """Function that calculates the mean velocity given a Chi solution through moment of group velocity in BZ. Note that
    this makes use of the equilibrium carrier density.
    Parameters:
        chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in chi form.
        df (dataframe): Electron DataFrame indexed by kpt containing the group velocity associated with each state in eV.

    Returns:
        v_ave (double): The value of the average velocity for a given steady-state solution of the BTE in m/s.
    """
    f0 = df['k_FD'].values
    f = chi + f0
    v_ave = np.sum(f * np.abs(df['vx [m/s]'])) / np.sum(f)
    return v_ave


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
    Nuc = pp.kgrid ** 3
    normalization = 1 / Nuc / c.Vuc
    n = 2 * np.sum(f) * normalization
    # print('Non-eq carrier density (including chi) is {:.10E}'.format(n * 1E-6) + ' per cm^{-3}')
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
    f = chi + f0
    meanE = np.sum(f * df['energy [eV]']) / np.sum(f)
    eq_en = np.sum(f0 * df['energy [eV]']) / np.sum(f0)
    # print('Carrier density (including chi) is {:.10E}'.format(n * 1E-6) + ' per cm^{-3}')
    print('Mean carrier energy is {:.10E} [eV]'.format(meanE))
    # print('Equilibrium carrier energy is {:.10E} [eV]'.format(eq_en))

    return meanE


def mean_kx(chi, df):
    """Function that calculates the mean kx by a sum of the electron energy over the BZ. Note that this makes use of
    the equilbrium carrier density.
    Parameters:
        chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in chi form.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        meanE (double): The value of the mean carrier energy in eV.
    """
    f0 = df['k_FD'].values
    f = chi + f0
    meankx = np.sum(f * df['kx [1/A]']) / np.sum(f)

    return meankx


def calc_mobility(F, df):
    """Calculate mobility as per Wu Li PRB 92, 2015. Solution must be fed in as F with no field provided.
    Parameters:
        F (nparray): Numpy array containing a solution of the steady Boltzmann equation in F form or as psi
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        mobility (double): The value of the mobility carrier energy in m^2/V-s.
    """
    Nuc = pp.kgrid ** 3
    print('Field not specified. Mobility calculated using linear in E formula.')
    prefactor = 2 * c.e ** 2 / (c.Vuc * c.kb_joule * pp.T * Nuc)
    conductivity = prefactor * np.sum(df['k_FD'] * (1 - df['k_FD']) * df['vx [m/s]'] * F)
    n = calculate_density(df)
    mobility = conductivity / c.e / n
    print('Carrier density is {:.8E}'.format(n * 1E-6) + ' per cm^{-3}')
    print('Mobility is {:.10E} (cm^2 / V / s)'.format(mobility * 1E4))
    return mobility


def calc_linear_mobility(chi, df, field):
    """Calculate linear mobility as per general definition of conductivity. Solution must be fed in as chi. I just used
    Wu Li's formula and modified to acept chi as an input.
    Parameters:
        chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in F form or as psi
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        field (dbl): Value of the electric field in V/m.
    Returns:
        mobility (double): The value of the mobility carrier energy in m^2/V-s.
    """
    Nuc = pp.kgrid ** 3
    print('Field specified. Mobility calculated using general definition of conductivity')
    n = calculate_density(df)
    prefactor = 2 * c.e / c.Vuc / Nuc / field
    conductivity = prefactor * np.sum(df['vx [m/s]'] * chi)
    mobility = conductivity / c.e / n
    # print('Carrier density is {:.8E}'.format(n * 1E-6) + ' per cm^{-3}')
    print('Mobility is {:.10E} (cm^2 / V / s)'.format(mobility * 1E4))
    print('Conductivity is {:.10E} (S / m)'.format(conductivity))
    return mobility


def calc_popsplit(chi, df):
    """Function that calculates the carrrier populations in the Gamma, L, and X valleys given a Chi solution.
    Parameters:
        chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in chi form.
        df (dataframe): Electron DataFrame indexed by kpt
        get_X (Bool): Boolean signifying whether the calculation should also re`turn X valley indices. Not every grid
        contains the X valleys. True -> Get X valley inds
    Returns:
        ng (double): The value of the gamma carrier population in m^-3.
        nl (double): The value of the L carrier population in m^-3.
        nx (dobule): The value of the X carrier population in m^-3.
    """
    f0 = df['k_FD'].values
    f = chi + f0
    g_inds,l_inds,x_inds = gaas_split_valleys(df,False)

    Nuc = pp.kgrid ** 3
    n_g = 2 / Nuc / c.Vuc * np.sum(f[g_inds])
    n_l = 2 / Nuc / c.Vuc * np.sum(f[l_inds])
    if pp.getX:
        n_x = 2 / Nuc / c.Vuc * np.sum(f[x_inds])
    else:
        n_x = 0
    n = 2 / Nuc / c.Vuc * np.sum(f)
    return n_g, n_l, n_x, n


def calc_popinds(chi, df, inds):
    """Function that calculates the carrrier populations in the given indices of a chi solution.
    Parameters:
        chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in chi form.
        df (dataframe): Electron DataFrame indexed by kpt
        inds (nparray): Indexing array that you want to calculate the population in.
    Returns:
        ng (double): The value carrier population contained in inds in m^-3.

    """
    f0 = df['k_FD'].values
    f = chi + f0
    Nuc = pp.kgrid ** 3
    n = 2 / Nuc / c.Vuc * np.sum(f[inds])

    return n


def f2chi(f, df, field):
    """Convert F_k from low field approximation iterative scheme into chi which is easy to plot. 
    Since the solution we obtain from cg and from iterative scheme is F_k where chi_k = eE/kT * f0(1-f0) * F_k
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
    # prefactor = field * c.e / c.kb_joule / pp.T * f0
    chi = np.squeeze(f) * np.squeeze(prefactor)
    return chi


if __name__ == '__main__':
    out_loc = pp.outputLoc
    in_loc = pp.inputLoc
    eldf, phdf = load_el_ph_data(in_loc)
    fermi_distribution(eldf)
    conc = calculate_density(eldf)
    print('Carrier concentration is {:.2E} cm^-3'.format(conc * 1E-6))
