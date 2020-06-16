import numpy as np
import constants as c
import os
import pandas as pd
import occupation_plotter
import problem_parameters as pp
import matplotlib.pyplot as plt


def split_valleys(df, plot_Valleys = True):
    """Hardcoded for GaAs, obtains the indices for Gamma valley, the 8 L valleys, and the 6 X valleys and returns these
    as zero-indexed vectors.
    Parameters:
        df (DataFrame): Dataframe that has coordinates that have already been shifted back into the FBZ.
        plot_Valleys (Bool): Boolean signifying whether to generate plots of the three distinct valley types.
    Returns:
        valley_key_G (nparray): Numpy array containing the zero-indexed Gamma valley indices.
        valley_key_L (nparray): Numpy array containing the zero-indexed L valley indices.
        valley_key_X (nparray): Numpy array containing the zero-indexed X valley inds. If get_X = False, return an empty
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
        occupation_plotter.bz_3dscatter(g_df, True, False)
        occupation_plotter.bz_3dscatter(l_df, True, False)
        if pp.getX:
            occupation_plotter.bz_3dscatter(x_df, True, False)
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
    _, L_inds, _ = split_valleys(df, False)
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
        ax.scatter(x[valley_key_L1], y[valley_key_L1], z[valley_key_L1])
        ax.scatter(x[valley_key_L2], y[valley_key_L2], z[valley_key_L2])
        ax.scatter(x[valley_key_L3], y[valley_key_L3], z[valley_key_L3])
        ax.scatter(x[valley_key_L4], y[valley_key_L4], z[valley_key_L4])
        ax.scatter(x[valley_key_L5], y[valley_key_L5], z[valley_key_L5])
        ax.scatter(x[valley_key_L6], y[valley_key_L6], z[valley_key_L6])
        ax.scatter(x[valley_key_L7], y[valley_key_L7], z[valley_key_L7])
        ax.scatter(x[valley_key_L8], y[valley_key_L8], z[valley_key_L8])
        ax.set_xlabel(r'$kx/2\pi a$')
        ax.set_ylabel(r'$ky/2\pi a$')
        ax.set_zlabel(r'$kz/2\pi a$')

    return valley_key_L1, valley_key_L2, valley_key_L3, valley_key_L4, valley_key_L5, valley_key_L6, valley_key_L7, valley_key_L8


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
    print('The average absolute value of SCM column sum is {:E}'.format(np.average(np.abs(cs))))
    print('The largest SCM column sum is {:E}'.format(cs.max()))
    # print('The matrix is symmetric: {0!s}'.format(check_symmetric(matrix)))
    # print('The average absolute value of element is {:E}'.format(np.average(np.abs(matrix))))
    print('The average value of on-diagonal element is {:E}'.format(np.average(np.diag(matrix))))


def load_el_ph_data(inputLoc):
    if not (os.path.isfile(inputLoc + pp.prefix + '_full_electron_data.parquet') and
            os.path.isfile(inputLoc + pp.prefix + '_enq.parquet')):
        exit('Electron or phonon dataframes could not be found. ' +
             'You can create it using preprocessing.create_el_ph_dataframes.')
    else:
        el_df = pd.read_parquet(inputLoc + pp.prefix + '_full_electron_data.parquet')
        ph_df = pd.read_parquet(inputLoc + pp.prefix + '_enq.parquet')
    return el_df, ph_df


def translate_into_fbz(df):
    """Manually translate coordinates back into first Brillouin zone

    The way we do this is by finding all the planes that form the FBZ boundary and the vectors that are associated
    with these planes. Since the FBZ is centered on Gamma, the position vectors of the high symmetry points are also
    vectors normal to the plane. Once we have these vectors, we find the distance between a given point (u) and
    a plane (n) using the dot product of the difference vector (u-n). And if the distance is positive, then translate
    back into the FBZ.

    Args:
        df (dataframe): Electron dataframe containing the kpoints. Will have their data translated back into FBZ 
    """
    # First, find all the vectors defining the boundary
    coords = df[['kx [1/A]', 'ky [1/A]', 'kz [1/A]']]
    b1, b2, b3 = c.b1, c.b2, c.b3
    b1pos = 0.5 * b1[:, np.newaxis]
    b2pos = 0.5 * b2[:, np.newaxis]
    b3pos = 0.5 * b3[:, np.newaxis]
    lpos = 0.5 * (b1 + b2 + b3)[:, np.newaxis]
    b1neg = -1 * b1pos
    b2neg = -1 * b2pos
    b3neg = -1 * b3pos
    lneg = -1 * lpos
    xpos = -0.5 * (b1 + b3)[:, np.newaxis]
    ypos = 0.5 * (b2 + b3)[:, np.newaxis]
    zpos = 0.5 * (b1 + b2)[:, np.newaxis]
    xneg = -1 * xpos
    yneg = -1 * ypos
    zneg = -1 * zpos

    # Place them into octants to avoid problems when finding points
    # (naming is based on positive or negative for coordinate so octpmm means x+ y- z-. p=plus, m=minus)
    vecs_ppp = np.concatenate((b2pos, xpos, ypos, zpos), axis=1)[:, :, np.newaxis]
    vecs_ppm = np.concatenate((b1neg, xpos, ypos, zneg), axis=1)[:, :, np.newaxis]
    vecs_pmm = np.concatenate((lneg, xpos, yneg, zneg), axis=1)[:, :, np.newaxis]
    vecs_mmm = np.concatenate((b2neg, xneg, yneg, zneg), axis=1)[:, :, np.newaxis]
    vecs_mmp = np.concatenate((b1pos, xneg, yneg, zpos), axis=1)[:, :, np.newaxis]
    vecs_mpp = np.concatenate((lpos, xneg, ypos, zpos), axis=1)[:, :, np.newaxis]
    vecs_mpm = np.concatenate((b3pos, xneg, ypos, zneg), axis=1)[:, :, np.newaxis]
    vecs_pmp = np.concatenate((b3neg, xpos, yneg, zpos), axis=1)[:, :, np.newaxis]
    # Construct matrix which is 3 x 4 x 8 where we have 3 Cartesian coordinates, 4 vectors per octant, and 8 octants
    allvecs = np.concatenate((vecs_ppp, vecs_ppm, vecs_pmm, vecs_mmm, vecs_mmp, vecs_mpp, vecs_mpm, vecs_pmp), axis=2)

    # Since the number of points in each octant is not equal, can't create array of similar shape. Instead the 'octant'
    # array below is used as a boolean map where 1 (true) indicates positive, and 0 (false) indicates negative
    octants = np.array([[1, 1, 1],
                        [1, 1, 0],
                        [1, 0, 0],
                        [0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 1],
                        [0, 1, 0],
                        [1, 0, 1]])

    fbzcoords = coords.copy(deep=True).values
    exitvector = np.zeros((8, 1))
    iteration = 0
    while not np.all(exitvector):  # don't exit until all octants have points inside
        exitvector = np.zeros((8, 1))
        for i in range(8):
            oct_vecs = allvecs[:, :, i]
            whichoct = octants[i, :]
            if whichoct[0]:
                xbool = fbzcoords[:, 0] > 0
            else:
                xbool = fbzcoords[:, 0] < 0
            if whichoct[1]:
                ybool = fbzcoords[:, 1] > 0
            else:
                ybool = fbzcoords[:, 1] < 0
            if whichoct[2]:
                zbool = fbzcoords[:, 2] > 0
            else:
                zbool = fbzcoords[:, 2] < 0
            octindex = np.logical_and(np.logical_and(xbool, ybool), zbool)
            octcoords = fbzcoords[octindex, :]
            allplanes = 0
            for j in range(oct_vecs.shape[1]):
                diffvec = octcoords[:, :] - np.tile(oct_vecs[:, j], (octcoords.shape[0], 1))
                dist2plane = np.dot(diffvec, oct_vecs[:, j]) / np.linalg.norm(oct_vecs[:, j])
                outside = dist2plane[:] > 0
                if np.any(outside):
                    octcoords[outside, :] = octcoords[outside, :] - \
                                            (2 * np.tile(oct_vecs[:, j], (np.count_nonzero(outside), 1)))
                    # Times 2 because the vectors that define FBZ are half of the full recip latt vectors
                    # print('number outside this plane is %d' % np.count_nonzero(outside))
                else:
                    allplanes += 1
            if allplanes == 4:
                exitvector[i] = 1
            fbzcoords[octindex, :] = octcoords
        iteration += 1
        print('Finished %d iterations of bringing points into FBZ' % iteration)

    uniqkx = np.sort(np.unique(fbzcoords[:, 0]))
    deltakx = np.diff(uniqkx)
    smalldkx = np.concatenate((deltakx < (np.median(deltakx) * 1E-2), [False]))
    if np.any(smalldkx):
        for kxi in np.nditer(np.nonzero(smalldkx)):
            kx = uniqkx[kxi]
            fbzcoords[fbzcoords[:, 0] == kx, 0] = uniqkx[kxi+1]
        print('Shifted points that were slightly misaligned in kx.\n')
    df[['kx [1/A]', 'ky [1/A]', 'kz [1/A]']] = fbzcoords
    print('Done bringing points into FBZ!')
    return df


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
    vd = np.sum(f * df['vx [m/s]']) / np.sum(f)
    return vd


def mean_velocity(chi, df):
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
    n = calculate_density(df)
    meanE = np.sum(f * df['energy [eV]']) / np.sum(f)
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
    Nuc = pp.kgrid ** 3
    print('Field not specified. Mobility calculated using linear in E formula.')
    prefactor = 2 * c.e ** 2 / (c.Vuc * c.kb_joule * pp.T * Nuc)
    conductivity = prefactor * np.sum(df['k_FD'] * (1 - df['k_FD']) * df['vx [m/s]'] * F)
    n = calculate_density(df)
    mobility = conductivity / c.e / n
    # print('Carrier density is {:.8E}'.format(n * 1E-6) + ' per cm^{-3}')
    # print('Mobility is {:.10E} (cm^2 / V / s)'.format(mobility * 1E4))
    return mobility


def calc_diff_mobility(chi, df, field):
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
    Nuc = pp.kgrid ** 3
    print('Field specified. Mobility calculated using general definition of conductivity')
    n = calculate_density(df)
    prefactor = 2 * c.e / c.Vuc / Nuc / field
    conductivity = prefactor * np.sum(df['vx [m/s]'] * chi)
    mobility = conductivity / c.e / n
    # print('Carrier density is {:.8E}'.format(n * 1E-6) + ' per cm^{-3}')
    # print('Mobility is {:.10E} (cm^2 / V / s)'.format(mobility * 1E4))
    return mobility


def calc_popsplit(chi, df):
    """Function that calculates the carrrier populations in the Gamma, L, and X valleys given a Chi solution.
    Parameters:
        chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in chi form.
        df (dataframe): Electron DataFrame indexed by kpt
        get_X (Bool): Boolean signifying whether the calculation should also return X valley indices. Not every grid
        contains the X valleys. True -> Get X valley inds
    Returns:
        ng (double): The value of the gamma carrier population in m^-3.
        nl (double): The value of the L carrier population in m^-3.
        nx (dobule): The value of the X carrier population in m^-3.
    """
    f0 = df['k_FD'].values
    f = chi + f0
    g_inds,l_inds,x_inds = split_valleys(df,False)

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
    # prefactor = field * c.e / c.kb_joule / pp.T * f0 * (1 - f0)
    prefactor = field * c.e / c.kb_joule / pp.T * f0
    chi = np.squeeze(f) * np.squeeze(prefactor)
    return chi


if __name__ == '__main__':
    out_loc = pp.outputLoc
    in_loc = pp.inputLoc
    eldf, phdf = load_el_ph_data(in_loc)
    fermi_distribution(eldf)
    conc = calculate_density(eldf)
    print('Carrier concentration is {:.2E} cm^-3'.format(conc * 1E-6))
    # split_L_valleys(eldf, plot_Valleys=True)
    # plt.show()
