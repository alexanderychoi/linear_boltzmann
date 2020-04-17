import numpy as np
import problemparameters as pp
import constants as c
import os
import pandas as pd


def load_electron_df(inLoc):
    """Loads the electron dataframe from the .VEL output from Perturbo, transforms into cartesian coordinates, and
    translates the points back into the FBZ.
    Parameters:
        inLoc (str): String containing the location of the directory containing the input text file.
    Returns:
        None. Just prints the values of the problem parameters.
    """

    os.chdir(inLoc)
    kvel = np.loadtxt('gaas.vel', skiprows=3)
    kvel_df = pd.DataFrame(data=kvel,
                           columns=['k_inds', 'bands', 'energy', 'kx [2pi/alat]', 'ky [2pi/alat]', 'kz [2pi/alat]',
                                    'vx_dir', 'vy_dir', 'vz_dir', 'v_mag [m/s]'])
    kvel_df[['k_inds']] = kvel_df[['k_inds']].astype(int)
    cart_kpts = kvel_df.copy(deep=True)
    cart_kpts['kx [2pi/alat]'] = cart_kpts['kx [2pi/alat]'].values * 2 * np.pi / c.a
    cart_kpts['ky [2pi/alat]'] = cart_kpts['ky [2pi/alat]'].values * 2 * np.pi / c.a
    cart_kpts['kz [2pi/alat]'] = cart_kpts['kz [2pi/alat]'].values * 2 * np.pi / c.a
    cart_kpts.columns = ['k_inds', 'bands', 'energy', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]', 'vx_dir', 'vy_dir',
                         'vz_dir', 'v_mag [m/s]']
    cart_kpts['vx [m/s]'] = np.multiply(cart_kpts['vx_dir'].values, cart_kpts['v_mag [m/s]'])
    cart_kpts = cart_kpts.drop(['bands'], axis=1)
    cart_kpts = cart_kpts.drop(['vx_dir', 'vy_dir', 'vz_dir'], axis=1)

    cart_kpts['FD'] = (np.exp((cart_kpts['energy'].values * c.e - c.mu * c.e)
                              / (c.kb_joule * pp.T)) + 1) ** (-1)
    reciplattvecs = np.concatenate((c.b1[np.newaxis, :], c.b2[np.newaxis, :], c.b3[np.newaxis, :]), axis=0)
    fbzcartkpts = translate_into_fbz(cart_kpts.values[:, 2:5], reciplattvecs)
    fbzcartkpts = pd.DataFrame(data=fbzcartkpts, columns=['kx [1/A]', 'ky [1/A]', 'kz [1/A]'])
    fbzcartkpts = pd.concat([cart_kpts[['k_inds', 'vx [m/s]', 'energy']], fbzcartkpts], axis=1)
    fbzcartkpts.to_pickle(in_Loc + 'electron_df.pkl')


def translate_into_fbz(coords, rlv):
    """Manually translate coordinates back into first Brillouin zone

    The way we do this is by finding all the planes that form the FBZ boundary and the vectors that are associated
    with these planes. Since the FBZ is centered on Gamma, the position vectors of the high symmetry points are also
    vectors normal to the plane. Once we have these vectors, we find the distance between a given point (u) and
    a plane (n) using the dot product of the difference vector (u-n). And if the distance is positive, then translate
    back into the FBZ.

    Args:
        rlv: numpy array of vectors where the rows are the reciprocal lattice vectors given in Cartesian basis
        coords: numpy array of coordinates where each row is a point. For N points, coords is N x 3

    Returns:
        fbzcoords:
    """
    # First, find all the vectors defining the boundary
    b1, b2, b3 = rlv[0, :], rlv[1, :], rlv[2, :]
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

    fbzcoords = np.copy(coords)
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
    for kxi in np.nditer(np.nonzero(smalldkx)):
        kx = uniqkx[kxi]
        fbzcoords[fbzcoords[:, 0] == kx, 0] = uniqkx[kxi+1]
    print('Done bringing points into FBZ!')
    return fbzcoords


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
    chi = np.squeeze(f) * np.squeeze(prefactor)
    return chi


if __name__ == '__main__':
    out_Loc = pp.outputLoc
    in_Loc = pp.inputLoc
    read_problem_params(in_Loc)