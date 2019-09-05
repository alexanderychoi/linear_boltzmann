#!/usr/bin/env python
"""Data processing module for the electron-phonon collision matrix

This is meant to be a frills-free calculation of the electron-phonon collision matrix utilizing the data from
Jin Jian Zhou for GaAs. Updated 7/19/19.
"""

import numpy as np
import itertools
import plotly
import os
import pandas as pd
from tqdm import tqdm, trange

import plotting


class PhysicalConstants:
    """A class with constants to be passed into any method"""
    # Physical parameters
    a = 5.5563606                    # Lattice constant for GaAs [Angstrom]
    kb = 1.38064852*10**(-23)        # Boltzmann constant in SI [m^2 kg s^-2 K^-1]
    T = 300                          # Lattice temperature [K]
    e = 1.602*10**(-19)              # Fundamental electronic charge [C]
    mu = 5.780                       # Chemical potential [eV]
    b = 8/1000                       # Gaussian broadening [eV]
    h = 1.054*10**(-34)              # Reduced Planck's constant [J/s]

    # Lattice vectors
    a1 = np.array([-2.7781803, 0.0000000, 2.7781803])
    a2 = np.array([+0.0000000, 2.7781803, 2.7781803])
    a3 = np.array([-2.7781803, 2.7781803, 0.0000000])

    b1 = np.array([-1.1308095, -1.1308095, +1.1308095])
    b2 = np.array([+1.1308095, +1.1308095, +1.1308095])
    b3 = np.array([-1.1308095, +1.1308095, -1.1308095])
    # b1 = np.array([+1.1308095, -1.1308095, +1.1308095])
    # b2 = np.array([+1.1308095, +1.1308095, -1.1308095])
    # b3 = np.array([-1.1308095, +1.1308095, +1.1308095])


def loadfromfile():
    """Takes the raw text files and converts them into useful dataframes."""
    if os.path.isfile('matrix_el.h5'):
        g_df = pd.read_hdf('matrix_el.h5', key='df')
        print('loaded matrix elements from hdf5')
    else:
        data = pd.read_csv('gaas.eph_matrix', sep='\t', header=None, skiprows=(0,1))
        data.columns = ['0']
        data_array = data['0'].values
        new_array = np.zeros((len(data_array),7))
        for i1 in trange(len(data_array)):
            new_array[i1,:] = data_array[i1].split()

        g_df = pd.DataFrame(data=new_array, columns=['k_inds','q_inds','k+q_inds','m_band','n_band','im_mode','g_element'])
        g_df[['k_inds', 'q_inds', 'k+q_inds', 'm_band', 'n_band', 'im_mode']] = g_df[['k_inds', 'q_inds', 'k+q_inds', 'm_band','n_band', 'im_mode']].apply(pd.to_numeric, downcast='integer')
        g_df = g_df.drop(["m_band", "n_band"],axis=1)
        g_df.to_hdf('matrix_el.h5', key='df')

    return g_df


def vectorbasis2cartesian(coords, vecs):
    """Transform any coordinates written in lattice vector basis into Cartesian coordinates

    Given that the inputs are correct, then the transformation is a simple matrix multiply

    Parameters:
        vecs (numpy array): Array of vectors where the rows are the basis vectors given in Cartesian basis
        coords (numpy array): Array of coordinates where each row is an independent point, so that column 1 corresponds
            to the amount of the basis vector in row 1 of vecs, column 2 is amount of basis vector in row 2 of vecs...

    Returns:
        cartcoords (array): same size as coords but in Cartesian coordinates
    """

    cartcoords = np.matmul(coords, vecs)
    return cartcoords


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

    print('Done bringing points into FBZ!')

    return fbzcoords


def load_vel_data(dirname,cons):
    """Dirname is the name of the directory where the .VEL file is stored.
    The k-points are given in Cartesian coordinates and are not yet shifted
    back into the first Brillouin Zone.
    The result of this function is a Pandas DataFrame containing the columns:
    [k_inds][bands][energy (eV)][kx (1/A)][ky (1/A)[kz (1/A)]"""

    kvel = pd.read_csv(dirname, sep='\t', header=None, skiprows=[0, 1, 2])
    kvel.columns = ['0']
    kvel_array = kvel['0'].values
    new_kvel_array = np.zeros((len(kvel_array), 10))
    for i1 in trange(len(kvel_array)):
        new_kvel_array[i1, :] = kvel_array[i1].split()
    kvel_df = pd.DataFrame(data=new_kvel_array,
                           columns=['k_inds', 'bands', 'energy', 'kx [2pi/alat]', 'ky [2pi/alat]', 'kz [2pi/alat]',
                                    'vx_dir', 'vy_dir', 'vz_dir', 'v_mag [m/s]'])
    kvel_df[['k_inds']] = kvel_df[['k_inds']].apply(pd.to_numeric, downcast='integer') # downcast indices to integer
    cart_kpts_df = kvel_df.copy(deep=True)
    cart_kpts_df['kx [2pi/alat]'] = cart_kpts_df['kx [2pi/alat]'].values * 2 * np.pi / cons.a
    cart_kpts_df['ky [2pi/alat]'] = cart_kpts_df['ky [2pi/alat]'].values * 2 * np.pi / cons.a
    cart_kpts_df['kz [2pi/alat]'] = cart_kpts_df['kz [2pi/alat]'].values * 2 * np.pi / cons.a
    cart_kpts_df.columns = ['k_inds', 'bands', 'energy', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]', 'vx_dir', 'vy_dir',
                            'vz_dir', 'v_mag [m/s]']

    cart_kpts_df['vx [m/s]'] = np.multiply(cart_kpts_df['vx_dir'].values,cart_kpts_df['v_mag [m/s]'])

    cart_kpts_df = cart_kpts_df.drop(['bands'], axis=1)
    cart_kpts_df = cart_kpts_df.drop(['vx_dir','vy_dir','vz_dir'], axis=1)

    cart_kpts_df['FD'] = (np.exp((cart_kpts_df['energy'].values * cons.e - cons.mu * cons.e) / (cons.kb * cons.T)) + 1) ** (-1)

    cart_kpts_df['FD_der [J]'] = -np.multiply(cart_kpts_df['FD'].values**2/(cons.kb*cons.T),np.exp((cart_kpts_df['energy'].values * cons.e - cons.mu * cons.e) / (cons.kb * cons.T)))

    # from sklearn.cluster import KMeans
    # num_clusters = 9
    #
    # my_list = np.linspace(0, num_clusters - 1, num_clusters)
    # k_means = KMeans(n_clusters=num_clusters, n_init=10)
    # X = cart_kpts_df[['kx [1/A]', 'ky [1/A]', 'kz [1/A]']].values
    # k_means.fit(X)
    # k_means_cluster_centers = k_means.cluster_centers_
    #
    # cart_kpts_df['labels'] = k_means.labels_
    #
    # temp_label = cart_kpts_df['labels'].drop_duplicates().values
    # clustering = cart_kpts_df['labels'].values
    # for i0 in range(len(my_list)):
    #     inds = clustering == temp_label[i0]
    #     cart_kpts_df.loc[inds, 'labels'] = my_list[i0] + 1
    #
    # cart_kpts_df['labels'] = cart_kpts_df['labels'].values.astype(int)
    #
    # cart_kpts_df['slice_inds'] = cart_kpts_df.sort_values(['ky [1/A]', 'kz [1/A]','labels'], ascending=True).groupby(
    #     ['ky [1/A]', 'kz [1/A]','labels']).ngroup()

    return cart_kpts_df


def load_enq_data(dirname):
    """Dirname is the name of the directory where the .ENQ file is stored.
    im_mode is the corresponding phonon polarization.
    The result of this function is a Pandas DataFrame containing the columns:
    [q_inds][im_mode][energy (Ryd)]"""

    enq = pd.read_csv(dirname, sep='\t', header=None)
    enq.columns = ['0']
    enq_array = enq['0'].values
    new_enq_array = np.zeros((len(enq_array), 3))
    for i1 in trange(len(enq_array)):
        new_enq_array[i1, :] = enq_array[i1].split()

    enq_df = pd.DataFrame(data=new_enq_array, columns=['q_inds', 'im_mode', 'energy [Ryd]'])
    enq_df[['q_inds', 'im_mode']] = enq_df[['q_inds', 'im_mode']].apply(pd.to_numeric, downcast='integer')

    return enq_df


def load_qpt_data(dirname):
    """Dirname is the name of the directory where the .QPT file is stored.
    The result of this function is a Pandas DataFrame containing the columns:
    [q_inds][b1][b2][b3]"""

    qpts = pd.read_csv(dirname, sep='\t', header=None)
    qpts.columns = ['0']
    qpts_array = qpts['0'].values
    new_qpt_array = np.zeros((len(qpts_array), 4))

    for i1 in trange(len(qpts_array)):
        new_qpt_array[i1, :] = qpts_array[i1].split()

    qpts_df = pd.DataFrame(data=new_qpt_array, columns=['q_inds', 'b1', 'b2', 'b3'])
    qpts_df[['q_inds']] = qpts_df[['q_inds']].apply(pd.to_numeric, downcast='integer')

    return qpts_df


def load_g_data(dirname):
    """Dirname is the name of the directory where the .eph_matrix file is stored.
    The result of this function is a Pandas DataFrame containing the columns:
    [k_inds][q_inds][k+q_inds][m_band][n_band][im_mode][g_element]"""
    data = pd.read_csv('gaas.eph_matrix', sep='\t', header=None, skiprows=(0, 1))
    data.columns = ['0']
    data_array = data['0'].values
    new_array = np.zeros((len(data_array), 7))
    for i1 in trange(len(data_array)):
        new_array[i1, :] = data_array[i1].split()

    g_df = pd.DataFrame(data=new_array,
                        columns=['k_inds', 'q_inds', 'k+q_inds', 'm_band', 'n_band', 'im_mode', 'g_element'])
    g_df[['k_inds', 'q_inds', 'k+q_inds', 'm_band', 'n_band', 'im_mode']] = g_df[
        ['k_inds', 'q_inds', 'k+q_inds', 'm_band', 'n_band', 'im_mode']].apply(pd.to_numeric, downcast='integer')

    g_df = g_df.drop(["m_band", "n_band"], axis=1)
    return g_df


# Now process the data to be in a more convenient form for calculations
def fermi_distribution(g_df,mu,T):
    """This function is designed to take a Pandas DataFrame containing e-ph data and return
    the Fermi-Dirac distribution associated with both the pre- and post- collision states.
    The distribution is calculated with respect to a given chemical potential, mu"""

    # Physical constants
    e = 1.602 * 10 ** (-19)  # fundamental electronic charge [C]
    kb = 1.38064852 * 10 ** (-23);  # Boltzmann constant in SI [m^2 kg s^-2 K^-1]

    g_df['k_FD'] = (np.exp((g_df['k_en [eV]'].values * e - mu * e) / (kb * T)) + 1) ** (-1)
    g_df['k+q_FD'] = (np.exp((g_df['k+q_en [eV]'].values * e - mu * e) / (kb * T)) + 1) ** (-1)

    return g_df


def bose_distribution(g_df,T):
    """This function is designed to take a Pandas DataFrame containing e-ph data and return
    the Bose-Einstein distribution associated with the mediating phonon mode."""
    # Physical constants
    e = 1.602*10**(-19) # fundamental electronic charge [C]
    kb = 1.38064852*10**(-23); # Boltzmann constant in SI [m^2 kg s^-2 K^-1]

    g_df['BE'] = (np.exp((g_df['q_en [eV]'].values*e)/(kb*T)) - 1)**(-1)
    return g_df


def bosonic_processing(g_df,enq_df,T):
    """This function takes the e-ph DataFrame and assigns a phonon energy to each collision
    and calculates the Bose-Einstein distribution"""
    # Physical constants
    e = 1.602 * 10 ** (-19)  # fundamental electronic charge [C]
    kb = 1.38064852 * 10 ** (-23);  # Boltzmann constant in SI [m^2 kg s^-2 K^-1]

    modified_g_df = g_df.copy(deep=True)
    modified_g_df.set_index(['q_inds', 'im_mode'], inplace=True)
    modified_g_df = modified_g_df.sort_index()
    modified_enq_df = enq_df.copy(deep=True)
    modified_enq_df.set_index(['q_inds', 'im_mode'], inplace=True)
    modified_enq_df = modified_enq_df.sort_index()
    modified_enq_df = modified_enq_df.loc[modified_g_df.index.unique()]

    modified_enq_df = modified_enq_df.reset_index()
    modified_enq_df = modified_enq_df.sort_values(['q_inds', 'im_mode'], ascending=True)
    modified_enq_df = modified_enq_df[['q_inds', 'im_mode', 'energy [Ryd]']]
    modified_enq_df['q_id'] = modified_enq_df.groupby(['q_inds', 'im_mode']).ngroup()
    g_df['q_id'] = g_df.sort_values(['q_inds', 'im_mode'], ascending=True).groupby(['q_inds', 'im_mode']).ngroup()

    g_df['q_en [eV]'] = modified_enq_df['energy [Ryd]'].values[g_df['q_id'].values] * 13.6056980659

    g_df = bose_distribution(g_df, T)

    return g_df


def fermionic_processing(g_df,cart_kpts_df,mu,T):
    """This function takes the e-ph DataFrame and assigns the relevant pre and post collision energies
    as well as the Fermi-Dirac distribution associated with both states."""

    # Pre-collision
    modified_g_df_k = g_df.copy(deep=True)
    modified_g_df_k.set_index(['k_inds'], inplace=True)
    modified_g_df_k = modified_g_df_k.sort_index()

    modified_k_df = cart_kpts_df.copy(deep=True)
    modified_k_df.set_index(['k_inds'], inplace=True)
    modified_k_df = modified_k_df.sort_index()
    modified_k_df = modified_k_df.loc[modified_g_df_k.index.unique()]

    modified_k_df = modified_k_df.reset_index()
    modified_k_df = modified_k_df.sort_values(['k_inds'], ascending=True)
    modified_k_df = modified_k_df[['k_inds', 'energy', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]']]

    modified_k_df['k_id'] = modified_k_df.groupby(['k_inds']).ngroup()
    g_df['k_id'] = g_df.sort_values(['k_inds'], ascending=True).groupby(['k_inds']).ngroup()
    g_df['k_en [eV]'] = modified_k_df['energy'].values[g_df['k_id'].values]


    g_df['kx [1/A]'] = modified_k_df['kx [1/A]'].values[g_df['k_id'].values]
    g_df['ky [1/A]'] = modified_k_df['ky [1/A]'].values[g_df['k_id'].values]
    g_df['kz [1/A]'] = modified_k_df['kz [1/A]'].values[g_df['k_id'].values]

    # Post-collision
    modified_g_df_kq = g_df.copy(deep=True)
    modified_g_df_kq.set_index(['k_inds'], inplace=True)
    modified_g_df_kq = modified_g_df_kq.sort_index()

    modified_k_df = cart_kpts_df.copy(deep=True)
    modified_k_df.set_index(['k_inds'], inplace=True)
    modified_k_df = modified_k_df.sort_index()
    modified_k_df = modified_k_df.loc[modified_g_df_kq.index.unique()]

    modified_k_df = modified_k_df.reset_index()
    modified_k_df = modified_k_df.sort_values(['k_inds'], ascending=True)
    modified_k_df = modified_k_df[['k_inds', 'energy', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]']]

    modified_k_df['k+q_id'] = modified_k_df.groupby(['k_inds']).ngroup()
    g_df['k+q_id'] = g_df.sort_values(['k+q_inds'], ascending=True).groupby(['k+q_inds']).ngroup()
    g_df['k+q_en [eV]'] = modified_k_df['energy'].values[g_df['k+q_id'].values]


    g_df['kqx [1/A]'] = modified_k_df['kx [1/A]'].values[g_df['k+q_id'].values]
    g_df['kqy [1/A]'] = modified_k_df['ky [1/A]'].values[g_df['k+q_id'].values]
    g_df['kqz [1/A]'] = modified_k_df['kz [1/A]'].values[g_df['k+q_id'].values]

    g_df = fermi_distribution(g_df, mu, T)

    g_df = g_df.drop(['k_id', 'k+q_id'], axis=1)

    return g_df

def gaussian_weight(g_df,n):
    """This function assigns the value of the delta function of the energy conservation
    approimated by a Gaussian with broadening n"""

    energy_delta_ems = g_df['k_en [eV]'].values - g_df['k+q_en [eV]'].values - g_df['q_en [eV]'].values
    energy_delta_abs = g_df['k_en [eV]'].values - g_df['k+q_en [eV]'].values + g_df['q_en [eV]'].values

    g_df['abs_gaussian'] = 1 / np.sqrt(np.pi) * 1 / n * np.exp(-(energy_delta_abs / n) ** 2)
    g_df['ems_gaussian'] = 1 / np.sqrt(np.pi) * 1 / n * np.exp(-(energy_delta_ems / n) ** 2)

    return g_df


def populate_reciprocals(g_df,b):
    """The g^2 elements are invariant under substitution of the pre-and post- collision indices.
    Therefore, the original e-ph matrix DataFrame only contains half the set, since the other
    half is obtainable. This function populates the appropriate reciprocal elements."""

    modified_g_df = g_df.copy(deep=True)

    flipped_inds = g_df['k_inds'] > g_df['k+q_inds']
    modified_g_df.loc[flipped_inds, 'k_inds'] = g_df.loc[flipped_inds, 'k+q_inds']
    modified_g_df.loc[flipped_inds, 'k+q_inds'] = g_df.loc[flipped_inds, 'k_inds']

    modified_g_df.loc[flipped_inds, 'k_FD'] = g_df.loc[flipped_inds, 'k+q_FD']
    modified_g_df.loc[flipped_inds, 'k+q_FD'] = g_df.loc[flipped_inds, 'k_FD']

    modified_g_df.loc[flipped_inds, 'k_en [eV]'] = g_df.loc[flipped_inds, 'k+q_en [eV]']
    modified_g_df.loc[flipped_inds, 'k+q_en [eV]'] = g_df.loc[flipped_inds, 'k_en [eV]']

    modified_g_df.loc[flipped_inds, 'kqx [1/A]'] = g_df.loc[flipped_inds, 'kx [1/A]']
    modified_g_df.loc[flipped_inds, 'kqy [1/A]'] = g_df.loc[flipped_inds, 'ky [1/A]']
    modified_g_df.loc[flipped_inds, 'kqz [1/A]'] = g_df.loc[flipped_inds, 'kz [1/A]']
    modified_g_df.loc[flipped_inds, 'kx [1/A]'] = g_df.loc[flipped_inds, 'kqx [1/A]']
    modified_g_df.loc[flipped_inds, 'ky [1/A]'] = g_df.loc[flipped_inds, 'kqy [1/A]']
    modified_g_df.loc[flipped_inds, 'kz [1/A]'] = g_df.loc[flipped_inds, 'kqz [1/A]']

    modified_g_df['k_pair_id'] = modified_g_df.groupby(['k_inds', 'k+q_inds']).ngroup()

    reverse_df = modified_g_df.copy(deep=True)

    reverse_df['k_inds'] = modified_g_df['k+q_inds']
    reverse_df['k+q_inds'] = modified_g_df['k_inds']

    reverse_df['k_FD'] = modified_g_df['k+q_FD']
    reverse_df['k+q_FD'] = modified_g_df['k_FD']

    reverse_df['k_en [eV]'] = modified_g_df['k+q_en [eV]']
    reverse_df['k+q_en [eV]'] = modified_g_df['k_en [eV]']

    reverse_df['kqx [1/A]'] = modified_g_df['kx [1/A]']
    reverse_df['kqy [1/A]'] = modified_g_df['ky [1/A]']
    reverse_df['kqz [1/A]'] = modified_g_df['kz [1/A]']
    reverse_df['kx [1/A]'] = modified_g_df['kqx [1/A]']
    reverse_df['ky [1/A]'] = modified_g_df['kqy [1/A]']
    reverse_df['kz [1/A]'] = modified_g_df['kqz [1/A]']

    full_g_df = modified_g_df.append(reverse_df)

    full_g_df = gaussian_weight(full_g_df, b)

    # full_g_df.to_hdf('full_matrix_el.h5', key='df')

    return full_g_df


def cartesian_q_points(qpts_df, con):
    """Given a dataframe containing indexed q-points in terms of the crystal lattice vector, return the dataframe with cartesian q coordinates.
    Parameters:
    -----------
    con :  instance of the physical_constants class
    qpts_df : pandas dataframe containing:

        q_inds : vector_like, shape (n,1)
        Index of q point

        kx : vector_like, shape (n,1)
        x-coordinate in momentum space [1/A]

        ky : vector_like, shape (n,1)
        y-coordinate in momentum space [1/A]

        kz : vector_like, shape (n,1)
        z-coordinate in momentum space [1/A]

    For FCC lattice, use the momentum space primitive vectors as per:
    http://lampx.tugraz.at/~hadley/ss1/bzones/fcc.php

    b1 = 2 pi/a (kx - ky + kz)
    b2 = 2 pi/a (kx + ky - kz)
    b3 = 2 pi/a (-kx + ky + kz)

    Returns:
    --------
    cart_kpts_df : pandas dataframe containing:

        q_inds : vector_like, shape (n,1)
        Index of q point

        kx : vector_like, shape (n,1)
        x-coordinate in Cartesian momentum space [1/m]

        ky : vector_like, shape (n,1)
        y-coordinate in Cartesian momentum space [1/m]

        kz : vector_like, shape (n,1)
        z-coordinate in Cartesian momentum space [1/m]
    """
    cartesian_df = pd.DataFrame(columns=['q_inds', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]'])

    con1 = pd.DataFrame(columns=['kx [1/A]', 'ky [1/A]', 'kz [1/A]'])
    con1['kx [1/A]'] = np.ones(len(qpts_df)) * -1
    con1['ky [1/A]'] = np.ones(len(qpts_df)) * -1
    con1['kz [1/A]'] = np.ones(len(qpts_df)) * 1

    con2 = con1.copy(deep=True)
    con2['kx [1/A]'] = con2['kx [1/A]'].values * -1
    con2['ky [1/A]'] = con2['ky [1/A]'].values * -1

    con3 = con1.copy(deep=True)
    con3['ky [1/A]'] = con2['ky [1/A]'].values
    con3['kz [1/A]'] = con3['kz [1/A]'].values * -1

    cartesian_df['kx [1/A]'] = np.multiply(qpts_df['b1'].values, (con1['kx [1/A]'].values)) + np.multiply(
        qpts_df['b2'].values, (con2['kx [1/A]'].values)) + np.multiply(qpts_df['b3'].values, (con3['kx [1/A]'].values))
    cartesian_df['ky [1/A]'] = np.multiply(qpts_df['b1'].values, (con1['ky [1/A]'].values)) + np.multiply(
        qpts_df['b2'].values, (con2['ky [1/A]'].values)) + np.multiply(qpts_df['b3'].values, (con3['ky [1/A]'].values))
    cartesian_df['kz [1/A]'] = np.multiply(qpts_df['b1'].values, (con1['kz [1/A]'].values)) + np.multiply(
        qpts_df['b2'].values, (con2['kz [1/A]'].values)) + np.multiply(qpts_df['b3'].values, (con3['kz [1/A]'].values))

    cartesian_df['q_inds'] = qpts_df['q_inds'].values

    cartesian_df_edit = cartesian_df.copy(deep=True)

    qx_plus = cartesian_df['kx [1/A]'] > 0.5
    qx_minus = cartesian_df['kx [1/A]'] < -0.5

    qy_plus = cartesian_df['ky [1/A]'] > 0.5
    qy_minus = cartesian_df['ky [1/A]'] < -0.5

    qz_plus = cartesian_df['kz [1/A]'] > 0.5
    qz_minus = cartesian_df['kz [1/A]'] < -0.5

    cartesian_df_edit.loc[qx_plus, 'kx [1/A]'] = cartesian_df.loc[qx_plus, 'kx [1/A]'] - 1
    cartesian_df_edit.loc[qx_minus, 'kx [1/A]'] = cartesian_df.loc[qx_minus, 'kx [1/A]'] + 1

    cartesian_df_edit.loc[qy_plus, 'ky [1/A]'] = cartesian_df.loc[qy_plus, 'ky [1/A]'] - 1
    cartesian_df_edit.loc[qy_minus, 'ky [1/A]'] = cartesian_df.loc[qy_minus, 'ky [1/A]'] + 1

    cartesian_df_edit.loc[qz_plus, 'kz [1/A]'] = cartesian_df.loc[qz_plus, 'kz [1/A]'] - 1
    cartesian_df_edit.loc[qz_minus, 'kz [1/A]'] = cartesian_df.loc[qz_minus, 'kz [1/A]'] + 1

    return cartesian_df, cartesian_df_edit


def shift_into_fbz(qpts_df,kvel_df, con):
    """Shifting k vectors back into BZ."""

    kvel_edit = kvel_df.copy(deep=True)

    # Shift the points back into the first BZ
    kx_plus = kvel_df['kx [2pi/alat]'] > 0.5
    kx_minus = kvel_df['kx [2pi/alat]'] < -0.5

    ky_plus = kvel_df['ky [2pi/alat]'] > 0.5
    ky_minus = kvel_df['ky [2pi/alat]'] < -0.5

    kz_plus = kvel_df['kz [2pi/alat]'] > 0.5
    kz_minus = kvel_df['kz [2pi/alat]'] < -0.5

    kvel_edit.loc[kx_plus, 'kx [2pi/alat]'] = kvel_df.loc[kx_plus, 'kx [2pi/alat]'] -1
    kvel_edit.loc[kx_minus,'kx [2pi/alat]'] = kvel_df.loc[kx_minus,'kx [2pi/alat]'] +1

    kvel_edit.loc[ky_plus,'ky [2pi/alat]'] = kvel_df.loc[ky_plus,'ky [2pi/alat]'] -1
    kvel_edit.loc[ky_minus,'ky [2pi/alat]'] = kvel_df.loc[ky_minus,'ky [2pi/alat]'] +1

    kvel_edit.loc[kz_plus,'kz [2pi/alat]'] = kvel_df.loc[kz_plus,'kz [2pi/alat]'] -1
    kvel_edit.loc[kz_minus,'kz [2pi/alat]'] = kvel_df.loc[kz_minus,'kz [2pi/alat]'] +1

    kvel_df = kvel_edit.copy(deep=True)
    kvel_df.head()

    cart_kpts_df = kvel_df.copy(deep=True)
    cart_kpts_df['kx [2pi/alat]'] = cart_kpts_df['kx [2pi/alat]'].values*2*np.pi/con.a
    cart_kpts_df['ky [2pi/alat]'] = cart_kpts_df['ky [2pi/alat]'].values*2*np.pi/con.a
    cart_kpts_df['kz [2pi/alat]'] = cart_kpts_df['kz [2pi/alat]'].values*2*np.pi/con.a

    cart_kpts_df.columns = ['k_inds', 'bands', 'energy', 'kx [1/A]', 'ky [1/A]','kz [1/A]', 'vx_dir', 'vy_dir', 'vz_dir', 'v_mag [m/s]']

    # Making Cartesian qpoints
    cart_qpts_df,edit_cart_qpts_df = cartesian_q_points(qpts_df)
    return cart_qpts_df,edit_cart_qpts_df


def relaxation_times(g_df,cart_kpts_df):
    """This function calculates the on-diagonal scattering rates, the relaxation times, as per Mahan's Eqn. 11.127.
    Also returns the off-diagonal scattering term.

    CHECK THE FORMULAS FROM MAHAN"""
    g_df['ems_weight'] = np.multiply(
        np.multiply(g_df['BE'].values + 1 - g_df['k+q_FD'].values, g_df['g_element'].values),
        g_df['ems_gaussian']) / 13.6056980659
    g_df['abs_weight'] = np.multiply(np.multiply((g_df['BE'].values + g_df['k+q_FD'].values), g_df['g_element'].values),
                                     g_df['abs_gaussian']) / 13.6056980659

    g_df['weight'] = g_df['ems_weight'].values + g_df['abs_weight'].values

    sr = g_df.groupby(['k_inds'])['weight'].agg('sum') * 2 * np.pi * 2.418 * 10 ** (17) * 10 ** (-12) / len(
        np.unique(g_df['q_id'].values))

    scattering = sr.to_frame().reset_index()
    scattering_array = np.zeros(len(np.unique(cart_kpts_df['k_inds'])))
    scattering_array[scattering['k_inds'].values-1] = scattering['weight'].values

    g_df['OD_abs_weight'] =  np.multiply(np.multiply(g_df['BE'].values +1 - g_df['k_FD'].values,
                                                     g_df['abs_gaussian'].values),g_df['g_element'])/ 13.6056980659
    g_df['OD_ems_weight'] = np.multiply(np.multiply(g_df['BE'].values + g_df['k_FD'].values,
                                                    g_df['ems_gaussian'].values),g_df['g_element'])/ 13.6056980659

    g_df['OD_weight'] = g_df['OD_ems_weight'].values + g_df['OD_abs_weight'].values
    OD_sr = g_df.groupby(['k_inds'])['OD_weight'].agg('sum') * 2 * np.pi * 2.418 * 10 ** (17) * 10 ** (-12) / len(
        np.unique(g_df['q_id'].values))

    OD_scattering = OD_sr.to_frame().reset_index()
    OD_scattering_array = np.zeros(len(np.unique(cart_kpts_df['k_inds'])))
    OD_scattering_array[OD_scattering['k_inds'].values-1] = OD_scattering['weight'].values


    return scattering_array,OD_scattering_array


def RTA_calculation(g_df,cart_kpts_df,E,cons):
    """This function calculates the solution to the one-dimensional Boltzmann Equation under the relaxation time
    approximation."""

    scattering_array,OD_scattering_array = relaxation_times(g_df,cart_kpts_df) # in 1/seconds

    diagonal_arg = cons.h/(cons.e*E)*scattering_array
    matrix_exp = np.diag(np.exp(diagonal_arg))
    neg_matrix_exp = np.diag(np.exp(-diagonal_arg))
    inhomo = cons.h/(cons.kb*cons.T)*np.multiply(cart_kpts_df['vx [m/s]'].values,cart_kpts_df['FD'].values)
    factor = np.dot(matrix_exp,inhomo)
    cart_kpts_df['inhomo'] = inhomo
    cart_kpts_df['factor'] = factor
    kx_sorted_df = cart_kpts_df.copy(deep=True).sort_values(['kx [1/A]'], ascending=True)
    kx_sorted_df['cumint'] = kx_sorted_df.groupby(['slice_inds']).apply(m)['cum_int']



    cart_kpts_df = kx_sorted_df.copy(deep=True).sort_values(['k_inds'], ascending=True)

    soln = np.dot(neg_matrix_exp,cart_kpts_df['cumint'])

    return cart_kpts_df


def practice_calc(g_df, cart_kpts_df, E, cons):
    """This function calculates the solution to the one-dimensional Boltzmann Equation under the relaxation time
     approximation."""

    scattering_array,OD_scattering_array = relaxation_times(g_df, cart_kpts_df)  # in 1/seconds

    cart_kpts_df['RT [s]'] = np.reciprocal(scattering_array)
    cart_kpts_df['MFP [nm]'] = np.multiply(cart_kpts_df['RT [s]'],cart_kpts_df['v_mag [m/s]'])*10**(9)
    kx_array = cart_kpts_df['kx [1/A]'].values*10**(10)

    diagonal_arg = np.multiply(cons.h / (cons.e * E) * scattering_array,kx_array)
    matrix_exp = np.diag(np.exp(-diagonal_arg))
    neg_matrix_exp = np.diag(np.exp(diagonal_arg))
    inhomo = cons.h* np.multiply(cart_kpts_df['vx [m/s]'].values, cart_kpts_df['FD_der [J]'].values)


    factor = np.dot(matrix_exp, inhomo)
    cart_kpts_df['inhomo'] = inhomo
    cart_kpts_df['factor'] = factor
    kx_sorted_df = cart_kpts_df.copy(deep=True).sort_values(['kx [1/A]'], ascending=True)
    kx_sorted_df['cumint'] = kx_sorted_df.groupby(['slice_inds']).apply(m)['cum_int']

    cart_kpts_df = kx_sorted_df.copy(deep=True).sort_values(['k_inds'], ascending=True)

    soln = np.dot(neg_matrix_exp, cart_kpts_df['cumint'])

    return cart_kpts_df


def m(x):
    y = integrate.cumtrapz(x['factor'].values,x['kx [1/A]'].values,initial = 0)
    return pd.DataFrame({'cum_int':y,'factor':x['factor'],'kx [1/A]':x['kx [1/A]']})


def main():

    con = PhysicalConstants()
    reciplattvecs = np.concatenate((con.b1[np.newaxis, :], con.b2[np.newaxis, :], con.b3[np.newaxis, :]), axis=0)

    files_loc = '/home/peishi/nvme/k100-0.3eV'
    os.chdir(files_loc)

    enq_df = load_enq_data('gaas.enq')
    print('Phonon energies loaded')
    print('Phonon qpts loaded')
    cart_kpts_df = load_vel_data('gaas.vel', con)
    print('Electron kpts loaded')
    print('Electron energies loaded')
    g_df = loadfromfile()
    print('E-ph data loaded')

    # At this point, the processed data are:
    # e-ph matrix elements = g_df['k_inds','q_inds','k+q_inds','m_band','n_band','im_mode']
    # k-points = kpts_df['k_inds','b1','b2','b3']
    # q-points = qpts_df['q_inds','b1','b2','b3']
    # k-energy = enk_df['k_inds','band_inds','energy [Ryd]']
    # q-energy = enq_df['q_inds','im_mode','energy [Ryd]']

    # cartesian_df, cartesian_df_edit = cartesian_q_points(qpt_df, con)

    print('Data loading completed. Starting data processing:')

    if os.path.isfile('full_g_df.h5'):
        full_g_df = pd.read_hdf('full_g_df.h5', key='df')
        print('Loaded the fully processed dataframe from hdf5')
    else:
        g_df = bosonic_processing(g_df, enq_df, con.T)
        print('Bosonic processing completed')

        g_df = fermionic_processing(g_df, cart_kpts_df, con.mu, con.T)
        print('Fermionic processing completed')

        full_g_df = populate_reciprocals(g_df, con.b)
        print('Reciprocals populated')

        del g_df
        full_g_df = full_g_df[['k_inds', 'q_inds', 'k+q_inds', 'im_mode', 'g_element', 'q_id',
                               'q_en [eV]', 'BE', 'k_en [eV]', 'k+q_en [eV]',
                               'k_FD', 'k+q_FD', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]', 'kqx [1/A]',
                               'kqy [1/A]', 'kqz [1/A]', 'k_pair_id', 'abs_gaussian', 'ems_gaussian']]
        print('g_dataframe processed')
        full_g_df.to_hdf('full_g_df.h5', key='df')

    # print('Now chunking the full_g_df so each k_ind is in its own hdf5 file')

    # if not os.path.isdir('chunked'):
    #     os.mkdir('chunked')
    # os.chdir('chunked')
    # nk = len(np.unique(full_g_df['k_inds']))
    # if len([name for name in os.listdir('.') if os.path.isfile(name)]) == nk:
    #     print('Data already chunked (probably). Not running chunking code.')
    # else:
    #     for k in np.nditer(np.unique(full_g_df['k_inds'])):
    #         thisdf = full_g_df[full_g_df['k_inds'] == k]
    #         # NOTE: The fill length may change depending on the number of kpoints you have. I am using 4 here because I
    #         # know that there are 9999 or less unique k so that the max number for k_ind is only 4 digits. If you need
    #         # to increase fill length, just change {:04d} to {:0xd} where x is the largest number of digits
    #         thisdf.to_hdf('full_g_{:04d}.h5'.format(k), key='df')
    #
    #     print('Done chunking all of the data.')


if __name__ == '__main__':
    main()



