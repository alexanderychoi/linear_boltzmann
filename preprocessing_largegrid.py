#!/usr/bin/env python
"""Data processing module for the electron-phonon collision matrix

This is meant to be a frills-free calculation of the electron-phonon collision matrix utilizing the data from
Jin Jian Zhou for GaAs.
"""

import numpy as np
import os
import pandas as pd
import sys
import multiprocessing as mp
import time
from functools import partial
import cProfile
import numba


class PhysicalConstants:
    """A class with constants to be passed into any method

    Doesn't really have an instantiation function but works for now...
    """
    # Physical parameters
    a = 5.5563606                    # Lattice constant for GaAs [Angstrom]
    kb_joule = 1.38064852*10**(-23)  # Boltzmann constant in SI [m^2 kg s^-2 K^-1]
    kb_ev = 8.617333 * (10 ** -5)    # Boltzmann constant in eV/K
    T = 300                          # Lattice temperature [K]
    e = 1.602*10**(-19)              # Fundamental electronic charge [C]
    # mu = 5.780                       # Chemical potential [eV] old 10/8/2019
    mu = 6.03                        # Chemical potential [eV] based on carrier density cited in JJ Zhou paper
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


def loadfromfile(data_dir, matrixel=True):
    """Takes the raw text files and converts them into useful dataframes."""
    os.chdir(data_dir)

    if matrixel:  # maybe sometimes don't need matrix elements, just need the other data
        if os.path.isfile('matrix_el.h5'):
            g_df = pd.read_hdf('matrix_el.h5', key='df')
            print('Loaded matrix elements from hdf5')
        else:
            print('Reading in matrix elements from text file')
            data = np.loadtxt('gaas.eph_matrix', skiprows=1)

            g_df = pd.DataFrame(data=data, columns=['k_inds','q_inds','k+q_inds','m_band','n_band','im_mode','g_element'])
            g_df[['k_inds', 'q_inds', 'k+q_inds', 'm_band', 'n_band', 'im_mode']] = g_df[[
                'k_inds', 'q_inds', 'k+q_inds','m_band','n_band', 'im_mode']].astype(int)
            g_df = g_df.drop(["m_band", "n_band"], axis=1)
            # g_df.to_hdf('matrix_el.h5', key='df')
    else:
        g_df = []

    # Importing electron k points
    if os.path.isfile('gaas.kpts.parquet'):
        kpts = pd.read_parquet('gaas.kpts.parquet')
    else:
        kpts_array = np.loadtxt('gaas.kpts')
        kpts = pd.DataFrame(data=kpts_array, columns=['k_inds', 'b1', 'b2', 'b3'])
        kpts['k_inds'] = kpts['k_inds'].astype(int)
        kpts.to_parquet('gaas.kpts.parquet')

    # Import electron energy library
    if os.path.isfile('gaas.enk.parquet'):
        enk = pd.read_parquet('gaas.enk.parquet')
    else:
        enk_array = np.loadtxt('gaas.enk')
        enk = pd.DataFrame(data=enk_array, columns=['k_inds', 'band_inds', 'energy [Ryd]'])
        enk[['k_inds', 'band_inds']] = enk[['k_inds', 'band_inds']].astype(int)
        enk.to_parquet('gaas.enk.parquet')

    # Import phonon q-point index
    if os.path.isfile('gaas.qpts.parquet'):
        qpts = pd.read_parquet('gaas.qpts.parquet')
    else:
        qpts_array = np.loadtxt('gaas.qpts')
        qpts = pd.DataFrame(data=qpts_array, columns=['q_inds', 'b1', 'b2', 'b3'])
        qpts['q_inds'] = qpts['q_inds'].astype(int)
        qpts.to_parquet('gaas.qpts.parquet')

    # Import phonon energy library
    if os.path.isfile('gaas.enq.parquet'):
        enq = pd.read_parquet('gaas.enq.parquet')
    else:
        enq_array = np.loadtxt('gaas.enq')
        enq = pd.DataFrame(data=enq_array, columns=['q_inds', 'im_mode', 'energy [Ryd]'])
        enq[['q_inds', 'im_mode']] = enq[['q_inds', 'im_mode']].astype(int)
        enq.to_parquet('gaas.enq.parquet')

    return g_df, kpts, enk, qpts, enq


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


def load_vel_data(data_dir, cons):
    """Dirname is the name of the directory where the .VEL file is stored.
    The k-points are given in Cartesian coordinates and are not yet shifted
    back into the first Brillouin Zone.
    The result of this function is a Pandas DataFrame containing the columns:
    [k_inds][bands][energy (eV)][kx (1/A)][ky (1/A)[kz (1/A)]"""

    os.chdir(data_dir)

    if False:
    # if os.path.isfile('gaas.vel.parquet'):
        cart_kpts = pd.read_parquet('gaas.vel.parquet')
    else:
        kvel = np.loadtxt('gaas.vel', skiprows=3)
        kvel_df = pd.DataFrame(data=kvel,
                               columns=['k_inds', 'bands', 'energy', 'kx [2pi/alat]', 'ky [2pi/alat]', 'kz [2pi/alat]',
                                        'vx_dir', 'vy_dir', 'vz_dir', 'v_mag [m/s]'])
        kvel_df[['k_inds']] = kvel_df[['k_inds']].astype(int)
        cart_kpts = kvel_df.copy(deep=True)
        cart_kpts['kx [2pi/alat]'] = cart_kpts['kx [2pi/alat]'].values * 2 * np.pi / cons.a
        cart_kpts['ky [2pi/alat]'] = cart_kpts['ky [2pi/alat]'].values * 2 * np.pi / cons.a
        cart_kpts['kz [2pi/alat]'] = cart_kpts['kz [2pi/alat]'].values * 2 * np.pi / cons.a
        cart_kpts.columns = ['k_inds', 'bands', 'energy', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]', 'vx_dir', 'vy_dir',
                             'vz_dir', 'v_mag [m/s]']

        cart_kpts['vx [m/s]'] = np.multiply(cart_kpts['vx_dir'].values, cart_kpts['v_mag [m/s]'])

        cart_kpts = cart_kpts.drop(['bands'], axis=1)
        cart_kpts = cart_kpts.drop(['vx_dir', 'vy_dir', 'vz_dir', 'v_mag [m/s]'], axis=1)

        cart_kpts['FD'] = (np.exp((cart_kpts['energy'].values * cons.e - cons.mu * cons.e)
                                  / (cons.kb_joule * cons.T)) + 1) ** (-1)
        cart_kpts.to_parquet('gaas.vel.parquet')

    return cart_kpts


def bosonic_processing(g_df, enq_key, nb, T):
    """This function takes the e-ph DataFrame and assigns a phonon energy to each collision
    and calculates the Bose-Einstein distribution"""
    # Physical constants

    e = 1.602 * 10 ** (-19)  # fundamental electronic charge [C]
    kb = 1.38064852 * 10 ** (-23)  # Boltzmann constant in SI [m^2 kg s^-2 K^-1]

    qindex = ((np.array(g_df['q_inds']) - 1)*nb + np.array(g_df['im_mode'])).astype(int) - 1

    g_df['q_en [eV]'] = enq_key[qindex]  # It's already in eV!!!

    def bose_distribution(df, temp):
        """This function is designed to take a Pandas DataFrame containing e-ph data and return
        the Bose-Einstein distribution associated with the mediating phonon mode."""

        df['BE'] = (np.exp((df['q_en [eV]'].values * e) / (kb * temp)) - 1) ** (-1)
        return df

    g_df = bose_distribution(g_df, T)

    return g_df


def fermionic_processing(g_df, enk_key, mu, T):
    """This function takes the e-ph DataFrame and assigns the relevant pre and post collision energies
    as well as the Fermi-Dirac distribution associated with both states."""
    k = (g_df['k_inds'].values)[0]
    print('k={:d} at T={:.0f}'.format(int(k), T))

    # Pre-collision
    g_df['k_en [eV]'] = enk_key[np.array(g_df['k_inds']).astype(int) - 1]

    # Post-collision
    g_df['k+q_en [eV]'] = enk_key[np.array(g_df['k+q_inds']).astype(int) - 1]

    def fermi_distribution(df, fermilevel, temp):
        """This function is designed to take a Pandas DataFrame containing e-ph data and return
        the Fermi-Dirac distribution associated with both the pre- and post- collision states.
        The distribution is calculated with respect to a given chemical potential, mu"""

        # Physical constants
        e = 1.602 * 10 ** (-19)  # fundamental electronic charge [C]
        kb = 1.38064852 * 10 ** (-23)  # Boltzmann constant in SI [m^2 kg s^-2 K^-1]

        df['k_FD'] = (np.exp((df['k_en [eV]'].values * e - fermilevel * e) / (kb * temp)) + 1) ** (-1)
        df['k+q_FD'] = (np.exp((df['k+q_en [eV]'].values * e - fermilevel * e) / (kb * temp)) + 1) ** (-1)

        return df

    g_df = fermi_distribution(g_df, mu, T)

    return g_df


def memmap_par(kq, data):
    # The columns of data are ['k_inds', 'q_inds', 'k+q_inds', 'im_mode', 'g_element']
    kqi = int(kq)
    kmap = np.memmap('k{:05d}.mmap'.format(kqi), dtype='float64', mode='r+', shape=(100000, 4))
    thiskq = data[data[:, 2] == kqi, :]
    flipped = thiskq[:, [1, 0, 3, 4]]
    nlines = thiskq.shape[0]
    startind = recip_line_key[kqi-1]
    kmap[startind:startind+nlines, :] = flipped
    del kmap
    recip_line_key[kqi-1] += nlines
    if startind + nlines > 100000:
        print('There were more than 100000 k+q lines for kq={:d}'.format(kqi))


def gaussian_weight_inchunks(k_ind):
    """Function that is easy to use with multiprocessing"""
    print('Doing k={:d}'.format(k_ind))

    # If no contributions from energy diffs larger than 8 meV then that implies that the Gaussian function should be
    # near zero across an 8 meV span. We know that integrating from -3*sigma to +3*sigma gives 99.7% of a Gaussian, so
    # let's take the width of 6 sigma to be equal to 8 meV which implies sigma is 8/6 meV
    # sigma = 8 / 4 / 1000
    # eta = np.sqrt(2) * sigma  # Gaussian broadening parameter which is equal to sqrt(2) * sigma (the stddev) in [eV]
    eta = 5

    df = pd.read_parquet('k{:05d}.parquet'.format(k_ind))
    energy_delta_ems = df['k_en [eV]'].values - df['k+q_en [eV]'].values - df['q_en [eV]'].values
    energy_delta_abs = df['k_en [eV]'].values - df['k+q_en [eV]'].values + df['q_en [eV]'].values

    df['abs_gaussian'] = 1 / np.sqrt(np.pi) * 1 / eta * np.exp(-1 * (energy_delta_abs / eta) ** 2)
    df['ems_gaussian'] = 1 / np.sqrt(np.pi) * 1 / eta * np.exp(-1 * (energy_delta_ems / eta) ** 2)

    df.to_parquet('k{:05d}.parquet'.format(k_ind))
    del df


def chunkify(fname, size=512 * 1024 * 1024):
    """Function that can be used as a python iterator"""
    fileEnd = os.path.getsize(fname)
    f = open(fname, 'rb')
    # Want to readline for first line with headings so that numpy doesn't try to convert it to float.
    headings = f.readline().decode('utf-8')
    print('CHUNKING using chunkify function')
    chunkEnd = f.tell()
    while True:
        chunkStart = chunkEnd
        f.seek(size, 1)
        f.readline()
        chunkEnd = f.tell()
        yield chunkStart, chunkEnd - chunkStart
        if chunkEnd > fileEnd:
            break
    f.close()


def chunk_linebyline(matrixel_path, chunkloc):
    """Load the matrix elements in chunks, calculate additional info needed, and store into file for each kpoint

    Parameters:
        matrixel_path (str): String with absolute path to matrix elements file
        chunkloc (str): String with absolute path to where you want to store the chunks

    Returns:
        None. Just a function call to load shit and process it into the right form
    """
    f = open(matrixel_path)
    nGB = 0
    nlines = 0

    for chunkStart, chunkSize in chunkify(matrixel_path):
        f.seek(chunkStart)
        all_lines = np.array([float(i) for i in f.read(chunkSize).split()])
        print('Finished read for GB number {:d}'.format(nGB))
        data = np.reshape(all_lines, (-1, 7), order='C')
        nlines += data.shape[0]
        this_df = pd.DataFrame(data=data,
                               columns=['k_inds', 'q_inds', 'k+q_inds', 'm_band', 'n_band', 'im_mode', 'g_element'])
        this_df.drop(columns=['m_band', 'n_band'], inplace=True)
        for k in np.nditer(np.unique(this_df['k_inds'])):
            to_store = this_df[this_df['k_inds'] == k].copy()
            k_fname = chunkloc+'k{:05d}.parquet'.format(int(k))
            to_store.drop(columns=['k_inds'], inplace=True)
            if os.path.isfile(k_fname):
                prevdata = pd.read_parquet(k_fname)
                to_store = pd.concat([prevdata, to_store])
            to_store.to_parquet(k_fname, index=False)
        nGB += 1

    print('Total number of lines is {:d}'.format(nlines))
    os.chdir('/home/peishi/nvme/k200-0.4eV')
    ln = open('totallines', 'w')
    ln.write('Total number of lines is {:d}'.format(nlines))


def create_q_en_key(df):
    """Create a n by 1 vector of phonon energies where n is total number of phonon modes = qpts x polarizations.

    This is useful because now to figure out what the phonon energies are, all you need to do is arithmetic on the q_ind
    and im_mode where (q_ind - 1)*(n_bands) + im_mode = the index in this en_q_key
    """

    df.sort_values(by=['q_inds', 'im_mode'], inplace=True)
    en_q_key = np.array(df['energy [Ryd]']) * 13.6056980659  # convert from Ryd to eV
    nb = np.max(df['im_mode'])  # need total number of bands

    return en_q_key, nb


def chunked_bosonic_fermionic(k_ind, ph_energies, nb, el_energies, constants):
    """Add data to each chunked kpoint file like electron/phonon energies, occupation factors, gaussian weights.

    Function written per k since the function must be self contained at the module level to be run in a parallel way using multiprocessing.

    Parameters:
        k_ind (int): Unique index for kpoint
        ph_energies (numpy vector): Phonon energies in a 1D vector where the index of the entry corresponds
        nb (int): Total number of phonon bands. Used to index the phonon energies
        el_energies (numpy vector): Electron energies in 1D vector, same scheme as above but since electrons only in
            one band, the length of the vector is the same as number of kpts
        constants (object): Instance of PhysicalConstants class
    """
    # print('doing k={:d}'.format(k_ind))
    df_k = pd.read_parquet('k{:05d}.parquet'.format(int(k_ind)))

    if not np.any(df_k.columns == 'k_inds'):
        df_k['k_inds'] = k_ind * np.ones(len(df_k.index))

    # if not np.any(df_k.columns == 'q_en [eV]'):
    df_k = bosonic_processing(df_k, ph_energies, nb, con.T)

    # if not np.any(df_k.columns == 'k_en [eV]'):
    df_k = fermionic_processing(df_k, el_energies, con.mu, con.T)

    df_k.to_parquet('k{:05d}.parquet'.format(int(k_ind)))


def creating_mmap(dir, n, nl):
    """Creates the memory mapped numpy arrays which can be opened and have data added to them later"""
    os.chdir(dir)
    for k in range(1, n+1):
        kmap = np.memmap('k{:05d}.mmap'.format(k), dtype='float64', mode='w+', shape=(nl, 4))
        del kmap


if __name__ == '__main__':
    con = PhysicalConstants()

    # data_loc = '/home/peishi/nvme/k100-0.3eV/'
    # chunk_loc = '/home/peishi/nvme/k100-0.3eV/chunked/'
    # recip_loc = '/home/peishi/nvme/k100-0.3eV/recips/'
    data_loc = '/home/peishi/nvme/k200-0.4eV/'
    chunk_loc = '/home/peishi/nvme/k200-0.4eV/chunked/'
    recip_loc = '/home/peishi/nvme/k200-0.4eV/recips/'

    nkpts = 42433
    mmaplines = 100000
    nthreads = 6

    # For the 200x200x200 kpoints will need to load and process line by line since file too large.
    # First load all the other stuff.
    load_data = True
    if load_data:
        print('Loading data from ' + data_loc)
        load_matrix_elements = False
        _, kpts_df, enk_df, qpts_df, enq_df = loadfromfile(data_loc, matrixel=load_matrix_elements)

        print('Matrix elements loaded (%s), electron kpoints and energies, phonon qpoints and energies loaded'
              % load_matrix_elements)

        cart_kpts_df = load_vel_data(data_loc, con)
        print('Electron kpts and energies loaded')

        # At this point, the processed data are:
        # e-ph matrix elements = g_df['k_inds','q_inds','k+q_inds','m_band','n_band','im_mode']
        # k-points = kpts_df['k_inds','b1','b2','b3']
        # q-points = qpts_df['q_inds','b1','b2','b3']
        # k-energy = enk_df['k_inds','band_inds','energy [Ryd]']
        # q-energy = enq_df['q_inds','im_mode','energy [Ryd]']
        print('Non matrix element data loaded.')
    else:
        print('Did not load non-matrix element data')

    # Chunking matrix element data into separate files for each kpoint
    if len([name for name in os.listdir(chunk_loc) if os.path.isfile(chunk_loc + name)]) == nkpts:
        print('Data already chunked (probably). Not running chunking code.')
    else:
        pass
        # chunk_linebyline(data_loc + 'gaas.eph_matrix', chunk_loc)

    # After chunking the matrix elements, need to populate each one with the reciprocal data. Doing this using numpy
    # memmap arrays for each kpoint since they are really fast.
    populate_memmaps = False  # Finished doing this. Takes 9 hours
    if populate_memmaps:
        print('Populating reciprocals by adding data into memmapped files')
        os.chdir(recip_loc)  # THIS IS REALLY IMPORTANT FOR SOME REASON

        # recip_line_key keeps track of where the next line should go for each memmap array, since appending is hard.
        recip_line_key = mp.Array('i', [0]*nkpts, lock=False)
        pool = mp.Pool(nthreads)

        creating_mmap(recip_loc, nkpts, mmaplines)

        counter = 1
        f = open(data_loc + 'gaas.eph_matrix')
        for chunkStart, chunkSize in chunkify(data_loc + 'gaas.eph_matrix', size=512*(1024**2)):
            f.seek(chunkStart)
            all_lines = np.array([float(i) for i in f.read(chunkSize).split()])
            print('Finished READ IN for {:d} chunks of 512 MB'.format(counter))
            # The columns are ['k_inds', 'q_inds', 'k+q_inds', 'm_band', 'n_band', 'im_mode', 'g_element']
            # We only use columns 0, 1, 2, 5, 6
            chunkdata = np.reshape(all_lines, (-1, 7), order='C')
            chunkdata = chunkdata[:, [0, 1, 2, 5, 6]]
            kqinds = np.unique(chunkdata[:, 2])
            print('There are {:d} unique k+q inds for the number {:d} chunk'.format(len(kqinds), counter))
            start = time.time()
            pool.map(partial(memmap_par, data=chunkdata), kqinds)
            end = time.time()
            print('Processing reciprocals took {:.2f} seconds'.format(end - start))
            counter += 1

    # Now need to add the data from each memmap file into the corresponding kpoint
    memmap_to_parquets = False  # DO NOT RUN THIS UNLESS YOU KNOW YOU NEED TO DO IT!!!
    if memmap_to_parquets:
        print('Adding data from memmaps into parquet chunks')
        os.chdir(chunk_loc)
        for i in range(nkpts):
            k = i + 1
            kmap = np.memmap(recip_loc+'k{:05d}.mmap'.format(k), dtype='float64', mode='r+', shape=(mmaplines, 4))
            kdf = pd.read_parquet(chunk_loc+'k{:05d}.parquet'.format(k))
            inds = kmap[:, 0] != 0
            recipdf = pd.DataFrame(data=kmap[inds, :], columns=kdf.columns)
            fulldf = pd.concat([kdf, recipdf])
            fulldf.to_parquet(chunk_loc+'k{:05d}.parquet'.format(k))
            if k % 100 == 0:
                print('Added memmap data for k={:d}'.format(k))

    # Since the above only chunks the matrix elements line by line, need to add the rest of the data. I'm doing this in
    # a parallel, distributed memory fashion. Each process runs independently of the others.
    do_fermionic_bosonic_gaussianweights = True
    if do_fermionic_bosonic_gaussianweights:
        print('Processing auxillary information for each kpoint file')
        os.chdir(chunk_loc)

        pool = mp.Pool(nthreads)

        k_inds = [k0 + 1 for k0 in range(nkpts)]
        # k_inds = [1]

        # Don't need a separate key for k energies since only one band. I checked for both datasets
        k_en_key = cart_kpts_df.sort_values(by=['k_inds'])
        k_en_key = np.array(k_en_key['energy'])
        q_en_key, nphononbands = create_q_en_key(enq_df)  # need total number of bands

        start = time.time()
        pool.map(partial(chunked_bosonic_fermionic, ph_energies=q_en_key, nb=nphononbands, el_energies=k_en_key,
                         constants=con), k_inds)
        end = time.time()
        print('Parallel fermionic and bosonic processing took {:.2f} seconds'.format(end - start))

        # start = time.time()
        # pool.map(gaussian_weight_inchunks, k_inds)
        # end = time.time()
        # print('Parallel gaussian weights took {:.2f} seconds'.format(end - start))




