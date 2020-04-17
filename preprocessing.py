#!/usr/bin/env python
import numpy as np
import os
import pandas as pd
import sys
import multiprocessing as mp
import time
from functools import partial
import cProfile
import constants as c
import problem_parameters as pp


def load_optional_data(data_dir):
    """Create dataframes for kpts and qpts in crystal coordinates and Ryd energy.
    Not typically used, so in a separate function."""

     # Electron kpoints in crystal coordinates
    kpts_array = np.loadtxt('gaas.kpts')
    kpts = pd.DataFrame(data=kpts_array, columns=['k_inds', 'b1', 'b2', 'b3'])
    kpts['k_inds'] = kpts['k_inds'].astype(int)

    # Electron energies
    enk_array = np.loadtxt('gaas.enk')
    enk_ryd = pd.DataFrame(data=enk_array, columns=['k_inds', 'band_inds', 'energy [Ryd]'])
    enk_ryd[['k_inds', 'band_inds']] = enk[['k_inds', 'band_inds']].astype(int)

    # Phonon qpoints in crystal coordinates
    qpts_array = np.loadtxt('gaas.qpts')
    qpts = pd.DataFrame(data=qpts_array, columns=['q_inds', 'b1', 'b2', 'b3'])
    qpts['q_inds'] = qpts['q_inds'].astype(int)

    return kpts, enk_ryd, qpts


def create_el_ph_dataframes(data_dir, overwrite=False):
    """Create dataframes from text files output from perturbo. They contain information
    used later in the calculation. This should only have to be run once.

    Parameters:
        data_dir (str): absolute file path to the perturbo text files

    Outputs:
        None. Just saves the dataframes as parquet files.
    """
    if not overwrite and \
       (os.path.isfile(data_dir + 'gaas_enq.parquet')
        or os.path.isfile('gaas_full_electron_data.parquet')):
        exit('The dataframes already exist and you did not explicitly request an overwrite.')

    # Phonon energies
    enq_array = np.loadtxt('gaas.enq')
    enq = pd.DataFrame(data=enq_array, columns=['q_inds', 'im_mode', 'energy [Ryd]'])
    enq[['q_inds', 'im_mode']] = enq[['q_inds', 'im_mode']].astype(int)
    enq['energy [eV]'] = enq['energy [Ryd]'] * c.ryd2ev
    enq.to_parquet('gaas_enq.parquet')

    # Electron data
    alldat = np.loadtxt('gaas_fullgrd.kpt', skiprows=4)
    colheadings = ['k_inds', 'bands', 'energy', 'kx [frac]', 'ky [frac]', 'kz [frac]',
                   'vx_dir', 'vy_dir', 'vz_dir', 'v_mag [m/s]']
    electron_df = pd.DataFrame(data=alldat, columns=colheadings)
    kvel_df['k_inds'] = kvel_df['k_inds'].astype(int)
    cart_kpts = kvel_df.copy(deep=True)
    cart_kpts['kx [1/A]'] = cart_kpts['frac'] * 2 * np.pi / c.a
    cart_kpts['ky [1/A]'] = cart_kpts['frac'] * 2 * np.pi / c.a
    cart_kpts['kz [1/A]'] = cart_kpts['frac'] * 2 * np.pi / c.a

    cart_kpts['vx [m/s]'] = cart_kpts['vx_dir'] * cart_kpts['v_mag [m/s]']
    # Drop band indces since only one band
    cart_kpts = cart_kpts.drop(['bands'], axis=1)
    cart_kpts = cart_kpts.drop(['vx_dir', 'vy_dir', 'vz_dir'], axis=1)
    cart_kpts.to_parquet('gaas_full_electron_data.parquet')


def create_q_en_key(df):
    """Create a n by 1 vector of phonon energies where n is total number of phonon modes = qpts x polarizations.

    This is useful because now to figure out what the phonon energies are, all you need to do is arithmetic on the q_ind
    and im_mode where (q_ind - 1)*(n_bands) + im_mode = the index in this en_q_key
    """

    df.sort_values(by=['q_inds', 'im_mode'], inplace=True)
    en_q_key = np.array(df['energy [Ryd]']) * 13.6056980659  # convert from Ryd to eV
    nb = np.max(df['im_mode'])  # need total number of bands

    return en_q_key, nb


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
    """Python iterator to give location in file while chunking and chunk size.

    Parameters:
        fname (str):
        size (float):

    Yields:
        chunkStart (file):
        chunkEnd (file):
    """
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


def chunk_linebyline(dataloc, chunkloc):
    """Load the matrix elements in chunks, calculate additional info needed, and store into file for each kpoint

    Parameters:
        matrixel_path (str): String with absolute path to matrix elements file
        chunkloc (str): String with absolute path to where you want to store the chunks

    Returns:
        None. Just a function call to load shit and process it into the right form
    """

    matrixel_path = dataloc +  'gaas.eph_matrix'
    f = open(matrixel_path)
    nGB = 0
    nlines = 0

    for chunkStart, chunkSize in chunkify(matrixel_path):
        f.seek(chunkStart)
        all_lines = np.array([float(i) for i in f.read(chunkSize).split()])
        print('Finished read for GB number {:d}'.format(nGB))
        data = np.reshape(all_lines, (-1, 5), order='C')
        nlines += data.shape[0]
        this_df = pd.DataFrame(data=data,
                               columns=['k_inds', 'q_inds', 'k+q_inds', 'im_mode', 'g_element'])
        # this_df.drop(columns=['m_band', 'n_band'], inplace=True)
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
    os.chdir(dataloc)
    ln = open('totallines', 'w')
    ln.write('Total number of lines is {:d}'.format(nlines))
    # For the k160-0.4eV matrix, it should have 628241287 lines including the header


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
    for k in range(1, n+1):
        kmap = np.memmap(dir + 'k{:05d}.mmap'.format(k), dtype='float64', mode='w+', shape=(nl, 4))
        del kmap

chunk_and_pop_recips(data_dir, n_th):
    """"""
    nkpts = 23643
    mmaplines = 100000
    # n_th is number of threads

    if len([name for name in os.listdir(chunk_loc) if os.path.isfile(chunk_loc + name)]) == nkpts:
        print('Data already chunked (probably). Not running chunking code.')
    else:
        chunk_linebyline(data_loc, chunk_loc)

     # After chunking the matrix elements, need to populate each one with the reciprocal data. Doing this using numpy
    # memmap arrays for each kpoint since they are really fast.
    populate_memmaps = True  # Finished doing this. Takes 9 hours
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
            chunkdata = np.reshape(all_lines, (-1, 5), order='C')
            # chunkdata = chunkdata[:, [0, 1, 2, 5, 6]]
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


def add_occ_and_delta_weights(data_dir, n_th):
    """Add Ferm Dirac and Bose occupation function data along with delta function weights
    into the parquets for each chunk

    Parameters:
        data_dir (str): absolute file path to location of the data
        n_th (int): number of threads to use for multiprocessing

    Returns:
        None. Just adds data into the parquet chunks. Overwrites existing data
    """
    print('Processing auxillary information for each kpoint file')

    chunk_loc = data_dir + 'chunked'
    os.chdir(chunk_loc)

    pool = mp.Pool(n_th)

    k_inds = [k0 + 1 for k0 in range(nkpts)]

    # Don't need a separate key for k energies since only one band. I checked for both datasets
    k_en_key = cart_kpts_df.sort_values(by=['k_inds'])
    k_en_key = np.array(k_en_key['energy'])
    q_en_key, nphononbands = create_q_en_key(enq_df)  # need total number of bands

    start = time.time()
    pool.map(partial(chunked_bosonic_fermionic, ph_energies=q_en_key, nb=nphononbands, el_energies=k_en_key,
                     constants=con), k_inds)
    end = time.time()
    print('Parallel fermionic and bosonic processing took {:.2f} seconds'.format(end - start))

    start = time.time()
    pool.map(gaussian_weight_inchunks, k_inds)
    end = time.time()
    print('Parallel gaussian weights took {:.2f} seconds'.format(end - start))


if __name__ == '__main__':
    # Point to inputs and outputs
    in_loc = pp.inputLoc
    out_loc = pp.outputLoc
    nthreads = 24

    create_dataframes = False
    chunk_mat_pop_recips = False
    occ_func_and_delta_weights = False

    if create_dataframes:
        create_el_ph_dataframes(in_loc, overwrite=False)
    if chunk_mat_pop_recips:
        chunk_and_pop_recips(in_loc, nthreads)
    if occ_func_and_delta_weights:
        add_occ_and_delta_weights(in_loc, nthreads, pp.T)
