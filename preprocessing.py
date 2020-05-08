#!/usr/bin/env python
import numpy as np
import os
import pandas as pd
import sys
import multiprocessing as mp
import time
from functools import partial
import constants as c
import utilities


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
    """
    if not overwrite and \
       (os.path.isfile(data_dir + 'gaas_enq.parquet')
        or os.path.isfile('gaas_full_electron_data.parquet')):
        exit('The dataframes already exist and you did not explicitly request an overwrite.')

    # Phonon energies
    enq_array = np.loadtxt(data_dir + 'gaas.enq')
    enq = pd.DataFrame(data=enq_array, columns=['q_inds', 'im_mode', 'energy [Ryd]'])
    enq[['q_inds', 'im_mode']] = enq[['q_inds', 'im_mode']].astype(int)
    enq['energy [eV]'] = enq['energy [Ryd]'] * c.ryd2ev
    enq.to_parquet(data_dir + 'gaas_enq.parquet')

    # Electron data
    alldat = np.loadtxt(data_dir + 'gaas_fullgrid.kpt', skiprows=4)
    colheadings = ['k_inds', 'bands', 'energy [eV]', 'kx [frac]', 'ky [frac]', 'kz [frac]',
                   'vx_dir', 'vy_dir', 'vz_dir', 'v_mag [m/s]']
    electron_df = pd.DataFrame(data=alldat, columns=colheadings)
    electron_df['k_inds'] = electron_df['k_inds'].astype(int)
    electron_df['kx [1/A]'] = electron_df['kx [frac]'] * 2 * np.pi / c.a
    electron_df['ky [1/A]'] = electron_df['ky [frac]'] * 2 * np.pi / c.a
    electron_df['kz [1/A]'] = electron_df['kz [frac]'] * 2 * np.pi / c.a
    electron_df['vx [m/s]'] = electron_df['vx_dir'] * electron_df['v_mag [m/s]']
    # Drop band indces since only one band
    electron_df = electron_df.drop(['bands'], axis=1)
    electron_df.to_parquet(data_dir + 'gaas_full_electron_data.parquet')


def recip2memmap_par(kq, reciploc, data, nl_tot):
    # The columns of data are ['k_inds', 'q_inds', 'k+q_inds', 'im_mode', 'g_element']
    kqi = int(kq)
    kmap = np.memmap(reciploc + 'k{:05d}.mmap'.format(kqi), 
                     dtype='float64', mode='r+', shape=(nl_tot, 4))
    thiskq = data[data[:, 2] == kqi, :]
    flipped = thiskq[:, [1, 0, 3, 4]]
    nlines = thiskq.shape[0]
    startind = recip_line_key[kqi-1]
    if startind + nlines > nl_tot:
        exit('There were more than {:d} lines for k+q={:d} than lines allocated in the memmap.\n'
             .format(kqi, int(nl_tot)) + 'Increase the number in the chunk_and_pop_recips function.')
    else:
        kmap[startind:startind+nlines, :] = flipped
        del kmap
        recip_line_key[kqi-1] += nlines


def chunk_iterator(fname, size=512 * 1024 * 1024):
    """Python iterator to give location in file and chunk size.

    Parameters:
        fname (str): name of file to chunk
        size (float): the chunksize to be read into memory
    Yields:
        chunkStart (int): bit where a particular chunk starts in the file
        chunkEnd-chunkStart (int): number of bits in this chunk
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
    """
    matrixel_path = dataloc +  'gaas.eph_matrix'
    f = open(matrixel_path)
    nGB = 0
    nlines = 0

    for chunkStart, chunkSize in chunk_iterator(matrixel_path):
        f.seek(chunkStart)
        all_lines = np.array([float(i) for i in f.read(chunkSize).split()])
        print('Finished read for GB number {:d}'.format(nGB))
        data = np.reshape(all_lines, (-1, 5), order='C')
        nlines += data.shape[0]
        this_df = pd.DataFrame(data=data,
                               columns=['k_inds', 'q_inds', 'k+q_inds', 'im_mode', 'g_element'])
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
    ln = open(dataloc + 'totallines', 'w')
    ln.write('Total number of lines is {:d}'.format(nlines))
    # For the k160-0.4eV matrix, it should have 628241287 lines including the header


def chunk_and_pop_recips(data_dir, n_th, el_df):
    """Split matrix elements by kpoint and populate reciprocal matrix elements.
    
    Parameters:
        data_dir (str): absolute file path to directory containing matrix elements
        n_th (int): number of multiprocessing threads
        el_df (dataframe): contains electron energies, velocities and cartesian kpoints
    """
    nkpts = len(el_df['k_inds'])
    mmaplines = 100000

    chunk_loc = data_dir + 'chunked/'
    if not os.path.isdir(chunk_loc):
        os.mkdir(chunk_loc)
    # Creation of matrix chunks
    if len([name for name in os.listdir(chunk_loc) if os.path.isfile(chunk_loc + name)]) == nkpts:
        print('\nData already chunked (probably). Not running chunking code.')
    else:
        chunk_linebyline(data_dir, chunk_loc)

    # After chunking the matrix elements, need to populate each one with the reciprocal data. 
    # Doing this using memmap arrays for each kpoint since they are really fast.
    print('\nGenerating reciprocal matrix elements into memmapped files')  
    if pp.scratchLoc:
        recip_loc = pp.scratchLoc
        print('A scratch location is specified. Reciprocal elements stored in {:s}'.format(scratch_loc))
    else:
        recip_loc = data_dir + 'recips/'
        if not os.path.isdir(recip_loc):
            os.mkdir(recip_loc)
        print('A scratch location was NOT specified. Creating recip folder in {:s}'.format(data_dir))

    # Create memory mapped arrays which can be opened and have data added to them later
    for k in range(1, nkpts+1):
        kmap = np.memmap(recip_loc + 'k{:05d}.mmap'.format(k), dtype='float64', mode='w+', shape=(mmaplines, 4))
        del kmap
    # Populate the memmaps by using matrix elements from original .eph_matrix file
    # recip_line_key keeps track of where the next line should go for each memmap array.
    pool = mp.Pool(n_th)
    global recip_line_key
    recip_line_key = mp.Array('i', [0]*nkpts, lock=False)
    counter = 1
    f = open(data_dir + 'gaas.eph_matrix')
    for chunkStart, chunkSize in chunk_iterator(data_dir + 'gaas.eph_matrix', size=512*(1024**2)):
        f.seek(chunkStart)
        all_lines = np.array([float(i) for i in f.read(chunkSize).split()])
        print('Finished READ IN for {:d} chunk of {:2E} bits.'.format(counter, chunkSize))
        chunkdata = np.reshape(all_lines, (-1, 5), order='C')
        kqinds = np.unique(chunkdata[:, 2])
        print('There are {:d} unique k+q inds for the number {:d} chunk'.format(len(kqinds), counter))
        start = time.time()
        pool.map(partial(recip2memmap_par, data=chunkdata, reciploc=recip_loc, nl_tot=mmaplines), kqinds)
        end = time.time()
        print('Finished populating reciprocals. It took {:.2f} seconds'.format(end - start))
        counter += 1

    # Now need to add the data from each memmap file into the corresponding kpoint
    print('\nAdding data from memmaps into parquet chunks')
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
        print('Finished adding memmap data into parquet chunks.')


def occupation_functions_par(k_ind, ph_energies, nb, el_energies, chunkdir):
    """Multithreaded calculation of phonon and electron occupation functions

    Parameters:
        k_ind (int): Unique index for kpoint
        ph_energies (numpy vector): Phonon energies in a 1D vector where the index of the entry corresponds
        nb (int): Total number of phonon bands. Used to index the phonon energies
        el_energies (numpy vector): Electron energies in 1D vector, same scheme as above but since electrons only in
            one band, the length of the vector is the same as number of kpts
        chunkdir (str): absolute file path to directory for the chunks(parquets)
    """
    df_k = pd.read_parquet(chunkdir + 'k{:05d}.parquet'.format(int(k_ind)))
    # Add column of k_ind which should be same number for a given chunk. Takes up space...
    if not np.any(df_k.columns == 'k_inds'):
        df_k['k_inds'] = k_ind * np.ones(len(df_k.index))
    # Phonon occupations: Bose Einstein distribution
    qindex = ((g_df['q_inds'] - 1) * nb + g_df['im_mode']).astype(int) - 1
    g_df['q_en [eV]'] = enq_key[qindex]
    g_df['BE'] = (np.exp(df['q_en [eV]'] * c.e / c.kb_joule / pp.T) - 1) ** (-1)
    # Electron occupations: Fermi Dirac distribution
    g_df['k_en [eV]'] = enk_key[np.array(g_df['k_inds']).astype(int) - 1]
    g_df['k+q_en [eV]'] = enk_key[np.array(g_df['k+q_inds']).astype(int) - 1]
    df['k_FD'] = (np.exp((df['k_en [eV]'] - fermilevel)*c.e / c.kb_joule / pp.T) + 1) ** -1
    df['k+q_FD'] = (np.exp((df['k+q_en [eV]'] - fermilevel)*c.e / c.kb_joule / pp.T) + 1) ** -1

    df_k.to_parquet('k{:05d}.parquet'.format(int(k_ind)))


def delta_weight_par(k_ind):
    """Multithreaded calculation of value of delta function approximated as a gaussian

    Parameters:
        k_ind (int): integer 
    """
    # If no contributions from energy diffs larger than 8 meV then that implies that 
    # the Gaussian function should be near zero across an 8 meV span. We know that 
    # integrating from -3*sigma to +3*sigma gives 99.7% of a Gaussian, so let's take 
    # the width of 6 sigma to be equal to 8 meV which implies sigma is 8/6 meV
    # sigma = 8 / 4 / 1000
    # eta = np.sqrt(2) * sigma  # Gaussian broadening parameter which is equal to sqrt(2) * sigma (the stddev) in [eV]
    eta = 5

    df = pd.read_parquet('k{:05d}.parquet'.format(k_ind))
    energy_delta_ems = df['k_en [eV]'] - df['k+q_en [eV]'] - df['q_en [eV]']
    energy_delta_abs = df['k_en [eV]'] - df['k+q_en [eV]'] + df['q_en [eV]']

    df['abs_gaussian'] = 1 / np.sqrt(np.pi) * 1 / eta * np.exp(-1 * (energy_delta_abs / eta) ** 2)
    df['ems_gaussian'] = 1 / np.sqrt(np.pi) * 1 / eta * np.exp(-1 * (energy_delta_ems / eta) ** 2)

    df.to_parquet('k{:05d}.parquet'.format(k_ind))
    del df


def add_occ_and_delta_weights(data_dir, n_th, el_df, ph_df):
    """Add Ferm Dirac and Bose occupation function data along with delta function weights
    into the parquets for each chunk

    Parameters:
        data_dir (str): absolute file path to location of the data
        n_th (int): number of threads to use for multiprocessing
        el_df (dataframe): contains electron energies, velocities, Cartesian kpointss
        ph_df (dataframe): contains phonon energies
    """
    print('Processing auxillary information for each kpoint file')
    print('Adding electron and phonon occupations corresponding to T={:.1f} K'.format(pp.T) )

    chunk_loc = data_dir + 'chunked/'
    if not os.path.isdir(chunk_loc):
        os.mkdir(chunk_loc)

    k_inds = [k0 + 1 for k0 in range(nkpts)]
    
    # Create nkpts by 1 vector of electron energies
    k_en_key = np.array(el_df['energy'].sort_values(by=['k_inds']))
    # Create a nqpts by 1 vector of phonon energies where nqpts is total number of phonon modes.
    # This is useful because now to figure out what the phonon energies are, all you need to do 
    # is make a single index where (q_ind - 1)*(n_bands) + im_mode = the index into this en_q_key
    ph_df.sort_values(by=['q_inds', 'im_mode'], inplace=True)
    q_en_key = np.array(ph_df['energy [eV]'])
    n_ph_bands = np.max(ph_df['im_mode'])

    pool = mp.Pool(n_th)
    start = time.time()
    pool.map(partial(occupation_functions_par, ph_energies=q_en_key, nb=n_ph_bands, 
                     el_energies=k_en_key, chunkdir=chunk_loc), k_inds)
    end = time.time()
    print('Calc. occupation functions for el and ph took {:.2f} seconds'.format(end - start))

    start = time.time()
    pool.map(delta_weight_par, k_inds)
    end = time.time()
    print('Calc. delta function weight using gaussian took {:.2f} seconds'.format(end - start))


def process_perturbo_matrix(data_dir,el_df):
    """Take the scattering matrix created by perturbo and puts it into a memmap array"""
    nk = len(el_df['k_inds'])
    matrix = np.memmap(data_dir + 'scatt_mat_pert.mmap', dtype='float64', mode='w+', shape=(nk, nk))

    hbar_evs = 6.582119569E-16
    ryd2ev = 13.605698

    counter = 1
    f = open(data_dir + 'gaas.scatt_mat')
    for chunkStart, chunkSize in chunk_iterator(data_dir + 'gaas.scatt_mat', size=512*(1024**2)):
        f.seek(chunkStart)
        all_lines = np.array([float(i) for i in f.read(chunkSize).split()])
        print('Finished READ IN for {:d} chunk of {:2E} bits.'.format(counter, chunkSize))
        chunkdata = np.reshape(all_lines, (-1, 3), order='C')
        inds = chunkdata[:, :2].astype(int) - 1
        matrix[inds[:,0], inds[:,1]] = chunkdata[:, 2] * ryd2ev / hbar_evs
        counter += 1

    # The diagonal is empty. Put the scattering rates into it.
    # data = np.loadtxt(data_dir + 'gaas.rates', skiprows=5)
    # rates = data[:, 3] / 1000 / hbar_evs  # in 1/s
    # diag = np.arange(nk)
    # matrix[diag, diag] = rates
    
    print('Scattering matrix constructed directly from perturbo')


if __name__ == '__main__':
    import problem_parameters as pp
    in_loc = pp.inputLoc
    out_loc = pp.outputLoc
    nthreads = 24

    create_dataframes = False
    create_pert_scatt_mat = True
    chunk_mat_pop_recips = False
    occ_func_and_delta_weights = False

    if create_dataframes:
        create_el_ph_dataframes(in_loc, overwrite=False)

    electron_df, phonon_df = utilities.load_el_ph_data(in_loc)

    if create_pert_scatt_mat:
        process_perturbo_matrix(in_loc, electron_df)
    if chunk_mat_pop_recips:
        chunk_and_pop_recips(in_loc, nthreads, electron_df)
    if occ_func_and_delta_weights:
        add_occ_and_delta_weights(in_loc, nthreads, electron_df, phonon_df)
