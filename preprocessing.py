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
    enk_ryd[['k_inds', 'band_inds']] = enk_ryd[['k_inds', 'band_inds']].astype(int)

    # Phonon qpoints in crystal coordinates
    qpts_array = np.loadtxt('gaas.qpts')
    qpts = pd.DataFrame(data=qpts_array, columns=['q_inds', 'b1', 'b2', 'b3'])
    qpts['q_inds'] = qpts['q_inds'].astype(int)

    return kpts, enk_ryd, qpts


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
    for kxi in np.nditer(np.nonzero(smalldkx)):
        kx = uniqkx[kxi]
        fbzcoords[fbzcoords[:, 0] == kx, 0] = uniqkx[kxi+1]
    df[['kx [1/A]', 'ky [1/A]', 'kz [1/A]']] = fbzcoords
    print('Done bringing points into FBZ!')

    return df


def create_el_ph_dataframes(data_dir, overwrite=False):
    """Create dataframes from text files output from perturbo. They contain information
    used later in the calculation. This should only have to be run once.

    Parameters:
        data_dir (str): absolute file path to the perturbo text files
        overwrite (bool): True if you want to overwrite the existing dataframe
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
    electron_df = translate_into_fbz(electron_df)
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
             .foreat(kqi, int(nl_tot)) + 'Increase the number in the chunk_and_pop_recips function.')
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
    print('ITERATING over chunks of the large file.')
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
    print('\nChunking the big text file of all matrix elements, and populating reciprocal elements')
    nkpts = len(el_df['k_inds'])
    mmaplines = 100000

    # Creation of chunks by kpoint. Delete if they exist beforehand to prevent issues
    chunk_loc = data_dir + 'chunked/'
    if os.path.isdir(chunk_loc):
        print('Existing folder of chunks found. Removing to prevent unintended appending.')
        import shutil
        shutil.rmtree(chunk_loc)    
    os.mkdir(chunk_loc)
    chunk_linebyline(data_dir, chunk_loc)

    # After chunking the matrix elements, need to populate each one with the reciprocal data. 
    # Doing this using memmap arrays for each kpoint since they are really fast.
    print('\nGenerating reciprocal matrix elements into memmapped files')  
    if pp.scratchLoc:
        recip_loc = pp.scratchLoc
        print('A scratch location is specified. Reciprocal elements stored in {:s}'.format(recip_loc))
    else:
        recip_loc = data_dir + 'recips/'
        if not os.path.isdir(recip_loc):
            os.mkdir(recip_loc)
            print('A scratch location was NOT specified. Creating recip folder in {:s}'.format(data_dir))
        else:
            print('A scratch location was NOT specified. Using the recip folder in {:s}'.format(data_dir))
    # Create memory mapped arrays which can be opened and have data added to them later
    for k in range(1, nkpts+1):
        kmap = np.memmap(recip_loc + 'k{:05d}.mmap'.format(k), dtype='float64', mode='w+', shape=(mmaplines, 4))
        del kmap
    # Populate the memmaps by using matrix elements from original .eph_matrix file
    # recip_line_key keeps track of where the next line should go for each memmap array.
    pool = mp.Pool(n_th)
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
        inds = kmap[:, 0] != 0
        labels = ['q_inds', 'k+q_inds', 'im_mode', 'g_element']
        recipdf = pd.DataFrame(data=kmap[inds, :], columns=labels)
        if os.path.isfile(chunk_loc+'k{:05d}.parquet'.format(k)):
            kdf = pd.read_parquet(chunk_loc+'k{:05d}.parquet'.format(k))
            fulldf = pd.concat([kdf, recipdf])
        else:
            fulldf = recipdf 
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
    qindex = ((df_k['q_inds'] - 1) * nb + df_k['im_mode']).astype(int) - 1
    df_k['q_en [eV]'] = ph_energies[qindex.values]
    df_k['BE'] = (np.exp(df_k['q_en [eV]'] * c.e / c.kb_joule / pp.T) - 1) ** (-1)
    # Electron occupations: Fermi Dirac distribution
    df_k['k_en [eV]'] = el_energies[np.array(df_k['k_inds']).astype(int) - 1]
    df_k['k+q_en [eV]'] = el_energies[np.array(df_k['k+q_inds']).astype(int) - 1]
    df_k['k_FD'] = (np.exp((df_k['k_en [eV]'] - pp.mu)*c.e / c.kb_joule / pp.T) + 1) ** -1
    df_k['k+q_FD'] = (np.exp((df_k['k+q_en [eV]'] - pp.mu)*c.e / c.kb_joule / pp.T) + 1) ** -1

    df_k.to_parquet(chunkdir + 'k{:05d}.parquet'.format(int(k_ind)))


def delta_weight_par(k_ind, chunkdir):
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

    df = pd.read_parquet(chunkdir + 'k{:05d}.parquet'.format(k_ind))
    energy_delta_ems = df['k_en [eV]'] - df['k+q_en [eV]'] - df['q_en [eV]']
    energy_delta_abs = df['k_en [eV]'] - df['k+q_en [eV]'] + df['q_en [eV]']

    df['abs_gaussian'] = 1 / np.sqrt(np.pi) * 1 / eta * np.exp(-1 * (energy_delta_abs / eta) ** 2)
    df['ems_gaussian'] = 1 / np.sqrt(np.pi) * 1 / eta * np.exp(-1 * (energy_delta_ems / eta) ** 2)

    df.to_parquet(chunkdir + 'k{:05d}.parquet'.format(k_ind))


def add_occ_and_delta_weights(data_dir, n_th, el_df, ph_df):
    """Add Ferm Dirac and Bose occupation function data along with delta function weights
    into the parquets for each chunk

    Parameters:
        data_dir (str): absolute file path to location of the data
        n_th (int): number of threads to use for multiprocessing
        el_df (dataframe): contains electron energies, velocities, Cartesian kpointss
        ph_df (dataframe): contains phonon energies
    """
    print('\nProcessing auxillary information for each kpoint file')
    print('Adding electron and phonon occupations corresponding to T={:.1f} K'.format(pp.T) )

    chunk_loc = data_dir + 'chunked/'
    if not os.path.isdir(chunk_loc):
        os.mkdir(chunk_loc)

    nkpts = len(el_df['k_inds'])
    k_inds = [k0 + 1 for k0 in range(nkpts)]
    
    # Create nkpts by 1 vector of electron energies
    el_df.sort_values(by=['k_inds'], inplace=True)
    k_en_key = np.array(el_df['energy [eV]'])
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
    pool.map(partial(delta_weight_par, chunkdir=chunk_loc), k_inds)
    end = time.time()
    print('Calc. delta function weight using gaussian took {:.2f} seconds'.format(end - start))


def process_perturbo_matrix(data_dir, el_df):
    """Take the scattering matrix created by perturbo and puts it into a memmap array"""
    nk = len(el_df['k_inds'])
    matrix = np.memmap(data_dir + 'scatt_mat_pert.mmap', dtype='float64', mode='w+', shape=(nk, nk))

    counter = 1
    f = open(data_dir + 'gaas.scatt_mat')
    for chunkStart, chunkSize in chunk_iterator(data_dir + 'gaas.scatt_mat', size=512*(1024**2)):
        f.seek(chunkStart)
        all_lines = np.array([float(i) for i in f.read(chunkSize).split()])
        print('Finished READ IN for {:d} chunk of {:2E} bits.'.format(counter, chunkSize))
        chunkdata = np.reshape(all_lines, (-1, 3), order='C')
        inds = chunkdata[:, :2].astype(int) - 1
        matrix[inds[:,0], inds[:,1]] = chunkdata[:, 2] * c.ryd2ev / c.hbar_ev
        counter += 1

    print('Scattering matrix constructed directly from perturbo')


if __name__ == '__main__':
    import problem_parameters as pp
    in_loc = pp.inputLoc
    out_loc = pp.outputLoc
    nthreads = 6

    create_dataframes = False
    create_pert_scatt_mat = False
    chunk_mat_pop_recips = False
    occ_func_and_delta_weights = True

    if create_dataframes:
        create_el_ph_dataframes(in_loc, overwrite=False)

    electron_df, phonon_df = utilities.load_el_ph_data(in_loc)

    if create_pert_scatt_mat:
        process_perturbo_matrix(in_loc, electron_df)
    if chunk_mat_pop_recips:
        global recip_line_key
        recip_line_key = mp.Array('i', [0]*len(np.unique(electron_df['k_inds'])), lock=False)
        chunk_and_pop_recips(in_loc, nthreads, electron_df)
    if occ_func_and_delta_weights:
        add_occ_and_delta_weights(in_loc, nthreads, electron_df, phonon_df)
