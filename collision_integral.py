#!/usr/bin/env python
# import preprocessing_largegrid_comet as preprocessing_largegrid
import numpy as np
import multiprocessing as mp
from functools import partial
import os
import pandas as pd
import time
import constants as c
import problemparameters as pp


def relaxation_times_parallel(k, nlambda):
    """This function calculates the on-diagonal scattering rates, the relaxation times, as per Mahan's Eqn. 11.127.
    Also returns the off-diagonal scattering term. Calculates the Fermi Dirac and Bose Einstein distributions on the fly
    Parameters:
        k (int): Index of the parquet to open and process
        nlambda (int): Number of phonon modes

    Returns:
        Writes the scattering rate to file as .rate. Updates a numpy array called scattering_rates to include the on-diag
        elements.
    """
    prefactor = c.rydberg_to_eV

    def bose_distribution(df):
        """This function is designed to take a Pandas DataFrame containing e-ph data and return
        the Bose-Einstein distribution associated with the mediating phonon mode."""

        df['BE'] = (np.exp((df['q_en [eV]'].values * c.e) / (c.kb_joule * pp.T)) - 1) ** (-1)
        return df

    def fermi_distribution(df):
        """This function is designed to take a Pandas DataFrame containing e-ph data and return
        the Fermi-Dirac distribution associated with both the pre- and post- collision states.
        The distribution is calculated with respect to a given chemical potential, mu"""

        # Physical constants
        df['k_FD'] = (np.exp((df['k_en [eV]'].values * c.e - pp.mu * c.e) / (c.kb_joule * pp.T)) + 1) ** (-1)
        df['k+q_FD'] = (np.exp((df['k+q_en [eV]'].values * c.e - pp.mu * c.e) / (c.kb_joule * pp.T)) + 1) ** (-1)
        return df

    g_df = pd.read_parquet('k{:05d}.parquet'.format(k))
    g_df = fermi_distribution(g_df)
    g_df = bose_distribution(g_df)
    ems_weight = np.multiply(np.multiply(g_df['BE'].values + 1 - g_df['k+q_FD'].values, g_df['g_element'].values),
                             g_df['ems_gaussian'])
    abs_weight = np.multiply(np.multiply((g_df['BE'].values + g_df['k+q_FD'].values), g_df['g_element'].values),
                             g_df['abs_gaussian'])
    g_df['weight'] = ems_weight + abs_weight
    sr = np.sum(g_df['weight'].to_numpy()) * 2 * np.pi / c.hbar_eV * (10 ** -12) / nlambda * prefactor**2
    print(r'For k={:d}, the scattering rate (1/ps) is {:.24E}'.format(k, sr))
    f = open('k{:05d}.rate'.format(k), 'w')
    f.write('{:.24E}'.format(sr))
    f.close()
    scattering_rates[k-1] = sr


def save_weights(k, nlambda):
    """This function calculates and saves the absorption and emission weights from Eqn 11 in Peishi's write up. This
    will allow us to evaluate whether energy is being conserved or not
    Parameters:
        k (int): Index of the parquet to open and process
        nlambda (int): Number of phonon modes

    Returns:
        Writes the scattering rate to file as .rate. Updates a numpy array called scattering_rates to include the on-diag
        elements.
    """
    g_df = pd.read_parquet('k{:05d}.parquet'.format(k))
    simple_abs = np.multiply(np.multiply(g_df['BE'].values + g_df['k+q_FD'].values,g_df['k_FD'].values),1-g_df['k_FD'].values)
    canonical_abs = np.multiply(np.multiply(1-g_df['k+q_FD'].values,g_df['k_FD'].values),g_df['BE'].values)
    simple_ems = np.multiply(np.multiply(1-g_df['k_FD'].values,g_df['k_FD'].values),g_df['BE'].values+1-g_df['k+q_FD'].values)
    canonical_ems = np.multiply(np.multiply(1-g_df['k_FD'].values,g_df['k+q_FD'].values),g_df['BE'].values)
    wuli_ems = np.multiply(np.multiply(g_df['k_FD'].values,1-g_df['k+q_FD'].values),1+g_df['BE'].values)
    abs_s[k-1] = np.sum(simple_abs)
    abs_c[k-1] = np.sum(canonical_abs)
    ems_s[k-1] = np.sum(simple_ems)
    ems_c[k-1] = np.sum(canonical_ems)
    ems_w[k-1] = np.sum(wuli_ems)


def parse_scatteringrates():
    """This function takes .rates located in the chunk_loc and aggregates the scattering rate into a single numpy array.
    Parameters:
        None. Chunk loc intrinsically loaded.

    Returns:
        rates: (nparray) Updates a numpy array called rates to include the on-diagonal elements.
    """
    os.chdir(chunk_loc)
    rates = np.zeros(nkpts)
    for k in range(nkpts):
        thisk = k+1
        f = open('k{:05d}.rate'.format(thisk), 'r')
        rates[k] = float(f.read())
        f.close()
        os.remove('k{:05d}.rate'.format(thisk))
        if thisk % 100 == 0:
            print('Done with k={:d}'.format(thisk))
    return rates


def rta_mobility(datadir, enk, vels):
    """Calculate mobility using RTA and near equilibrium approximation and TDF from Jin Jian's paper.
    Parameters:
        datadir (str): String location containing the scattering_rates.npy
        enk (nparray): Array containing the energy of each electron in eV
        vels (nparray): Array containing the group velocity in the transport direction in m/s
    Returns:
        mobility (dbl): Value of the RTA mobility in m^2/V-s
    """
    os.chdir(datadir)
    rates = np.load('scattering_rates.npy') * 36.5  # arbitrary factor to test if I can get the mobility
    taus = 1 / rates * 1E-12  # in seconds
    npts = 4000  # number of points in the KDE
    ssigma = np.zeros(npts)  # capital sigma as defined in Jin Jian's paper Eqn 3
    # Need to define the energy range that I'm doing integration over
    en_axis = np.linspace(enk.min(), enk.max(), npts)
    dx = (en_axis.max() - en_axis.min()) / npts
    beta = 1 / (c.kb_ev * pp.T)
    fermi = pp.mu  # Fermi level
    spread = 100*dx

    def dfde(x):
        # Approximate dF/dE using dF_0/dE
        return (-1 * beta) * np.exp((x-fermi) * beta) * (np.exp((x-fermi) * beta) + 1)**(-2)

    def gaussian(x, mu, sigma=spread):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)
    # Calculating carrier concentration using Eqn 22 in W. Li PRB 92 (2015)
    nc = (2 / len(enk) / c.Vuc) * np.sum((np.exp((enk - fermi) * beta) + 1) ** -1)
    print('The carrier concentration is {:.3E} in m^-3'.format(nc))

    for k in range(len(enk)):
        istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4*spread/dx), 0))
        iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4*spread/dx), npts - 1))
        ssigma[istart:iend] += vels[k]**2 * taus[k] * gaussian(en_axis[istart:iend], enk[k])
    ssigma = ssigma * 2 / c.Vuc
    conductivity = (c.e**2) / c.e * np.trapz(np.multiply(ssigma, -1 * dfde(en_axis)), en_axis)  # divided by e for eV to J
    print('Conductivity is {:.3E}'.format(conductivity))
    print('dF/dE is {:.3E}'.format(np.sum(-1 * dfde(en_axis))))
    mobility = conductivity / nc / c.e * 1E4 / len(enk)  # 1E4 to get from m^2 to cm^2
    print('Mobility is {:.3E}'.format(mobility))
    font = {'size': 14}
    mpl.rc('font', **font)
    plt.plot(en_axis, np.multiply(ssigma, -1 * dfde(en_axis)), '.')
    plt.xlabel('Energy (eV)')
    plt.ylabel('TDF * dF/dE (a.u.)')
    plt.show()
    return mobility


def assemble_full_matrix(mat_row_dir,simpleLin=True):
    """Once all of the rows have been created, can put them all together. This function enters the mat_row_dir and reads
    the memmaps written by matrixrows_par and assembles a full memmaped array.
    Parameters:
        mat_row_dir (str): String location containing the row-by-row matrix memmaps.
        simpleLin (bool): If "True", matrix will be written in simple linearization.

    Returns:
        Writes scattering_matrix.mmap to file.
    """
    if simpleLin:
        mat_str = '_simple'
    else:
        mat_str = '_canonical'
    matrix = np.memmap('scattering_matrix'+mat_str+'.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
    for k in range(nkpts):
        kind = k + 1
        krow = np.memmap(mat_row_dir+'k{:05d}.mmap'.format(kind), dtype='float64', mode='r', shape=nkpts)
        matrix[k, :] = krow
        if kind % 100 == 0:
            print('Finished k={:d}'.format(kind))


def matrixrows_par(k, nlambda, nk, simpleLin=True):
    """Calculate the full scattering matrix row by row and in parallel. We do this because the data is chunked by
    kpoint, so the most efficient way to do this is by calculating each row of the scattering matrix since the
    information required for each row is contained in each kpoint chunk. This allows the calculation to be done in
    parallel. For each kpoint, will input the row data into a memmap. Afterwards, the full matrix can be assembled
    quickly in serial.
    Parameters:
        k (int): Index of the parquet to open and process
        nlambda (int): Number of phonon modes
        nk (int): Number of total kpts
        simpleLin (bool): If "True", matrix will be written in simple linearization.
    """
    # Create memmap of the row
    istart = time.time()
    if os.path.isfile('k{:05d}.mmap'.format(k)):
        krow = np.memmap('k{:05d}.mmap'.format(k), dtype='float64', mode='r+', shape=nk)
    else:
        krow = np.memmap('k{:05d}.mmap'.format(k), dtype='float64', mode='w+', shape=nk)

    ryd2ev = c.rydberg_to_eV  # Should probably make these originate from constants
    hbar_ev = c.hbar_eV  # Should probably make these originate from constants
    # # Diagonal term
    # # In canonical scattering matrix, the diagonal element is not the scattering rate
    g_df = pd.read_parquet(chunk_loc + 'k{:05d}.parquet'.format(k))
    if simpleLin:
        diag_abs_weight = (g_df['BE'] + g_df['k+q_FD']) * g_df['g_element'] * g_df['abs_gaussian']
        diag_ems_weight = (g_df['BE'] + 1 - g_df['k+q_FD']) * g_df['g_element'] * g_df['ems_gaussian']
    else:
        diag_abs_weight = np.multiply(np.multiply(np.multiply(np.multiply(
            g_df['k_FD'].values, 1 - g_df['k+q_FD'].values), g_df['BE'].values), g_df['g_element']), g_df['abs_gaussian'])

        diag_ems_weight = np.multiply(np.multiply(np.multiply(np.multiply(
            1 - g_df['k_FD'].values, g_df['k+q_FD'].values), g_df['BE'].values), g_df['g_element']), g_df['ems_gaussian'])

    diagterm = np.sum(diag_ems_weight + diag_abs_weight) * 2*np.pi / hbar_ev / nlambda * ryd2ev**2
    krow[k-1] = (-1) * diagterm
    # Calculating nondiagonal terms
    nkprime = np.unique(g_df['k+q_inds'])
    for kp in np.nditer(nkprime):
        kpi = int(kp)
        kp_rows = g_df[g_df['k+q_inds'] == kpi]
        # Commented out section below is the scattering matrix with simple linearization
        if simpleLin:
            abs_weight = np.multiply(np.multiply((kp_rows['BE'].values + 1 - kp_rows['k_FD'].values),
                                                 kp_rows['g_element'].values), kp_rows['abs_gaussian'])
            ems_weight = np.multiply(np.multiply(kp_rows['BE'].values + kp_rows['k_FD'].values,
                                                 kp_rows['g_element'].values), kp_rows['ems_gaussian'])
        else:
            abs_weight = np.multiply(np.multiply(np.multiply(np.multiply(
                kp_rows['k_FD'].values, 1 - kp_rows['k+q_FD'].values), kp_rows['BE'].values), kp_rows['g_element']),
                kp_rows['abs_gaussian'])

            ems_weight = np.multiply(np.multiply(np.multiply(np.multiply(
                1 - kp_rows['k_FD'].values, kp_rows['k+q_FD'].values), kp_rows['BE'].values), kp_rows['g_element']),
                kp_rows['ems_gaussian'])
        tot_weight = abs_weight + ems_weight
        krow[kpi - 1] = np.sum(tot_weight) * 2 * np.pi / hbar_ev / nlambda * ryd2ev**2
    del krow
    iend = time.time()
    print('Row calc for k={:d} took {:.2f} seconds'.format(k, iend - istart))


def matrix_check_colsum(sm,df):
    """We know that the matrix should have columns that sum to zero by the conservation of particle number. This function
    calculates the sum of each column and writes it out.
    Parameters:
        sm (.mmap): scattering matrix

    Returns:
        colsum (dbl): array containing the column sum
    """
    nkpts = len(df)
    colsum = np.zeros(nkpts)
    for k in range(nkpts):
        colsum[k] = np.sum(sm[:, k])
        # print(k)
        # print('Finished k={:d}'.format(k+1))
    return colsum


if __name__ == '__main__':
    # Point to various directories
    data_loc = '/home/alexchoi/storage/k160-0.4eV-200K/'
    chunk_loc = '/home/alexchoi/storage/k160-0.4eV-200K/chunked/'
    nthreads = 48  # The requested number of threads on cluster. Ensure consistency with ntasks in the batch file.

    # Set to true to return matrix in the simple linearization. Be sure to create directories for the matrix rows in
    # data_loc. "mat_rows_simple" for simple and "mat_rows_canonical" for cannonical.
    simplelinear = True

    # Set calc_scattering_rates to "True" to calculate the on-diagonal elements in the simple linearization and save to
    # scattering_rates.npy. Also calculates the off-diagonal elements. Will populate the chunk_loc with .rates for each
    # parquet.
    calc_scattering_rates = True

    # Set calc_weights to "True" to output the sum over qpts of the canonical and simple emission and absoprtion scattering
    # rates, indexed by kpt.
    calc_weights = True

    # Set calc_matrix_rows to True to calculate the scattering matrix row-by-row and store as .mmaps in a subdirectory.
    # These rows can later be assembled.
    calc_matrix_rows = True

    # Set assemble_matrix to True to assemble the scattering matrix from the row parquets stored in mat_row_loc.
    # NOTE: The assemble_full_matrix function will overwrite previous matrix. Be careful
    assemble_matrix = True

    if simplelinear:
        mat_rows_loc = data_loc+'mat_rows_simple'
        mat_str = 'scattering_matrix_simple.mmap'
    else:
        mat_rows_loc = data_loc+'mat_rows_canonical'
        mat_str = 'scattering_matrix_canonical.mmap'

    _, kpts_df, _, _, enq_df = preprocessing_largegrid.loadfromfile(data_loc, matrixel=False)
    nkpts = len(np.unique(kpts_df['k_inds']))
    n_ph_modes = len(np.unique(enq_df['q_inds'])) * len(np.unique(enq_df['im_mode']))
    kinds = np.arange(1, nkpts + 1)
    operatingsystem = 'linux'  # NOTE: Change this to windows if you need

    if calc_scattering_rates:
        os.chdir(chunk_loc)
        scattering_rates = mp.Array('d', [0] * nkpts, lock=False)
        nthreads = 48
        pool = mp.Pool(nthreads)

        start = time.time()
        if operatingsystem is 'windows':
            # NOTE: You also have to manually comment out the last line in relaxation_times_parallel because there's
            # some weird bug I don't know how to fix yet.
            pool.map(partial(relaxation_times_parallel, nlambda=n_ph_modes), kinds)
            sr = parse_scatteringrates()
            np.save(data_loc + 'scattering_rates', sr)
        elif operatingsystem is 'linux':
            pool.map(partial(relaxation_times_parallel, nlambda=n_ph_modes), kinds)
            scattering_rates = np.array(scattering_rates)
            np.save(data_loc + 'scattering_rates', scattering_rates)
        else:
            exit('No operating system specified. Don''t know how to aggregate scattering rates.')
        end = time.time()
        print('Parallel relaxation time calc took {:.2f} seconds'.format(end - start))

    if calc_weights:
        os.chdir(chunk_loc)
        abs_s = mp.Array('d', [0]*nkpts, lock=False)
        abs_c = mp.Array('d', [0]*nkpts, lock=False)
        ems_s = mp.Array('d', [0]*nkpts, lock=False)
        ems_c = mp.Array('d', [0]*nkpts, lock=False)
        ems_w = mp.Array('d', [0]*nkpts, lock=False)
        pool = mp.Pool(nthreads)
        start = time.time()
        pool.map(partial(save_weights, nlambda=n_ph_modes), kinds)
        abs_s = np.array(abs_s)
        abs_c = np.array(abs_c)
        ems_s = np.array(ems_s)
        ems_c = np.array(ems_c)
        ems_w = np.array(ems_w)
        np.save(data_loc + 'abs_s',abs_s)
        np.save(data_loc + 'abs_c',abs_c)
        np.save(data_loc + 'ems_s',ems_s)
        np.save(data_loc + 'ems_c',ems_c)
        np.save(data_loc + 'ems_w',ems_w)
        end = time.time()
        print('Weight calc took {:.2f} seconds'.format(end-start))

    if calc_matrix_rows:
        os.chdir(data_loc)
        # Multiprocessing version
        os.chdir(mat_rows_loc)
        pool = mp.Pool(nthreads)
        start = time.time()
        pool.map(partial(matrixrows_par, nlambda=n_ph_modes, nk=nkpts, simpleLin=simplelinear), kinds)
        end = time.time()
        print('Calc of scattering matrix rows took {:.2f} seconds'.format(end - start))

    if assemble_matrix:
        assemble_full_matrix(mat_rows_loc+'/',simplelinear)

    # Run printout checks on the matrix
    matrix = np.memmap(data_loc + mat_str, dtype='float64', mode='r', shape=(nkpts, nkpts))
    cs = matrix_check_colsum(matrix,kpts_df)
    print('The average absolute value of column sum is {:E}'.format(np.average(np.abs(cs))))
    print('The largest column sum is {:E}'.format(cs.max()))

