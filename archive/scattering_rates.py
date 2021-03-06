#!/usr/bin/env python
import preprocessing_largegrid
import numpy as np
import multiprocessing as mp
from functools import partial
import os
import pandas as pd
import time
<<<<<<< HEAD
import matplotlib.pyplot as plt
import matplotlib as mpl
import constants as c
=======
import re
import matplotlib.pyplot as plt


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def calc_sparsity():
    matrix = np.memmap(data_loc + 'scattering_matrix.mmap', dtype='float64', mode='r', shape=(nkpts, nkpts))
    sparsity = 1 - (np.count_nonzero(matrix) / nkpts**2)
    nelperrow = np.zeros(nkpts)
    for ik in range(nkpts):
        nelperrow[ik] = np.count_nonzero(matrix[ik, :])
        print('For row {:d}, the number of nozero elements is {:f}'.format(ik+1, nelperrow[ik]))
    return sparsity, nelperrow
>>>>>>> 7cb41ec26fb589c276f566fe7e3ea2787a8d302c


def relaxation_times_parallel(k, nlambda):
    """This function calculates the on-diagonal scattering rates, the relaxation times, as per Mahan's Eqn. 11.127.
    Also returns the off-diagonal scattering term.

    CHECK THE FORMULAS FROM MAHAN"""
    mbands = 6
    # divisor = 121904  # number of unique q_ids for k100
    # divisor = 3043002  # number of unique q_ids for k200
    prefactor = 13.6056980659
    ryd2ev = 13.605693122994
    hbar_ev = 6.582119569 * 1E-16

    g_df = pd.read_parquet('k{:05d}.parquet'.format(k))

    ems_weight = (g_df['BE'] + 1 - g_df['k+q_FD']) * (g_df['g_element']) * g_df['ems_gaussian']
    abs_weight = (g_df['BE'] + g_df['k+q_FD']) * (g_df['g_element']) * g_df['abs_gaussian']

    g_df['weight'] = ems_weight + abs_weight

    sr = np.sum(g_df['weight'].to_numpy()) * 2 * np.pi / hbar_ev / nlambda * ryd2ev**2

    print(r'For k={:d}, the scattering rate (1/ps) is {:.24E}'.format(k, sr))

    f = open('k{:05d}.rate'.format(k), 'w')
    f.write('{:.24E}'.format(sr))
    f.close()
    scattering_rates[k-1] = sr


<<<<<<< HEAD
def parse_scatteringrates():
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


=======
>>>>>>> setup_module
def rta_mobility(datadir, enk, vels):
    """Calculate mobility using RTA and near equilibrium approximation.
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
    beta = 1 / (c.kb_ev * c.T)
    fermi = c.mu  # Fermi level
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
<<<<<<< HEAD
    font = {'size': 14}
    mpl.rc('font', **font)
=======

>>>>>>> 7cb41ec26fb589c276f566fe7e3ea2787a8d302c
    plt.plot(en_axis, np.multiply(ssigma, -1 * dfde(en_axis)), '.')
    plt.xlabel('Energy (eV)')
    plt.ylabel('TDF * dF/dE (a.u.)')
    plt.show()
    return mobility


def assemble_full_matrix(mat_row_dir):
<<<<<<< HEAD
    """Once all of the rows have been created, can put them all together. This function enters the mat_row_dir and reads
    the memmaps written by matrixrows_par and assembles a full memmaped array.
    Parameters:
        mat_row_dir (str): String location containing the row-by-row matrix memmaps.

    Returns:
        Writes scattering_matrix.mmap to file.
    """
    matrix = np.memmap('scattering_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
=======
    """Once all of the rows have been created, can put them all together."""
    matrix = np.memmap(data_loc + 'scattering_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
>>>>>>> 7cb41ec26fb589c276f566fe7e3ea2787a8d302c
    for k in range(nkpts):
        kind = k + 1
        krow = np.memmap(mat_row_dir+'k{:05d}.mmap'.format(kind), dtype='float64', mode='r', shape=nkpts)
        matrix[k, :] = krow
        if kind % 100 == 0:
            print('Finished k={:d}'.format(kind))


def matrixrows_par(k, nlambda, nk):
    """Calculate the full scattering matrix row by row and in parallel. We do this because the data is chunked by
    kpoint, so the most efficient way to do this is by calculating each row of the scattering matrix since the
    information required for each row is contained in each kpoint chunk. This allows the calculation to be done in
    parallel. For each kpoint, will input the row data into a memmap. Afterwards, the full matrix can be assembled
    quickly in serial.
    Parameters:
        k
    """
    # Create memmap of the row
    istart = time.time()
    if os.path.isfile('k{:05d}.mmap'.format(k)):
        krow = np.memmap('k{:05d}.mmap'.format(k), dtype='float64', mode='r+', shape=nk)
    else:
        krow = np.memmap('k{:05d}.mmap'.format(k), dtype='float64', mode='w+', shape=nk)

    ryd2ev = 13.605693122994
    hbar_ev = 6.582119569 * 1E-16
    # Diagonal term
    # In canonical scattering matrix, the diagonal element is not the scattering rate
    g_df = pd.read_parquet(chunk_loc + 'k{:05d}.parquet'.format(k))

    # diag_abs_weight = (g_df['BE'] + g_df['k+q_FD']) * g_df['g_element'] * g_df['abs_gaussian']
    # diag_ems_weight = (g_df['BE'] + 1 - g_df['k+q_FD']) * g_df['g_element'] * g_df['ems_gaussian']
    diag_abs_weight = g_df['k_FD'] * (1 - g_df['k+q_FD']) * g_df['BE'] * g_df['g_element'] * g_df['abs_gaussian']
    diag_ems_weight = (1 - g_df['k_FD']) * g_df['k+q_FD'] * g_df['BE'] * g_df['g_element'] * g_df['ems_gaussian']

    diagterm = np.sum(diag_ems_weight + diag_abs_weight) * 2*np.pi / hbar_ev / nlambda * ryd2ev**2
    krow[k-1] = (-1) * diagterm

    # Calculating nondiagonal terms
    nkprime = np.unique(g_df['k+q_inds'])
    for kp in np.nditer(nkprime):
        kpi = int(kp)
        kp_rows = g_df[g_df['k+q_inds'] == kpi]
        # Scattering matrix with simple linearization
        # abs_weight = (kp_rows['BE'] + 1 - kp_rows['k_FD']) * kp_rows['g_element'] * kp_rows['abs_gaussian']
        # ems_weight = (kp_rows['BE'] + kp_rows['k_FD']) * kp_rows['g_element'] * kp_rows['ems_gaussian']
        # Scattering matrix with canonical (symmetric) linearization
        abs_weight = kp_rows['k_FD'] * (1 - kp_rows['k+q_FD']) * kp_rows['BE'] * kp_rows['g_element'] * kp_rows['abs_gaussian']
        ems_weight = (1 - kp_rows['k_FD']) * kp_rows['k+q_FD'] * kp_rows['BE'] * kp_rows['g_element'] * kp_rows['ems_gaussian']
        tot_weight = abs_weight + ems_weight
        krow[kpi - 1] = np.sum(tot_weight) * 2 * np.pi / hbar_ev / nlambda * ryd2ev**2

    del krow
    iend = time.time()
    print('Row calc for k={:d} took {:.2f} seconds'.format(k, iend - istart))


def matrix_check_colsum(sm):
    colsum = np.zeros(nkpts)
    for k in range(nkpts):
        colsum[k] = np.sum(sm[:, k])
        print(k)
        # print('Finished k={:d}'.format(k+1))
    return colsum


if __name__ == '__main__':
    con = preprocessing_largegrid.PhysicalConstants()

    data_loc = '/home/peishi/nvme/k200-0.4eV/'
    chunk_loc = '/home/peishi/nvme/k200-0.4eV/chunked/'
    # data_loc = '/home/peishi/storage/k200-0.4eV/'  # for Comet
    # chunk_loc = '/home/peishi/storage/chunked/'
    # data_loc = '/p/work3/peishi/k200-0.4eV/'  # for gaffney (navy cluster)
    # chunk_loc = '/p/work3/peishi/chunked/'

    _, kpts_df, enk_df, qpts_df, enq_df = preprocessing_largegrid.loadfromfile(data_loc, matrixel=False)
    cartkpts = preprocessing_largegrid.load_vel_data(data_loc, con)

    k_en = (cartkpts.sort_values(by=['k_inds']))['energy'].values
    kvel = cartkpts[['k_inds', 'vx [m/s]']]
    kvel = (kvel.sort_values(by=['k_inds']))['vx [m/s]'].values

    nkpts = len(np.unique(kpts_df['k_inds']))
    n_ph_modes = len(np.unique(enq_df['q_inds'])) * len(np.unique(enq_df['im_mode']))
    kinds = np.arange(1, nkpts + 1)

    operatingsystem = 'linux'  # NOTE: Change this to windows if you need
    calc_scattering_rates = True
    if calc_scattering_rates:
        os.chdir(chunk_loc)
        scattering_rates = mp.Array('d', [0] * nkpts, lock=False)
        nthreads = 6
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

    # rta_mobility(data_loc, k_en, kvel)

    calc_matrix_rows = False
    if calc_matrix_rows:
        os.chdir(data_loc)
        rta_rates = np.load('scattering_rates.npy')

        # Multiprocessing version
        # Need to create a directory called mat_rows inside the data_loc directory to store rows
        os.chdir(data_loc + 'mat_rows')
        # scattering_rates = mp.Array('d', rta_rates, lock=False)
        nthreads = 48
        pool = mp.Pool(nthreads)

        start = time.time()
        pool.map(partial(matrixrows_par, nlambda=n_ph_modes, nk=nkpts), kinds)
        end = time.time()
        print('Calc of scattering matrix rows took {:.2f} seconds'.format(end - start))

    # NOTE: The assemble_full_matrix function will overwrite previous matrix. Be careful
    # assemble_full_matrix(data_loc + 'mat_rows/')

    # matrix = np.memmap(data_loc + 'scattering_matrix.mmap', dtype='float64', mode='r', shape=(nkpts, nkpts))
    # cs = matrix_check_colsum(matrix)
    # print('The average absolute value of column sum is {:E}'.format(np.average(np.abs(cs))))
    # print('The largest column sum is {:E}'.format(cs.max()))
    # a = check_symmetric(matrix)
    # print('Result of check symmetric is ' + str(a))

    # scm = np.memmap(data_loc + 'scattering_matrix.mmap', dtype='float64', mode='r', shape=(nkpts, nkpts))
    # renormalize_matrix(scm)
