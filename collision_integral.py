#!/usr/bin/env python
# import preprocessing_largegrid
import numpy as np
import multiprocessing as mp
from functools import partial
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import constants as c
import utilities


def scattering_rates_parallel(k, nlambda, chunk_loc):
    """This function calculates the on-diagonal scattering rates, the relaxation times, as per Mahan's Eqn. 11.127."""
    ryd2ev = 13.6056980659
    hbar_ev = 6.582119569 * 1E-16

    g_df = pd.read_parquet(chunk_loc + 'k{:05d}.parquet'.format(k))

    abs_weight = (g_df['BE'] + g_df['k+q_FD']) * g_df['g_element'] * g_df['abs_gaussian']
    ems_weight = (g_df['BE'] + 1 - g_df['k+q_FD']) * g_df['g_element'] * g_df['ems_gaussian']
    tot_weight = ems_weight + abs_weight

    sr = np.sum(tot_weight) * 2*np.pi / hbar_ev / nlambda * ryd2ev**2 * 1E-12
    rates_array[k-1] = sr


def rta_tdf_mobility(datadir, el_df):
    """Calculate mobility using RTA and a transport distribution function like Bernardi does"""
    enk = el_df['energy [eV]']
    vels = el_df['vx [m/s]']
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
    font = {'size': 14}
    mpl.rc('font', **font)
    plt.plot(en_axis, np.multiply(ssigma, -1 * dfde(en_axis)), '.')
    plt.xlabel('Energy (eV)')
    plt.ylabel('TDF * dF/dE (a.u.)')
    plt.show()
    return mobility


def assemble_full_matrix(data_dir, electron_df):
    """Create a scattering matrix from the matrix rows"""
    nkpts = len(np.unique(el_df['k_inds']))
    matrix = np.memmap(data_dir + 'scattering_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
    mat_row_dir = data_dir + 'mat_rows/'
    for k in range(nkpts):
        kind = k + 1
        krow = np.memmap(mat_row_dir+'k{:05d}.mmap'.format(kind), dtype='float64', mode='r', shape=nkpts)
        matrix[k, :] = krow
        if kind % 100 == 0:
            print('Finished k={:d}'.format(kind))


def matrixrows_par(k, nlambda, nk, simple=True):
    """Calculate the full scattering matrix row by row and in parallel
    
    We do this because the data is chunked by kpoint, so the most efficient way to do this is by calculating each row
    of the scattering matrix since the information required for each row is contained in each kpoint chunk. This allows 
    the calculation to be done in parallel. For each kpoint, will input the row data into a memmap. Afterwards, the 
    full matrix can be assembled quickly in serial.
    """
    istart = time.time()
    # Create memmap of the row
    krow = np.memmap('k{:05d}.mmap'.format(k), dtype='float64', mode='w+', shape=nk)

    ryd2ev = 13.6056980659
    hbar_ev = 6.582119569 * 1E-16

    # Diagonal term
    g_df = pd.read_parquet(chunk_loc + 'k{:05d}.parquet'.format(k))

    if simple:
        diag_abs_weight = (g_df['BE'] + g_df['k+q_FD']) * g_df['g_element'] * g_df['abs_gaussian']
        diag_ems_weight = (g_df['BE'] + 1 - g_df['k+q_FD']) * g_df['g_element'] * g_df['ems_gaussian']
    else:
        diag_abs_weight = g_df['k_FD'] * (1 - g_df['k+q_FD']) * g_df['BE'] * g_df['g_element'] * g_df['abs_gaussian']
        diag_ems_weight = g_df['k_FD'] * (1 - g_df['k+q_FD']) * (g_df['BE'] + 1) * g_df['g_element'] * g_df['ems_gaussian']

    diagterm = np.sum(diag_ems_weight + diag_abs_weight) * 2*np.pi / hbar_ev / nlambda * ryd2ev**2
    krow[k-1] = (-1) * diagterm

    # Calculating nondiagonal terms
    nkprime = np.unique(g_df['k+q_inds'])
    for kp in np.nditer(nkprime):
        kpi = int(kp)
        kp_rows = g_df[g_df['k+q_inds'] == kpi]

        if simple:
            # Scattering matrix with simple linearization
            abs_weight = (kp_rows['BE'] + 1 - kp_rows['k_FD']) * kp_rows['g_element'] * kp_rows['abs_gaussian']
            ems_weight = (kp_rows['BE'] + kp_rows['k_FD']) * kp_rows['g_element'] * kp_rows['ems_gaussian']
        else:
            # Scattering matrix with canonical (symmetric) linearization
            abs_weight = kp_rows['k_FD'] * (1 - kp_rows['k+q_FD']) * kp_rows['BE'] * kp_rows['g_element'] * kp_rows['abs_gaussian']
            ems_weight = kp_rows['k_FD'] * (1 - kp_rows['k+q_FD']) * (kp_rows['BE'] + 1) * kp_rows['g_element'] * kp_rows['ems_gaussian']

        tot_weight = abs_weight + ems_weight
        krow[kpi - 1] = np.sum(tot_weight) * 2 * np.pi / hbar_ev / nlambda * ryd2ev**2

    del krow
    iend = time.time()
    print('Row calc for k={:d} took {:.2f} seconds'.format(k, iend - istart))


def matrix_check_colsum(data_loc, el_df):
    nkpts = len(np.unique(el_df['k_inds']))
    matrix = np.memmap(data_loc + 'scattering_matrix.mmap', dtype='float64', mode='r', shape=(nkpts, nkpts))

    colsum = np.zeros(nkpts)
    for k in range(nkpts):
        colsum[k] = np.sum(sm[:, k])
    print('The average absolute value of column sum is {:E}'.format(np.average(np.abs(colsum))))
    print('The largest column sum is {:E}'.format(colsum.max()))
    return colsum


def scattering_rates(data_loc, el_df, ph_df, n_th)
    print('\nCalculating the scattering rates for each kpoint using {:d} threads'.format(n_th))
    nkpts = len(np.unique(el_df['k_inds']))
    n_ph_modes = len(np.unique(ph_df['q_inds'])) * len(np.unique(ph_df['im_mode']))
    kinds = np.arange(1, nkpts + 1)
    pool = mp.Pool(n_th)
    chunk_dir = data_loc + 'chunked/'

    start = time.time()
    pool.map(partial(scattering_rates_parallel, nlambda=n_ph_modes, chunk_loc=chunk_dir), kinds)
    aggregated_rates = np.array(rates_array)
    np.save(data_loc + 'scattering_rates', aggregated_rates)
    end = time.time()
    print('Parallel relaxation time calc took {:.2f} seconds'.format(end - start))


def scattering_matrix(data_loc, el_df, n_th, simplebool)
    print('\nCalculating the scattering matrix row by row...')
    nkpts = len(np.unique(el_df['k_inds']))
    n_ph_modes = len(np.unique(ph_df['q_inds'])) * len(np.unique(ph_df['im_mode']))
    kinds = np.arange(1, nkpts + 1)
    pool = mp.Pool(n_th)
    mat_row_dir = data_loc + 'mat_rows/'

    start = time.time()
    pool.map(partial(matrixrows_par, nlambda=n_ph_modes, nk=nkpts, row_dir=mat_row_dir, simple=simplebool), kinds)
    end = time.time()
    print('Calc of scattering matrix rows took {:.2f} seconds'.format(end - start))
    print('\nCreating scattering matrix using the rows. Prior matrices of same name overwritten.')
    assemble_full_matrix(data_loc)


if __name__ == '__main__':
    import problem_parameters as pp
    in_loc = pp.inputLoc
    out_loc = pp.outputLoc
    electron_df, phonon_df = utilities.load_el_ph_data(in_loc)
    nthreads = 8

    calc_scattering_rates = True
    calc_rta_mobility = False
    build_scattering_matrix = False
    check_column_sums = False

    if calc_scattering_rates:
        rates_array = mp.Array('d', [0] * len(np.unique(electron_df['k_inds'])), lock=False)
        scattering_rates(in_loc, electron_df, phonon_df, nthreads)
    if calc_rta_mobility:
        rta_tdf_mobility(in_loc, electron_df)
    if build_scattering_matrix:
        scattering_matrix(in_loc, electron_df, nthreads, pp.simpleBool)
        matrix_check_colsum(in_loc, electron_df)
