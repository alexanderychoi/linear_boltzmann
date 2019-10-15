#!/usr/bin/env python

import preprocessing_largegrid
import numpy as np
import multiprocessing as mp
import matplotlib as mpl
from functools import partial
import os
import pandas as pd
import time
import numba
import re

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import plotly.offline as py
import plotly.graph_objs as go
import plotly


def coupling_matrix_calc(g_df):
    """
    This function takes a list of k-point indices and returns the Fermi-distributions and energies associated with each k-point on that list. The Fermi distributions are calculated with respect to a particular chemical potential.
    Parameters:
    -----------

    abs_g_df : pandas dataframe containing:

        k_inds : vector_like, shape (n,1)
        Index of k point (pre-collision)

        q_inds : vector_like, shape (n,1)
        Index of q point

        k+q_inds : vector_like, shape (n,1)
        Index of k point (post-collision)

        m_band : vector_like, shape (n,1)
        Band index of post-collision state

        n_band : vector_like, shape (n,1)
        Band index of pre-collision state

        im_mode : vector_like, shape (n,1)
        Polarization of phonon mode

        g_element : vector_like, shape (n,1)
        E-ph matrix element

        k_fermi_dist : vector_like, shape (n,1)
        Fermi distribution of pre collision state

        k+q_fermi_dist : vector_like, shape (n,1)
        Fermi distribution of post collision state

        k_energy : vector_like, shape (n,1)
        Energy of the pre collision state

        k+q_energy : vector_like, shape (n,1)
        Energy of the post collision state


    T : scalar
    Lattice temperature in Kelvin

    Returns:
    --------

    """
    # Physical constants
    e = 1.602 * 10 ** (-19)  # fundamental electronic charge [C]
    kb = 1.38064852 * 10 ** (-23);  # Boltzmann constant in SI [m^2 kg s^-2 K^-1]
    h = 1.0545718 * 10 ** (-34)

    g_df_ems = g_df.loc[(g_df['collision_state'] == -1)].copy(deep=True)
    g_df_abs = g_df.loc[(g_df['collision_state'] == 1)].copy(deep=True)

    g_df_ems['weight'] = np.multiply(
        np.multiply((g_df_ems['BE'].values + 1 - g_df_ems['k+q_FD'].values), g_df_ems['g_element'].values),
        g_df_ems['gaussian']) / 13.6056980659
    g_df_abs['weight'] = np.multiply(
        np.multiply((g_df_abs['BE'].values + g_df_abs['k+q_FD'].values), g_df_abs['g_element'].values),
        g_df_abs['gaussian']) / 13.6056980659

    abs_sr = g_df_abs.groupby(['k_inds', 'k+q_inds'])['weight'].agg('sum')
    summed_abs_df = abs_sr.to_frame().reset_index()

    ems_sr = g_df_ems.groupby(['k_inds', 'k+q_inds'])['weight'].agg('sum')
    summed_ems_df = ems_sr.to_frame().reset_index()

    return summed_abs_df, summed_ems_df


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def relaxation_times_parallel(k, nlambda):
    """This function calculates the on-diagonal scattering rates, the relaxation times, as per Mahan's Eqn. 11.127.
    Also returns the off-diagonal scattering term.

    CHECK THE FORMULAS FROM MAHAN"""
    mbands = 6
    # divisor = 121904  # number of unique q_ids for k100
    # divisor = 3043002  # number of unique q_ids for k200
    prefactor = 13.6056980659

    g_df = pd.read_parquet('k{:05d}.parquet'.format(k))

    ems_weight = np.multiply(np.multiply(g_df['BE'].values + 1 - g_df['k+q_FD'].values, g_df['g_element'].values),
                             g_df['ems_gaussian'])
    abs_weight = np.multiply(np.multiply((g_df['BE'].values + g_df['k+q_FD'].values), g_df['g_element'].values),
                             g_df['abs_gaussian'])

    g_df['weight'] = ems_weight + abs_weight

    sr = np.sum(g_df['weight'].to_numpy()) * 2 * np.pi / (6.582119 * 10 ** -16) * (10 ** -12) / nlambda * prefactor**2

    print(r'For k={:d}, the scattering rate (1/ps) is {:.14E}'.format(k, sr))

    # scattering_rates[k-1] = sr


def parse_scatteringrates():
    f = open('rates.out')
    alltext = f.read()

    matches = re.findall(r'For k=(\d+), the scattering rate \(1\/ps\) is (\d+\.\d+E.\d+)', alltext)

    rates = np.zeros(nkpts)
    for match in matches:
        thisk = np.int(match[0])
        thisrate = np.float(match[1])
        rates[thisk-1] = thisrate

    return rates


def relaxation_times(g_df, cart_kpts_df):
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

    g_df['OD_abs_weight'] = np.multiply(np.multiply(g_df['BE'].values + 1 - g_df['k_FD'].values,
                                                    g_df['abs_gaussian'].values), g_df['g_element']) / 13.6056980659
    g_df['OD_ems_weight'] = np.multiply(np.multiply(g_df['BE'].values + g_df['k_FD'].values,
                                                    g_df['ems_gaussian'].values), ['g_element']) / 13.6056980659

    g_df['OD_weight'] = g_df['OD_ems_weight'].values + g_df['OD_abs_weight'].values
    OD_sr = g_df.groupby(['k_inds'])['OD_weight'].agg('sum') * 2 * np.pi * 2.418 * 10 ** (17) * 10 ** (-12) / len(
        np.unique(g_df['q_id'].values))

    OD_scattering = OD_sr.to_frame().reset_index()
    OD_scattering_array = np.zeros(len(np.unique(cart_kpts_df['k_inds'])))
    OD_scattering_array[OD_scattering['k_inds'].values-1] = OD_scattering['weight'].values

    return scattering_array


def rta_mobility(datadir, enk, vels):
    """Calculate mobility using RTA and near equilibrium approximation"""
    os.chdir(datadir)
    rates = np.load('scattering_rates.npy') * 36.5  # arbitrary factor to test if I can get the mobility
    taus = 1 / rates * 1E-12  # in seconds
        
    npts = 4000  # number of points in the KDE
    ssigma = np.zeros(npts)  # capital sigma as defined in Jin Jian's paper Eqn 3
    # Need to define the energy range that I'm doing integration over
    # en_axis = np.linspace(enk.min(), enk.min() + 0.4, npts)
    en_axis = np.linspace(enk.min(), enk.max(), npts)
    dx = (en_axis.max() - en_axis.min()) / npts

    vuc = 1E-30 * np.dot(np.cross(con.a1, con.a2), con.a3)  # volume of the unit cell in m^3
    
    e = 1.602 * 10 ** (-19)  # fundamental electronic charge [C]
    # kb = 1.38064852 * 10 ** (-23)  # Boltzmann constant in SI [m^2 kg s^-2 K^-1]
    kb = 8.617333 * (10**-5)  # Boltzmann constant in eV / K
    temperature = 300  # Kelvin
    beta = 1 / (kb * temperature)
    fermi = con.mu  # Fermi level
    # fermi = con.mu + 0.3  # Fermi level
    # fermi = enk.min() - (1.424 / 2)  # Fermi level
    spread = 100*dx

    def dfde(x):
        # Approximate dF/dE using dF_0/dE
        return (-1 * beta) * np.exp((x-fermi) * beta) * (np.exp((x-fermi) * beta) + 1)**(-2)

    def gaussian(x, mu, sigma=spread):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)

    # Calculating carrier concentration using Eqn 22 in W. Li PRB 92 (2015)
    nc = 2E18  # the conc (m^-3) that makes the calculation work for a midgap fermi level
    nc = (2 / len(enk) / vuc) * np.sum((np.exp((enk - fermi) * beta) + 1) ** -1)
    print('The carrier concentration is {:.3E} in m^-3'.format(nc))

    for k in range(len(enk)):
        istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4*spread/dx), 0))
        iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4*spread/dx), npts - 1))
        ssigma[istart:iend] += vels[k]**2 * taus[k] * gaussian(en_axis[istart:iend], enk[k])

    ssigma = ssigma * 2 / vuc
    conductivity = (e**2) / e * np.trapz(np.multiply(ssigma, -1 * dfde(en_axis)), en_axis)  # divided by e for eV to J
    print('Conductivity is {:.3E}'.format(conductivity))
    print('dF/dE is {:.3E}'.format(np.sum(-1 * dfde(en_axis))))
    mobility = conductivity / nc / e * 1E4 / len(enk)  # 1E4 to get from m^2 to cm^2
    print('Mobility is {:.3E}'.format(mobility))

    font = {'size': 14}
    mpl.rc('font', **font)
    plt.plot(en_axis, np.multiply(ssigma, -1 * dfde(en_axis)), '.')
    plt.xlabel('Energy (eV)')
    plt.ylabel('TDF * dF/dE (a.u.)')
    plt.show()

    return mobility


def assemble_full_matrix():
    """Once all of the rows have been created, can put them all together."""
    if os.path.isfile('scattering_matrix.mmap'):
        matrix = np.memmap('scattering_matrix.mmap', dtype='float64', mode='r+', shape=(nkpts, nkpts))
    else:
        matrix = np.memmap('scattering_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))


# @numba.jit(parallel=True, nopython=True)
def numba_matrixrows(nk, nlambda, kra):
    thisrow = np.zeros(nkpts)
    for i in numba.prange(len(nkprime)):
        kpi = int(nkprime[i])
        kpinds = kra[:, 1].astype(int)
        kp_rows = kra[kpinds == kpi, :]
        abs_weight = np.multiply(np.multiply((kp_rows[:, 6] + 1 - kp_rows[:, 9]),
                                             kp_rows[:, 3]), kp_rows[:, -2])
        ems_weight = np.multiply(np.multiply(kp_rows[:, 6] + kp_rows[:, 9],
                                             kp_rows[:, 3]), kp_rows[:, -1])
        tot_weight = abs_weight + ems_weight
        thisrow[kpi - 1] = np.sum(tot_weight) * 2 * np.pi / (6.582119 * 10 ** -16) * (10 ** -12) / nlambda

    return thisrow


def matrixrows_par(k, nlambda):
    """Calculate the full scattering matrix row by row and in parallel
    
    We do this because the data is chunked by kpoint, so the most efficient way to do this is by calculating each row
    of the scattering matrix since the information required for each row is contained in each kpoint chunk. This allows 
    the calculation to be done in parallel. For each kpoint, will input the row data into a memmap. Afterwards, the 
    full matrix can be assembled quickly in serial.
    """
    # Create memmap of the row
    istart = time.time()
    if os.path.isfile('k{:05d}.mmap'.format(k)):
        krow = np.memmap('k{:05d}.mmap'.format(k), dtype='float64', mode='r+', shape=nkpts)
    else:
        krow = np.memmap('k{:05d}.mmap'.format(k), dtype='float64', mode='w+', shape=nkpts)

    # Diagonal term
    krow[k-1] = scattering_rates[k-1]

    # Calculating nondiagonal terms
    g_df = pd.read_parquet(chunk_loc + 'k{:05d}.parquet'.format(k))
    nkprime = np.unique(g_df['k+q_inds'])
    for kp in np.nditer(nkprime):
        kpi = int(kp)
        kp_rows = g_df[g_df['k+q_inds'] == kpi]
        abs_weight = np.multiply(np.multiply((kp_rows['BE'].values + 1 - kp_rows['k_FD'].values),
                                             kp_rows['g_element'].values), kp_rows['abs_gaussian'])
        ems_weight = np.multiply(np.multiply(kp_rows['BE'].values + kp_rows['k_FD'].values,
                                             kp_rows['g_element'].values), kp_rows['ems_gaussian'])
        tot_weight = abs_weight + ems_weight
        krow[kpi - 1] = np.sum(tot_weight) * 2 * np.pi / (6.582119 * 10 ** -16) * (10 ** -12) / nlambda

    del krow
    iend = time.time()
    print('Row calc for k={:d} took {:.2f} seconds'.format(k, iend - istart))


if __name__ == '__main__':
    con = preprocessing_largegrid.PhysicalConstants()

    # data_loc = '/home/peishi/nvme/k100-0.3eV/'
    # chunk_loc = '/home/peishi/nvme/k100-0.3eV/chunked/'
    data_loc = '/home/peishi/nvme/k200-0.4eV/'
    chunk_loc = '/home/peishi/nvme/k200-0.4eV/chunked/'

    # mp.set_start_method('spawn')

    _, kpts_df, enk_df, qpts_df, enq_df = preprocessing_largegrid.loadfromfile(data_loc, matrixel=False)
    cartkpts = preprocessing_largegrid.load_vel_data(data_loc, con)

    k_en = (cartkpts.sort_values(by=['k_inds']))['energy'].values
    kvel = cartkpts[['k_inds', 'vx [m/s]']]
    kvel = (kvel.sort_values(by=['k_inds']))['vx [m/s]'].values

    nkpts = len(np.unique(kpts_df['k_inds']))
    n_ph_modes = len(np.unique(enq_df['q_inds'])) * len(np.unique(enq_df['im_mode']))
    kinds = np.arange(1, nkpts + 1)

    calc_scattering_rates = False
    if calc_scattering_rates:
        os.chdir(chunk_loc)
        # scattering_rates = mp.Array('d', [0] * nkpts, lock=False)
        nthreads = 6
        pool = mp.Pool(nthreads)

        start = time.time()
        pool.map(partial(relaxation_times_parallel, nlambda=n_ph_modes), kinds)
        end = time.time()
        print('Parallel relaxation time calc took {:.2f} seconds'.format(end - start))
        # scattering_rates = np.array(scattering_rates)

        # # Scattering rate calculation when you have the whole dataframe
        # full_g_df = pd.read_hdf('full_g_df.h5', key='df')
        # kpts = preprocessing_largegrid.load_vel_data(data_loc, con)
        # scattering_rates = relaxation_times(full_g_df, kpts)

    sr = parse_scatteringrates()
    np.save(data_loc + 'scattering_rates_test', sr)

    calc_matrix_rows = False
    if calc_matrix_rows:
        os.chdir(data_loc)
        rta_rates = np.load('scattering_rates.npy')

        # Multiprocessing version
        # Need to create a directory called mat_rows inside the data_loc directory to store rows
        os.chdir(data_loc + 'mat_rows')
        scattering_rates = mp.Array('d', rta_rates, lock=False)
        nthreads = 6
        pool = mp.Pool(nthreads)

        start = time.time()
        pool.map(partial(matrixrows_par, nlambda=n_ph_modes), np.arange(1, 20))
        end = time.time()
        print('Calc of scattering matrix rows took {:.2f} seconds'.format(end - start))

        # # Numba version
        # os.chdir(data_loc + 'mat_rows')
        # for k in range(nkpts):
        #     kind = k+1
        #     istart = time.time()
        #     krow = np.memmap('k{:05d}.mmap'.format(kind), dtype='float64', mode='w+', shape=nkpts)
        #     kind = k + 1
        #     g_df = pd.read_parquet(chunk_loc + 'k{:05d}.parquet'.format(kind))
        #     nkprime = np.unique(g_df['k+q_inds'])
        #     kra = g_df.values
        #     krow = numba_matrixrows(nkpts, n_ph_modes, kra)
        #     del krow
        #     iend = time.time()
        #     print('Row calc for k={:d} took {:.2f} seconds'.format(k, iend - istart))

    # rta_mobility(data_loc, k_en, kvel)


