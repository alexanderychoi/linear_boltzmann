#!/usr/bin/env python

import preprocessing_largegrid
import numpy as np
import multiprocessing as mp
from functools import partial
import os
import pandas as pd
import time

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

    sr = np.sum(g_df['weight'].to_numpy()) * 2 * np.pi / (6.582119 * 10 ** -16) * (10 ** -12) / nlambda

    print(r'For k={:d}, the scattering rate (1/ps) is {:f}'.format(k, sr))

    scattering_rates[k-1] = sr

    # scattering = sr.to_frame().reset_index()
    # scattering_array = np.zeros(nkpts)
    # scattering_array[scattering['k_inds'].values-1] = scattering['weight'].values
    #
    # offdiag_abs_weight = np.multiply(np.multiply(g_df['BE'].values + 1 - g_df['k_FD'].values,
    #                                                 g_df['abs_gaussian'].values), g_df['g_element']) / 13.6056980659
    # offdiag_ems_weight = np.multiply(np.multiply(g_df['BE'].values + g_df['k_FD'].values,
    #                                                 g_df['ems_gaussian'].values), ['g_element']) / 13.6056980659
    #
    # g_df['OD_weight'] = offdiag_abs_weight + offdiag_ems_weight
    # offdiag_sr = g_df.groupby(['k_inds'])['OD_weight'].agg('sum') * 2 * np.pi * (2.418 * 10 ** 17) * (10 ** -12) / len(
    #     np.unique(g_df['q_id'].values))
    #
    # OD_scattering = offdiag_sr.to_frame().reset_index()
    # OD_scattering_array = np.zeros(len(np.unique(cart_kpts_df['k_inds'])))
    # OD_scattering_array[OD_scattering['k_inds'].values-1] = OD_scattering['weight'].values


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

    # g_df['OD_abs_weight'] = np.multiply(np.multiply(g_df['BE'].values + 1 - g_df['k_FD'].values,
    #                                                 g_df['abs_gaussian'].values), g_df['g_element']) / 13.6056980659
    # g_df['OD_ems_weight'] = np.multiply(np.multiply(g_df['BE'].values + g_df['k_FD'].values,
    #                                                 g_df['ems_gaussian'].values), ['g_element']) / 13.6056980659
    #
    # g_df['OD_weight'] = g_df['OD_ems_weight'].values + g_df['OD_abs_weight'].values
    # OD_sr = g_df.groupby(['k_inds'])['OD_weight'].agg('sum') * 2 * np.pi * 2.418 * 10 ** (17) * 10 ** (-12) / len(
    #     np.unique(g_df['q_id'].values))
    #
    # OD_scattering = OD_sr.to_frame().reset_index()
    # OD_scattering_array = np.zeros(len(np.unique(cart_kpts_df['k_inds'])))
    # OD_scattering_array[OD_scattering['k_inds'].values-1] = OD_scattering['weight'].values

    return scattering_array


if __name__ == '__main__':
    con = preprocessing_largegrid.PhysicalConstants()

    # data_loc = '/home/peishi/nvme/k100-0.3eV/'
    # chunk_loc = '/home/peishi/nvme/k100-0.3eV/chunked/'
    data_loc = '/home/peishi/nvme/k200-0.4eV/'
    chunk_loc = '/home/peishi/nvme/k200-0.4eV/chunked/'

    _, kpts_df, enk_df, qpts_df, enq_df = preprocessing_largegrid.loadfromfile(data_loc, matrixel=False)

    nkpts = len(np.unique(kpts_df['k_inds']))
    n_ph_modes = len(np.unique(enq_df['q_inds'])) * len(np.unique(enq_df['im_mode']))
    kinds = np.arange(1, nkpts + 1)

    os.chdir(chunk_loc)
    scattering_rates = mp.Array('d', [0] * nkpts, lock=False)
    nthreads = 6
    pool = mp.Pool(nthreads)

    start = time.time()
    pool.map(partial(relaxation_times_parallel, nlambda=n_ph_modes), kinds)
    end = time.time()
    print('Parallel relaxation time calc took {:.2f} seconds'.format(end - start))
    scattering_rates = np.array(scattering_rates)

    # # Scattering rate calculation when you have the whole dataframe
    # full_g_df = pd.read_hdf('full_g_df.h5', key='df')
    # kpts = preprocessing_largegrid.load_vel_data(data_loc, con)
    #
    # scattering_rates = relaxation_times(full_g_df, kpts)

    np.save(data_loc + 'scattering_rates', scattering_rates)
