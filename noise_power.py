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


def steady_state_solns(kindices, numkpts, fullkpts_df, field):
    """Get steady state solutions"""
    e = 1.602 * 10 ** (-19)  # fundamental electronic charge [C]
    kb = 1.38064852 * 10 ** (-23)  # Boltzmann constant in SI [m^2 kg s^-2 K^-1]

    # Calculate fermi distribution function for all the kpoints. Currently only in chunked dataframes.
    # Want them all in one place here
    def fermi_distribution(df, fermilevel=con.mu, temp=con.T):
        df['k_FD'] = (np.exp((df['energy'].values * e - fermilevel * e) / (kb * temp)) + 1) ** (-1)
        return df

    kptdata = fullkpts_df[['k_inds', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]', 'energy', 'vx [m/s]']]
    fermi_distribution(kptdata)

    # Create a lookup table (whichintegrands) where the index corresponds to the kpoint, and the entry is the row number
    # in the (integrands) array, so for given a kpoint we can easily find its corresponding integrand, since there are
    # fewer unique integrands than there kpoints.
    uniq_yz = np.unique(kptdata[['ky [1/A]', 'kz [1/A]']].values, axis=0)
    uniq_x = np.sort(np.unique(kptdata[['kx [1/A]']]))
    integrands = np.zeros((uniq_yz.shape[0], len(uniq_x)))
    whichintegrands = np.zeros(numkpts)

    prefactor = e * field / kb / con.T
    loopstart = time.time()
    # ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
    # whichkpts = np.array(np.logical_and(kptdata[['ky [1/A]']] == ky, kptdata[['kz [1/A]']] == kz))
    # subset = kptdata[whichkpts].sort_values(by=['kx [1/A]'])
    # kx_pts = np.isin(uniq_x, subset[['kx [1/A]']].values)
    # integrands[i, kx_pts] = thisintegrand[:, 0]
    # whichintegrands[kptdata[['k_inds']][whichkpts] - 1] = i)

    b = prefactor * kptdata[['vx [m/s]']].values * kptdata[['k_FD']].values * \
        (1 - kptdata['k_FD'])[:, np.newaxis]

    start = time.time()
    x = np.linalg.solve(matrix, b)
    end = time.time()
    print('Solve successful. Took {:.2f}s'.format(end-start))


if __name__ == '__main__':
    data_loc = '/home/peishi/nvme/k200-0.4eV/'
    chunk_loc = '/home/peishi/nvme/k200-0.4eV/chunked/'

    con = preprocessing_largegrid.PhysicalConstants()

    _, kpts_df, enk_df, qpts_df, enq_df = preprocessing_largegrid.loadfromfile(data_loc, matrixel=False)
    cartkpts = preprocessing_largegrid.load_vel_data(data_loc, con)

    nkpts = len(np.unique(kpts_df['k_inds']))
    n_ph_modes = len(np.unique(enq_df['q_inds'])) * len(np.unique(enq_df['im_mode']))
    kinds = np.arange(1, nkpts + 1)

    # steady_state_solns(kinds, nkpts, cartkpts, 1)


