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


#def steady_state_solns(kindices, numkpts, fullkpts_df, field):
#    """Get steady state solutions"""
#    e = 1.602 * 10 ** (-19)  # fundamental electronic charge [C]
#    kb = 1.38064852 * 10 ** (-23)  # Boltzmann constant in SI [m^2 kg s^-2 K^-1]
#
#    # Calculate fermi distribution function for all the kpoints. Currently only in chunked dataframes.
#    # Want them all in one place here
#    def fermi_distribution(df, fermilevel=con.mu, temp=con.T):
#        df['k_FD'] = (np.exp((df['energy'].values * e - fermilevel * e) / (kb * temp)) + 1) ** (-1)
#        return df
#
#    kptdata = fullkpts_df[['k_inds', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]', 'energy', 'vx [m/s]']]
#    fermi_distribution(kptdata)
#
#    # Create a lookup table (whichintegrands) where the index corresponds to the kpoint, and the entry is the row number
#    # in the (integrands) array, so for given a kpoint we can easily find its corresponding integrand, since there are
#    # fewer unique integrands than there kpoints.
#    uniq_yz = np.unique(kptdata[['ky [1/A]', 'kz [1/A]']].values, axis=0)
#    uniq_x = np.sort(np.unique(kptdata[['kx [1/A]']]))
#    integrands = np.zeros((uniq_yz.shape[0], len(uniq_x)))
#    whichintegrands = np.zeros(numkpts)
#
#    # Calculate the unique integrands along kx
#    if os.path.isfile(data_loc + 'integrands.npy'):
#        integrands = np.load('integrands.npy')
#        whichintegrands = np.load('whichintegrands.npy')
#    else:
#        prefactor = e * field / kb / con.T
#        loopstart = time.time()
#        for i in range(uniq_yz.shape[0]):
#            start = time.time()
#            ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
#            whichkpts = np.array(np.logical_and(kptdata[['ky [1/A]']] == ky, kptdata[['kz [1/A]']] == kz))
#            subset = kptdata[whichkpts].sort_values(by=['kx [1/A]'])
#            thisintegrand = prefactor * subset[['vx [m/s]']].values * subset[['k_FD']].values * (1 - subset['k_FD'])[:, np.newaxis]
#            kx_pts = np.isin(uniq_x, subset[['kx [1/A]']].values)
#            integrands[i, kx_pts] = thisintegrand[:, 0]
#            whichintegrands[kptdata[['k_inds']][whichkpts] - 1] = i
#            end = time.time()
#            print('Done with the i={:d} unique integrand in {:.2f}s. There were {:d} kx entries'
#                  .format(i+1, end - start, np.count_nonzero(kx_pts)))
#        loopend = time.time()
#        print('Done creating integrands in {:.2f}s'.format(loopend - loopstart))
#        np.save(data_loc + 'integrands', integrands)
#        np.save(data_loc + 'whichintegrands', whichintegrands)
#
#    # Now integrate after multiplying integrands by the exponentiated eigenvalue for each kpoint
#    ptransp = np.memmap(data_loc + 'p_transpose.mmap', dtype='float64', mode='r', shape=(numkpts, numkpts))
#    eigvals = np.load(data_loc + 'eigenvalues.npy')
#    fullintegralvec = np.zeros(numkpts)
#    loopstart = time.time()
#    for k in range(numkpts):
#        start = time.time()
#        x = np.where(uniq_x == kptdata[['kx [1/A]']].values[k])[0][0]
#        expvec = np.exp(-1 * uniq_x[:x+1] * eigvals[k])
#        expmat = np.tile(expvec, (integrands.shape[0], 1))
#        k_integrands = np.multiply(integrands[:, :x+1], expmat)
#        integralvals = np.trapz(k_integrands, axis=1)
#        # The line below is where all the magic happens. Constructs a vector which is the value of all of the integrals
#        # for this particular kpoint (particular kx) by selecting the right integral value using (whichintegrands)
#        # With this vector, the result of the linear combination is simply a dot product with the row of P_transpose
#        k_intg_vec = np.array([integralvals[int(whichintegrands[kprime])] for kprime in range(numkpts)])
#        fullintegralvec[k] = np.dot(ptransp[k, :], k_intg_vec)
#        end = time.time()
#        print('Done with k={:d} in {:.3f}s'.format(k+1, end-start))
#    loopend = time.time()
#    print('Evaluated all linear combination of integrals multiplied by exponential for all kpoints in {:2f}s'
#          .format(loopend - loopstart))
def calc_sparsity():
    matrix = np.memmap(data_loc + 'scattering_matrix.mmap', dtype='float64', mode='r+', shape=(nkpts, nkpts))
    sparsity = 1 - (np.count_nonzero(matrix) / nkpts**2)
    nelperrow = np.zeros(nkpts)
    for ik in range(nkpts):
        nelperrow[ik] = np.count_nonzero(matrix[ik, :])
        print('For row {:d}, the number of nozero elements is {:f}'.format(ik+1, nelperrow[ik]))
    return sparsity, nelperrow


def centraldiff_matrix(fullkpts_df,step_size):
    # Do not  flush the memmap it will overwrite consecutively. 
    matrix = np.memmap(data_loc + 'scattering_matrix.mmap', dtype='float64', mode='r+', shape=(nkpts, nkpts))
    # Get the first and last rows since these are different because of the IC. Go through each.

    # Get the unique ky and kz values from the array for looping.
    kptdata = fullkpts_df[['k_inds', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]']]
    uniq_yz = np.unique(kptdata[['ky [1/A]', 'kz [1/A]']].values, axis=0)
    start = time.time()

    # If there are too few points in a slice < 5, we want to keep track of those points
    shortslice_inds = []
    lastslice_inds =[]

    # Loop through the unique ky and kz values
    for i in range(len(uniq_yz)):
        kind = i + 1
        ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]

        # Grab the "slice" of points in k space with the same ky and kz coordinate
        slice_df = kptdata.loc[(kptdata['ky [1/A]'] == ky) & (kptdata['kz [1/A]'] == kz)]
        slice_inds = slice_df['k_inds'].values

        # if 0 in slice_inds or 1 in slice_inds or len(kptdata) in slice_inds or len(kptdata)-1 in slice_inds:
        #     lastslice_inds.append(slice_inds)
        #     continue

        if len(slice_inds) > 4:
            # Subset is the slice sorted by kx value in ascending order. The index of subset still references kptdata.
            subset = slice_df.sort_values(by=['kx [1/A]'],ascending=True)
            ordered_slice_inds = subset.index

            # Set the "initial condition" i.e. the point with the most negative kx value is treated as being zero
            # (and virtual point below)
            matrix[ordered_slice_inds[0], ordered_slice_inds[1]] = matrix[ordered_slice_inds[0],ordered_slice_inds[1]] - \
                                                                  1/(2*step_size)
            matrix[ordered_slice_inds[1], ordered_slice_inds[2]] = matrix[ordered_slice_inds[1],ordered_slice_inds[2]] - \
                                                                  1/(2*step_size)

            # Set the other "boundary condition" i.e. the point with the most positive kx value is treated as being zero
            # (and virtual point above)
            last = len(ordered_slice_inds) - 1
            slast = len(ordered_slice_inds) - 2
            matrix[ordered_slice_inds[last], ordered_slice_inds[slast]] = \
                matrix[ordered_slice_inds[last], ordered_slice_inds[slast]] + 1/(2*step_size)
            matrix[ordered_slice_inds[slast], ordered_slice_inds[slast-1]] = \
                matrix[ordered_slice_inds[slast], ordered_slice_inds[slast-1]] + 1/(2*step_size)

            # Set the value of all other points in the slice
            inter_inds = ordered_slice_inds[2:slast]
            inter_inds_up = ordered_slice_inds[3:last]
            inter_inds_down = ordered_slice_inds[1:slast-1]

            # print('Ordered_slice_inds')
            # print(*ordered_slice_inds)
            # print('inter_inds')
            # print(*inter_inds)
            # print('inter_inds_up')
            # print(*inter_inds_up)
            # print('inter_inds_down')
            # print(*inter_inds_down)

            matrix[inter_inds, inter_inds_up] = matrix[inter_inds, inter_inds_up] - 1/(2*step_size)
            matrix[inter_inds, inter_inds_down] = matrix[inter_inds, inter_inds_down] + 1/(2*step_size)

        else:
            shortslice_inds.append(slice_inds)

        end = time.time()
        if kind % 10 == 0:
            print('Finished {:d} out of {:d} slices in {:.3f}s'.format(kind, len(uniq_yz),end-start))
    print('Scattering matrix modified to incorporate central difference contribution.')
    print('Not applied to {:d} points because fewer than 5 points on the slice.'.format(len(shortslice_inds)))
    return shortslice_inds


if __name__ == '__main__':
    # data_loc = '/home/peishi/nvme/k200-0.4eV/'
    # chunk_loc = '/home/peishi/nvme/k200-0.4eV/chunked/'

    data_loc = 'D:/Users/AlexanderChoi/GaAs_300K_10_19/k200-0.4eV/'
    chunk_loc = 'D:/Users/AlexanderChoi/GaAs_300K_10_19/chunked/'

    con = preprocessing_largegrid.PhysicalConstants()

    _, kpts_df, enk_df, qpts_df, enq_df = preprocessing_largegrid.loadfromfile(data_loc, matrixel=False)
    cartkpts = preprocessing_largegrid.load_vel_data(data_loc, con)

    nkpts = len(np.unique(kpts_df['k_inds']))
    n_ph_modes = len(np.unique(enq_df['q_inds'])) * len(np.unique(enq_df['im_mode']))
    kinds = np.arange(1, nkpts + 1)
    print('bumdiddly')


    # matrix = np.memmap(data_loc + 'scattering_matrix.mmap', dtype='float64', mode='r+', shape=(nkpts, nkpts))

    centraldiff_matrix(cartkpts,1)

    # steady_state_solns(kinds, nkpts, cartkpts, 1)
