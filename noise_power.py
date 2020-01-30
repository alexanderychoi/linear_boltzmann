import preprocessing_largegrid
import plotting
import numpy as np
import scipy.sparse.linalg
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib
from functools import partial
import os
import pandas as pd
import time
import numba
import re


def fermi_distribution(df, fermilevel=6.03, temp=300):
    e = 1.602 * 10 ** (-19)  # fundamental electronic charge [C]
    kb = 1.38064852 * 10 ** (-23)  # Boltzmann constant in SI [m^2 kg s^-2 K^-1]
    df['k_FD'] = (np.exp((df['energy'].values * e - fermilevel * e) / (kb * temp)) + 1) ** (-1)
    return df


def steady_state_solns(matrix, numkpts, fullkpts_df, field):
    """Get steady state solutions"""
    e = 1.602 * 10 ** (-19)  # fundamental electronic charge [C]
    kb = 1.38064852 * 10 ** (-23)  # Boltzmann constant in SI [m^2 kg s^-2 K^-1]

    # Calculate fermi distribution function for all the kpoints. Currently only in chunked dataframes.
    # Want them all in one place here


    kptdata = fullkpts_df[['k_inds', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]', 'energy', 'vx [m/s]']]
    fermi_distribution(kptdata)

    prefactor = e * field / kb / con.T

    b = prefactor * kptdata[['vx [m/s]']].values * kptdata[['k_FD']].values * \
        (1 - kptdata['k_FD'])[:, np.newaxis]

    start = time.time()
    x = np.linalg.solve(matrix, b)
    end = time.time()
    print('Direct inversion solve successful. Took {:.2f}s'.format(end - start))
    start = time.time()
    x_star = np.linalg.lstsq(matrix, b)
    end = time.time()
    print('Least squares solve successful. Took {:.2f}s'.format(end - start))

    return x, x_star


def calc_sparsity():
    matrix = np.memmap(data_loc + 'scattering_matrix.mmap', dtype='float64', mode='r+', shape=(nkpts, nkpts))
    sparsity = 1 - (np.count_nonzero(matrix) / nkpts**2)
    nelperrow = np.zeros(nkpts)
    for ik in range(nkpts):
        nelperrow[ik] = np.count_nonzero(matrix[ik, :])
        print('For row {:d}, the number of nozero elements is {:f}'.format(ik+1, nelperrow[ik]))
    return sparsity, nelperrow


def apply_centraldiff_matrix(matrix, fullkpts_df, E, cons, step_size=1):
    # Do not  flush the memmap it will overwrite consecutively.

    # Get the first and last rows since these are different because of the IC. Go through each.

    # Get the unique ky and kz values from the array for looping.
    kptdata = fullkpts_df[['k_inds', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]']]
    kptdata['kpt_mag'] = np.sqrt(kptdata['kx [1/A]'].values**2 + kptdta['ky [1/A]'].values**2 +
                                 kptdata['kz [1/A]'].values**2)
    kptdata.loc['valley',kptdata['kpt_mag'] < 0.3] = 1  # corresponds to Gamma valley
    kptdata.loc['valley',kptdata['kpt_mag'] > 0.3] = 0  # corresponds to L valley

    uniq_yz = np.unique(kptdata[['ky [1/A]', 'kz [1/A]']].values, axis=0)

    # If there are too few points in a slice < 5, we want to keep track of those points
    shortslice_inds = []
    icinds = []
    lvalley_inds = []

    start = time.time()
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

        # Skip all slices that intersect an L valley. Save the L valley indices
        if np.any(slice_df['valley'] == 0):
            lvalley_inds.append(slice_inds)
            continue

        if len(slice_inds) > 4:
            # Subset is the slice sorted by kx value in ascending order. The index of subset still references kptdata.
            subset = slice_df.sort_values(by=['kx [1/A]'], ascending=True)
            ordered_inds = subset['k_inds'].values - 1  # indices of matrix
            icinds.append(ordered_inds[0] + 1)  # +1 to get the k_inds values

            # Set the "initial condition" i.e. the point with the most negative kx value is treated as being zero
            # (and virtual point below)
            matrix[ordered_inds[0], ordered_inds[1]] += - 1/(2*step_size)*cons.e*E/cons.h
            matrix[ordered_inds[1], ordered_inds[2]] += - 1/(2*step_size)*cons.e*E/cons.h

            # Set the other "boundary condition" i.e. the point with the most positive kx value is treated as being zero
            # (and virtual point above)
            last = len(ordered_inds) - 1
            slast = len(ordered_inds) - 2
            matrix[ordered_inds[last], ordered_inds[slast]] += 1/(2*step_size)*cons.e*E/cons.h
            matrix[ordered_inds[slast], ordered_inds[slast-1]] += 1/(2*step_size)*cons.e*E/cons.h

            # Set the value of all other points in the slice
            inter_inds = ordered_inds[2:slast]
            inter_inds_up = ordered_inds[3:last]
            inter_inds_down = ordered_inds[1:slast-1]

            matrix[inter_inds, inter_inds_up] = matrix[inter_inds, inter_inds_up] - 1/(2*step_size)*cons.e*E/cons.h
            matrix[inter_inds, inter_inds_down] = matrix[inter_inds, inter_inds_down] + 1/(2*step_size)*cons.e*E/cons.h

        else:
            shortslice_inds.append(slice_inds)

        end = time.time()
        if kind % 10 == 0:
            print('Finished {:d} out of {:d} slices in {:.3f}s'.format(kind, len(uniq_yz),end-start))
    print('Scattering matrix modified to incorporate central difference contribution.')
    print('Not applied to {:d} points because fewer than 5 points on the slice.'.format(len(shortslice_inds)))
    print('Finite difference not applied to L valleys. Derivative treated as zero for these points.')
    return shortslice_inds, icinds, lvalley_inds, matrix


def iterative_solver(kptdf, matrix):
    # sr = np.load(data_loc + 'scattering_rates_direct.npy')
    # tau = 1 / sr
    prefactor = (np.diag(matrix))**(-1)
    f_0 = np.squeeze(kptdf['vx [m/s]'] * kptdf['k_FD']) * (1 - kptdf['k_FD']) * prefactor
    # f_0 = kptdf['vx [m/s]'] * tau
    # matrix -= np.diag(np.diag(matrix))

    errpercent = 1
    counter = 0
    f_prev = f_0
    while errpercent > 1E-6 and counter < 20:
        s1 = time.time()
        mvp = np.matmul(matrix, f_prev)
        e1 = time.time()
        print('Matrix vector multiplication took {:.2f}s'.format(e1-s1))
        # Remove the part of the vector from the diagonal multiplication
        offdiagsum = mvp - (np.diag(matrix) * f_prev)
        f_next = f_0 - (prefactor * offdiagsum)
        errvecnorm = np.linalg.norm(f_next - f_prev)
        errpercent = errvecnorm / np.linalg.norm(f_prev)
        f_prev = f_next
        counter += 1
        print('Iteration {:d}: Error percent is {:.3E}. Abs error norm is {:.3E}'
              .format(counter, errpercent, errvecnorm))
    return f_next, f_0


def conj_grad_soln(kptdf, matrix):
    b = np.squeeze(kptdf[['vx [m/s]']].values * kptdf[['k_FD']].values) * (1 - kptdf['k_FD'])

    ts = time.time()
    x, status = scipy.sparse.linalg.cg(matrix, b)
    te = time.time()
    print('Conjugate gradient solve successful. Took {:.2f}s'.format(te - ts))
    return x

def calc_mobility(F,fullkpts_df,cons):
    """Calculate mobility as per Wu Li PRB 92, 2015"""
    kptdata = fullkpts_df[['k_inds', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]', 'energy', 'vx [m/s]']]
    fermi_distribution(kptdata)
    V = np.dot(np.cross(cons.b1,cons.b2),cons.b3)
    prefactor = cons.e**2/(V*cons.kb*cons.T*len(kptdata))*10**30

    mobility =  prefactor*np.sum(np.multiply(np.multiply(np.multiply(kptdata['k_FD'].values,1-kptdata['k_FD'].values)
                                                         ,kptdata['vx [m/s']),F))
    return mobility


if __name__ == '__main__':
    # data_loc = '/home/peishi/nvme/k200-0.4eV/'
    # chunk_loc = '/home/peishi/nvme/k200-0.4eV/chunked/'

    data_loc = 'D:/Users/AlexanderChoi/GaAs_300K_10_19/k200-0.4eV/'
    chunk_loc = 'D:/Users/AlexanderChoi/GaAs_300K_10_19/chunked/'

    con = preprocessing_largegrid.PhysicalConstants()

    _, kpts_df, enk_df, qpts_df, enq_df = preprocessing_largegrid.loadfromfile(data_loc, matrixel=False)
    cartkpts = preprocessing_largegrid.load_vel_data(data_loc, con)

    reciplattvecs = np.concatenate((con.b1[np.newaxis, :], con.b2[np.newaxis, :], con.b3[np.newaxis, :]), axis=0)
    fbzcartkpts = preprocessing_largegrid.translate_into_fbz(cartkpts.to_numpy()[:, 2:5], reciplattvecs)
    fbzcartkpts = pd.DataFrame(data=fbzcartkpts, columns=['kx [1/A]', 'ky [1/A]', 'kz [1/A]'])
    fbzcartkpts = pd.concat([cartkpts['k_inds'], fbzcartkpts], axis=1)

    # fermi_distribution(fbzcartkpts, fermilevel=con.mu, temp=con.T)
    fermi_distribution(cartkpts, fermilevel=con.mu, temp=con.T)

    nkpts = len(np.unique(kpts_df['k_inds']))
    n_ph_modes = len(np.unique(enq_df['q_inds'])) * len(np.unique(enq_df['im_mode']))
    kinds = np.arange(1, nkpts + 1)

    field = 1  # V/m ??? Just making up a number here

    approach = 'iterative'
    if approach is 'matrix':
        scm = np.memmap(data_loc + 'scattering_matrix.mmap', dtype='float64', mode='r+', shape=(nkpts, nkpts))
        _, edgepoints, lpts, modified_matrix = apply_centraldiff_matrix(scm, fbzcartkpts, field, con)

        edgepoints = fbzcartkpts[np.isin(fbzcartkpts['k_inds'], np.array(edgepoints))]
        fo = plotting.bz_3dscatter(con, fbzcartkpts, enk_df)
        plotting.highlighted_points(fo, edgepoints, con)

        f, f_star = steady_state_solns(modified_matrix, nkpts, cartkpts, 1)
    elif approach is 'iterative':
        scm = np.memmap(data_loc + 'scattering_matrix.mmap', dtype='float64', mode='r', shape=(nkpts, nkpts))
        itsoln = False
        if itsoln:
            start = time.time()
            f_iter, f_rta = iterative_solver(cartkpts, scm)
            end = time.time()
            print('Iterative solver took {:.2f} seconds'.format(end - start))
            np.save('f_iterative', f_iter)
            np.save('f_rta', f_rta)
        else:
            try:
                f_iter = np.load('f_iterative.npy')
                f_rta = np.load('f_rta.npy')
            except:
                exit('Iterative solution not calculated and not stored on file.')

        cgcalc = False
        if cgcalc:
            f_cg = conj_grad_soln(cartkpts, scm)
            np.save('f_conjgrad', f_cg)
        else:
            try:
                f_cg = np.load('f_conjgrad.npy')
            except:
                exit('Conjugate gradient solution not calculated and not stored on file.')

        print('The norm of difference vector of iterative and cg is {:.3E}'.format(np.linalg.norm(f_iter - f_cg)))
        print('The percent difference is {:.3E}'.format(np.linalg.norm(f_iter-f_cg)/np.linalg.norm(f_iter)))
        font = {'size': 14}
        matplotlib.rc('font', **font)
        plt.plot(f_cg, linewidth=1, label='CG')
        plt.plot(f_iter, linewidth=1, label='Iterative')
        plt.plot(f_rta, linewidth=1, label='RTA')
        plt.xlabel('kpoint index')
        plt.ylabel('deviational occupation')
        plt.legend()
        plt.show()