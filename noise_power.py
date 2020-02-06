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
    kb_joule = 1.38064852 * 10 ** (-23)  # Boltzmann constant in SI [m^2 kg s^-2 K^-1]
    df['k_FD'] = (np.exp((df['energy'].values * e - fermilevel * e) / (kb_joule * temp)) + 1) ** (-1)
    return df


def steady_state_solns(matrix, numkpts, fullkpts_df, field):
    """Get steady state solutions"""
    e = 1.602 * 10 ** (-19)  # fundamental electronic charge [C]
    kb = 1.38064852 * 10 ** (-23)  # Boltzmann constant in SI [m^2 kg s^-2 K^-1]

    # Calculate fermi distribution function for all the kpoints. Currently only in chunked dataframes.
    # Want them all in one place her

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
    step_size = 0.005654047459752398 * 1E10  # 1/Angstrom to 1/m

    kptdata = fullkpts_df[['k_inds', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]']]
    kptdata['kpt_mag'] = np.sqrt(kptdata['kx [1/A]'].values**2 + kptdata['ky [1/A]'].values**2 +
                                 kptdata['kz [1/A]'].values**2)
    kptdata['ingamma'] = kptdata['kpt_mag'] < 0.3  # Boolean. In gamma if kpoint magnitude less than some amount

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
        if np.any(slice_df['ingamma'] == 0):
            lvalley_inds.append(slice_inds)
            print('Not applied to {:d} slice because skip L valley'.format(kind))
            continue

        if len(slice_inds) > 4:
            # Subset is the slice sorted by kx value in ascending order. The index of subset still references kptdata.
            subset = slice_df.sort_values(by=['kx [1/A]'], ascending=True)
            ordered_inds = subset['k_inds'].values - 1  # indices of matrix
            icinds.append(ordered_inds[0] + 1)  # +1 to get the k_inds values

            # Set the "initial condition" i.e. the point with the most negative kx value is treated as being zero
            # (and virtual point below)
            matrix[ordered_inds[0], ordered_inds[1]] += - 1/(2*step_size)*cons.e*E/cons.hbar_joule
            matrix[ordered_inds[1], ordered_inds[2]] += - 1/(2*step_size)*cons.e*E/cons.hbar_joule

            # Set the other "boundary condition" i.e. the point with the most positive kx value is treated as being zero
            # (and virtual point above)
            last = len(ordered_inds) - 1
            slast = len(ordered_inds) - 2
            matrix[ordered_inds[last], ordered_inds[slast]] += 1/(2*step_size)*cons.e*E/cons.hbar_joule
            matrix[ordered_inds[slast], ordered_inds[slast-1]] += 1/(2*step_size)*cons.e*E/cons.hbar_joule

            # Set the value of all other points in the slice
            inter_inds = ordered_inds[2:slast]
            inter_inds_up = ordered_inds[3:last]
            inter_inds_down = ordered_inds[1:slast-1]

            matrix[inter_inds, inter_inds_up] += (-1) * 1/(2*step_size)*cons.e*E/cons.hbar_joule
            matrix[inter_inds, inter_inds_down] += 1/(2*step_size)*cons.e*E/cons.hbar_joule

        else:
            shortslice_inds.append(slice_inds)

        end = time.time()
        if kind % 10 == 0:
            print('Finished {:d} out of {:d} slices in {:.3f}s'.format(kind, len(uniq_yz),end-start))
    print('Scattering matrix modified to incorporate central difference contribution.')
    print('Not applied to {:d} points because fewer than 5 points on the slice.'.format(len(shortslice_inds)))
    print('Finite difference not applied to L valleys. Derivative treated as zero for these points.')
    return shortslice_inds, icinds, lvalley_inds, matrix


def iterative_solver_simple(b, matrix, convergence=1E-4):
    """An iterative solver that just takes a b vector (RHS of Ax=b) and a matrix (A in Ax=b) and solves for x with an
    iterative method. Start by finding an x0 which has components x0_i = b_i / A_ii and then add off diagonal components
    until convergence criteria.
    :parameter b: The RHS of Ax=b
    :parameter matrix: The matrix in Ax=b
    :parameter convergence: Convergence criteria given as |x_i+1 - x_i| / |x_i|

    :returns x_next: The converged solution
    :returns x_0: The starting vector for comparison"""

    # The prefactor is just the inverse of the diagonal of the matrix
    prefactor = (np.diag(matrix)) ** (-1)
    x_0 = b * prefactor

    errpercent = 1
    counter = 0
    x_prev = x_0
    while errpercent > convergence and counter < 20:
        s1 = time.time()
        mvp = np.matmul(matrix, x_prev)
        e1 = time.time()
        print('Matrix vector multiplication took {:.2f}s'.format(e1-s1))
        # Remove diagonal terms from the matrix multiplication (prevent double counting of diagonal term)
        offdiagsum = mvp - (np.diag(matrix) * x_prev)
        x_next = x_0 + (prefactor * offdiagsum)
        errvecnorm = np.linalg.norm(x_next - x_prev)
        errpercent = errvecnorm / np.linalg.norm(x_prev)
        x_prev = x_next
        counter += 1
        print('Iteration {:d}: Error percent is {:.3E}. Abs error norm is {:.3E}'
              .format(counter, errpercent, errvecnorm))
    return x_next, x_0


def iterative_solver_lowfield(kptdf, matrix):
    """Iterative solver hard coded for solving the BTE in low field approximation"""
    # sr = np.load(data_loc + 'scattering_rates_direct.npy')
    # tau = 1 / sr
    prefactor = (-1) * (np.diag(matrix))**(-1)
    f_0 = np.squeeze(kptdf['vx [m/s]'] * kptdf['k_FD']) * (1 - kptdf['k_FD']) * prefactor
    # f_0 = kptdf['vx [m/s]'] * tau
    # matrix -= np.diag(np.diag(matrix))

    errpercent = 1
    counter = 0
    f_prev = f_0
    while errpercent > 1E-4 and counter < 20:
        s1 = time.time()
        mvp = np.matmul(matrix, f_prev)
        e1 = time.time()
        print('Matrix vector multiplication took {:.2f}s'.format(e1-s1))
        # Remove the part of the vector from the diagonal multiplication
        offdiagsum = mvp - (np.diag(matrix) * f_prev)
        f_next = f_0 + (prefactor * offdiagsum)
        errvecnorm = np.linalg.norm(f_next - f_prev)
        errpercent = errvecnorm / np.linalg.norm(f_prev)
        f_prev = f_next
        counter += 1
        print('Iteration {:d}: Error percent is {:.3E}. Abs error norm is {:.3E}'
              .format(counter, errpercent, errvecnorm))
    return f_next, f_0


def drift_velocity(f, cons):
    """Assuming that f is in the form of chi/(eE/kbT)"""
    vd = np.sum(f)*cons.e*cons.E/(cons.kb_ev*cons.T)/len(f)
    return vd


def conj_grad_soln(kptdf, matrix):
    b = (-1) * np.squeeze(kptdf[['vx [m/s]']].values * kptdf[['k_FD']].values) * (1 - kptdf['k_FD'])

    ts = time.time()
    x, status = scipy.sparse.linalg.cg(matrix, b)
    te = time.time()
    print('Conjugate gradient solve successful. Took {:.2f}s'.format(te - ts))
    return x


def calc_mobility(F, kptdata, cons):
    """Calculate mobility as per Wu Li PRB 92, 2015"""
    # kptdata = fullkpts_df[['k_inds', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]', 'energy', 'vx [m/s]']]
    # fermi_distribution(kptdata)
    V = np.dot(np.cross(cons.b1, cons.b2), cons.b3) * 1E-30  # unit cell volume in m^3
    Nuc = len(kptdata)
    prefactor = 2 * cons.e**2 / (V*cons.kb_ev*cons.T*Nuc)

    conductivity = prefactor * np.sum(kptdata['FD'] * (1 - kptdata['FD']) * kptdata['vx [m/s]'] * F)
    carrier_dens = 2 / Nuc / V * np.sum(kptdata['FD'])
    mobility = conductivity / cons.e / carrier_dens
    print('Mobility is {:.10E}'.format(mobility))


def g_conj_grad_soln(kptdf, matrix, f, cons):
    """Assuming that f is in the form of chi/(eE/kbT)"""
    vd = drift_velocity(f, cons)
    b = (vd-kptdf[['vx [m/s]']].values)/(cons.e*cons.E)*cons.kb*cons.T
    ts = time.time()
    x, status = scipy.sparse.linalg.cg(matrix, b)
    te = time.time()
    print('Conjugate gradient solve successful. Took {:.2f}s'.format(te - ts))
    return x


def steady_state_full_drift_iterative_solver(matrix_sc, matrix_fd, kptdf, c, convergence=1E-4):
    field = 1E4  # in Volts/meter

    _, _, _, matrix_fd = apply_centraldiff_matrix(matrix_fd, kptdf, field, c)

    b = (-1)*c.e*field/c.kb_joule/c.T * np.squeeze(kptdf['vx [m/s]'] * kptdf['k_FD']) * (1 - kptdf['k_FD'])
    # chi2psi is used to give the finite difference matrix the right factors in front since substitution made
    chi2psi = np.squeeze(kptdf['k_FD'] * (1 - kptdf['k_FD']))
    invdiag = (np.diag(matrix_sc)) ** (-1)
    x_0 = b * invdiag

    errpercent = 1
    counter = 0
    x_prev = x_0
    while errpercent > convergence and counter < 20:
        s1 = time.time()
        mvp_sc = np.matmul(matrix_sc, x_prev)
        # Remove diagonal terms from the scattering matrix multiplication (prevent double counting of diagonal term)
        # Also include  2pi^2 factor that we believe is the conversion between radians and seconds
        offdiag_sc = (mvp_sc - (np.diag(matrix_sc) * x_prev)) * (2 * np.pi)**2
        # There's no diagonal component of the finite difference matrix so matmul directly gives contribution
        offdiag_fd = np.matmul(matrix_fd, chi2psi * x_prev)
        e1 = time.time()
        print('Two matrix vector multiplications took {:.2f}s'.format(e1 - s1))
        x_next = x_0 + (invdiag * (offdiag_fd - offdiag_sc))

        errvecnorm = np.linalg.norm(x_next - x_prev)
        errpercent = errvecnorm / np.linalg.norm(x_prev)
        x_prev = x_next
        counter += 1
        print('Iteration {:d}: Error percent is {:.3E}. Abs error norm is {:.3E}'
              .format(counter, errpercent, errvecnorm))
    return x_next, x_0


if __name__ == '__main__':
    data_loc = '/home/peishi/nvme/k200-0.4eV/'
    chunk_loc = '/home/peishi/nvme/k200-0.4eV/chunked/'
    # data_loc = '/home/peishi/storage/k200-0.4eV/'  # for Comet
    # chunk_loc = '/home/peishi/storage/chunked/'
    # data_loc = '/p/work3/peishi/k200-0.4eV/'  # for gaffney (navy cluster)
    # chunk_loc = '/p/work3/peishi/chunked/'
    # data_loc = 'D:/Users/AlexanderChoi/GaAs_300K_10_19/k200-0.4eV/'
    # chunk_loc = 'D:/Users/AlexanderChoi/GaAs_300K_10_19/chunked/'

    con = preprocessing_largegrid.PhysicalConstants()

    _, kpts_df, enk_df, qpts_df, enq_df = preprocessing_largegrid.loadfromfile(data_loc, matrixel=False)
    cartkpts = preprocessing_largegrid.load_vel_data(data_loc, con)

    reciplattvecs = np.concatenate((con.b1[np.newaxis, :], con.b2[np.newaxis, :], con.b3[np.newaxis, :]), axis=0)
    fbzcartkpts = preprocessing_largegrid.translate_into_fbz(cartkpts.values[:, 2:5], reciplattvecs)
    fbzcartkpts = pd.DataFrame(data=fbzcartkpts, columns=['kx [1/A]', 'ky [1/A]', 'kz [1/A]'])
    fbzcartkpts = pd.concat([cartkpts[['k_inds', 'vx [m/s]', 'energy']], fbzcartkpts], axis=1)

    fbzcartkpts = fermi_distribution(fbzcartkpts, fermilevel=con.mu, temp=con.T)
    cartkpts = fermi_distribution(cartkpts, fermilevel=con.mu, temp=con.T)

    nkpts = len(np.unique(kpts_df['k_inds']))
    n_ph_modes = len(np.unique(enq_df['q_inds'])) * len(np.unique(enq_df['im_mode']))
    kinds = np.arange(1, nkpts + 1)

    trythis = False
    approach = 'iterative'
    if approach is 'matrix' and trythis:
        scm = np.memmap(data_loc + 'scattering_matrix.mmap', dtype='float64', mode='r+', shape=(nkpts, nkpts))
        # _, edgepoints, lpts, modified_matrix = apply_centraldiff_matrix(scm, fbzcartkpts, field, con)

        edgepoints = fbzcartkpts[np.isin(fbzcartkpts['k_inds'], np.array(edgepoints))]
        fo = plotting.bz_3dscatter(con, fbzcartkpts, enk_df)
        plotting.highlighted_points(fo, edgepoints, con)

        f, f_star = steady_state_solns(modified_matrix, nkpts, cartkpts, 1)
    elif approach is 'iterative' and trythis:
        scm = np.memmap(data_loc + 'scattering_matrix.mmap', dtype='float64', mode='r', shape=(nkpts, nkpts))
        itsoln = False
        if itsoln:
            start = time.time()
            f_iter, f_rta = iterative_solver_lowfield(cartkpts, scm)
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

        # print('The norm of difference vector of iterative and cg is {:.3E}'.format(np.linalg.norm(f_iter - f_cg)))
        # print('The percent difference is {:.3E}'.format(np.linalg.norm(f_iter-f_cg)/np.linalg.norm(f_iter)))

        # plotting.plot_cg_iter_rta(f_cg, f_iter, f_rta)
        # plotting.plot_1dim_steady_soln(f_iter, cartkpts)

        # print('RTA mobility')
        # calc_mobility(f_rta, cartkpts, con)
        # print('Iterative mobility')
        # calc_mobility(f_iter, cartkpts, con)
        # print('CG mobility')
        # calc_mobility(f_cg, cartkpts, con)

    solve_full_steadystatebte = True
    if solve_full_steadystatebte:
        scm = np.memmap(data_loc + 'scattering_matrix.mmap', dtype='float64', mode='r', shape=(nkpts, nkpts))
        fdm = np.memmap(data_loc + 'finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
        steady_state_full_drift_iterative_solver(scm, fdm, cartkpts, con)
