import numpy as np
import scipy.sparse.linalg
import multiprocessing as mp
from functools import partial
import os
import pandas as pd
import time
import re
import glob
import matplotlib.pyplot as plt
import pickle
import constants as c
import utilities


# The following set of functions calculate solutions to the steady Boltzmann Equation and write the solutions to file.
def iterative_solver_lowfield(df, scm, simplelin=True, applyscmFac=False):
    """Iterative solver hard coded for solving the BTE in low field approximation. Sign convention by Wu Li. Returns f,
    which is equal to chi/(eE/kT).

    Parameters:
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        scm (memmap): Memory-mapped array containing the scattering matrix, assumed simple linearization by default.
        simplelin (bool): Boolean that specifies whether the scattering matrix is assumed simple or canonical
        linearization, assumed simple by default (i.e. simplelin=True).
        applyscmFac (bool): Boolean that specifies whether or not to apply the 2*pi squared factor.

    Returns:
        f_next (nparray): Numpy array containing the (hopefully) converged iterative solution as psi/(eE/kT).
        f_0 (nparray): Numpy array containing the RTA solution as psi_0/(eE/kT).
    """
    if applyscmFac:
        scmfac = (2*np.pi)**2
        print('Applying 2 Pi-squared factor.')
    else:
        scmfac = 1
    prefactor = (np.diag(scm) * scmfac)**(-1)
    f_0 = (-1) * np.squeeze(df['vx [m/s]'] * df['k_FD']) * (1 - df['k_FD']) * prefactor
    print('The avg abs val of f_rta is {:.3E}'.format(np.average(np.abs(f_0))))
    print('The sum over f_rta is {:.3E}'.format(np.sum(f_0)))
    chi2psi = np.squeeze(df['k_FD'] * (1 - df['k_FD']))
    errpercent = 1
    counter = 0
    f_prev = f_0
    while errpercent > 5E-4 and counter < 25:
        s1 = time.time()
        mvp = np.matmul(scm, f_prev) * scmfac
        e1 = time.time()
        print('Matrix vector multiplication took {:.2f}s'.format(e1-s1))
        # Remove the part of the vector from the diagonal multiplication
        offdiagsum = mvp - (np.diag(scm) * f_prev * scmfac)
        f_next = f_0 - (prefactor * offdiagsum)
        print('The avg abs val of offdiag part is {:.3E}'.format(np.average(np.abs(prefactor * offdiagsum))))
        errvecnorm = np.linalg.norm(f_next - f_prev)
        errpercent = errvecnorm / np.linalg.norm(f_prev)
        f_prev = f_next
        counter += 1
        print('Iteration {:d}: Error percent is {:.3E}. Abs error norm is {:.3E}\n'
              .format(counter, errpercent, errvecnorm))

    if simplelin:
        print('Converting chi to psi since matrix in simple linearization. Returning solution as F.')
        f_next = f_next / chi2psi
        f_0 = f_0 / chi2psi
    return f_next, f_0


def write_iterative_solver_lowfield(outLoc,inLoc,df,simplelin2=True,applyscmFac2=False):
    """Calls the iterative solver hard coded for solving the BTE in low field approximation and writes the single F_psi
     solution to file.

    Parameters:
        outLoc (str): String containing the location of the directory to write the chi solutions.
        inLoc (str): String containing the location of the directory containing the scattering matrix, assumed simple
        linearization by default.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        simplelin2 (bool): Boolean that specifies whether the scattering matrix is assumed simple or canonical
        linearization, assumed simple by default (i.e. canonical=False).
        applyscmFac2 (bool): Boolean that specifies whether or not to apply the 2*pi squared factor.

    Returns:
        None. Just writes the F_psi solution to file. FPsi_#. #1 corresponds to low-field RTA, #2 corresponds to
        low-field iterative.
    """
    nkpts = len(np.unique(df['k_inds']))
    scm = np.memmap(inLoc + 'scattering_matrix_5.87_simple.mmap', dtype='float64', mode='r', shape=(nkpts, nkpts))
    f_next, f_0 = iterative_solver_lowfield(df, scm, simplelin=simplelin2, applyscmFac=applyscmFac2)
    np.save(outLoc +'f_' + '1', f_0)
    np.save(outLoc +'f_' + '2', f_next)
    print('f solutions written to file.')


def apply_centraldiff_matrix(matrix,fullkpts_df,E,step_size=1):
    """Given a scattering matrix, calculate a modified matrix using the central difference stencil and apply bc. In the
    current version, bc is not applied to points in the L valley.

    Parameters:
        matrix (memmap): Scattering matrix in simple linearization.
        fullkpts_df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        E (dbl): Value of the electric field in V/m.
        scm (memmap): Memory-mapped array containing the scattering matrix, assumed simple linearization by default.
        step_size (dbl): Specifies the spacing between consecutive k-pts for the integration.

    Returns:
        shortslice_inds (numpyarray): Array of indices associated with the points along ky-kz lines with fewer than 5 pts.
        np.array(icinds) (numpyarray): Array of indices associated with the points where the ic is applied.
        np.array(icinds) (numpyarray): Array of indices associated with the points where the ic is applied.
        matrix (memmap): Memory-mapped array containing the modified scattering matrix, accounting for the FDM.
    """
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
            # print('Not applied to {:d} slice because skip L valley'.format(kind))
            continue

        if len(slice_inds) > 4:
            # Subset is the slice sorted by kx value in ascending order. The index of subset still references kptdata.
            subset = slice_df.sort_values(by=['kx [1/A]'], ascending=True)
            ordered_inds = subset['k_inds'].values - 1  # indices of matrix
            icinds.append(ordered_inds[0] + 1)  # +1 to get the k_inds values

            # Set the "initial condition" i.e. the point with the most negative kx value is treated as being zero
            # (and virtual point below)
            matrix[ordered_inds[0], ordered_inds[1]] += 1/(2*step_size)*c.e*E/c.hbar_joule
            matrix[ordered_inds[1], ordered_inds[2]] += 1/(2*step_size)*c.e*E/c.hbar_joule

            # Set the other "boundary condition" i.e. the point with the most positive kx value is treated as being zero
            # (and virtual point above)
            last = len(ordered_inds) - 1
            slast = len(ordered_inds) - 2
            matrix[ordered_inds[last], ordered_inds[slast]] += -1 * 1/(2*step_size)*c.e*E/c.hbar_joule
            matrix[ordered_inds[slast], ordered_inds[slast-1]] += -1 * 1/(2*step_size)*c.e*E/c.hbar_joule

            # Set the value of all other points in the slice
            inter_inds = ordered_inds[2:slast]
            inter_inds_up = ordered_inds[3:last]
            inter_inds_down = ordered_inds[1:slast-1]

            matrix[inter_inds, inter_inds_up] += 1/(2*step_size)*c.e*E/c.hbar_joule
            matrix[inter_inds, inter_inds_down] += -1 * 1/(2*step_size)*c.e*E/c.hbar_joule

        else:
            shortslice_inds.append(slice_inds)

        end = time.time()
        if kind % 10 == 0:
            pass
            # print('Finished {:d} out of {:d} slices in {:.3f}s'.format(kind, len(uniq_yz),end-start))
    print('Scattering matrix modified to incorporate central difference contribution.')
    print('Not applied to {:d} points because fewer than 5 points on the slice.'.format(len(shortslice_inds)))
    print('Finite difference not applied to L valleys. Derivative treated as zero for these points.')
    return shortslice_inds, np.array(icinds), lvalley_inds, matrix


def steady_state_full_drift_iterative_solver(matrix_sc, matrix_fd, kptdf, field, canonical=False, applyscmFac=False,
                                             convergence=5E-4):
    """Iterative solver for calculating steady BTE solution in the form of Chi using the full finite difference matrix.

    Parameters:
        matrix_sc (memmap): Scattering matrix in simple linearization by default..
        matrix_fd (memmap): Finite difference matrix, generated by apply_centraldiff_matrix.
        kptdf (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        field (dbl): Value of the electric field in V/m.
        canonical (bool): Boolean that specifies whether the scattering matrix is assumed simple or canonical
        applyscmFac (bool): Boolean that specifies whether or not to apply the 2*pi squared factor.
        convergence (dbl): Specifies the percentage threshold for convergence.

    Returns:
        x_next (nparray): Numpy array containing the (hopefully) converged iterative solution as chi.
        x_0 (nparray): Numpy array containing the RTA solution as chi.
    """
    print('Starting steady_state_full_drift_iterative solver for {:.3E}'.format(field))
    if applyscmFac:
        scmfac = (2*np.pi)**2
        print('Applying 2 Pi-squared factor.')
    else:
        scmfac = 1

    _, icinds, _, matrix_fd = apply_centraldiff_matrix(matrix_fd, kptdf, field)

    b = (-1)*c.e*field/c.kb_joule/c.T * np.squeeze(kptdf['vx [m/s]'] * kptdf['k_FD']) * (1 - kptdf['k_FD'])
    # chi2psi is used to give the finite difference matrix the right factors in front since substitution made
    chi2psi = np.squeeze(kptdf['k_FD'] * (1 - kptdf['k_FD']))
    invdiag = (np.diag(matrix_sc) * scmfac) ** (-1)
    x_0 = b * invdiag
    print('The avg abs val of f_rta is {:.3E}'.format(np.average(np.abs(x_0))))
    print('The sum over f_rta is {:.3E}'.format(np.sum(x_0)))

    errpercent = 1
    counter = 0
    x_prev = x_0

    from scipy.linalg import get_blas_funcs
    print('Starting convergence loop')
    loopstart = time.time()
    while errpercent > convergence and counter < 40:
        # Directly make the boundary condition points zero. icinds is 1-indexed. Subtract 1.
        # x_prev[icinds - 1] = 0
        s1 = time.time()
        mvp_sc = np.matmul(matrix_sc, x_prev) * scmfac

        # Remove diagonal terms from the scattering matrix multiplication (prevent double counting of diagonal term)
        # Also include  2pi^2 factor that we believe is the conversion between radians and seconds
        offdiag_sc = mvp_sc - (np.diag(matrix_sc) * x_prev * scmfac)
        # There's no diagonal component of the finite difference matrix so matmul directly gives contribution
        # If using simple linearization (chi instead of psi) then don't use chi2psi term
        if not canonical:
            offdiag_fd = np.matmul(matrix_fd, x_prev)
        else:
            offdiag_fd = np.matmul(matrix_fd, chi2psi * x_prev)
        e1 = time.time()
        print('Matrix vector multiplications took {:.2f}s'.format(e1 - s1))
        print('The 2-norm of offdiag FDM part is {:.3E}'.format(np.linalg.norm(offdiag_fd)))
        print('The 2-norm of offdiag scattering part is {:.3E}'.format(np.linalg.norm(offdiag_sc)))
        x_next = x_0 + (invdiag * (offdiag_fd - offdiag_sc))

        errvecnorm = np.linalg.norm(x_next - x_prev)
        errpercent = errvecnorm / np.linalg.norm(x_prev)
        x_prev = x_next
        counter += 1
        print('Iteration {:d}: Error percent is {:.3E}. Abs error norm is {:.3E}\n'
              .format(counter, errpercent, errvecnorm))
    loopend = time.time()
    print('Convergence took {:.2f}s'.format(loopend - loopstart))
    if canonical:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in canonical linearization')
        x_next = x_next * chi2psi
        x_0 = x_0 * chi2psi
    return x_next, x_0


def write_iterative_solver_fdm(outLoc, inLoc, fieldVector, df, canonical2=False, applyscmFac2=False,
                               convergence2=5E-4):
    """Calls the iterative solver hard coded for solving the BTE with full FDM and writes the chis to file.

    Parameters:
        outLoc (str): String containing the location of the directory to write the chi solutions.
        inLoc (str): String containing the location of the directory containing the scattering matrix, assumed simple
        linearization by default.
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        canonical2 (bool): Boolean that specifies whether the scattering matrix is assumed simple or canonical
        linearization, assumed simple by default (i.e. canonical=False).
        applyscmFac2 (bool): Boolean that specifies whether or not to apply the 2*pi squared factor.
        convergence2 (dbl): Specifies the percentage threshold for convergence.

    Returns:
        None. Just writes the chi solutions to file. chi_#_EField. #1 corresponds to low-field RTA, #2 corresponds to
        low-field iterative, #3 corresponds to full finite-difference iterative.
    """
    nkpts = len(np.unique(df['k_inds']))
    scm = np.memmap(inLoc + 'scattering_matrix_5.87_simple.mmap', dtype='float64', mode='r', shape=(nkpts, nkpts))
    for i in range(len(fieldVector)):
        fdm = np.memmap(inLoc + 'finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
        EField = fieldVector[i]
        x_next, _ = steady_state_full_drift_iterative_solver(scm, fdm, df, EField, canonical2, applyscmFac2, convergence2)
        del fdm
        np.save(outLoc + 'chi_' + '3_' + "{:.1e}".format(EField), x_next)
        print('Solution written to file for ' + "{:.1e}".format(EField))


# The following set of functions calculate the solutions to the effective Boltzmann equation and write the solutions
def eff_distr_g_rta(chi,matrix_sc,df,simplelin=True,applyscmFac=False):
    """For calculating effective BTE solution in the form of g_Chi using the low-field RTA solution.
    Parameters:
        chi (nparray): Solution for the steady distribution function in chi form.
        matrix_sc (memmap): Scattering matrix in simple linearization by default..
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        simplelin (bool): Boolean that specifies whether the scattering matrix is assumed simple or canonical
        applyscmFac (bool): Boolean that specifies whether or not to apply the 2*pi squared factor.

    Returns:
        g_linear (nparray): Numpy array containing the effective distribution function for the low-field approx."""
    if applyscmFac:
        scmfac = (2*np.pi)**2
        print('Applying 2 Pi-squared factor.')
    else:
        scmfac = 1
    psi2chi = np.squeeze(df['k_FD'] * (1 - df['k_FD']))
    f = chi + df['k_FD']
    vd = drift_velocity(chi,df)
    invdiag = (np.diag(matrix_sc) * scmfac) ** (-1)
    print('Invdiag mean')
    print(np.mean(invdiag))
    g_lin = (vd-df['vx [m/s]']) * invdiag * f
    g0 = (-df['vx [m/s]']) * invdiag * df['k_FD']
    print('Sum g0')
    print(np.sum(g0))
    if not simplelin:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in simple linearization')
        g_lin = g_next * psi2chi
    return g_lin, g0


def eff_distr_g_iterative_solver(chi, matrix_sc, matrix_fd, df, field, simplelin=True, applyscmFac=False, convergence=5E-4):
    """Iterative solver for calculating effective BTE solution in the form of g_Chi using the full finite difference matrix.

    Parameters:
        chi (nparray): Solution for the steady distribution function in chi form.
        matrix_sc (memmap): Scattering matrix in simple linearization by default..
        matrix_fd (memmap): Finite difference matrix, generated by apply_centraldiff_matrix.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        field (dbl): Value of the electric field in V/m.
        simplelin (bool): Boolean that specifies whether the scattering matrix is assumed simple or canonical
        applyscmFac (bool): Boolean that specifies whether or not to apply the 2*pi squared factor.
        convergence (dbl): Specifies the percentage threshold for convergence.

    Returns:
        g_next (nparray): Numpy array containing the (hopefully) converged iterative solution as g_chi.
        g_0 (nparray): Numpy array containing the RTA solution as g0_chi.
    """
    print('Starting eff_distr_g_iterative_solver solver for {:.3E}'.format(field))
    if applyscmFac:
        scmfac = (2*np.pi)**2
        print('Applying 2 Pi-squared factor.')
    else:
        scmfac = 1

    _, _, _, matrix_fd = apply_centraldiff_matrix(matrix_fd, df, field, c)
    # Will only be able to run if you have a precalculated chi stored on file
    vd = drift_velocity(chi, df)
    f0 = df['k_FD'].values
    f = chi + f0
    b = (-1) * ((df['vx [m/s]'] - vd) * f)
    # chi2psi is used to give the finite difference matrix the right factors in front since substitution made
    psi2chi = np.squeeze(df['k_FD'] * (1 - df['k_FD']))
    invdiag = (np.diag(matrix_sc) * scmfac) ** (-1)
    g_0 = b * invdiag
    print('The avg abs val of g_0 is {:.3E}'.format(np.average(np.abs(g_0))))
    print('The sum over g_0 is {:.3E}'.format(np.sum(g_0)))

    errpercent = 1
    counter = 0
    g_prev = g_0
    loopstart = time.time()
    while errpercent > convergence and counter < 40:
        s1 = time.time()
        mvp_sc = np.matmul(matrix_sc, g_prev) * scmfac
        offdiag_sc = mvp_sc - (np.diag(matrix_sc) * g_prev * scmfac)
        # There's no diagonal component of the finite difference matrix so matmul directly gives contribution
        # If using simple linearization (chi instead of psi) then don't use chi2psi term
        if simplelin:
            offdiag_fd = np.matmul(matrix_fd, g_prev)
        else:
            offdiag_fd = np.matmul(matrix_fd, psi2chi * g_prev)
        e1 = time.time()
        print('Matrix vector multiplications took {:.2f}s'.format(e1 - s1))
        print('The 2-norm of offdiag FDM part is {:.3E}'.format(np.linalg.norm(offdiag_fd)))
        print('The 2-norm of offdiag scattering part is {:.3E}'.format(np.linalg.norm(offdiag_sc)))
        g_next = g_0 + (invdiag * (offdiag_fd - offdiag_sc))
        errvecnorm = np.linalg.norm(g_next - g_prev)
        errpercent = errvecnorm / np.linalg.norm(g_prev)
        g_prev = g_next
        counter += 1
        print('Iteration {:d}: Error percent is {:.3E}. Abs error norm is {:.3E}\n'
              .format(counter, errpercent, errvecnorm))
    loopend = time.time()
    print('Convergence took {:.2f}s'.format(loopend - loopstart))
    if not simplelin:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in simple linearization')
        g_next = g_next * psi2chi
        g_0 = g_0 * psi2chi
    return g_next, g_0


def write_iterative_solver_g(outLoc, inLoc, fieldVector, df, simplelin2=True, applyscmFac2 = False,
                               convergence2=5E-4):
    """Calls the iterative solver hard coded for solving the effective BTE w/FDM and writes the chis to file.

    Parameters:
        outLoc (str): String containing the location of the directory to write the chi solutions and ready steady state chis.
        inLoc (str): String containing the location of the directory containing the scattering matrix, assumed simple
        linearization by default.
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        simplelin2 (bool): Boolean that specifies whether the scattering matrix is assumed simple or canonical
        linearization, assumed simple by default (i.e. canonical=False).
        applyscmFac2 (bool): Boolean that specifies whether or not to apply the 2*pi squared factor.
        convergence2 (dbl): Specifies the percentage threshold for convergence.

    Returns:
        None. Just writes the chi solutions to file. chi_#_EField. #1 corresponds to low-field RTA, #2 corresponds to
        low-field iterative, #3 corresponds to full finite-difference iterative.
    """
    nkpts = len(np.unique(df['k_inds']))
    scm = np.memmap(inLoc + 'scattering_matrix_5.87_simple.mmap', dtype='float64', mode='r', shape=(nkpts, nkpts))
    for i in range(len(fieldVector)):
        EField = fieldVector[i]
        chi = np.load(outLoc + 'chi_3_{:.1e}.npy'.format(EField))
        fdm = np.memmap(inLoc + 'finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
        g_next, g_3_johnson = eff_distr_g_iterative_solver(chi, scm, fdm, df, EField, simplelin2, applyscmFac2, convergence2)
        del fdm
        f_i = np.load(outLoc + 'f_1.npy')
        chi_1_i = f2chi(f_i, df, EField)
        g_rta, g_1_johnson = eff_distr_g_rta(chi_1_i, scm, df, simplelin2, applyscmFac2)
        np.save(outLoc + 'g_' + '1_' + "{:.1e}".format(EField), g_rta)
        np.save(outLoc + 'g_' + '3_' + "{:.1e}".format(EField), g_next)
        print('Solution written to file for ' + "{:.1e}".format(EField))
    np.save(outLoc + 'g_' + '1_johnson', g_1_johnson)
    np.save(outLoc + 'g_' + '3_johnson', g_3_johnson)


def load_problem_params(inLoc):
    print('Physical constants loaded from' + in_Loc)
    print('Temperature is {:.1e} K'.format(c.T))
    print('Fermi Level is {:.1e} eV'.format(c.mu))
    print('Gaussian broadening is {:.1e} eV'.format(c.b))
    print('Grid density is {:.1e} cubed'.format(c.gD))


if __name__ == '__main__':
    # out_Loc = 'D:/Users/AlexanderChoi/Dropbox (Minnich Lab)/Minnich Lab Team ' \
    #           'Folder/Peishi+Alex/BoltzmannGreensFunctionSolver/#1_Problem/1_Pipeline/Output/ '
    # in_Loc = 'D:/Users/AlexanderChoi/Dropbox (Minnich Lab)/Minnich Lab Team ' \
    #           'Folder/Peishi+Alex/BoltzmannGreensFunctionSolver/#1_Problem/0_Data/'

    out_Loc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/#1_Problem/1_Pipeline/Output/'
    in_Loc = 'E:/Dropbox (Minnich Lab)/Alex_Peishi_Noise_Calcs/BoltzmannGreenFunctionNoise/#1_Problem/0_Data/'

    load_problem_params(in_Loc)
    fields = np.array([2e1,3e1,4e1,5e1,6e1,7e1,8e1,9e1,1e2,2e2,3e2,4e2,5e2,6e2,7e2,8e2,9e2])
    electron_df = pd.read_pickle(in_Loc+'electron_df.pkl')
    electron_df = utilities.fermi_distribution(electron_df)
    # Steady state solutions
    #write_iterative_solver_lowfield(out_Loc, in_Loc, electron_df, True, True)
    write_iterative_solver_fdm(out_Loc, in_Loc, fields, electron_df, False, True,5E-4)
    write_iterative_solver_g(out_Loc, in_Loc, fields, electron_df,True, True,5E-4)
