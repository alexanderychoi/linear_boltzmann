import numpy as np
import pandas as pd
import time
import constants as c
import utilities
import problemparameters as pp
import matplotlib.pyplot as plt
from scipy.spatial import distance
import numpy.linalg
from scipy.sparse import linalg


# The following set of functions calculate solutions to the steady Boltzmann Equation and write the solutions to file.
def iterative_solver_lowfield(df, scm, simplelin=True, applyscmFac=False,doIt=False):
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

    invdiag = np.asarray(np.diag(scm).tolist())

    if doIt:
        inds = np.asarray(df.loc[df['kx [1/A]'] < 0].index)
        invdiag[inds] = invdiag[inds]*3
        plt.figure()
        plt.plot(df['kx [1/A]'].values,-invdiag,'.')
        plt.show()

    prefactor = (invdiag * scmfac)**(-1)
    f_0 = (-1) * np.squeeze(df['vx [m/s]'] * df['k_FD']) * (1 - df['k_FD']) * prefactor
    print('The avg abs val of f_rta is {:.3E}'.format(np.average(np.abs(f_0))))
    print('The sum over f_rta is {:.3E}'.format(np.sum(f_0)))
    chi2psi = np.squeeze(df['k_FD'] * (1 - df['k_FD']))
    errpercent = 1
    counter = 0
    f_prev = f_0
    while errpercent > 5E-4 and counter < 200:
    # while errpercent > 3e-3:
        s1 = time.time()
        mvp = np.matmul(scm, f_prev) * scmfac
        e1 = time.time()
        print('The sum over f_prev is {:3E}'.format(np.sum(f_prev)))
        print('Matrix vector multiplication took {:.2f}s'.format(e1-s1))
        # Remove the part of the vector from the diagonal multiplication
        offdiagsum = mvp - (invdiag * f_prev * scmfac)
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
    b = -np.squeeze(df['vx [m/s]'] * df['k_FD']) * (1 - df['k_FD'])/scmfac
    print('Checking b.')
    b_check = np.dot(scm,f_next)
    error = np.linalg.norm(b_check - b)/np.linalg.norm(b)
    print(error)

    print('Cosine similarity.')
    cosdist = distance.cosine(b_check,b)
    cos_sim = 1-cosdist
    print(cos_sim)

    # plt.figure()
    # plt.plot(b, label ='Matrix Vector Product')
    #
    # plt.plot(b_check, label='Forcing',alpha =0.6)
    # plt.title('Low field iterative error check')
    # plt.xlabel('kpt index')
    # plt.legend()
    # plt.show()

    # plt.figure()
    # plt.plot(df['kx [1/A]'].values,df['vx [m/s]'].values,'.')
    # plt.xlabel('kpt index')
    # plt.ylabel('vx [m/s]')
    # plt.show()

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
    field = 1
    nkpts = len(np.unique(df['k_inds']))
    scm = np.memmap(inLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    f_next, f_0 = iterative_solver_lowfield(df, scm, simplelin=simplelin2, applyscmFac=applyscmFac2)
    np.save(outLoc +'f_' + '1', f_0)
    np.save(outLoc +'f_' + '2', f_next)
    print('f solutions written to file.')
    # pos_kx_loc = df.loc[df['kx [1/A]'] > 0].index
    # neg_kx_loc = df.loc[df['kx [1/A]'] < 0].index
    # velocity_factor = np.asarray([1,1.5,1.75,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.10])
    # f_next = []
    # f_0 = []
    # f_next_pop = []
    # Nuc = len(df)
    # for i in range(len(velocity_factor)):
    #     temp_df = electron_df.copy(deep=True)
    #     temp_df.loc[temp_df['kx [1/A]'] < 0, 'vx [m/s]'] = temp_df.loc[temp_df['kx [1/A]'] < 0, 'vx [m/s]'] * velocity_factor[i]
    #     temp_df.loc[temp_df['kx [1/A]'] > 0, 'vx [m/s]'] = temp_df.loc[temp_df['kx [1/A]'] > 0, 'vx [m/s]'] * velocity_factor[i]
    #
    #     f_next_temp, f_0_temp = iterative_solver_lowfield(temp_df, scm, simplelin=simplelin2, applyscmFac=applyscmFac2)
    #     f_next_pop.append(10**-6 * 2 / Nuc / c.Vuc * np.sum(utilities.f2chi(f_next_temp, df, field)+df['k_FD'].values))
    #     f_0.append(f_0_temp)
    # plt.figure()
    # #plt.plot(velocity_factor, f_next_pop,label='Applied to both halves')
    #
    # f_next_pop = []
    # for i in range(len(velocity_factor)):
    #     temp_df = electron_df.copy(deep=True)
    #     temp_df.loc[temp_df['kx [1/A]'] < 0, 'vx [m/s]'] = temp_df.loc[temp_df['kx [1/A]'] < 0, 'vx [m/s]'] * velocity_factor[i]
    #
    #     f_next_temp, f_0_temp = iterative_solver_lowfield(temp_df, scm, simplelin=simplelin2, applyscmFac=applyscmFac2)
    #     f_next_pop.append(10**-6 * 2 / Nuc / c.Vuc * np.sum(utilities.f2chi(f_next_temp, df, field)+df['k_FD'].values))
    #     f_0.append(f_0_temp)
    # plt.plot(velocity_factor, f_next_pop,label='Only applied to left-half')
    # f_next_pop = []
    # for i in range(len(velocity_factor)):
    #     temp_df = electron_df.copy(deep=True)
    #     temp_df.loc[temp_df['kx [1/A]'] > 0, 'vx [m/s]'] = temp_df.loc[temp_df['kx [1/A]'] > 0, 'vx [m/s]'] * velocity_factor[i]
    #     f_next_temp, f_0_temp = iterative_solver_lowfield(temp_df, scm, simplelin=simplelin2, applyscmFac=applyscmFac2)
    #     f_next_pop.append(10**-6 * 2 / Nuc / c.Vuc * np.sum(utilities.f2chi(f_next_temp, df, field)+df['k_FD'].values))
    #     f_0.append(f_0_temp)
    # plt.plot(velocity_factor, f_next_pop,label='Only applied to right-half')
    # plt.xlabel('Velocity Asymmetry Factor')
    # plt.ylabel('Population (cm^-3)')
    # plt.title('0.03% step deviation locked {:3E} kV/cm'.format(field*1e-5))
    # plt.legend()
    # plt.show()


# def write_iterative_solver_lowfield(outLoc,inLoc,df,simplelin2=True,applyscmFac2=False):
#     """Calls the iterative solver hard coded for solving the BTE in low field approximation and writes the single F_psi
#      solution to file.
#
#     Parameters:
#         outLoc (str): String containing the location of the directory to write the chi solutions.
#         inLoc (str): String containing the location of the directory containing the scattering matrix, assumed simple
#         linearization by default.
#         df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
#         simplelin2 (bool): Boolean that specifies whether the scattering matrix is assumed simple or canonical
#         linearization, assumed simple by default (i.e. canonical=False).
#         applyscmFac2 (bool): Boolean that specifies whether or not to apply the 2*pi squared factor.
#
#     Returns:
#         None. Just writes the F_psi solution to file. FPsi_#. #1 corresponds to low-field RTA, #2 corresponds to
#         low-field iterative.
#     """
#     nkpts = len(np.unique(df['k_inds']))
#     scm = np.memmap(inLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
#     f_next, f_0 = iterative_solver_lowfield(df, scm, simplelin=simplelin2, applyscmFac=applyscmFac2,doIt=False)
#     # np.save(outLoc +'f_' + '1', f_0)
#     # np.save(outLoc +'f_' + '2', f_next)
#     # print('f solutions written to file.')
#
#     print(2 / nkpts / c.Vuc * np.sum(utilities.f2chi(f_next, df, 10**6)+electron_df['k_FD']))


# This is the original, which is wrong because it does not define the inter_inds correctly.
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
    step_size = 0.0070675528500652425 * 1E10  # 1/Angstrom to 1/m (for 160^3)
    # step_size = 0.005654047459752398 * 1E10  # 1/Angstrom to 1/m (for 200^3)


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
            ordered_inds = subset['k_inds'].values - 1  # indices of matrix (zero indexed)
            icinds.append(ordered_inds[0] + 1)  # +1 to get the k_inds values (one indexed)
            icinds.append(ordered_inds[-1] +1)
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
        if kind % 10 == 0:
            pass
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
    # cond_num = np.linalg.cond(matrix_sc-matrix_fd)
    # sparsity = utilities.calc_sparsity(matrix_sc-matrix_fd)
    # cs = utilities.matrix_check_colsum(matrix_fd,len(kptdf))
    # print('The average absolute value of FDM column sum is {:E}'.format(np.average(np.abs(cs))))
    # print('The largest FDM column sum is {:E}'.format(cs.max()))
    #
    # cs = utilities.matrix_check_colsum(matrix_sc,len(kptdf))
    # print('The average absolute value of SCM column sum is {:E}'.format(np.average(np.abs(cs))))
    # print('The largest SCM column sum is {:E}'.format(cs.max()))

    b = (-1)*c.e*field/c.kb_joule/pp.T * np.squeeze(kptdf['vx [m/s]'] * kptdf['k_FD']) * (1 - kptdf['k_FD'])
    # chi2psi is used to give the finite difference matrix the right factors in front since substitution made
    chi2psi = np.squeeze(kptdf['k_FD'] * (1 - kptdf['k_FD']))
    invdiag = (np.diag(matrix_sc) * scmfac) ** (-1)
    x_0 = b * invdiag
    Nuc = len(kptdf)

    print('The avg abs val of f_rta is {:.3E}'.format(np.average(np.abs(x_0))))
    print('The sum over f_rta is {:.3E}'.format(np.sum(x_0)))
    print('Pop change f_rta is {:3E}'.format(2 / Nuc / c.Vuc * np.sum(x_0)))

    errpercent = 1
    counter = 0
    x_prev = x_0
    print('Starting convergence loop')
    loopstart = time.time()
    print('Original population is {:3E}'.format(utilities.calculate_density(kptdf)))
    pos_kx_loc = kptdf.loc[kptdf['kx [1/A]'] > 0].index
    neg_kx_loc = kptdf.loc[kptdf['kx [1/A]'] < 0].index

    while errpercent > convergence and counter < 150:
        # Directly make the boundary condition points zero. icinds is 1-indexed. Subtract 1.
        # x_prev[icinds - 1] = 0
        s1 = time.time()
        mvp_sc = np.matmul(matrix_sc, x_prev) * scmfac
        print('The sum over mvp_sc is {:3E}'.format(np.sum(mvp_sc)))
        print('Pop change mvp_sc is {:3E}'.format(2 / Nuc / c.Vuc * np.sum(mvp_sc)))
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
        print('The sum over offdiag_fd is {:3E}'.format(np.sum(offdiag_fd)))
        print('Pop change offdiag_fd is {:3E}'.format(2 / Nuc / c.Vuc * np.sum(offdiag_fd)))
        print('Matrix vector multiplications took {:.2f}s'.format(e1 - s1))
        print('The 2-norm of offdiag FDM part is {:.3E}'.format(np.linalg.norm(offdiag_fd)))
        print('The 2-norm of offdiag scattering part is {:.3E}'.format(np.linalg.norm(offdiag_sc)))
        x_next = x_0 + (invdiag * (offdiag_fd - offdiag_sc))
        errvecnorm = np.linalg.norm(x_next - x_prev)
        errpercent = errvecnorm / np.linalg.norm(x_prev)
        print('The sum over x_prev is {:3E}'.format(np.sum(x_prev)))
        print('Pop change x_prev is {:3E}'.format(2 / Nuc / c.Vuc * np.sum(x_prev)))
        print('The sum over x_next is {:3E}'.format(np.sum(x_next)))
        print('Pop change x_next is {:3E}'.format(2 / Nuc / c.Vuc * np.sum(x_next)))
        print('The sum over x_FDM is {:3E}'.format(np.sum(invdiag * offdiag_fd)))
        print('Pop change x_FDM is {:3E}'.format(2 / Nuc / c.Vuc * np.sum(invdiag * offdiag_fd)))
        print('The sum over x_SCM is {:3E}'.format(-1 * np.sum(invdiag * offdiag_sc)))
        print('Pop change x_SCM is {:3E}'.format(-2 / Nuc / c.Vuc * np.sum(invdiag * offdiag_sc)))
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

    print('Checking b.')
    b_check = np.dot(matrix_sc-matrix_fd,x_next)
    error = np.linalg.norm(b_check - b/scmfac)/np.linalg.norm(b/scmfac)
    print(error)

    print('Cosine similarity.')
    cosdist = distance.cosine(b_check,b/scmfac)
    cos_sim = 1-cosdist
    print(cos_sim)

    # plt.figure()
    # plt.plot(b_check,label=r'MVP $= (SCM-FDM)\chi_k$')
    # plt.plot(b/scmfac,label='Forcing = b',alpha=0.6)
    # plt.title('FDM error check E = 1 kV/cm')
    # plt.xlabel('kpt index')
    # plt.ylabel('b value (1/s)')
    # plt.legend()
    # plt.show()

    # return x_next, x_0, icinds, error, cos_sim, cond_num, sparsity
    return x_next, x_0, icinds, error, cos_sim


# def steady_state_full_drift_iterative_solver(matrix_sc, matrix_fd, kptdf, field, canonical=False, applyscmFac=False,
#                                              convergence=5E-4):
#     """Iterative solver for calculating steady BTE solution in the form of Chi using the full finite difference matrix.
#     Parameters:
#         matrix_sc (memmap): Scattering matrix in simple linearization by default..
#         matrix_fd (memmap): Finite difference matrix, generated by apply_centraldiff_matrix.
#         kptdf (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
#         field (dbl): Value of the electric field in V/m.
#         canonical (bool): Boolean that specifies whether the scattering matrix is assumed simple or canonical
#         applyscmFac (bool): Boolean that specifies whether or not to apply the 2*pi squared factor.
#         convergence (dbl): Specifies the percentage threshold for convergence.
#     Returns:
#         x_next (nparray): Numpy array containing the (hopefully) converged iterative solution as chi.
#         x_0 (nparray): Numpy array containing the RTA solution as chi.
#     """
#     print('Starting steady_state_full_drift_iterative solver for {:.3E}'.format(field))
#     if applyscmFac:
#         scmfac = (2*np.pi)**2
#         print('Applying 2 Pi-squared factor.')
#     else:
#         scmfac = 1
#     _, icinds, _, matrix_fd = apply_centraldiff_matrix(matrix_fd, kptdf, field)
#     b = (-1)*c.e*field/c.kb_joule/pp.T * np.squeeze(kptdf['vx [m/s]'] * kptdf['k_FD']) * (1 - kptdf['k_FD'])
#     # chi2psi is used to give the finite difference matrix the right factors in front since substitution made
#     chi2psi = np.squeeze(kptdf['k_FD'] * (1 - kptdf['k_FD']))
#     invdiag = (np.diag(matrix_sc) * scmfac) ** (-1)
#     x_0 = b * invdiag
#     print('The avg abs val of f_rta is {:.3E}'.format(np.average(np.abs(x_0))))
#     print('The sum over f_rta is {:.3E}'.format(np.sum(x_0)))
#     errpercent = 1
#     counter = 0
#     x_prev = x_0
#     print('Starting convergence loop')
#     loopstart = time.time()
#     while errpercent > convergence and counter < 40:
#         # Directly make the boundary condition points zero. icinds is 1-indexed. Subtract 1.
#         # x_prev[icinds - 1] = 0
#         s1 = time.time()
#         mvp_sc = np.matmul(matrix_sc, x_prev) * scmfac
#         # Remove diagonal terms from the scattering matrix multiplication (prevent double counting of diagonal term)
#         # Also include  2pi^2 factor that we believe is the conversion between radians and seconds
#         offdiag_sc = mvp_sc - (np.diag(matrix_sc) * x_prev * scmfac)
#         # There's no diagonal component of the finite difference matrix so matmul directly gives contribution
#         # If using simple linearization (chi instead of psi) then don't use chi2psi term
#         if not canonical:
#             offdiag_fd = np.matmul(matrix_fd, x_prev)
#         else:
#             offdiag_fd = np.matmul(matrix_fd, chi2psi * x_prev)
#         e1 = time.time()
#         print('Matrix vector multiplications took {:.2f}s'.format(e1 - s1))
#         print('The 2-norm of offdiag FDM part is {:.3E}'.format(np.linalg.norm(offdiag_fd)))
#         print('The 2-norm of offdiag scattering part is {:.3E}'.format(np.linalg.norm(offdiag_sc)))
#         x_next = x_0 + (invdiag * (offdiag_fd - offdiag_sc))
#         errvecnorm = np.linalg.norm(x_next - x_prev)
#         errpercent = errvecnorm / np.linalg.norm(x_prev)
#         x_prev = x_next
#         counter += 1
#         print('Iteration {:d}: Error percent is {:.3E}. Abs error norm is {:.3E}\n'
#               .format(counter, errpercent, errvecnorm))
#     loopend = time.time()
#     print('Convergence took {:.2f}s'.format(loopend - loopstart))
#     if canonical:
#         # Return chi in all cases so there's not confusion in plotting
#         print('Converting psi to chi since matrix in canonical linearization')
#         x_next = x_next * chi2psi
#         x_0 = x_0 * chi2psi
#     return x_next, x_0


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
    scm = np.memmap(inLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    error = []
    cos_sim =[]
    condnumber = []
    sparsity = []
    for EField in fieldVector:
        fdm = np.memmap(inLoc + 'finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
        # x_next, _, icinds, temp_e, temp_cos, temp_cond, temp_sparse = steady_state_full_drift_iterative_solver(scm, fdm, df, EField, canonical2, applyscmFac2, convergence2)
        x_next, _, icinds, temp_e, temp_cos = steady_state_full_drift_iterative_solver(scm, fdm, df, EField, canonical2, applyscmFac2, convergence2)
        error.append(temp_e)
        cos_sim.append(temp_cos)
        # condnumber.append(temp_cond)
        # sparsity.append(temp_sparse)
        del fdm
        np.save(outLoc + 'chi_' + '3_' + "{:.1e}".format(EField), x_next)
        print('Solution written to file for ' + "{:.1e}".format(EField))
        np.save(outLoc + 'icinds',icinds)
    plt.figure()
    plt.plot(fieldVector * 1E-5,error)
    plt.xlabel('EField (kV/cm)')
    plt.ylabel('|b-bp|/|b|')

    plt.figure()
    plt.plot(fieldVector * 1E-5,cos_sim)
    plt.xlabel('EField (kV/cm)')
    plt.ylabel('Cosine Similarity MVP & b')

    # plt.figure()
    # plt.plot(fieldVector * 1E-5,condnumber)
    # plt.xlabel('EField (kV/cm)')
    # plt.ylabel('Condition Number')
    #
    # plt.figure()
    # plt.plot(fieldVector * 1E-5,sparsity)
    # plt.xlabel('EField (kV/cm)')
    # plt.ylabel('Sparsity')

    plt.show()

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
    vd = utilities.drift_velocity(chi,df)
    invdiag = (np.diag(matrix_sc) * scmfac) ** (-1)
    g_lin = (vd-df['vx [m/s]']) * invdiag * f
    g0 = (-df['vx [m/s]']) * invdiag * df['k_FD']
    if not simplelin:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in simple linearization')
        g_lin = g_lin * psi2chi
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
    vd = utilities.drift_velocity(chi, df)
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
    while errpercent > convergence and counter < 100:
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
    scm = np.memmap(inLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    for i in range(len(fieldVector)):
        EField = fieldVector[i]
        chi = np.load(outLoc + 'chi_3_{:.1e}.npy'.format(EField))
        fdm = np.memmap(inLoc + 'finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
        g_next, g_3_johnson = eff_distr_g_iterative_solver(chi, scm, fdm, df, EField, simplelin2, applyscmFac2, convergence2)
        del fdm
        f_i = np.load(outLoc + 'f_1.npy')
        chi_1_i = utilities.f2chi(f_i, df, EField)
        g_rta, g_1_johnson = eff_distr_g_rta(chi_1_i, scm, df, simplelin2, applyscmFac2)
        np.save(outLoc + 'g_' + '1_' + "{:.1e}".format(EField), g_rta)
        np.save(outLoc + 'g_' + '3_' + "{:.1e}".format(EField), g_next)
        print('Solution written to file for ' + "{:.1e}".format(EField))
    np.save(outLoc + 'g_' + '1_johnson', g_1_johnson)
    np.save(outLoc + 'g_' + '3_johnson', g_3_johnson)


if __name__ == '__main__':
    # Right now, the functions are hardcoded to look for a scattering matrix named 'scattering_matrix_5.87_simple.mmap'
    # in the in_Loc. This can be modified later to look for just scattering_matrix_simple, or the name can be passed as
    # an argument. I kind of lean towards the argument, because it would force you to change the name each time you ran
    # with a new scattering matrix, which is probably good so we don't mess up.

    # Point to inputs and outputs
    out_Loc = pp.outputLoc
    in_Loc = pp.inputLoc

    # Read problem parameters and specify electron DataFrame
    utilities.load_electron_df(in_Loc)
    utilities.read_problem_params(in_Loc)
    electron_df = pd.read_pickle(in_Loc+'electron_df.pkl')
    electron_df = utilities.fermi_distribution(electron_df)

    # Steady state solutions
    # fields = np.array([1e1,2e1,3e1,4e1,5e1,6e1,7e1,8e1,9e1,1e2,2e2,3e2,4e2,5e2,6e2,7e2,8e2,9e2,2e3,4e3,6e3,8e3,1e4,2e4,4e4,6e4,8e4,1.1e5,1.2e5,1.3e5,1.4e5,1.5e5,1.6e5,1.7e5,1.8e5,1.9e5,2e5])
    # fields = np.array([1.6e5,1.7e5,1.8e5,1.9e5,2e5])
    # fields = np.logspace(0,5,6)
    # fields = np.array([1,10,100,1e3,1e4,1e5])
    fields = np.array([1e2,1e3,1e4,2.5e4,5e4,7.5e4,1e5,2e5,3e5])
    # fields = np.array([1e5])
    applySCMFac = pp.scmBool
    simpleLin = pp.simpleBool
    writeLowfield = True
    writeFDM = False
    writeEffective = False


    if writeLowfield:
        write_iterative_solver_lowfield(out_Loc, in_Loc, electron_df, simpleLin, applySCMFac)
        print('Low field solutions written to file as Fs.')
    if writeFDM:
        write_iterative_solver_fdm(out_Loc, in_Loc, fields, electron_df, not simpleLin, applySCMFac,5E-4)
        print('FDM solutions written to file as chis.')
    if writeEffective:
        write_iterative_solver_g(out_Loc, in_Loc, fields, electron_df, simpleLin, applySCMFac,5E-4)
        print('Effective distribution solutions written to file.')