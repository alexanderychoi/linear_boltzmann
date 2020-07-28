import numpy as np
import constants as c
import utilities
import problem_parameters as pp
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import pandas as pd
import noise_solver
import occupation_plotter
from matplotlib.font_manager import FontProperties
import preprocessing
from mpl_toolkits.mplot3d import Axes3D
import material_plotter
import occupation_solver
import time
import silicon_scratch
import numpy.linalg
from scipy.sparse import linalg
import scipy


def si_6_fdm(matrix,fullkpts_df,E, unitProjection):
    """Calculate a finite difference matrix using the specified difference stencil and apply bc. In the current version,
    it is only applied in the kx direction, but it is easy to generalize for a general axis. Only applied to the 6 equiv
    valleys of Silicon.

    Parameters:
        matrix (memmap): Scattering matrix in simple linearization.
        fullkpts_df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        E (dbl): Value of the electric field in V/m.

    Returns:
        shortslice_inds (numpyarray): Array of indices associated with the points along ky-kz lines with fewer than 5 pts.
        np.array(icinds) (numpyarray): Array of indices associated with the points where the ic is applied.
        np.array(icinds) (numpyarray): Array of indices associated with the points where the ic is applied.
        matrix (memmap): Memory-mapped array containing the modified scattering matrix, accounting for the FDM.
    """
    # Do not  flush the memmap it will overwrite consecutively.
    # Get the first and last rows since these are different because of the IC. Go through each.
    # Get the unique ky and kz values from the array for looping.
    # This is not robust and should be replaced.
    if pp.kgrid == 160:
        step_size = 0.0070675528500652425 * 1E10  # 1/Angstrom to 1/m (for 160^3)
    if pp.kgrid == 200:
        step_size = 0.005654047459752398 * 1E10  # 1/Angstrom to 1/m (for 200^3)
    if pp.kgrid == 80:
        step_size = 0.0070675528500652425*2*1E10  # 1/Angstron for 1/m (for 80^3)
    if pp.kgrid == 100:
        step_size = 0.0231 * 1e10
    print('Step size is {:.3f} inverse Angstroms'.format(step_size/1e10))
    kptdata = fullkpts_df[['k_inds', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]','energy [eV]']]
    s1_inds,s2_inds,s3_inds,s4_inds,s5_inds,s6_inds = silicon_scratch.silicon_splitter(fullkpts_df)
    valley_list = [s1_inds,s2_inds,s3_inds,s4_inds,s5_inds,s6_inds]

    xProj = unitProjection[0]
    yProj = unitProjection[1]
    zProj = unitProjection[2]

    shortslice_inds = []
    l_icinds = []
    r_icinds = []
    kx_spacing_vec = []
    if pp.fdmName == 'Hybrid Difference':
        print('Applying hybrid FDM scheme.')
    for i1 in range(len(valley_list)):
        print('Applying to {} L valley'.format(i1))
        l_df = kptdata.loc[valley_list[i1]]
        uniq_yz = np.unique(l_df[['ky [1/A]', 'kz [1/A]']].values, axis=0)
        uniq_xz = np.unique(l_df[['kx [1/A]', 'kz [1/A]']].values, axis=0)
        uniq_xy = np.unique(l_df[['kx [1/A]', 'ky [1/A]']].values, axis=0)

        # If there are too few points in a slice < 5, we want to keep track of those points
        start = time.time()
        # Loop through the unique ky and kz values in the Gamma valley
        if xProj !=0:
            print('Applying FDM along ky-kz.')
            for i in range(len(uniq_yz)):
                ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
                # Grab the "slice" of points in k space with the same ky and kz coordinate
                slice_df = l_df.loc[(l_df['ky [1/A]'] == ky) & (l_df['kz [1/A]'] == kz)]
                slice_inds = slice_df['k_inds'].values
                if len(slice_inds) > 3:
                    # Subset is the slice sorted by kx value in ascending order. The index of subset still references kptdata.
                    subset = slice_df.sort_values(by=['kx [1/A]'], ascending=True)
                    ordered_inds = subset['k_inds'].values - 1  # indices of matrix (zero indexed)
                    kx_spacing_vec.append(np.abs(np.diff(subset['kx [1/A]'].values)))
                    l_icinds.append(ordered_inds[0])  # +1 to get the k_inds values (one indexed)
                    r_icinds.append(ordered_inds[-1])
                    last = len(ordered_inds) - 1
                    slast = len(ordered_inds) - 2
                    if pp.fdmName == 'Column Preserving Central Difference':
                        # Set the "initial condition" i.e. point with most negative kx value is treated as being zero
                        # (and virtual point below)
                        matrix[ordered_inds[0], ordered_inds[1]] += 1 / (2 * step_size) * c.e * E / c.hbar_joule * xProj
                        matrix[ordered_inds[1], ordered_inds[2]] += 1 / (2 * step_size) * c.e * E / c.hbar_joule * xProj
                        # Set the other "boundary condition" i.e. point with most positive kx value is treated as being zero
                        # (and virtual point above)
                        matrix[ordered_inds[last], ordered_inds[slast]] += -1 * 1 / (2 * step_size) * c.e * E / c.hbar_joule * xProj
                        matrix[ordered_inds[slast], ordered_inds[slast - 1]] += -1 * 1 / (2 * step_size) *c.e*E/c.hbar_joule * xProj
                        # Set the value of all other points in the slice
                        inter_inds = ordered_inds[2:slast]
                        inter_inds_up = ordered_inds[3:last]
                        inter_inds_down = ordered_inds[1:slast - 1]
                        matrix[inter_inds, inter_inds_up] += 1 / (2 * step_size) * c.e * E / c.hbar_joule  * xProj
                        matrix[inter_inds, inter_inds_down] += -1 * 1 / (2 * step_size) * c.e * E / c.hbar_joule * xProj
                    if pp.fdmName == 'Backwards Difference':
                        # Set the "initial condition" i.e. the point with the most negative kx value has virtual point below
                        # that is assumed to be zero
                        inter_inds = ordered_inds[1:last + 1]
                        inter_inds_down = ordered_inds[0:last]
                        matrix[ordered_inds, ordered_inds] += 1 / (step_size) * c.e * E / c.hbar_joule
                        matrix[inter_inds, inter_inds_down] += -1 * 1 / (step_size) * c.e * E / c.hbar_joule
                else:
                    shortslice_inds.append(slice_inds)

        if yProj != 0:
            print('Applying FDM along kx-kz.')
            for i in range(len(uniq_xz)):
                kx, kz = uniq_xz[i, 0], uniq_xz[i, 1]
                # Grab the "slice" of points in k space with the same kx and kz coordinate
                slice_df = l_df.loc[(l_df['kx [1/A]'] == kx) & (l_df['kz [1/A]'] == kz)]
                slice_inds = slice_df['k_inds'].values
                if len(slice_inds) > 3:
                    # Subset is the slice sorted by ky value in ascending order. The index of subset still references kptdata.
                    subset = slice_df.sort_values(by=['ky [1/A]'], ascending=True)
                    ordered_inds = subset['k_inds'].values - 1  # indices of matrix (zero indexed)
                    l_icinds.append(ordered_inds[0])  # +1 to get the k_inds values (one indexed)
                    r_icinds.append(ordered_inds[-1])
                    last = len(ordered_inds) - 1
                    slast = len(ordered_inds) - 2
                    if pp.fdmName == 'Column Preserving Central Difference':
                        # Set the "initial condition" i.e. point with most negative kx value is treated as being zero
                        # (and virtual point below)
                        matrix[ordered_inds[0], ordered_inds[1]] += 1 / (2 * step_size) * c.e * E / c.hbar_joule * yProj
                        matrix[ordered_inds[1], ordered_inds[2]] += 1 / (2 * step_size) * c.e * E / c.hbar_joule * yProj
                        # Set the other "boundary condition" i.e. point with most positive kx value is treated as being zero
                        # (and virtual point above)
                        matrix[ordered_inds[last], ordered_inds[slast]] += -1 * 1 / (
                                    2 * step_size) * c.e * E / c.hbar_joule * yProj
                        matrix[ordered_inds[slast], ordered_inds[slast - 1]] += -1 * 1 / (
                                    2 * step_size) * c.e * E / c.hbar_joule * yProj
                        # Set the value of all other points in the slice
                        inter_inds = ordered_inds[2:slast]
                        inter_inds_up = ordered_inds[3:last]
                        inter_inds_down = ordered_inds[1:slast - 1]
                        matrix[inter_inds, inter_inds_up] += 1 / (2 * step_size) * c.e * E / c.hbar_joule * yProj
                        matrix[inter_inds, inter_inds_down] += -1 * 1 / (2 * step_size) * c.e * E / c.hbar_joule * yProj
                    if pp.fdmName == 'Backwards Difference':
                        # Set the "initial condition" i.e. the point with the most negative kx value has virtual point below
                        # that is assumed to be zero
                        inter_inds = ordered_inds[1:last + 1]
                        inter_inds_down = ordered_inds[0:last]
                        matrix[ordered_inds, ordered_inds] += 1 / (step_size) * c.e * E / c.hbar_joule
                        matrix[inter_inds, inter_inds_down] += -1 * 1 / (step_size) * c.e * E / c.hbar_joule
                else:
                    shortslice_inds.append(slice_inds)

        if zProj != 0:
            print('Applying FDM along kx-ky.')
            for i in range(len(uniq_xy)):
                kind = i + 1
                kx, ky = uniq_xy[i, 0], uniq_xy[i, 1]
                # Grab the "slice" of points in k space with the same kx and kz coordinate
                slice_df = l_df.loc[(l_df['kx [1/A]'] == kx) & (l_df['ky [1/A]'] == ky)]
                slice_inds = slice_df['k_inds'].values
                if len(slice_inds) > 3:
                    # Subset is the slice sorted by kz value in ascending order. The index of subset still references kptdata.
                    subset = slice_df.sort_values(by=['kz [1/A]'], ascending=True)
                    ordered_inds = subset['k_inds'].values - 1  # indices of matrix (zero indexed)
                    l_icinds.append(ordered_inds[0])  # +1 to get the k_inds values (one indexed)
                    r_icinds.append(ordered_inds[-1])
                    last = len(ordered_inds) - 1
                    slast = len(ordered_inds) - 2
                    if pp.fdmName == 'Column Preserving Central Difference':
                        # Set the "initial condition" i.e. point with most negative kx value is treated as being zero
                        # (and virtual point below)
                        matrix[ordered_inds[0], ordered_inds[1]] += 1 / (2 * step_size) * c.e * E / c.hbar_joule * zProj
                        matrix[ordered_inds[1], ordered_inds[2]] += 1 / (2 * step_size) * c.e * E / c.hbar_joule * zProj
                        # Set the other "boundary condition" i.e. point with most positive kx value is treated as being zero
                        # (and virtual point above)
                        matrix[ordered_inds[last], ordered_inds[slast]] += -1 * 1 / (
                                    2 * step_size) * c.e * E / c.hbar_joule * zProj
                        matrix[ordered_inds[slast], ordered_inds[slast - 1]] += -1 * 1 / (
                                    2 * step_size) * c.e * E / c.hbar_joule * zProj
                        # Set the value of all other points in the slice
                        inter_inds = ordered_inds[2:slast]
                        inter_inds_up = ordered_inds[3:last]
                        inter_inds_down = ordered_inds[1:slast - 1]
                        matrix[inter_inds, inter_inds_up] += 1 / (2 * step_size) * c.e * E / c.hbar_joule * zProj
                        matrix[inter_inds, inter_inds_down] += -1 * 1 / (2 * step_size) * c.e * E / c.hbar_joule * zProj
                    if pp.fdmName == 'Backwards Difference':
                        # Set the "initial condition" i.e. the point with the most negative kx value has virtual point below
                        # that is assumed to be zero
                        inter_inds = ordered_inds[1:last + 1]
                        inter_inds_down = ordered_inds[0:last]
                        matrix[ordered_inds, ordered_inds] += 1 / (step_size) * c.e * E / c.hbar_joule
                        matrix[inter_inds, inter_inds_down] += -1 * 1 / (step_size) * c.e * E / c.hbar_joule
                else:
                    shortslice_inds.append(slice_inds)
    print('Scattering matrix modified to incorporate central difference contribution.')
    shortslice_inds = np.concatenate(shortslice_inds, axis=0)
    kx_spacing_vec = np.concatenate(kx_spacing_vec, axis=0)

    print('Not applied to {:d} L valley points because fewer than 4 points on the slice.'.format(len(shortslice_inds)))
    print('This represents {:1f} % of points in the particular valley.'.format(len(shortslice_inds) / len(fullkpts_df) * 100))
    print('Finite difference applied to L valleys.')
    end = time.time()
    print('Finite difference generation took {:.2f}s'.format(end - start))

    return shortslice_inds, np.array(l_icinds), np.array(r_icinds), matrix, kx_spacing_vec


def si_inverse_relaxation_operator(b, matrix_sc, matrix_fd, kptdf, field, unitProjection, freq):
    """Generalized minimal residual solver for calculating the solution to the matix equation Ax = b, where b is to be
    specified and A is the transient relaxation operator (time + drift + scm).
    Parameters:
        b (nparray) : The forcing for the matrix equation.
        matrix_sc (memmap): Scattering matrix in simple linearization by default..
        matrix_fd (memmap): Finite difference matrix, generated by apply_centraldiff_matrix.
        kptdf (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        field (dbl): Value of the electric field in V/m.
        freq (dbl): Frequency in GHz to be used for the transient solution
    Returns:
        x_next (nparray): Numpy array containing the (hopefully) converged iterative solution as chi.
        x_0 (nparray): Numpy array containing the RTA solution as chi.
        counter.niter (dbl): Number of iterations to reach desired relative convergence
    """
    counter = occupation_solver.gmres_counter()
    print('Starting inverse relaxation solver for {:.3E} and {:.3E} GHz'.format(field,freq))
    if pp.scmBool:
        scmfac = pp.scmVal
        print('Applying correction factor to the scattering matrix.')
    else:
        scmfac = 1
        print('Not applying correction factor to the scattering matrix.')
    _,_,_,matrix_fd,_ = si_6_fdm(matrix_fd, kptdf, field,unitProjection)

    loopstart = time.time()
    chi2psi = np.squeeze(kptdf['k_FD'] * (1 - kptdf['k_FD']))
    invdiag = (np.diag(matrix_sc) * scmfac) ** (-1)
    x_0 = b * invdiag
    tau_0 = (np.sum(np.diag(matrix_sc)*kptdf['k_FD'])/np.sum(kptdf['k_FD']))**(-1)

    x_smrta = b * np.ones(len(kptdf))*tau_0
    if freq > 0:
        freq_matrix = np.diag(np.ones(len(kptdf))*1j*10**9*2*np.pi*freq)  # Positive quantity
        print('Starting GMRES solver.')
        x_next, criteria = linalg.gmres(freq_matrix+matrix_sc*scmfac-matrix_fd, b,x0=x_0,tol=pp.relConvergence,
                                        callback=counter,atol=pp.absConvergence)
    if freq == 0:
        print('Starting GMRES solver.')
        x_next, criteria = linalg.gmres(matrix_sc*scmfac-matrix_fd, b,x0=x_0,tol=pp.relConvergence,
                                        callback=counter,atol=pp.absConvergence)
        freq_matrix = np.diag(np.zeros(len(kptdf)))
    print('GMRES convergence criteria: {:3E}'.format(criteria))
    # The following step is the calculation of the relative residual, which involves another MVP. This adds expense. If
    # we're confident in the convergence, we can omit this check to increase speed.
    if pp.verboseError:
        b_check = np.dot(freq_matrix+matrix_sc*scmfac-matrix_fd,x_next)
        error = np.linalg.norm(b_check - b)/np.linalg.norm(b)
        print('Norm of b is {:3E}'.format(np.linalg.norm(b)))
        print('Absolute residual error is {:3E}'.format(np.linalg.norm(b_check-b)))
        print('Relative residual error is {:3E}'.format(error))
    else:
        error = 0
        print('Error not stored.')
    loopend = time.time()
    print('Convergence took {:.2f}s'.format(loopend - loopstart))
    if not pp.simpleBool:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in canonical linearization')
        x_next = x_next * chi2psi
        x_0 = x_0 * chi2psi
        x_smrta = x_smrta * chi2psi
    return x_next, x_smrta, error, counter.niter


def write_steady(fieldVector, df, unitProjection):
    """Calls the GMRES solver hard coded for solving the BTE with full FDM and writes the chis to file. Also calculates
    low-field and RTA chis.
    Parameters:
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
    Returns:
        None. Just writes the chi solutions to file. chi_#_EField_freq. #1 corresponds to RTA, #2 corresponds to
        low-field, #3 corresponds to full finite-difference.
    """
    nkpts = len(np.unique(df['k_inds']))
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    for ee in fieldVector:
        fdm = np.memmap(pp.inputLoc + '/finite_difference_matrix2.mmap', dtype='float64', mode='w+',
                        shape=(nkpts, nkpts))
        xProj = unitProjection[0]  # Projection of the FDM along kx
        yProj = unitProjection[1]  # Projection of the FDM along ky
        zProj = unitProjection[2]  # Projection of the FDM along kz

        b = (-1) * c.e * ee / c.kb_joule / pp.T * np.squeeze((xProj*df['vx [m/s]']+yProj*df['vy [m/s]']+zProj*df['vz [m/s]']) * df['k_FD']) * (1 - df['k_FD'])
        chi_3_i,_,_,_ = si_inverse_relaxation_operator(b, scm, fdm, df, ee, unitProjection, 0)
        np.save(
            pp.outputLoc + 'Steady/' + 'chi_3_' + "E_{:.1e}".format(ee) + '_FD_{:.1f}_{:.1f}_{:.1f}'.format(xProj,
                                                                                                               yProj,
                                                                                                               zProj),
            chi_3_i)
        del fdm


def write_correlation(fieldVector,df,freqVector,unitProjection):
    """Calls the GMRES solver hard coded for solving the effective BTE w/FDM for the Gamma-Gamma population
    autocorrelation for two-valleys and writes to file.
    Parameters:
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        freq (dbl): Frequency in GHz to be used for the transient solution
    Returns:
        None. Just writes the chi solutions to file. chi_#_EField. #1 corresponds to low-field RTA, #2 corresponds to
        low-field iterative, #3 corresponds to full finite-difference iterative.
    """
    nkpts = len(np.unique(df['k_inds']))
    xProj = unitProjection[0]
    yProj = unitProjection[1]
    zProj = unitProjection[2]
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    utilities.check_matrix_properties(scm)
    for freq in freqVector:
        for ee in fieldVector:
            fdm = np.memmap(pp.inputLoc + '/finite_difference_matrix2.mmap', dtype='float64', mode='w+',
                            shape=(nkpts, nkpts))
            chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}".format(ee) + '_FD_{:.1f}_{:.1f}_{:.1f}.npy'.format(xProj, yProj, zProj))
            f0 = df['k_FD'].values
            f = chi_3_i + f0
            Vd_x, Vd_y, Vd_z = silicon_scratch.projected_drift(chi_3_i, df, unitProjection)
            b = (-1) * (xProj*(df['vx [m/s]'] - Vd_x) + yProj*(df['vy [m/s]'] - Vd_y) + zProj*(df['vz [m/s]'] - Vd_z)) * f
            corr_3_long, _, _, _ = si_inverse_relaxation_operator(b, scm, fdm, df, ee, unitProjection, freq)
            np.save(pp.outputLoc + 'SB_Density/' + 'corr_long_' + '3_' + "f_{:.1e}_E_{:.1e}".format(freq,ee) +'_FD_{:.1f}_{:.1f}_{:.1f}'.format(xProj,yProj,zProj),corr_3_long)
            del fdm
            print('Transient solution written to file for ' + "{:.1e} V/m and {:.1e} GHz".format(ee,freq))
            print('\n \n')


def density(chi, EField,df,freq,unitProjection):
    f0 = df['k_FD'].values
    f = chi + f0
    Nuc = pp.kgrid ** 3
    xProj = unitProjection[0]
    yProj = unitProjection[1]
    zProj = unitProjection[2]


    print(unitProjection)
    conductivity_long = 2 * c.e / (Nuc * c.Vuc * EField) * np.sum(xProj*chi * df['vx [m/s]']+yProj*chi * df['vy [m/s]']+zProj*chi * df['vz [m/s]'])

    n = utilities.calculate_noneq_density(chi,df)
    mobility = conductivity_long/c.e/n
    print('Mobility is {:3f} cm^2/(V-s)'.format(mobility*100**2))

    corr_long = np.load(pp.outputLoc + 'SB_Density/' + 'corr_long_' + '3_' + "f_{:.1e}_E_{:.1e}".format(freq,EField) +'_FD_{:.1f}_{:.1f}_{:.1f}.npy'.format(xProj,yProj,zProj))

    prefactor = (c.e / c.Vuc / Nuc) ** 2
    S_long = 8*np.real(prefactor*np.sum(corr_long*(xProj*df['vx [m/s]']+yProj*df['vy [m/s]']+zProj*df['vz [m/s]'])))
    return S_long, conductivity_long


def plot_density(fieldVector,freqVector,df,unitProjection):
    for freq in freqVector:
        S_long_vector = []
        cond_long_vector = []
        xProj = unitProjection[0]
        yProj = unitProjection[1]
        zProj = unitProjection[2]
        fielddirStr = '< %.2f %.2f %.2f >' % (xProj, yProj, zProj)
        for ee in fieldVector:
            chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}".format(ee) + '_FD_{:.1f}_{:.1f}_{:.1f}.npy'.format(xProj, yProj, zProj))
            S_long, cond_long = density(chi_3_i, ee,df,freq,unitProjection)
            S_long_vector.append(S_long)
            cond_long_vector.append(cond_long)
            kvcm = np.array(fieldVector) * 1e-5

        Nuc = pp.kgrid ** 3
        fig, ax = plt.subplots()
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = '\n'.join((r'$f = %.1f GHz $' % (freq,), pp.fdmName,fielddirStr))
        ax.plot(kvcm, S_long_vector, label=r'$S^{long}$')
        ax.axhline(np.array(cond_long_vector[0]) * 4 * c.kb_joule * pp.T / c.Vuc / Nuc, linestyle='--',
                    label=r'$\frac{4 kT \sigma_{xx}^{eq}}{V_0 N_0}$')
        plt.legend()
        plt.xlabel('Field [kV/cm]')
        plt.ylabel('Spectral Density [A^2/m^4/Hz]')
        plt.title(pp.title_str)
        ax.text(0.55, 0.9, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)

    S_xx_vector = []
    for freq in freqVector:
        plotfield = fieldVector[-3]
        chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}".format(plotfield) + '_FD_{:.1f}_{:.1f}_{:.1f}.npy'.format(xProj, yProj, zProj))
        S_xx, conductivity_xx = density(chi_3_i, plotfield, df, freq,unitProjection)
        S_xx_vector.append(S_xx)
        Nuc = pp.kgrid ** 3
        print('freq')
    fig, ax = plt.subplots()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join(
        (r'$f = %.1f GHz \, \, (100) $' % (freq,), pp.fdmName, r'$E = kV/cm%.1f$' % (plotfield / 1e5,)))
    ax.plot(freqVector, S_xx_vector, label=r'$S^{xx}$')
    plt.legend()
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Spectral Density [A^2/m^4/Hz]')
    plt.title(pp.title_str)
    ax.text(0.55, 0.9, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)


def plot_steady_transport_moments(df,fieldVector,unitProjection):
    """Takes chi solutions which are already calculated and plots transport moments: average energy, drift velocity,
    carrier population, carrier mobility
    Parameters:
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        freq (dbl): Frequency in GHz to be used for the transient solution
    Returns:
        Nothing. Just the plots.
    """
    vd_2, meanE_2, Vd_x2, Vd_y2, Vd_z2, nl_2 = ([] for i in range(6))

    s1_inds,s2_inds,s3_inds,s4_inds,s5_inds,s6_inds = silicon_scratch.silicon_splitter(df)
    n2_s1 = []
    n2_s2 = []
    n2_s3 = []
    n2 = []
    xProj = unitProjection[0]
    yProj = unitProjection[1]
    zProj = unitProjection[2]
    fielddirStr = '< %.2f %.2f %.2f >' % (xProj,yProj,zProj)

    denom = []
    thermal_energy = utilities.mean_energy(np.zeros(len(df)), df)

    mom_RT = []
    nkpts = len(df)
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    if pp.simpleBool:
        rates = (-1) * np.diag(scm) * pp.scmVal * 1E-12

    for ee in fieldVector:
        chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}".format(ee) +'_FD_{:.1f}_{:.1f}_{:.1f}.npy'.format(xProj,yProj,zProj))

        mom_RT.append(np.sum((df['k_FD']+chi_3_i) * rates ** (-1)) / np.sum(df['k_FD']))
        vd_2.append(utilities.mean_velocity(chi_3_i,df))
        meanE_2.append(utilities.mean_energy(chi_3_i,df))
        vdx2,vdy2,vdz2 = silicon_scratch.projected_drift(chi_3_i,df,unitProjection)
        Vd_x2.append(vdx2)
        Vd_y2.append(vdy2)
        Vd_z2.append(vdz2)

        denom.append((xProj*vdx2+yProj*vdy2+zProj*vdz2)*ee)

        n2_s1.append(utilities.calc_popinds(chi_3_i,df,s1_inds))
        n2_s2.append(utilities.calc_popinds(chi_3_i,df,s2_inds))
        n2_s3.append(utilities.calc_popinds(chi_3_i,df,s3_inds))
        n2.append(utilities.calculate_noneq_density(chi_3_i,df))


    kvcm = np.array(fieldVector)*1e-5
    excessEnergy = np.array(meanE_2) - thermal_energy
    energyRT = excessEnergy/np.array(denom)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


    fig, ax = plt.subplots()
    plt.plot(kvcm,np.array(meanE_2) - thermal_energy)
    plt.ylabel('Excess Energy [eV]')
    plt.xlabel('Efield [kV/cm]')
    textstr = fielddirStr
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.title(pp.title_str)

    fig, ax = plt.subplots()
    ax.plot(kvcm,energyRT*1e12)
    plt.ylabel('Energy RT [ps]')
    plt.xlabel('Efield [kV/cm]')
    textstr = fielddirStr
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.title(pp.title_str)

    fig, ax = plt.subplots()
    ax.plot(kvcm,c.e*np.array(denom),'o-', linewidth=2)
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'eE*vd [W]')
    plt.title(pp.title_str)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)

    fig, ax = plt.subplots()
    plt.plot(kvcm,np.array(mom_RT),'o-', linewidth=2, label='FDM')
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Momentum Relaxation Time [ps]')
    plt.title(pp.title_str)
    textstr = fielddirStr
    ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.title(pp.title_str)

    fig, ax = plt.subplots()
    plt.plot(kvcm,Vd_x2,label='kx')
    plt.plot(kvcm,Vd_y2,label='ky')
    plt.plot(kvcm,Vd_z2,label='kz')
    plt.plot(kvcm,np.sqrt(np.array(Vd_x2)**2+np.array(Vd_y2)**2+np.array(Vd_z2)**2),label = 'Total')
    plt.legend(loc='lower right')
    plt.ylabel('Drift velocity [m/s]')
    plt.xlabel('Efield [kV/cm]')
    textstr = fielddirStr
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.title(pp.title_str)


    fig, ax = plt.subplots()
    plt.plot(kvcm,n2_s1,label='S1')
    plt.plot(kvcm,n2_s2,label='S2')
    plt.plot(kvcm,n2_s3,label='S3')
    # plt.plot(kvcm,n2)
    plt.xlabel('Efield [kV/cm]')
    plt.ylabel('Carrier Density [m^-3]')
    textstr = fielddirStr
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.title(pp.title_str)
    plt.ylim([0,1.2*np.max(n2_s1)])
    plt.legend(loc='lower left')



if __name__ == '__main__':
    fields = pp.fieldVector
    freqs = pp.freqVector
    unitProjection = utilities.cartesian_projection(pp.fieldDirection)
    print(unitProjection)
    # preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)
    df = electron_df.copy(deep=True)

    # It's important to coarsen the description of the kx since the point shifting doens't work exactly
    df['kx [1/A]'] = np.around(df['kx [1/A]'].values,4)
    df['ky [1/A]'] = np.around(df['ky [1/A]'].values,4)
    df['kz [1/A]'] = np.around(df['kz [1/A]'].values,4)
    nkpts = len(df)
    # fdm = np.memmap(pp.inputLoc + '/finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
    # shortslice_inds, left_inds, right_inds, matrix, kx_spacing = si_6_fdm(fdm, df, 1,unitProjection)

    # plt.figure()
    # plt.hist(kx_spacing, bins='auto')


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = df['kx [1/A]'].values
    # y = df['ky [1/A]'].values
    # z = df['kz [1/A]'].values
    # ax.scatter(x[left_inds], y[left_inds], z[left_inds], color='black', s=1.5)

    # write_steady(fields, df, unitProjection)
    plot_steady_transport_moments(df, fields, unitProjection)
    # write_correlation(fields, df, freqs, unitProjection)
    plot_density(fields, freqs, df, unitProjection)

    xProj = unitProjection[0]
    yProj = unitProjection[1]
    zProj = unitProjection[2]

    fielddirStr = '< %.2f %.2f %.2f >' % (xProj,yProj,zProj)


    s1_inds,s2_inds,s3_inds,s4_inds,s5_inds,s6_inds = silicon_scratch.silicon_splitter(df)
    plt.figure()
    for ee in fields:
        chi_3 = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}".format(ee) +'_FD_{:.1f}_{:.1f}_{:.1f}.npy'.format(xProj,yProj,zProj))
        chi_3_s1 = chi_3[s1_inds]
        silicon_scratch.momentum_distribution_kde(chi_3_s1, df.loc[s1_inds].reset_index(), ee, fielddirStr + ' S1 Valley')

    plt.figure()
    for ee in fields:
        chi_3 = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}".format(ee) +'_FD_{:.1f}_{:.1f}_{:.1f}.npy'.format(xProj,yProj,zProj))
        chi_3_s2 = chi_3[s2_inds]
        silicon_scratch.momentum_distribution_kde(chi_3_s2, df.loc[s2_inds].reset_index(), ee, fielddirStr + ' S2 Valley')

    plt.figure()
    for ee in fields:
        chi_3 = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}".format(ee) +'_FD_{:.1f}_{:.1f}_{:.1f}.npy'.format(xProj,yProj,zProj))
        chi_3_s3 = chi_3[s3_inds]
        silicon_scratch.momentum_distribution_kde(chi_3_s3, df.loc[s3_inds].reset_index(), ee, fielddirStr + ' S3 Valley')

    plt.show()

