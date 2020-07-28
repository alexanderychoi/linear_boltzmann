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
import numpy.linalg
from scipy.sparse import linalg
import scipy


def steady_low_field(df, scm):
    """GMRES solver hard coded for solving the BTE in low field approximation. Sign convention by Wu Li. Returns f,
    which is equal to chi/(eE/kT).
    Parameters:
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        scm (memmap): Memory-mapped array containing the scattering matrix, assumed simple linearization by default.
    Returns:
        f_next (nparray): Numpy array containing the (hopefully) converged GMRES solution as psi/(eE/kT).
        f_0 (nparray): Numpy array containing the RTA solution as psi_0/(eE/kT).
    """
    counter = occupation_solver.gmres_counter()
    print('Starting steady BTE low-field occupancy solver')
    if pp.scmBool:
        scmfac = pp.scmVal
        print('Applying correction factor to the scattering matrix.')
    else:
        scmfac = 1
        print('Not applying correction factor to the scattering matrix.')
    loopstart = time.time()
    invdiag = (np.diag(scm) * scmfac) ** (-1)
    unitProjection = utilities.cartesian_projection(pp.fieldDirection)
    xProj = unitProjection[0]  # Projection of the FDM along kx
    yProj = unitProjection[1]  # Projection of the FDM along ky
    zProj = unitProjection[2]  # Projection of the FDM along kz

    b = (-1) * np.squeeze((df['vx [m/s]']*xProj + df['vy [m/s]']*yProj +df['vz [m/s]']*zProj) * df['k_FD'] * (1 - df['k_FD']))
    f_0 = b * invdiag
    f_next, criteria = linalg.gmres(scm*scmfac, b, x0=f_0, tol=pp.relConvergence, atol=pp.absConvergence,
                                    callback=counter)
    print('GMRES convergence criteria: {:3E}'.format(criteria))
    if pp.verboseError:
        b_check = np.dot(scm*scmfac,f_next)
        error = np.linalg.norm(b_check - b)/np.linalg.norm(b)
        print('Norm of b is {:3E}'.format(np.linalg.norm(b)))
        print('Absolute residual error is {:3E}'.format(np.linalg.norm(b_check-b)))
        print('Relative residual error is {:3E}'.format(error))
    else:
        error = 0
        print('Error not stored.')
    loopend = time.time()
    print('Convergence took {:.2f}s'.format(loopend - loopstart))
    if pp.simpleBool:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in simple linearization')
        chi2psi = np.squeeze(df['k_FD'] * (1 - df['k_FD']))
        f_next = f_next / chi2psi
        f_0 = f_0 / chi2psi
    return f_next, f_0, error, counter.niter


def silicon_splitter(df):
    kx_norm = df['kx [1/A]'] / (2 * np.pi / c.alat)
    ky_norm = df['ky [1/A]'] / (2 * np.pi / c.alat)
    kz_norm = df['kz [1/A]'] / (2 * np.pi / c.alat)

    sz = 0.3
    x = df['kx [1/A]'].values / (2 * np.pi / c.alat)
    y = df['ky [1/A]'].values / (2 * np.pi / c.alat)
    z = df['kz [1/A]'].values / (2 * np.pi / c.alat)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, label='S1', s=sz,color='red')

    s1_inds = np.array(kx_norm > 0.5)
    s2_inds = np.array(kx_norm < -0.5)
    s3_inds = np.array(ky_norm > 0.5)
    s4_inds = np.array(ky_norm < -0.5)
    s5_inds = np.array(kz_norm > 0.5)
    s6_inds = np.array(kz_norm < -0.5)

    return s1_inds,s2_inds,s3_inds,s4_inds,s5_inds,s6_inds


def plot_silicon_split(df):
    s1_inds,s2_inds,s3_inds,s4_inds,s5_inds,s6_inds = silicon_splitter(df)
    sz = 0.3
    x = df['kx [1/A]'].values / (2 * np.pi / c.alat)
    y = df['ky [1/A]'].values / (2 * np.pi / c.alat)
    z = df['kz [1/A]'].values / (2 * np.pi / c.alat)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[s1_inds], y[s1_inds], z[s1_inds], label='S1', s=sz,color='red')
    ax.scatter(x[s2_inds], y[s2_inds], z[s2_inds], label='S2', s=sz,color='blue')
    ax.scatter(x[s3_inds], y[s3_inds], z[s3_inds], label='S3', s=sz,color='green')
    ax.scatter(x[s4_inds], y[s4_inds], z[s4_inds], label='S4', s=sz,color='yellow')
    ax.scatter(x[s5_inds], y[s5_inds], z[s5_inds], label='S5', s=sz,color='purple')
    ax.scatter(x[s6_inds], y[s6_inds], z[s6_inds], label='S6', s=sz,color='black')

    ax.set_xlabel(r'$kx/2\pi a$')
    ax.set_ylabel(r'$ky/2\pi a$')
    ax.set_zlabel(r'$kz/2\pi a$')


def write_steady(fieldVector, df):
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
    f_next, f_0,_,_ = steady_low_field(df, scm)
    unitProjection = utilities.cartesian_projection(pp.fieldDirection)
    xProj = unitProjection[0]  # Projection of the FDM along kx
    yProj = unitProjection[1]  # Projection of the FDM along ky
    zProj = unitProjection[2]  # Projection of the FDM along kz

    np.save(pp.outputLoc + 'Steady/' + 'f_1_FD_{:.1f}_{:.1f}_{:.1f}'.format(xProj,yProj,zProj),f_0)
    np.save(pp.outputLoc + 'Steady/' + 'f_2_FD_{:.1f}_{:.1f}_{:.1f}'.format(xProj,yProj,zProj),f_next)
    for ee in fieldVector:
        np.save(pp.outputLoc + 'Steady/' + 'chi_' + '2_' + "E_{:.1e}".format(ee)+'_FD_{:.1f}_{:.1f}_{:.1f}'.format(xProj,yProj,zProj), utilities.f2chi(f_next, df, ee))
        np.save(pp.outputLoc + 'Steady/' + 'chi_' + '1_' + "E_{:.1e}".format(ee)+'_FD_{:.1f}_{:.1f}_{:.1f}'.format(xProj,yProj,zProj)
                , utilities.f2chi(f_0, df, ee))


def momentum_distribution_kde(chi, df,ee,title=[]):
    """Takes chi solutions which are already calculated and plots the KDE of the distribution in velocity space
    Parameters:
        chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in chi form.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        title (str): String containing the desired name of the plot
    Returns:
        Nothing. Just the plots.
    """
    mom = df['kx [1/A]']
    npts = 600  # number of points in the KDE
    kdist = np.zeros(npts)
    kdist_tot = np.zeros(npts)
    kdist_f0 = np.zeros(npts)
    # Need to define the energy range that I'm doing integration over
    # en_axis = np.linspace(enk.min(), enk.min() + 0.4, npts)
    k_ax = np.linspace(mom.min(), mom.max(), npts)
    dx = (k_ax.max() - k_ax.min()) / npts
    f0 = np.squeeze(df['k_FD'].values)
    if pp.kgrid == 200:
        spread = 22 * dx # For 200^3
    if pp.kgrid == 160:
        spread = 25 * dx  # For 160^3
    if pp.kgrid == 100:
        spread = 35 * dx  # For 100^3
    if pp.kgrid == 80:
        spread = 70 *dx  # For 80^3
    def gaussian(x, mu, sigma=spread):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)
    for k in range(len(chi)):
        istart = int(np.maximum(np.floor((mom[k] - k_ax[0]) / dx) - (4 * spread / dx), 0))
        iend = int(np.minimum(np.floor((mom[k] - k_ax[0]) / dx) + (4 * spread / dx), npts - 1))
        kdist_tot[istart:iend] += (chi[k] + f0[k]) * gaussian(k_ax[istart:iend], mom[k])
        kdist_f0[istart:iend] += f0[k] * gaussian(k_ax[istart:iend], mom[k])
        kdist[istart:iend] += chi[k] * gaussian(k_ax[istart:iend], mom[k])

    ax = plt.axes([0.18, 0.15, 0.76, 0.76])
    ax.plot(k_ax, [0] * len(k_ax), 'k')
    # ax.plot(k_ax, kdist_f0, '--', linewidth=2, label='Equilbrium')
    ax.plot(k_ax, kdist_tot, linewidth=2, label='{:.3f} kV/cm'.format(ee/1e5))
    ax.fill(k_ax, kdist, '--', linewidth=2, color='Red')
    ax.set_xlabel('kx [1/A]')
    ax.set_ylabel(r'Occupation [arb.]')
    plt.legend()
    if title:
        plt.title(title, fontsize=8)


def plot_mom_KDEs(fieldVector, df, fieldDirection):
    """Wrapper script for velocity_distribution_kde. Can do for the various solution schemes saved to file.
        Parameters:
        outLoc (str): String containing the location of the directory to write the chi solutions and ready steady state chis.
        field (dbl): the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        Nothing. Just the plots.
    """
    s1_inds,s2_inds,s3_inds,s4_inds,s5_inds,s6_inds = silicon_splitter(df)
    xProj = fieldDirection[0]
    yProj = fieldDirection[1]
    zProj = fieldDirection[2]

    plt.figure()
    for ee in fieldVector:
        chi_2 = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '2_' + "E_{:.1e}".format(ee) +'_FD_{:.1f}_{:.1f}_{:.1f}.npy'.format(xProj,yProj,zProj))
        chi_2_s1 = chi_2[s1_inds]
        momentum_distribution_kde(chi_2_s1, df.loc[s1_inds].reset_index(), ee)

    plt.figure()
    for ee in fieldVector:
        chi_2 = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '2_' + "E_{:.1e}".format(ee) +'_FD_{:.1f}_{:.1f}_{:.1f}.npy'.format(xProj,yProj,zProj))
        chi_2_s2 = chi_2[s2_inds]
        momentum_distribution_kde(chi_2_s2, df.loc[s2_inds].reset_index(), ee)

    plt.figure()
    for ee in fieldVector:
        chi_2 = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '2_' + "E_{:.1e}".format(ee) +'_FD_{:.1f}_{:.1f}_{:.1f}.npy'.format(xProj,yProj,zProj))
        chi_2_s3 = chi_2[s3_inds]
        momentum_distribution_kde(chi_2_s3, df.loc[s3_inds].reset_index(), ee)

def projected_drift(chi,df,unitProjection):
    xProj = unitProjection[0]
    yProj = unitProjection[1]
    zProj = unitProjection[2]

    f0 = df['k_FD'].values

    f = chi + f0

    drift = np.sum(xProj * f * df['vx [m/s]'] + yProj * f * df['vy [m/s]'] + zProj * f * df['vz [m/s]'])/ np.sum(f0)

    Vd_x = xProj * drift
    Vd_y = yProj * drift
    Vd_z = zProj * drift
    return Vd_x, Vd_y, Vd_z


def longitudinal_lowfield_correlation(chi2, matrix_sc, df, field, freq, unitProjection):
    """Generalized minimal residual solver for calculating transient Gamma vd-vd autocorrelation using the full
    finite difference matrix.
    Parameters:
        chi (nparray): Solution for the steady distribution function in chi form.
        matrix_sc (memmap): Scattering matrix in simple linearization by default..
        matrix_fd (memmap): Finite difference matrix, generated by apply_centraldiff_matrix.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        field (dbl): Value of the electric field in V/m.
        freq (dbl): Frequency in GHz to be used for the transient solution
    Returns:
        g_next (nparray): Numpy array containing the (hopefully) converged iterative solution as g_chi.
        g_0 (nparray): Numpy array containing the RTA solution as g0_chi.
    """
    xProj = unitProjection[0]
    yProj = unitProjection[1]
    zProj = unitProjection[2]
    counter = occupation_solver.gmres_counter()
    print('Starting transient correlation XX longitudinal solver for {:.3E} V/m and {:E} GHz'.format(field,freq))
    freq_matrix = np.diag(np.ones(len(df))*1j*10**9*2*np.pi*freq)  # Positive quantity
    if pp.scmBool:
        scmfac = pp.scmVal
        print('Applying correction factor to the scattering matrix.')
    else:
        scmfac = 1
        print('Not applying correction factor to the scattering matrix.')
    loopstart = time.time()
    # Will only be able to run if you have a precalculated chi stored on file
    b_chi = (-1) * c.e * field / c.kb_joule / pp.T * np.squeeze(xProj*(df['vx [m/s]']+yProj*df['vy [m/s]']+zProj*df['vz [m/s]']) * df['k_FD']) * (1 - df['k_FD'])
    if pp.verboseError:
        b_chi_check = np.dot(matrix_sc*scmfac,chi2)
        error_chi = np.linalg.norm(b_chi_check - b_chi)/np.linalg.norm(b_chi)
        print('Residual error of chi is {:3E}'.format(error_chi))
    f0 = df['k_FD'].values
    f = chi2 + f0
    Vd_x,Vd_y,Vd_z = projected_drift(chi2, df, unitProjection)
    b = (-1) * (xProj*(df['vx [m/s]'] - Vd_x) + yProj*(df['vy [m/s]'] - Vd_y) + zProj*(df['vz [m/s]'] - Vd_z)) * f

    # chi2psi is used to give the finite difference matrix the right factors in front since substitution made
    psi2chi = np.squeeze(df['k_FD'] * (1 - df['k_FD']))
    invdiag = (np.diag(matrix_sc) * scmfac) ** (-1)
    g_0 = b * invdiag
    g_next, criteria = linalg.gmres(freq_matrix+matrix_sc * scmfac, b,x0=g_0,callback=counter,
                                    tol=pp.relConvergence, atol=pp.absConvergence)
    if pp.verboseError:
        b_check = np.dot(freq_matrix+matrix_sc*scmfac,g_next)
        error = np.linalg.norm(b_check - b)/np.linalg.norm(b)
        print('Norm of b is {:3E}'.format(np.linalg.norm(b)))
        print('Absolute residual error is {:3E}'.format(np.linalg.norm(b_check-b)))
        print('Relative residual error is {:3E}'.format(error))
        print('GMRES convergence criteria: {:3E}'.format(criteria))
    else:
        error = 0
        print('Error not stored.')
    loopend = time.time()
    print('Convergence took {:.2f}s'.format(loopend - loopstart))
    if not pp.simpleBool:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in simple linearization')
        g_next = g_next * psi2chi
        g_0 = g_0 * psi2chi
    return g_next, g_0,error, counter.niter


def transverse_lowfield_correlation(chi2, matrix_sc, df, field, freq, unitProjection):
    """Generalized minimal residual solver for calculating transient Gamma vd-vd autocorrelation using the full
    finite difference matrix.
    Parameters:
        chi (nparray): Solution for the steady distribution function in chi form.
        matrix_sc (memmap): Scattering matrix in simple linearization by default..
        matrix_fd (memmap): Finite difference matrix, generated by apply_centraldiff_matrix.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        field (dbl): Value of the electric field in V/m.
        freq (dbl): Frequency in GHz to be used for the transient solution
    Returns:
        g_next (nparray): Numpy array containing the (hopefully) converged iterative solution as g_chi.
        g_0 (nparray): Numpy array containing the RTA solution as g0_chi.
    """
    xProj = unitProjection[0]
    yProj = unitProjection[1]
    zProj = unitProjection[2]

    transverseProjection = np.array([1.0,-1.0,1.0])  # take a random vector
    transverseProjection -= transverseProjection.dot(unitProjection) * unitProjection  # make it orthogonal to k
    transverseProjection /= np.linalg.norm(transverseProjection)  # normalize it

    t_xProj = transverseProjection[0]
    t_yProj = transverseProjection[1]
    t_zProj = transverseProjection[2]

    counter = occupation_solver.gmres_counter()
    print('Starting transient correlation YY transverse solver for {:.3E} V/m and {:E} GHz'.format(field,freq))
    freq_matrix = np.diag(np.ones(len(df))*1j*10**9*2*np.pi*freq)  # Positive quantity
    if pp.scmBool:
        scmfac = pp.scmVal
        print('Applying correction factor to the scattering matrix.')
    else:
        scmfac = 1
        print('Not applying correction factor to the scattering matrix.')
    loopstart = time.time()
    # Will only be able to run if you have a precalculated chi stored on file
    b_chi = (-1) * c.e * field / c.kb_joule / pp.T * np.squeeze(
        xProj * (df['vx [m/s]'] + yProj * df['vy [m/s]'] + zProj * df['vz [m/s]']) * df['k_FD']) * (1 - df['k_FD'])
    if pp.verboseError:
        b_chi_check = np.dot(matrix_sc * scmfac, chi2)
        error_chi = np.linalg.norm(b_chi_check - b_chi) / np.linalg.norm(b_chi)
        print('Residual error of chi is {:3E}'.format(error_chi))
    f0 = df['k_FD'].values
    f = chi2 + f0

    b = (-1) * (t_xProj*(df['vx [m/s]']) + t_yProj*(df['vy [m/s]']) + t_zProj*(df['vz [m/s]'])) * f

    # chi2psi is used to give the finite difference matrix the right factors in front since substitution made
    psi2chi = np.squeeze(df['k_FD'] * (1 - df['k_FD']))
    invdiag = (np.diag(matrix_sc) * scmfac) ** (-1)
    g_0 = b * invdiag
    g_next, criteria = linalg.gmres(freq_matrix+matrix_sc * scmfac, b,x0=g_0,callback=counter,
                                    tol=pp.relConvergence, atol=pp.absConvergence)
    if pp.verboseError:
        b_check = np.dot(freq_matrix+matrix_sc*scmfac,g_next)
        error = np.linalg.norm(b_check - b)/np.linalg.norm(b)
        print('Norm of b is {:3E}'.format(np.linalg.norm(b)))
        print('Absolute residual error is {:3E}'.format(np.linalg.norm(b_check-b)))
        print('Relative residual error is {:3E}'.format(error))
        print('GMRES convergence criteria: {:3E}'.format(criteria))
    else:
        error = 0
        print('Error not stored.')
    loopend = time.time()
    print('Convergence took {:.2f}s'.format(loopend - loopstart))
    if not pp.simpleBool:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in simple linearization')
        g_next = g_next * psi2chi
        g_0 = g_0 * psi2chi
    return g_next, g_0,error, counter.niter,transverseProjection


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
            chi_2_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '2_' + "E_{:.1e}".format(ee) + '_FD_{:.1f}_{:.1f}_{:.1f}.npy'.format(xProj, yProj, zProj))

            corr_2_long,_,_,_ = longitudinal_lowfield_correlation(chi_2_i, scm, df, ee, freq, unitProjection)
            corr_2_tran, _, _, _, transverseProjection = transverse_lowfield_correlation(chi_2_i, scm, df, ee, freq, unitProjection)
            np.save(pp.outputLoc + 'SB_Density/' + 'corr_long_' + '2_' + "f_{:.1e}_E_{:.1e}".format(freq,ee) +'_FD_{:.1f}_{:.1f}_{:.1f}'.format(xProj,yProj,zProj),corr_2_long)
            np.save(pp.outputLoc + 'SB_Density/' + 'corr_tran_' + '2_' + "f_{:.1e}_E_{:.1e}".format(freq,ee) +'_FD_{:.1f}_{:.1f}_{:.1f}'.format(xProj,yProj,zProj),corr_2_tran)
            np.save(pp.outputLoc + 'SB_Density/' + 'tranProjection' +'_FD_{:.1f}_{:.1f}_{:.1f}'.format(xProj,yProj,zProj),transverseProjection)

            print('Transient solution written to file for ' + "{:.1e} V/m and {:.1e} GHz".format(ee,freq))
            print('\n \n')


def density(chi, EField,df,freq,unitProjection):
    f0 = df['k_FD'].values
    f = chi + f0
    Nuc = pp.kgrid ** 3
    xProj = unitProjection[0]
    yProj = unitProjection[1]
    zProj = unitProjection[2]

    transverseProjection = np.load(pp.outputLoc + 'SB_Density/' + 'tranProjection' + '_FD_{:.1f}_{:.1f}_{:.1f}.npy'.format(xProj, yProj, zProj))
    t_xProj = transverseProjection[0]
    t_yProj = transverseProjection[1]
    t_zProj = transverseProjection[2]

    print(unitProjection)
    print(transverseProjection)
    conductivity_long = 2 * c.e / (Nuc * c.Vuc * EField) * np.sum(xProj*chi * df['vx [m/s]']+yProj*chi * df['vy [m/s]']+zProj*chi * df['vz [m/s]'])

    n = utilities.calculate_noneq_density(chi,df)
    mobility = conductivity_long/c.e/n
    print('Mobility is {:3f} cm^2/(V-s)'.format(mobility*100**2))

    corr_long = np.load(pp.outputLoc + 'SB_Density/' + 'corr_long_' + '2_' + "f_{:.1e}_E_{:.1e}".format(freq,EField) +'_FD_{:.1f}_{:.1f}_{:.1f}.npy'.format(xProj,yProj,zProj))
    corr_tran = np.load(pp.outputLoc + 'SB_Density/' + 'corr_tran_' + '2_' + "f_{:.1e}_E_{:.1e}".format(freq,EField) +'_FD_{:.1f}_{:.1f}_{:.1f}.npy'.format(xProj,yProj,zProj))

    prefactor = (c.e / c.Vuc / Nuc) ** 2
    S_xx = 8*np.real(prefactor*np.sum(corr_long*(xProj*df['vx [m/s]']+yProj*df['vy [m/s]']+zProj*df['vz [m/s]'])))
    S_yy = 8*np.real(prefactor*np.sum(corr_tran*(t_xProj*df['vx [m/s]']+t_yProj*df['vy [m/s]']+t_zProj*df['vz [m/s]'])))
    return S_xx, S_yy, conductivity_long


def plot_density(fieldVector,freqVector,df,unitProjection):
    for freq in freqVector:
        S_long_vector = []
        S_tran_vector = []
        cond_long_vector = []
        xProj = unitProjection[0]
        yProj = unitProjection[1]
        zProj = unitProjection[2]
        for ee in fieldVector:
            chi_2_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '2_' + "E_{:.1e}".format(ee) + '_FD_{:.1f}_{:.1f}_{:.1f}.npy'.format(xProj, yProj, zProj))
            S_long, S_tran, cond_long = density(chi_2_i, ee,df,freq,unitProjection)
            S_long_vector.append(S_long)
            S_tran_vector.append(S_tran)
            cond_long_vector.append(cond_long)
            kvcm = np.array(fieldVector) * 1e-5

            Nuc = pp.kgrid ** 3
        fig, ax = plt.subplots()
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = '\n'.join((r'$f = %.1f GHz $' % (freq,), pp.fdmName))
        ax.plot(kvcm, S_long_vector, label=r'$S^{xx}$')
        ax.plot(kvcm, S_tran_vector, label=r'$S^{yy}$')
        ax.axhline(np.array(cond_long_vector[0]) * 4 * c.kb_joule * pp.T / c.Vuc / Nuc, linestyle='--',
                    label=r'$\frac{4 kT \sigma_{xx}^{eq}}{V_0 N_0}$')
        plt.legend()
        plt.xlabel('Field [kV/cm]')
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
    vd_1, meanE_1, n_1, mu_1, ng_1, nl_1 = ([] for i in range(6))
    vd_2, meanE_2, Vd_x2, Vd_y2, Vd_z2, nl_2 = ([] for i in range(6))

    s1_inds,s2_inds,s3_inds,s4_inds,s5_inds,s6_inds = silicon_splitter(df)
    n2_s1 = []
    n2_s3 = []
    n2 = []
    xProj = unitProjection[0]
    yProj = unitProjection[1]
    zProj = unitProjection[2]

    for ee in fieldVector:
        chi_2_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '2_' + "E_{:.1e}".format(ee) +'_FD_{:.1f}_{:.1f}_{:.1f}.npy'.format(xProj,yProj,zProj))
        vd_2.append(utilities.mean_velocity(chi_2_i,df))
        meanE_2.append(utilities.mean_energy(chi_2_i,df))
        vdx2,vdy2,vdz2 = projected_drift(chi_2_i,df,unitProjection)
        Vd_x2.append(vdx2)
        Vd_y2.append(vdy2)
        Vd_z2.append(vdz2)
        n2_s1.append(utilities.calc_popinds(chi_2_i,df,s1_inds))
        n2_s3.append(utilities.calc_popinds(chi_2_i,df,s3_inds))
        n2.append(utilities.calculate_noneq_density(chi_2_i,df))

    kvcm = np.array(fieldVector)*1e-5

    plt.figure()
    plt.plot(kvcm,meanE_2)

    plt.figure()
    plt.plot(kvcm,Vd_x2,label='kx')
    plt.plot(kvcm,Vd_y2,label='ky')
    plt.plot(kvcm,Vd_z2,label='kz')
    plt.plot(kvcm,np.sqrt(np.array(Vd_x2)**2+np.array(Vd_y2)**2+np.array(Vd_z2)**2),label = 'Total')
    plt.legend()

    plt.figure()
    plt.plot(kvcm,n2_s1)
    plt.plot(kvcm,n2_s3)
    plt.plot(kvcm,n2)



if __name__ == '__main__':
    fields = pp.fieldVector
    freqs = pp.freqVector
    unitProjection = utilities.cartesian_projection(pp.fieldDirection)
    print(unitProjection)
    # preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)
    # material_plotter.bz_3dscatter(electron_df, True, False)

    # write_steady(fields,electron_df)
    #plot_silicon_split(electron_df)
    df = electron_df.loc[(electron_df['kx [1/A]']<-0.5)&(electron_df['ky [1/A]']>-0.5)].sort_values(['k_inds','energy [eV]'])
    #plot_mom_KDEs(fields,electron_df,unitProjection)
    #plot_steady_transport_moments(electron_df, fields, unitProjection)

    #write_correlation(fields, electron_df, freqs, unitProjection)
    #plot_density(fields, freqs, electron_df, unitProjection)

    # df = electron_df.loc[(electron_df['kz [1/A]']>0.5)&(np.abs(electron_df['kx [1/A]'])<0.5)].sort_values(['k_inds','energy [eV]'])
    df['kx [1/A]'] = np.around(df['kx [1/A]'].values,11)
    df['ky [1/A]'] = np.around(df['ky [1/A]'].values,11)
    df['kz [1/A]'] = np.around(df['kz [1/A]'].values,11)

    # band_vec = np.zeros(len(df))
    # kinds_vec = np.unique(df['k_inds'].values)
    # for i1 in range(len(kinds_vec)):
    #     inds = np.where(df['k_inds'].values== kinds_vec[i1])
    #     inds = inds[0]
    #     band_vec[inds[0]] = 1
    #     band_vec[inds[1]] = 2
    # df['band'] = band_vec
    # print('Done assigning band index.')


    plt.figure()
    plt.plot(df['kx [1/A]'],df['energy [eV]'],'.')
    plt.ylabel('Energy [eV]')
    plt.xlabel('kx [1/A]')

    df2 = df.loc[df['bands'] ==1]
    s1_inds,s2_inds,_,_,_,_ = silicon_splitter(electron_df)
    df2 = electron_df.loc[s2_inds]

    fig, ax = plt.subplots()
    uniq_yz = np.unique(df2[['ky [1/A]', 'kz [1/A]']].values, axis=0)
    for i in range(len(uniq_yz)):
        kind = i + 1
        ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
        slice_df = df2.loc[(df2['ky [1/A]'] == ky) & (df2['kz [1/A]'] == kz)].sort_values(by=['kx [1/A]'],
                                                                                                   ascending=True)
        ax.plot(slice_df['kx [1/A]'],slice_df['k_FD'],color='black',linewidth=0.5)
        ax.plot(slice_df['kx [1/A]'],slice_df['k_FD'],'.',color='black')
    plt.yscale('log')
    plt.xlabel('kx [1/A]')
    plt.ylabel('Fermi-Dirac Occupation [unitless]')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = 'kx-kz slices' + '\n <-1 0 0> Valley'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.title(pp.title_str)


    fig, ax = plt.subplots()
    for i in range(len(uniq_yz)):
        kind = i + 1
        ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
        slice_df = df2.loc[(df2['ky [1/A]'] == ky) & (df2['kz [1/A]'] == kz)].sort_values(by=['kx [1/A]'],
                                                                                         ascending=True)
        ax.plot(slice_df['kx [1/A]'],slice_df['energy [eV]']-pp.mu,color='black',linewidth=0.5)
        ax.plot(slice_df['kx [1/A]'],slice_df['energy [eV]']-pp.mu,'.',color='black')
    plt.xlabel('kx [1/A]')
    plt.ylabel('Band Energy [eV]')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = 'kx-kz slices' + '\n <0 1 0> Valley'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.title(pp.title_str)


    fig, ax = plt.subplots()
    for i in range(len(uniq_yz)):
        kind = i + 1
        ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
        slice_df = df2.loc[(df2['ky [1/A]'] == ky) & (df2['kz [1/A]'] == kz)].sort_values(by=['kx [1/A]'],
                                                                                          ascending=True)
        ax.plot(slice_df['kx [1/A]'], slice_df['vx [m/s]'], color='black',linewidth=0.5)
        ax.plot(slice_df['kx [1/A]'], slice_df['vx [m/s]'], '.', color='black')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = 'kx-kz slices' + '\n <0 1 0> Valley'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.title(pp.title_str)
    plt.xlabel('kx [1/A]')
    plt.ylabel('vx [m/s]')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = df2['kx [1/A]'].values
    y = df2['ky [1/A]'].values
    z = df2['kz [1/A]'].values
    ax.scatter(x, y, z, color='black',s=1.5)
    for i in range(len(uniq_yz)):
        kind = i + 1
        ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
        # Grab the "slice" of points in k space with the same ky and kz coordinate
        slice_df = df2.loc[(df2['ky [1/A]'] == ky) & (df2['kz [1/A]'] == kz)].sort_values(by=['kx [1/A]'],ascending=True)
        x = slice_df['kx [1/A]'].values
        y = slice_df['ky [1/A]'].values
        z = slice_df['kz [1/A]'].values
        ax.plot(x, y, z,'-r',LineWidth=0.5)

    ax.set_xlabel('kx [1/A]')
    ax.set_ylabel('ky [1/A]')
    ax.set_zlabel('kz [1/A]')
    plt.title('ky-kz contours' + pp.title_str)




    # df2 = df.loc[df['bands'] ==2]
    #
    # fig, ax = plt.subplots()
    # uniq_yz = np.unique(df2[['ky [1/A]', 'kz [1/A]']].values, axis=0)
    # for i in range(len(uniq_yz)):
    #     kind = i + 1
    #     ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
    #     slice_df = df2.loc[(df['ky [1/A]'] == ky) & (df2['kz [1/A]'] == kz)].sort_values(by=['kx [1/A]'],
    #                                                                                      ascending=True)
    #     ax.plot(slice_df['kx [1/A]'],slice_df['k_FD'],color='black',linewidth=0.5)
    #     ax.plot(slice_df['kx [1/A]'],slice_df['k_FD'],'.',color='black')
    # plt.yscale('log')
    #
    #
    # fig, ax = plt.subplots()
    # for i in range(len(uniq_yz)):
    #     kind = i + 1
    #     ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
    #     slice_df = df2.loc[(df2['ky [1/A]'] == ky) & (df2['kz [1/A]'] == kz)].sort_values(by=['kx [1/A]'],
    #                                                                                      ascending=True)
    #     ax.plot(slice_df['kx [1/A]'],slice_df['energy [eV]'],color='black',linewidth=0.5)
    #     ax.plot(slice_df['kx [1/A]'],slice_df['energy [eV]'],'.',color='black')

    plt.show()