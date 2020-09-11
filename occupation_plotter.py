import numpy as np
import utilities
import problem_parameters as pp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import distance
import numpy.linalg
import constants as c
# import paper_figures
from matplotlib.lines import Line2D
# import davydov_utilities
import matplotlib

# PURPOSE: THIS MODULE WILL GENERATE DIRECT VISUALIZATIONS OF THE STEADY DISTRIBUTION FUNCTION AS A FUNCTION OF ELECTRIC
# FIELD. THE VISUALIZATIONS ARE KERNEL DENSITY ESTIMATES OF THE #2 (LOW FIELD + PERTURBO SCM) AND #3 (FINITE DIFFERENCE
# + PERTURBO SCM) DISTRIBUTION FUNCTIONS IN VELOCITY, MOMENTUM, AND ENERGY SPACE.

# ORDER: THIS MODULE CAN BE RUN AFTER OCCUPATION_SOLVER.PY HAS STORED SOLUTIONS IN THE OUTPUT LOC.

# OUTPUT: THIS MODULE RENDERS FIGURES FOR EXPLORATORY DATA ANALYSIS. IT DOES NOT SAVE FIGURES.

import matplotlib as mpl
# font = {'size': 11}
# mpl.rc('font', **font)
# triFigSize = (2.25*1.25,2.25*1.25)
# # squareFigSize = (2.53125, 2.53125)
# squareFigSize = (3, 3)
#
# singleCol_doubleSize = (3.375,3.375/1.4)
# # SMALL_SIZE = 7.6
# SMALL_SIZE = 8
# # SMALL_SIZE = 6
# # MEDIUM_SIZE = 8.5
# MEDIUM_SIZE = 9
# BIGGER_SIZE = 10
# different_small_size = 5.7
#
#
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#
# from matplotlib import rcParams
# rcParams.update({'figure.autolayout': True})
# plt.rcParams["font.family"] = "sans-serif"
#
# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'


def plot_steady_transient_difference(fieldVector, freq):
    """Plotting code to compare the steady-state and transient solutions at a given frequency. Generates plots of the
    residual between the real parts of the AC and DC solutions. Currently only implemented for #3 solutions
    (full derivative + Perturbo SCM).

    Parameters:
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        freq (dbl): Frequency in GHz to be used for the transient solution

    Returns:
        Nothing. Just the plots.
    """
    ac_dc_error = []
    ac_dc_cosine = []
    for ee in fieldVector:
        transient_chi = np.load(pp.outputLoc + 'Transient/' + 'chi_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq,ee))
        steady_chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        ac_dc_error.append(np.linalg.norm(transient_chi - steady_chi) / np.linalg.norm(steady_chi))
        ac_dc_cosine.append(distance.cosine(steady_chi,transient_chi)*-1)
        print('Relative residual is {:E}'.format(ac_dc_error[-1]))
        print('Cosine similarity is {:E}'.format(ac_dc_cosine[-1]))
    plt.figure()
    plt.plot(fieldVector*1E-5, ac_dc_error)
    plt.xlabel('EField (kV/cm)')
    plt.ylabel('Re[{:.1e} GHz]'.format(freq) + '/DC Residual')
    plt.title(pp.title_str)

    plt.figure()
    plt.plot(fieldVector*1E-5, ac_dc_cosine)
    plt.xlabel('EField (kV/cm)')
    plt.ylabel('Re[{:.1e} GHz]'.format(freq) + '/DC Cosine Similarity')
    plt.title(pp.title_str)


def momentum_distribution_kde(chi, df, ee, title=[], saveData=False, lowfield=False):
    """Takes chi solutions from file and plots the KDE of the distribution in momentum space.

    Parameters:
        chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in chi form.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        title (str): String containing the desired name of the plot

    Returns:
        Nothing. Just the plots.

    TODO: CURRENTLY THE SPACING PARAMETER IS HARDCODED FOR DIFFERENT GRID SIZES. WOULD BE NICE TO MAKE THIS MORE ROBUST
    IN THE FUTURE.
    """
    mom = df['kx [1/A]']
    npts = 400  # number of points in the KDE
    kdist = np.zeros(npts)
    kdist_tot = np.zeros(npts)
    kdist_f0 = np.zeros(npts)
    # Need to define the energy range that I'm doing integration over
    # en_axis = np.linspace(enk.min(), enk.min() + 0.4, npts)
    k_ax = np.linspace(mom.min(), mom.max(), npts)
    dx = (k_ax.max() - k_ax.min()) / npts
    f0 = np.zeros(len(df))
    if pp.kgrid == 200:
        spread = 22 * dx # For 200^3
    if pp.kgrid == 160:
        spread = 25 * dx  # For 160^3
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
    # ax.fill(k_ax, kdist, '--', linewidth=2, label='Non-equilibrium deviation', color='Red')
    ax.set_xlabel('kx [1/A]')
    ax.set_ylabel(r'Occupation [arb.]')
    plt.legend()
    if title:
        plt.title(title, fontsize=8)

    # Sometimes, it is useful to save the KDE data for paper figures. This toggles on Boolean.
    if saveData and lowfield:
        np.save(pp.outputLoc + 'Momentum_KDE/' + 'k_ax_' + '2_' + "E_{:.1e}".format(ee), k_ax)
        np.save(pp.outputLoc + 'Momentum_KDE/' + 'k_dist_f0_' + '2_' + "E_{:.1e}".format(ee), kdist_f0)
        np.save(pp.outputLoc + 'Momentum_KDE/' + 'k_dist' + '2_' + "E_{:.1e}".format(ee), kdist)
    if saveData and not lowfield:
        np.save(pp.outputLoc + 'Momentum_KDE/' + 'k_ax_' + '3_' + "E_{:.1e}".format(ee), k_ax)
        np.save(pp.outputLoc + 'Momentum_KDE/' + 'k_dist_f0_' + '3_' + "E_{:.1e}".format(ee), kdist_f0)
        np.save(pp.outputLoc + 'Momentum_KDE/' + 'k_dist' + '3_' + "E_{:.1e}".format(ee), kdist)


def velocity_distribution_kde(chi, df, title=[]):
    """Takes chi solutions which are already calculated and plots the KDE of the distribution in velocity space
    Parameters:
        chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in chi form.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        title (str): String containing the desired name of the plot
    Returns:
        Nothing. Just the plots.

    TODO: CURRENTLY THE SPACING PARAMETER IS HARDCODED FOR DIFFERENT GRID SIZES. WOULD BE NICE TO MAKE THIS MORE ROBUST
    IN THE FUTURE.
    """
    vel = df['vx [m/s]']
    npts = 600  # number of points in the KDE
    vdist = np.zeros(npts)
    vdist_tot = np.zeros(npts)
    vdist_f0 = np.zeros(npts)
    # Need to define the energy range that I'm doing integration over
    # en_axis = np.linspace(enk.min(), enk.min() + 0.4, npts)
    v_ax = np.linspace(vel.min(), vel.max(), npts)
    dx = (v_ax.max() - v_ax.min()) / npts
    f0 = np.squeeze(df['k_FD'].values)
    if pp.kgrid == 200:
        spread = 22 * dx # For 200^3
    if pp.kgrid == 160:
        spread = 25 * dx  # For 160^3
    if pp.kgrid == 80:
        spread = 70 *dx  # For 80^3
    def gaussian(x, mu, sigma=spread):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)

    for k in range(len(chi)):
        istart = int(np.maximum(np.floor((vel[k] - v_ax[0]) / dx) - (4 * spread / dx), 0))
        iend = int(np.minimum(np.floor((vel[k] - v_ax[0]) / dx) + (4 * spread / dx), npts - 1))
        vdist_tot[istart:iend] += (chi[k] + f0[k]) * gaussian(v_ax[istart:iend], vel[k])
        vdist_f0[istart:iend] += f0[k] * gaussian(v_ax[istart:iend], vel[k])
        vdist[istart:iend] += chi[k] * gaussian(v_ax[istart:iend], vel[k])

    ax = plt.axes([0.18, 0.15, 0.76, 0.76])
    ax.plot(v_ax, [0] * len(v_ax), 'k')
    ax.plot(v_ax, vdist_f0, '--', linewidth=2, label='Equilbrium')
    ax.plot(v_ax, vdist_tot, linewidth=2, label='Hot electron distribution')
    # plt.fill(v_ax, vdist, label='non-eq distr', color='red')
    ax.fill(v_ax, vdist, '--', linewidth=2, label='Non-equilibrium deviation', color='Red')
    ax.set_xlabel(r'Velocity [ms$^{-1}$]')
    ax.set_ylabel(r'Occupation [arb.]')
    plt.legend()
    if title:
        plt.title(title, fontsize=8)


def plot_vel_KDEs(field, df):
    """Wrapper script for velocity_distribution_kde. Can do for the various solution schemes saved to file, but for now
    only implemented for steady #3 solutions.
        Parameters:
        field (dbl): the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        Nothing. Just the plots.
    """
    chi_3 = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(field))
    g_inds,l_inds,x_inds = utilities.gaas_split_valleys(df,False)
    chi_3_g = chi_3[g_inds]
    chi_3_l = chi_3[l_inds]
    if pp.getX:
        chi_3_x = chi_3[x_inds]
    # velocity_distribution_kde(chi_3, df, title='DC Chi FDM {:.1e} V/m '.format(field) + pp.title_str)

    velocity_distribution_kde(chi_3_g, df.loc[g_inds].reset_index(), title='Gamma Chi FDM {:.1e} V/m '.format(field) + pp.title_str)
    velocity_distribution_kde(chi_3_l, df.loc[l_inds].reset_index(), title='L Chi FDM {:.1e} V/m '.format(field) + pp.title_str)
    if pp.getX:
        velocity_distribution_kde(chi_3_x, df.loc[x_inds].reset_index(), title='X Chi FDM {:.1e} V/m '.format(field) + pp.title_str)


def plot_mom_KDEs(fieldVector, df, saveData = False):
    """Wrapper script for momentum_distribution_kde. Can do for the various solution schemes saved to file, but for now
    only implemented for steady #2 or #3 solutions, on Boolean toggle.
        Parameters:
        fieldVector (nparray): the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        lowField (bool): Should the momentum distribution be calculated for #2 or #3 solutions.
        saveData (bool): Should the kde be written to file?

    Returns:
        Nothing. Just the plots.
    """
    g_inds,l_inds,x_inds = utilities.gaas_split_valleys(df,False)
    plt.figure()
    for ee in fieldVector:
        chi_2 = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '2_' + "E_{:.1e}.npy".format(ee))
        chi_2_g = chi_2[g_inds]
        momentum_distribution_kde(chi_2_g, df.loc[g_inds].reset_index(), ee,'',saveData,True)
    plt.figure()
    for ee in fieldVector:
        chi_3 = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        chi_3_g = chi_3[g_inds]
        momentum_distribution_kde(chi_3_g, df.loc[g_inds].reset_index(), ee,'',saveData,False)
    plt.figure()
    freq = pp.freqVector[0]

    for ee in fieldVector:
        chi_3 = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        momentum_distribution_kde(chi_3[g_inds], df.loc[g_inds].reset_index(), ee,'',saveData,False)

    plt.figure()
    vx = df['vx [m/s]'].values
    for ee in fieldVector:
        chi_3 = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        momentum_distribution_kde(chi_3[g_inds]*(vx[g_inds]-utilities.mean_velocity(chi_3,df)), df.loc[g_inds].reset_index(), ee,'',saveData,False)

    plt.figure()
    for ee in fieldVector:
        chi_3 = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        momentum_distribution_kde((vx[g_inds]-utilities.mean_velocity(chi_3,df)), df.loc[g_inds].reset_index(), ee,'',saveData,False)

    plt.figure()
    for ee in fieldVector:
        g_k = np.real(np.load(pp.outputLoc + '/SB_Density/xx_3_f_{:.1e}_E_{:.1e}.npy'.format(freq, ee)))
        momentum_distribution_kde(g_k[g_inds], df.loc[g_inds].reset_index(), ee,'',saveData,False)

    plt.figure()
    for ee in fieldVector:
        g_k = np.real(np.load(pp.outputLoc + '/SB_Density/xx_3_f_{:.1e}_E_{:.1e}.npy'.format(freq, ee)))
        momentum_distribution_kde(g_k[g_inds]*vx[g_inds], df.loc[g_inds].reset_index(), ee,'',saveData,False)



def occupation_v_energy_sep(chi, enk, kptsdf):
    """Takes chi solutions which are already calculated and plots the KDE of the distribution in energy space. The Gamma,
    L, and X valleys are calculated separately.
        Parameters:
            chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in chi form.
            enk (nparray): Containing the energy associated with each state in eV.
            kptsdf (datafrane): Electron dataframe containing momentum and velocity information.
        Returns:
            Nothing. Just the plots.

        TODO: CURRENTLY THE SPACING PARAMETER IS HARDCODED FOR DIFFERENT GRID SIZES. WOULD BE NICE TO MAKE THIS MORE ROBUST
        IN THE FUTURE.
        """

    g_inds, l_inds, x_inds = utilities.gaas_split_valleys(kptsdf,False)

    vmags = kptsdf['v_mag [m/s]']

    npts = 400  # number of points in the KDE
    g_chiax = np.zeros(npts)
    g_ftot = np.zeros(npts)
    g_f0ax = np.zeros(npts)

    l_chiax = np.zeros(npts)
    l_ftot = np.zeros(npts)
    l_f0ax = np.zeros(npts)

    f0ax = np.zeros(npts)

    # Need to define the energy range that I'm doing integration over
    en_axis = np.linspace(enk.min(), enk.max(), npts)
    g_en_axis = np.linspace(enk.min(), enk.max(), npts)
    l_en_axis = np.linspace(enk.min(), enk.max(), npts)

    g_chi = chi[g_inds]
    l_chi = chi[l_inds]
    g_enk = enk[g_inds]
    l_enk = enk[l_inds]

    dx = (g_en_axis.max() - g_en_axis.min()) / npts
    g_f0 = np.squeeze(kptsdf.loc[g_inds,'k_FD'].values)
    l_f0 = np.squeeze(kptsdf.loc[l_inds,'k_FD'].values)

    f0 = np.squeeze(kptsdf['k_FD'])
    # (Peishi): I used 35 for the spread for npts=400 and it works pretty well. Don't need a ton of npts.
    spread = 35 * dx
    spread = 350 * dx  # This is a value that has to change to get better smoothing. Also depends on the number of pts.
    # spread = 120 * dx
    # spread = 200 * dx
    # spread = 600 * dx

    def gaussian(x, mu, vmag, stdev=spread):
        sigma = stdev - (vmag/1E6) * 1.0 * stdev
        vals = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)
        return vals

    for k in range(len(f0)):
        istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
        iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
        f0ax[istart:iend] += f0[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k])

    for k in range(len(g_chi)):
        istart = int(np.maximum(np.floor((g_enk[k] - g_en_axis[0]) / dx) - (4*spread/dx), 0))
        iend = int(np.minimum(np.floor((g_enk[k] - g_en_axis[0]) / dx) + (4*spread/dx), npts - 1))
        g_ftot[istart:iend] += (g_chi[k] + g_f0[k]) * gaussian(g_en_axis[istart:iend], g_enk[k],vmags[k])
        g_f0ax[istart:iend] += g_f0[k] * gaussian(g_en_axis[istart:iend], g_enk[k], vmags[k])
        g_chiax[istart:iend] += g_chi[k] * gaussian(g_en_axis[istart:iend], g_enk[k], vmags[k])

    for k in range(len(l_chi)):
        istart = int(np.maximum(np.floor((l_enk[k] - l_en_axis[0]) / dx) - (4 * spread / dx), 0))
        iend = int(np.minimum(np.floor((l_enk[k] - l_en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
        l_ftot[istart:iend] += (l_chi[k] + l_f0[k]) * gaussian(l_en_axis[istart:iend], l_enk[k], vmags[k])
        l_f0ax[istart:iend] += l_f0[k] * gaussian(l_en_axis[istart:iend], l_enk[k], vmags[k])
        l_chiax[istart:iend] += l_chi[k] * gaussian(l_en_axis[istart:iend], l_enk[k], vmags[k])

    if pp.getX:
        x_chiax = np.zeros(npts)  # capital sigma as defined in Jin Jian's paper Eqn 3
        x_ftot = np.zeros(npts)
        x_f0ax = np.zeros(npts)
        x_en_axis = np.linspace(enk.min(), enk.max(), npts)
        x_chi = chi[x_inds]
        x_enk = enk[x_inds]
        x_f0 = np.squeeze(kptsdf.loc[x_inds,'k_FD'].values)
        for k in range(len(x_chi)):
            istart = int(np.maximum(np.floor((x_enk[k] - x_en_axis[0]) / dx) - (4 * spread / dx), 0))
            iend = int(np.minimum(np.floor((x_enk[k] - x_en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
            x_ftot[istart:iend] += (x_chi[k] + x_f0[k]) * gaussian(x_en_axis[istart:iend], x_enk[k], vmags[k])
            x_f0ax[istart:iend] += x_f0[k] * gaussian(x_en_axis[istart:iend], x_enk[k], vmags[k])
            x_chiax[istart:iend] += x_chi[k] * gaussian(x_en_axis[istart:iend], x_enk[k], vmags[k])
    else:
        x_en_axis = []
        x_ftot = []
        x_chi = []
        x_chiax = []
        x_f0ax =[]
    return g_en_axis, g_ftot, g_chiax, g_f0ax, l_en_axis, l_ftot, l_chiax, l_f0ax, x_en_axis, x_ftot, x_chiax, x_f0ax, f0ax, en_axis


def plot_energy_sep(df, fields):
    """Wrapper script for occupation_v_energy_sep. Can do for the various solution schemes saved to file, but for now
    only implemented for steady #3 solutions.
        Parameters:
        fields (nparray): Containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        Nothing. Just the plots.
    """
    plt.figure(figsize=(6, 6))
    for ee in fields:
        chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        g_en_axis, g_ftot, g_chiax, g_f0ax, l_en_axis, l_ftot, l_chiax, l_f0ax, x_en_axis, x_ftot, x_chiax, x_f0ax, f0ax, en_axis = \
        occupation_v_energy_sep(chi_3_i, df['energy [eV]'].values, df)
        plt.plot(g_en_axis-np.min(df['energy [eV]']), g_chiax, label=r'$\Gamma$'+' Valley')
        plt.plot(l_en_axis-np.min(df['energy [eV]']), l_chiax, '--', label='L Valley')
        plt.plot(en_axis-np.min(df['energy [eV]']), f0ax, '-', label='Equilibrium')
        if pp.getX:
            plt.plot(x_en_axis - np.min(df['energy [eV]']), x_chiax, '--', label='X Valley')
    plt.ylim([-0.02,0.015])
    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel(r'FDM deviational occupation ($\delta f_{\mathbf{k}}$) [arb]')
    plt.title(pp.title_str)


def plot_energy_sep_lf(df, fields):
    """Wrapper script for occupation_v_energy_sep. Can do for the various solution schemes saved to file, but for now
    only implemented for steady #2 solutions.
        Parameters:
        fields (nparray): Containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        Nothing. Just the plots.
    """
    plt.figure(figsize=(6, 6))
    for ee in fields:
        chi_2_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '2_' + "E_{:.1e}.npy".format(ee))
        g_en_axis, g_ftot, g_chiax, g_f0ax, l_en_axis, l_ftot, l_chiax, l_f0ax, x_en_axis, x_ftot, x_chiax, x_f0ax = \
            occupation_v_energy_sep(chi_2_i, df['energy [eV]'].values, df)
        plt.plot(g_en_axis - np.min(df['energy [eV]']), g_chiax, label=r'$\Gamma$' + ' Valley')
        plt.plot(l_en_axis - np.min(df['energy [eV]']), l_chiax, '--', label='L Valley')
        if pp.getX:
            plt.plot(x_en_axis - np.min(df['energy [eV]']), x_chiax, '--', label='X Valley')
    plt.ylim([-0.02,0.015])
    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel(r'LF deviational occupation ($\delta f_{\mathbf{k}}$) [arb]')
    plt.title(pp.title_str)


def plot_2d_dist(df, field):
    """Plots a 3D scatter where x and y are momentum coordinates and z is the distribution function.
        Parameters:
        fields (nparray): Containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        Nothing. Just the plots.
    """
    g_inds,_,_ = utilities.gaas_split_valleys(df, True)
    plt.figure(figsize=(6, 6))
    chi_3 = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(field))

    x = df.loc[g_inds, 'kx [1/A]'].values / (2 * np.pi / c.alat)
    y = df.loc[g_inds, 'ky [1/A]'].values / (2 * np.pi / c.alat)
    z = np.log10(chi_3[g_inds] + df.loc[g_inds,'k_FD'])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.title(pp.title_str)


def plot_noise_kde(el_df, big_e,freq):
    vmags = el_df['v_mag [m/s]']
    vx = el_df['vx [m/s]']
    enk = el_df['energy [eV]'] - el_df['energy [eV]'].min()

    lw = 1.5
    alpha = 1
    npts = 400  # number of points in the KDE
    # Define the energy range over intergration
    en_axis = np.linspace(enk.min(), enk.max(), npts)
    v_axis = np.linspace(vx.min(), vx.max(), npts)
    dx = (en_axis.max() - en_axis.min()) / npts
    spread = 40 * dx

    def gaussian(x, mu, vmag, stdev=spread):
        sigma = stdev - (vmag/1E6) * 0.9 * stdev
        vals = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)
        return vals

    # Define colormap
    cmap = plt.cm.get_cmap('YlOrRd', 9)
    colors = []
    # for i in range(cmap.N):
    #     rgb = cmap(i)[:3]  # will return rgba, we take only first 3 so we get rgb
    #     colors.append(mpl.colors.rgb2hex(rgb))

    colors = ['#0C1446','#FF3F00']

    noise_tot = np.zeros(len(big_e))
    plt.figure()
    enplot = plt.axes()
    for i, ee in enumerate(big_e):
        g_k = np.real(np.load(pp.outputLoc + '/SB_Density/xx_3_f_{:.1e}_E_{:.1e}.npy'.format(freq, ee)))
        noise_k = 2 * (2 * c.e / pp.kgrid**3 / c.Vuc)**2 * np.real(g_k * vx)
        # noise_k = np.abs(vx)
        noise_tot[i] = np.sum(noise_k)
        noise_en = np.zeros(npts)

        for k in range(len(noise_k)):
            istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
            iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
            noise_en[istart:iend] += noise_k[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k])

        enplot.plot(en_axis, noise_en, color=colors[i], label='{:.1f} V/cm'.format(ee / 100))
    enplot.set_xlabel('Energy above CBM (eV)')
    # plt.yticks([])
    # plt.ylabel(r'Noise power per energy ($A^2m^{-4}Hz^{-1}eV^{-1}$)')
    # plt.ylabel(r'Noise power per energy')
    enplot.set_ylabel(r'Effective distribution (g)')
    enplot.set_xlim([0, 0.45])
    enplot.legend()

    plt.figure(figsize=squareFigSize)

    vy = el_df['vy [m/s]']
    vxplot = plt.axes()
    for i, ee in enumerate(big_e):
        g_k = np.real(np.load(pp.outputLoc + '/SB_Density/xx_3_f_{:.1e}_E_{:.1e}.npy'.format(pp.freqVector[-1], ee)))
        g_k_2 = np.real(np.load(pp.outputLoc + '/SB_Density/yy_3_f_{:.1e}_E_{:.1e}.npy'.format(pp.freqVector[-1], ee)))

        chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
        chi_t_i = np.load(pp.outputLoc + '/Transient/chi_3_f_{:.1e}_E_{:.1e}.npy'.format(pp.freqVector[-1], ee))
        dv = utilities.mean_velocity(chi, el_df)
        noise_k = 2 * (2 * c.e / pp.kgrid**3 / c.Vuc)**2 * np.real(g_k * vx)
        noise_k_2 = 2 * (2 * c.e / pp.kgrid**3 / c.Vuc)**2 * np.real(g_k_2 * vy)

        noise_vx = np.zeros(npts)
        noise_vx_2 = np.zeros(npts)
        density_kx = np.zeros(npts)
        for k in range(len(noise_k)):
            istart = int(np.maximum(np.floor((vx[k] - v_axis[0]) / dx) - (4 * spread / dx), 0))
            iend = int(np.minimum(np.floor((vx[k] - v_axis[0]) / dx) + (4 * spread / dx), npts - 1))
            noise_vx[istart:iend] += noise_k[k] * gaussian(v_axis[istart:iend], vx[k], vmags[k], stdev=spread)
            noise_vx_2[istart:iend] += noise_k_2[k] * gaussian(v_axis[istart:iend], vx[k], vmags[k], stdev=spread)
            density_kx[istart:iend] += chi[k] * gaussian(v_axis[istart:iend], vx[k], vmags[k], stdev=spread)
        if i == 0:
            normed = np.trapz(noise_vx,v_axis*1e-6)
        vxplot.plot(v_axis * 1E-3, noise_vx/normed, color=colors[i], label='{:.0f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$',alpha=alpha)
        # vxplot.plot(v_axis * 1E-3, noise_vx_2/normed,'--',color=colors[i],label='{:.0f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$',alpha=alpha)
        vxplot.axvline(dv * 1E-3, color=colors[i], linestyle='--')
        # vxplot.plot(v_axis * 1E-3, density_kx/np.min(density_kx), color=colors[4+i], label='{:.1f} V/cm'.format(ee / 100))
    vxplot.set_xlabel(r'Group velocity $\, (\rm km \, s^{-1}$)')
    vxplot.set_ylabel(r'Noise weight '+ r'$\rm  (A^2 \, m^3 \, Hz^{-2})$')
    vxplot.legend(loc='upper left',frameon=False,prop={'size': different_small_size})
    plt.xlim([np.min(v_axis * 1E-3),np.max(v_axis * 1E-3)])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    freqTHz = pp.freqVector[-1]/1000
    textstr = 'f = % .0f THz' %freqTHz
    # textstr = 'f = % .0f GHz' %pp.freqVector[-1]
    plt.ylim([0,1.5])
    vxplot.text(0.98,0.98,textstr, transform=vxplot.transAxes, fontsize=8, verticalalignment='top', bbox=props,ha ='right',va='top')
    # plt.ylim([0,2.25e-5])
    plt.savefig(pp.figureLoc+'Longitudinal_noiseDecomp_10THz.png', bbox_inches='tight',dpi=600)


    lw = 1
    vx = el_df['vy [m/s]']
    v_axis = np.linspace(vx.min(), vx.max(), npts)
    dx = (v_axis.max() - v_axis.min()) / npts
    spread = 40 * dx
    alpha = 0.5
    f,(ax1,ax2) = plt.subplots(1,2,figsize=singleCol_doubleSize,sharey=True)
    for i, ee in enumerate(big_e):
        g_k = np.real(np.load(pp.outputLoc + '/SB_Density/yy_3_f_{:.1e}_E_{:.1e}.npy'.format(pp.freqVector[0], ee)))
        chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
        chi_t_i = np.load(pp.outputLoc + '/Transient/chi_3_f_{:.1e}_E_{:.1e}.npy'.format(pp.freqVector[0], ee))
        dv = utilities.mean_velocity(chi, el_df)
        noise_k = 2 * (2 * c.e / pp.kgrid ** 3 / c.Vuc) ** 2 * np.real(g_k * vx)
        noise_vx = np.zeros(npts)
        density_kx = np.zeros(npts)
        for k in range(len(noise_k)):
            istart = int(np.maximum(np.floor((vx[k] - v_axis[0]) / dx) - (4 * spread / dx), 0))
            iend = int(np.minimum(np.floor((vx[k] - v_axis[0]) / dx) + (4 * spread / dx), npts - 1))
            noise_vx[istart:iend] += noise_k[k] * gaussian(v_axis[istart:iend], vx[k], vmags[k], stdev=spread)
            density_kx[istart:iend] += chi[k] * gaussian(v_axis[istart:iend], vx[k], vmags[k], stdev=spread)
        if i == 0:
            normed = np.trapz(noise_vx, v_axis * 1e-6)
            ax2.plot(v_axis * 1E-3, noise_vx / normed, color=colors[i], alpha=alpha, linewidth=lw)
            ax2.fill(v_axis * 1E-3, noise_vx / normed, color=colors[i],
                        label='{:.0f} '.format(ee / 100) + r'$\rm V \, cm^{-1}$', alpha=alpha)
        if i > 0:
            ax2.plot(v_axis * 1E-3, noise_vx / normed, color=colors[i], alpha=alpha, linewidth=lw)
            ax2.fill(v_axis * 1E-3, noise_vx / normed, color=colors[i],
                        label=r'{:.0f} '.format(ee / 100) + r'$\rm V \, cm^{-1}$', alpha=alpha)

        # vxplot.plot(v_axis * 1E-3, noise_vx, color=colors[4+i],lw=lw)
    ax2.axvline(0 * 1E-3, color='black', linestyle='--', linewidth=lw,alpha=alpha)
        # vxplot.plot(v_axis * 1E-3, density_kx/np.min(density_kx), color=colors[4+i], label='{:.1f} V/cm'.format(ee / 100))
    # vxplot.plot(v_axis * 1E-3, noise_vy / normed, linestyle=(0,(lw,1,1,1)),color=colors[i],
    #             label=r'$S_{\perp}$ '+'{:.0f} '.format(ee / 100) + r'$\rm V \, cm^{-1}$', alpha=alpha, linewidth=lw)
    ax2.set_xlabel(r'$v_{\perp} \, (\rm km \, s^{-1}$)')
    # vxplot.set_ylabel(r'Noise weight ' + r'$\rm  (A^2 \, m^3 \, Hz^{-2})$')
    # ax2.legend(markerscale=0.02,loc='upper left', frameon=False,handletextpad=0.4,handlelength=1.2,borderaxespad=0.2,borderpad=0.2)
    ax2.set_xlim([np.min(v_axis * 1E-3), np.max(v_axis * 1E-3)])
    ax2.set_title('Transverse')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    # freqTHz = freq/1000
    # textstr = 'f = % .0f THz' %freqTHz
    textstr = 'f = % .0f GHz' % pp.freqVector[0]
    # ax2.set_ylim([0, 1.5])
    # vxplot.text(0.98,0.98,textstr, transform=vxplot.transAxes, fontsize=8, verticalalignment='top', bbox=props,ha ='right',va='top')
    # plt.ylim([0,2.25e-5])
    # plt.savefig(pp.figureLoc+'Transverse_noiseDecomp_0GHz.png', bbox_inches='tight',dpi=600)


    vx = el_df['vx [m/s]']
    v_axis = np.linspace(vx.min(), vx.max(), npts)
    dx = (v_axis.max() - v_axis.min()) / npts
    spread = 40 * dx
    alpha = 0.5
    for i, ee in enumerate(big_e):
        g_k = np.real(np.load(pp.outputLoc + '/SB_Density/xx_3_f_{:.1e}_E_{:.1e}.npy'.format(pp.freqVector[0], ee)))
        chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
        chi_t_i = np.load(pp.outputLoc + '/Transient/chi_3_f_{:.1e}_E_{:.1e}.npy'.format(pp.freqVector[0], ee))
        dv = utilities.mean_velocity(chi, el_df)
        noise_k = 2 * (2 * c.e / pp.kgrid ** 3 / c.Vuc) ** 2 * np.real(g_k * vx)
        noise_vx = np.zeros(npts)
        density_kx = np.zeros(npts)
        for k in range(len(noise_k)):
            istart = int(np.maximum(np.floor((vx[k] - v_axis[0]) / dx) - (4 * spread / dx), 0))
            iend = int(np.minimum(np.floor((vx[k] - v_axis[0]) / dx) + (4 * spread / dx), npts - 1))
            noise_vx[istart:iend] += noise_k[k] * gaussian(v_axis[istart:iend], vx[k], vmags[k], stdev=spread)
            density_kx[istart:iend] += chi[k] * gaussian(v_axis[istart:iend], vx[k], vmags[k], stdev=spread)
        if i == 0:
            normed = np.trapz(noise_vx, v_axis * 1e-6)
            ax1.plot(v_axis * 1E-3, noise_vx / normed, color=colors[i],alpha=alpha, linewidth=lw)
            ax1.fill(v_axis * 1E-3, noise_vx / normed, color=colors[i],
                        label='{:.0f} '.format(ee / 100) + r'$\rm V \, cm^{-1}$', alpha=alpha)
        if i > 0:
            ax1.plot(v_axis * 1E-3, noise_vx / normed, color=colors[i], alpha=alpha, linewidth=lw)
            ax1.fill(v_axis * 1E-3, noise_vx / normed, color=colors[i],
                        label=r'{:.0f} '.format(ee / 100) + r'$\rm V \, cm^{-1}$', alpha=alpha)

        # vxplot.plot(v_axis * 1E-3, noise_vx, color=colors[4+i],lw=lw)
        ax1.axvline(dv * 1E-3, color=colors[i], linestyle='--', linewidth=lw,alpha=alpha+0.2)
        # vxplot.plot(v_axis * 1E-3, density_kx/np.min(density_kx), color=colors[4+i], label='{:.1f} V/cm'.format(ee / 100))
    # vxplot.plot(v_axis * 1E-3, noise_vy / normed, linestyle=(0,(lw,1,1,1)),color=colors[i],
    #             label=r'$S_{\perp}$ '+'{:.0f} '.format(ee / 100) + r'$\rm V \, cm^{-1}$', alpha=alpha, linewidth=lw)
    ax1.set_xlabel(r'$v_{\parallel} \, (\rm km \, s^{-1}$)')
    ax1.set_ylabel(r'Noise weight ' + r'$\rm  (A^2 \, m^3 \, Hz^{-2})$')
    ax1.legend(markerscale=0.02,loc='upper left', frameon=False,handletextpad=0.4,handlelength=1.2,borderaxespad=0.2,borderpad=0.2)
    ax1.set_xlim([np.min(v_axis * 1E-3), np.max(v_axis * 1E-3)])
    ax1.set_title('Longitudinal')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    # freqTHz = freq/1000
    # textstr = 'f = % .0f THz' %freqTHz
    textstr = 'f = % .0f GHz' % pp.freqVector[0]
    ax1.set_ylim([0, 1.5])
    # vxplot.text(0.98,0.98,textstr, transform=vxplot.transAxes, fontsize=8, verticalalignment='top', bbox=props,ha ='right',va='top')
    # plt.ylim([0,2.25e-5])
    plt.savefig(pp.figureLoc + 'Longitudinal_noiseDecomp_0GHz.png', bbox_inches='tight', dpi=600)

    # plt.figure()
    # vxplot = plt.axes()
    # for i, ee in enumerate(big_e):
    #     g_k = np.real(np.load(pp.outputLoc + '/Small_Signal/decomp_cond_3_f_{:.1e}_E_{:.1e}.npy'.format(freq, ee)))
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     chi_t_i = np.load(pp.outputLoc + '/Transient/chi_3_f_{:.1e}_E_{:.1e}.npy'.format(freq, ee))
    #     dv = utilities.mean_velocity(chi, el_df)
    #     noise_k = np.real(chi)*c.kb_joule*pp.T/(c.e*ee)*vx
    #     noise_vx = np.zeros(npts)
    #     density_kx = np.zeros(npts)
    #     for k in range(len(noise_k)):
    #         istart = int(np.maximum(np.floor((vx[k] - v_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((vx[k] - v_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         noise_vx[istart:iend] += noise_k[k] * gaussian(v_axis[istart:iend], vx[k], vmags[k], stdev=spread)
    #         density_kx[istart:iend] += chi[k] * gaussian(v_axis[istart:iend], vx[k], vmags[k], stdev=spread)
    #     vxplot.plot(v_axis * 1E-3, noise_vx, color=colors[i], label='{:.1f} V/cm'.format(ee / 100),alpha = alpha)
    #     # vxplot.plot(v_axis * 1E-3, noise_vx, color=colors[4+i],lw=lw)
    #     vxplot.axvline(dv * 1E-3, color=colors[i], linestyle='--')
    # # vxplot.plot(v_axis * 1E-3, density_kx/np.min(density_kx), color=colors[4+i], label='{:.1f} V/cm'.format(ee / 100))
    # vxplot.set_xlabel(r'Longitudinal group velocity ($\rm km \, s^{-1}$)')
    # vxplot.set_ylabel(r'DC conductivity weight ($\rm S \, m^{-2} \, Hz^{-1}$)')
    # vxplot.legend()
    # plt.savefig(pp.figureLoc + 'DC_condKDE.png', bbox_inches='tight', dpi=600)
    #
    # nkpts = len(el_df)
    # scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    # thermal_energy = utilities.mean_energy(np.zeros(nkpts), el_df)
    # energy_above_thermal = el_df['energy [eV]']-thermal_energy
    #
    # plt.figure(figsize=triFigSize)
    # vxplot = plt.axes()
    # for i, ee in enumerate(big_e):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     dv = utilities.mean_velocity(chi, el_df)
    #     power_k = np.dot(scm,chi)*energy_above_thermal
    #     power_vx = np.zeros(npts)
    #     for k in range(len(power_k)):
    #         istart = int(np.maximum(np.floor((vx[k] - v_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((vx[k] - v_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         power_vx[istart:iend] += power_k[k] * gaussian(v_axis[istart:iend], vx[k], vmags[k], stdev=spread)
    #     vxplot.plot(v_axis * 1E-3, power_vx, color=colors[i], label='{:.1f} V/cm'.format(ee / 100),alpha = alpha)
    #     vxplot.axvline(dv * 1E-3, color=colors[i], linestyle='--')
    # vxplot.set_xlabel(r'Longitudinal group velocity ($\rm km \, s^{-1}$)')
    # vxplot.set_ylabel(r'Power weight ($\rm eV \, m^{-1}$)')
    # vxplot.legend()
    # plt.savefig(pp.figureLoc + 'power.png', bbox_inches='tight', dpi=600)
    #
    # plt.figure(figsize=triFigSize)
    # vxplot = plt.axes()
    # for i, ee in enumerate(big_e):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     dv = utilities.mean_velocity(chi, el_df)
    #     power_k = np.dot(scm,chi)*energy_above_thermal
    #     power_vx = np.zeros(npts)
    #     for k in range(len(power_k)):
    #         istart = int(np.maximum(np.floor((vx[k] - v_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((vx[k] - v_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         power_vx[istart:iend] += power_k[k] * gaussian(v_axis[istart:iend], vx[k], vmags[k], stdev=spread)
    #     vxplot.plot(v_axis * 1E-3, power_vx, color=colors[i], label='{:.1f} V/cm'.format(ee / 100),alpha = alpha)
    #     vxplot.axvline(dv * 1E-3, color=colors[i], linestyle='--')
    # vxplot.set_xlabel(r'Longitudinal group velocity ($\rm km \, s^{-1}$)')
    # vxplot.set_ylabel(r'Power weight ($\rm eV \, m^{-1}$)')
    # vxplot.legend()
    # plt.savefig(pp.figureLoc + 'power.png', bbox_inches='tight', dpi=600)
    #
    # plt.figure(figsize=triFigSize)
    # vxplot = plt.axes()
    # for i, ee in enumerate(big_e):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     dv = utilities.mean_velocity(chi, el_df)
    #     power_k = np.dot(scm,chi)
    #     power_vx = np.zeros(npts)
    #     for k in range(len(power_k)):
    #         istart = int(np.maximum(np.floor((vx[k] - v_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((vx[k] - v_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         power_vx[istart:iend] += power_k[k] * gaussian(v_axis[istart:iend], vx[k], vmags[k], stdev=spread)
    #     vxplot.plot(v_axis * 1E-3, power_vx, color=colors[i], label='{:.1f} V/cm'.format(ee / 100),alpha = alpha)
    #     vxplot.axvline(dv * 1E-3, color=colors[i], linestyle='--')
    # vxplot.set_xlabel(r'Longitudinal group velocity ($\rm km \, s^{-1}$)')
    # vxplot.set_ylabel(r'Density flux weight ($\rm s \, m^{-1}$)')
    # vxplot.legend()
    # plt.savefig(pp.figureLoc + 'densityFlux.png', bbox_inches='tight', dpi=600)
    #
    # plt.figure(figsize=triFigSize)
    # vxplot = plt.axes()
    # for i, ee in enumerate(big_e):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     dv = utilities.mean_velocity(chi, el_df)
    #     power_k = energy_above_thermal
    #     power_vx = np.zeros(npts)
    #     normalizing_vx = np.zeros(npts)
    #     for k in range(len(power_k)):
    #         istart = int(np.maximum(np.floor((vx[k] - v_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((vx[k] - v_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         power_vx[istart:iend] += power_k[k] * gaussian(v_axis[istart:iend], vx[k], vmags[k], stdev=spread)
    #         normalizing_vx[istart:iend] += (chi+el_df['k_FD'])[k] * gaussian(v_axis[istart:iend], vx[k], vmags[k], stdev=spread)
    #     vxplot.plot(v_axis * 1E-3, power_vx/normalizing_vx, color=colors[i], label='{:.1f} V/cm'.format(ee / 100),alpha = alpha)
    #     vxplot.axvline(dv * 1E-3, color=colors[i], linestyle='--')
    # vxplot.set_xlabel(r'Longitudinal group velocity ($\rm km \, s^{-1}$)')
    # vxplot.set_ylabel(r'Energy above thermal ($\rm ev \, s \, m^{-1}$)')
    # vxplot.legend()
    # plt.savefig(pp.figureLoc + 'excessEnergyKDE.png', bbox_inches='tight', dpi=600)
    #
    # vx = el_df['vy [m/s]']
    # v_axis = np.linspace(vx.min(), vx.max(), npts)
    # dx = (v_axis.max() - v_axis.min()) / npts
    # spread = 40 * dx
    #
    # plt.figure()
    # vxplot = plt.axes()
    # for i, ee in enumerate(big_e):
    #     g_k = np.real(np.load(pp.outputLoc + '/SB_Density/yy_3_f_{:.1e}_E_{:.1e}.npy'.format(freq, ee)))
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     chi_t_i = np.load(pp.outputLoc + '/Transient/chi_3_f_{:.1e}_E_{:.1e}.npy'.format(freq, ee))
    #     dv = utilities.mean_velocity(chi, el_df)
    #     noise_k = 2 * (2 * c.e / pp.kgrid**3 / c.Vuc)**2 * np.real(g_k * vx)
    #     noise_vx = np.zeros(npts)
    #     density_kx = np.zeros(npts)
    #     for k in range(len(noise_k)):
    #         istart = int(np.maximum(np.floor((vx[k] - v_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((vx[k] - v_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         noise_vx[istart:iend] += noise_k[k] * gaussian(v_axis[istart:iend], vx[k], vmags[k], stdev=spread)
    #         density_kx[istart:iend] += chi[k] * gaussian(v_axis[istart:iend], vx[k], vmags[k], stdev=spread)
    #     vxplot.plot(v_axis * 1E-3, noise_vx, color=colors[i], label='{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$',alpha = alpha)
    #     # vxplot.plot(v_axis * 1E-3, noise_vx, color=colors[4+i],lw=lw)
    # vxplot.axvline(0, color='black', linestyle='--')
    # # vxplot.plot(v_axis * 1E-3, density_kx/np.min(density_kx), color=colors[4+i], label='{:.1f} V/cm'.format(ee / 100))
    # vxplot.set_xlabel(r'Transverse group velocity ($\rm km \, s^{-1}$)')
    # vxplot.set_ylabel(r'Transverse noise weight '+ r'$\rm  (A^2 \, m^3 \, Hz^{-2})$')
    # vxplot.legend(loc='upper left',frameon=False)
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    # freqTHz = freq/1000
    # # textstr = 'f = % .0f THz' %freqTHz
    # textstr = 'f = % .0f GHz' %freq
    # vxplot.text(0.98,0.98,textstr, transform=vxplot.transAxes, fontsize=8, verticalalignment='top', bbox=props,ha ='right',va='top')
    # # plt.ylim([0,0.017])
    # # plt.ylim([0,2.25e-5])
    # plt.savefig(pp.figureLoc+'Transverse_noiseKDE.png', bbox_inches='tight',dpi=600)
    #
    # print('Thermal energy is {:.3f} eV'.format(thermal_energy))
    # plt.figure(figsize=triFigSize)
    # plt.plot(el_df['vx [m/s]'],el_df['energy [eV]'],'.')
    # plt.xlabel(r'Transverse group velocity ($\rm km \, s^{-1}$)')
    # plt.ylabel(r'State energy (eV)')
    # plt.savefig(pp.figureLoc+'energy_vx.png', bbox_inches='tight',dpi=600)

    return noise_k


def field_dependence_noise(big_e, el_df):
    """Plot velocity variance and effective scattering rate versus field"""
    pA = 4.4481e-23 # 13 Problem
    pB = 1000 * pA  # Energy RT coefficient
    eRT_Ratio = 65
    inverseFrohlichTime = 5.8271028e12 # 13 Problem
    opticalPhononEnergy = 35/1000*c.e

    g_inds, _, _ = utilities.gaas_split_valleys(el_df, False)
    g_df = el_df.loc[g_inds]
    g_df.reset_index(drop=True)
    carrierEnergy = (g_df['energy [eV]'].values - np.min(g_df['energy [eV]']))
    incr_en = np.argsort(carrierEnergy)

    nkpts = len(np.unique(el_df['k_inds']))
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    sr = np.diag(scm)
    tau = -1 * sr**-1
    tau_adp, _ = davydov_utilities.acoustic_davydovRTs(g_df, pA, eRT_Ratio)
    tau_froh = davydov_utilities.frohlichScattering(g_df, opticalPhononEnergy, inverseFrohlichTime)
    tau_froh = tau_froh**-1

    # def davydov_f1(davdist):
        # c.e *

    fig, ax = plt.subplots()
    variance = []
    tau_eff = []
    for e in big_e:
        chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(e))
        fs = el_df['k_FD'] + chi
        fs_froh = davydov_utilities.frohlich_davydovDistribution(g_df.reset_index(drop=True), e, opticalPhononEnergy, inverseFrohlichTime, eRT_Ratio)
        fs_adp = davydov_utilities.acoustic_davydovDistribution(g_df.reset_index(drop=True), e, pA, eRT_Ratio)
        var_pert = np.var(fs * el_df['vx [m/s]'])
        var_adp = np.var(fs_adp * g_df['vx [m/s]'])
        var_froh = np.var(fs_froh * g_df['vx [m/s]'].values[incr_en])
        variance.append([var_pert, var_adp, var_froh])
        tau_eff.append([np.sum(fs*tau)/np.sum(fs), np.sum(fs_adp*tau_adp)/np.sum(fs_adp), np.sum(fs_froh*tau_froh[incr_en])/np.sum(fs_froh)])
        ax.semilogy(g_df['energy [eV]'], fs_adp, '.', markersize=4)
    variance = np.array(variance)
    tau_eff = np.array(tau_eff) * 1e15

    big_e = big_e * 1e-5
    plt.figure()
    plt.plot(big_e, tau_eff[:, 0], label='Perturbo', color='C1')
    plt.plot(big_e, tau_eff[:, 1], label='ADP', linestyle='dotted', color='Black')
    plt.plot(big_e, tau_eff[:, 2], label='Frohlich', linestyle='dashed', color='Black')
    plt.ylabel('Weighted lifetime [fs]')
    plt.xlabel('Field [kV/cm]')
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(big_e, variance[:, 0], label='Perturbo', color='C1')
    plt.plot(big_e, variance[:, 1], label='ADP', linestyle='dotted', color='Black')
    plt.plot(big_e, variance[:, 2], label='Frohlich', linestyle='dashed', color='Black')
    plt.ylabel('Velocity Variance [(m/s)^2]')
    plt.xlabel('Field [kV/cm]')
    plt.legend()
    plt.tight_layout()


def mom_en_relaxation(el_df, fields):
    enk = el_df['energy [eV]'].values
    enk = enk - enk.min()
    kx = el_df['kx [1/A]'].values / (2 * np.pi / c.alat)
    vmags = el_df['v_mag [m/s]'].values
    nkpts = len(np.unique(el_df['k_inds']))
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    g_inds, l_inds, x_inds = utilities.gaas_split_valleys(el_df, plot_Valleys=False)

    xtext = 'Electron energy (eV)'
    ytext = r'Average fractional momentum change'
    valleycolors = ['#004369', '#01949A']
    if len(fields) < 10:
        cmap = plt.cm.get_cmap('YlOrRd', len(fields) + 2)
    else:
        cmap = plt.cm.get_cmap('inferno', len(fields) + 2)
    font = {'size': 11}
    mpl.rc('font', **font)
    mpl.rcParams['savefig.dpi'] = 300
    fieldcolors = []
    for i in range(cmap.N):
        rgb = cmap(i)[:3]  # will return rgba, we take only first 3 so we get rgb
        fieldcolors.append(mpl.colors.rgb2hex(rgb))

    npts = 400  # number of points in the KDE
    en_axis = np.linspace(enk.min(), enk.max(), npts)
    k_axis = np.linspace(kx.min(), kx.max(), npts)
    dx = (en_axis.max() - en_axis.min()) / npts
    dkx = (k_axis.max() - k_axis.min()) / npts
    spread = 35 * dx
    kspread = 35 * dkx

    def gaussian(x, mu, stdev=kspread):
        vals = (1 / (stdev * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / stdev) ** 2)
        return vals

    def smeared_gauss(x, mu, vmag, stdev=spread):
        sigma = stdev - (vmag / 1E6) * 0.90 * stdev
        vals = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)
        return vals

    # allk = el_df[['kx [1/A]', 'ky [1/A]', 'kz [1/A]']].values / (2 * np.pi / c.alat)
    kx_abs = np.abs(el_df['kx [1/A]'].values / (2 * np.pi / c.alat))
    # allk = np.linalg.norm(el_df[['kx [1/A]', 'ky [1/A]', 'kz [1/A]']].values / (2 * np.pi / c.alat), axis=1)
    decaymatrix1 = np.zeros((nkpts, nkpts))
    decaymatrix2 = np.zeros((nkpts, nkpts))
    for ik in range(nkpts):
        # deltak = np.linalg.norm(allk - np.tile(allk[ik, :], (nkpts, 1)), axis=1)
        # deltak = allk[ik] - allk  # |k| - |k'| so positive is momentum decay
        # deltak = kx_abs[ik] - kx_abs
        deltak = kx[ik] - kx
        delta_en = enk[ik] - enk
        if enk[ik] == 0:
            continue
        colk = scm[:, ik].copy()
        rowk = scm[ik, :].copy()
        # Should remove the diagonal contribution, but since the deltak for element k is zero, cancels out
        # decaymatrix[ik, :] = (deltak * rowk * -1) / np.linalg.norm(allk[ik]) / (-1 * scm[ik, ik])
        # decaymatrix[ik, ik] = np.dot(deltak, colk) / np.linalg.norm(allk[ik]) / (-1 * scm[ik, ik])
        decaymatrix1[ik, :] = (deltak * rowk * -1)
        decaymatrix1[ik, ik] = np.dot(deltak, colk)
        # decaymatrix[ik, :] = (delta_en * rowk * -1) / (-1 * scm[ik, ik]) / enk[ik]
        # decaymatrix[ik, ik] = np.dot(delta_en, colk) / (-1 * scm[ik, ik]) / enk[ik]
        decaymatrix2[ik, :] = (delta_en * rowk * -1)
        decaymatrix2[ik, ik] = np.dot(delta_en, colk)
        if ik % 1000 == 0:
            print('Did k = {}'.format(str(ik)))

    # Plot of loss versus fields
    plt.figure()
    ax1 = plt.axes()
    plt.tight_layout()
    plt.figure()
    ax2 = plt.axes()
    plt.tight_layout()
    plt.figure()
    ax3 = plt.axes()
    plt.tight_layout()
    te = np.zeros(len(fields))
    tm = np.zeros(len(fields))
    excess_en = np.zeros(len(fields))
    excess_kx = np.zeros(len(fields))
    dfkbelow = np.zeros(len(fields))
    dfkscale = np.zeros(len(fields))
    avg_en_lost = np.zeros(len(fields))
    avg_kx_lost = np.zeros(len(fields))
    for i, field in enumerate(fields):
        chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(field))
        mvp = scm @ chi
        excess_en[i] = np.sum(chi * enk)
        excess_kx[i] = np.sum(chi * kx)
        full_kx = (decaymatrix1 @ chi) / excess_kx[i]
        full_en = (decaymatrix2 @ chi) / excess_en[i]
        # full_kx = (decaymatrix1 @ chi)
        # full_en = (decaymatrix2 @ chi)
        # full_kx = (decaymatrix1 @ chi) / np.abs(mvp)
        # full_en = (decaymatrix2 @ chi) / np.abs(mvp)
        avg_kx_lost[i] = np.average(full_kx)
        avg_en_lost[i] = np.average(full_en)
        dfkbelow[i] = np.sum(chi[enk < 0.040])
        full_kx_kde = np.zeros(npts)
        full_en_kde = np.zeros(npts)
        mvp_kde = np.zeros(npts)
        for k in range(nkpts):
            # if np.isin(k, l_inds):
            #     continue
            istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
            iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
            full_kx_kde[istart:iend] += full_kx[k] * smeared_gauss(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
            full_en_kde[istart:iend] += full_en[k] * smeared_gauss(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
            mvp_kde[istart:iend] += np.abs(mvp[k]) * smeared_gauss(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
        ax1.plot(en_axis, full_kx_kde, label='{:.0f} V/cm'.format(field/100), color=fieldcolors[i+1])
        ax2.plot(en_axis, full_en_kde, label='{:.0f} V/cm'.format(field/100), color=fieldcolors[i+1])
        ax3.plot(en_axis, np.abs(mvp_kde), label='{:.0f} V/cm'.format(field/100), color=fieldcolors[i + 1])
        tm[i] = np.trapz(full_kx_kde, en_axis)
        te[i] = np.trapz(full_en_kde, en_axis)
        momenratio = tm[i] / te[i]
        print('At {:.0f} V/cm, the momentum:energy relaxation ratio is {:.1f}'.format(field/100, momenratio))
    ax1.axhline(0, linestyle='--', color='Black', lw=0.5)
    # ax1.set_ylabel('Momentum loss (norm by excess kx)')
    ax1.set_ylabel('Mom. loss / total scattering')
    ax1.set_xlabel(xtext)
    # ax1.legend()
    ax2.axhline(0, linestyle='--', color='Black', lw=0.5)
    # ax2.set_ylabel('Energy loss (norm by excess en)')
    ax2.set_ylabel('En. loss / total scattering')
    ax2.set_xlabel(xtext)
    # ax2.legend()
    # ax3.axhline(0, linestyle='--', color='Black', lw=0.5)
    ax3.set_ylabel(r'Total scattering at given energy (1/s)')
    ax3.set_xlabel(xtext)

    plt.figure()
    plt.plot(fields/100, avg_en_lost, label='Energy')
    plt.plot(fields/100, avg_kx_lost, label='Momentum')
    plt.legend()
    plt.ylabel('Avg quantity lost')
    plt.xlabel('Fields')

    plt.figure()
    plt.plot(fields/100, tm * 1e-12, label='Momentum')
    plt.plot(fields/100, te * 1e-12, label='Energy')
    plt.plot(fields/100, (te[0]*1e-12) * (dfkbelow/dfkbelow[0]) / (excess_en/excess_en[0]),
             linestyle='--', color='C1', label='Extrapolation')
    plt.plot(fields / 100, (tm[0] * 1e-12) * (fields/fields[0]) / (excess_kx/excess_kx[0]),
             linestyle='--', color='C0', label='Extrapolation')
    plt.ylabel('Relaxation rate (1/ps)')
    plt.xlabel('Fields (V/cm)')
    plt.legend(frameon=False)

    # plt.figure()
    # plt.plot(fields / 100, (odchange-1)*100, label='Momentum')
    # plt.ylabel('Pct. increase full vs diag')
    # plt.xlabel('Fields (V/cm)')
    # plt.ylim([100, 120])

    # # Plot of diagonal versus off diagonal momentum loss for chi of field specified below
    # thisfield = fields[3]
    # print('Doing diagonal/off diagonal comparison for {:.1f} V/cm'.format(thisfield/100))
    # chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(thisfield))
    # excess_en = np.sum(chi * enk)
    # excess_kx = np.sum(chi * kx)
    # full_kx = (decaymatrix1 @ chi) / excess_kx
    # diag_kx = np.diag(decaymatrix1) * chi / excess_kx
    # full_en = (decaymatrix2 @ chi) / excess_en
    # diag_en = np.diag(decaymatrix2) * chi / excess_en
    # full_kx_kde = np.zeros(npts)
    # diag_kx_kde = np.zeros(npts)
    # full_en_kde = np.zeros(npts)
    # diag_en_kde = np.zeros(npts)
    # # k_diagdecay = np.zeros(npts)
    # # k_full_enax = np.zeros(npts)
    # for k in range(nkpts):
    #     istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
    #     iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #     full_kx_kde[istart:iend] += full_kx[k] * smeared_gauss(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
    #     diag_kx_kde[istart:iend] += diag_kx[k] * smeared_gauss(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
    #     full_en_kde[istart:iend] += full_en[k] * smeared_gauss(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
    #     diag_en_kde[istart:iend] += diag_en[k] * smeared_gauss(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
    #     # istart = int(np.maximum(np.floor((kx[k] - k_axis[0]) / dkx) - (4 * kspread / dkx), 0))
    #     # iend = int(np.minimum(np.floor((kx[k] - k_axis[0]) / dkx) + (4 * kspread / dkx), npts - 1))
    #     # k_diagdecay[istart:iend] += diagdecay_fs[k] * gaussian(en_axis[istart:iend], kx[k], stdev=kspread)
    #     # k_full_enax[istart:iend] += fulldecay[k] * gaussian(en_axis[istart:iend], kx[k], stdev=kspread)
    # plt.figure()
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.plot(en_axis, diag_kx_kde, label='Diagonal only', color=fieldcolors[i+1], linestyle='--')
    # plt.plot(en_axis, full_kx_kde, label='Including off-diagonal', color=fieldcolors[i+1])
    # plt.title('Field is {:.0f} V/cm'.format(thisfield/100))
    # plt.ylabel('Momentum loss (norm by excess kx)')
    # plt.xlabel(xtext)
    # plt.legend()
    # plt.tight_layout()
    # pctdiff = np.trapz(full_kx_kde, en_axis) / np.trapz(diag_kx_kde, en_axis)
    # print('The full solution for momentum is {:.1f}% of the diagonal solution'.format((pctdiff)*100))
    # plt.figure()
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.plot(en_axis, diag_en_kde, label='Diagonal only', color=fieldcolors[i + 1], linestyle='--')
    # plt.plot(en_axis, full_en_kde, label='Including off-diagonal', color=fieldcolors[i + 1])
    # plt.title('Field is {:.0f} V/cm'.format(thisfield / 100))
    # plt.ylabel('Energy loss (norm by excess en)')
    # plt.xlabel(xtext)
    # plt.legend()
    # plt.tight_layout()
    # pctdiff = np.trapz(full_en_kde, en_axis) / np.trapz(diag_en_kde, en_axis)
    # print('The full solution for energy is {:.1f}% of the diagonal solution'.format((pctdiff) * 100))

    # Plot of momentum loss rates parametrically against energy
    # diagdecayrate = np.diag(decaymatrix1)
    # plt.figure()
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # # plt.axhline(4e12, linestyle='--', color='Black', linewidth=1, alpha=0.6)
    # plt.plot(enk[g_inds], diagdecayrate[g_inds], '.', label='Gamma', markersize=3.5, color=valleycolors[0])
    # # plt.plot(enk[l_inds], diagdecayrate[l_inds], '.', label='L', markersize=3.5, color=valleycolors[1])
    # plt.ylabel('Momentum loss')
    # plt.xlabel(xtext)
    # legend_elements = [Line2D([0], [0], marker='o', lw=0, color=valleycolors[0], label='Gamma', markersize=5),
    #                    Line2D([0], [0], marker='o', lw=0, color=valleycolors[1], label='L', markersize=5)]
    # plt.legend(handles=legend_elements)
    # plt.tight_layout()

    # Plot of momentum loss rates parametrically against energy
    diagdecayrate = np.diag(decaymatrix2)
    plt.figure()
    plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    plt.axvline(0.040, linestyle='--', color='Black', linewidth=0.5)
    # plt.axhline(4e12, linestyle='--', color='Black', linewidth=1, alpha=0.6)
    plt.plot(enk[g_inds], diagdecayrate[g_inds], '.', label='Gamma', markersize=3.5, color=valleycolors[0])
    # plt.plot(enk[l_inds], diagdecayrate[l_inds], '.', label='L', markersize=3.5, color=valleycolors[1])
    plt.ylabel(r'Energy loss $\Gamma$ only (eV/s/eV)')
    plt.xlabel(xtext)
    legend_elements = [Line2D([0], [0], marker='o', lw=0, color=valleycolors[0], label='Gamma', markersize=5),
                       Line2D([0], [0], marker='o', lw=0, color=valleycolors[1], label='L', markersize=5)]
    # plt.legend(handles=legend_elements)
    plt.tight_layout()
    # # plt.savefig(pp.figureLoc + 'avg_energy_loss.png', bbox_inches='tight', dpi=400)

    # Plot of loss rates in momentum space
    diagdecayrate = np.diag(decaymatrix1)
    plt.figure()
    plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.axhline(4e12, linestyle='--', color='Black', linewidth=1, alpha=0.6)
    plt.plot(kx[g_inds], diagdecayrate[g_inds], '.', label='Gamma', markersize=3.5, color=valleycolors[0])
    # plt.plot(kx[l_inds], diagdecayrate[l_inds], '.', label='L', markersize=3.5, color=valleycolors[1])
    plt.ylabel(r'kx momentum loss $\Gamma$ only (1/s)')
    plt.xlabel('kx')
    legend_elements = [Line2D([0], [0], marker='o', lw=0, color=valleycolors[0], label='Gamma', markersize=5),
                       Line2D([0], [0], marker='o', lw=0, color=valleycolors[1], label='L', markersize=5)]
    plt.legend(handles=legend_elements)
    plt.tight_layout()

    # # Plot of momentum loss rates diagonal vs off diagonal
    # diagdecayrate = np.diag(decaymatrix2)
    # incl_off_loss = np.sum(decaymatrix2, axis=1)
    # plt.figure()
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # # plt.axhline(4e12, linestyle='--', color='Black', linewidth=1, alpha=0.6)
    # plt.plot(enk, diagdecayrate, '.', markersize=3.5, color='gray')
    # plt.plot(enk, incl_off_loss, '.', markersize=3.5, color='black')
    # plt.ylabel('Energy loss')
    # plt.xlabel(xtext)
    # legend_elements = [Line2D([0], [0], marker='o', lw=0, color='gray', label='diagonal', markersize=6),
    #                    Line2D([0], [0], marker='o', lw=0, color='black', label='offdiagonal', markersize=6)]
    # plt.legend(handles=legend_elements)
    # plt.tight_layout()

    # # Plot of momentum loss rates diagonal vs off diagonal
    # diagdecayrate = np.diag(decaymatrix1)
    # incl_off_loss = np.sum(decaymatrix1, axis=1)
    # plt.figure()
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # # plt.axhline(4e12, linestyle='--', color='Black', linewidth=1, alpha=0.6)
    # plt.plot(enk, diagdecayrate, '.', markersize=3.5, color='gray')
    # plt.plot(enk, incl_off_loss, '.', markersize=3.5, color='black')
    # plt.ylabel('Momentum loss')
    # plt.xlabel(xtext)
    # legend_elements = [Line2D([0], [0], marker='o', lw=0, color='gray', label='diagonal', markersize=6),
    #                    Line2D([0], [0], marker='o', lw=0, color='black', label='offdiagonal', markersize=6)]
    # plt.legend(handles=legend_elements)
    # plt.tight_layout()


def anisotropy_of_scattering(el_df):
    enk = el_df['energy [eV]'].values
    enk = enk - enk.min()
    kx = el_df['kx [1/A]']
    nkpts = len(np.unique(el_df['k_inds']))
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    g_inds, l_inds, x_inds = utilities.gaas_split_valleys(el_df, plot_Valleys=False)

    ytext = r'Total energy loss (eV/s)'
    xtext = 'kx (1/A) for state k'
    valleycolors = ['#01949A', '#004369']

    allk = el_df[['kx [1/A]', 'ky [1/A]', 'kz [1/A]']].values / (2 * np.pi / c.alat)
    kmag = np.linalg.norm(allk, axis=1)
    # scatteredk = np.zeros(nkpts)
    scatteredk = np.zeros((nkpts, 3))
    avgangle = np.zeros(nkpts)
    for ik in range(nkpts):
        if ik == 0:
            continue
        colk = scm[:, ik].copy()
        colk[ik] = 0
        kpi = np.nonzero(colk)
        # avgkprime = np.sum(allk * np.tile(colk[:, np.newaxis], (1, 3)), axis=0)  # sum of all k' weighted by theta
        # scatteredk[ik, :] = avgkprime / scm[ik, ik] * -1  # avg k' magnitude, dividing out theta_tot
        # scatteredk[ik] = np.linalg.norm(avgkprime) / scm[ik, ik] * -1  # avg k' magnitude, dividing out theta_tot
        # kprimes = allk[kpi] * np.tile(colk[kpi[0], np.newaxis], (1, 3)) / scm[ik, ik] * -1
        dotprods = np.sum(allk[kpi] * np.tile(allk[ik, :], (len(kpi), 1)), axis=1)
        kprimemag = np.linalg.norm(allk[kpi], axis=1)
        angles = np.degrees(np.arccos(dotprods / kprimemag / np.linalg.norm(allk[ik, :])))
        avgangle[ik] = np.average(angles[~np.isnan(angles)] * colk[kpi][~np.isnan(angles)] / scm[ik, ik] * -1)
        if ik % 1000 == 0:
            print('Did k = {}'.format(str(ik)))
    # kprimemag = np.linalg.norm(scatteredk, axis=1)
    # cosangle = np.degrees(np.arccos(np.sum(allk * scatteredk, axis=1) / kprimemag / kmag))  # cosine angle of the dot product

    plt.figure()
    # plt.plot(kx[g_inds], cosangle[g_inds], '.', label='Gamma', markersize=3.5, color=valleycolors[0])
    # plt.plot(kx[l_inds], cosangle[l_inds], '.', label='L', markersize=3.5, color=valleycolors[1])
    plt.plot(kx[g_inds], avgangle[g_inds], '.', label='Gamma', markersize=3.5, color=valleycolors[0])
    plt.plot(kx[l_inds], avgangle[l_inds], '.', label='L', markersize=3.5, color=valleycolors[1])
    plt.ylabel('Angle between k\'_avg and k', fontsize=12)
    plt.xlabel(xtext, fontsize=11)
    legend_elements = [Line2D([0], [0], marker='o', lw=0, color=valleycolors[0], label='Gamma', markersize=5),
                       Line2D([0], [0], marker='o', lw=0, color=valleycolors[1], label='L', markersize=5)]
    plt.legend(handles=legend_elements)


def mom_en_rel_histograms(el_df):
    enk = el_df['energy [eV]'].values
    enk = enk - enk.min()
    kx = el_df['kx [1/A]'].values / (2 * np.pi / c.alat)
    vmags = el_df['v_mag [m/s]'].values
    nkpts = len(np.unique(el_df['k_inds']))
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    g_inds, l_inds, x_inds = utilities.gaas_split_valleys(el_df, plot_Valleys=False)

    font = {'size': 11}
    mpl.rc('font', **font)
    mpl.rcParams['savefig.dpi'] = 300

    allk = el_df[['kx [1/A]', 'ky [1/A]', 'kz [1/A]']].values / (2 * np.pi / c.alat)
    diagloss_k = np.zeros(nkpts)  # Momentum
    diagloss_en = np.zeros(nkpts)  # Energy
    for ik in range(nkpts):
        deltak = np.linalg.norm(allk - np.tile(allk[ik, :], (nkpts, 1)), axis=1)
        delta_en = enk[ik] - enk
        if enk[ik] == 0:
            continue
        colk = scm[:, ik].copy()
        # Should remove the diagonal contribution, but since the deltak for element k is zero, cancels out
        diagloss_k[ik] = np.dot(deltak, colk) / (-1 * scm[ik, ik])
        diagloss_en[ik] = np.dot(delta_en, colk) / (-1 * scm[ik, ik])
        if ik % 1000 == 0:
            print('Did k = {}'.format(str(ik)))
    diagloss_en = diagloss_en * 1000  # so it's in meV instead of eV

    # Parametrically plotting just to see
    plt.figure()
    plt.plot(enk[g_inds], diagloss_en[g_inds] * 1000, '.', markersize=3.5, label='Gamma')
    plt.plot(enk[l_inds], diagloss_en[l_inds] * 1000, '.', markersize=3.5, label='L')
    plt.xlabel('Electron energy (eV)')
    plt.ylabel('Avg. en. lost per scattering (meV)')
    plt.legend()
    plt.figure()
    plt.plot(enk[g_inds], diagloss_k[g_inds], '.', markersize=3.5, label='Gamma')
    plt.plot(enk[l_inds], diagloss_k[l_inds], '.', markersize=3.5, label='L')
    plt.xlabel('Electron energy (eV)')
    plt.ylabel('Avg. mom. lost per scattering (norm 2pi/a)')
    plt.legend()

    # Histograms
    nbins = 50
    en_width = (diagloss_en.max() - diagloss_en.min()) / (nbins+10)  # adding 10 to divisor to give a little gap in bars
    k_width = (diagloss_k.max() - diagloss_k.min()) / (nbins+10)
    en_bins = np.linspace(diagloss_en.min(), diagloss_en.max() + en_width, nbins)
    k_bins = np.linspace(diagloss_k.min(), diagloss_k.max() + k_width, nbins)
    en_bin_inds = np.searchsorted(en_bins, diagloss_en, side='right')
    k_bin_inds = np.searchsorted(k_bins, diagloss_k, side='right')
    en_hist = np.zeros(nbins)
    k_hist = np.zeros(nbins)
    for i in range(len(en_bin_inds)):
        en_hist[en_bin_inds[i]] += 1
        k_hist[k_bin_inds[i]] += 1

    plt.figure()
    plt.bar(en_bins, en_hist, width=en_width)
    plt.ylabel('Number of states that lose given avg. energy')
    plt.xlabel('Average energy lost per scattering event')
    plt.tight_layout()
    plt.figure()
    plt.bar(k_bins, k_hist, width=k_width)
    plt.ylabel('Number of states that lose given avg. momentum')
    plt.xlabel('Average momentum lost per scattering event')
    plt.tight_layout()


def energy_KDEs(el_df, fields):
    lw = 1.5
    nkpts = len(np.unique(el_df['k_inds']))
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    ytext = r'Momentum loss (1/s)'
    xtext = 'Electron energy (eV)'
    valleycolors = ['#004369', '#01949A']
    cmap = plt.cm.get_cmap('YlOrRd', 6)
    fieldcolors = []
    # for i in range(cmap.N):
    #     rgb = cmap(i)[:3]  # will return rgba, we take only first 3 so we get rgb
    #     fieldcolors.append(mpl.colors.rgb2hex(rgb))

    fieldcolors = ['#0C1446', '#2740DA', '#FF3F00']

    npts = 400  # number of points in the KDE

    neg_k_inds = el_df['kx [1/A]'] < 0
    neg_enk = el_df.loc[neg_k_inds,'energy [eV]'].values
    neg_enk = neg_enk - neg_enk.min()
    neg_en_axis = np.linspace(neg_enk.min(), neg_enk.max(), npts)
    neg_vmags = el_df.loc[neg_k_inds,'v_mag [m/s]'].values
    dx = (neg_en_axis.max() - neg_en_axis.min()) / npts
    spread = 35 * dx

    def gaussian(x, mu, vmag, stdev=spread):
        sigma = stdev - (vmag / 1E6) * 0.90 * stdev
        vals = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)
        return vals

    T_vector = np.geomspace(300,500,1000)
    energy_vector = paper_figures.calculate_electron_temperature(el_df,T_vector)
    carrierEnergy = (el_df['energy [eV]'] - np.min(el_df['energy [eV]']))

    plt.figure(figsize=triFigSize)
    for i,ee in enumerate(fields):
        chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
        chi3 = np.load(pp.outputLoc + '/Steady/chi_4_E_{:.1e}.npy'.format(ee))
        f3 = el_df['k_FD'] + chi3
        A3 = np.sum(el_df['k_FD']) / np.sum(f3)
        chi3 = chi3 * A3
        pertEnergy = np.sum(carrierEnergy * (chi + el_df['k_FD'])) / np.sum(el_df['k_FD'])
        pert_Temp = np.interp(pertEnergy, energy_vector, T_vector)
        hot_boltzdist = (np.exp((el_df['energy [eV]'].values * c.e - pp.mu * c.e) / (c.kb_joule * pert_Temp))) ** (-1)
        hot_boltzdist = hot_boltzdist * np.sum(el_df['k_FD']) / np.sum(hot_boltzdist)
        fulldecay = chi[neg_k_inds]
        fulldecay2 = hot_boltzdist[neg_k_inds] - el_df.loc[neg_k_inds,'k_FD'].values
        fulldecay4 = chi3[neg_k_inds]
        # Plot of the momentum loss KDE vs energy for the chi of field designated above
        decay = np.zeros(npts)
        decay2 = np.zeros(npts)
        decay4 = np.zeros(npts)
        # for k in np.nonzero(pos_k_inds)[0]:
        for k in range(len(fulldecay)):
            istart = int(np.maximum(np.floor((neg_enk[k] - neg_en_axis[0]) / dx) - (4 * spread / dx), 0))
            iend = int(np.minimum(np.floor((neg_enk[k] - neg_en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
            decay[istart:iend] += fulldecay[k] * gaussian(neg_en_axis[istart:iend], neg_enk[k], neg_vmags[k], stdev=spread)
            decay2[istart:iend] += fulldecay2[k] * gaussian(neg_en_axis[istart:iend], neg_enk[k], neg_vmags[k], stdev=spread)
            decay4[istart:iend] += fulldecay4[k] * gaussian(neg_en_axis[istart:iend], neg_enk[k], neg_vmags[k],
                                                            stdev=spread)
        plt.plot(neg_en_axis*1000, decay, color=fieldcolors[i], label='Hot+Full{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
        # plt.plot(neg_en_axis*1000, decay2, '-.',color=fieldcolors[i+1], label='Hot M-B{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
        plt.plot(neg_en_axis*1000, decay4, '-.',color=fieldcolors[i], label='Hot+RTA{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')

    plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    plt.ylabel(r'$\Delta f_{k}$')
    plt.xlabel('Electron energy (meV)')
    plt.title(r'Negative $k_x$')
    plt.legend(frameon=False)
    plt.savefig(pp.figureLoc+'energy_KDE_neg.png', dpi=600)

    pos_k_inds = el_df['kx [1/A]'] > 0
    pos_enk = el_df.loc[pos_k_inds,'energy [eV]'].values
    pos_enk = pos_enk - pos_enk.min()
    pos_en_axis = np.linspace(pos_enk.min(), pos_enk.max(), npts)
    pos_vmags = el_df.loc[pos_k_inds,'v_mag [m/s]'].values
    dx = (pos_en_axis.max() - pos_en_axis.min()) / npts
    spread = 35 * dx

    plt.figure(figsize=triFigSize)
    for i,ee in enumerate(fields):
        chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
        chi3 = np.load(pp.outputLoc + '/Steady/chi_4_E_{:.1e}.npy'.format(ee))
        f3 = el_df['k_FD']+chi3
        A3 = np.sum(el_df['k_FD'])/np.sum(f3)
        chi3 = chi3*A3
        pertEnergy = np.sum(carrierEnergy * (chi + el_df['k_FD'])) / np.sum(el_df['k_FD'])
        pert_Temp = np.interp(pertEnergy, energy_vector, T_vector)
        hot_boltzdist = (np.exp((el_df['energy [eV]'].values * c.e - pp.mu * c.e) / (c.kb_joule * pert_Temp))) ** (-1)
        hot_boltzdist = hot_boltzdist * np.sum(el_df['k_FD']) / np.sum(hot_boltzdist)
        fulldecay = chi[pos_k_inds]
        fulldecay2 = hot_boltzdist[pos_k_inds] - el_df.loc[pos_k_inds,'k_FD'].values
        fulldecay4 = chi3[pos_k_inds]
        # Plot of the momentum loss KDE vs energy for the chi of field designated above
        decay = np.zeros(npts)
        decay2 = np.zeros(npts)
        decay4 = np.zeros(npts)
        # for k in np.nonzero(pos_k_inds)[0]:
        for k in range(len(fulldecay)):
            istart = int(np.maximum(np.floor((pos_enk[k] - pos_en_axis[0]) / dx) - (4 * spread / dx), 0))
            iend = int(np.minimum(np.floor((pos_enk[k] - pos_en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
            decay[istart:iend] += fulldecay[k] * gaussian(pos_en_axis[istart:iend], pos_enk[k], pos_vmags[k], stdev=spread)
            decay2[istart:iend] += fulldecay2[k] * gaussian(pos_en_axis[istart:iend], pos_enk[k], pos_vmags[k],
                                                          stdev=spread)
            decay4[istart:iend] += fulldecay4[k] * gaussian(pos_en_axis[istart:iend], pos_enk[k], pos_vmags[k],
                                                          stdev=spread)
        plt.plot(pos_en_axis*1000, decay, color=fieldcolors[i], label='Hot+Full {:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
        # plt.plot(pos_en_axis*1000, decay2,'-.', color=fieldcolors[i+1], label='Hot M-B {:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
        plt.plot(pos_en_axis*1000, decay4,'-.', color=fieldcolors[i], label='Hot_RTA {:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')

    plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    plt.ylabel(r'$\Delta f_{k}$')
    plt.xlabel('Electron energy (meV)')
    plt.title(r'Positive $k_x$')
    plt.legend(frameon=False)
    plt.savefig(pp.figureLoc+'energy_KDE_pos.png', dpi=600)

    enk = el_df['energy [eV]']
    enk = enk - enk.min()
    vmags = el_df['v_mag [m/s]']
    en_axis = np.linspace(pos_enk.min(), pos_enk.max(), npts)
    dx = (en_axis.max() - en_axis.min()) / npts
    spread = 55 * dx

    plt.figure(figsize=squareFigSize)
    ax = plt.axes([0.24, 0.21, 0.71, 0.71])
    for i, ee in enumerate(fields):
        chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
        chi2 = np.load(pp.outputLoc + '/Steady/chi_2_E_{:.1e}.npy'.format(ee))
        chi3 = np.load(pp.outputLoc + '/Steady/chi_4_E_{:.1e}.npy'.format(ee))

        f3 = el_df['k_FD']+chi3
        A3 = np.sum(el_df['k_FD'])/np.sum(f3)
        chi3 = chi3*A3

        print('checksum')
        print(np.sum(f3*A3))
        print(np.sum(chi+el_df['k_FD']))

        pertEnergy = np.sum(carrierEnergy * (chi + el_df['k_FD'])) / np.sum(el_df['k_FD'])
        pert_Temp = np.interp(pertEnergy, energy_vector, T_vector)
        hot_boltzdist = (np.exp((el_df['energy [eV]'].values * c.e - pp.mu * c.e) / (c.kb_joule * pert_Temp))) ** (-1)
        hot_boltzdist = hot_boltzdist*np.sum(el_df['k_FD'])/np.sum(hot_boltzdist)
        print(pert_Temp)
        fulldecay = chi
        fulldecay2 = hot_boltzdist - el_df['k_FD']
        fulldecay3 = chi2
        fulldecay4 = chi3
        print(np.sum(fulldecay))
        print(np.sum(fulldecay2))
        # Plot of the momentum loss KDE vs energy for the chi of field designated above
        decay = np.zeros(npts)
        decay2 = np.zeros(npts)
        decay3 = np.zeros(npts)
        decay4 = np.zeros(npts)
        plotcolors = ['#4C5355','#98A4B0','#FF3F00']
        plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
        for k in range(len(fulldecay)):
            istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
            iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
            decay[istart:iend] += fulldecay[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
            decay2[istart:iend] += fulldecay2[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
            decay3[istart:iend] += fulldecay3[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
            decay4[istart:iend] += fulldecay4[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)

        # plt.plot(en_axis * 1000, decay, color=fieldcolors[i + 3],
        #          label='PERTURBO {:.1f} '.format(ee / 100) + r'$\rm V \, cm^{-1}$')
        # plt.plot(en_axis * 1000, decay2, '--',color=fieldcolors[i + 3],
        #          label='Hot M-B {:.1f} '.format(ee / 100) + r'$\rm V \, cm^{-1}$')
        # # plt.plot(en_axis * 1000, decay3, '-.',color=fieldcolors[i + 1],
        # #          label='{:.1f} '.format(ee / 100) + r'$\rm V \, cm^{-1}$')
        # plt.plot(en_axis * 1000, decay4, '-.',color=fieldcolors[i + 3],
        #          label='RTA {:.1f} '.format(ee / 100) + r'$\rm V \, cm^{-1}$')
        plt.plot(en_axis * 1000, decay4, '-', color=plotcolors[0],
                 label='RTA',linewidth=lw)
        plt.plot(en_axis * 1000, decay2, '-', color=plotcolors[1],
                 label='Hot M-B',linewidth=lw)
        # plt.plot(en_axis * 1000, decay3, '-.',color=fieldcolors[i + 1],
        #          label='{:.1f} '.format(ee / 100) + r'$\rm V \, cm^{-1}$')
        plt.plot(en_axis * 1000, decay, color=plotcolors[2],
                 label='PERTURBO',linewidth=lw)
    plt.ylabel(r'Deviational occupation $\rm \Delta f_{\mathbf{k}}$')
    plt.xlabel('Energy (meV)')
    # plt.title(r'All $k_x$')
    plt.xlim([0,np.max(en_axis)*1000])
    plt.legend(frameon=False)
    plt.savefig(pp.figureLoc+'energy_KDE_all.png', dpi=600)


    plt.figure(figsize=squareFigSize)
    for i, ee in enumerate(fields):
        chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
        pertEnergy = np.sum(carrierEnergy * (chi + el_df['k_FD'])) / np.sum(el_df['k_FD'])
        pert_Temp = np.interp(pertEnergy, energy_vector, T_vector)
        hot_boltzdist = (np.exp((el_df['energy [eV]'].values * c.e - pp.mu * c.e) / (c.kb_joule * pert_Temp))) ** (-1)
        hot_boltzdist = hot_boltzdist*np.sum(el_df['k_FD'])/np.sum(hot_boltzdist)
        print(pert_Temp)
        fulldecay = 2 * c.e / c.Vuc / ee * el_df['vx [m/s]'] * chi
        fulldecay2 = chi+el_df['k_FD']
        # Plot of the momentum loss KDE vs energy for the chi of field designated above
        decay = np.zeros(npts)
        decay2 = np.zeros(npts)
        plotcolors = ['#4C5355','#98A4B0','#FF3F00']
        plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
        for k in range(len(fulldecay)):
            istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
            iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
            decay[istart:iend] += fulldecay[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
            decay2[istart:iend] += fulldecay2[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
        plt.plot(en_axis * 1000, decay/decay2, color=plotcolors[i],
                 label='{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
    plt.ylabel(r'Conductivity contribution (Pop removed)')
    plt.xlabel('Energy (meV)')
    # plt.title(r'All $k_x$')
    plt.xlim([0,np.max(en_axis)*1000])
    plt.legend(frameon=False)
    plt.savefig(pp.figureLoc+'conductivity_energy_kde.png',dpi=600)

    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))

    plt.figure(figsize=squareFigSize)
    for i, ee in enumerate(fields):
        chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
        pertEnergy = np.sum(carrierEnergy * (chi + el_df['k_FD'])) / np.sum(el_df['k_FD'])
        pert_Temp = np.interp(pertEnergy, energy_vector, T_vector)
        hot_boltzdist = (np.exp((el_df['energy [eV]'].values * c.e - pp.mu * c.e) / (c.kb_joule * pert_Temp))) ** (-1)
        hot_boltzdist = hot_boltzdist*np.sum(el_df['k_FD'])/np.sum(hot_boltzdist)
        print(pert_Temp)
        fulldecay = chi/ee/(el_df['k_FD']*(1-el_df['k_FD']))/el_df['v_mag [m/s]']*el_df['vx [m/s]']
        fulldecay2 = np.ones(len(el_df))
        fulldecay3 = el_df['k_FD']+chi
        # Plot of the momentum loss KDE vs energy for the chi of field designated above
        decay = np.zeros(npts)
        decay2 = np.zeros(npts)
        decay3 = np.zeros(npts)
        plotcolors = ['#4C5355','#98A4B0','#FF3F00']
        plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
        for k in range(len(fulldecay)):
            istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
            iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
            decay[istart:iend] += fulldecay[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
            decay2[istart:iend] += fulldecay2[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
            decay3[istart:iend] += fulldecay3[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
        plt.plot(en_axis * 1000, decay/decay2*10**9, color=plotcolors[i],
                 label='{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
    plt.ylabel(r'MFPs (nm)')
    plt.xlabel('Energy (meV)')
    # plt.title(r'All $k_x$')
    plt.xlim([0,np.max(en_axis)*1000])
    # plt.legend(frameon=False)
    plt.savefig(pp.figureLoc+'MFP_kde.png',dpi=600)

    freq = pp.freqVector[0]
    plt.figure(figsize=squareFigSize)
    for i, ee in enumerate(fields):
        corr_xx = np.load(pp.outputLoc + 'SB_Density/' +'xx_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, ee))
        fulldecay = np.real(corr_xx)
        # Plot of the momentum loss KDE vs energy for the chi of field designated above
        decay = np.zeros(npts)
        plotcolors = ['#4C5355','#98A4B0','#FF3F00']
        plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
        for k in range(len(fulldecay)):
            istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
            iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
            decay[istart:iend] += fulldecay[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
        plt.plot(en_axis * 1000, decay, color=plotcolors[i],
                 label='{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
    plt.ylabel(r'G_eff')
    plt.xlabel('Energy (meV)')
    # plt.title(r'All $k_x$')
    plt.xlim([0,np.max(en_axis)*1000])
    # plt.legend(frameon=False)
    plt.savefig(pp.figureLoc+'g_kde.png',dpi=600)

    # plt.figure(figsize=triFigSize)
    # for i,ee in enumerate(fields):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     fulldecay = np.dot(scm,chi)
    #     fulldecay = fulldecay[neg_k_inds]
    #     # Plot of the momentum loss KDE vs energy for the chi of field designated above
    #     decay = np.zeros(npts)
    #     for k in range(len(fulldecay)):
    #         istart = int(np.maximum(np.floor((neg_enk[k] - neg_en_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((neg_enk[k] - neg_en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         decay[istart:iend] += fulldecay[k] * gaussian(neg_en_axis[istart:iend], neg_enk[k], neg_vmags[k], stdev=spread)
    #     plt.plot(neg_en_axis*1000, decay, color=fieldcolors[i], label='{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.ylabel(r'Density flux weight ($\rm eV^{-1} \, s^{-1}$)')
    # plt.xlabel('Electron energy (meV)')
    # plt.legend(frameon=False)
    # plt.title(r'Negative $k_x$')
    # plt.savefig(pp.figureLoc+'density_flux_neg.png',dpi=600)
    #
    #
    #
    # plt.figure(figsize=triFigSize)
    # for i,ee in enumerate(fields):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     fulldecay = np.dot(scm,chi)
    #     fulldecay = fulldecay[pos_k_inds]
    #     # Plot of the momentum loss KDE vs energy for the chi of field designated above
    #     decay = np.zeros(npts)
    #     for k in range(len(fulldecay)):
    #         istart = int(np.maximum(np.floor((pos_enk[k] - pos_en_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((pos_enk[k] - pos_en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         decay[istart:iend] += fulldecay[k] * gaussian(pos_en_axis[istart:iend], pos_enk[k], pos_vmags[k], stdev=spread)
    #     plt.plot(pos_en_axis*1000, decay, color=fieldcolors[i], label='{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.ylabel(r'Density flux weight ($\rm eV^{-1} \, s^{-1}$)')
    # plt.xlabel('Electron energy (meV)')
    # plt.legend(frameon=False)
    # plt.title(r'Positive $k_x$')
    # plt.savefig(pp.figureLoc+'density_flux_pos.png',dpi=600)
    #
    #
    # plt.figure(figsize=triFigSize)
    # for i,ee in enumerate(fields):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     fulldecay = np.dot(scm,chi)
    #     # Plot of the momentum loss KDE vs energy for the chi of field designated above
    #     decay = np.zeros(npts)
    #     for k in range(len(fulldecay)):
    #         istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         decay[istart:iend] += fulldecay[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
    #     plt.plot(en_axis*1000, decay, color=fieldcolors[i], label='{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.ylabel(r'Density flux weight ($\rm eV^{-1} \, s^{-1}$)')
    # plt.xlabel('Electron energy (meV)')
    # plt.legend(frameon=False)
    # plt.title(r'All $k_x$')
    # plt.savefig(pp.figureLoc+'density_flux_all.png',dpi=600)
    #
    # plt.figure(figsize=triFigSize)
    # dx = (en_axis.max() - en_axis.min()) / npts
    # spread = 70 * dx
    # for i,ee in enumerate(fields):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     thermal_energy = utilities.mean_energy(np.zeros(nkpts), el_df)
    #     energy_above_thermal = el_df['energy [eV]'].values-thermal_energy
    #     fulldecay = np.dot(scm, chi)*energy_above_thermal
    #     # Plot of the momentum loss KDE vs energy for the chi of field designated above
    #     decay = np.zeros(npts)
    #     for k in range(len(fulldecay)):
    #         istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         decay[istart:iend] += fulldecay[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
    #     plt.plot(en_axis*1000, decay, color=fieldcolors[i], label='{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.ylabel(r'Power weight ($\rm s^{-1}$)')
    # plt.xlabel('Electron energy (meV)')
    # plt.legend(frameon=False)
    # plt.title(r'All $k_x$')
    # plt.savefig(pp.figureLoc+'energy_flux_all.png',dpi=600)
    #
    # plt.figure(figsize=triFigSize)
    # dx = (en_axis.max() - en_axis.min()) / npts
    # spread = 70 * dx
    # for i,ee in enumerate(fields):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     thermal_energy = utilities.mean_energy(np.zeros(nkpts), el_df)
    #     energy_above_thermal = el_df['energy [eV]'].values-np.min(el_df['energy [eV]'])
    #     fulldecay = np.abs(chi*energy_above_thermal/np.dot(scm, chi)*energy_above_thermal)
    #     # Plot of the momentum loss KDE vs energy for the chi of field designated above
    #     decay = np.zeros(npts)
    #     for k in range(len(fulldecay)):
    #         istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         decay[istart:iend] += fulldecay[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
    #     plt.plot(en_axis*1000, decay, color=fieldcolors[i], label='{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.ylabel(r'Energy relaxation time ($\rm s\, eV^{-1}$)')
    # plt.xlabel('Electron energy (meV)')
    # plt.legend(frameon=False)
    # plt.title(r'All $k_x$')
    # plt.savefig(pp.figureLoc+'energy_flux_all.png',dpi=600)
    #
    #
    # plt.figure(figsize=triFigSize)
    # dx = (en_axis.max() - en_axis.min()) / npts
    # spread = 70 * dx
    # for i,ee in enumerate(fields):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     thermal_energy = utilities.mean_energy(np.zeros(nkpts), el_df)
    #     energy_above_thermal = el_df['energy [eV]'].values-np.min(el_df['energy [eV]'])
    #     fulldecay = energy_above_thermal*chi
    #     # Plot of the momentum loss KDE vs energy for the chi of field designated above
    #     decay = np.zeros(npts)
    #     for k in range(len(fulldecay)):
    #         istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         decay[istart:iend] += fulldecay[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
    #     plt.plot(en_axis*1000, decay, color=fieldcolors[i], label='{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.ylabel(r'Excess energy ($\rm eV$)')
    # plt.xlabel('Electron energy (meV)')
    # plt.legend(frameon=False)
    # # plt.title(r'All $k_x$')
    # plt.savefig(pp.figureLoc+'excess_energy.png',dpi=600)
    #
    #
    # plt.figure(figsize=triFigSize)
    # dx = (neg_en_axis.max() - neg_en_axis.min()) / npts
    # spread = 70 * dx
    # for i,ee in enumerate(fields):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     thermal_energy = utilities.mean_energy(np.zeros(nkpts), el_df)
    #     energy_above_thermal = el_df['energy [eV]'].values-thermal_energy
    #     fulldecay = np.dot(scm, chi)*energy_above_thermal
    #     fulldecay = fulldecay[neg_k_inds]
    #     # Plot of the momentum loss KDE vs energy for the chi of field designated above
    #     decay = np.zeros(npts)
    #     for k in range(len(fulldecay)):
    #         istart = int(np.maximum(np.floor((neg_enk[k] - neg_en_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((neg_enk[k] - neg_en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         decay[istart:iend] += fulldecay[k] * gaussian(neg_en_axis[istart:iend], neg_enk[k], neg_vmags[k], stdev=spread)
    #     plt.plot(neg_en_axis*1000, decay, color=fieldcolors[i], label='{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.ylabel(r'Power weight ($\rm s^{-1}$)')
    # plt.xlabel('Electron energy (meV)')
    # plt.title(r'Negative $k_x$')
    # plt.legend(frameon=False)
    # plt.savefig(pp.figureLoc+'energy_flux_neg.png',dpi=600)
    #
    #
    # plt.figure(figsize=triFigSize)
    # dx = (pos_en_axis.max() - pos_en_axis.min()) / npts
    # spread = 70 * dx
    # for i,ee in enumerate(fields):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     thermal_energy = utilities.mean_energy(np.zeros(nkpts), el_df)
    #     energy_above_thermal = el_df['energy [eV]'].values-thermal_energy
    #     fulldecay = np.dot(scm, chi)*energy_above_thermal
    #     fulldecay = fulldecay[pos_k_inds]
    #
    #     # Plot of the momentum loss KDE vs energy for the chi of field designated above
    #     decay = np.zeros(npts)
    #     for k in range(len(fulldecay)):
    #         istart = int(np.maximum(np.floor((pos_enk[k] - pos_en_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((pos_enk[k] - pos_en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         decay[istart:iend] += fulldecay[k] * gaussian(pos_en_axis[istart:iend], pos_enk[k], pos_vmags[k], stdev=spread)
    #     plt.plot(pos_en_axis*1000, decay, color=fieldcolors[i], label='{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.ylabel(r'Power weight ($\rm s^{-1}$)')
    # plt.xlabel('Electron energy (meV)')
    # plt.title(r'Positive $k_x$')
    # plt.legend(frameon=False)
    # plt.savefig(pp.figureLoc+'energy_flux_pos.png',dpi=600)
    #
    # kmag = np.sqrt(el_df['kx [1/A]'].values ** 2 + el_df['ky [1/A]'].values ** 2 + el_df['kz [1/A]'].values ** 2)
    # plt.figure(figsize=triFigSize)
    # dx = (en_axis.max() - en_axis.min()) / npts
    # spread = 70 * dx
    # for i,ee in enumerate(fields):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     fulldecay = np.dot(scm, chi)*el_df['kx [1/A]'].values
    #     # fulldecay = np.dot(scm, chi)*kmag
    #     # Plot of the momentum loss KDE vs energy for the chi of field designated above
    #     decay = np.zeros(npts)
    #     for k in range(len(fulldecay)):
    #         istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         decay[istart:iend] += fulldecay[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
    #     plt.plot(en_axis*1000, decay, color=fieldcolors[i], label='{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.ylabel(r'Momentum weight ($\rm eV^{-1} \, \AA^{-1} \, s^{-1}$)')
    # plt.xlabel('Electron energy (meV)')
    # plt.legend(frameon=False)
    # plt.title(r'All $k_x$')
    # plt.savefig(pp.figureLoc+'momentum_flux_all.png',dpi=600)
    #
    #
    # plt.figure(figsize=triFigSize)
    # dx = (en_axis.max() - en_axis.min()) / npts
    # spread = 70 * dx
    # for i,ee in enumerate(fields):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     # fulldecay = el_df['kx [1/A]'].values*chi
    #     fulldecay = kmag*chi
    #     # fulldecay = np.dot(scm, chi)*kmag
    #     # Plot of the momentum loss KDE vs energy for the chi of field designated above
    #     decay = np.zeros(npts)
    #     for k in range(len(fulldecay)):
    #         istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         decay[istart:iend] += fulldecay[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
    #     plt.plot(en_axis*1000, decay, color=fieldcolors[i], label='{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.ylabel(r'Excess momentum ($\rm \AA^{-1}$)')
    # plt.xlabel('Electron energy (meV)')
    # plt.legend(frameon=False)
    # # plt.title(r'All $k_x$')
    # plt.savefig(pp.figureLoc+'excess_momentum.png',dpi=600)
    #
    #
    # plt.figure(figsize=triFigSize)
    # dx = (en_axis.max() - en_axis.min()) / npts
    # spread = 70 * dx
    # for i,ee in enumerate(fields):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     fulldecay = np.abs(el_df['kx [1/A]']*chi/np.dot(scm, chi)*el_df['kx [1/A]'])
    #     # fulldecay = np.dot(scm, chi)*kmag
    #     # Plot of the momentum loss KDE vs energy for the chi of field designated above
    #     decay = np.zeros(npts)
    #     for k in range(len(fulldecay)):
    #         istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         decay[istart:iend] += fulldecay[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
    #     plt.plot(en_axis*1000, decay, color=fieldcolors[i], label='{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.ylabel(r'Momentum relaxation time ($\rm s \, eV^{-1}$)')
    # plt.xlabel('Electron energy (meV)')
    # plt.legend(frameon=False)
    # plt.title(r'All $k_x$')
    # plt.savefig(pp.figureLoc+'momentum_flux_all.png',dpi=600)
    #
    #
    #
    # plt.figure(figsize=triFigSize)
    # dx = (neg_en_axis.max() - neg_en_axis.min()) / npts
    # spread = 70 * dx
    # for i,ee in enumerate(fields):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     fulldecay = np.dot(scm, chi)*el_df['kx [1/A]'].values
    #     # fulldecay = np.dot(scm, chi)*kmag
    #     fulldecay = fulldecay[neg_k_inds]
    #     # Plot of the momentum loss KDE vs energy for the chi of field designated above
    #     decay = np.zeros(npts)
    #     for k in range(len(fulldecay)):
    #         istart = int(np.maximum(np.floor((neg_enk[k] - neg_en_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((neg_enk[k] - neg_en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         decay[istart:iend] += fulldecay[k] * gaussian(neg_en_axis[istart:iend], neg_enk[k], neg_vmags[k], stdev=spread)
    #     plt.plot(neg_en_axis*1000, decay, color=fieldcolors[i], label='{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.ylabel(r'Momentum weight ($\rm eV^{-1} \, \AA^{-1} \, s^{-1}$)')
    # plt.xlabel('Electron energy (meV)')
    # plt.title(r'Negative $k_x$')
    # plt.legend(frameon=False)
    # plt.savefig(pp.figureLoc+'momentum_flux_neg.png',dpi=600)
    #
    # plt.figure(figsize=triFigSize)
    # dx = (pos_en_axis.max() - pos_en_axis.min()) / npts
    # spread = 70 * dx
    # for i,ee in enumerate(fields):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     fulldecay = np.dot(scm, chi)*el_df['kx [1/A]'].values
    #     # fulldecay = np.dot(scm, chi)*kmag
    #     fulldecay = fulldecay[pos_k_inds]
    #
    #     # Plot of the momentum loss KDE vs energy for the chi of field designated above
    #     decay = np.zeros(npts)
    #     for k in range(len(fulldecay)):
    #         istart = int(np.maximum(np.floor((pos_enk[k] - pos_en_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((pos_enk[k] - pos_en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         decay[istart:iend] += fulldecay[k] * gaussian(pos_en_axis[istart:iend], pos_enk[k], pos_vmags[k], stdev=spread)
    #     plt.plot(pos_en_axis*1000, decay, color=fieldcolors[i], label='{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.ylabel(r'Momentum weight ($\rm eV^{-1} \, \AA^{-1} \, s^{-1}$)')
    # plt.xlabel('Electron energy (meV)')
    # plt.title(r'Positive $k_x$')
    # plt.legend(frameon=False)
    # plt.savefig(pp.figureLoc+'momentum_flux_pos.png',dpi=600)
    #
    # plt.figure(figsize=triFigSize)
    # for i,ee in enumerate(fields):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(ee))
    #     spread = 70 * dx
    #     thermal_energy = utilities.mean_energy(np.zeros(nkpts), el_df)
    #     energy_above_thermal = el_df['energy [eV]'] - thermal_energy
    #     fulldecay = energy_above_thermal*chi
    #
    #     # Plot of the momentum loss KDE vs energy for the chi of field designated above
    #     decay = np.zeros(npts)
    #     for k in range(len(fulldecay)):
    #         istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         decay[istart:iend] += fulldecay[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
    #     plt.plot(en_axis * 1000, decay, color=fieldcolors[i], label='{:.1f} '.format(ee / 100)+r'$\rm V \, cm^{-1}$')
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.ylabel(r'Excess energy weight (unitless)')
    # plt.xlabel('Electron energy (meV)')
    # plt.legend()
    # decaymatrix = np.zeros((nkpts, nkpts))
    # allk = el_df[['kx [1/A]', 'ky [1/A]', 'kz [1/A]']].values / (2 * np.pi / c.alat)
    # # allk = np.linalg.norm(el_df[['kx [1/A]', 'ky [1/A]', 'kz [1/A]']].values / (2 * np.pi / c.alat), axis=1)
    # for ik in range(nkpts):
    #     deltak = np.linalg.norm(allk - np.tile(allk[ik, :], (nkpts, 1)), axis=1)
    #     # deltak = allk[ik] - allk  # |k| - |k'| so positive is momentum decay
    #     if enk[ik] == 0:
    #         continue
    #     delta_en = enk[ik] - enk
    #     colk = scm[:, ik].copy()
    #     rowk = scm[ik, :].copy()
    #     # Should remove the diagonal contribution, but since the deltak for element k is zero, cancels out
    #     decaymatrix[ik, :] = (deltak * rowk * -1) / np.linalg.norm(allk[ik])
    #     decaymatrix[ik, ik] = np.dot(deltak, colk) / np.linalg.norm(allk[ik])
    #     # decaymatrix[ik, :] = (delta_en * rowk * -1) / enk[ik]
    #     # decaymatrix[ik, ik] = np.dot(delta_en, colk) / enk[ik]
    #     if ik % 500 == 0:
    #         print('Did k = {}'.format(str(ik)))
    # diagdecayrate = np.diag(decaymatrix)
    #
    # for i, field in enumerate(fields):
    #     chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(field))
    #     fulldecay = decaymatrix @ fs
    #     full_enax = np.zeros(npts)
    #     for k in range(len(fulldecay)):
    #         istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
    #         iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #         full_enax[istart:iend] += fulldecay[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
    #     plt.plot(en_axis, full_enax, label='{:.0f} V/cm'.format(field/100), color=fieldcolors[i+1])
    # plt.axhline(0, linestyle='--', color='Black')
    # plt.ylabel(ytext)
    # plt.xlabel(xtext)
    # plt.legend()
    # plt.tight_layout()
    #
    # # Plot of diagonal versus off diagonal momentum loss for chi of field specified below
    # chi = np.load(pp.outputLoc + '/Steady/chi_3_E_{:.1e}.npy'.format(fields[0]))
    # # fs = chi + el_df['k_FD']
    # fs = chi
    # fulldecay = decaymatrix @ fs
    # diagdecay_fs = diagdecayrate * fs
    # diagdecay = np.zeros(npts)
    # full_enax = np.zeros(npts)
    # for k in range(len(diagdecayrate)):
    #     istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
    #     iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #     diagdecay[istart:iend] += diagdecay_fs[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
    #     full_enax[istart:iend] += fulldecay[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
    # plt.figure()
    # plt.axhline(0, linestyle='-', color='Black', linewidth=0.5)
    # plt.plot(en_axis, diagdecay, label='Diagonal only', color=fieldcolors[i+1], linestyle='--')
    # plt.plot(en_axis, full_enax, label='Including off-diagonal', color=fieldcolors[i+1])
    # plt.ylabel(ytext)
    # plt.xlabel(xtext)
    # plt.legend()
    # plt.tight_layout()
    #
    # # Plot of the momentum loss KDE vs energy for the chi of field designated above
    # g_decay = np.zeros(npts)
    # l_decay = np.zeros(npts)
    # for k in np.nonzero(g_inds)[0]:
    #     istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
    #     iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #     g_decay[istart:iend] += fulldecay[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
    # for k in np.nonzero(l_inds)[0]:
    #     istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4 * spread / dx), 0))
    #     iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    #     l_decay[istart:iend] += fulldecay[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k], stdev=spread)
    # plt.figure()
    # plt.plot(en_axis, g_decay, label='Gamma', color=valleycolors[0])
    # plt.plot(en_axis, l_decay, label='L', color=valleycolors[1])
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.ylabel(ytext)
    # plt.xlabel(xtext)
    # plt.legend()
    #
    # # Plot of momentum loss rates parametrically against energy
    # plt.figure()
    # plt.axhline(0, linestyle='--', color='Black', linewidth=0.5)
    # plt.plot(enk[g_inds], diagdecayrate[g_inds], '.', label='Gamma', markersize=3.5, color=valleycolors[0])
    # plt.plot(enk[l_inds], diagdecayrate[l_inds], '.', label='L', markersize=3.5, color=valleycolors[1])
    # plt.ylabel(ytext)
    # plt.xlabel(xtext)
    # legend_elements = [Line2D([0], [0], marker='o', lw=0, color=valleycolors[0], label='Gamma', markersize=5),
    #                    Line2D([0], [0], marker='o', lw=0, color=valleycolors[1], label='L', markersize=5)]
    # plt.legend(handles=legend_elements)
    # plt.tight_layout()


if __name__ == '__main__':
    # Create electron and phonon dataframes
    # preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)
    fields = pp.small_signal_fields
    freqs = pp.freqVector
    # freq = 6.2e2
    # print(freq)

    # plot_noise_kde(electron_df, fields)
    # material_plotter.bz_3dscatter(electron_df,True,False)
    # plot_steady_transient_difference(fields,freq)
    # plot_mom_KDEs(fields, electron_df,saveData=True)
    # fields = pp.fieldVector
    freq = pp.freqGHz

    # anisotropy_of_scattering(electron_df)
    # mom_en_relaxation(electron_df, np.array([1e2, 1e4, 5e4]))
    # mom_en_relaxation(electron_df, pp.moment_fields)
    mom_en_rel_histograms(electron_df)
    # field_dependence_noise(fields, electron_df)
    # energy_KDEs(electron_df, [1e4, 3e4, 5e4])
    # plot_noise_kde(electron_df, fields)
    # material_plotter.bz_3dscatter(electron_df,True,False)
    # plot_steady_transient_difference(fields,freq)
    plot_mom_KDEs(fields, electron_df, saveData=True)
    # plot_vel_KDEs(fields[-1],electron_df)
    # plot_energy_sep(electron_df, fields)
    # plot_energy_sep_lf(electron_df, fields)
    # plot_2d_dist(electron_df, fields[-1])

    # plot_noise_kde(electron_df, [1e-3,5e4],freqs[0])
    # energy_KDEs(electron_df, pp.small_signal_fields)
    # plot_steady_transient_difference(5e4, freq)
    # plot_noise_kde(electron_df, [1e-3,5e4],freqs[0])
    # energy_KDEs(electron_df, pp.small_signal_fields[2:])
    plt.show()