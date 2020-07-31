import numpy as np
import constants as c
import utilities
import problem_parameters as pp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import psd_solver
import occupation_plotter
import davydov_utilities
from scipy import integrate
import rt_solver

# Set the parameters for the paper figures
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# PURPOSE: THIS MODULE CONTAINS PLOTTING CODE TO RENDER THE PLOTS THAT WILL GO INTO THE PAPER.

# ORDER: THIS MODULE REQUIRES THAT STEADY SOLUTIONS, TRANSIENT SOLUTIONS, SMALL SIGNAL CONDUCTIVITY, AND THE EFFECTIVE
# DISTRIBUTIONS HAVE BEEN WRITTEN TO FILE ALREADY.

# OUTPUT: THIS MODULE PRODUCES MOMENTUM AND ENERGY KDES FOR THE STEADY DEVIATIONAL DISTRIBUTION, A PLOT OF THE OHMIC
# MOBILITY, A PLOT OF THE SMALL SIGNAL MOBILITY, THE SPECTRAL DENSITY VS FREQUENCY AT SELECTED FIELDS


def linear_mobility_paperplot(fieldVector,df):
    """Make a paper plot for the Ohmic (or linear) mobility of the RTA, low-field, and full-drift solutions."""
    vcm = np.array(fieldVector) * 1e-2
    lw = 2
    mu_1 = []
    mu_2 = []
    mu_3 = []
    meanE_1 = []
    meanE_2 = []
    meanE_3 = []
    for ee in fieldVector:
        chi_1_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '1_' + "E_{:.1e}.npy".format(ee))
        chi_2_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '2_' + "E_{:.1e}.npy".format(ee))
        chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        mu_1.append(utilities.calc_linear_mobility(chi_1_i, df, ee) * 10 ** 4)
        mu_2.append(utilities.calc_linear_mobility(chi_2_i, df, ee) * 10 ** 4)
        mu_3.append(utilities.calc_linear_mobility(chi_3_i, df, ee) * 10 ** 4)
        meanE_1.append(utilities.mean_energy(chi_1_i,df))
        meanE_2.append(utilities.mean_energy(chi_2_i,df))
        meanE_3.append(utilities.mean_energy(chi_3_i,df))

    plt.figure()
    plt.plot(vcm,mu_1,'--',linewidth=lw,label='RTA')
    plt.plot(vcm,mu_2,'--',linewidth=lw,label='Low Field')
    plt.plot(vcm,mu_3,'-',linewidth=lw,label='Full Drift')

    plt.xlabel(r'Field ($V \, cm^{-1}$)')
    plt.ylabel(r'Ohmic Mobility ($cm^2 \, V^{-1}\, s^{-1}$)')
    plt.ylim([0.8*np.min(mu_1),1.1*np.max(mu_3)])
    plt.legend(ncol=3,loc='lower center')
    plt.savefig(pp.figureLoc +'linear_mobility2.png', bbox_inches='tight',dpi=600)

    plt.figure()
    lw = 2
    mup = np.min(df['energy [eV]']) - pp.mu
    plt.plot(vcm,(np.array(meanE_1) -pp.mu - mup)/c.kb_ev*2/3,'-', linewidth=lw, label='RTA')
    plt.plot(vcm,(np.array(meanE_2) -pp.mu - mup)/c.kb_ev*2/3,'-', linewidth=lw, label='L-F')
    plt.plot(vcm,(np.array(meanE_3) -pp.mu - mup)/c.kb_ev*2/3,'-', linewidth=lw, label='FDM')
    plt.xlabel(r'Field [$kV/cm$]')
    plt.ylabel(r'Electron Temperature [$K$]')
    plt.title(pp.title_str)
    plt.legend()


def small_signal_mobility_paperplot(fieldVector,freqVector,df):
    """Make and save a paper plot for the small signal AC conductivity and save to file."""
    vcm = np.array(fieldVector)*1e-2
    n = utilities.calculate_density(df)
    lw = 2
    fig, ax = plt.subplots()
    for freq in freqVector:
        cond = []
        mu_3 = []
        for ee in fieldVector:
            chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
            mu_3.append(utilities.calc_linear_mobility(chi_3_i, df, ee) * 10 ** 4)
            cond.append(np.load(pp.outputLoc + 'Small_Signal/' + 'cond_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, ee)))
        ax.plot(vcm, np.array(cond)/c.e/n*100**2, '-', label='{:.1f} GHz'.format(freq),linewidth=lw)
    ax.plot(vcm,mu_3,'-',label = 'Ohmic Mobility',linewidth=lw)
    plt.xlabel(r'Field ($V \, cm^{-1}$)')
    plt.ylabel(r'Longitudinal AC Mobility ($cm^2 \, V^{-1} \, s^{-1}$)')
    plt.ylim([-0.4*np.max(mu_3),np.max(mu_3)*1.2])
    # ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.legend(ncol=3,loc='lower center')
    plt.savefig(pp.figureLoc+'ac_mobility.png', bbox_inches='tight',dpi=600)

    fig, ax = plt.subplots()
    i  = 0
    for ee in fieldVector:
        colorList = ['black', 'dodgerblue', 'tomato']
        cond = []
        cond_linear = []
        for freq in freqVector:
            mu_3 = []
            chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
            mu_3.append(utilities.calc_linear_mobility(chi_3_i, df, ee) * 10 ** 4)
            cond.append(np.load(pp.outputLoc + 'Small_Signal/' + 'cond_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, ee)))
            cond_linear.append(np.load(pp.outputLoc + 'Small_Signal/' + 'linear_cond_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, ee)))

        ax.plot(freqVector, np.array(cond)/c.e/n*100**2, '-', label='E = {:.0f} '.format(ee/100)+r'$V \, cm^{-1}$',linewidth=lw,color = colorList[i])
        ax.plot(freqVector, np.array(cond_linear)/c.e/n*100**2, '-.', label='E = {:.0f} L '.format(ee/100)+r'$V \, cm^{-1}$',linewidth=lw,color = colorList[i])

        i = i + 1
    plt.xlabel(r'Frequency ($GHz$)')
    plt.ylabel(r'Longitudinal AC Mobility ($cm^2 \, V^{-1}\, s^{-1}$)')
    plt.ylim([-0.4*np.max(mu_3),np.max(mu_3)*1.2])
    plt.xscale('log')
    plt.legend()
    # ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.savefig(pp.figureLoc+'ac_mobility2.png', bbox_inches='tight',dpi=600)


def momentum_kde_paperplot(fields):
    """Make a paper plot for the momentum KDE of the low-field, and full-drift solutions."""
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    axisList = [ax1,ax2,ax3]
    i =0

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for ee in fields:
        ee_Vcm = ee/100
        textstr = r'$E_{k_x}\, = \, %.1f \, V \, cm^{-1}$' % ee_Vcm
        k_ax = np.load(pp.outputLoc + 'Momentum_KDE/' + 'k_ax_' + '2_' + "E_{:.1e}.npy".format(ee))
        kdist_f0_2 = np.load(pp.outputLoc + 'Momentum_KDE/' + 'k_dist_f0_' + '2_' + "E_{:.1e}.npy".format(ee))
        kdist_2 = np.load(pp.outputLoc + 'Momentum_KDE/' + 'k_dist' + '2_' + "E_{:.1e}.npy".format(ee))

        kdist_f0_3 = np.load(pp.outputLoc + 'Momentum_KDE/' + 'k_dist_f0_' + '3_' + "E_{:.1e}.npy".format(ee))
        kdist_3 = np.load(pp.outputLoc + 'Momentum_KDE/' + 'k_dist' + '3_' + "E_{:.1e}.npy".format(ee))
        axisList[i].fill(k_ax, kdist_2/np.max(kdist_f0_2), '--', linewidth=2, alpha = 0.6, label='Low Field '+r'$\Delta f$',color='blue')
        axisList[i].fill(k_ax, kdist_3/np.max(kdist_f0_2), '--', linewidth=2, alpha = 0.6, label='Full Drift '+r'$\Delta f$',color='red')
        axisList[i].plot(k_ax, kdist_2/np.max(kdist_f0_2), '-', linewidth=1,color='blue')
        axisList[i].plot(k_ax, kdist_3/np.max(kdist_f0_2), '-', linewidth=1,color='red')
        axisList[i].plot(k_ax, kdist_f0_2/np.max(kdist_f0_2), '-', linewidth=2, label='Equilibrium Dist.',color='black')
        axisList[i].yaxis.set_major_formatter(FormatStrFormatter('%g'))
        axisList[i].locator_params(axis='y', nbins=2)
        axisList[i].locator_params(axis='x', nbins=5)
        axisList[i].set_xlim(-0.06,0.06)
        axisList[i].text(0.02, 0.92, textstr, transform=axisList[i].transAxes, verticalalignment='top', bbox=props)

        i = i+1
    plt.xlabel(r'$k_x \, \, (\AA^{-1})$')
    ax2.set_ylabel('Occupation Probability (norm.)')
    axisList[0].legend(loc ="upper right")
    plt.savefig(pp.figureLoc+'momentum_KDE.png', bbox_inches='tight',dpi=600)


def pop_below_cutoff(ee,df,cutoffVector):
    n_below = []
    fig, ax = plt.subplots()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    Vcm = np.array(ee)/100
    textstr = 'E = %.1f V/cm' %Vcm
    n = utilities.calculate_density(df)
    for cutoff in cutoffVector:
        chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        energyInds = np.array(df['energy [eV]'].values < cutoff+ np.min(df['energy [eV]']))
        n_below.append(utilities.calculate_noneq_density(chi_3_i[energyInds],df.loc[energyInds]))
    ax.plot(cutoffVector,np.array(n_below)/n)
    ax.text(0.8, 0.9, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.xlabel(r'Cutoff (eV)')
    plt.ylabel(r'Population fraction below cutoff')


def plotDavydov(df,fields,pA):
    fig, ax = plt.subplots()
    ax.plot(df['energy [eV]'] -np.min(df['energy [eV]']), df['k_FD'], '.',color='black',label='Fermi-Dirac')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for ee in fields:
        chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        f = chi_3_i+df['k_FD']
        davydovDist = davydov_utilities.davydovDistribution(df,ee,pA,pB)
        ax.plot(df['energy [eV]']-np.min(df['energy [eV]']),davydovDist,'.',color='red',label='Acoustic Davydov')
        ax.plot(df['energy [eV]'] - np.min(df['energy [eV]']), f, '.',color='blue',label='Full Drift')
    eeVcm = ee/100
    textstr = 'Electric Field = %.1f V/cm \n' %eeVcm + 'ADP Coefficient = %.1e ' % pA +r'$J^{1/2} \, s$'
    ax.text(0.60, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.legend(loc ='lower left')
    plt.yscale('log')
    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel('Occupation (unitless)')

    fig, ax = plt.subplots()
    davydovPartial = davydov_utilities.davydovPartialEnergy(df,ee,pA,pB)
    ax.plot(df['energy [eV]'] - np.min(df['energy [eV]']), -davydovPartial*c.e, '.', color='red', label='Acoustic Davydov')
    plt.xlabel('Energy above CBM (eV)')
    plt.yscale('log')
    plt.ylabel('Partial Energy (1/eV)')


def plotkykxplane(df,ee,pA):
    chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
    f = chi_3_i + df['k_FD']
    # davydovDist = davydov_utilities.davydovDistribution(df, ee,pA,pB)

    df['fullDrift'] = f
    # df['acousticDavydov'] = davydovDist
    g_inds,_,_ = utilities.gaas_split_valleys(df,False)
    g_df = df.loc[g_inds]
    uniq_yz = np.unique(g_df[['ky [1/A]', 'kz [1/A]']].values, axis=0)
    uniq_z = uniq_yz[:,1]
    uniq_y_small_z_inds = np.where(np.abs(uniq_yz[:,1])==0)
    uniq_y_small_z = uniq_yz[uniq_y_small_z_inds[0]]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    max_kx = np.max(g_df['kx [1/A]'])
    max_ky = np.max(g_df['ky [1/A]'])

    for i in range(len(uniq_y_small_z)):
        kind = i + 1
        ky, kz = uniq_y_small_z[i, 0], uniq_y_small_z[i, 1]
        # Grab the "slice" of points in k space with the same ky and kz coordinate
        slice_df = g_df.loc[(g_df['ky [1/A]'] == ky) & (g_df['kz [1/A]'] == kz)].sort_values(by=['kx [1/A]'],ascending=True)
        # ax.scatter(slice_df['kx [1/A]'],slice_df['ky [1/A]'],np.log10(slice_df['fullDrift']),c='blue')
        # ax.scatter(slice_df['kx [1/A]'],slice_df['ky [1/A]'],np.log10(slice_df['acousticDavydov']),c='red')
        # ax.scatter(slice_df['kx [1/A]'],slice_df['ky [1/A]'])

    ax.set_xlabel('kx (norm.)')
    ax.set_zlim3d([-13,-3])
    ax.set_ylabel('ky (norm.)')
    ax.set_zlabel('log(f) (unitless)')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='z', nbins=5)
    # plt.title('Equilibrium Distribution')
    plt.title('Acoustic Davydov {:.1f} V/cm'.format(ee/100))
    # plt.title('Full Drift {:.1f} V/cm'.format(ee/100))


def momentum_kde2_paperplot(fields):
    """Make a paper plot for the momentum KDE of the low-field, and full-drift solutions."""
    fig, ax = plt.subplots()
    colorList = ['dodgerblue','tomato']
    lw = 2
    i = 0
    meankx_2 = []
    meankx_3 = []
    k_ax = np.load(pp.outputLoc + 'Momentum_KDE/' + 'k_ax_' + '2_' + "E_{:.1e}.npy".format(fields[0]))
    ax.plot(k_ax, np.zeros(len(k_ax)), '-', linewidth=lw, color='black', label='Equilibrium')
    for ee in fields:
        ee_Vcm = ee/100
        k_ax = np.load(pp.outputLoc + 'Momentum_KDE/' + 'k_ax_' + '2_' + "E_{:.1e}.npy".format(ee))
        kdist_f0_2 = np.load(pp.outputLoc + 'Momentum_KDE/' + 'k_dist_f0_' + '2_' + "E_{:.1e}.npy".format(ee))
        kdist_2 = np.load(pp.outputLoc + 'Momentum_KDE/' + 'k_dist' + '2_' + "E_{:.1e}.npy".format(ee))
        kdist_f0_3 = np.load(pp.outputLoc + 'Momentum_KDE/' + 'k_dist_f0_' + '3_' + "E_{:.1e}.npy".format(ee))
        kdist_3 = np.load(pp.outputLoc + 'Momentum_KDE/' + 'k_dist' + '3_' + "E_{:.1e}.npy".format(ee))

        chi_2_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '2_' + "E_{:.1e}.npy".format(ee))
        meankx_2.append(utilities.mean_kx(chi_2_i, electron_df))
        chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        meankx_3.append(utilities.mean_kx(chi_3_i, electron_df))

        ax.plot(k_ax, kdist_2, '--', linewidth=lw,color = colorList[i],label=r'Low Field {:.0f} '.format(ee/100)+r'$V \, cm^{-1}$')
        ax.plot(k_ax, kdist_3, '-', linewidth=lw,color = colorList[i],label=r'Full Drift {:.0f} '.format(ee/100)+r'$V \, cm^{-1}$')
        i = i + 1
    # ax.plot(k_ax, kdist_f0_3, '--', linewidth=lw, color='black', label=r'$f_0$')
    # ax.plot(meankx_2,np.mean(abs(kdist_2))*np.ones(len(meankx_3)), '-', linewidth=lw, color='black')
    # ax.plot(meankx_3,np.mean(abs(kdist_3))*np.ones(len(meankx_3)), '-', linewidth=lw, color='black')

    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.locator_params(axis='y', nbins=3)
    ax.locator_params(axis='x', nbins=5)
    ax.set_xlim(-0.06,0.06)

    plt.xlabel(r'$k_x \, \, (\AA^{-1})$')
    plt.ylabel(r'Deviational occupation $\delta f_{\mathbf{k}}$ (norm.)')
    # plt.ylabel(r'$\delta f_{\mathbf{k}}/f_{\mathbf{k}}^0$')
    # plt.ylim([-1,1])
    plt.legend(loc ="upper left")
    plt.savefig(pp.figureLoc+'momentum_KDE2.png', bbox_inches='tight',dpi=600)


def energy_kde_paperplot(fields,df):
    """Make a energy plot for the momentum KDE of the low-field, and full-drift solutions."""
    plt.figure()
    i = 0
    colorList = ['dodgerblue','tomato']
    lw = 2

    meanE_2 = []
    meanE_3 = []
    mup = np.min(df['energy [eV]']) - pp.mu
    chi_0 = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '2_' + "E_{:.1e}.npy".format(fields[0]))
    g_en_axis, _, _, _, _, _, _, _, _, _, _, _ = \
        occupation_plotter.occupation_v_energy_sep(chi_0, df['energy [eV]'].values, df)
    plt.plot(g_en_axis - np.min(df['energy [eV]']), np.zeros(len(g_en_axis)), '-', color='black', lineWidth=lw,label='Equilibrium')

    for ee in fields:
        chi_2_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '2_' + "E_{:.1e}.npy".format(ee))
        # meanE_2 = utilities.mean_energy(chi_2_i,df)
        g_en_axis, g_ftot, g_chiax, g_f0ax, _, _, _, _, _, _, _, _ = \
            occupation_plotter.occupation_v_energy_sep(chi_2_i, df['energy [eV]'].values, df)
        plt.plot(g_en_axis - np.min(df['energy [eV]']), g_chiax,'--',color = colorList[i],lineWidth=lw,label=r'Low Field {:.0f} '.format(ee/100)+r'$V \, cm^{-1}$')
        print(integrate.trapz(g_chiax,g_en_axis))

        # plt.plot(meanE_2-np.min(df['energy [eV]']),0,'.')
        chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        g_en_axis, g_ftot, g_chiax, g_f0ax, _, _, _, _, _, _, _, _ = \
            occupation_plotter.occupation_v_energy_sep(chi_3_i, df['energy [eV]'].values, df)
        plt.plot(g_en_axis - np.min(df['energy [eV]']), g_chiax,color = colorList[i],lineWidth=lw,label=r'Full Drift {:.0f} '.format(ee/100)+r'$V \, cm^{-1}$')
        print(integrate.trapz(g_chiax,g_en_axis))

        i = i + 1
    # plt.plot(g_en_axis - np.min(df['energy [eV]']), g_f0ax, '--', color='black', lineWidth=lw,label=r'$f_0$')

    plt.legend()
    # plt.ylim([-0.02, 0.015])
    plt.xlabel(r'Energy above CBM ($eV$)')
    plt.ylabel(r'Deviational occupation $\delta f_{\mathbf{k}}$ (norm.)')
    # plt.ylabel(r'$\delta f_{\mathbf{k}}/f_{\mathbf{k}}^0$')
    plt.savefig(pp.figureLoc+'energy_KDE.png', bbox_inches='tight',dpi=600)

    plt.figure()
    plt.plot(g_en_axis,g_chiax)

    plt.figure()
    Z, xedges, yedges = np.histogram2d(df['kx [1/A]']*chi_3_i,df['ky [1/A]']*chi_3_i)
    plt.pcolormesh(xedges, yedges, Z.T)

    from scipy.stats.kde import gaussian_kde
    g_inds,_,_ = utilities.gaas_split_valleys(df,False)
    g_df = df.loc[g_inds]

    x = g_df['kx [1/A]']*(chi_3_i[g_inds]+g_df['k_FD'])
    y = g_df['ky [1/A]']*(chi_3_i[g_inds]+g_df['k_FD'])

    # y = g_df['energy [eV]']*(chi_3_i[g_inds]+g_df['k_FD'])
    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min():x.max():x.size ** 0.5 * 1j, y.min():y.max():y.size ** 0.5 * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    fig = plt.figure(figsize=(7, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    # alpha=0.5 will make the plots semitransparent
    ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5)
    ax2.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)

    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylim(y.min(), y.max())
    ax2.set_xlim(x.min(), x.max())
    ax2.set_ylim(y.min(), y.max())


def plot_energy_transient(freqVector,df):
    meanEnergy = []
    fig, ax = plt.subplots()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for freq in freqVector:
        chi_3t_i = np.load(pp.outputLoc + 'Transient/' + 'chi_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, 1e4))
        meanEnergy.append(utilities.mean_energy(chi_3t_i,df))
    ax.plot(freqVector, np.array(meanEnergy)-np.min(df['energy [eV]']),label = r'100 ($V \, cm^{-1}$)')
    meanEnergy = []

    for freq in freqVector:
        chi_3t_i = np.load(pp.outputLoc + 'Transient/' + 'chi_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, 4e4))
        meanEnergy.append(utilities.mean_energy(chi_3t_i,df))
    ax.plot(freqVector, np.array(meanEnergy)-np.min(df['energy [eV]']),label = r'400 ($V \, cm^{-1}$)')
    ax.axhline(utilities.mean_energy(np.zeros(len(df)),df)-np.min(df['energy [eV]']), linestyle='--',
               label='Thermal Energy',color='black')
    plt.legend()
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Mean energy above CBM (eV)')
    plt.xscale('log')
    plt.savefig(pp.figureLoc + 'Freq_Dependent_Energy.png', bbox_inches='tight', dpi=600)


def plot_density_v_field(fieldVector, freq, df, pA, pB):
    fig, ax = plt.subplots()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # textstr = '\n'.join((r'$f = %.1f GHz \, \, (100) $' % (freq,), pp.fdmName))
    ratio = pA/pB
    textstr = 'ADP Coefficient = %.1e ' % pA + r'$J^{1/2} \, s$'+ '\n' + r'$\tau_p/\tau_{\epsilon} = %.2f$' % ratio
    S_xx_vector = []
    S_xx_RTA_vector = []
    S_yy_vector = []
    conductivity_xx_vector = []
    S_xx_Davy = []
    davyEnergy = []
    fermiNoise = []
    for ee in fieldVector:
        chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        S_xx, S_xx_RTA, S_yy, S_yy_RTA, conductivity_xx = psd_solver.density(chi, ee, df, freq, False, 0)
        S_xx_vector.append(S_xx)
        S_xx_RTA_vector.append(S_xx_RTA)
        S_yy_vector.append(S_yy)
        # S_xx_Davy.append(davydov_utilities.davydovNoise(electron_df, ))
        conductivity_xx_vector.append(conductivity_xx)
        kvcm = np.array(fieldVector) * 1e-5
        davyEnergy.append(davydov_utilities.davydovMeanEnergy(df,ee,pA,pB)/c.e)
        # fermiNoise.append(davydov_utilities.fermiDiracNoise(df,pA,pB))
    Nuc = pp.kgrid ** 3
    ax.plot(kvcm, S_xx_vector, label=r'$S^{xx}$')
    ax.plot(kvcm, S_yy_vector, label=r'$S^{yy}$')
    # ax.plot(kvcm, np.array(S_xx_Davy),color='red', label=r'$S_{l,DD}$')
    # ax.plot(kvcm, np.array(fermiNoise),color='blue', label=r'$S_{l,FD}$')
    ax.axhline(np.array(conductivity_xx_vector[0]) * 4 * c.kb_joule * pp.T / c.Vuc / Nuc, linestyle='--',
               label=r'$\frac{4 kT \sigma_{xx}^{eq}}{V_0 N_0}$')
    plt.legend()
    plt.xlabel('Field [kV/cm]')
    plt.ylabel('Spectral Density [A^2/m^4/Hz]')
    print('Sxx at 400 V/cm is {:3e}'.format(S_xx_vector[-1]))
    print('Sxx at 0 V/cm is {:3e}'.format(S_xx_vector[0]))
    n = utilities.calculate_density(df)
    print('The mobility corresponding to 0 V/cm is {:3e} cm^2/V/s'.format(conductivity_xx_vector[0]*100**2/c.e/n))
    # ax.text(0.6, 0.9, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)

    # fig, ax = plt.subplots()
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # textstr = 'ADP Coefficient = %.1e ' % pA + r'$J^{1/2} \, s$'+ '\n' + r'$\tau_p/\tau_{\epsilon} = %.2f$' % ratio
    # ax.text(0.6, 0.9, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    # ax.plot(kvcm,davyEnergy)
    # plt.xlabel('Field [kV/cm]')
    # plt.ylabel('Mean Davydov [eV]')

    print('Factor is {:3e}'.format(Nuc*c.Vuc))


def plot_density(fieldVector, freqVector, df, cutoff):
    for freq in freqVector:
        conductivity_xx_vector = []
        for ee in fieldVector:
            chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
            S_xx, S_xx_RTA, S_yy, S_yy_RTA, conductivity_xx = psd_solver.density(chi, ee, df, freq,False,cutoff)
            conductivity_xx_vector.append(conductivity_xx)
            Nuc = pp.kgrid ** 3
    S_xx_vector = []
    S_xx_RTA_vector = []
    S_yy_vector = []
    cond = []
    for freq in freqVector:
        plotfield = fieldVector[0]
        chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(plotfield))
        S_xx, S_xx_RTA, S_yy, S_yy_RTA, conductivity_xx = psd_solver.density(chi, plotfield, df, freq,False,cutoff)
        S_xx_vector.append(S_xx)
        S_xx_RTA_vector.append(S_xx_RTA)
        S_yy_vector.append(S_yy)
        Nuc = pp.kgrid ** 3
        cond.append(np.load(pp.outputLoc + 'Small_Signal/' + 'cond_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, plotfield)))
    fig, ax = plt.subplots()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # textstr = '\n'.join((pp.fdmName, r'$E = %.1f kV/cm  \, \, (100)$' % (plotfield / 1e5,)))
    ax.axhline(np.array(conductivity_xx_vector[0]) * 4 * c.kb_joule * pp.T / c.Vuc / Nuc, linestyle='--',
               label=r'$\frac{4 kT \sigma_{xx}^{eq}}{V_0 N_0}$',color='black')
    # ax.plot(freqVector, np.array(cond) * 4 * c.kb_joule * pp.T / c.Vuc / Nuc, linestyle='--',
    #            label=r'$\frac{4 k_B T_0}{\mathcal{V}_0} \Re [\sigma^{\alpha \beta}_{\omega}(\mathcal{E} = 0)]$', color='black')

    # ax.plot(freqVector, np.array(cond) * 4 * c.kb_joule * pp.T / c.Vuc / Nuc, linestyle='--',
    #         label=r'$\frac{4 k_B T_0}{\mathcal{V}_0} \Re [\sigma^{\alpha \beta}_{\omega}(\mathcal{E} = 0)]$',
    #         color='black')

    ax.plot(freqVector, np.array(cond) * 4 * c.kb_joule * pp.T / c.Vuc / Nuc, linestyle='--',
            label=r'Nyquist-Johnson',
            color='black')
    ax.plot(freqVector, S_xx_vector,color='black', label=r'$S_{lt}$' + '  E = {:.1f} '.format(plotfield/1e2)+r'$V\,cm^{-1}$')

    fitQuant, RT, A, = rt_solver.fit_single_lorentzian_rt(freqVector,S_xx_vector)
    print(RT)
    # ax.plot(freqVector,fitQuant,'-.',label='SL: '+r'$A = {:.0f} \, A^2/m^4, $ '.format(A) + r'$\tau = {:.0f} \, fs$'.format(RT), color = 'blue')
    # ax.plot(freqVector, S_yy_vector, label=r'$S_{t}$' + '  E = {:.1f} '.format(plotfield/1e2)+r'$V\,cm^{-1}$')

    S_xx_vector = []
    S_xx_RTA_vector = []
    S_yy_vector = []
    for freq in freqVector:
        plotfield = fieldVector[-1]
        chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(plotfield))
        S_xx, S_xx_RTA, S_yy, S_yy_RTA, conductivity_xx = psd_solver.density(chi, plotfield, df, freq,False,cutoff)
        S_xx_vector.append(S_xx)
        S_xx_RTA_vector.append(S_xx_RTA)
        S_yy_vector.append(S_yy)
        Nuc = pp.kgrid ** 3
        print('freq')
    ax.plot(freqVector, S_yy_vector,color='tomato',linestyle='-.', label=r'$S_{t}$' + '  E = {:.1f} '.format(plotfield/1e2)+r'$V\,cm^{-1}$')
    fitQuant, RT, A, = rt_solver.fit_single_lorentzian_rt(freqVector,S_yy_vector)
    print(RT)
    # ax.plot(freqVector,fitQuant,'-.',label='SL: '+r'$A = {:.0f} \, A^2/m^4, $ '.format(A) + r'$\tau = {:.0f} \, fs$'.format(RT), color = 'green')
    ax.plot(freqVector, S_xx_vector,color='tomato', label=r'$S_{l}$' + '  E = {:.1f} '.format(plotfield/1e2)+r'$V\,cm^{-1}$')
    array = np.column_stack((freqVector,np.array(S_xx_vector)))
    np.save(pp.outputLoc+'Sxx_freq',array)

    fitQuant, RT, A, = rt_solver.fit_single_lorentzian_rt(freqVector,S_xx_vector)
    # ax.plot(freqVector,fitQuant,'-.',label='SL: '+r'$A = {:.0f} \, A^2/m^4, $ '.format(A) + r'$\tau = {:.0f} \, fs$'.format(RT), color = 'pink')


    # fitQuant, RT1, RT2, A1, A2 = rt_solver.fit_double_lorentzian_rt(freqVector,S_xx_vector,A,RT/1e6)
    # print(RT1)
    # print(RT2)
    # ax.plot(freqVector,fitQuant,'-.',label='DL: ' + r'$A = {:.0f} \, A^2/m^4, $ '.format(A2) + r'$\tau_1 = {:.0f} \, fs, \,$'.format(RT2) + r'$B = {:.0f} \, A^2/m^4, $ '.format(A1) + r'$\tau_2 = {:.0f} \, fs$'.format(RT1), color = 'red')


    plt.legend(loc='lower left')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Spectral Density ' r'$[A^2 \, m^4 \, Hz^{-1}]$')
    # plt.title(pp.title_str)
    # ax.text(0.05, 0.15, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.xscale('log')
    # plt.title('Energy cutoff at {:.2f} eV'.format(cutoff))
    plt.savefig(pp.figureLoc + 'Freq_Dependent_PSD.png', bbox_inches='tight', dpi=600)

    diffVector = np.array(S_yy_vector)-np.array(S_xx_vector)
    diffVector = diffVector + np.abs(np.min(diffVector))

    inds = np.where(freqVector > 100)
    fitQuant, RT, A, = rt_solver.fit_single_lorentzian_rt(freqVector[inds],np.array(S_xx_vector)[inds])
    print('Hah')
    print(RT)

    inds = np.where(freqVector < 100)
    fitQuant, RT, A, = rt_solver.fit_single_lorentzian_rt(freqVector[inds],diffVector[inds])
    print('Heh')
    print(RT)

    plt.figure()
    plt.plot(freqVector, diffVector,color='tomato',linestyle='-.')
    plt.plot(freqVector[inds], diffVector[inds],color='tomato',linestyle='-', label=r'$S_{l,max}-S_{l}$' + '  E = {:.1f} '.format(plotfield/1e2)+r'$V\,cm^{-1}$')
    plt.plot(freqVector[inds],fitQuant,label='SL: '+r'$A = {:.0f} \, A^2/m^4, $ '.format(A) + r'$\tau = {:.0f} \, fs$'.format(RT))
    plt.xscale('log')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Spectral Density [A^2/m^4/Hz]')
    plt.legend()
    plt.title('Roll off for Energy RT')


def plotScatteringRate():
    dat_300 = np.loadtxt(pp.inputLoc + 'relaxation_time_300K.dat', skiprows=1)
    enk_300 = dat_300[:, 3]  # eV
    taus_300 = dat_300[:, 4]  # fs
    rates_300 = dat_300[:, 5]  # THz
    plt.figure()
    plt.plot(enk_300-np.min(enk_300), rates_300, '.', markersize=3)
    plt.ylabel('Scattering rate (THz)')
    plt.xlabel('Energy (eV)')
    plt.xlim([-0.01,0.32])
    plt.ylim([-1,20])
    plt.savefig(pp.figureLoc + 'ScatteringRate.png', bbox_inches='tight', dpi=600)



def plotDavydovRTs(df,pA,pB):
    nkpts = len(df)
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    rates = (-1) * np.diag(scm) * 1E-15
    g_inds,_,_ = utilities.gaas_split_valleys(df,False)
    g_df = df.loc[g_inds]
    momRTs,energyRTs = davydov_utilities.davydovRTs(g_df, pA, pB)

    fig, ax = plt.subplots()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = 'ADP Coefficient = %.1e ' % pA +r'$J^{1/2} \, s$'

    ax.plot(g_df['energy [eV]'] - np.min(g_df['energy [eV]']),momRTs*1e15,label='Davydov Momentum')
    # ax.plot(g_df['energy [eV]'] - np.min(g_df['energy [eV]']),energyRTs*1e15,label='Davydov Energy')
    ax.plot(g_df['energy [eV]'] - np.min(g_df['energy [eV]']),1/rates[g_inds], '.' ,label='Perturbo')
    ax.text(0.6, 0.82, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)

    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel('Relaxation time (fs)')
    plt.legend()


if __name__ == '__main__':
    fields = pp.fieldVector
    freqs = pp.freqVector

    # Define a custom range of fields for the paper plots
    mom_kde_fields = np.array([1e2,1e4,4e4])
    energy_kde_fields = np.array([1e4,4e4])
    moment_fields = np.geomspace(1e2,4e4,20)
    small_signal_fields = np.array([1e-3,1e4,4e4])
    cutoffVector = np.geomspace(0.01,0.5,30)

    # Calculate the steady and transient solutions for moment_fields + small_signal_fields and the geomspaced frequencies
    # Calculate the small signal conductivity for the small_signal_fields and the geomspaced frequencies
    # Calculate the PSD for the moment_fields + small_signal_fields and the geomspaced frequencies
    # Then you can make the plots here.

    # preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)
    pop_below_cutoff(4e4, electron_df, cutoffVector)

    # momentum_kde_paperplot(mom_kde_fields)
    momentum_kde2_paperplot(energy_kde_fields)
    linear_mobility_paperplot(small_signal_fields, electron_df)
    energy_kde_paperplot(energy_kde_fields, electron_df)
    small_signal_mobility_paperplot(small_signal_fields,freqs,electron_df)
    # pA = 4.60082e-22 # Prefactor for energy time dependence [see Lundstrom]
    pA = 4.60082e-23*0.83/0.99531*1.11929/1.02537796976  # Modified to give the same mobility
    pB = 1000*pA  # Energy RT coefficient
    plot_density(small_signal_fields, freqs, electron_df,0.5)
    plotkykxplane(electron_df,4e4,pA)
    plot_density_v_field(small_signal_fields, freqs[0], electron_df, pA,pB)
    plot_energy_transient(freqs, electron_df)
    plotScatteringRate()
    plt.show()