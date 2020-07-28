import numpy as np
import constants as c
import utilities
import problem_parameters as pp
import matplotlib.pyplot as plt
import occupation_plotter
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
import material_plotter
font = {'size': 11}
mpl.rc('font', **font)

# PURPOSE: THIS MODULE WILL GENERATE PLOTS OF THE RELEVANT OBSERVABLES BY TAKING THE APPROPRIATE MOMENTS OVER THE STEADY
# BOLTZMANN SOLUTIONS CALCULATED IN OCCUPATION_SOLVER.PY. DRIFT VELOCITY, MOBILITY, AVERAGE CARRIER ENERGY, E TEMP ARE
# CALCULATED AND PLOTTED.

# ORDER: THIS MODULE CAN BE RUN AFTER OCCUPATION_SOLVER.PY HAS STORED SOLUTIONS IN THE OUTPUT LOC.

# OUTPUT: THIS MODULE RENDERS FIGURES FOR EXPLORATORY DATA ANALYSIS. IT DOES NOT SAVE FIGURES.

# TODO: THE PLOTS ARE NOT NICELY OR UNIFORMLY FORMATTED. IT WOULD BE GOOD TO GO THROUGH AND APPLY SOME CONSISTENT
# FORMATTING IN THE FUTURE.

def plot_steady_transport_moments(df,fieldVector):
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
    vd_2, meanE_2, n_2, mu_2, ng_2, nl_2 = ([] for i in range(6))
    vd_3, meanE_3, n_3, mu_3, ng_3, nl_3, nx_3, vd_3g, vd_3l, vd_3x, vm_3g, vm_3l, vm_3x, meanE_3g, meanE_3l, meanE_3x \
        , ng_left, ng_right = ([] for i in range(18))
    # vd_3t, meanE_3t, n_3t, mu_3t, ng_3t, nl_3t = ([] for i in range(6))
    diff_new = []
    super_cutoff_pop = []
    cutoff = np.min(df['energy [eV]'])+pp.cutoff
    superinds = df['energy [eV]']>cutoff
    f_1 = np.load(pp.outputLoc + 'Steady/' + 'f_1.npy')
    f_2 = np.load(pp.outputLoc + 'Steady/' + 'f_2.npy')
    g_inds, l_inds, x_inds = utilities.gaas_split_valleys(df, False)
    gamma_icinds_l = np.load(pp.outputLoc + 'Gamma_left_icinds.npy')
    gamma_icinds_r = np.load(pp.outputLoc + 'Gamma_right_icinds.npy')
    # l_icinds_l = np.load(pp.outputLoc + 'L_left_icinds.npy')
    # l_icinds_r = np.load(pp.outputLoc + 'L_right_icinds.npy')
    meankx_1 = []
    meankx_2 = []
    meankx_3 = []
    mom_RT = []
    nkpts = len(df)
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    if pp.simpleBool:
        rates = (-1) * np.diag(scm) * pp.scmVal * 1E-12

    thermal_tau = np.sum(df['k_FD'] * rates ** (-1)) / np.sum(df['k_FD'])
    print('Thermal relaxation time is {:3f} ps'.format(thermal_tau))

    for ee in fieldVector:
        chi_1_i = utilities.f2chi(f_1, df, ee)
        chi_2_i = utilities.f2chi(f_2, df, ee)
        chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        # chi_3t_i = np.load(pp.outputLoc + 'Transient/' + 'chi_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq,ee))

        mom_RT.append(np.sum((df['k_FD']+chi_3_i) * rates ** (-1))*1000 / np.sum(df['k_FD']))

        vd_1.append(utilities.mean_velocity(chi_1_i,df))
        vd_2.append(utilities.mean_velocity(chi_2_i,df))
        vd_3.append(utilities.mean_velocity(chi_3_i,df))
        # vd_3t.append(utilities.mean_velocity(chi_3t_i,df))

        meanE_1.append(utilities.mean_energy(chi_1_i,df))
        meanE_2.append(utilities.mean_energy(chi_2_i,df))
        meanE_3.append(utilities.mean_energy(chi_3_i,df))

        meankx_1.append(utilities.mean_kx(chi_1_i,df))
        meankx_2.append(utilities.mean_kx(chi_2_i,df))
        meankx_3.append(utilities.mean_kx(chi_3_i,df))
        # meanE_3t.append(utilities.mean_energy(chi_3t_i,df))

        meanE_3g.append(utilities.mean_energy(chi_3_i[g_inds],df.loc[g_inds]))
        meanE_3l.append(utilities.mean_energy(chi_3_i[l_inds],df.loc[l_inds]))

        mu_1.append(utilities.calc_linear_mobility(chi_1_i,df,ee) * 10 ** 4)
        mu_2.append(utilities.calc_linear_mobility(chi_2_i,df,ee) * 10 ** 4)
        mu_3.append(utilities.calc_linear_mobility(chi_3_i,df,ee) * 10 ** 4)
        # mu_3t.append(utilities.calc_linear_mobility(chi_3t_i,df,ee) * 10 ** 4)

        n_1.append(utilities.calculate_noneq_density(chi_1_i,df))
        n_2.append(utilities.calculate_noneq_density(chi_2_i,df))
        n_3.append(utilities.calculate_noneq_density(chi_3_i,df))
        # n_3t.append(utilities.calculate_noneq_density(chi_3t_i,df))

        ng_left.append(utilities.calc_popinds(chi_3_i,df,gamma_icinds_l))
        ng_right.append(utilities.calc_popinds(chi_3_i,df,gamma_icinds_r))

        ng,nl,nx,n = utilities.calc_popsplit(chi_1_i,df)
        ng_1.append(ng)
        nl_1.append(nl)

        ng,nl,nx,n = utilities.calc_popsplit(chi_2_i,df)
        ng_2.append(ng)
        nl_2.append(nl)

        ng,nl,nx,n = utilities.calc_popsplit(chi_3_i,df)
        ng_3.append(ng)
        nl_3.append(nl)

        super_cutoff_pop.append(utilities.calc_popinds(chi_3_i,df,superinds))

        if pp.getX:
            meanE_3x.append((utilities.mean_energy(chi_3_i[x_inds],df.loc[x_inds])))
            nx_3.append(nx)
            vd_3x.append(utilities.mean_velocity(chi_3_i[x_inds],df.loc[x_inds]))
            vm_3x.append(utilities.mean_xvelocity_mag(chi_3_i[x_inds],df.loc[x_inds]))

        ng,nl,nx,n = utilities.calc_popsplit(chi_3_i,df)
        # ng_3t.append(ng)
        # nl_3t.append(nl)

        vd_3g.append(utilities.mean_velocity(chi_3_i[g_inds],df.loc[g_inds]))
        vd_3l.append(utilities.mean_velocity(chi_3_i[l_inds],df.loc[l_inds]))
        vm_3g.append(utilities.mean_xvelocity_mag(chi_3_i[g_inds],df.loc[g_inds]))
        vm_3l.append(utilities.mean_xvelocity_mag(chi_3_i[l_inds],df.loc[l_inds]))
        N = np.sum(chi_3_i+df['k_FD'])
        Ng = np.sum(chi_3_i[g_inds]+df.loc[g_inds,'k_FD'])
        Nl = np.sum(chi_3_i[l_inds]+df.loc[l_inds,'k_FD'])

        diff_new.append((vd_3g[-1]-vd_3l[-1])**2/(2*N**2)*(Ng*Nl))

    kvcm = np.array(fieldVector)*1e-5

    plt.figure()
    plt.plot(kvcm,diff_new,'o-', linewidth=2, label='RTA')
    plt.xlabel(r'Field [kV/cm]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,vd_1,'o-', linewidth=2, label='RTA')
    plt.plot(kvcm,vd_2,'o-', linewidth=2, label='L-F')
    plt.plot(kvcm,vd_3,'o-', linewidth=2, label='FDM')
    # plt.plot(kvcm,np.array(mu_3)/1e4*np.array(fieldVector), label=r'$\mu E$')
    # plt.plot(kvcm,vd_3t,'o-', linewidth=2, label='FDM {:.1e} GHz'.format(freq))
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Drift velocity [m/s]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,n_1,'o-', linewidth=2, label='RTA')
    plt.plot(kvcm,n_2,'o-', linewidth=2, label='L-F')
    plt.plot(kvcm,n_3,'o-', linewidth=2, label='FDM')
    # plt.plot(kvcm,np.array(mu_3)/1e4*np.array(fieldVector), label=r'$\mu E$')
    # plt.plot(kvcm,n_3t,'o-', linewidth=2, label='FDM {:.1e} GHz'.format(freq))

    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Carrier Population [m$^-3$]')
    plt.ylim(bottom=0,top=1.2*np.max(n_3))
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm*1000,mu_1,'-', linewidth=2, label='RTA')
    plt.plot(kvcm*1000,mu_2,'-', linewidth=2, label='Low Field')
    plt.plot(kvcm*1000,mu_3,'-', linewidth=2, label='Full Drift')
    # plt.plot(kvcm,mu_3t,'o-', linewidth=2, label='FDM {:.1e} GHz'.format(freq))
    plt.xlabel(r'Electric field [$V \, cm^{-1}$]')
    plt.ylabel(r'Linear mobility [$cm^2 V^{-1} s^{-1}$]')
    plt.legend()
    # plt.savefig(pp.outputLoc+'Paper_Figures/'+'Ohmic_Mobility.png', bbox_inches='tight',dpi=600)


    plt.figure()
    plt.plot(kvcm,np.array(meanE_1) -pp.mu,'o-', linewidth=2, label='RTA')
    plt.plot(kvcm,np.array(meanE_2) -pp.mu,'o-', linewidth=2, label='L-F')
    plt.plot(kvcm,np.array(meanE_3) -pp.mu,'o-', linewidth=2, label='FDM')
    # plt.plot(kvcm,meanE_3t,'o-', linewidth=2, label='FDM {:.1e} GHz'.format(freq))
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Mean energy above Fermi Level [eV]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,meankx_1,'o-', linewidth=2, label='RTA')
    plt.plot(kvcm,meankx_2,'o-', linewidth=2, label='L-F')
    plt.plot(kvcm,meankx_3,'o-', linewidth=2, label='FDM')
    # plt.plot(kvcm,meanE_3t,'o-', linewidth=2, label='FDM {:.1e} GHz'.format(freq))
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Mean kx [1/A]')
    plt.title(pp.title_str)
    plt.legend()


    mup = np.min(df['energy [eV]']) - pp.mu


    plt.figure()
    plt.plot(kvcm,(np.array(meanE_1) -pp.mu - mup)/c.kb_ev*2/3,'o-', linewidth=2, label='RTA')
    plt.plot(kvcm,(np.array(meanE_2) -pp.mu - mup)/c.kb_ev*2/3,'o-', linewidth=2, label='L-F')
    plt.plot(kvcm,(np.array(meanE_3) -pp.mu - mup)/c.kb_ev*2/3,'o-', linewidth=2, label='FDM')
    # plt.plot(kvcm,meanE_3t,'o-', linewidth=2, label='FDM {:.1e} GHz'.format(freq))
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Electron Temperature [K]')
    plt.title(pp.title_str)
    plt.legend()

    thermal_energy = utilities.mean_energy(np.zeros(len(df)), df)
    denom = np.array(fieldVector)*np.array(vd_3)
    excess_energy = np.array(meanE_3) - thermal_energy
    energy_RT = excess_energy/denom

    plt.figure()
    plt.plot(kvcm,energy_RT*1e12,'o-', linewidth=2, label='FDM')
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Energy Relaxation Time [ps]')
    plt.title(pp.title_str)
    plt.ylim([0,12])
    plt.legend()

    plt.figure()
    plt.plot(kvcm,excess_energy,'o-', linewidth=2, label='FDM')
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Excess Energy [eV]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,c.e*denom,'o-', linewidth=2, label='FDM')
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'eE*vd [W]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,np.array(mom_RT),'o-', linewidth=2, label='FDM')
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Momentum Relaxation Time [fs]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,np.array(meanE_1)-np.min(df['energy [eV]']),'o-', linewidth=2, label='RTA')
    plt.plot(kvcm,np.array(meanE_2)-np.min(df['energy [eV]']),'o-', linewidth=2, label='L-F')
    plt.plot(kvcm,np.array(meanE_3)-np.min(df['energy [eV]']),'o-', linewidth=2, label='FDM')
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Mean energy above CBM [eV]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(meanE_1,mu_1,'o-', linewidth=2, label='RTA')
    plt.plot(meanE_2,mu_2,'o-', linewidth=2, label='L-F')
    plt.plot(meanE_3,mu_3,'o-', linewidth=2, label='FDM')
    # plt.plot(meanE_3t,mu_3t,'o-', linewidth=2, label='FDM {:.1e} GHz'.format(freq))
    plt.xlabel(r'Mean energy [eV]')
    plt.ylabel(r'Linear mobility [$cm^2 V^{-1} s^{-1}$]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(meanE_1,np.array(mu_1)*c.e*n/100**2,'o-', linewidth=2, label='RTA')
    plt.plot(meanE_2,np.array(mu_2)*c.e*n/100**2,'o-', linewidth=2, label='L-F')
    plt.plot(meanE_3,np.array(mu_3)*c.e*n/100**2,'o-', linewidth=2, label='FDM')
    # plt.plot(meanE_3t,mu_3t,'o-', linewidth=2, label='FDM {:.1e} GHz'.format(freq))
    plt.xlabel(r'Mean energy [eV]')
    plt.ylabel(r'Linear conductivity [$S/m$]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,meanE_3g,'o-', linewidth=2, label='FDM Gamma')
    plt.plot(kvcm,meanE_3l,'o-', linewidth=2, label='FDM L')
    if pp.getX:
        plt.plot(kvcm, meanE_3x, 'o-', linewidth=2, label='FDM X')
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Mean energy [eV]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,ng_right,'o-', linewidth=2, label=r'Gamma Left')
    plt.plot(kvcm,ng_left,'o-', linewidth=2, label=r'Gamma Right')
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Carrier Population [m$^-3$]')
    plt.title(pp.title_str)
    plt.legend()

    x = df.loc[gamma_icinds_r,'kx [1/A]'].values / (2 * np.pi / c.alat)
    y = df.loc[gamma_icinds_r,'ky [1/A]'].values / (2 * np.pi / c.alat)
    z = df.loc[gamma_icinds_r,'kz [1/A]'].values / (2 * np.pi / c.alat)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    plt.figure()
    plt.plot(kvcm,ng_1,'o-', linewidth=2, label=r'RTA Gamma')
    plt.plot(kvcm,ng_2,'o-', linewidth=2, label=r'L-F Gamma')
    plt.plot(kvcm,ng_3,'o-', linewidth=2, label=r'FDM Gamma')
    # plt.plot(kvcm,ng_3t,'o-', linewidth=2, label=r'FDM {:.1e} GHz Gamma'.format(freq))

    plt.plot(kvcm,nl_1,'o-', linewidth=2, label=r'RTA L')
    plt.plot(kvcm,nl_2,'o-', linewidth=2, label=r'L-F L')
    plt.plot(kvcm,nl_3,'o-', linewidth=2, label=r'FDM L')
    # plt.plot(kvcm,nl_3t,'o-', linewidth=2, label=r'FDM {:.1e} GHz L'.format(freq))
    if pp.getX:
        plt.plot(kvcm, nx_3, 'o-', linewidth=2, label='FDM X')
        # plt.plot(kvcm,np.array(nx_3)+np.array(nl_3)+np.array(ng_3),'--',linewidth=2,label='Sum')

    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Carrier Population [m$^-3$]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,vd_3g,'o-', linewidth=2, label=r'FDM Gamma')
    plt.plot(kvcm,vd_3l,'o-', linewidth=2, label=r'FDM L')
    if pp.getX:
        plt.plot(kvcm, vd_3x, 'o-', linewidth=2, label='FDM X')
        plt.plot(kvcm, np.array(vd_3l) * np.array(nl_3) / np.real(n) + np.array(vd_3g) * np.array(ng_3) / np.real(n) +
                 np.array(vd_3x) * np.array(nx_3) / np.real(n),
                 '-o', linewidth=2, label=r'Bulk Drift')
    else:
        plt.plot(kvcm,np.array(vd_3l)*np.array(nl_3)/np.real(np.array(ng_3)+np.array(nl_3))+np.array(vd_3g)*np.array(ng_3)/np.real(np.array(ng_3)+np.array(nl_3)),'-o', linewidth=2, label=r'Bulk Drift')

    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Drift velocity [m/s]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,vm_3g,'o-', linewidth=2, label=r'FDM Gamma')
    plt.plot(kvcm,vm_3l,'o-', linewidth=2, label=r'FDM L')
    if pp.getX:
        plt.plot(kvcm, vm_3x, 'o-', linewidth=2, label=r'FDM X')
    plt.plot(kvcm,np.array(vd_3l)*np.array(nl_3)/np.real(n)+np.array(vd_3g)*np.array(ng_3)/np.real(n),'-o', linewidth=2, label=r'Bulk Drift')
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Average x-velocity magnitude [m/s]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,super_cutoff_pop/n*100,'o-', linewidth=2)
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Population percentage above cutoff')
    plt.title(pp.title_str)
    plt.legend()


def plot_L_valley_drift(df,fieldVector):
    """Plot the drift velocities arising out of each L valley. The notion of drift velocity is tricky when applied to
    an inidividual valley. We have seen that the contribution out of each valley is non-zero, even at equilibrium. It
    might be better to intrepret this as an average velocity in the kx direction.

    Parameters:
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        Nothing. Just the plots.
    """
    L1_inds,L2_inds,L3_inds,L4_inds,L5_inds,L6_inds,L7_inds,L8_inds = utilities.split_L_valleys(df)
    vd_3, vd_3g, vd_3l, vd_3l1, vd_3l2, vd_3l3, vd_3l4, vd_3l5, vd_3l6, vd_3l7, vd_3l8 = ([] for i in range(11))
    n_3l1, n_3l2, n_3l3, n_3l4, n_3l5, n_3l6, n_3l7, n_3l8 = ([] for i in range(8))

    g_inds, l_inds, x_inds = utilities.gaas_split_valleys(df, False)
    for ee in fieldVector:
        chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        vd_3.append(utilities.mean_velocity(chi_3_i,df))
        vd_3g.append(utilities.mean_velocity(chi_3_i[g_inds],df.loc[g_inds]))
        vd_3l.append(utilities.mean_velocity(chi_3_i[l_inds],df.loc[l_inds]))
        vd_3l1.append(utilities.mean_velocity(chi_3_i[L1_inds],df.loc[L1_inds]))
        vd_3l2.append(utilities.mean_velocity(chi_3_i[L2_inds],df.loc[L2_inds]))
        vd_3l3.append(utilities.mean_velocity(chi_3_i[L3_inds],df.loc[L3_inds]))
        vd_3l4.append(utilities.mean_velocity(chi_3_i[L4_inds],df.loc[L4_inds]))
        vd_3l5.append(utilities.mean_velocity(chi_3_i[L5_inds],df.loc[L5_inds]))
        vd_3l6.append(utilities.mean_velocity(chi_3_i[L6_inds],df.loc[L6_inds]))
        vd_3l7.append(utilities.mean_velocity(chi_3_i[L7_inds],df.loc[L7_inds]))
        vd_3l8.append(utilities.mean_velocity(chi_3_i[L8_inds],df.loc[L8_inds]))

        n_3l1.append(utilities.calc_popinds(chi_3_i,df,L1_inds))
        n_3l2.append(utilities.calc_popinds(chi_3_i,df,L2_inds))
        n_3l3.append(utilities.calc_popinds(chi_3_i,df,L3_inds))
        n_3l4.append(utilities.calc_popinds(chi_3_i,df,L4_inds))
        n_3l5.append(utilities.calc_popinds(chi_3_i,df,L5_inds))
        n_3l6.append(utilities.calc_popinds(chi_3_i,df,L6_inds))
        n_3l7.append(utilities.calc_popinds(chi_3_i,df,L7_inds))
        n_3l8.append(utilities.calc_popinds(chi_3_i,df,L8_inds))

    kvcm = np.array(fieldVector) * 1e-5
    plt.figure()
    # plt.plot(kvcm,vd_3, 'o-', linewidth=2, label=r'Total')
    # plt.plot(kvcm,vd_3g, 'o-', linewidth=2, label=r'$\Gamma$')

    plt.plot(kvcm, vd_3l1, 'o-', linewidth=2, label='L1')
    plt.plot(kvcm, vd_3l2, 'o-', linewidth=2, label='L2')
    plt.plot(kvcm, vd_3l3, 'o-', linewidth=2, label='L3')
    plt.plot(kvcm, vd_3l4, 'o-', linewidth=2, label='L4')
    plt.plot(kvcm, vd_3l5, 'o-', linewidth=2, label='L5')
    plt.plot(kvcm, vd_3l6, 'o-', linewidth=2, label='L6')
    plt.plot(kvcm, vd_3l7, 'o-', linewidth=2, label='L7')
    plt.plot(kvcm, vd_3l8, 'o-', linewidth=2, label='L8')
    plt.plot(kvcm,vd_3l, 'o-', linewidth=2, label='Total L')
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Drift velocity [m/s]')
    plt.title(pp.title_str)
    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(prop=fontP, loc='upper left', ncol=2)


    plt.figure()
    plt.plot(kvcm, n_3l1, 'o-', linewidth=2, label='L1')
    plt.plot(kvcm, n_3l2, 'o-', linewidth=2, label='L2')
    plt.plot(kvcm, n_3l3, 'o-', linewidth=2, label='L3')
    plt.plot(kvcm, n_3l4, 'o-', linewidth=2, label='L4')
    plt.plot(kvcm, n_3l5, 'o-', linewidth=2, label='L5')
    plt.plot(kvcm, n_3l6, 'o-', linewidth=2, label='L6')
    plt.plot(kvcm, n_3l7, 'o-', linewidth=2, label='L7')
    plt.plot(kvcm, n_3l8, 'o-', linewidth=2, label='L8')
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Population [m^-3]')
    plt.title(pp.title_str)
    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(prop=fontP, loc='upper left', ncol=2)



    chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(fieldVector[0]))
    occupation_plotter.velocity_distribution_kde(chi_3_i[L1_inds], df.loc[L1_inds].reset_index())

    plt.figure()
    plt.plot(df.loc[L1_inds,'k_FD'], df.loc[L1_inds,'vx [m/s]'], 'o', linewidth=2, label='L1')

    print(np.sum(df.loc[L1_inds,'k_FD']*df.loc[L1_inds,'vx [m/s]'])/np.sum(df.loc[L1_inds,'k_FD']))
    print(np.sum(df.loc[L2_inds,'k_FD']*df.loc[L2_inds,'vx [m/s]'])/np.sum(df.loc[L2_inds,'k_FD']))
    print(np.sum(df.loc[L3_inds,'k_FD']*df.loc[L3_inds,'vx [m/s]'])/np.sum(df.loc[L3_inds,'k_FD']))
    print(np.sum(df.loc[L4_inds,'k_FD']*df.loc[L4_inds,'vx [m/s]'])/np.sum(df.loc[L4_inds,'k_FD']))
    print(np.sum(df.loc[L5_inds,'k_FD']*df.loc[L5_inds,'vx [m/s]'])/np.sum(df.loc[L5_inds,'k_FD']))
    print(np.sum(df.loc[L6_inds,'k_FD']*df.loc[L6_inds,'vx [m/s]'])/np.sum(df.loc[L6_inds,'k_FD']))
    print(np.sum(df.loc[L7_inds,'k_FD']*df.loc[L7_inds,'vx [m/s]'])/np.sum(df.loc[L7_inds,'k_FD']))
    print(np.sum(df.loc[L8_inds,'k_FD']*df.loc[L8_inds,'vx [m/s]'])/np.sum(df.loc[L8_inds,'k_FD']))
    counts, bins = np.histogram(df.loc[L1_inds,'vx [m/s]']*df.loc[L1_inds,'k_FD'],bins=1001)
    plt.figure()
    plt.hist(df.loc[L1_inds,'vx [m/s]'].values, bins = 21, alpha = 0.3, label = 'Positive kx octant')
    plt.hist(df.loc[L8_inds,'vx [m/s]'].values, bins = 21, alpha =0.3, label = 'Negative kx octant')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Bin count')
    plt.legend()

    plt.figure()
    plt.plot(df.loc[L1_inds,'kx [1/A]'].values, df.loc[L1_inds,'energy [eV]'].values,'.')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    plt.ylabel('Velocity [m/s]')
    plt.xlabel('k-axis [1/A]')
    plt.legend()

    plt.figure()
    plt.plot(df.loc[L1_inds,'kz [1/A]'].values, df.loc[L1_inds,'energy [eV]'].values,'.')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    plt.ylabel('Velocity [m/s]')
    plt.xlabel('k-axis [1/A]')
    plt.legend()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = df['kx [1/A]'].values / (2 * np.pi / c.a)
    y = df['ky [1/A]'].values / (2 * np.pi / c.a)
    z = df['kz [1/A]'].values / (2 * np.pi / c.a)
    ax.scatter(x[g_inds], y[g_inds], df.loc[g_inds,'energy [eV]'], s=5, c = df.loc[g_inds,'energy [eV]'])
    ax.set_xlabel(r'$kx/2\pi a$')
    ax.set_ylabel(r'$ky/2\pi a$')
    ax.set_zlabel(r'Energy [eV]')
    fontP = FontProperties()
    fontP.set_size('small')
    plt.title('Gamma valley')


if __name__ == '__main__':
    fields = pp.fieldVector
    freq = pp.freqGHz
    freqs = pp.freqVector
    # preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)

    plot_steady_transport_moments(electron_df, fields)
    # plot_L_valley_drift(electron_df,fields)
    material_plotter.bz_3dscatter(electron_df, True, False)


    # plot_small_signal_conductivity(fields,freqs,electron_df)


    # This is for comparing parabolic energy to the Perturbo generated energy. Can just leave commented.
    # mup = np.min(electron_df['energy [eV]']) - pp.mu
    # electron_df['par_energy [eV]'] = c.hbar_joule ** 2 / (2 * 0.063 * 9.11e-31) * (
    #             (electron_df['kx [1/A]'] * 10 ** 10) ** 2 + (electron_df['ky [1/A]'] * 10 ** 10) ** 2 + (
    #                 electron_df['kz [1/A]'] * 10 ** 10) ** 2) / c.e + pp.mu + mup
    # electron_df['k_FD_par'] = (np.exp(
    #     (electron_df['par_energy [eV]'].values * c.e - pp.mu * c.e) / (c.kb_joule * pp.T)) + 1) ** (-1)
    # g_inds, _, _ = utilities.gaas_split_valleys(electron_df, False)
    # print(mup)
    # df = electron_df.copy(deep=True)
    # df = df.loc[g_inds]
    # plt.figure()
    # plt.plot(df['kx [1/A]'], df['par_energy [eV]'] - pp.mu, '.', label='Parabolic')
    # plt.plot(electron_df['kx [1/A]'], electron_df['energy [eV]'] - pp.mu, '.', label='Perturbo')
    # plt.legend()
    # plt.ylabel('Energy Above Fermi Level [eV]')
    # plt.xlabel('kx [1/A]')
    # plt.title(pp.title_str)
    # print((np.sum((df['par_energy [eV]']) * df['k_FD_par']) / np.sum(df['k_FD_par']) - pp.mu - mup) / c.kb_ev / pp.T)
    # print(utilities.mean_energy(np.zeros(len(electron_df)), electron_df) / (c.kb_ev * pp.T) - pp.mu / (
    #             c.kb_ev * pp.T) - mup / (c.kb_ev * pp.T))

    plt.show()