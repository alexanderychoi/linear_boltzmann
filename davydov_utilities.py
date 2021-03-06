import numpy as np
import constants as c
import utilities
import problem_parameters as pp
import scipy.special as sc
from scipy import integrate
import matplotlib.pyplot as plt
import psd_solver
import matplotlib
import matplotlib.font_manager
import paper_figures
from scipy import interpolate

# Set the parameters for the paper figures
# SMALL_SIZE = 7.6
SMALL_SIZE = 8
# MEDIUM_SIZE = 8.5
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
different_small_size = 5.7


# For frequency PSD comment out otherwise
singleCol_doubleSize = (3.375/2,3.375/1.6)
# SMALL_SIZE = 7.6
SMALL_SIZE = 6
# MEDIUM_SIZE = 8.5
MEDIUM_SIZE = 9.5
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=different_small_size)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.rcParams["font.family"] = "sans-serif"

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

# perturboColor = 'firebrick'
# frohlichColor = '#2E8BC0'
# adpColor = '#0C2D48'

perturboColor = 'firebrick'
frohlichColor = '#D7A449'
adpColor = '#0B1F65'

eq_color = '#0C1446'
med_color = '#2740DA'
high_color = '#FF3F00'

columnwidth = 3.375
triFigSize = (2.25, 2.5)
dualFigSize = (2.58,2.58)
squareFigSize = (2.53125, 2.53125)

def acoustic_davydovDistribution(df,ee,pA,eRT_Ratio):
    carrierEnergy = (df['energy [eV]'] - pp.mu)*c.e
    hyperGeom = sc.hyp2f1(1, 2, 3, -3 * carrierEnergy * c.kb_joule * pp.T * c.m_gaas / (2 * pA**2 * eRT_Ratio * c.e**2 * ee**2))
    prefactor = -3*c.m_gaas*carrierEnergy**2/(4*pA**2*eRT_Ratio*c.e**2*ee**2)
    davydov = np.exp(prefactor*hyperGeom)
    normalizing = np.sum(davydov)/np.sum(df['k_FD'])

    return davydov/normalizing


def acoustic_davydovNoise_lowfreq(df,davydovDist,momRT):
    # Taken from GGK 3.76
    Nuc = pp.kgrid ** 3
    carrierEnergy = (df['energy [eV]'] - pp.mu)*c.e
    n = utilities.calculate_density(df)
    davyNoise = 4/3*c.e**2*n/(c.m_gaas*Nuc*c.Vuc)*np.sum(davydovDist*momRT*carrierEnergy**(1.5))/np.sum(carrierEnergy**(0.5)*davydovDist)

    return davyNoise


def acoustic_davydovNoise_hifreq(df,davydovDist,momRT,freq):
    # Taken from GGK 3.76
    Nuc = pp.kgrid ** 3
    angFreq = 2*np.pi*freq*1e9
    carrierEnergy = (df['energy [eV]'] - pp.mu)*c.e
    n = utilities.calculate_density(df)
    davyNoise = 4/3*c.e**2*n/(c.m_gaas*Nuc*c.Vuc*angFreq**2)*np.sum(davydovDist*1/momRT*carrierEnergy**(1.5))/np.sum(carrierEnergy**(0.5)*davydovDist)

    return davyNoise


def frohlich_davydovNoise_lowfreq(df,davydovDist,momRT):
    # Taken from GGK 3.76
    Nuc = pp.kgrid ** 3
    carrierEnergy = (df['energy [eV]'] - pp.mu)*c.e
    sortedInds = np.argsort(carrierEnergy)

    n = utilities.calculate_density(df)
    davyNoise = 4/3*c.e**2*n/(c.m_gaas*Nuc*c.Vuc)*np.sum(davydovDist*momRT*carrierEnergy[sortedInds]**(1.5))/np.sum(carrierEnergy**(0.5)*davydovDist)

    return davyNoise


def frohlich_davydovNoise_hifreq(df,davydovDist,momRT,freq):
    # Taken from GGK 3.76
    angFreq = 2*np.pi*freq*1e9
    Nuc = pp.kgrid ** 3
    carrierEnergy = (df['energy [eV]'] - pp.mu)*c.e
    sortedInds = np.argsort(carrierEnergy)

    n = utilities.calculate_density(df)
    davyNoise = 4/3*c.e**2*n/(2*np.pi*c.m_gaas*Nuc*c.Vuc*angFreq**2)*np.sum(davydovDist*1/momRT*carrierEnergy[sortedInds]**(1.5))/np.sum(carrierEnergy**(0.5)*davydovDist)

    return davyNoise


def frohlichDiffusionRatio(df,davydovDist,momRT,eRT_ratio):
    # Taken from GGK 3.78
    Nuc = pp.kgrid ** 3
    carrierEnergy = (df['energy [eV]'] - pp.mu)*c.e
    sortedInds = np.argsort(carrierEnergy)

    momRTFro = 1 / frohlichScattering(g_df, opticalPhononEnergy, inverseFrohlichTime)
    eRTFro = eRT_Ratio*momRTFro

    theta = 2*c.e**2/(3*c.m_gaas*c.kb_joule*pp.T)*momRTFro*eRTFro  # GGK 3.80




def acoustic_davydovRTs(df,pA,eRT_Ratio):
    carrierEnergy = (df['energy [eV]'] - pp.mu) * c.e
    momRT = pA*(carrierEnergy)**(-0.5)
    eRT = eRT_Ratio*pA*(carrierEnergy)**(-0.5)

    return momRT,eRT


def acoustic_davydovPartialEnergy(df,ee,pA,eRT_Ratio):
    carrierEnergy = (df['energy [eV]'] - pp.mu) * c.e
    davydovDist = acoustic_davydovDistribution(df, ee, pA, eRT_Ratio)
    hypergeom_a = sc.hyp2f1(1, 2, 3, -3 * carrierEnergy * c.kb_joule * pp.T * c.m_gaas / (2 * pA**2 * eRT_Ratio * c.e**2 * ee**2))
    hypergeom_b = sc.hyp2f1(2, 3, 4, -3 * carrierEnergy * c.kb_joule * pp.T * c.m_gaas / (2 * pA**2 * eRT_Ratio * c.e**2 * ee**2))

    term_a = -3*carrierEnergy*c.m_gaas/(2*pA**2*eRT_Ratio*c.e**2*ee**2)*hypergeom_a
    term_b = -carrierEnergy**2*c.m_gaas/(2*pA**2*eRT_Ratio*c.e**2*carrierEnergy**2)*hypergeom_b

    return davydovDist*(term_a+term_b)


def acoustic_davydovMeanEnergy(df,ee,pA,eRT_Ratio):
    carrierEnergy = (df['energy [eV]'] - pp.mu) * c.e
    davydovDist = acoustic_davydovDistribution(df, ee, pA, eRT_Ratio)
    np.sum(davydovDist*carrierEnergy)/np.sum(davydovDist)

    return np.sum(davydovDist*carrierEnergy)/np.sum(davydovDist)


def frohlichScattering(df,opticalPhononEn,inverseFrohlichTime):
    parEnergy = c.hbar_joule ** 2 / (2 * c.m_gaas) * (
                (df['kx [1/A]'] * 10 ** 10) ** 2 + (df['ky [1/A]'] * 10 ** 10) ** 2 + (
                    df['kz [1/A]'] * 10 ** 10) ** 2)+ opticalPhononEn/100

    boseEinstein = 1/(np.exp(opticalPhononEn/(c.kb_joule*pp.T))-1)
    ephRatio = np.abs(parEnergy/opticalPhononEn)
    scatteringRate = 1/np.sqrt(ephRatio)*(boseEinstein*np.sqrt(np.arcsinh(ephRatio))+(boseEinstein+1)*np.sqrt(np.arcsinh(ephRatio-1)+0j))

    return np.real(scatteringRate)*inverseFrohlichTime


def acoustic_davydovIntegrand(df,ee,pA,eRT_Ratio):
    carrierEnergy = (df['energy [eV]'].values - pp.mu)
    sortedInds = np.argsort(carrierEnergy)

    plt.figure()
    plt.plot(np.arange(len(sortedInds+1)),carrierEnergy[sortedInds],'.')

    momRT,eRT = acoustic_davydovRTs(df, pA, eRT_Ratio)
    integrand = -1/(c.kb_joule*pp.T + 2*c.e**2*ee**2*momRT*eRT/(3*c.m_gaas))

    plt.figure()
    plt.plot(carrierEnergy[sortedInds]-np.min(carrierEnergy),integrand[sortedInds],'.')
    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel('Integrand')

    davydovArgument = integrate.cumtrapz(integrand[sortedInds],carrierEnergy[sortedInds]*c.e,initial=0)
    plt.figure()
    plt.plot(carrierEnergy[sortedInds]-np.min(carrierEnergy),davydovArgument,'.')
    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel('Cumtrapz')

    normalizing = np.sum(df['k_FD'])/np.sum(np.exp(davydovArgument))
    numericalDavydov = normalizing*np.exp(davydovArgument)

    plt.figure()
    plt.plot(carrierEnergy[sortedInds]-np.min(carrierEnergy),numericalDavydov,'-',label='Numerical Integration')
    plt.plot(carrierEnergy-np.min(carrierEnergy),acoustic_davydovDistribution(df,ee,pA,eRT_Ratio),'.',label='Analytical Integration')
    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel('Acoustic Davydov Distribution (unitless)')
    plt.legend()


def plot_scattering(df,pA,eRT_Ratio,opticalPhononEnergy, inverseFrohlichTime):
    lw = 1.5
    momRT,eRT = acoustic_davydovRTs(df, pA, eRT_Ratio)
    carrierEnergy = (df['energy [eV]'].values - pp.mu)

    nkpts = len(electron_df)
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    # utilities.check_matrix_properties(scm)
    rates = (-1) * np.diag(scm)
    g_inds, _, _ = utilities.gaas_split_valleys(electron_df, False)

    scatteringRate = frohlichScattering(df, opticalPhononEnergy, inverseFrohlichTime)
    sortedInds = np.argsort(carrierEnergy)
    inverseFrohlichTimeps = inverseFrohlichTime/1e12
    opticalPhononEnergymeV = opticalPhononEnergy*1000/c.e
    ratio = 1/eRT_Ratio
    fig, ax = plt.subplots()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = 'ADP Coefficient = %.1e ' % pA +r'$J^{1/2} \, s$'+ '\n' + r'$\tau_p/\tau_{\epsilon} = %.2f$' % ratio + \
              '\n' + r'Frohlich inverse RT = %.2f 1/ps' % inverseFrohlichTimeps + \
              '\n' + r'Optical Phonon Energy = %.2f meV' % opticalPhononEnergymeV

    ax.plot(g_df['energy [eV]'] - np.min(g_df['energy [eV]']), rates[g_inds], '.', label='PERTURBO')
    ax.plot(carrierEnergy[sortedInds] - np.min(carrierEnergy), scatteringRate[sortedInds], '--', label = r'$\rm Fr\"{o}hlich$')
    ax.plot(g_df['energy [eV]'] - np.min(g_df['energy [eV]']), 1/momRT, '--', label='ADP')
    # ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)

    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel('Scattering Rate '+r'$\rm (fs^{-1})$')
    plt.legend()

    dat_300 = np.loadtxt(pp.inputLoc+'relaxation_time_dense.dat', skiprows=1)
    taus_300 = dat_300[:, 4]  # fs
    enk_300 = dat_300[:, 3]  # eV


    fig, ax = plt.subplots(figsize=triFigSize)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = 'ADP Coefficient = %.1e ' % pA +r'$J^{1/2} \, s$'+ '\n' + r'$\tau_p/\tau_{\epsilon} = %.2f$' % ratio + \
              '\n' + r'Frohlich inverse RT = %.2f 1/ps' % inverseFrohlichTimeps + \
              '\n' + r'Optical Phonon Energy = %.2f meV' % opticalPhononEnergymeV
    ax.plot(g_df['energy [eV]'] - np.min(g_df['energy [eV]']), 1/momRT/1e12, '--', label='ADP',color=adpColor,linewidth=lw)
    ax.plot(carrierEnergy[sortedInds] - np.min(carrierEnergy), scatteringRate[sortedInds]/1e12, '--', label=r'$\rm Fr\"{o}hlich$',color=frohlichColor,linewidth=lw)
    ax.plot(enk_300 - np.min(enk_300), 1/taus_300*1000, '.', label='PERTURBO',color=perturboColor,markersize=2)
    ax.axvline(39/1000,linestyle='--',color='black',lw=0.5)
    ax.annotate(r'$\hbar \, \omega_{LO}$', xy=(39/1000, 450), xytext=(60/1000, 440),size=14,
                arrowprops=dict(facecolor='black', arrowstyle="->"),
                )
    # ax.text(0.57, 0.77, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Scattering rate ' + r'$(ps^{-1})$')
    plt.xlim([-0.02,0.32])
    plt.ylim([-2,27])
    plt.legend()
    # plt.savefig(pp.figureLoc +'davydovRates.png', bbox_inches='tight',dpi=600)


    fig, ax = plt.subplots(figsize=squareFigSize)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = 'ADP Coefficient = %.1e ' % pA +r'$J^{1/2} \, s$'+ '\n' + r'$\tau_p/\tau_{\epsilon} = %.2f$' % ratio + \
              '\n' + r'Frohlich inverse RT = %.2f 1/ps' % inverseFrohlichTimeps + \
              '\n' + r'Optical Phonon Energy = %.2f meV' % opticalPhononEnergymeV
    df['momRT_ADP'] = momRT
    adp_df = df.sort_values(by='energy [eV]',ascending=True)

    ax.plot((adp_df['energy [eV]'] - np.min(adp_df['energy [eV]']))*1000, adp_df['momRT_ADP'].values*1e15, '--', label='ADP',color=adpColor,linewidth=lw)
    ax.plot((carrierEnergy[sortedInds] - np.min(carrierEnergy))*1000, 1/scatteringRate[sortedInds]*1e15, '--', label=r'$\rm Fr\"{o}hlich$',color=frohlichColor,linewidth=lw)
    ax.plot((enk_300 - np.min(enk_300))*1000, taus_300, '.', label='PERTURBO',color=perturboColor,markersize=1.5)
    ax.axvline(39,linestyle='--',color='black',lw=0.5)
    ax.annotate(r'$\hbar \, \omega_{LO}$', xy=(39, 50), xytext=(60, 40),size=SMALL_SIZE,
                arrowprops=dict(facecolor='black', arrowstyle="->"),
                )
    # ax.text(0.57, 0.77, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.xlabel('Energy (meV)')
    plt.ylabel('Relaxation time (fs)')
    plt.xlim([0,0.3*1000])
    plt.ylim([0,700])
    plt.legend(frameon=False)
    plt.savefig(pp.figureLoc +'davydovTimes.png', bbox_inches='tight',dpi=600)


def frohlich_davydovDistribution(df,ee,opticalPhononEnergy,inverseFrohlichTime,eRT_Ratio):
    carrierEnergy = (df['energy [eV]'] - pp.mu)
    sortedInds = np.argsort(carrierEnergy)
    scatteringRate = frohlichScattering(df, opticalPhononEnergy, inverseFrohlichTime)
    frohlichRTs = 1/scatteringRate
    integrand = -1 / (c.kb_joule * pp.T + 2 * c.e ** 2 * ee ** 2 *frohlichRTs**2 * eRT_Ratio / (3 * c.m_gaas))
    davydovArgument = integrate.cumtrapz(integrand[sortedInds], carrierEnergy[sortedInds] * c.e, initial=0)
    normalizing = np.sum(df['k_FD'])/np.sum(np.exp(davydovArgument))

    return normalizing*np.exp(davydovArgument)


def plotdavydovDistributions(ee, df, pA, opticalPhononEnergy, inverseFrohlichTime,eRT_Ratio):
    chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))

    f = chi + electron_df['k_FD']
    davydovFro = frohlich_davydovDistribution(df,ee,opticalPhononEnergy,inverseFrohlichTime,eRT_Ratio)
    davydovAco = acoustic_davydovDistribution(df, ee, pA, eRT_Ratio)
    carrierEnergy = (df['energy [eV]'] - np.min(df['energy [eV]']))
    sortedInds = np.argsort(carrierEnergy)

    fig, ax = plt.subplots()
    ax.plot(electron_df['energy [eV]'] - np.min(electron_df['energy [eV]']), f, '.', label='PERTURBO',color='black')
    ax.plot(carrierEnergy[sortedInds], davydovFro,'.',color='tomato',label = r'$\rm Fr\"{o}hlich$')
    ax.plot(carrierEnergy,davydovAco, '.',color='dodgerblue', label='Acoustic')
    plt.legend()
    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel('Occupation (unitless)')
    plt.yscale('log')


def plot_davydov_density_v_field(fieldVector, freq, df, pA, opticalPhononEnergy, inverseFrohlichTime,eRT_Ratio):
    lw = 1.5
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # textstr = '\n'.join((r'$f = %.1f GHz \, \, (100) $' % (freq,), pp.fdmName))
    ratio = 1/eRT_Ratio
    textstr = 'ADP Coefficient = %.1e ' % pA + r'$J^{1/2} \, s$'+ '\n' + r'$\tau_p/\tau_{\epsilon} = %.2f$' % ratio
    S_xx_vector = []
    S_yy_vector = []
    conductivity_xx_vector = []
    S_xx_acoustic_Davy = []
    S_xx_frohlich_Davy = []
    pertEnergy = []
    davyAcoEnergy = []
    davyFroEnergy = []
    carrierEnergy = (df['energy [eV]'] - np.min(df['energy [eV]']))
    sortedInds = np.argsort(carrierEnergy)
    for ee in fieldVector:
        chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        S_xx, _, S_yy, _, conductivity_xx = psd_solver.density(chi, ee, electron_df, freq, False, 0)
        S_xx_vector.append(S_xx)
        S_yy_vector.append(S_yy)
        davydovFro = frohlich_davydovDistribution(df,ee,opticalPhononEnergy,inverseFrohlichTime,eRT_Ratio)
        davydovAco = acoustic_davydovDistribution(df, ee, pA, eRT_Ratio)
        momRTAco, _ = acoustic_davydovRTs(df, pA, eRT_Ratio)
        momRTFro = 1/frohlichScattering(df,opticalPhononEnergy,inverseFrohlichTime)
        S_xx_acoustic_Davy.append(acoustic_davydovNoise_lowfreq(df,davydovAco, momRTAco))
        S_xx_frohlich_Davy.append(frohlich_davydovNoise_lowfreq(df,davydovFro, momRTFro[sortedInds]))
        conductivity_xx_vector.append(conductivity_xx)
        # pertEnergy.append(utilities.mean_energy(chi[g_inds],df))
        pertEnergy.append(np.sum(carrierEnergy*(chi[g_inds]+df['k_FD']))/np.sum(df['k_FD']))
        davyAcoEnergy.append(np.sum(carrierEnergy*davydovAco)/np.sum(davydovAco))
        davyFroEnergy.append(np.sum(carrierEnergy[sortedInds]*davydovFro)/np.sum(davydovFro))

    vcm = np.array(fieldVector) * 1e-2
    Nuc = pp.kgrid ** 3
    fig, ax = plt.subplots(figsize=squareFigSize)
    # ax.axhline(np.array(conductivity_xx_vector[0]) * 4 * c.kb_joule * pp.T / c.Vuc / Nuc, linestyle='--',color='black',lw=0.5)
    ax.axhline(np.array(S_xx_vector[0]/S_xx_vector[0]) , linestyle='--', color='black',lw=0.5)
    ax.plot(vcm, np.array(S_xx_acoustic_Davy)/S_xx_vector[0],'--',color=adpColor, label='ADP',linewidth=lw)
    ax.plot(vcm, np.array(S_xx_frohlich_Davy)/S_xx_vector[0],'--',color=frohlichColor, label = r'$\rm Fr\"{o}hlich$',linewidth=lw)
    ax.plot(vcm, np.array(S_xx_vector)/S_xx_vector[0], label='PERTURBO',color=perturboColor,linewidth=lw)

    plt.legend(loc='lower left',frameon=False)
    plt.xlabel(r'Electric field ($\rm V \, cm^{-1}$)')
    # plt.ylabel(r'$S_{\parallel}$ '+ r'$\rm (A^2 \, m^{-4} \, Hz^{-1})$')
    plt.ylabel('Current PSD ' r'(norm.)')
    ax.locator_params(axis='x', nbins=6)
    # plt.ylim([0,1.5])
    plt.xlim([0,500])
    ax.locator_params(nbins=6, axis='y')
    plt.savefig(pp.figureLoc +'density_vField.png', bbox_inches='tight',dpi=600)

    T_vector = np.geomspace(300,500,1000)
    energy_vector = paper_figures.calculate_electron_temperature(df,T_vector)

    aco_Temp = np.interp(davyAcoEnergy,energy_vector,T_vector)
    fro_Temp = np.interp(davyFroEnergy,energy_vector,T_vector)
    minEnergy = np.min(df['energy [eV]'])
    pertEnergy_red = np.array(pertEnergy)-minEnergy
    pert_Temp = np.interp(pertEnergy,energy_vector,T_vector)

    fig, ax = plt.subplots(figsize=squareFigSize)
    ax.plot(vcm, aco_Temp,'--',color=adpColor, label='ADP',linewidth=lw)
    ax.plot(vcm, fro_Temp,'--',color=frohlichColor, label = r'$\rm Fr\"{o}hlich$',linewidth=lw)
    ax.plot(vcm, pert_Temp, color = perturboColor, label='PERTURBO',linewidth=lw)
    plt.legend(frameon=False)
    plt.xlabel(r'Electric field ($\rm V \, cm^{-1}$)')
    plt.ylabel('Electron temperature (K)')
    plt.xlim([0,500])
    ax.locator_params(axis='x', nbins=6)
    plt.savefig(pp.figureLoc +'meanEnergy.png', bbox_inches='tight',dpi=600)


def plot_density(ee, freqVector, full_df, pA, opticalPhononEnergy, inverseFrohlichTime, eRT_Ratio):
    lw = 1.5
    g_inds, _, _ = utilities.gaas_split_valleys(full_df, False)
    g_df = full_df[g_inds].reset_index(drop=True)

    davydovFro = frohlich_davydovDistribution(g_df, ee, opticalPhononEnergy, inverseFrohlichTime, eRT_Ratio)
    davydovAco = acoustic_davydovDistribution(g_df, ee, pA, eRT_Ratio)
    momRTAco, _ = acoustic_davydovRTs(g_df, pA, eRT_Ratio)
    momRTFro = 1 / frohlichScattering(g_df, opticalPhononEnergy, inverseFrohlichTime)
    carrierEnergy = (g_df['energy [eV]'] - np.min(g_df['energy [eV]']))
    sortedInds = np.argsort(carrierEnergy)

    # First plot the Perturbo spectral density as a function of frequency and the nyquist spectral density
    Nuc = pp.kgrid ** 3
    S_xx_vector = []
    S_xx_RTA_vector = []
    S_yy_vector = []
    cond = []
    for freq in freqVector:
        chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        S_xx, S_xx_RTA, S_yy, S_yy_RTA, conductivity_xx = psd_solver.density(chi, ee, full_df, freq, False, 0)
        S_xx_vector.append(S_xx)
        S_xx_RTA_vector.append(S_xx_RTA)
        S_yy_vector.append(S_yy)
        cond.append(
            np.load(pp.outputLoc + 'Small_Signal/' + 'cond_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, 1e-3)))

    initial = np.array(cond[0]) * 4 * c.kb_joule * pp.T / c.Vuc / Nuc
    fig, ax = plt.subplots(figsize=singleCol_doubleSize)
    ax.plot(freqVector, np.array(cond) * 4 * c.kb_joule * pp.T / c.Vuc / Nuc / initial, linestyle='-',
            label=r'0 $\rm V \, cm^{-1}$',color=eq_color)

    ax.plot(freqVector, np.array(S_xx_vector)/initial, color=high_color,
            label=r'$S_{\parallel}$ '+r'{:.0f} '.format(ee/100) + r'$\rm V \, cm^{-1}$',linewidth=lw)
    ax.plot(freqVector, np.array(S_yy_vector)/initial, color=high_color, linestyle=(0,(lw,1,1,1)),
            label=r'$S_{\perp}$ ' +r'{:.0f} '.format(ee/100) + r'$\rm V \, cm^{-1}$',linewidth=lw)

    S_xx_fro_Davy_lowfreq = frohlich_davydovNoise_lowfreq(g_df, davydovFro, momRTFro[sortedInds])
    S_yy_fro_Davy_lowfreq = frohlich_davydovNoise_lowfreq(g_df, davydovFro, momRTFro[sortedInds])/0.958181 # Taken from diffusion ratio calc
    lowfreq = freqVector[freqVector < 1]


    # ax.plot(lowfreq, np.ones(len(lowfreq))*S_xx_fro_Davy_lowfreq, color=frohlichColor, linestyle='-',
    #         label=r'$S_{\parallel}$' +' ' + r'$\rm Fr\"{o}hlich$'+r' {:.1f} '.format(ee/100) + r'$\rm V \, cm^{-1}$',linewidth=lw)
    # ax.plot(lowfreq, np.ones(len(lowfreq))*S_yy_fro_Davy_lowfreq, color=frohlichColor, linestyle='-.',
    #         label=r'$S_{\perp}$' +' ' + r'$\rm Fr\"{o}hlich$'+r' {:.1f} '.format(ee/100) + r'$\rm V \, cm^{-1}$',linewidth=lw)


    # Now plot the high frequency Davydov spectral density
    hifreq = freqVector[freqVector > 1000]
    S_xx_acoustic_Davy_hifreq = []
    S_xx_frohlich_Davy_hifreq = []

    for freq in hifreq:
        S_xx_acoustic_Davy_hifreq.append(acoustic_davydovNoise_hifreq(g_df, davydovAco, momRTAco, freq))
        S_xx_frohlich_Davy_hifreq.append(frohlich_davydovNoise_hifreq(g_df, davydovFro, momRTFro[sortedInds], freq))

    # ax.plot(hifreq, S_xx_frohlich_Davy_hifreq, color=frohlichColor, linestyle='-',linewidth=lw)


    plt.legend(loc='lower left',frameon=False,handletextpad=0.4,handlelength=1.2,borderaxespad=0.2,borderpad=0.2)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Current PSD ' r'(norm.)')
    plt.xlim([freqVector[0],freqVector[-1]])
    plt.xscale('log')
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=6)
    ax.xaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1,
                                          numticks=100)
    ax.xaxis.set_minor_locator(locmin)
    # xlabels=np.array([1, 2, 3,4,5])
    # ax.set_xticks(xlabels)
    # ax.set_xticks(10 ** xlabels)
    # ax.set_xticklabels(xlabels)
    # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.ylim([0,1.05])
    plt.xlim([freqVector[0],freqVector[-1]])
    # ax.locator_params(nbins=10, axis='x')

    # ax.annotate('Low-Frequency Limit', xy=(0.28, 3400), xytext=(0.065, 2800),size=8,
    #             arrowprops=dict(facecolor='black', arrowstyle="->"),
    #             )
    # ax.annotate('High-Frequency Limit', xy=(3700, 50), xytext=(800, 650),size=8,
    #             arrowprops=dict(facecolor='black', arrowstyle="->"),
    #             )
    f = interpolate.interp1d(S_xx_vector,freqVector)
    f_1p = f(S_xx_vector[0]*1.01)
    f_3dB = f(np.max(S_xx_vector)/2)
    f_20dB = f(np.max(S_xx_vector)/100)
    ax.axvline(f_1p,color='black',alpha=0.8,linestyle='--',lw=0.5,label='99 %')
    ax.axvline(f_3dB,color='black',alpha=0.6,linestyle='--',lw=0.5,label='50%')
    ax.axvline(f_20dB,color='black',alpha=0.4,linestyle='--',lw=0.5,label='1%')

    f2 = interpolate.interp1d(S_yy_vector, freqVector)
    f_1p_2 = f2(S_yy_vector[0]*0.99)
    f_3dB_2 = f2(S_yy_vector[0]/2)
    f_20dB_2 = f2(S_yy_vector[0]/100)

    # ax.axvline(f_1p_2,color='black',alpha=0.8,linestyle='-.',lw=0.5,label='99%')
    # ax.axvline(f_3dB_2,color='black',alpha=0.6,linestyle='-.',lw=0.5,label='50%')
    # ax.axvline(f_20dB_2,color='black',alpha=0.4,linestyle='-.',lw=0.5,label='1%')
    plt.title('Sxx lines')
    plt.savefig(pp.figureLoc + 'Freq_Dependent_PSD.png', bbox_inches='tight', dpi=600)



if __name__ == '__main__':
    fields = pp.moment_fields
    freqs = pp.freqVector

    # preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)
    g_inds,_,_ = utilities.gaas_split_valleys(electron_df,False)

    g_df = electron_df.loc[g_inds]

    # Set Davydov parameters
    # Momentum relaxation time coefficient for ADP
    # pA = 4.5558419e-23 # 12 Problem
    pA = 4.4481e-23 # 13 Problem

    # Ratio of energy to moment relaxation times for Frohlich and ADP
    eRT_Ratio = 6

    # Inverse Frohlich time as per Vatsal's paper Eqn. 1
    # inverseFrohlichTime = 5.6892967e12 # 12 Problem
    inverseFrohlichTime = 5.8271028e12 # 13 Problem
    # inverseFrohlichTime = 7.567429*10**12 # This is the value predicted by the band parameters

    # Optical phonon energy in J
    opticalPhononEnergy = 35/1000*c.e


    # acoustic_davydovIntegrand(g_df.reset_index(drop=True), 4e4, pA, eRT_Ratio)
    # plot_scattering(g_df.reset_index(drop=True),pA,eRT_Ratio,opticalPhononEnergy,inverseFrohlichTime)
    plot_davydov_density_v_field(pp.moment_fields, 0.1, g_df.reset_index(drop=True), pA, opticalPhononEnergy, inverseFrohlichTime, eRT_Ratio)
    # plotdavydovDistributions(4e4, g_df.reset_index(drop=True), pA, opticalPhononEnergy, inverseFrohlichTime, eRT_Ratio)
    plot_density(5e4, freqs, electron_df, pA, opticalPhononEnergy, inverseFrohlichTime, eRT_Ratio)
    plt.show()