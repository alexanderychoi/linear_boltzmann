import numpy as np
import constants as c
import utilities
import problem_parameters as pp
import scipy.special as sc
from scipy import integrate
import matplotlib.pyplot as plt
import psd_solver

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

def acoustic_davydovDistribution(df,ee,pA,eRT_Ratio):
    carrierEnergy = (df['energy [eV]'] - pp.mu)*c.e
    hyperGeom = sc.hyp2f1(1, 2, 3, -3 * carrierEnergy * c.kb_joule * pp.T * c.m_gaas / (2 * pA * eRT_Ratio * c.e**2 * ee**2))
    prefactor = -3*c.m_gaas*carrierEnergy**2/(4*pA*eRT_Ratio*c.e**2*ee**2)
    davydov = np.exp(prefactor*hyperGeom)
    normalizing = np.sum(davydov)/np.sum(df['k_FD'])

    return davydov/normalizing


def davydovNoise(df,davydovDist,momRT):
    # Taken from GGK 3.76
    Nuc = pp.kgrid ** 3
    carrierEnergy = (df['energy [eV]'] - pp.mu)*c.e
    n = utilities.calculate_density(df)
    davyNoise = 4/3*c.e**2*n/(c.m_gaas*Nuc*c.Vuc)*np.sum(davydovDist*momRT*carrierEnergy**(1.5))/np.sum(carrierEnergy**(0.5)*davydovDist)

    return davyNoise


def acoustic_davydovRTs(df,pA,eRT_Ratio):
    carrierEnergy = (df['energy [eV]'] - pp.mu) * c.e
    momRT = pA*(carrierEnergy)**(-0.5)
    eRT = eRT_Ratio*(carrierEnergy)**(-0.5)

    return momRT,eRT


def acoustic_davydovPartialEnergy(df,ee,pA,eRT_Ratio):
    carrierEnergy = (df['energy [eV]'] - pp.mu) * c.e
    davydovDist = acoustic_davydovDistribution(df, ee, pA, eRT_Ratio)
    hypergeom_a = sc.hyp2f1(1, 2, 3, -3 * carrierEnergy * c.kb_joule * pp.T * c.m_gaas / (2 * pA * eRT_Ratio * c.e**2 * ee**2))
    hypergeom_b = sc.hyp2f1(2, 3, 4, -3 * carrierEnergy * c.kb_joule * pp.T * c.m_gaas / (2 * pA * eRT_Ratio * c.e**2 * ee**2))

    term_a = -3*carrierEnergy*c.m_gaas/(2*pA*eRT_Ratio*c.e**2*ee**2)*hypergeom_a
    term_b = -carrierEnergy**2*c.m_gaas/(2*pA*eRT_Ratio*c.e**2*carrierEnergy**2)*hypergeom_b

    return davydovDist*(term_a+term_b)


def acoustic_davydovMeanEnergy(df,ee,pA,eRT_Ratio):
    carrierEnergy = (df['energy [eV]'] - pp.mu) * c.e
    davydovDist = acoustic_davydovDistribution(df, ee, pA, eRT_Ratio)
    np.sum(davydovDist*carrierEnergy)/np.sum(davydovDist)

    return np.sum(davydovDist*carrierEnergy)/np.sum(davydovDist)


def frohlichScattering(df,opticalPhononEn,inverseFrohlichTime):
    parEnergy = c.hbar_joule ** 2 / (2 * 0.063 * 9.11e-31) * (
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
    integrand = -1/(c.kb_joule*pp.T + 2*c.e**2*ee**2*momRT*eRT/(3*0.063*9.11e-31))

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
    ratio = eRT_Ratio
    fig, ax = plt.subplots()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = 'ADP Coefficient = %.1e ' % pA +r'$J^{1/2} \, s$'+ '\n' + r'$\tau_p/\tau_{\epsilon} = %.2f$' % ratio + \
              '\n' + r'Frohlich inverse RT = %.2f 1/ps' % inverseFrohlichTimeps + \
              '\n' + r'Optical Phonon Energy = %.2f meV' % opticalPhononEnergymeV

    ax.plot(g_df['energy [eV]'] - np.min(g_df['energy [eV]']), rates[g_inds], '.', label='Perturbo')
    # ax.plot(carrierEnergy[sortedInds] - np.min(carrierEnergy), scatteringRate[sortedInds], '.', label='Frohlich')
    ax.plot(carrierEnergy[sortedInds] - np.min(carrierEnergy), scatteringRate[sortedInds], '.', label='Frohlich')
    ax.plot(g_df['energy [eV]'] - np.min(g_df['energy [eV]']), 1/momRT, '.', label='Acoustic Deformation')
    ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)

    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel('Scattering Rate '+r'$(fs^{-1})$')
    plt.legend()

    dat_300 = np.loadtxt(pp.inputLoc+'relaxation_time_300K.dat', skiprows=1)
    taus_300 = dat_300[:, 4]  # fs
    enk_300 = dat_300[:, 3]  # eV

    fig, ax = plt.subplots()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = 'ADP Coefficient = %.1e ' % pA +r'$J^{1/2} \, s$'+ '\n' + r'$\tau_p/\tau_{\epsilon} = %.2f$' % ratio + \
              '\n' + r'Frohlich inverse RT = %.2f 1/ps' % inverseFrohlichTimeps + \
              '\n' + r'Optical Phonon Energy = %.2f meV' % opticalPhononEnergymeV
    ax.plot(g_df['energy [eV]'] - np.min(g_df['energy [eV]']), momRT*1e15, '-', label='Acoustic Deformation',color='dodgerblue')
    ax.plot(carrierEnergy[sortedInds] - np.min(carrierEnergy), 1/scatteringRate[sortedInds]*1e15, '-', label='Frohlich',color='tomato')
    ax.plot(enk_300 - np.min(enk_300), taus_300, '.', label='Perturbo',color='black')
    ax.text(0.57, 0.77, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)

    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel('Relaxation Times (fs)')
    plt.legend()
    # plt.savefig(pp.figureLoc +'davydovTimes.png', bbox_inches='tight',dpi=600)


def frohlich_davydovDistribution(df,ee,opticalPhononEnergy,inverseFrohlichTime,eRT_Ratio):
    carrierEnergy = (df['energy [eV]'] - pp.mu)*c.e
    sortedInds = np.argsort(carrierEnergy)
    scatteringRate = frohlichScattering(df, opticalPhononEnergy, inverseFrohlichTime)
    frohlichRTs = 1/scatteringRate/1e15
    integrand = -1 / (c.kb_joule * pp.T + 2 * c.e ** 2 * ee ** 2 *frohlichRTs[sortedInds]**2 * eRT_Ratio / (3 * 0.063 * 9.11e-31))
    davydovArgument = integrate.cumtrapz(integrand[sortedInds], carrierEnergy[sortedInds] * c.e, initial=0)
    normalizing = np.sum(df['k_FD'])/np.sum(np.exp(davydovArgument))

    return normalizing*np.exp(davydovArgument)


def plot_davydov_density_v_field(fieldVector, freq, df, pA, opticalPhononEnergy, inverseFrohlichTime,eRT_Ratio):
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # textstr = '\n'.join((r'$f = %.1f GHz \, \, (100) $' % (freq,), pp.fdmName))
    ratio = eRT_Ratio
    textstr = 'ADP Coefficient = %.1e ' % pA + r'$J^{1/2} \, s$'+ '\n' + r'$\tau_p/\tau_{\epsilon} = %.2f$' % ratio
    S_xx_vector = []
    S_yy_vector = []
    conductivity_xx_vector = []
    S_xx_acoustic_Davy = []
    S_xx_frohlich_Davy = []
    pertEnergy = []
    davyAcoEnergy = []
    davyFroEnergy = []
    carrierEnergy = (df['energy [eV]'] - np.min(df['energy [eV]'])) * c.e
    sortedInds = np.argsort(carrierEnergy)
    for ee in fieldVector:
        chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        S_xx, _, S_yy, _, conductivity_xx = psd_solver.density(chi, ee, df, freq, False, 0)
        S_xx_vector.append(S_xx)
        S_yy_vector.append(S_yy)
        davydovFro = frohlich_davydovDistribution(df, ee, opticalPhononEnergy, inverseFrohlichTime, eRT_Ratio)
        davydovAco = acoustic_davydovDistribution(df, ee, pA, eRT_Ratio)
        momRTAco, _ = acoustic_davydovRTs(df, pA, eRT_Ratio)
        momRTFro = 1/frohlichScattering(df,opticalPhononEnergy,inverseFrohlichTime)
        S_xx_acoustic_Davy.append(davydovNoise(electron_df,davydovAco, momRTAco))
        S_xx_frohlich_Davy.append(davydovNoise(electron_df,davydovFro, momRTFro))
        conductivity_xx_vector.append(conductivity_xx)
        pertEnergy.append(utilities.mean_energy(chi,df))
        davyAcoEnergy.append(np.sum(carrierEnergy*davydovAco)/np.sum(davydovAco))
        davyFroEnergy.append(np.sum(carrierEnergy[sortedInds]*davydovFro)/np.sum(davydovFro))

    kvcm = np.array(fieldVector) * 1e-5
    Nuc = pp.kgrid ** 3
    fig, ax = plt.subplots()
    ax.axhline(np.array(conductivity_xx_vector[0]) * 4 * c.kb_joule * pp.T / c.Vuc / Nuc, linestyle='--',
               label=r'$\frac{4 kT \sigma_{xx}^{eq}}{V_0 N_0}$')
    ax.plot(kvcm, S_xx_vector, label=r'$S^{xx}$')
    ax.plot(kvcm, np.array(S_xx_acoustic_Davy),color='red', label=r'$S_{l,DD}$')
    ax.plot(kvcm, np.array(S_xx_frohlich_Davy),color='red', label=r'$S_{l,DD}$')
    plt.legend()
    plt.xlabel('Field [kV/cm]')
    plt.ylabel('Spectral Density [A^2/m^4/Hz]')

    fig, ax = plt.subplots()
    ax.plot(kvcm, np.array(pertEnergy)-np.min(df['energy [eV]']), label=r'$S^{xx}$')
    ax.plot(kvcm, davyAcoEnergy,color='red', label=r'$S_{l,DD}$')
    ax.plot(kvcm, davyFroEnergy,color='red', label=r'$S_{l,DD}$')
    plt.legend()
    plt.xlabel('Field [kV/cm]')
    plt.ylabel('Mean energy above CBM (eV)')


if __name__ == '__main__':
    fields = pp.fieldVector
    freqs = pp.freqVector

    # preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)
    g_inds,_,_ = utilities.gaas_split_valleys(electron_df,False)

    g_df = electron_df.loc[g_inds]
    # pA = 4.60082e-22 # Prefactor for energy time dependence [see Lundstrom]
    pA = 4.60082e-23*0.83/0.99531*1.11929/1.02537796976  # Modified to give the same mobility
    eRT_Ratio = 0.02
    inverseFrohlichTime = 1e13*0.8  # s
    opticalPhononEnergy = 40/1000*c.e

    acoustic_davydovIntegrand(g_df.reset_index(drop=True), 4e4, pA, eRT_Ratio)
    plot_scattering(g_df.reset_index(drop=True),pA,eRT_Ratio,opticalPhononEnergy,inverseFrohlichTime)
    # plot_davydov_density_v_field(pp.moment_fields, 0.1, electron_df, pA, opticalPhononEnergy, inverseFrohlichTime, eRT_Ratio)
    plt.show()