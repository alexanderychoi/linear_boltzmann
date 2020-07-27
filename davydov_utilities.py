import numpy as np
import constants as c
import utilities
import problem_parameters as pp
import scipy.special as sc
from scipy import integrate
import matplotlib.pyplot as plt


def acoustic_davydovDistribution(df,ee,pA,pB):
    carrierEnergy = (df['energy [eV]'] - pp.mu)*c.e
    hyperGeom = sc.hyp2f1(1, 2, 3, -3 * carrierEnergy * c.kb_joule * pp.T * c.m_gaas / (2 * pA * pB * c.e**2 * ee**2))
    prefactor = -3*c.m_gaas*carrierEnergy**2/(4*pA*pB*c.e**2*ee**2)
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


def acoustic_davydovRTs(df,pA,pB):
    carrierEnergy = (df['energy [eV]'] - pp.mu) * c.e
    momRT = pA*(carrierEnergy)**(-0.5)
    eRT = pB*(carrierEnergy)**(-0.5)

    return momRT,eRT


def acoustic_davydovPartialEnergy(df,ee,pA,pB):
    carrierEnergy = (df['energy [eV]'] - pp.mu) * c.e
    davydovDist = acoustic_davydovDistribution(df, ee, pA, pB)
    hypergeom_a = sc.hyp2f1(1, 2, 3, -3 * carrierEnergy * c.kb_joule * pp.T * c.m_gaas / (2 * pA * pB * c.e**2 * ee**2))
    hypergeom_b = sc.hyp2f1(2, 3, 4, -3 * carrierEnergy * c.kb_joule * pp.T * c.m_gaas / (2 * pA * pB * c.e**2 * ee**2))

    term_a = -3*carrierEnergy*c.m_gaas/(2*pA*pB*c.e**2*ee**2)*hypergeom_a
    term_b = -carrierEnergy**2*c.m_gaas/(2*pA*pB*c.e**2*carrierEnergy**2)*hypergeom_b

    return davydovDist*(term_a+term_b)


def acoustic_davydovMeanEnergy(df,ee,pA,pB):
    carrierEnergy = (df['energy [eV]'] - pp.mu) * c.e
    davydovDist = acoustic_davydovDistribution(df, ee, pA, pB)
    np.sum(davydovDist*carrierEnergy)/np.sum(davydovDist)

    return np.sum(davydovDist*carrierEnergy)/np.sum(davydovDist)


def frohlichScattering(df,opticalPhononEn,inverseFrohlichTime):
    # opticalPhononEn = 40/1000*c.e #  Optical phonon energy at BZ center, approximated as 40 meV
    parEnergy = c.hbar_joule ** 2 / (2 * 0.063 * 9.11e-31) * (
                (df['kx [1/A]'] * 10 ** 10) ** 2 + (df['ky [1/A]'] * 10 ** 10) ** 2 + (
                    df['kz [1/A]'] * 10 ** 10) ** 2)+ opticalPhononEn/100

    boseEinstein = 1/(np.exp(opticalPhononEn/(c.kb_joule*pp.T))-1)
    ephRatio = np.abs(parEnergy/opticalPhononEn)
    scatteringRate = 1/np.sqrt(ephRatio)*(boseEinstein*np.sqrt(np.arcsinh(ephRatio))+(boseEinstein+1)*np.sqrt(np.arcsinh(ephRatio-1)+0j))

    return scatteringRate*inverseFrohlichTime


def acoustic_davydovIntegrand(df,ee,pA,pB):
    carrierEnergy = (df['energy [eV]'].values - pp.mu)
    sortedInds = np.argsort(carrierEnergy)

    plt.figure()
    plt.plot(np.arange(len(sortedInds+1)),carrierEnergy[sortedInds],'.')

    momRT,eRT = acoustic_davydovRTs(df, pA, pB)
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
    plt.plot(carrierEnergy-np.min(carrierEnergy),acoustic_davydovDistribution(df,ee,pA,pB),'.',label='Analytical Integration')
    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel('Acoustic Davydov Distribution (unitless)')
    plt.legend()


def plot_frohlich_scattering(df,pA,pB,opticalPhononEnergy, inverseFrohlichTime):
    momRT,eRT = acoustic_davydovRTs(df, pA, pB)
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
    ratio = pA/pB
    fig, ax = plt.subplots()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = 'ADP Coefficient = %.1e ' % pA +r'$J^{1/2} \, s$'+ '\n' + r'$\tau_p/\tau_{\epsilon} = %.2f$' % ratio + \
              '\n' + r'Frohlich inverse RT = %.2f 1/ps' % inverseFrohlichTimeps + \
              '\n' + r'Optical Phonon Energy = %.2f meV' % opticalPhononEnergymeV

    ax.plot(g_df['energy [eV]'] - np.min(g_df['energy [eV]']), rates[g_inds], '.', label='Perturbo')
    ax.plot(carrierEnergy[sortedInds] - np.min(carrierEnergy), scatteringRate[sortedInds], '.', label='Frohlich')
    ax.plot(g_df['energy [eV]'] - np.min(g_df['energy [eV]']), 1/momRT, '.', label='Acoustic Deformation')
    ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)

    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel('Scattering Rate '+r'$(fs^{-1})$')
    plt.legend()


def frohlich_davydovDistribution(df,ee,opticalPhononEnergy,inverseFrohlichTime,eRT_Ratio):
    carrierEnergy = (df['energy [eV]'] - pp.mu)*c.e
    sortedInds = np.argsort(carrierEnergy)
    scatteringRate = frohlichScattering(df, opticalPhononEnergy, inverseFrohlichTime)
    frohlichRTs = 1/scatteringRate/1e15
    integrand = -1 / (c.kb_joule * pp.T + 2 * c.e ** 2 * ee ** 2 *frohlichRTs[sortedInds]**2 * eRT_Ratio / (3 * 0.063 * 9.11e-31))
    davydovArgument = integrate.cumtrapz(integrand[sortedInds], carrierEnergy[sortedInds] * c.e, initial=0)
    normalizing = np.sum(df['k_FD'])/np.sum(np.exp(davydovArgument))

    return normalizing*np.exp(davydovArgument)


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
    pB = 10*pA  # Modified to give the same mobility
    inverseFrohlichTime = 1e13*0.8  # s
    opticalPhononEnergy = 80/1000*c.e

    acoustic_davydovIntegrand(g_df.reset_index(drop=True), 4e4, pA, pB)
    plot_frohlich_scattering(g_df.reset_index(drop=True),pA,pB,opticalPhononEnergy,inverseFrohlichTime)

    fr_davydov = frohlich_davydovDistribution(g_df.reset_index(drop=True), 4e4, opticalPhononEnergy, inverseFrohlichTime, 10)

    plt.figure()
    plt.plot(fr_davydov,'.')
    plt.show()