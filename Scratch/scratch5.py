import numpy as np
import constants as c
import utilities
import problem_parameters as pp
import scipy.special as sc
from scipy import integrate
import matplotlib.pyplot as plt
import psd_solver
import matplotlib as mpl
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

# mpl.rcParams['figure.figsize'] = [7.5, 5.0]
# mpl.rcParams['figure.dpi'] = 200

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def davydovDistribution_acoustic(df,ee,pA,pB):
    carrierEnergy = (df['energy [eV]'] - pp.mu)*c.e
    hyperGeom = sc.hyp2f1(1, 2, 3, -3 * carrierEnergy * c.kb_joule * pp.T * c.m_gaas / (2 * pA * pB * c.e**2 * ee**2))
    prefactor = -3*c.m_gaas*carrierEnergy**2/(4*pA*pB*c.e**2*ee**2)
    davydov = np.exp(prefactor*hyperGeom)
    normalizing = np.sum(davydov)/np.sum(df['k_FD'])

    return davydov/normalizing


def acousticRTs(df,pA,pB):
    carrierEnergy = (df['energy [eV]'] - pp.mu) * c.e
    momRT = pA*(carrierEnergy)**(-0.5)
    eRT = pB*(carrierEnergy)**(-0.5)

    return momRT,eRT


def davydovMeanEnergy_acoustic(df,ee,pA,pB):
    carrierEnergy = (df['energy [eV]'] - pp.mu) * c.e
    davydovDist = davydovDistribution_acoustic(df, ee, pA, pB)
    np.sum(davydovDist*carrierEnergy)/np.sum(davydovDist)

    return np.sum(davydovDist*carrierEnergy)/np.sum(davydovDist)


def numericalDavydov(df,ee,momRT,eRT):
    carrierEnergy = (df['energy [eV]'].values - pp.mu)
    sortedInds = np.argsort(carrierEnergy)
    integrand = -1/(c.kb_joule*pp.T + 2*c.e**2*ee**2*momRT*eRT/(3*0.063*9.11e-31))
    davydovArgument = integrate.cumtrapz(integrand[sortedInds], carrierEnergy[sortedInds] * c.e, initial=0)
    normalizing = np.sum(df['k_FD'])/np.sum(np.exp(davydovArgument))

    return normalizing*np.exp(davydovArgument)


# def davydovMobility(df,davydovDist,ee,momRT,eRT):



def plot_numerical_analytic_davydov(df,ee,pA,pB):
    analyticDavy = davydovDistribution_acoustic(df, ee, pA, pB)
    momRT,eRT = acousticRTs(df,pA,pB)
    numericalDavy = numericalDavydov(df,ee,momRT,eRT)
    carrierEnergy = (df['energy [eV]'].values - pp.mu)
    sortedInds = np.argsort(carrierEnergy)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ratio = pA/pB
    Vcm = ee/100
    textstr = 'ADP Coefficient = %.1e ' % pA +r'$J^{1/2} \, s$'+ '\n' + r'$\tau_p/\tau_{\epsilon} = %.2f$' % ratio + \
              '\n' + r'Electric Field = %.2f V/cm' % Vcm
    fig, ax = plt.subplots()
    ax.plot(carrierEnergy[sortedInds]-np.min(carrierEnergy),analyticDavy[sortedInds],label='Analytical Acoustic Davydov')
    ax.plot(carrierEnergy[sortedInds]-np.min(carrierEnergy),numericalDavy,'-.',label='Numerical Acoustic Davydov')
    ax.text(0.60, 0.8, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel('Distribution Fcn. (unitless)')
    plt.legend()


def frohlichRT(df,opticalPhononEn,FrohlichTime,Temp):
    parEnergy = c.hbar_joule ** 2 / (2 * 0.063 * 9.11e-31) * (
                (df['kx [1/A]'] * 10 ** 10) ** 2 + (df['ky [1/A]'] * 10 ** 10) ** 2 + (
                    df['kz [1/A]'] * 10 ** 10) ** 2) + opticalPhononEn/100
    # I have added a small offset to the parabolic energy so that there is no problem at energies ~ 0
    boseEinstein = 1/(np.exp(opticalPhononEn/(c.kb_joule*Temp))-1)
    # boseEinstein = 1 / (np.exp(opticalPhononEn / (c.kb_joule * 77)) - 1)
    ephRatio = np.abs(parEnergy/opticalPhononEn)
    scatteringRate = 1/np.sqrt(ephRatio)*(boseEinstein*np.sqrt(np.arcsinh(ephRatio))+(boseEinstein+1)*np.real(np.sqrt(np.arcsinh(ephRatio-1)+0j)))
    # The second arcsinh becomes negative if the electron energies are less than the phonon energy, so I only take the real part of the sqrt.

    return 1/scatteringRate*FrohlichTime


def plot_davydov(df,ee,pA,pB,opticalPhononEn,FrohlichTime):
    acousticDavydov = davydovDistribution_acoustic(df, ee, pA, pB)
    ratio = pA/pB
    fRT = frohlichRT(df, opticalPhononEn, FrohlichTime,pp.T)
    frohlichDavydov = numericalDavydov(df, ee, fRT, fRT/ratio)
    carrierEnergy = (df['energy [eV]'].values - pp.mu)
    sortedInds = np.argsort(carrierEnergy)

    # Load BTE calculated solution
    chi_3 = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
    f = chi_3 + electron_df['k_FD']
    g_inds,_,_ = utilities.gaas_split_valleys(electron_df,False)

    # Distribution plot
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ratio = pA/pB
    FrohlichTimeps = FrohlichTime*1e15
    opticalPhononEnergymeV = opticalPhononEnergy*1000/c.e

    Vcm = ee/100
    textstr = 'ADP Coefficient = %.1e ' % pA +r'$J^{1/2} \, s$'+ '\n' + r'$\tau_p/\tau_{\epsilon} = %.2f$' % ratio + \
              '\n' + r'Frohlich RT = %.2f fs' % FrohlichTimeps + \
              '\n' + r'Optical Phonon Energy = %.2f meV' % opticalPhononEnergymeV + \
              '\n' + r'Electric Field = %.2f V/cm' % Vcm
    fig, ax = plt.subplots()
    ax.plot(carrierEnergy[sortedInds]-np.min(carrierEnergy),acousticDavydov[sortedInds],label='Davy.+ ADP')
    ax.plot(carrierEnergy[sortedInds]-np.min(carrierEnergy),frohlichDavydov,'-.',label='Davy.+ Frohlich')
    ax.plot(carrierEnergy-np.min(carrierEnergy),f[g_inds],'.',color='black',label='BTE + Pert.')

    ax.text(0.60, 0.75, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel('Distribution fcn. (unitless)')
    plt.legend()


def plot_RTs(df,pA,pB,opticalPhononEn,FrohlichTime):
    # Get the Frohlich and Acousic Deformation Potential RTs
    carrierEnergy = (df['energy [eV]'].values - pp.mu)
    momRT,eRT = acousticRTs(df,pA,pB)
    frRT = frohlichRT(df,opticalPhononEn,FrohlichTime, pp.T)
    sortedInds = np.argsort(carrierEnergy)

    # Get the Perturbo RTs
    nkpts = len(electron_df)
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    rates = (-1) * np.diag(scm)
    g_inds, _, _ = utilities.gaas_split_valleys(electron_df, False)

    # Relaxation time plot
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ratio = pA/pB
    FrohlichTimeps = FrohlichTime*1e15
    opticalPhononEnergymeV = opticalPhononEnergy*1000/c.e
    textstr = 'ADP Coefficient = %.1e ' % pA +r'$J^{1/2} \, s$'+ '\n' + r'$\tau_p/\tau_{\epsilon} = %.2f$' % ratio + \
              '\n' + r'Frohlich RT = %.2f fs' % FrohlichTimeps + \
              '\n' + r'Optical Phonon Energy = %.2f meV' % opticalPhononEnergymeV
    fig, ax = plt.subplots()
    ax.plot(carrierEnergy[sortedInds]-np.min(carrierEnergy),momRT[sortedInds]*1e15,label='Acoustic Deformation Potential')
    ax.plot(carrierEnergy[sortedInds]-np.min(carrierEnergy),frRT[sortedInds]*1e15,'-.',label='Fr'+r'$\"o$'+'hlich')
    ax.plot(df['energy [eV]'] - np.min(df['energy [eV]']), 1/rates[g_inds]*1e15, '.', label='Perturbo',color='black')
    ax.text(0.57, 0.76, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel('Relaxation Times (fs)')
    plt.legend()
    plt.savefig(pp.outputLoc + 'Paper_Figures/' + 'RT_comparison.png', bbox_inches='tight', dpi=600)


    # Scattering Rate Plot
    ratio = pA/pB
    FrohlichTimeps = FrohlichTime*1e15
    opticalPhononEnergymeV = opticalPhononEnergy*1000/c.e
    textstr = 'ADP Coefficient = %.1e ' % pA +r'$J^{1/2} \, s$'+ '\n' + r'$\tau_p/\tau_{\epsilon} = %.2f$' % ratio + \
              '\n' + r'Frohlich RT = %.2f fs' % FrohlichTimeps + \
              '\n' + r'Optical Phonon Energy = %.2f meV' % opticalPhononEnergymeV
    fig, ax = plt.subplots()
    ax.plot(carrierEnergy[sortedInds]-np.min(carrierEnergy),1/momRT[sortedInds]*1e-12,label='ADP')
    ax.plot(carrierEnergy[sortedInds]-np.min(carrierEnergy),1/frRT[sortedInds]*1e-12,'-.',label='Frohlich')
    ax.plot(df['energy [eV]'] - np.min(df['energy [eV]']), rates[g_inds]*1e-12, '.', label='Perturbo',color='black')
    ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel('Scattering Rate '+r'$(ps^{-1})$')
    plt.legend()

    frRT_300K = frohlichRT(df,opticalPhononEn,FrohlichTime, 300)
    frRT_77K = frohlichRT(df,opticalPhononEn,FrohlichTime, 77)
    frRT_4K = frohlichRT(df,opticalPhononEn,FrohlichTime, 4)

    # Temperature Dependent Frohlich
    fig, ax = plt.subplots()
    ax.plot(carrierEnergy[sortedInds]-np.min(carrierEnergy),frRT_300K[sortedInds]*1e15,'.',label='Davydov + Frohlich 300 K')
    ax.plot(carrierEnergy[sortedInds]-np.min(carrierEnergy),frRT_77K[sortedInds]*1e15,'.',label='Davydov + Frohlich 250 K')
    # ax.plot(carrierEnergy[sortedInds]-np.min(carrierEnergy),frRT_4K[sortedInds]*1e15,'-.',label='Davy.+ Frohlich 4 K')

    ax.text(0.55, 0.65, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel('Relaxation Times (fs)')
    plt.yscale('Log')
    plt.legend()


def davydovNoise(df,davydovDist,momRT):
    # Taken from GGK 3.76
    Nuc = pp.kgrid ** 3
    carrierEnergy = (df['energy [eV]'] - pp.mu)*c.e
    n = utilities.calculate_density(df)
    davyNoise = 4/3*c.e**2*n/(c.m_gaas*Nuc*c.Vuc)*np.sum(davydovDist*momRT*carrierEnergy**(1.5))/np.sum(carrierEnergy**(0.5)*davydovDist)

    return davyNoise


def plotDavyNoise(df,fieldVector,pA,pB,opticalPhononEn,FrohlichTime):
    ratio = pA/pB
    fRT = frohlichRT(df, opticalPhononEn, FrohlichTime, pp.T)
    adpRT,_ = acousticRTs(df,pA,pB)
    carrierEnergy = (df['energy [eV]'].values - pp.mu)
    sortedInds = np.argsort(carrierEnergy)
    g_inds,_,_ = utilities.gaas_split_valleys(electron_df,False)
    Nuc = pp.kgrid ** 3

    conductivity_xx_vector = []

    adpNoise = []
    froNoise = []
    S_xx_vector = []

    adpMeanE = []
    froMeanE = []
    bteMeanE = []

    adpMeanRT = []
    froMeanRT = []
    bteMeanRT = []

    # Get the Perturbo RTs
    nkpts = len(electron_df)
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    rates = (-1) * np.diag(scm)
    g_inds, _, _ = utilities.gaas_split_valleys(electron_df, False)

    # Calculate distributions and noise as a function of electric field
    for ee in fieldVector:
        # Store noise in list for ADP, Frohlich, and Perturbo
        acousticDavydov = davydovDistribution_acoustic(df, ee, pA, pB)
        frohlichDavydov = numericalDavydov(df, ee, fRT, fRT/ratio)
        adpNoise.append(davydovNoise(df, acousticDavydov, adpRT))
        froNoise.append(davydovNoise(df, frohlichDavydov, fRT[sortedInds]))
        chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        f = chi+electron_df['k_FD']
        S_xx, S_xx_RTA, S_yy, S_yy_RTA, conductivity_xx = psd_solver.density(chi, ee, electron_df, 0.1, False, 0)
        S_xx_vector.append(S_xx)
        conductivity_xx_vector.append(conductivity_xx)

        # Store average energy in list for ADP, Frohlich, and Perturbo
        adpMeanE.append(np.sum(acousticDavydov*df['energy [eV]'])/np.sum(acousticDavydov))
        froMeanE.append(np.sum(frohlichDavydov*df.loc[sortedInds,'energy [eV]'])/np.sum(frohlichDavydov))
        bteMeanE.append(utilities.mean_energy(chi,electron_df))

        # Store average RT in list for ADP, Frohlich, and Perturbo
        adpMeanRT.append(np.sum(acousticDavydov*adpRT)/np.sum(acousticDavydov))
        froMeanRT.append(np.sum(frohlichDavydov*fRT[sortedInds])/np.sum(frohlichDavydov))
        bteMeanRT.append(np.sum(f*1/rates)/np.sum(frohlichDavydov))

    # Noise plot
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ratio = pA/pB
    FrohlichTimeps = FrohlichTime*1e15
    opticalPhononEnergymeV = opticalPhononEnergy*1000/c.e

    textstr = 'ADP Coefficient = %.1e ' % pA +r'$J^{1/2} \, s$'+ '\n' + r'$\tau_p/\tau_{\epsilon} = %.2f$' % ratio + \
              '\n' + r'Frohlich RT = %.2f fs' % FrohlichTimeps + \
              '\n' + r'Optical Phonon Energy = %.2f meV' % opticalPhononEnergymeV

    Vcm = np.array(fieldVector)/100
    fig, ax = plt.subplots()
    # ax.axhline(np.array(conductivity_xx_vector[0]) * 4 * c.kb_joule * pp.T / c.Vuc / Nuc, linestyle='--',
    #            label=r'Nyquist-Johnson',color='black')
    # ax.plot(Vcm,adpNoise,label='Davydov + ADP')
    ax.plot(Vcm,froNoise,'-.',label='Davydov + Frohlich')
    # ax.plot(Vcm,S_xx_vector,'-.',label='Full Drift + Perturbo',color='black')
    ax.text(0.57, 0.96, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.ylabel('Spectral Density' + r'$(A^2 \, m^{-4} \, Hz^{-1})$')
    plt.xlabel('Electric Field '+r'$(V \, cm^{-1})$')
    plt.legend()
    plt.savefig(pp.outputLoc + 'Paper_Figures/' + 'EFieldDepNoiseCompare.png', bbox_inches='tight', dpi=600)


    adpMeanE = np.array(adpMeanE)
    froMeanE = np.array(froMeanE)
    bteMeanE = np.array(bteMeanE)


    adpMeanRT = np.array(adpMeanRT)
    froMeanRT = np.array(froMeanRT)
    bteMeanRT = np.array(bteMeanRT)

    # Energy plot
    fig, ax = plt.subplots()
    ax.plot(Vcm,(adpMeanE-np.min(df['energy [eV]']))/c.kb_ev/pp.T,label='Davydov + ADP')
    ax.plot(Vcm,(froMeanE-np.min(df['energy [eV]']))/c.kb_ev/pp.T,'-.',label='Davydov + Frohlich')
    ax.plot(Vcm,(bteMeanE-np.min(df['energy [eV]']))/c.kb_ev/pp.T,'-.',label='Full Drift + Perturbo',color='black')
    ax.text(0.57, 0.96, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.ylabel('Mean Energy above CBM (kT)')
    plt.xlabel('Electric Field '+r'$(V \, cm^{-1})$')
    plt.legend()
    plt.savefig(pp.outputLoc + 'Paper_Figures/' + 'MeanEnergyAboveCBM.png', bbox_inches='tight', dpi=600)

    # Average scattering rate
    fig, ax = plt.subplots()
    ax.plot(Vcm,1/(adpMeanRT/adpMeanRT[0]*1e-15),label='Davy.+ ADP')
    ax.plot(Vcm,1/(froMeanRT/froMeanRT[0]*1e-15),'-.',label='Davy.+ Frohlich')
    ax.plot(Vcm,1/(bteMeanRT/bteMeanRT[0]*1e-15),'-.',label='BTE + Perturbo',color='black')
    ax.text(0.62, 0.96, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.ylabel('MMSR (norm. to eq. value)')
    plt.xlabel('Electric Field '+r'$(V \, cm^{-1})$')
    plt.legend()



if __name__ == '__main__':
    fields = pp.fieldVector
    freqs = pp.freqVector
    small_signal_fields = np.array([1e-3,1e4,4e4])

    # preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)
    g_inds,_,_ = utilities.gaas_split_valleys(electron_df,False)

    g_df = electron_df.loc[g_inds]
    g_df = g_df.reset_index(drop=True)
    pA = 4.60082e-23*0.83/0.99531*1.11929*0.97989949748  # Modified to give the same mobility
    pB = 50*pA  # Modified to give the same mobility
    # pB = 10*pA
    # pB = 0.1*pA
    inverseFrohlichTime = 1.02576620559*1e13*0.9951417004  # s
    opticalPhononEnergy = 40/1000*c.e

    plot_numerical_analytic_davydov(g_df, 4e4, pA, pB)
    plot_RTs(g_df, pA, pB, opticalPhononEnergy, 1/inverseFrohlichTime)
    plot_davydov(g_df, 1e-3, pA, pB, opticalPhononEnergy, 1/inverseFrohlichTime)
    denseFields = np.geomspace(1e-5,4e4,1000)
    # plotDavyNoise(g_df, denseFields, pA, pB, opticalPhononEnergy, 1/inverseFrohlichTime)

    plotDavyNoise(g_df, small_signal_fields, pA, pB, opticalPhononEnergy, 1/inverseFrohlichTime)
    plt.show()