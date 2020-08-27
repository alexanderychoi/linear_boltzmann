import numpy as np
import constants as c
import utilities
import problem_parameters as pp
import scipy.special as sc
from scipy import integrate
import matplotlib.pyplot as plt
import psd_solver
import davydov_utilities

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


def frohlichScattering_temperature(df,opticalPhononEn,inverseFrohlichTime,Temp):
    parEnergy = c.hbar_joule ** 2 / (2 * c.m_gaas) * (
                (df['kx [1/A]'] * 10 ** 10) ** 2 + (df['ky [1/A]'] * 10 ** 10) ** 2 + (
                    df['kz [1/A]'] * 10 ** 10) ** 2)+ opticalPhononEn/100

    boseEinstein = 1/(np.exp(opticalPhononEn/(c.kb_joule*Temp))-1)
    ephRatio = np.abs(parEnergy/opticalPhononEn)
    abs = 1/np.sqrt(ephRatio)*boseEinstein*np.sqrt(np.arcsinh(ephRatio))
    ems = 1/np.sqrt(ephRatio)*(boseEinstein+1)*np.sqrt(np.arcsinh(ephRatio-1)+0j)
    scatteringRate = abs+ems

    return np.real(abs)*inverseFrohlichTime,np.real(ems)*inverseFrohlichTime



def frodfDistribution(df,ee,eRT_Ratio):
    carrierEnergy = (df['energy [eV]'] - pp.mu)
    frohlichRTs = df['froRT']
    integrand = -1 / (c.kb_joule * pp.T + 2 * c.e ** 2 * ee ** 2 *frohlichRTs**2 * eRT_Ratio / (3 * c.m_gaas))
    davydovArgument = integrate.cumtrapz(integrand, carrierEnergy * c.e, initial=0)
    normalizing = np.sum(df['k_FD'])/np.sum(np.exp(davydovArgument))

    return normalizing*np.exp(davydovArgument)

def froTheta(df,eRT_Ratio):

    return 2*c.e**2/(3*c.m_gaas*c.kb_joule*pp.T)*eRT_Ratio*df['froRT']**2


def froAvRT(df,ee,eRT_Ratio):
    carrierEnergy = (df['energy [eV]'] - pp.mu)
    theta = froTheta(df,eRT_Ratio)
    froDist = frodfDistribution(df,ee,eRT_Ratio)
    dF = -froDist/(c.kb_joule*pp.T*(1+ee**2*theta))

    num_integrand = df['froRT']*(carrierEnergy*c.e)**(3/2)*dF
    den_integrand = (carrierEnergy*c.e)**(1/2)*froDist

    num = integrate.trapz(num_integrand,carrierEnergy*c.e)
    den = integrate.trapz(den_integrand,carrierEnergy*c.e)

    return num/den*(-2/3)


def froKappa(df,ee,eRT_Ratio):
    carrierEnergy = (df['energy [eV]'] - pp.mu)
    froDist = frodfDistribution(df,ee,eRT_Ratio)
    gradF = np.gradient(froDist,carrierEnergy*c.e)
    integrand = df['froRT']*(carrierEnergy*c.e)**(3/2)*gradF + 3/2*froAvRT(df,ee,eRT_Ratio)*(carrierEnergy*c.e)**(1/2)*froDist

    kappa = np.zeros(len(df))
    for i in range(len(df)):
        ext = len(integrand+1)
        kappa[i] = -np.trapz(integrand[0:ext-i],carrierEnergy[0:ext-i]*c.e)
    return kappa


def diffusion_ratio(df,ee,eRT_Ratio):
    carrierEnergy = (df['energy [eV]'] - pp.mu)
    theta = froTheta(df,eRT_Ratio)
    kappa = froKappa(df,ee,eRT_Ratio)
    froDist = frodfDistribution(df, ee, eRT_Ratio)
    num1_integrand = theta*(0**2-(df['froRT']*(carrierEnergy*c.e)**(3/2)*froDist)**2)
    num2_integrand = (1+ee*2*theta)*df['froRT']*(carrierEnergy*c.e)**(3/2)*froDist
    den_integrand = df['froRT']*(carrierEnergy*c.e)**(3/2)*froDist

    num = np.trapz(num1_integrand/num2_integrand,carrierEnergy*c.e)
    den = np.trapz(den_integrand,carrierEnergy*c.e)

    return 1 + (ee/4)**2*num/den

if __name__ == '__main__':
    fields = pp.fieldVector
    freqs = pp.freqVector

    # preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)
    g_inds,_,_ = utilities.gaas_split_valleys(electron_df,False)

    g_df = electron_df.loc[g_inds]
    # pA = 4.60082e-22 # Prefactor for energy time dependence [see Lundstrom]
    pA = 4.60082e-23*0.83/0.99531*1.11929/1.02537796976/0.9954  # Modified to give the same mobility
    eRT_Ratio = 60
    inverseFrohlichTime = 1e13*1.05/1.0291504*1.00094492441*0.755939*1.010259  # Hz This is the value that gives matching PSD

    # inverseFrohlichTime = 7.567429*10**12 # This is the value predicted by the band parameters
    opticalPhononEnergy = 35/1000*c.e

    fro_df_small = g_df.drop_duplicates(subset='energy [eV]')
    fro_df_small = fro_df_small.sort_values('energy [eV]',ascending=True)
    fro_df_small = fro_df_small.reset_index(drop=True)
    fro_df_small['froRT'] = 1/davydov_utilities.frohlichScattering(fro_df_small,opticalPhononEnergy,inverseFrohlichTime)
    this = froAvRT(fro_df_small, 4e4, eRT_Ratio)

    new_moment_fields = np.geomspace(1e-2,1e6,30)


    avRT = []
    for ee in pp.moment_fields:
        avRT.append(froAvRT(fro_df_small, ee, eRT_Ratio))
    Vcm = pp.moment_fields/100
    plt.figure()
    plt.plot(Vcm,np.array(avRT)*1e15)
    plt.xlabel('Field (V/cm)')
    plt.ylabel('Average Davydov RT (fs)')

    plt.figure()
    for ee in new_moment_fields:
        plt.plot(fro_df_small['energy [eV]'],frodfDistribution(fro_df_small,ee,eRT_Ratio))
    plt.xlabel('Energy (eV)')
    plt.ylabel('Frohlich Distribution')

    froDist = frodfDistribution(fro_df_small,4e4,eRT_Ratio)
    carrierEnergy = (fro_df_small['energy [eV]'] - pp.mu)
    plt.figure()
    plt.plot(carrierEnergy,np.gradient(froDist,carrierEnergy))

    kappa = froKappa(fro_df_small, 2e5, eRT_Ratio)
    plt.figure()
    plt.plot(carrierEnergy,kappa)

    ratio_1 = []
    ratio_2 = []
    ratio_3 = []
    eRT_Ratio_1 = 1
    eRT_Ratio_2 = 60
    eRT_Ratio_3 = 120
    for ee in pp.moment_fields:
        ratio_1.append(diffusion_ratio(fro_df_small, ee, eRT_Ratio_1))
        ratio_2.append(diffusion_ratio(fro_df_small, ee, eRT_Ratio_2))
        ratio_3.append(diffusion_ratio(fro_df_small, ee, eRT_Ratio_3))

    Vcm = pp.moment_fields/100
    plt.figure()
    plt.plot(Vcm,ratio_1,label= r'$\tau_p/ \tau_{{\epsilon}} = {:.3f}$'.format(1/eRT_Ratio_1))
    plt.plot(Vcm,ratio_2,label= r'$\tau_p/ \tau_{{\epsilon}} = {:.3f}$'.format(1/eRT_Ratio_2))
    plt.plot(Vcm,ratio_3,label= r'$\tau_p/ \tau_{{\epsilon}} = {:.3f}$'.format(1/eRT_Ratio_3))

    plt.xlabel('Field (V/cm)')
    plt.ylabel('Frohlich Diffusion Ratio')
    plt.legend()


    fig, ax = plt.subplots()
    froRT_77K_abs,froRT_77K_ems  = frohlichScattering_temperature(fro_df_small,opticalPhononEnergy,inverseFrohlichTime,77)
    froRT_200K_abs,froRT_200K_ems = frohlichScattering_temperature(fro_df_small,opticalPhononEnergy,inverseFrohlichTime,200)
    froRT_300K_abs,froRT_300K_ems = frohlichScattering_temperature(fro_df_small,opticalPhononEnergy,inverseFrohlichTime,300)
    froRT_500K_abs,froRT_500K_ems = frohlichScattering_temperature(fro_df_small,opticalPhononEnergy,inverseFrohlichTime,500)

    ax.plot(fro_df_small['energy [eV]'],froRT_77K_abs/1e12,'-.',label='77 K ABS',color='dodgerblue')
    ax.plot(fro_df_small['energy [eV]'],froRT_77K_ems/1e12,'--',label='77 K EMS',color='dodgerblue')
    ax.plot(fro_df_small['energy [eV]'],froRT_300K_abs/1e12,'-.',label='300 K ABS',color='coral')
    ax.plot(fro_df_small['energy [eV]'],froRT_300K_ems/1e12,'--',label='300 K EMS',color='coral')
    ax.plot(fro_df_small['energy [eV]'],froRT_500K_abs/1e12,'-.',label='500 K ABS',color='firebrick')
    ax.plot(fro_df_small['energy [eV]'],froRT_500K_ems/1e12,'--',label='500 K EMS',color='firebrick')
    #
    # ax.plot(fro_df_small['energy [eV]'],(froRT_77K_ems+froRT_77K_abs)/1e12,'-',label='77 K',color='dodgerblue')
    # ax.plot(fro_df_small['energy [eV]'],(froRT_300K_ems+ froRT_300K_abs)/1e12,'-',label='300 K',color='coral')
    # ax.plot(fro_df_small['energy [eV]'],(froRT_500K_ems+ froRT_500K_abs)/1e12,'-',label='500 K',color='firebrick')

    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel('Scattering Rate '+r'$(ps^{-1})$')
    plt.legend(ncol=3)
    plt.show()