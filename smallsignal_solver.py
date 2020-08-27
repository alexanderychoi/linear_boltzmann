import numpy as np
import constants as c
import utilities
import problem_parameters as pp
import preprocessing
import occupation_solver
import matplotlib.pyplot as plt
import psd_solver

# PURPOSE: THIS MODULE CONTAINS FUNCTIONS USED TO CALCULATE AND PLOT THE SMALL SIGNAL AC CONDUCTIVITY. RIGHT NOW THIS IS
# HARDCODED FOR GAAS WITH THE FIELD IN THE [1 0 0], BUT CAN BE EASILY GENERALIZED.

# ORDER: THIS MODULE REQUIRES THAT STEADY AND TRANSIENT SOLUTIONS HAVE ALREADY BEEN WRITTEN TO FILE BY
# OCCUPATION_SOLVER.PY.

# OUTPUT: THIS MODULE SAVES THE SMALL SIGNAL CONDUCTIVITY INDEXED BY FREQUENCY AND FIELD TO THE SUBPROBLEM OUTPUT. IT CAN
# PRODUCE PLOTS OF THE SMALL SIGNAL CONDUCTIVITY FOR EXPLORATORY DATA ANALYSIS BUT IT DOES NOT SAVE THOSE PLOTS.


def longitudinal_small_signal_conductivity(df, fieldVector, freqVector):
    """Calculate small signal AC conductivity using Hartnagel's book, Eqn. 2.32.
    Parameters:
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        fieldVector (nparray): Containing values of the electric field in V/m
        freqVector (nparray) : Containing the frequencies in GHz
    Returns:
        Nothing. Saves the value of the small signal conductivity indexed by frequency and field.
    """
    Nuc = pp.kgrid ** 3
    nkpts = len(df)
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    df['df0/dkx'] = -c.hbar_joule/c.kb_joule/pp.T*df['vx [m/s]']*df['k_FD']*(1-df['k_FD'])
    for freq in freqVector:
        for ee in fieldVector:
            print('Starting calculation of small signal AC conductivity at {:.1e} GHz and E = {:.1e} kV/cm'.format(freq,ee/1e5))
            if ee != 0:
                fdm = np.memmap(pp.inputLoc + '/finite_difference_matrix.mmap', dtype='float64', mode='w+',
                                shape=(nkpts, nkpts))
                _, _, _, fdm = occupation_solver.gaas_gamma_fdm(fdm, df, ee)
                if pp.derL:
                    _, _, _, fdm = occupation_solver.gaas_l_fdm(fdm, df, ee)
                chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
                b = -np.dot(fdm, chi_3_i) / (c.e * ee) * c.hbar_joule
                del fdm
                if freq == 0:
                    chi_3t_i = chi_3_i
                else:
                    chi_3t_i = np.load(
                        pp.outputLoc + 'Transient/' + 'chi_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, ee))
                fdm2 = np.memmap(pp.inputLoc + '/finite_difference_matrix2.mmap', dtype='float64', mode='w+',
                                 shape=(nkpts, nkpts))
                bp, _, _, _ = occupation_solver.gaas_inverse_relaxation_operator(b, scm, fdm2, df, ee, freq)
                del fdm2
                bp2 = bp / (c.hbar_joule) * df['vx [m/s]']
                prefactor = 2 * c.e / c.Vuc / Nuc / ee
                linear_conductivity = prefactor * np.sum(df['vx [m/s]'] * chi_3t_i)
                ac_conductivity = linear_conductivity - 2 * c.e ** 2 / c.Vuc / Nuc * np.sum(bp2)
                np.save(pp.outputLoc + 'Small_Signal/' + 'cond_3_' + "f_{:.1e}_E_{:.1e}".format(freq, ee),
                        ac_conductivity)  # This is the total, composed of eq and noneq derivative contributions
                np.save(pp.outputLoc + 'Small_Signal/' + 'linear_cond_3_' + "f_{:.1e}_E_{:.1e}".format(freq, ee),
                        linear_conductivity)  # This is the portion which is proportional to the equilibrium derivative.
                                              # It makes sense to split these up because we know the derivative of the eq.
                                              # tern analytically.
                np.save(pp.outputLoc + 'Small_Signal/'+'decomp_cond_3_'+ "f_{:.1e}_E_{:.1e}".format(freq, ee),df['vx [m/s]']*(chi_3t_i/ee - c.e/c.hbar_joule*bp))

            if ee == 0:  # This is a special case, see Overleaf. Chi = 0 obviously, so have to calculate differently.
                print('Field is zero. Calculating accordingly.')
                fdm2 = np.memmap(pp.inputLoc + '/finite_difference_matrix2.mmap', dtype='float64', mode='w+',
                                 shape=(nkpts, nkpts))
                b = (-1) * c.hbar_joule / c.kb_joule / pp.T * np.squeeze(df['vx [m/s]'] * df['k_FD']) * (1 - df['k_FD'])
                bp, _, _, _ = occupation_solver.gaas_inverse_relaxation_operator(b, scm, fdm2, df, ee, freq)
                bp2 = bp / (c.hbar_joule) * df['vx [m/s]']
                prefactor = -2*c.e**2/c.Vuc/Nuc
                ac_conductivity = prefactor*np.sum(bp2)
                np.save(pp.outputLoc + 'Small_Signal/' + 'cond_3_' + "f_{:.1e}_E_{:.1e}".format(freq, ee),
                        ac_conductivity)
                del fdm2


def plot_small_signal_conductivity(fieldVector,freqVector,df):
    kvcm = np.array(fieldVector)*1e-5
    fig, ax = plt.subplots()
    for freq in freqVector:
        cond = []
        for ee in fieldVector:
            cond.append(np.load(pp.outputLoc + 'Small_Signal/' + 'cond_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, ee)))
        ax.plot(kvcm, cond, 'o-', label='Longitudinal {:.3f} GHz'.format(freq))

    plt.xlabel('Field [kV/cm]')
    plt.ylabel('Longitudinal AC Conductivity (S/m)')
    plt.title(pp.title_str)
    # ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.legend(loc='lower left')

    n = utilities.calculate_density(df)

    fig, ax = plt.subplots()
    for freq in freqVector:
        cond = []
        mu_3 = []
        for ee in fieldVector:
            chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
            mu_3.append(utilities.calc_linear_mobility(chi_3_i, df, ee) * 10 ** 4)
            cond.append(np.load(pp.outputLoc + 'Small_Signal/' + 'cond_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, ee)))
        ax.plot(kvcm, np.array(cond)/c.e/n*100**2, '-', label='{:.1f} GHz'.format(freq))
    ax.plot(kvcm,mu_3,'-',label = 'Ohmic Mobility')
    plt.xlabel('Field [kV/cm]')
    plt.ylabel(r'Longitudinal AC Mobility ($cm^2 \, V^{-1}\, s^{-1}$)')
    plt.ylim([-0.4*np.max(mu_3),np.max(mu_3)*1.2])
    # ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.legend(ncol=3,loc='lower center',fontsize=8)


if __name__ == '__main__':
    # Create electron and phonon dataframes
    # preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)
    fields = pp.small_signal_fields
    freqs = pp.freqVector

    #occupation_solver.write_steady(pp.moment_fields, electron_df)
    #occupation_solver.write_transient(fields, electron_df, freqs)

    longitudinal_small_signal_conductivity(electron_df, fields, freqs)
    plot_small_signal_conductivity(fields, freqs, electron_df)

    #psd_solver.write_correlation(fields,electron_df,freqs)
    #psd_solver.write_correlation(pp.moment_fields, electron_df, np.array([0.1]))
    #psd_solver.plot_density(fields, freqs, electron_df)

    # newFreqs = np.geomspace(1,100,100)
    # psd_solver.write_energy_correlation(fields,electron_df,newFreqs)
    # psd_solver.plot_energy_density(fields, np.unique(np.concatenate((freqs, newFreqs))), electron_df)
    # psd_solver.plot_energy_density(fields, newFreqs, electron_df)

    # psd_solver.plot_energy_density(fields, freqs, electron_df)
    plt.show()