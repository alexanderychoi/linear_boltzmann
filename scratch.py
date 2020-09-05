import utilities
def plot_density(ee, freqVector, full_df, pA, opticalPhononEnergy, inverseFrohlichTime,eRT_Ratio):
    g_inds,_,_ = utilities.gaas_split_valleys(full_df,False)
    g_df = full_df[g_inds].reset_index(drop=True)

    # First plot the Perturbo spectral density as a function of frequency and the nyquist spectral density
    Nuc = pp.kgrid ** 3
    S_xx_vector = []
    S_xx_RTA_vector = []
    S_yy_vector = []
    cond = []
    for freq in freqVector:
        chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        S_xx, S_xx_RTA, S_yy, S_yy_RTA, conductivity_xx = psd_solver.density(chi, ee, full_df, freq, False,0)
        S_xx_vector.append(S_xx)
        S_xx_RTA_vector.append(S_xx_RTA)
        S_yy_vector.append(S_yy)
        cond.append(
            np.load(pp.outputLoc + 'Small_Signal/' + 'cond_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, ee)))

    fig, ax = plt.subplots()
    ax.plot(freqVector, np.array(cond) * 4 * c.kb_joule * pp.T / c.Vuc / Nuc, linestyle='--',
            label=r'Nyquist-Johnson',
            color='black')
    ax.plot(freqVector, S_yy_vector, color='tomato', linestyle='-.',
            label=r'$S_{t}$' + '  E = {:.1f} '.format(ee / 1e2) + r'$V\,cm^{-1}$')
    ax.plot(freqVector, S_xx_vector, color='tomato',
            label=r'$S_{l}$' + '  E = {:.1f} '.format(ee / 1e2) + r'$V\,cm^{-1}$')

    # Now plot the high frequency Davydov spectral density
    hifreq = freqVector[freqVector > 200]
    S_xx_acoustic_Davy_hifreq = []
    S_xx_frohlich_Davy_hifreq = []

    davydovFro = frohlich_davydovDistribution(df, plotfield, opticalPhononEnergy, inverseFrohlichTime, eRT_Ratio)
    davydovAco = acoustic_davydovDistribution(df, plotfield, pA, eRT_Ratio)
    momRTAco, _ = acoustic_davydovRTs(df, pA, eRT_Ratio)
    momRTFro = 1 / frohlichScattering(df, opticalPhononEnergy, inverseFrohlichTime)

    for freq in lowfreq:
        S_xx_acoustic_Davy_hifreq.append(acoustic_davydovNoise_hifreq(g_df,davydovAco,momRTAco,freq))
        S_xx_frohlich_Davy_hifreq.append(frohlich_davydovNoise_hifreq(g_df,davydovFro,momRTFro,freq))

    ax.plot(lowfreq, acoustic_davydovNoise_hifreq, color='dodgerblue', linestyle='-.',
            label=r'$S_{t}$' + '  E = {:.1f} '.format(ee / 1e2) + r'$V\,cm^{-1}$')
    ax.plot(lowfreq, frohlich_davydovNoise_hifreq, color='green', linestyle='-.',
            label=r'$S_{t}$' + '  E = {:.1f} '.format(ee / 1e2) + r'$V\,cm^{-1}$')

    plt.legend(loc='lower left')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Spectral Density ' r'$[A^2 \, m^4 \, Hz^{-1}]$')
    plt.xscale('log')
    plt.savefig(pp.figureLoc + 'Freq_Dependent_PSD.png', bbox_inches='tight', dpi=600)