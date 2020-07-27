import numpy as np
import time
import constants as c
import utilities
import problem_parameters as pp
import matplotlib.pyplot as plt
import numpy.linalg
from scipy.sparse import linalg
import preprocessing
import pandas as pd


# Bug fix here. Short slice inds still being stored incorrectly. Need to subtract 1 to get the right value.
def apply_centraldiff_matrix(matrix,fullkpts_df,E):
    """Given a scattering matrix, calculate a modified matrix using the central difference stencil and apply bc. In the
    current version, bc is not applied to points in the L, X valleys.

    Parameters:
        matrix (memmap): Scattering matrix in simple linearization.
        fullkpts_df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        E (dbl): Value of the electric field in V/m.
        scm (memmap): Memory-mapped array containing the scattering matrix, assumed simple linearization by default.

    Returns:
        shortslice_inds (numpyarray): Array of indices associated with the points along ky-kz lines with fewer than 5 pts.
        np.array(icinds) (numpyarray): Array of indices associated with the points where the ic is applied.
        np.array(icinds) (numpyarray): Array of indices associated with the points where the ic is applied.
        matrix (memmap): Memory-mapped array containing the modified scattering matrix, accounting for the FDM.
    """
    # Do not  flush the memmap it will overwrite consecutively.
    # Get the first and last rows since these are different because of the IC. Go through each.
    # Get the unique ky and kz values from the array for looping.
    # This is not robust and should be replaced.
    if pp.kgrid == 160:
        step_size = 0.0070675528500652425 * 1E10  # 1/Angstrom to 1/m (for 160^3)
    if pp.kgrid == 200:
        step_size = 0.005654047459752398 * 1E10  # 1/Angstrom to 1/m (for 200^3)
    if pp.kgrid == 80:
        step_size = 0.0070675528500652425*2*1E10  # 1/Angstron for 1/m (for 80^3)

    kptdata = fullkpts_df[['k_inds', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]']]

    g_inds,l_inds,x_inds=utilities.gaas_split_valleys(fullkpts_df,False)
    g_df = kptdata.loc[g_inds]  # Only apply condition in the Gamma valley
    # g_df = fullkpts_df  # Changeline
    uniq_yz = np.unique(g_df[['ky [1/A]', 'kz [1/A]']].values, axis=0)

    # If there are too few points in a slice < 4, we want to keep track of those points
    shortslice_inds = []
    l_icinds = []
    r_icinds = []
    if pp.fdmName == 'Column Preserving Central Difference':
        print('Applying column preserving central difference scheme.')
    if pp.fdmName == 'Hybrid Difference':
        print('Applying hybrid FDM scheme.')
    if pp.fdmName == 'Backwards Difference':
        print('Applying backward difference scheme.')
    start = time.time()
    # Loop through the unique ky and kz values in the Gamma valley
    for i in range(len(uniq_yz)):
        ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
        # Grab the "slice" of points in k space with the same ky and kz coordinate
        slice_df = g_df.loc[(g_df['ky [1/A]'] == ky) & (g_df['kz [1/A]'] == kz)]
        slice_inds = slice_df['k_inds'].values-1

        if len(slice_inds) > 3:
            # Subset is the slice sorted by kx value in ascending order. The index of subset still references kptdata.
            subset = slice_df.sort_values(by=['kx [1/A]'], ascending=True)
            ordered_inds = subset['k_inds'].values - 1  # indices of matrix (zero indexed)
            l_icinds.append(ordered_inds[0])  # +1 to get the k_inds values (one indexed)
            r_icinds.append(ordered_inds[-1])
            last = len(ordered_inds) - 1
            slast = len(ordered_inds) - 2

            if pp.fdmName == 'Column Preserving Central Difference':
                # Set the "initial condition" i.e. the point with the most negative kx value is treated as being zero
                # (and virtual point below)
                matrix[ordered_inds[0], ordered_inds[1]] += 1/(2*step_size)*c.e*E/c.hbar_joule
                matrix[ordered_inds[1], ordered_inds[2]] += 1/(2*step_size)*c.e*E/c.hbar_joule
                # Set the other "boundary condition" i.e. the point with the most positive kx value is treated as being zero
                # (and virtual point above)
                matrix[ordered_inds[last], ordered_inds[slast]] += -1 * 1/(2*step_size)*c.e*E/c.hbar_joule
                matrix[ordered_inds[slast], ordered_inds[slast-1]] += -1 * 1/(2*step_size)*c.e*E/c.hbar_joule
                # Set the value of all other points in the slice
                inter_inds = ordered_inds[2:slast]
                inter_inds_up = ordered_inds[3:last]
                inter_inds_down = ordered_inds[1:slast-1]
                matrix[inter_inds, inter_inds_up] += 1/(2*step_size)*c.e*E/c.hbar_joule
                matrix[inter_inds, inter_inds_down] += -1 * 1/(2*step_size)*c.e*E/c.hbar_joule

            if pp.fdmName == 'Backwards Difference':
                print('Doing this.')
                # Set the "initial condition" i.e. the point with the most negative kx value has virtual point below
                # that is assumed to be zero
                inter_inds = ordered_inds[1:last+1]
                inter_inds_down = ordered_inds[0:last]
                matrix[ordered_inds, ordered_inds] += 1/(step_size)*c.e*E/c.hbar_joule
                matrix[inter_inds, inter_inds_down] += -1 * 1/(step_size)*c.e*E/c.hbar_joule

            if pp.fdmName == 'Hybrid Difference':
                matrix[ordered_inds[0],ordered_inds[0]] = -1 * 1/(step_size)*c.e*E/c.hbar_joule
                matrix[ordered_inds[0], ordered_inds[1]] = 1 / (step_size) * c.e * E / c.hbar_joule
                matrix[ordered_inds[1], ordered_inds[0]] = -1 / (2*step_size) * c.e * E / c.hbar_joule
                matrix[ordered_inds[slast],ordered_inds[last]] = 1/(2*step_size)*c.e*E/c.hbar_joule
        else:
            shortslice_inds.append(slice_inds)
    print('Scattering matrix modified to incorporate central difference contribution.')
    shortslice_inds = np.concatenate(shortslice_inds,axis=0)  # Changeline
    print('Not applied to {:d} Gamma points because fewer than 4 points on the slice.'.format(len(shortslice_inds)))
    print('This represents {:1f} % of points in the Gamma valley.'.format(len(shortslice_inds)/len(g_df)*100))
    end = time.time()
    print('Finite difference generation took {:.2f}s'.format(end - start))

    return matrix, shortslice_inds, np.array(l_icinds), np.array(r_icinds)



def apply_centraldiff_matrix_L(matrix,fullkpts_df,E,step_size=1):
    """Given a scattering matrix, calculate a modified matrix using the central difference stencil and apply bc. In the
    current version, bc is not applied to points in the L, X valleys.

    Parameters:
        matrix (memmap): Scattering matrix in simple linearization.
        fullkpts_df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        E (dbl): Value of the electric field in V/m.
        scm (memmap): Memory-mapped array containing the scattering matrix, assumed simple linearization by default.
        step_size (dbl): Specifies the spacing between consecutive k-pts for the integration.

    Returns:
        shortslice_inds (numpyarray): Array of indices associated with the points along ky-kz lines with fewer than 5 pts.
        np.array(icinds) (numpyarray): Array of indices associated with the points where the ic is applied.
        np.array(icinds) (numpyarray): Array of indices associated with the points where the ic is applied.
        matrix (memmap): Memory-mapped array containing the modified scattering matrix, accounting for the FDM.
    """
    # Do not  flush the memmap it will overwrite consecutively.
    # Get the first and last rows since these are different because of the IC. Go through each.
    # Get the unique ky and kz values from the array for looping.
    # This is not robust and should be replaced.
    if pp.kgrid == 160:
        step_size = 0.0070675528500652425 * 1E10  # 1/Angstrom to 1/m (for 160^3)
    if pp.kgrid == 200:
        step_size = 0.005654047459752398 * 1E10  # 1/Angstrom to 1/m (for 200^3)
    if pp.kgrid == 80:
        step_size = 0.0070675528500652425*2*1E10  # 1/Angstron for 1/m (for 80^3)

    kptdata = fullkpts_df[['k_inds', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]','energy [eV]']]

    _,L_inds,_=utilities.gaas_split_valleys(kptdata,False)
    l1_inds,l2_inds,l3_inds,l4_inds,l5_inds,l6_inds,l7_inds,l8_inds = utilities.split_L_valleys(kptdata,False)
    L_list = [l1_inds,l2_inds,l3_inds,l4_inds,l5_inds,l6_inds,l7_inds,l8_inds]
    L_list = [l4_inds,l2_inds,l3_inds,l4_inds,l5_inds,l6_inds,l7_inds,l8_inds]

    shortslice_inds = []
    l_icinds = []
    r_icinds = []
    if pp.hybridFDM:
        print('Applying hybrid FDM scheme.')
    for i1 in range(len(L_list)):
        if i1>0:
            print('Breaking')
            break
        print('Applying to {} L valley'.format(i1))
        l_df = kptdata.loc[L_list[i1]]
        uniq_yz = np.unique(l_df[['ky [1/A]', 'kz [1/A]']].values, axis=0)
        # If there are too few points in a slice < 5, we want to keep track of those points
        start = time.time()
        # Loop through the unique ky and kz values in the Gamma valley
        for i in range(len(uniq_yz)):
            kind = i + 1
            ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
            # Grab the "slice" of points in k space with the same ky and kz coordinate
            slice_df = l_df.loc[(l_df['ky [1/A]'] == ky) & (l_df['kz [1/A]'] == kz)]
            slice_inds = slice_df['k_inds'].values
            if len(slice_inds) > 3:
                # Subset is the slice sorted by kx value in ascending order. The index of subset still references kptdata.
                subset = slice_df.sort_values(by=['kx [1/A]'], ascending=True)
                ordered_inds = subset['k_inds'].values - 1  # indices of matrix (zero indexed)
                l_icinds.append(ordered_inds[0])  # +1 to get the k_inds values (one indexed)
                r_icinds.append(ordered_inds[-1])
                # Set the "initial condition" i.e. the point with the most negative kx value is treated as being zero
                # (and virtual point below)
                matrix[ordered_inds[0], ordered_inds[1]] += 1/(2*step_size)*c.e*E/c.hbar_joule
                matrix[ordered_inds[1], ordered_inds[2]] += 1/(2*step_size)*c.e*E/c.hbar_joule
                # Set the other "boundary condition" i.e. the point with the most positive kx value is treated as being zero
                # (and virtual point above)
                last = len(ordered_inds) - 1
                slast = len(ordered_inds) - 2
                matrix[ordered_inds[last], ordered_inds[slast]] += -1 * 1/(2*step_size)*c.e*E/c.hbar_joule
                matrix[ordered_inds[slast], ordered_inds[slast-1]] += -1 * 1/(2*step_size)*c.e*E/c.hbar_joule
                # Set the value of all other points in the slice
                inter_inds = ordered_inds[2:slast]
                inter_inds_up = ordered_inds[3:last]
                inter_inds_down = ordered_inds[1:slast-1]
                matrix[inter_inds, inter_inds_up] += 1/(2*step_size)*c.e*E/c.hbar_joule
                matrix[inter_inds, inter_inds_down] += -1 * 1/(2*step_size)*c.e*E/c.hbar_joule
                if pp.hybridFDM:
                    matrix[ordered_inds[0], ordered_inds[0]] = -1 * 1 / (step_size) * c.e * E / c.hbar_joule
                    matrix[ordered_inds[0], ordered_inds[1]] = 1 / (step_size) * c.e * E / c.hbar_joule
                    matrix[ordered_inds[1], ordered_inds[0]] = -1 / (2 * step_size) * c.e * E / c.hbar_joule
                    matrix[ordered_inds[slast], ordered_inds[last]] = 1 / (2 * step_size) * c.e * E / c.hbar_joule
            else:
                shortslice_inds.append(slice_inds-1)
            if kind % 10 == 0:
                pass
    print('Scattering matrix modified to incorporate central difference contribution.')
    shortslice_inds = np.concatenate(shortslice_inds, axis=0)
    print('Not applied to {:d} L valley points because fewer than 4 points on the slice.'.format(len(shortslice_inds)))
    print('This represents {:1f} % of points in the L valley.'.format(len(shortslice_inds) / len(fullkpts_df.loc[L_inds]) * 100))
    print('Finite difference applied to L valleys.')
    if not pp.getX:
        pass
    else:
        print('Finite difference not applied to X valleys. Derivative treated as zero for these points.')
    end = time.time()
    print('Finite difference generation took {:.2f}s'.format(end - start))

    return shortslice_inds, np.array(l_icinds), np.array(r_icinds), matrix


def gamma_boundary_analysis(df,plotField,fieldVector):
    """This function is designed to perform analysis on solutions to the steady Boltzmann equation in GaAs using a
    finite difference approximation for the drift derivative. This is only for the Gamma valley In particular, this
    function will generate plots that detail:
    1.) The number of states with negative distribution functions.
    2.) The location of states with negative distribution functions.
    3.) The magnitude of the distribution function for ky-kz slices, and the boundary points.
    4.) The mean value of the distribution function at the boundaries compared to the mean value of the interior points
        on the same slice, as a function of average interior point energy.

    Parameters:
        matrix (memmap): Scattering matrix in simple linearization.
        fullkpts_df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        E (dbl): Value of the electric field in V/m.
        scm (memmap): Memory-mapped array containing the scattering matrix, assumed simple linearization by default.
        step_size (dbl): Specifies the spacing between consecutive k-pts for the integration.

    Returns:
        shortslice_inds (numpyarray): Array of indices associated with the points along ky-kz lines with fewer than 5 pts.
        np.array(icinds) (numpyarray): Array of indices associated with the points where the ic is applied.
        np.array(icinds) (numpyarray): Array of indices associated with the points where the ic is applied.
        matrix (memmap): Memory-mapped array containing the modified scattering matrix, accounting for the FDM.
    """
    fdmName = pp.fdmName

    nkpts = len(np.unique(electron_df['k_inds']))
    fdm = np.memmap(pp.inputLoc + '/finite_difference_matrix2.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
    matrix_fd, gamma_shortslice_inds, gamma_left_inds, gamma_right_inds = apply_centraldiff_matrix(fdm, df, 1)
    g_inds, l_inds, _ = utilities.gaas_split_valleys(df, False)
    g_df = df.loc[g_inds]
    left_df = electron_df.loc[gamma_left_inds]
    right_df = electron_df.loc[gamma_right_inds]
    short_df = electron_df.loc[gamma_shortslice_inds]

    plt.figure()
    plt.plot(g_df['kx [1/A]'].values,g_df['energy [eV]'].values,'.', label='Interior Gamma',color='black')
    plt.plot(left_df['kx [1/A]'].values,left_df['energy [eV]'].values,'.', label='Left Gamma')
    plt.plot(right_df['kx [1/A]'].values,right_df['energy [eV]'].values,'.', label='Right Gamma')
    plt.plot(short_df['kx [1/A]'].values,short_df['energy [eV]'].values,'.', label='Short Gamma')
    plt.xlabel('kx [1/A]')
    plt.ylabel('Energy [eV]')
    plt.title(pp.title_str)
    plt.legend()


    ng_left = []
    ng_right = []
    ng = []
    for ee in pp.fieldVector:
        chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        ng_left.append(utilities.calc_popinds(chi_3_i,electron_df,gamma_left_inds))
        ng_right.append(utilities.calc_popinds(chi_3_i,electron_df,gamma_right_inds))
        ng.append(utilities.calc_popinds(chi_3_i,electron_df,g_inds))
    kvcm = np.array(pp.fieldVector)*1e-5
    ng_left = np.array(ng_left)
    ng_right = np.array(ng_right)
    ng = np.array(ng)

    plt.figure()
    plt.plot(kvcm,ng_left/ng*100,label='Gamma Left Population')
    plt.plot(kvcm,ng_right/ng*100,label='Gamma Right Population')
    plt.ylabel('Percentage of total carrier number')
    plt.xlabel('Efield [kV/cm]')
    plt.title(pp.title_str)
    plt.legend()

    neg_states = []
    for ee in fieldVector:
        chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        g_df['f'] = chi_3_i[g_inds] + g_df['k_FD']
        g_df['abs_f'] = np.abs(g_df['f'].values)
        ng_df = g_df.loc[g_df['f'] < 0]
        neg_states.append(len(ng_df))

    plt.figure()
    plt.plot(kvcm,neg_states)
    plt.ylabel('Number of negative states')
    plt.xlabel('Efield [kV/cm]')
    plt.title(pp.title_str)
    plt.legend()

    chi_3 = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(plotField))
    g_df['f'] = chi_3[g_inds] + g_df['k_FD']
    g_df['abs_f'] = np.abs(g_df['f'].values)
    textstr = '\n'.join((r'$E = %.1f kV/cm \, \, (100) $' % (plotField / 10 ** 5,), fdmName))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig, ax = plt.subplots()
    ax.plot(ng_df['kx [1/A]'].values, ng_df['abs_f'].values, '.',
            label='Negative distribution states')
    # plt.text(0, 1, 'E = {:1f}'.format(field))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    plt.legend()
    plt.xlabel('kx [1/A]')
    plt.yscale('log')
    plt.ylabel('|f| [arb]')
    plt.title(pp.title_str)

    fig, ax = plt.subplots()
    ax.plot(ng_df['kx [1/A]'].values, ng_df['energy [eV]'].values, '.',
            label='Negative distribution states')
    # plt.text(0, 1, 'E = {:1f}'.format(field))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    plt.legend()
    plt.xlabel('kx [1/A]')
    plt.yscale('log')
    plt.ylabel('Energy [eV]')
    plt.title(pp.title_str)


    # Plot 1: ky-kz slices plot(kx,distribution)
    uniq_yz = np.unique(g_df[['ky [1/A]', 'kz [1/A]']].values, axis=0)
    textstr='\n'.join((r'$E = %.1f kV/cm \, \, (100) $' % (plotField/10**5,),fdmName))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig, ax = plt.subplots()
    for i in range(len(uniq_yz)):
        kind = i + 1
        ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
        # Grab the "slice" of points in k space with the same ky and kz coordinate
        slice_df = g_df.loc[(g_df['ky [1/A]'] == ky) & (g_df['kz [1/A]'] == kz)].sort_values(by=['kx [1/A]'],ascending=True)
        ax.plot(slice_df['kx [1/A]'],slice_df['abs_f'],color='black',linewidth=0.5)
        ax.plot(slice_df['kx [1/A]'],slice_df['abs_f'],'.',color='black')
    ax.plot(g_df.loc[gamma_left_inds,'kx [1/A]'].values,g_df.loc[gamma_left_inds,'abs_f'].values,'.', label='Left Gamma')
    ax.plot(g_df.loc[gamma_right_inds,'kx [1/A]'].values,g_df.loc[gamma_right_inds,'abs_f'].values,'.', label='Right Gamma')
    # plt.text(0, 1, 'E = {:1f}'.format(field))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    plt.legend()
    plt.xlabel('kx [1/A]')
    plt.yscale('log')
    plt.ylim([1e-13,1e-1])
    plt.ylabel('f [arb]')
    plt.title(pp.title_str)

    # Plot 2: kx-kz slices plot(ky,distribution)
    uniq_xz = np.unique(g_df[['kx [1/A]', 'kz [1/A]']].values, axis=0)
    textstr='\n'.join((r'$E = %.1f kV/cm \,  \, (100) $' % (plotField/10**5,),fdmName))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig, ax = plt.subplots()
    for i in range(len(uniq_xz)):
        kind = i + 1
        kx, kz = uniq_xz[i, 0], uniq_xz[i, 1]
        # Grab the "slice" of points in k space with the same ky and kz coordinate
        slice_df = g_df.loc[(g_df['kx [1/A]'] == kx) & (g_df['kz [1/A]'] == kz)].sort_values(by=['ky [1/A]'],ascending=True)
        ax.plot(slice_df['ky [1/A]'],slice_df['abs_f'],color='black',linewidth=0.5)
        ax.plot(slice_df['ky [1/A]'],slice_df['abs_f'],'.',color='black')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    plt.legend()
    plt.xlabel('ky [1/A]')
    plt.yscale('log')
    plt.ylim([1e-13,1e-1])
    plt.ylabel('f [arb]')
    plt.title(pp.title_str)
    # Plot 3: ky-kz slices plot(mean_interior_energy,distribution)
    textstr='\n'.join((r'$E = %.1f kV/cm \, \, (100) $' % (plotField/10**5,),fdmName,'ky-kz slices'))
    fig, ax = plt.subplots()
    for i in range(len(uniq_yz)):
        kind = i + 1
        ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
        # Grab the "slice" of points in k space with the same ky and kz coordinate

        slice_df = g_df.loc[(g_df['ky [1/A]'] == ky) & (g_df['kz [1/A]'] == kz)].sort_values(by=['kx [1/A]'],
                                                                                             ascending=True)
        if len(slice_df)>3:
            min_kx = np.min(slice_df['kx [1/A]'])
            max_kx = np.max(slice_df['kx [1/A]'])
            left_f = slice_df.loc[slice_df['kx [1/A]'] == min_kx, 'abs_f'].values
            right_f = slice_df.loc[slice_df['kx [1/A]'] == max_kx, 'abs_f'].values

            interior_df = slice_df.loc[(slice_df['kx [1/A]']<max_kx)&(slice_df['kx [1/A]']>min_kx)]
            ax.plot(np.mean(interior_df['energy [eV]'].values), np.mean(interior_df['abs_f'].values), '.', color='black')
            ax.plot(np.mean(interior_df['energy [eV]'].values), left_f, '.', color='orange')
            ax.plot(np.mean(interior_df['energy [eV]'].values), right_f, '.', color='blue')

    ax.plot(np.mean(interior_df['energy [eV]'].values), np.mean(interior_df['abs_f'].values), '.', color='black',label='Gamma Mean Interior')
    ax.plot(np.mean(interior_df['energy [eV]'].values), left_f, '.', color='orange',label='Gamma kx Left')
    ax.plot(np.mean(interior_df['energy [eV]'].values), right_f, '.', color='blue',label='Gamma kx Right')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    plt.legend()
    plt.xlabel('Mean Interior Energy [eV]')
    plt.yscale('log')
    plt.ylabel('f [arb]')
    plt.ylim([1e-11,1e-1])
    plt.title(pp.title_str)

    # Plot 4: kx-kz slices plot(mean_interior_energy,distribution)
    textstr='\n'.join((r'$E = %.1f kV/cm \, \, (100) $' % (plotField/10**5,),fdmName,'kx-kz slices'))
    fig, ax = plt.subplots()
    for i in range(len(uniq_xz)):
        kind = i + 1
        kx, kz = uniq_xz[i, 0], uniq_xz[i, 1]
        # Grab the "slice" of points in k space with the same ky and kz coordinate

        slice_df = g_df.loc[(g_df['kx [1/A]'] == kx) & (g_df['kz [1/A]'] == kz)].sort_values(by=['ky [1/A]'],
                                                                                             ascending=True)
        if len(slice_df)>3:
            min_ky = np.min(slice_df['ky [1/A]'])
            max_ky = np.max(slice_df['ky [1/A]'])
            left_f = slice_df.loc[slice_df['ky [1/A]'] == min_ky, 'abs_f'].values
            right_f = slice_df.loc[slice_df['ky [1/A]'] == max_ky, 'abs_f'].values

            interior_df = slice_df.loc[(slice_df['ky [1/A]']<max_ky)&(slice_df['ky [1/A]']>min_ky)]
            ax.plot(np.mean(interior_df['energy [eV]'].values), np.mean(interior_df['abs_f'].values), '.', color='black')
            ax.plot(np.mean(interior_df['energy [eV]'].values), left_f, '.', color='orange')
            ax.plot(np.mean(interior_df['energy [eV]'].values), right_f, '.', color='blue')

    ax.plot(np.mean(interior_df['energy [eV]'].values), np.mean(interior_df['abs_f'].values), '.', color='black',label='Gamma Mean Interior')
    ax.plot(np.mean(interior_df['energy [eV]'].values), left_f, '.', color='orange',label='Gamma ky Left')
    ax.plot(np.mean(interior_df['energy [eV]'].values), right_f, '.', color='blue',label='Gamma ky Right')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    plt.legend()
    plt.xlabel('Mean Interior Energy [eV]')
    plt.yscale('log')
    plt.ylabel('f [arb]')
    plt.ylim([1e-11,1e-2
              ])
    plt.title(pp.title_str)


if __name__ == '__main__':
    preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)

    gamma_boundary_analysis(electron_df, pp.fieldVector[-1], pp.fieldVector)
    #
    # l = 5
    # # h = np.array(np.meshgrid(range(1, l + 1), range(1, l + 1), range(1, l + 1))).T.reshape(-1, 3)
    # h = np.array(np.meshgrid(range(1,l+1), 1, 1)).T.reshape(-1, 3)
    # d = {'kx [1/A]': h[:, 0], 'ky [1/A]': h[:, 1], 'kz [1/A]': h[:, 2]}
    # practice_df = pd.DataFrame(data=d).sort_values(by=['kx [1/A]', 'ky [1/A]', 'kz [1/A]'], ascending=True).reset_index(
    #     drop=True)
    # practice_df['k_inds'] = np.array(range(1, len(practice_df) + 1))
    # nkpts = len(practice_df)
    # fdm2 = np.memmap(pp.inputLoc + '/finite_difference_matrix4.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
    #
    # m,_,_,_ = apply_centraldiff_matrix(fdm2, practice_df, 1 * c.hbar_joule / c.e * 0.0070675528500652425 * 1E10)
    #
    # fig = plt.figure()
    # fig.set_size_inches(6, 6)
    # plt.matshow(m, fignum=1)
    # plt.colorbar()
    # plt.title(pp.fdmName + ' E = ({:.1f},{:.1f},{:.1f})'.format(pp.fieldDirection[0], pp.fieldDirection[1],
    #                                                                          pp.fieldDirection[2]))

    plt.show()

    # nkpts = len(np.unique(electron_df['k_inds']))
    # fdm = np.memmap(pp.inputLoc + '/finite_difference_matrix2.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
    # gamma_shortslice_inds, gamma_left_inds, gamma_right_inds, _, matrix_fd = apply_centraldiff_matrix(fdm, electron_df, 1)
    # del matrix_fd
    # del fdm
    # g_inds,l_inds,_ = utilities.gaas_split_valleys(electron_df,False)
    #
    # g_df = electron_df.loc[g_inds]
    # left_df = electron_df.loc[gamma_left_inds]
    # right_df = electron_df.loc[gamma_right_inds]
    # short_df = electron_df.loc[gamma_shortslice_inds]
    #
    # # field = pp.fieldVector[-1]
    # field = 1e5
    # field = pp.fieldVector[0]
    # chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(field))
    #
    # g_df['f'] = chi_3_i[g_inds] + g_df['k_FD']
    # g_df['abs_f'] = np.abs(g_df['f'].values)

    # ng_df = g_df.loc[g_df['f'] < 0]
    # plt.figure()
    # plt.plot(ng_df['kx [1/A]'].values,ng_df['energy [eV]'].values,'.', label='Negative Distribution')
    # plt.xlabel('kx [1/A]')
    # plt.ylabel('Energy [eV]')
    # plt.title(pp.title_str)
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(ng_df['kx [1/A]'].values,-ng_df['f'].values/ng_df['k_FD'].values,'.', label='Negative Distribution')
    # plt.xlabel('kx [1/A]')
    # plt.ylabel('f/f0')
    # plt.yscale('log')
    # plt.title(pp.title_str)
    # plt.legend()
    #
    # common1 = np.nonzero(np.in1d(ng_df['k_inds'], g_df.loc[gamma_left_inds,'k_inds']))[0]
    # print(common1)
    # common2 = np.nonzero(np.in1d(ng_df['k_inds'], g_df.loc[gamma_right_inds,'k_inds']))[0]
    # print(common2)
    #
    #
    # plt.figure()
    # plt.plot(g_df['kx [1/A]'].values,g_df['energy [eV]'].values,'.', label='Interior Gamma',color='black')
    # plt.plot(left_df['kx [1/A]'].values,left_df['energy [eV]'].values,'.', label='Left Gamma')
    # plt.plot(right_df['kx [1/A]'].values,right_df['energy [eV]'].values,'.', label='Right Gamma')
    # plt.plot(short_df['kx [1/A]'].values,short_df['energy [eV]'].values,'.', label='Short Gamma')
    # plt.xlabel('kx [1/A]')
    # plt.ylabel('Energy [eV]')
    # plt.title(pp.title_str)
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(g_df['kx [1/A]'].values,g_df['energy [eV]'].values,'.', label='Interior Gamma',color='black')
    # plt.plot(left_df['kx [1/A]'].values,left_df['energy [eV]'].values,'.', label='Left Gamma')
    # plt.plot(right_df['kx [1/A]'].values,right_df['energy [eV]'].values,'.', label='Right Gamma')
    # plt.plot(short_df['kx [1/A]'].values,short_df['energy [eV]'].values,'.', label='Short Gamma')
    # plt.plot(ng_df['kx [1/A]'].values,ng_df['energy [eV]'].values,'.', label='Negative Distribution')
    # plt.xlabel('kx [1/A]')
    # plt.ylabel('Energy [eV]')
    # plt.title(pp.title_str)
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(g_df['kx [1/A]'].values,g_df['f'].values,'.', label='Interior Gamma',color='black')
    # plt.plot(g_df.loc[gamma_left_inds,'kx [1/A]'].values,g_df.loc[gamma_left_inds,'f'].values,'.', label='Left Gamma')
    # plt.plot(g_df.loc[gamma_right_inds,'kx [1/A]'].values,g_df.loc[gamma_right_inds,'f'].values,'.', label='Right Gamma')
    #
    # plt.xlabel('kx [1/A]')
    # plt.ylabel('f [arb]')
    # plt.title(pp.title_str)
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(g_df['kx [1/A]'].values,g_df['f'].values,'.', label='Interior Gamma',color='black')
    # plt.plot(g_df.loc[gamma_left_inds,'kx [1/A]'].values,g_df.loc[gamma_left_inds,'f'].values,'.', label='Left Gamma')
    # plt.plot(g_df.loc[gamma_right_inds,'kx [1/A]'].values,g_df.loc[gamma_right_inds,'f'].values,'.', label='Right Gamma')
    # plt.xlabel('kx [1/A]')
    # plt.ylabel('f [arb]')
    # plt.title(pp.title_str)
    # plt.yscale('log')
    # plt.legend()

    # fdmName = 'Column Preserving Central Difference'
    #
    # uniq_yz = np.unique(g_df[['ky [1/A]', 'kz [1/A]']].values, axis=0)
    # textstr='\n'.join((r'$E = %.1f kV/cm $' % (field/10**5,),fdmName))
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # fig, ax = plt.subplots()
    # for i in range(len(uniq_yz)):
    #     kind = i + 1
    #     ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
    #     # Grab the "slice" of points in k space with the same ky and kz coordinate
    #     slice_df = g_df.loc[(g_df['ky [1/A]'] == ky) & (g_df['kz [1/A]'] == kz)].sort_values(by=['kx [1/A]'],ascending=True)
    #     ax.plot(slice_df['kx [1/A]'],slice_df['abs_f'],color='black',linewidth=0.5)
    #     ax.plot(slice_df['kx [1/A]'],slice_df['abs_f'],'.',color='black')
    # ax.plot(g_df.loc[gamma_left_inds,'kx [1/A]'].values,g_df.loc[gamma_left_inds,'abs_f'].values,'.', label='Left Gamma')
    # ax.plot(g_df.loc[gamma_right_inds,'kx [1/A]'].values,g_df.loc[gamma_right_inds,'abs_f'].values,'.', label='Right Gamma')
    # # plt.text(0, 1, 'E = {:1f}'.format(field))
    # ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
    #         verticalalignment='top', bbox=props)
    # plt.legend()
    # plt.xlabel('kx [1/A]')
    # plt.yscale('log')
    # plt.ylim([1e-13,1e-3])
    # plt.ylabel('f [arb]')
    # plt.title(pp.title_str)
    #
    #
    #
    # fig, ax = plt.subplots()
    # for i in range(len(uniq_yz)):
    #     kind = i + 1
    #     ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
    #     # Grab the "slice" of points in k space with the same ky and kz coordinate
    #
    #     slice_df = g_df.loc[(g_df['ky [1/A]'] == ky) & (g_df['kz [1/A]'] == kz)].sort_values(by=['kx [1/A]'],
    #                                                                                          ascending=True)
    #     if len(slice_df)>3:
    #         min_kx = np.min(slice_df['kx [1/A]'])
    #         max_kx = np.max(slice_df['kx [1/A]'])
    #         left_f = slice_df.loc[slice_df['kx [1/A]'] == min_kx, 'abs_f'].values
    #         right_f = slice_df.loc[slice_df['kx [1/A]'] == max_kx, 'abs_f'].values
    #
    #         interior_df = slice_df.loc[(slice_df['kx [1/A]']<max_kx)&(slice_df['kx [1/A]']>min_kx)]
    #         ax.plot(np.mean(interior_df['energy [eV]'].values), np.mean(interior_df['abs_f'].values), '.', color='black')
    #         ax.plot(np.mean(interior_df['energy [eV]'].values), left_f, '.', color='orange')
    #         ax.plot(np.mean(interior_df['energy [eV]'].values), right_f, '.', color='blue')
    #
    # ax.plot(np.mean(interior_df['energy [eV]'].values), np.mean(interior_df['abs_f'].values), '.', color='black',label='Gamma Mean Interior')
    # ax.plot(np.mean(interior_df['energy [eV]'].values), left_f, '.', color='orange',label='Gamma Left')
    # ax.plot(np.mean(interior_df['energy [eV]'].values), right_f, '.', color='blue',label='Gamma Right')
    # ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
    #         verticalalignment='top', bbox=props)
    # plt.legend()
    # plt.xlabel('Mean Interior Energy [eV]')
    # plt.yscale('log')
    # plt.ylabel('f [arb]')
    # plt.ylim([1e-11,1e-3])
    # plt.title(pp.title_str)



    # ng_left = []
    # ng_right = []
    # ng = []
    # for ee in pp.fieldVector:
    #     chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
    #     ng_left.append(utilities.calc_popinds(chi_3_i,electron_df,gamma_left_inds))
    #     ng_right.append(utilities.calc_popinds(chi_3_i,electron_df,gamma_right_inds))
    #     ng.append(utilities.calc_popinds(chi_3_i,electron_df,g_inds))
    # kvcm = np.array(pp.fieldVector)*1e-5
    # ng_left = np.array(ng_left)
    # ng_right = np.array(ng_right)
    # ng = np.array(ng)
    #
    # plt.figure()
    # plt.plot(kvcm,ng_left/ng*100,label='Gamma Left Population')
    # plt.plot(kvcm,ng_right/ng*100,label='Gamma Right Population')
    # plt.legend()
    # plt.show()

    # fdm = np.memmap(pp.inputLoc + '/finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
    # l_shortslice_inds, l_left_inds, l_right_inds, matrix_fd = apply_centraldiff_matrix_L(fdm, electron_df, 1)
    #
    # left_df = electron_df.loc[l_left_inds]
    # right_df = electron_df.loc[l_right_inds]
    # short_df = electron_df.loc[l_shortslice_inds]
    #
    # low_energy = right_df.loc[right_df['energy [eV]'] < 6.4]
    # # low_energy = left_df.loc[left_df['energy [eV]'] < 6.4]
    # uniq_yz = np.unique(low_energy[['ky [1/A]', 'kz [1/A]']].values, axis=0)
    # l1_inds,l2_inds,l3_inds,l4_inds,l5_inds,l6_inds,l7_inds,l8_inds = utilities.split_L_valleys(electron_df,False)
    # l1_df = electron_df.loc[l4_inds]
    #
    # plt.figure()
    # for i in range(len(uniq_yz)):
    #     kind = i + 1
    #     ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
    #     # Grab the "slice" of points in k space with the same ky and kz coordinate
    #     slice_df = l1_df.loc[(l1_df['ky [1/A]'] == ky) & (l1_df['kz [1/A]'] == kz)].sort_values(by=['kx [1/A]'],ascending=True)
    #     plt.plot(slice_df['kx [1/A]'],slice_df['energy [eV]'])
    #     plt.plot(slice_df['kx [1/A]'],slice_df['energy [eV]'],'.')
    # plt.xlabel('kx [1/A]')
    # plt.ylabel('Energy [eV]')
    # plt.title('Low Energy Right Boundaries L1 ' + pp.title_str)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = l1_df['kx [1/A]'].values
    # y = l1_df['ky [1/A]'].values
    # z = l1_df['kz [1/A]'].values
    # ax.scatter(x, y, z, color='black',s=0.5)
    # for i in range(len(uniq_yz)):
    #     kind = i + 1
    #     ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
    #     # Grab the "slice" of points in k space with the same ky and kz coordinate
    #     slice_df = l1_df.loc[(l1_df['ky [1/A]'] == ky) & (l1_df['kz [1/A]'] == kz)].sort_values(by=['kx [1/A]'],ascending=True)
    #     x = slice_df['kx [1/A]'].values
    #     y = slice_df['ky [1/A]'].values
    #     z = slice_df['kz [1/A]'].values
    #     ax.plot(x, y, z,'-r')
    #
    # ax.set_xlabel('kx [1/A]')
    # ax.set_ylabel('ky [1/A]')
    # ax.set_zlabel('kz [1/A]')
    # plt.title('Low Energy Right Boundaries L8 ' + pp.title_str)
    #
    # plt.figure()
    # plt.plot(l1_df['kx [1/A]'].values,l1_df['energy [eV]'].values,'.', color = 'black', label='Interior L8')
    # plt.plot(left_df['kx [1/A]'].values,left_df['energy [eV]'].values,'.', label='Left L8')
    # plt.plot(right_df['kx [1/A]'].values,right_df['energy [eV]'].values,'.', label='Right L8')
    # plt.plot(short_df['kx [1/A]'].values,short_df['energy [eV]'].values,'.', label='Short L8')
    # plt.legend()
    # plt.xlabel('kx [1/A]')
    # plt.ylabel('Energy [eV]')
    # plt.title(pp.title_str)
    # plt.show()