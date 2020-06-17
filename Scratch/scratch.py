import numpy as np
import time
import constants as c
import utilities
import problem_parameters as pp
import matplotlib.pyplot as plt
import numpy.linalg
from scipy.sparse import linalg
import occupation_plotter
import preprocessing
import occupation_solver


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

    _,L_inds,_=utilities.split_valleys(kptdata,False)
    l1_inds,l2_inds,l3_inds,l4_inds,l5_inds,l6_inds,l7_inds,l8_inds = utilities.split_L_valleys(kptdata,False)
    L_list = [l1_inds,l2_inds,l3_inds,l4_inds,l5_inds,l6_inds,l7_inds,l8_inds]
    shortslice_inds = []
    l_icinds = []
    r_icinds = []

    for i1 in range(len(L_list)):
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
            if len(slice_inds) > 4:
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
            else:
                shortslice_inds.append(slice_inds)
            if kind % 10 == 0:
                pass
        print('Scattering matrix modified to incorporate central difference contribution.')
        print('Not applied to {:d} points because fewer than 5 points on the slice.'.format(len(shortslice_inds)))
        print('Finite difference applied to L valleys.')
    if not pp.getX:
        pass
    else:
        print('Finite difference not applied to X valleys. Derivative treated as zero for these points.')
    end = time.time()
    print('Finite difference generation took {:.2f}s'.format(end - start))

    return shortslice_inds, np.array(l_icinds), np.array(r_icinds), matrix


if __name__ == '__main__':
    preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)
    nkpts = len(electron_df)
    fdm = np.memmap(pp.inputLoc + '/finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
    _, left_inds,right_inds, matrix_fd = apply_centraldiff_matrix_L(fdm, electron_df, pp.fieldVector[0])

    x = electron_df['kx [1/A]'].values / (2 * np.pi / c.a)
    y = electron_df['ky [1/A]'].values / (2 * np.pi / c.a)
    z = electron_df['kz [1/A]'].values / (2 * np.pi / c.a)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[left_inds], y[left_inds], z[left_inds],s=0.5,label='left')
    ax.scatter(x[right_inds], y[right_inds], z[right_inds],s=0.5,label='right')
    ax.set_xlabel(r'$kx/2\pi a$')
    ax.set_ylabel(r'$ky/2\pi a$')
    ax.set_zlabel(r'$kz/2\pi a$')
    plt.legend()

    g_inds,l_inds,x_inds = utilities.split_valleys(electron_df,False)

    l1,l2,l3,l4,l5,l6,l7,l8 = utilities.split_L_valleys(electron_df,False)
    plt.figure()
    plt.scatter(electron_df.loc[np.intersect1d(np.where(l1),left_inds),'kx [1/A]'],electron_df.loc[np.intersect1d(np.where(l1),left_inds),'energy [eV]'],s=0.5,label='left')
    plt.scatter(electron_df.loc[np.intersect1d(np.where(l1),right_inds),'kx [1/A]'],electron_df.loc[np.intersect1d(np.where(l1),right_inds),'energy [eV]'],s=0.5,label='right')
    plt.title('L1 Valley Boundary Points')
    plt.xlabel('kx [1/A]')
    plt.ylabel('Energy [eV]')
    plt.ylim([6.3,6.55])
    plt.legend()

    fig = plt.figure()
    plt.scatter(electron_df.loc[l1,'kx [1/A]'],electron_df.loc[l1,'energy [eV]'],s=0.5)
    plt.xlabel('kx [1/A]')
    plt.ylabel('Energy [eV]')
    plt.ylim([6.3,6.55])
    plt.title('L1 Valleys All Points')
    plt.show()