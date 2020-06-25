import utilities
import problem_parameters as pp
import time
import numpy as np
import constants as c
import pandas as pd
import matplotlib.pyplot as plt


def gaas_Gamma_FDM(matrix,fullkpts_df,E):
    """Calculate a finite difference matrix using the columnsum-preserving central difference stencil and apply bc.
    Allows for a field oriented along an arbitrary direction, specified in problem_parameters.

    Parameters:
        matrix (memmap): Scattering matrix in simple linearization.
        fullkpts_df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        E (dbl): Value of the electric field in V/m. Direction specified in problem_parameters.

    Returns:
        matrix (memmap): Memory-mapped array FDM. Arranged for LHS of BTE (i.e. + df/dkx)
        shortslice_inds (numpyarray): Array of indices associated with the points along ky-kz lines with fewer than 4pts
        np.array(icinds) (numpyarray): Array of indices associated with the points where the ic is applied.
        np.array(icinds) (numpyarray): Array of indices associated with the points where the ic is applied.
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

    g_df = fullkpts_df  # Need to change this

    # For an arbitrary field direction, we calculate the FDM along the three Cartesian axes and sum them in accordance
    # with the projection of the field along the axes.
    uniq_yz = np.unique(g_df[['ky [1/A]', 'kz [1/A]']].values, axis=0)  # Unique combinations of ky-kz, kx FDM
    uniq_xz = np.unique(g_df[['kx [1/A]', 'kz [1/A]']].values, axis=0)  # Unique combinations of kx-kz, ky FDM
    uniq_xy = np.unique(g_df[['kx [1/A]', 'ky [1/A]']].values, axis=0)  # Unique combinations of kx-ky, kz FDM
    unitProjection = utilities.cartesian_projection(pp.fieldDirection)
    xProj = unitProjection[0]  # Projection of the FDM along kx
    yProj = unitProjection[1]  # Projection of the FDM along ky
    zProj = unitProjection[2]  # Projection of the FDM along kz

    # If there are too few points in a slice < 4, we want to keep track of those points. We also want to keep track of
    # which points are considered boundary.
    shortslice_inds = []
    l_icinds = []
    r_icinds = []

    start = time.time()
    # First do kx projection of the FDM. Loop through unique ky and kz values in the Gamma valley.
    if xProj is not 0:
        print('Calculating projection of FDM along kx-axis.')
        for i in range(len(uniq_yz)):
            ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
            # Grab the "slice" of points in k space with the same ky and kz coordinate
            slice_df = g_df.loc[(g_df['ky [1/A]'] == ky) & (g_df['kz [1/A]'] == kz)]
            slice_inds = slice_df['k_inds'].values

            if len(slice_inds) > 3:
                # Subset is the slice sorted by kx value in ascending order. The index of subset still references kptdata.
                subset = slice_df.sort_values(by=['kx [1/A]'], ascending=True)
                ordered_inds = subset['k_inds'].values - 1  # indices of matrix (zero indexed)
                l_icinds.append(ordered_inds[0])  # +1 to get the k_inds values (one indexed)
                r_icinds.append(ordered_inds[-1])
                # Set the "initial condition" i.e. the point with the most negative kx value is treated as being zero
                # (and virtual point below)
                matrix[ordered_inds[0], ordered_inds[1]] += 1/(2*step_size)*c.e*E/c.hbar_joule*xProj
                matrix[ordered_inds[1], ordered_inds[2]] += 1/(2*step_size)*c.e*E/c.hbar_joule*xProj
                # Set the other "boundary condition" i.e. the point with the most positive kx value is treated as being zero
                # (and virtual point above)
                last = len(ordered_inds) - 1
                slast = len(ordered_inds) - 2
                matrix[ordered_inds[last], ordered_inds[slast]] += -1 * 1/(2*step_size)*c.e*E/c.hbar_joule*xProj
                matrix[ordered_inds[slast], ordered_inds[slast-1]] += -1 * 1/(2*step_size)*c.e*E/c.hbar_joule*xProj
                # Set the value of all other points in the slice
                inter_inds = ordered_inds[2:slast]
                inter_inds_up = ordered_inds[3:last]
                inter_inds_down = ordered_inds[1:slast-1]
                matrix[inter_inds, inter_inds_up] += 1/(2*step_size)*c.e*E/c.hbar_joule*xProj
                matrix[inter_inds, inter_inds_down] += -1 * 1/(2*step_size)*c.e*E/c.hbar_joule*xProj
            else:
                shortslice_inds.append(slice_inds-1)
        else:
            print('Field projection along kx is zero. kx-projection of FDM is zero.')

    # Second do ky projection of the FDM. Loop through unique kx and kz values in the Gamma valley.
    if yProj is not 0:
        print('Calculating projection of FDM along ky-axis.')
        for i in range(len(uniq_xz)):
            kx, kz = uniq_xz[i, 0], uniq_xz[i, 1]
            # Grab the "slice" of points in k space with the same kx and kz coordinate
            slice_df = g_df.loc[(g_df['kx [1/A]'] == kx) & (g_df['kz [1/A]'] == kz)]
            slice_inds = slice_df['k_inds'].values

            if len(slice_inds) > 3:
                # Subset is the slice sorted by ky value in ascending order. The index of subset still references kptdata.
                subset = slice_df.sort_values(by=['ky [1/A]'], ascending=True)
                ordered_inds = subset['k_inds'].values - 1  # indices of matrix (zero indexed)
                l_icinds.append(ordered_inds[0])  # +1 to get the k_inds values (one indexed)
                r_icinds.append(ordered_inds[-1])
                # Set the "initial condition" i.e. the point with the most negative ky value is treated as being zero
                # (and virtual point below)
                matrix[ordered_inds[0], ordered_inds[1]] += 1 / (2 * step_size) * c.e * E / c.hbar_joule * yProj
                matrix[ordered_inds[1], ordered_inds[2]] += 1 / (2 * step_size) * c.e * E / c.hbar_joule * yProj
                # Set the other "boundary condition" i.e. the point with the most positive ky value is treated as being zero
                # (and virtual point above)
                last = len(ordered_inds) - 1
                slast = len(ordered_inds) - 2
                matrix[ordered_inds[last], ordered_inds[slast]] += -1 * 1 / (
                            2 * step_size) * c.e * E / c.hbar_joule * yProj
                matrix[ordered_inds[slast], ordered_inds[slast - 1]] += -1 * 1 / (
                            2 * step_size) * c.e * E / c.hbar_joule * yProj
                # Set the value of all other points in the slice
                inter_inds = ordered_inds[2:slast]
                inter_inds_up = ordered_inds[3:last]
                inter_inds_down = ordered_inds[1:slast - 1]
                matrix[inter_inds, inter_inds_up] += 1 / (2 * step_size) * c.e * E / c.hbar_joule * yProj
                matrix[inter_inds, inter_inds_down] += -1 * 1 / (2 * step_size) * c.e * E / c.hbar_joule * yProj
            else:
                shortslice_inds.append(slice_inds-1)
        else:
            print('Field projection along ky is zero. ky-projection of FDM is zero.')

    # Second do ky projection of the FDM. Loop through unique kx and kz values in the Gamma valley.
    if zProj is not 0:
        print('Calculating projection of FDM along ky-axis.')
        for i in range(len(uniq_xy)):
            kx, ky = uniq_xy[i, 0], uniq_xy[i, 1]
            # Grab the "slice" of points in k space with the same kx and kz coordinate
            slice_df = g_df.loc[(g_df['kx [1/A]'] == kx) & (g_df['ky [1/A]'] == ky)]
            slice_inds = slice_df['k_inds'].values

            if len(slice_inds) > 3:
                # Subset is the slice sorted by kz value in ascending order. The index of subset still references kptdata.
                subset = slice_df.sort_values(by=['kz [1/A]'], ascending=True)
                ordered_inds = subset['k_inds'].values - 1  # indices of matrix (zero indexed)
                l_icinds.append(ordered_inds[0])  # +1 to get the k_inds values (one indexed)
                r_icinds.append(ordered_inds[-1])
                # Set the "initial condition" i.e. the point with the most negative ky value is treated as being zero
                # (and virtual point below)
                matrix[ordered_inds[0], ordered_inds[1]] += 1 / (2 * step_size) * c.e * E / c.hbar_joule * zProj
                matrix[ordered_inds[1], ordered_inds[2]] += 1 / (2 * step_size) * c.e * E / c.hbar_joule * zProj
                # Set the other "boundary condition" i.e. the point with the most positive kz value is treated as being zero
                # (and virtual point above)
                last = len(ordered_inds) - 1
                slast = len(ordered_inds) - 2
                matrix[ordered_inds[last], ordered_inds[slast]] += -1 * 1 / (
                        2 * step_size) * c.e * E / c.hbar_joule * zProj
                matrix[ordered_inds[slast], ordered_inds[slast - 1]] += -1 * 1 / (
                        2 * step_size) * c.e * E / c.hbar_joule * zProj
                # Set the value of all other points in the slice
                inter_inds = ordered_inds[2:slast]
                inter_inds_up = ordered_inds[3:last]
                inter_inds_down = ordered_inds[1:slast - 1]
                matrix[inter_inds, inter_inds_up] += 1 / (2 * step_size) * c.e * E / c.hbar_joule * zProj
                matrix[inter_inds, inter_inds_down] += -1 * 1 / (2 * step_size) * c.e * E / c.hbar_joule * zProj
            else:
                shortslice_inds.append(slice_inds-1)
        else:
            print('Field projection along kz is zero. ky-projection of FDM is zero.')

    # shortslice_inds = np.concatenate(shortslice_inds,axis=0)  # Change back
    print('Not applied to {:d} Gamma points because fewer than 4 points on the slice.'.format(len(shortslice_inds)))
    print('This represents {:1f} % of points in the Gamma valley.'.format(len(shortslice_inds)/len(g_df)*100))
    end = time.time()
    print('GaAs Gamma finite difference generation took {:.2f}s'.format(end - start))

    return matrix,shortslice_inds,l_icinds,r_icinds


if __name__ == '__main__':
    print(utilities.cartesian_projection([1,2,-2]))


    l = 5
    h = np.array(np.meshgrid(range(1,l+1), range(1,l+1), range(1,l+1))).T.reshape(-1, 3)
    # h = np.array(np.meshgrid(range(1,l+1), 1, 1)).T.reshape(-1, 3)
    d = {'kx [1/A]':h[:,0],'ky [1/A]':h[:,1],'kz [1/A]':h[:,2]}
    practice_df = pd.DataFrame(data=d).sort_values(by=['kx [1/A]','ky [1/A]','kz [1/A]'],ascending=True).reset_index(drop=True)
    practice_df['k_inds'] = np.array(range(1,len(practice_df)+1))
    nkpts = len(practice_df)
    fdm2 = np.memmap(pp.inputLoc + '/finite_difference_matrix4.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))

    m,_,_,_ = gaas_Gamma_FDM(fdm2,practice_df,1*c.hbar_joule / c.e * 0.0070675528500652425*2*1E10)

    fig = plt.figure()
    fig.set_size_inches(6,6)
    plt.matshow(m, fignum=1)
    plt.colorbar()
    plt.title('Column-sum preserving FDM, E = ({:.1f},{:.1f},{:.1f})'.format(pp.fieldDirection[0],pp.fieldDirection[1],pp.fieldDirection[2]))

    x = practice_df['kx [1/A]'].values
    y = practice_df['ky [1/A]'].values
    z = practice_df['kz [1/A]'].values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('kx [1/A]')
    ax.set_ylabel('ky [1/A]')
    ax.set_zlabel('kz [1/A]')
    plt.show()

    #
    # m1,m2 = np.nonzero(m)
    # row1_nonzero = np.nonzero(m[0,:])
    # row50_nonzero = np.nonzero(m[50,:])
    # row62_nonzero = np.nonzero(m[62,:])
    #
    # x0 = practice_df.loc[0,'kx [1/A]']
    # y0 = practice_df.loc[0,'ky [1/A]']
    # z0 = practice_df.loc[0,'kz [1/A]']
    #
    # xa = practice_df.loc[1,'kx [1/A]']
    # ya = practice_df.loc[1,'ky [1/A]']
    # za = practice_df.loc[1,'kz [1/A]']
    #
    # xb = practice_df.loc[5,'kx [1/A]']
    # yb = practice_df.loc[5,'ky [1/A]']
    # zb = practice_df.loc[5,'kz [1/A]']
    #
    # xc = practice_df.loc[25,'kx [1/A]']
    # yc = practice_df.loc[25,'ky [1/A]']
    # zc = practice_df.loc[25,'kz [1/A]']
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z,s=20)
    # ax.scatter(x0, y0, z0, c='black', s=45)
    # ax.scatter(xa, ya, za, c='r', s=45)
    # ax.scatter(xb, yb, zb, c='r', s=45)
    # ax.scatter(xc, yc, zc, c='r', s=45)
    # ax.set_xlabel('kx [1/A]')
    # ax.set_ylabel('ky [1/A]')
    # ax.set_zlabel('kz [1/A]')
    # plt.title('Exterior point')
    #
    # x0 = practice_df.loc[50,'kx [1/A]']
    # y0 = practice_df.loc[50,'ky [1/A]']
    # z0 = practice_df.loc[50,'kz [1/A]']
    #
    # xa = practice_df.loc[25,'kx [1/A]']
    # ya = practice_df.loc[25,'ky [1/A]']
    # za = practice_df.loc[25,'kz [1/A]']
    #
    # xb = practice_df.loc[51,'kx [1/A]']
    # yb = practice_df.loc[51,'ky [1/A]']
    # zb = practice_df.loc[51,'kz [1/A]']
    #
    # xc = practice_df.loc[55,'kx [1/A]']
    # yc = practice_df.loc[55,'ky [1/A]']
    # zc = practice_df.loc[55,'kz [1/A]']
    #
    # xd = practice_df.loc[75,'kx [1/A]']
    # yd = practice_df.loc[75,'ky [1/A]']
    # zd = practice_df.loc[75,'kz [1/A]']
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z,s=20)
    # ax.scatter(x0, y0, z0, c='black', s=45)
    # ax.scatter(xa, ya, za, c='r', s=45)
    # ax.scatter(xb, yb, zb, c='r', s=45)
    # ax.scatter(xc, yc, zc, c='r', s=45)
    # ax.scatter(xd, yd, zd, c='r', s=45)
    # ax.set_xlabel('kx [1/A]')
    # ax.set_ylabel('ky [1/A]')
    # ax.set_zlabel('kz [1/A]')
    # plt.title('Semi-interior point')
    #
    #
    #
    # x0 = practice_df.loc[62,'kx [1/A]']
    # y0 = practice_df.loc[62,'ky [1/A]']
    # z0 = practice_df.loc[62,'kz [1/A]']
    #
    # xa = practice_df.loc[37,'kx [1/A]']
    # ya = practice_df.loc[37,'ky [1/A]']
    # za = practice_df.loc[37,'kz [1/A]']
    #
    # xb = practice_df.loc[61,'kx [1/A]']
    # yb = practice_df.loc[61,'ky [1/A]']
    # zb = practice_df.loc[61,'kz [1/A]']
    #
    # xc = practice_df.loc[63,'kx [1/A]']
    # yc = practice_df.loc[63,'ky [1/A]']
    # zc = practice_df.loc[63,'kz [1/A]']
    #
    # xd = practice_df.loc[67,'kx [1/A]']
    # yd = practice_df.loc[67,'ky [1/A]']
    # zd = practice_df.loc[67,'kz [1/A]']
    #
    # xe = practice_df.loc[87,'kx [1/A]']
    # ye = practice_df.loc[87,'ky [1/A]']
    # ze = practice_df.loc[87,'kz [1/A]']
    #
    # xf = practice_df.loc[57,'kx [1/A]']
    # yf = practice_df.loc[57,'ky [1/A]']
    # zf = practice_df.loc[57,'kz [1/A]']
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z,s=20)
    # ax.scatter(x0, y0, z0, c='black', s=45)
    # ax.scatter(xa, ya, za, c='r', s=45)
    # ax.scatter(xb, yb, zb, c='r', s=45)
    # ax.scatter(xc, yc, zc, c='r', s=45)
    # ax.scatter(xd, yd, zd, c='r', s=45)
    # ax.scatter(xe, ye, ze, c='r', s=45)
    # ax.scatter(xf, yf, zf, c='r', s=45)
    #
    # ax.set_xlabel('kx [1/A]')
    # ax.set_ylabel('ky [1/A]')
    # ax.set_zlabel('kz [1/A]')
    # plt.title('Interior point')
    # plt.show()

    print('Hello')
    # d = {'kx [1/A]':[1,2,3,4],'ky [1/A]':[1,1,1,1],'kz [1/A]':[1,1,1,1],'k_inds':[1,2,3,4],'energy [eV]':[1,2,3,4]}
    # practice_df = pd.DataFrame(data=d)
    # nkpts = len(practice_df)
    # fdm2 = np.memmap(pp.inputLoc + '/finite_difference_matrix2.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))