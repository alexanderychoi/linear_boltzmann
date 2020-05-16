import numpy as np
import problemparameters as pp
import constants as c
import os
import pandas as pd
import utilities
import matrix_plotter


def load_electron_df(inLoc):
    """Loads the electron dataframe from the .VEL output from Perturbo, transforms into cartesian coordinates, and
    translates the points back into the FBZ.
    Parameters:
        inLoc (str): String containing the location of the directory containing the input text file.
    Returns:
        None. Just prints the values of the problem parameters.
    """

    os.chdir(inLoc)
    # kvel = np.loadtxt('gaas.vel', skiprows=3)
    # kvel_df = pd.DataFrame(data=kvel,
    #                        columns=['k_inds', 'bands', 'energy', 'kx [2pi/alat]', 'ky [2pi/alat]', 'kz [2pi/alat]',
    #                                 'vx_dir', 'vy_dir', 'vz_dir', 'v_mag [m/s]'])
    kvel_df = pd.read_parquet(inLoc+'gaas_full_electron_data.parquet',engine='pyarrow')
    kvel_df[['k_inds']] = kvel_df[['k_inds']].astype(int)
    cart_kpts = kvel_df.copy(deep=True)
    # cart_kpts['kx [2pi/alat]'] = cart_kpts['kx [2pi/alat]'].values * 2 * np.pi / c.a
    # cart_kpts['ky [2pi/alat]'] = cart_kpts['ky [2pi/alat]'].values * 2 * np.pi / c.a
    # cart_kpts['kz [2pi/alat]'] = cart_kpts['kz [2pi/alat]'].values * 2 * np.pi / c.a
    # cart_kpts.columns = ['k_inds', 'bands', 'energy', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]', 'vx_dir', 'vy_dir',
    #                      'vz_dir', 'v_mag [m/s]']
    cart_kpts['vx [m/s]'] = np.multiply(cart_kpts['vx_dir'].values, cart_kpts['v_mag [m/s]'])
    # cart_kpts = cart_kpts.drop(['bands'], axis=1)
    cart_kpts = cart_kpts.drop(['vx_dir', 'vy_dir', 'vz_dir'], axis=1)

    cart_kpts['FD'] = (np.exp((cart_kpts['energy [eV]'].values * c.e - pp.mu * c.e)
                              / (c.kb_joule * pp.T)) + 1) ** (-1)
    reciplattvecs = np.concatenate((c.b1[np.newaxis, :], c.b2[np.newaxis, :], c.b3[np.newaxis, :]), axis=0)
    vector_df = cart_kpts.values[:, 6:9]
    fbzcartkpts, delta_kx = utilities.translate_into_fbz(cart_kpts.values[:, 6:9], reciplattvecs)

    fbzcartkpts = pd.DataFrame(data=fbzcartkpts, columns=['kx [1/A]', 'ky [1/A]', 'kz [1/A]'])
    fbzcartkpts = pd.concat([cart_kpts[['k_inds', 'vx [m/s]', 'energy [eV]','v_mag [m/s]']], fbzcartkpts], axis=1)
    fbzcartkpts = fbzcartkpts.rename(columns={"energy [eV]": "energy"})
    fbzcartkpts.to_pickle(inLoc + 'electron_df.pkl')
    print('Wrote electron DF')


if __name__ == '__main__':
    out_Loc = pp.outputLoc
    in_Loc = pp.inputLoc
    load_electron_df(in_Loc)