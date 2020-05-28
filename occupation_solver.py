import numpy as np
import pandas as pd
import time
import constants as c
import utilities
import problemparameters as pp
import matplotlib.pyplot as plt
import numpy.linalg
from scipy.sparse import linalg
import occupation_plotter
import preprocessing


class gmres_counter(object):
    """A class object that can be called during GMRES to give stepwise iterative residual."""
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


def apply_centraldiff_matrix(matrix,fullkpts_df,E,step_size=1):
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

    if pp.gD == 160:
        step_size = 0.0070675528500652425 * 1E10  # 1/Angstrom to 1/m (for 160^3)
    if pp.gD == 200:
        step_size = 0.005654047459752398 * 1E10  # 1/Angstrom to 1/m (for 200^3)
    if pp.gD == 80:
        step_size = 0.0070675528500652425*2*1E10 # 1/Angstron for 1/m (for 80^3)

    kptdata = fullkpts_df[['k_inds', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]']]

    g_inds,l_inds,x_inds=utilities.split_valleys(fullkpts_df,False)
    g_df = kptdata.loc[g_inds]
    uniq_yz = np.unique(g_df[['ky [1/A]', 'kz [1/A]']].values, axis=0)
    # If there are too few points in a slice < 5, we want to keep track of those points
    shortslice_inds = []
    l_icinds = []
    r_icinds = []
    lvalley_inds = []
    start = time.time()
    # Loop through the unique ky and kz values
    for i in range(len(uniq_yz)):
        kind = i + 1
        ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
        # Grab the "slice" of points in k space with the same ky and kz coordinate
        slice_df = g_df.loc[(g_df['ky [1/A]'] == ky) & (g_df['kz [1/A]'] == kz)]
        slice_inds = slice_df['k_inds'].values
        # if 0 in slice_inds or 1 in slice_inds or len(kptdata) in slice_inds or len(kptdata)-1 in slice_inds:
        #     lastslice_inds.append(slice_inds)
        #     continue
        # Skip all slices that intersect an L valley. Save the L valley indices

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
    print('Finite difference not applied to L valleys. Derivative treated as zero for these points.')
    return shortslice_inds, np.array(l_icinds), np.array(r_icinds), lvalley_inds, matrix


def transient_full_drift(matrix_sc, matrix_fd, kptdf, field, freq):
    """Generalized minimal residual solver for calculating transient BTE solution in the form of Chi using the full
    finite difference matrix.
    Parameters:
        matrix_sc (memmap): Scattering matrix in simple linearization by default..
        matrix_fd (memmap): Finite difference matrix, generated by apply_centraldiff_matrix.
        kptdf (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        field (dbl): Value of the electric field in V/m.
        freq (dbl): Frequency in GHz to be used for the transient solution
    Returns:
        x_next (nparray): Numpy array containing the (hopefully) converged iterative solution as chi.
        x_0 (nparray): Numpy array containing the RTA solution as chi.
        icinds_l (nparray): Numpy array containing the left side indices where the initial condition is applied.
        icinds_r (nparray): Numpy array containing the right side indices where the initial condition is applied.
        counter.niter (dbl): Number of iterations to reach desired relative convergence
    """
    counter = gmres_counter()
    print('Starting transient BTE solver for {:.3E} and {:.3E} GHz'.format(field,freq))
    if pp.scmBool:
        scmfac = pp.scmVal
        print('Applying correction factor to the scattering matrix.')
    else:
        scmfac = 1
    _, icinds_l,icinds_r, _, matrix_fd = apply_centraldiff_matrix(matrix_fd, kptdf, field)

    loopstart = time.time()
    b = (-1)*c.e*field/c.kb_joule/pp.T * np.squeeze(kptdf['vx [m/s]'] * kptdf['k_FD']) * (1 - kptdf['k_FD'])
    chi2psi = np.squeeze(kptdf['k_FD'] * (1 - kptdf['k_FD']))
    invdiag = (np.diag(matrix_sc) * scmfac) ** (-1)
    x_0 = b * invdiag
    freq_matrix = np.diag(np.ones(len(kptdf))*1j*10**9*-1*2*np.pi*freq)
    x_next, criteria = linalg.gmres(freq_matrix+matrix_sc*scmfac-matrix_fd, b,x0=x_0,tol=pp.relConvergence,callback=counter,
                                    atol=pp.absConvergence)
    print('GMRES convergence criteria: {:3E}'.format(criteria))
    # The following step is the calculation of the relative residual, which involves another MVP. This adds expense. If
    # we're confident in the convergence, we can omit this check to increase speed.
    if pp.verboseError:
        b_check = np.dot(freq_matrix+matrix_sc*scmfac-matrix_fd,x_next)
        error = np.linalg.norm(b_check - b)/np.linalg.norm(b)
        print('Norm of b is {:3E}'.format(np.linalg.norm(b)))
        print('Absolute residual error is {:3E}'.format(np.linalg.norm(b_check-b)))
        print('Relative residual error is {:3E}'.format(error))
    else:
        error = 0

    loopend = time.time()
    print('Convergence took {:.2f}s'.format(loopend - loopstart))
    if not pp.simpleBool:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in canonical linearization')
        x_next = x_next * chi2psi
        x_0 = x_0 * chi2psi
    return x_next, x_0, error, icinds_l, icinds_r, counter.niter


def steady_low_field(df, scm):
    """GMRES solver hard coded for solving the BTE in low field approximation. Sign convention by Wu Li. Returns f,
    which is equal to chi/(eE/kT).
    Parameters:
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        scm (memmap): Memory-mapped array containing the scattering matrix, assumed simple linearization by default.
    Returns:
        f_next (nparray): Numpy array containing the (hopefully) converged GMRES solution as psi/(eE/kT).
        f_0 (nparray): Numpy array containing the RTA solution as psi_0/(eE/kT).
    """
    counter = gmres_counter()
    print('Starting steady low-field BTE solver')
    if pp.scmBool:
        scmfac = pp.scmVal
        print('Applying correction factor to the scattering matrix.')
    else:
        scmfac = 1
    loopstart = time.time()
    b = (-1)* np.squeeze(df['vx [m/s]'])
    # chi2psi is used to give the finite difference matrix the right factors in front since substitution made
    chi2psi = np.squeeze(df['k_FD'] * (1 - df['k_FD']))
    invdiag = (np.diag(scm) * scmfac) ** (-1)
    f_0 = b * invdiag
    f_next, criteria = linalg.gmres(scm*scmfac, b,x0=f_0,tol=pp.relConvergence, atol=pp.absConvergence,
                                    callback=counter)
    print('GMRES convergence criteria: {:3E}'.format(criteria))
    if pp.verboseError:
        b_check = np.dot(scm*scmfac,f_next)
        error = np.linalg.norm(b_check - b)/np.linalg.norm(b)
        print('Norm of b is {:3E}'.format(np.linalg.norm(b)))
        print('Absolute residual error is {:3E}'.format(np.linalg.norm(b_check-b)))
        print('Relative residual error is {:3E}'.format(error))
    else:
        error = 0
    loopend = time.time()
    print('Convergence took {:.2f}s'.format(loopend - loopstart))
    if not pp.simpleBool:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in canonical linearization')
        f_next = f_next * chi2psi
        f_0 = f_0 * chi2psi
    return f_next, f_0, error, counter.niter


def steady_full_drift(matrix_sc, matrix_fd, kptdf, field):
    """Generalized minimal residual solver for calculating transient BTE solution in the form of Chi using the full
    finite difference matrix.
    Parameters:
        matrix_sc (memmap): Scattering matrix in simple linearization by default..
        matrix_fd (memmap): Finite difference matrix, generated by apply_centraldiff_matrix.
        kptdf (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        field (dbl): Value of the electric field in V/m.
    Returns:
        x_next (nparray): Numpy array containing the (hopefully) converged iterative solution as chi.
        x_0 (nparray): Numpy array containing the RTA solution as chi.
        icinds_l (nparray): Numpy array containing the left side indices where the initial condition is applied.
        icinds_r (nparray): Numpy array containing the right side indices where the initial condition is applied.
        counter.niter (dbl): Number of iterations to reach desired relative convergence
    """
    counter = gmres_counter()
    print('Starting steady BTE solver for {:.3E}'.format(field))
    if pp.scmBool:
        scmfac = pp.scmVal
        print('Applying correction factor to the scattering matrix.')
    else:
        scmfac = 1
    _, icinds_l,icinds_r, _, matrix_fd = apply_centraldiff_matrix(matrix_fd, kptdf, field)

    loopstart = time.time()
    b = (-1)*c.e*field/c.kb_joule/pp.T * np.squeeze(kptdf['vx [m/s]'] * kptdf['k_FD']) * (1 - kptdf['k_FD'])
    chi2psi = np.squeeze(kptdf['k_FD'] * (1 - kptdf['k_FD']))
    invdiag = (np.diag(matrix_sc) * scmfac) ** (-1)
    x_0 = b * invdiag
    x_next, criteria = linalg.gmres(matrix_sc*scmfac-matrix_fd, b,x0=x_0,tol=pp.relConvergence,callback=counter,
                                    atol=pp.absConvergence)
    print('GMRES convergence criteria: {:3E}'.format(criteria))
    # The following step is the calculation of the relative residual, which involves another MVP. This adds expense. If
    # we're confident in the convergence, we can omit this check to increase speed.
    if pp.verboseError:
        b_check = np.dot(matrix_sc*scmfac-matrix_fd,x_next)
        error = np.linalg.norm(b_check - b)/np.linalg.norm(b)
        print('Norm of b is {:3E}'.format(np.linalg.norm(b)))
        print('Absolute residual error is {:3E}'.format(np.linalg.norm(b_check-b)))
        print('Relative residual error is {:3E}'.format(error))
    else:
        error = 0
    loopend = time.time()
    print('Convergence took {:.2f}s'.format(loopend - loopstart))
    if not pp.simpleBool:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in canonical linearization')
        x_next = x_next * chi2psi
        x_0 = x_0 * chi2psi
    return x_next, x_0, error, icinds_l, icinds_r, counter.niter


def write_steady(fieldVector, df):
    """Calls the GMRES solver hard coded for solving the BTE with full FDM and writes the chis to file. Also calculates
    low-field and RTA chis.
    Parameters:
        outLoc (str): String containing the location of the directory to write the chi solutions.
        inLoc (str): String containing the location of the directory containing the scattering matrix, assumed simple
        linearization by default.
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
    Returns:
        None. Just writes the chi solutions to file. chi_#_EField_freq. #1 corresponds to RTA, #2 corresponds to
        low-field, #3 corresponds to full finite-difference iterative.
    """
    nkpts = len(np.unique(df['k_inds']))
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    error = []
    iteration_count = []
    f_next, f_0,_,_ = steady_low_field(df, scm)
    np.save(pp.outputLoc + 'Steady/' + 'f_1',f_0)
    np.save(pp.outputLoc + 'Steady/' + 'f_2',f_next)

    for i in range(len(fieldVector)):
        fdm = np.memmap(pp.inputLoc + '/finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
        EField = fieldVector[i]
        x_next, _, temp_error, icinds_l, icinds_r, iterations = steady_full_drift(scm, fdm, df, EField)
        error.append(temp_error)
        iteration_count.append(iterations)
        del fdm
        np.save(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}".format(EField), x_next)
        np.save(pp.outputLoc + 'Steady/' + 'chi_' + '2_' + "E_{:.1e}".format(EField), utilities.f2chi(f_next,df,EField))
        np.save(pp.outputLoc + 'Steady/' + 'chi_' + '1_' + "E_{:.1e}".format(EField), utilities.f2chi(f_0,df, EField))

        print('Steady solution written to file for ' + "{:.1e} V/m ".format(EField))
        print('\n \n')
    np.save(pp.outputLoc + 'left_icinds', icinds_l)
    np.save(pp.outputLoc + 'right_icinds', icinds_r)

    plt.figure()
    plt.plot(fieldVector*1E-5,error)
    plt.xlabel('EField (kV/cm)')
    plt.ylabel(r'$|Ax_{f}-b|/|b|$')
    plt.title('Steady Occupation' + pp.title_str)

    plt.figure()
    plt.plot(fieldVector*1E-5, iteration_count)
    plt.xlabel('EField (kV/cm)')
    plt.ylabel('Iterations to convergence')
    plt.title('Steady Occupation' + pp.title_str)


def write_transient(fieldVector, df, freq):
    """Calls the GMRES solver hard coded for solving the BTE with full FDM and writes the chis to file.
    Parameters:
        outLoc (str): String containing the location of the directory to write the chi solutions.
        inLoc (str): String containing the location of the directory containing the scattering matrix, assumed simple
        linearization by default.
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        freq (dbl): Frequency in GHz to be used for the transient solution
    Returns:
        None. Just writes the chi solutions to file. chi_#_EField_freq. #1 corresponds to RTA, #2 corresponds to
        low-field, #3 corresponds to full finite-difference iterative.
    """
    nkpts = len(np.unique(df['k_inds']))
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    error = []
    iteration_count =[]
    for i in range(len(fieldVector)):
        fdm = np.memmap(pp.inputLoc + '/finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
        EField = fieldVector[i]
        x_next, _, temp_error, icinds_l, icinds_r, iterations = transient_full_drift(scm, fdm, df, EField, freq)
        error.append(temp_error)
        iteration_count.append(iterations)
        del fdm
        np.save(pp.outputLoc + 'Transient/' + 'chi_' + '3_' + "f_{:.1e}_E_{:.1e}".format(freq,EField), x_next)
        print('Transient solution written to file for ' + "{:.1e} V/m and {:.1e} GHz".format(EField,freq))
        print('\n \n')
    np.save(pp.outputLoc + 'left_icinds', icinds_l)
    np.save(pp.outputLoc + 'right_icinds', icinds_r)

    plt.figure()
    plt.plot(fieldVector*1E-5,error)
    plt.xlabel('EField (kV/cm)')
    plt.ylabel(r'$|Ax_{f}-b|/|b|$')
    plt.title('Occupation {:.1e} GHz'.format(freq) + pp.title_str)

    plt.figure()
    plt.plot(fieldVector*1E-5, iteration_count)
    plt.xlabel('EField (kV/cm)')
    plt.ylabel('Iterations to convergence')
    plt.title('Occupation {:.1e} GHz'.format(freq) + pp.title_str)


if __name__ == '__main__':
    # Create electron and phonon dataframes
    # preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)

    fields = pp.fieldVector
    freq = pp.freqGHz

    writeTransient = True
    writeSteady = True

    if writeTransient:
        write_transient(fields, electron_df, freq)
    if writeSteady:
        write_steady(fields, electron_df)

    occupation_plotter.bz_3dscatter(electron_df,True,True)
    plt.show()