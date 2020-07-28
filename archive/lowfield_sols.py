import numpy as np
import time
import constants as c
import problem_parameters as pp
import numpy.linalg
from scipy.sparse import linalg
import preprocessing
import utilities


class gmres_counter(object):
    """A class object that can be called during GMRES to print stepwise iterative residual."""
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


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
    print('Starting steady BTE low-field occupancy solver')
    if pp.scmBool:
        scmfac = pp.scmVal
        print('Applying correction factor to the scattering matrix.')
    else:
        scmfac = 1
        print('Not applying correction factor to the scattering matrix.')
    loopstart = time.time()
    invdiag = (np.diag(scm) * scmfac) ** (-1)
    b = (-1) * np.squeeze(df['vx [m/s]'] * df['k_FD']) * (1 - df['k_FD'])
    f_0 = b * invdiag
    f_next, criteria = linalg.gmres(scm*scmfac, b, x0=f_0, tol=pp.relConvergence, atol=pp.absConvergence,
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
        print('Error not stored.')
    loopend = time.time()
    print('Convergence took {:.2f}s'.format(loopend - loopstart))
    if pp.simpleBool:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in simple linearization')
        chi2psi = np.squeeze(df['k_FD'] * (1 - df['k_FD']))
        f_next = f_next / chi2psi
        f_0 = f_0 / chi2psi
    return f_next, f_0, error, counter.niter


def f2chi(f, df, field):
    """Convert F_k from low field approximation iterative scheme into chi which is easy to plot.
    Since the solution we obtain from cg and from iterative scheme is F_k where chi_k = eE/kT * f0(1-f0) * F_k
    then we need to bring these factors back in to get the right units
    Parameters:
        f (nparray): Numpy array containing a solution of the steady Boltzmann equation in f form.
        df (dataframe): Electron DataFrame indexed by kpt containing the group velocity associated with each state in eV.
        field (dbl): Value of the electric field in V/m.
    Returns:
        chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in chi form.
    """
    f0 = np.squeeze(df['k_FD'].values)
    prefactor = field * c.e / c.kb_joule / pp.T * f0 * (1 - f0)
    # prefactor = field * c.e / c.kb_joule / pp.T * f0
    chi = np.squeeze(f) * np.squeeze(prefactor)
    return chi


def write_steady(df):
    """Calls the GMRES solver hard coded for solving the BTE with full FDM and writes the chis to file. Also calculates
    low-field and RTA chis.
    Parameters:
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
    Returns:
        None. Just writes the chi solutions to file. chi_#_EField_freq. #1 corresponds to RTA, #2 corresponds to
        low-field, #3 corresponds to full finite-difference.
    """
    nkpts = len(np.unique(df['k_inds']))
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    # The steady low-field solution is returned as F from Wu Li's paper.
    f_next, f_0,_,_ = steady_low_field(df, scm)
    np.save(pp.outputLoc + 'Steady/' + 'f_1',f_0)
    np.save(pp.outputLoc + 'Steady/' + 'f_2',f_next)


def calc_mobility(F, df):
    """Calculate mobility as per Wu Li PRB 92, 2015. Solution must be fed in as F with no field provided.
    Parameters:
        F (nparray): Numpy array containing a solution of the steady Boltzmann equation in F form or as psi
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        mobility (double): The value of the mobility carrier energy in m^2/V-s.
    """
    Nuc = pp.kgrid ** 3
    print('Field not specified. Mobility calculated using linear in E formula.')
    prefactor = 2 * c.e ** 2 / (c.Vuc * c.kb_joule * pp.T * Nuc)
    conductivity = prefactor * np.sum(df['k_FD'] * (1 - df['k_FD']) * df['vx [m/s]'] * F)
    n = utilities.calculate_density(df)
    mobility = conductivity / c.e / n
    # print('Carrier density is {:.8E}'.format(n * 1E-6) + ' per cm^{-3}')
    print('Mobility is {:.10E} (cm^2 / V / s)'.format(mobility * 1E4))
    return mobility


if __name__ == '__main__':
    # Create electron and phonon dataframes
    # preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)
    fields = pp.fieldVector

    # writeSteady = True
    # if writeSteady:
    #     write_steady(fields, electron_df)

    f_2 = np.load(pp.outputLoc + 'Steady/' + 'f_2.npy')
    calc_mobility(f_2, electron_df)
