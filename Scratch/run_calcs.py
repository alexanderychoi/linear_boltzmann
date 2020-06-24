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
import correlation_solver

if __name__ == '__main__':
    # Create electron and phonon dataframes
    preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)

    fields = pp.fieldVector
    freq = pp.freqGHz

    writeTransient = True
    writeSteady = True
    occupation_solver.write_icinds(electron_df)
    if writeTransient:
        occupation_solver.write_transient(fields, electron_df, freq)
    if writeSteady:
        occupation_solver.write_steady(fields, electron_df)

    writeIntervalley = True
    writeThermal = True
    if pp.getX:
        if writeThermal:
            correlation_solver.write_vd_vd(fields, electron_df, freq)
        if writeIntervalley:
            correlation_solver.write_ngl_nlx(fields, electron_df, freq)
    else:
        if writeIntervalley:
            correlation_solver.write_ng_ng(fields, electron_df, freq)
        if writeThermal:
            correlation_solver.write_vd_vd(fields, electron_df, freq)

    freq = 1
    writeTransient = True
    writeSteady = True
    occupation_solver.write_icinds(electron_df)
    if writeTransient:
        occupation_solver.write_transient(fields, electron_df, freq)
    if writeSteady:
        occupation_solver.write_steady(fields, electron_df)

    writeIntervalley = True
    writeThermal = True
    if pp.getX:
        if writeThermal:
            correlation_solver.write_vd_vd(fields, electron_df, freq)
        if writeIntervalley:
            correlation_solver.write_ngl_nlx(fields, electron_df, freq)
    else:
        if writeIntervalley:
            correlation_solver.write_ng_ng(fields, electron_df, freq)
        if writeThermal:
            correlation_solver.write_vd_vd(fields, electron_df, freq)

    plt.show()