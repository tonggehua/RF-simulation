

from math import ceil, pi
import numpy as np

from bloch.spingroup_ps import NumSolverSpinGroup

def make_DIGGER_waveform(b1,a,b,duration,rf_raster_time):
    # DIGGER RF Pulse for RUFIS implementation
    n = ceil(duration / rf_raster_time)
    t = np.arange(-duration/2, duration/2, n)
    b1_waveform = b1*(-1j)*(np.sin(pi*a*t)*np.sin(pi*b*t))/(pi*t)

    return b1_waveform, t


if __name__ == '__main__':
    T1 = 1
    T2 = 1e-3
    PD = 1

    # Freq offsets
    fs = np.arange(-5e3,5e3,5)

    # Assume that (a-b) is the central band width and b is the saturated bands' width
    bw_slice = 5e3   # Hz
    bw_sides = 10e3
    a = (bw_slice + bw_sides)/2
    b = bw_sides

    spins = [NumSolverSpinGroup(loc=(0,0,0), pdt1t2=(1,0,0), df=f) for f in fs]


    for spin in spins:
        spin.apply_rf_store()
