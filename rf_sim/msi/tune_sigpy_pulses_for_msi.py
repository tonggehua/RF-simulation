from rf_sim.rf_simulations import simulate_rf
import sigpy.mri.rf as rf_ext
import numpy as np
from pypulseq.opts import Opts
import rf_sim.msi.sigpy2pulseq as sp
GAMMA_BAR = 42.58e6
from rf_sim.msi.msi_sim_make_rf_seqs import make_rf_profile_plot

def sim_90():
    system = Opts()
    rfbw = 500
    thk = 5e-3

    # 90
    rf_ex_phase = np.pi/2
    tb = 4
    t_ex = tb / rfbw
    pulse = rf_ext.slr.dzrf(n=int(round(t_ex / system.rf_raster_time)), tb=tb, ptype='st', ftype='ls',
                            d1=0.01, d2=0.01, cancel_alpha_phs=True)
    rf_ex, g_ss, _, _ = sp.sig_2_seq(pulse=pulse, flip_angle=np.pi / 2, system=system, duration=t_ex,
                                     slice_thickness=thk, phase_offset=rf_ex_phase, return_gz=True,
                                     time_bw_product=tb)
    # Simulate 90
    rf_ex_dt = rf_ex.t[1] - rf_ex.t[0]
    bw = 2*rfbw
    signals90, m90 = simulate_rf(bw_spins=bw, n_spins=200, pdt1t2=(1,0,0), flip_angle=90, dt=rf_ex_dt,
                       solver="RK45",
                       pulse_type='custom', pulse_shape=rf_ex.signal/GAMMA_BAR, display=False)
    # Display
    make_rf_profile_plot(bw,m90)

def sim_180():
    # Make a sigpy pulse - customize the parameters
    system = Opts()
    rfbw = 500
    thk = 5e-3

    # 180
    rf_ref_phase = 0
    tb = 4
    t_ref = tb / rfbw
    pulse = rf_ext.slr.dzrf(n=int(round(t_ref / system.rf_raster_time)), tb=tb, ptype='se', ftype='ls',
                            d1=0.01, d2=0.01, cancel_alpha_phs=False)
    rf_ref, gz, gzr, _ = sp.sig_2_seq(pulse=pulse, flip_angle=np.pi, system=system, duration=t_ref,
                                      slice_thickness=thk, phase_offset=rf_ref_phase, use='refocusing',
                                      return_gz=True, time_bw_product=tb)

    # Simulate 180
    rf_ref_dt = rf_ref.t[1] - rf_ref.t[0]
    bw = 2 * rfbw
    signals180, m180 = simulate_rf(bw_spins=bw, n_spins=200, pdt1t2=(1, 0, 0), flip_angle=180, dt=rf_ref_dt,
                                   solver="RK45",
                                   pulse_type='custom', pulse_shape=rf_ref.signal / GAMMA_BAR, display=False)

    make_rf_profile_plot(bw, m180)
if __name__ == '__main__':
    sim_180()