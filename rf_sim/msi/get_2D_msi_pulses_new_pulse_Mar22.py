# Create a sequence with only one 90 pulse, a TE delay, and one 180 pulse per slice/bin
# For simulating the spatial-spectral response of current MSI implementation
from rf_sim.msi.write_2D_MSI_sim_use import *
import numpy as np
# Sequence TSE-based
# 1. Initialize ----------------------------------------------------------------
system = set_June_system_limits()
seq = Sequence(system)
## System limits & setup
# Set system limits
ramp_time = 250e-6 # Ramp up/down time for all gradients where this is specified
# Set the gyromagnetic ratio constant for protons
GAMMA_BAR = 42.5775e6
# Set slice locations and bins
FOV = 128e-3
thk = 5e-3
gap = 5e-3
n_slices = 1
displacement = 0
L = (n_slices - 1) * (thk+gap)
#displacement = -9.2e-3
sl_locs = displacement + np.arange(-L/2, L/2 + thk + gap, thk + gap)

# Exp 2
#N_bins = 9
#bin_width = 2400 / N_bins

# Exp 1
N_bins = 19
bin_width = 800

bin_centers = bin_width * (np.arange(N_bins) - (N_bins - 1) / 2)

print('Slice locations: ', sl_locs)
# Set contrast params
TE = 500e-3

# 2. Calculate derived parameters -----------------------------------------------
dk = 1/FOV
N = 256
k_width = N * dk
# Flip angles
fa_exc = 90  # degrees
fa_ref = 180 # degrees
###  Readout duration
readout_time = 6.4e-3 + 2 * system.adc_dead_time
### Excitation pulse duration
t_ex = 2.5e-3
t_exwd = t_ex + system.rf_ringdown_time + system.rf_dead_time
### Refocusing pulse duration
t_ref = 2e-3
t_refwd = t_ref + system.rf_ringdown_time + system.rf_dead_time
### Time gaps for spoilers
t_sp = 0.5 * (TE - readout_time - t_refwd)  # time gap between pi-pulse and readout
t_spex = 0.5 * (TE - t_exwd - t_refwd)  # time gap between pi/2-pulse and pi-pulse
### Spoiling factors
fsp_r = 1  # in readout direction per refocusing
fsp_s = 0.5  # in slice direction per refocusing

# 3. Make sequence components ------------------------------------------------------
## RF pulses
rfbw = bin_width
### 90 deg pulse (+y')
rf_ex_phase = np.pi / 2
flip_ex = fa_exc * np.pi / 180
rf_ex, g_ss, _ = make_sinc_pulse(flip_angle=flip_ex, system=system, duration=t_ex, slice_thickness=thk,
                                 apodization=0.5, time_bw_product=t_ex*rfbw, phase_offset=rf_ex_phase, return_gz=True)
gs_ex = make_trapezoid(channel='z', system=system, amplitude= -g_ss.amplitude, flat_time=t_exwd,
                       rise_time=ramp_time)# Note that the 90-deg SS gradient's amplitude is reversed for 2D MSI purposes!!

### 180 deg pulse (+x')
rf_ref_phase = 0
flip_ref = fa_ref * np.pi / 180
rf_ref, gz, _ = make_sinc_pulse(flip_angle=flip_ref, system=system, duration=t_ref, slice_thickness=thk,
                                apodization=0.5, time_bw_product=t_ref*rfbw, phase_offset=rf_ref_phase, use='refocusing',
                                return_gz=True)
gs_ref = make_trapezoid(channel='z', system=system, amplitude=g_ss.amplitude, flat_time=t_refwd,
                        rise_time=ramp_time)


delay1 = make_delay(TE - calc_duration(rf_ex)/2 - calc_duration(rf_ref)/2)
delay2 = make_delay(TE - calc_duration(rf_ref)/2)

# Add building blocks to sequence
# Add a delay!
for q in range(N_bins):
    for s in range(n_slices):  # For each slice (multislice)
        offset_90, offset_180 = calculate_MSI_rf_params(z0=sl_locs[s], dz=thk, f0=bin_centers[q],
                                                        df=bin_width)

        # Modify the pulse frequencies
        rf_ex.freq_offset = offset_90
        rf_ex.phase_offset = rf_ex_phase - 2 * np.pi * rf_ex.freq_offset * calc_rf_center(rf_ex)[0]
        rf_ref.freq_offset = offset_180
        rf_ref.phase_offset = rf_ref_phase - 2 * np.pi * rf_ref.freq_offset * calc_rf_center(rf_ref)[0]

        # Add blocks - no encoding, only RF
        seq.add_block(rf_ex, gs_ex)
        seq.add_block(delay1)
        seq.add_block(rf_ref, gs_ref)
        seq.add_block(delay2)


#seq.plot(time_range=[0,100e-3])
seq.write(f'2DMSI.seq')