# Replicate the 3 figures of 2D MSI paper in simulation
import bloch.phantom as pht
import numpy as np
from pypulseq.Sequence.sequence import Sequence
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_gauss_pulse import make_gauss_pulse
from rf_sim.msi.write_2D_MSI_sim_use import *
import sigpy.mri.rf as rf_ext
import rf_sim.msi.sigpy2pulseq as sp

# Create shared phantom in 2D - z (slice),x (read) - stand-in


# Make sequence commands (only RF and if needed, slice gradients. No readout or phase gradients)
# 1. MAVRIC
# What type of RF?
# How many bins?
def make_mavric_RF_seq(nbins=5, bw=1500, TR=2):
    seq = Sequence()
    bw_bin = bw/nbins
    offsets = get_rf_freq_offsets(nbins,bin_sep=bw_bin)
    rf = make_block_pulse(flip_angle=np.pi / 2, duration=2.5e-3, freq_offset=0)
    adc_single = make_adc(num_samples=1, dwell=10e-6, freq_offset=0, phase_offset=0)
    gro = make_trapezoid(channel='x', area=0, flat_time=calc_duration(adc_single))
    delayTR = make_delay(TR - calc_duration(rf)-calc_duration(adc_single))

    for b in range(nbins):
        rf.freq_offset = offsets[b]
        seq.add_block(rf)
        seq.add_block(delayTR)
        seq.add_block(adc_single, gro)

    return seq

# 2. SEMAC
# What type of RF? - Gauss, + gradient
# Number of slices - consider overlap; display the slice overlap as a figure to help understanding the output
#
def make_semac_RF_seq(nbins=5, bw=1500, slab_thk=100e-3):
    seq = Sequence()
    return seq

# 3. 2D MSI

def make_2dmsi_RF_seq(TE=500e-3, nbins=1, n_slices=1, thk=5e-3, gap=5e-3, bw=500,
                      use_sigpy_90=False, use_sigpy_180=False):
    ### Setup ###
    system = Opts()
    #set_June_system_limits()
    ramp_time = 250e-6
    # Slices
    displacement = 0
    sl_locs = get_slice_locations(n_slices, thk, gap, displacement)
    # Bins
    bin_width = bw/nbins
    bin_centers = get_rf_freq_offsets(nbins,bin_sep=bin_width)
    # RF parameters
    t_ex = 2.5e-3
    t_ref = 2e-3
    rfbw = bin_width
    rf_ex_phase = np.pi / 2
    rf_ref_phase = 0


    ### RF pulses ###
    # 90 deg pulse (+y')
    if use_sigpy_90:
        tb = 4
        t_ex = tb / rfbw
        pulse = rf_ext.slr.dzrf(n=int(round(t_ex/system.rf_raster_time)), tb=tb, ptype='st', ftype='ls',
                                d1=0.01, d2=0.01, cancel_alpha_phs=True)
        rf_ex, g_ss, _, _ = sp.sig_2_seq(pulse=pulse, flip_angle=np.pi/2,system=system,duration=t_ex,
                                         slice_thickness=thk, phase_offset=rf_ex_phase,return_gz=True,
                                         time_bw_product=tb)
    else:
        rf_ex, g_ss, _ = make_sinc_pulse(flip_angle=np.pi/2, system=system, duration=t_ex, slice_thickness=thk,
                                         apodization=0.5, time_bw_product=t_ex * rfbw, phase_offset=rf_ex_phase,
                                         return_gz=True)

    # 180 deg pulse (+x')
    if use_sigpy_180:
        tb = 4
        t_ref = tb / rfbw
        pulse = rf_ext.slr.dzrf(n=int(round(t_ref / system.rf_raster_time)), tb=tb, ptype='st', ftype='ls',
                                d1=0.01, d2=0.01, cancel_alpha_phs=True)
        rf_ref, gz, gzr, _ = sp.sig_2_seq(pulse=pulse, flip_angle=np.pi, system=system, duration=t_ref,
                                      slice_thickness=thk, phase_offset=rf_ref_phase,use='refocusing',
                                      return_gz=True, time_bw_product=tb)
    else:
        rf_ref, gz, _ = make_sinc_pulse(flip_angle=np.pi, system=system, duration=t_ref, slice_thickness=thk,
                                        apodization=0.5, time_bw_product=t_ref * rfbw, phase_offset=rf_ref_phase,
                                        use='refocusing',
                                        return_gz=True)

    # Gradients
    #t_exwd = t_ex + system.rf_ringdown_time + system.rf_dead_time
    #t_refwd = t_ref + system.rf_ringdown_time + system.rf_dead_time
    t_exwd = calc_duration(rf_ex)
    t_refwd = calc_duration(rf_ref)
    # 90-deg slice selective
    gs_ex = make_trapezoid(channel='z', system=system, amplitude=-g_ss.amplitude, flat_time=t_exwd,
                           rise_time=ramp_time)  # Note that the 90-deg SS gradient's amplitude is reversed for 2D MSI purposes!!
    # 180-deg slice selective
    gs_ref = make_trapezoid(channel='z', system=system, amplitude=g_ss.amplitude, flat_time=t_refwd,
                            rise_time=ramp_time)
    # Spoilers
    readout_time = 6.4e-3 + 2 * system.adc_dead_time
    t_sp = 0.5 * (TE - readout_time - t_refwd)  # time gap between pi-pulse and readout
    t_spex = 0.5 * (TE - t_exwd - t_refwd)  # time gap between pi/2-pulse and pi-pulse
    # Slice refocusing + pre-180 spoiler
    sp_area = np.absolute(gs_ex.area*0.75)# from TSE based MSI sequence.
    # 0.5/1.5
    gs_spex = make_trapezoid(channel='z', system=system, area=sp_area - gs_ex.area/2, duration=t_spex, rise_time=ramp_time)
    # Post-180 spoiler
    gs_spr = make_trapezoid(channel='z', system=system, area=sp_area, duration=t_sp, rise_time=ramp_time)

    # Split gradients
    gs1, gs2, gs3, gs4, gs5 = make_split_gradients(gs_ex, gs_spex, gs_ref, gs_spr)

    #delay1 = make_delay(TE/2 - calc_duration(rf_ex) / 2 - calc_duration(rf_ref) / 2)
    #delay2 = make_delay(TE/2 - calc_duration(rf_ref) / 2)

    delayTE = make_delay(readout_time/2)

    # Single point ADC
    adc_single = make_adc(num_samples=1, dwell=10e-6, freq_offset=0, phase_offset=0)

    # Add building blocks to sequence
    seq_list = []
    for q in range(nbins):
        for s in range(n_slices):  # For each slice (multislice)
            seq = Sequence(system)
            offset_90, offset_180 = calculate_MSI_rf_params(z0=sl_locs[s], dz=thk, f0=bin_centers[q],
                                                            df=bin_width)
            # Modify the Tx/Rx frequencies
            rf_ex.freq_offset = offset_90
            rf_ex.phase_offset = rf_ex_phase - 2 * np.pi * rf_ex.freq_offset * calc_rf_center(rf_ex)[0]
            rf_ref.freq_offset = offset_180
            rf_ref.phase_offset = rf_ref_phase - 2 * np.pi * rf_ref.freq_offset * calc_rf_center(rf_ref)[0]
            adc_single.freq_offset = bin_centers[q]
            adc_single.phase_offset = 0


            # Add blocks - no readout!!!
            seq.add_block(gs1)
            seq.add_block(gs2, rf_ex)
            seq.add_block(gs3)
            seq.add_block(gs4, rf_ref)
            seq.add_block(gs5)
            seq.add_block(delayTE)


            # PREV
            #seq.add_block(rf_ex, gs_ex)
            #seq.add_block(delay1)
            #seq.add_block(rf_ref, gs_ref)
            #seq.add_block(delay2)

            seq_list.append(seq)

    return seq_list, sl_locs, bin_centers


def make_split_gradients(gs_ex, gs_spex, gs_ref, gs_spr):
    ch_ss = 'z'
    # Split gradients and recombine into blocks

    # gs1 : ramp up of gs_ex
    gs1_times = [0, gs_ex.rise_time]
    gs1_amp = [0, gs_ex.amplitude]
    gs1 = make_extended_trapezoid(channel=ch_ss, times=gs1_times, amplitudes=gs1_amp)

    # gs2 : flat part of gs_ex
    gs2_times = [0, gs_ex.flat_time]
    gs2_amp = [gs_ex.amplitude, gs_ex.amplitude]
    gs2 = make_extended_trapezoid(channel=ch_ss, times=gs2_times, amplitudes=gs2_amp)

    # gs3 : Bridged slice pre-spoiler
    gs3_times = [0, gs_spex.rise_time, gs_spex.rise_time + gs_spex.flat_time,
                 gs_spex.rise_time + gs_spex.flat_time + gs_spex.fall_time]
    gs3_amp = [gs_ex.amplitude, gs_spex.amplitude, gs_spex.amplitude, gs_ref.amplitude]
    gs3 = make_extended_trapezoid(channel=ch_ss, times=gs3_times, amplitudes=gs3_amp)

    # gs4 : Flat slice selector for pi-pulse
    gs4_times = [0, gs_ref.flat_time]
    gs4_amp = [gs_ref.amplitude, gs_ref.amplitude]
    gs4 = make_extended_trapezoid(channel=ch_ss, times=gs4_times, amplitudes=gs4_amp)

    # gs5 : Bridged slice post-spoiler
    gs5_times = [0, gs_spr.rise_time, gs_spr.rise_time + gs_spr.flat_time,
                 gs_spr.rise_time + gs_spr.flat_time + gs_spr.fall_time]
    gs5_amp = [gs_ref.amplitude, gs_spr.amplitude, gs_spr.amplitude, 0]
    gs5 = make_extended_trapezoid(channel=ch_ss, times=gs5_times, amplitudes=gs5_amp)


    return gs1,gs2,gs3,gs4,gs5



def get_slice_locations(n_slices, thk, gap, displacement):
    L = (n_slices - 1) * (thk + gap)
    sl_locs = displacement + np.arange(-L / 2, L / 2 + thk + gap, thk + gap)

    return sl_locs

def get_rf_freq_offsets(nbins, bin_sep):
    offsets = bin_sep * (np.arange(nbins) - (nbins - 1)/2)
    return offsets

def sim_msi_pulses(seq):
    # Get 90 pulse

    # Get 180 pulse

    # Simulate!
    return 0


# Test!
if __name__ == '__main__':
    seq_list, sl_locs, bin_centers = make_2dmsi_RF_seq(TE=500e-3, nbins=3, n_slices=3, thk=5e-3, gap=5e-3, bw=500)
