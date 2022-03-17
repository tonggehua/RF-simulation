from pypulseq.Sequence.sequence import Sequence
from math import pi
import numpy as np
from scipy.io import savemat, loadmat
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.opts import Opts
import math
import warnings


def write_2D_MSI(N=256,FOV=250e-3,enc='xyz',TR=2000e-3,TE=12.4e-3, thk=5e-3,turbo_factor=4,
                 slice_locations=[0],N_bins=19,bin_width=800,t_ex=2.5e-3,t_ref=2e-3,randomize_bin=False):
    # Generate a 2D-MSI sequence using Pypulseq
    # Sequence TSE-based
    # 1. Initialize ----------------------------------------------------------------
    system = set_June_system_limits()
    seq = Sequence(system)
    ## System limits & setup
    # Set system limits
    ramp_time = 250e-6 # Ramp up/down time for all gradients where this is specified

    # 2. Calculate derived parameters -----------------------------------------------
    dk = 1/FOV
    Nf, Np = (N, N)
    k_width = Nf * dk
    n_slices = len(slice_locations)

    ## Spatial encoding directions
    ch_ro = enc[0]
    ch_pe = enc[1]
    ch_ss = enc[2]

    ## Number of echoes per excitation (i.e. turbo factor)
    n_echo = turbo_factor
    # Flip angles
    fa_exc = 90  # degrees
    fa_ref = 180 # degrees

    ## Durations
    ###  Readout duration
    readout_time = 6.4e-3 + 2 * system.adc_dead_time
    ### Excitation pulse duration
    t_exwd = t_ex + system.rf_ringdown_time + system.rf_dead_time
    ### Refocusing pulse duration
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
    gs_ex = make_trapezoid(channel=ch_ss, system=system, amplitude= -g_ss.amplitude, flat_time=t_exwd,
                           rise_time=ramp_time)# Note that the 90-deg SS gradient's amplitude is reversed for 2D MSI purposes!!
    ### 180 deg pulse (+x')
    rf_ref_phase = 0
    flip_ref = fa_ref * np.pi / 180
    rf_ref, gz, _ = make_sinc_pulse(flip_angle=flip_ref, system=system, duration=t_ref, slice_thickness=thk,
                                    apodization=0.5, time_bw_product=t_ref*rfbw, phase_offset=rf_ref_phase, use='refocusing',
                                    return_gz=True)
    gs_ref = make_trapezoid(channel=ch_ss, system=system, amplitude=g_ss.amplitude, flat_time=t_refwd,
                            rise_time=ramp_time)

    """## Make gradients and ADC
    * gs_spex : slice direction spoiler between initial excitation and 1st 180 pulse
    * gs_spr : slice direction spoiler between 180 pulses 
    * gr_spr : readout direction spoiler; area is (fsp_r) x (full readout area)
    """

    # SS spoiling
    ags_ex = gs_ex.area / 2
    gs_spr = make_trapezoid(channel=ch_ss, system=system, area=ags_ex * (1 + fsp_s), duration=t_sp, rise_time=ramp_time)
    gs_spex = make_trapezoid(channel=ch_ss, system=system, area=ags_ex * fsp_s, duration=t_spex, rise_time=ramp_time)

    # Readout gradient and ADC
    gr_acq = make_trapezoid(channel=ch_ro, system=system, flat_area=k_width, flat_time=readout_time,
                            rise_time=ramp_time)

    # No need for risetime delay since it is set at beginning of flattime; delay is ADC deadtime
    adc = make_adc(num_samples=Nf, duration=gr_acq.flat_time - 40e-6, delay=20e-6) #

    # RO spoiling
    gr_spr = make_trapezoid(channel=ch_ro, system=system, area=gr_acq.area * fsp_r, duration=t_sp, rise_time=ramp_time)


    # Prephasing gradient in RO direction
    agr_preph = gr_acq.area / 2 + gr_spr.area
    gr_preph = make_trapezoid(channel=ch_ro, system=system, area=agr_preph, duration=t_spex, rise_time=ramp_time)

    ## **Phase encoding areas**
    # Number of readouts/echoes to be produced per TR
    n_ex = math.floor(Np / n_echo)
    pe_steps = np.arange(1, n_echo * n_ex + 1) - 0.5 * n_echo * n_ex - 1
    if divmod(n_echo, 2)[1] == 0:  # If there is an even number of echoes
        pe_steps = np.roll(pe_steps, -round(n_ex / 2))
    pe_order = pe_steps.reshape((n_ex, n_echo), order='F').T
    savemat('pe_info.mat', {'order': pe_order, 'dims': ['n_echo', 'n_ex']})
    phase_areas = pe_order * dk

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

    # gs7 : The gs3 for next pi-pulse
    gs7_times = [0, gs_spr.rise_time, gs_spr.rise_time + gs_spr.flat_time,
                 gs_spr.rise_time + gs_spr.flat_time + gs_spr.fall_time]
    gs7_amp = [0, gs_spr.amplitude, gs_spr.amplitude, gs_ref.amplitude]
    gs7 = make_extended_trapezoid(channel=ch_ss, times=gs7_times, amplitudes=gs7_amp)

    # gr3 : pre-readout gradient
    gr3 = gr_preph

    # gr5 : Readout post-spoiler
    gr5_times = [0, gr_spr.rise_time, gr_spr.rise_time + gr_spr.flat_time,
                 gr_spr.rise_time + gr_spr.flat_time + gr_spr.fall_time]
    gr5_amp = [0, gr_spr.amplitude, gr_spr.amplitude, gr_acq.amplitude]
    gr5 = make_extended_trapezoid(channel=ch_ro, times=gr5_times, amplitudes=gr5_amp)

    # gr6 : Flat readout gradient
    gr6_times = [0, readout_time]
    gr6_amp = [gr_acq.amplitude, gr_acq.amplitude]
    gr6 = make_extended_trapezoid(channel=ch_ro, times=gr6_times, amplitudes=gr6_amp)

    # gr7 : the gr3 for next repeat
    gr7_times = [0, gr_spr.rise_time, gr_spr.rise_time + gr_spr.flat_time,
                 gr_spr.rise_time + gr_spr.flat_time + gr_spr.fall_time]
    gr7_amp = [gr_acq.amplitude, gr_spr.amplitude, gr_spr.amplitude, 0]
    gr7 = make_extended_trapezoid(channel=ch_ro, times=gr7_times, amplitudes=gr7_amp)

    ### Timing (delay) calculations"""

    # delay_TR : delay at the end of each TSE pulse train (i.e. each TR)
    t_ex = gs1.t[-1] + gs2.t[-1] + gs3.t[-1]
    t_ref = gs4.t[-1] + gs5.t[-1] + gs7.t[-1] + readout_time
    t_end = gs4.t[-1] + gs5.t[-1]
    TE_train = t_ex + n_echo * t_ref + t_end
    TR_fill = (TR - n_slices * TE_train) / n_slices
    TR_fill = system.grad_raster_time * round(TR_fill / system.grad_raster_time)
    if TR_fill < 0:
        TR_fill = 1e-3
        print(f'TR too short, adapted to include all slices to: {1000 * n_slices * (TE_train + TR_fill)} ms')
    else:
        print(f'TR fill: {1000 * TR_fill} ms')

    delay_TR = make_delay(TR_fill)

    # 2D MSI frequency bins
    bin_centers = bin_width*(np.arange(N_bins)-(N_bins-1)/2)
    if randomize_bin:
        np.random.shuffle(bin_centers)


    # Add building blocks to sequence
    for q in range(N_bins):
        print("Adding another bin...")
        for k_ex in range(n_ex + 1):  # For each TR
            for s in range(n_slices):  # For each slice (multislice)
                offset_90, offset_180 = calculate_MSI_rf_params(z0=slice_locations[s],dz=thk,f0=bin_centers[q],df=bin_width)
                adc.freq_offset = bin_centers[q]

                rf_ex.freq_offset = offset_90
                rf_ex.phase_offset = rf_ex_phase - 2 * np.pi * rf_ex.freq_offset * calc_rf_center(rf_ex)[0]

                rf_ref.freq_offset = offset_180
                rf_ref.phase_offset = rf_ref_phase - 2 * np.pi * rf_ref.freq_offset * calc_rf_center(rf_ref)[0]

                seq.add_block(gs1)
                seq.add_block(gs2, rf_ex)  # make sure gs2 has channel ch_ss
                seq.add_block(gs3, gr3)

                for k_echo in range(n_echo):  # For each echo
                    if k_ex > 0:
                        phase_area = phase_areas[k_echo, k_ex - 1]
                    else:
                        # First TR is skipped so zero phase encoding is needed
                        phase_area = 0.0  # 0.0 and not 0 because -phase_area should successfully result in negative zero
                    gp_pre = make_trapezoid(channel=ch_pe, system=system, area=phase_area, duration=t_sp,
                                            rise_time=ramp_time)
                    gp_rew = make_trapezoid(channel=ch_pe, system=system, area=-phase_area, duration=t_sp,
                                            rise_time=ramp_time)
                    seq.add_block(gs4, rf_ref)
                    seq.add_block(gs5, gr5, gp_pre)
                    # Skipping first TR
                    if k_ex > 0:
                        seq.add_block(gr6, adc)
                    else:
                        seq.add_block(gr6)
                    seq.add_block(gs7, gr7, gp_rew)
                seq.add_block(gs4)
                seq.add_block(gs5)
                seq.add_block(delay_TR)

    return seq, bin_centers


def set_June_system_limits():
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130,
                      slew_unit='T/m/s', rf_ringdown_time=100e-6, # changed from 30e-6
                      rf_dead_time=100e-6, adc_dead_time=20e-6)
    return system


def calculate_MSI_rf_params(z0,dz,f0,df):
    """ Calculates the RF information for a single diamond 2D-MSI bin
    Params
    ------
    z0 : float
        Slice location from isocenter [meters]
    dz : float
        Slice thickness [meters]
    f0 : float
        Frequency bin center [Hz]
    df : float
        Frequency bin width [Hz]

    Returns
    -------
    msi_info : dict
        BW and offset of RF pulses as well as gradient amplitudes
    """
    dfdz = df/dz

    offset_90 = f0 - dfdz*z0
    offset_180 = f0 + dfdz*z0

    return offset_90, offset_180


if __name__ == '__main__':
    # ACR Slice locations (for default, set slice_locations to None)
    n_slices = 11
    thk = 5e-3
    gap = 5e-3
    L = (n_slices - 1) * (thk + gap)
    displacement = -4.4e-3 # change this
    acr_sl_locs = displacement + np.arange(-L / 2, L / 2 + thk + gap, thk + gap)
    print(acr_sl_locs)



    # # Yellow tray slice locations (used)
    # thk = 5e-3
    # gap = 5e-3
    # n_slices = 5
    # L = (n_slices - 1) * (thk+gap)
    # displacement = -9.2e-3
    # sl_locs = displacement + np.arange(-L/2, L/2 + thk + gap, thk + gap)
    #FOV = 128e-3 # For tray phantom

    # Set parameters
    TR = 2000e-3
    TE = 13e-3
    N = 256
    FOV = 250e-3

    sl_locs = acr_sl_locs[0:-1:2]
    sl_locs = sl_locs[::-1]



    # Set up same as 2nd exp. of last time
    N_bins = 9
    bin_width = 3*800 / N_bins

    # 081221: try 2 things
    # 1. Lengthen the RF pulse
    dur_factor = 1 # change duration of RF pulses; default is 1
    # 2. Change turbo factor
    tf = 16 # default is 16

    # Make sequence
    seq, bin_centers = write_2D_MSI(N=N,FOV=FOV,enc='xyz',TR=TR,TE=TE,thk=thk,turbo_factor=tf,slice_locations=sl_locs,
                       N_bins=N_bins,bin_width=bin_width,t_ex=dur_factor*2.5e-3, t_ref=dur_factor*2e-3, randomize_bin=True)
    print("Done")
    # Check sequence
    print(seq.test_report())
    # Plot
    seq.plot(time_range=[0,TR])
    # Save
    seq.write(f'2D_MSI_120321_TR{TR}_Ns{len(sl_locs)}_bins_randomized_Nbins{N_bins}_bw{N_bins*bin_width}_disp{displacement}_rfdur{dur_factor}x_tf{tf}.seq')
    savemat('2D_MSI_120321_bin_centers.mat',{'bin_centers': bin_centers})

