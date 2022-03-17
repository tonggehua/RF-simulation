# Simulate the spatial-spectral response from the MSI sequence in development
# Import functions
from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from bloch.spingroup_ps_t2star import *
import bloch.pulseq_blochsim_methods as blcsim
from rf_sim.msi.msi_sim_make_rf_seqs import make_2dmsi_RF_seq
import bloch.spingroup_ps as sg
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from rf_sim.rf_simulations import simulate_rf

GAMMA_BAR = 42.58e6

def diamond_sim_ver1():
    # Make a grid of spins in spatial-spectral space and apply pulse+gradients / delays
    # For now, just do {90(+Gz), delay, 180(-Gz)} (turbo_factor = 1)

    # Load mock sequence (?) and extract the necessary information
    seq = Sequence()
    seq.read('seqs/2dmsi_sim_single_slice_19bins_800Hz_per_bin_TE0.5.seq')

    # Extract RF(90), delay, RF(180)

    # Simulate with NumSolverSpinGroup class

    spins_list = []
    zs = np.linspace(-15e-3,15e-3,15, endpoint=True) # [meters]
    #dfs = np.linspace(-1200,1200,16, endpoint=True) # [Hz]
    dfs = np.linspace(-7600, 5400, 16, endpoint=True)

    for z in zs:
        for df in dfs:
            #newspin = sg.NumSolverSpinGroup(loc=(0, 0, z), pdt1t2=(1,0,0), df=df)
            newspin = SpinGroupT2star(loc=(0, 0, z), pdt1t2=(1,1,0.1), df=df, t2star=0.02, num_spins=100)
            spins_list.append(newspin)

    blk = 4
    count = 0
    while (blk+3) <= 7: # len(seq.dict_block_events): # Over all slices and all
        #TODO factor in RF phase and frequency displacements.

        rf90 = seq.get_block(blk).rf
        grad90 = seq.get_block(blk).gz
        rf180 = seq.get_block(blk+2).rf
        grad180 = seq.get_block(blk+2).gz
        delay_time_1 = seq.get_block(blk+1).delay.delay

        delay_time_2 = delay_time_1 + calc_duration(rf90)/2

        dt90 = rf90.t[1] - rf90.t[0]
        pulse_shape_90 = rf90.signal/GAMMA_BAR
        pulse_shape_90 *= np.exp(-1j*rf90.phase_offset)*np.exp(-1j*2*np.pi*rf90.freq_offset*rf90.t)
        grads_shape_90 = np.zeros((3, len(pulse_shape_90)))
        grads_shape_90[2,:] = np.interp(dt90*np.arange(len(rf90.signal)),
                                    np.cumsum([0,grad90.rise_time,grad90.flat_time,grad90.fall_time]),
                                    [0,grad90.amplitude/GAMMA_BAR,grad90.amplitude/GAMMA_BAR,0])

        dt180 = rf180.t[1] - rf180.t[0]
        pulse_shape_180 = rf180.signal/GAMMA_BAR
        pulse_shape_180 *= np.exp(-1j*rf180.phase_offset)*np.exp(-1j*2*np.pi*rf180.freq_offset*rf180.t)
        grads_shape_180 = np.zeros((3, len(pulse_shape_180)))
        grads_shape_180[2,:] = np.interp(dt180*np.arange(len(rf180.signal)),
                                    np.cumsum([0,grad180.rise_time,grad180.flat_time,grad180.fall_time]),
                                    [0,grad180.amplitude/GAMMA_BAR,grad180.amplitude/GAMMA_BAR,0])
        blk += 3

        spin = spins_list[0]
        results90 = [spin.apply_rf_store(pulse_shape_90, grads_shape_90, dt90)[0][-1] for spin in spins_list]
        __ = [spin.delay(delay_time_1) for spin in spins_list]
        results180 = [spin.apply_rf_store(pulse_shape_180, grads_shape_180, dt180)[0][-1] for spin in spins_list]
        __ = [spin.delay(delay_time_2) for spin in spins_list]

        for spin in spins_list: spin.reset()

        count += 1
        print(f'Pulse #{count} simulated.')

    savemat('simulated_Data/msi_sim_results_TE500ms_added_delay_t2star_solver_030822.mat',
            {'after_1st': results90, 'after_2nd': results180,'zs':zs, 'dfs':dfs, 'shape':np.array([len(dfs),len(zs)])})



def diamond_sim(save_data=False):
    # Spatial-spectral simulation

    # Make sequence
    seqs, sl_locs, bin_centers = make_2dmsi_RF_seq(TE=500e-3, nbins=1, n_slices=1, thk=5e-3, gap=5e-3, bw=500,
                                                   use_sigpy_90=False, use_sigpy_180=False)

    # Make spins
    # Use WM: t2 = 110 ms, t2* = 80 ms
    zs, dfs, spins_list = make_spatial_spectral_spins(fov=30e-3, nz=12, bw=2.5e3, nbins=13,
                                                      t2=110e-3, t2star=80e-3, num_spins=25)

    print('zs : ', zs)
    print('dfs : ', dfs)

    # Simulate!
    # one first; then do more
    seq = seqs[0]
    seq_info = blcsim.store_pulseq_commands(seq)
    results = [blcsim.apply_pulseq_commands(sg,seq_info,store_m=True) for sg in spins_list]

    signals_at_TE = [sg.get_m_signal() for sg in spins_list]
    #for spin in spins_list: spin.reset()

    display_spatial_spectral_plot(signals_at_TE, zs, dfs)
    if save_data:
        savemat('simulated_Data/results_msi2d_with_spoilers_n25_TE500_nosigpy.mat', {'results':results, 'signals_at_TE': signals_at_TE})
    return results, signals_at_TE, zs, dfs, sl_locs, bin_centers

def display_spatial_spectral_plot(signals_at_TE, zs, dfs):
    plt.figure()
    signals_grid = np.reshape(signals_at_TE, (len(zs),len(dfs)))
    plt.imshow(np.absolute(signals_grid))
    plt.show()

    return 0

def make_spatial_spectral_spins(fov, nz, bw, nbins, t2, t2star, num_spins):
    spins_list = []
    zs = np.linspace(-fov/2, fov/2, nz, endpoint=False) # [meters]
    dfs = np.linspace(-bw/2, bw/2, nbins, endpoint=False) # [Hz]

    for z in zs:
        for df in dfs:
            newspin = SpinGroupT2star(loc=(0, 0, z), pdt1t2=(1,0,t2), df=df, t2star=t2star, num_spins=num_spins)
            #newspin = NumSolverSpinGroup(loc=(0,0,z),pdt1t2=(1,0,t2),df=df)
            spins_list.append(newspin)

    return zs, dfs, spins_list

def sim_rfs_from_seq():
    seqs, sl_locs, bin_centers = make_2dmsi_RF_seq(TE=500e-3, nbins=1, n_slices=1, thk=5e-3, gap=5e-3, bw=800,
                                                   use_sigpy_90=False, use_sigpy_180=False)
    seq = seqs[0]
    rf90 = seq.get_block(2).rf
    rf180 = seq.get_block(4).rf

    rf_dt_90 = rf90.t[1] - rf90.t[0]
    rf_dt_180 = rf180.t[1] - rf180.t[0]

    bw = 800 * 2

    signal90, m90 = simulate_rf(bw_spins=bw, n_spins=200, pdt1t2=(1,0,0), flip_angle=90, dt=rf_dt_90,
                       solver="RK45",
                       pulse_type='custom', pulse_shape=rf90.signal/GAMMA_BAR, display=False)

    signal180, m180 = simulate_rf(bw_spins=bw, n_spins=200, pdt1t2=(1,0,0), flip_angle=180, dt=rf_dt_180,
                       solver="RK45",
                       pulse_type='custom', pulse_shape=rf180.signal/GAMMA_BAR, display=False)

    # Plot rf
    make_rf_profile_plot(bw=bw, m=m90)
    make_rf_profile_plot(bw=bw, m=m180)

    #savemat('msi_rf_sims_both.mat',{'m90':m90,'m180':m180})

    return 0

def make_rf_profile_plot(bw, m):
    mxy = m[:,0,-1] + 1j*m[:,1,-1]
    freqs = np.linspace(-bw/2, bw/2, m.shape[0])
    plt.figure(1)

    plt.subplot(311)
    plt.plot(freqs, np.absolute(mxy),)
    plt.title("|Mxy|")

    plt.subplot(312)
    plt.plot(freqs, np.angle(mxy),label="Phase")
    plt.title('Mxy phase')

    plt.subplot(313)
    plt.plot(freqs, np.squeeze(m[:,2,-1]))
    plt.title('Mz')


    plt.show()

    return 0




if __name__ == '__main__':
    #diamond_sim(save_data=True)
    sim_rfs_from_seq()

