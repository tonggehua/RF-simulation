# Simulate - replicate figure
import numpy as np
from scipy.io import loadmat, savemat
import bloch.phantom as pht
import time
import bloch.pulseq_blochsim_methods as blcsim
import bloch.spingroup_ps_t2star as sg2
import multiprocessing as mp
from rf_sim.msi.msi_sim_make_rf_seqs import *
GAMMA_BAR = 42.58e6

# Make phantom
def make_fig_phantom(bw,vsize=1e-3,T2=110e-3, T2s=80e-3):
    df_data = loadmat('bmap_msi_fig.mat')
    b0 = df_data['b0']
    dBmap = np.zeros((b0.shape[1],1,b0.shape[0]))
    dBmap[:,0,:] = np.transpose((bw/2) * (b0 / np.max(b0)))

    T1map = np.zeros(dBmap.shape)
    T2map = T2 * np.ones(dBmap.shape)
    PDmap = np.ones(dBmap.shape)
    T2starmap = T2s * np.ones(dBmap.shape)

    fig_phantom = pht.Phantom(T1map=T1map,T2map=T2map,PDmap=PDmap,vsize=vsize, T2star_map=T2starmap,
                    dBmap=dBmap,Dmap=0,loc=(0,0,0))
    return fig_phantom

def simulate_figure(seq, phantom, num_spins):
    # Time the code: Tic
    start_time = time.time()
    # -----------------------------------------------------------
    loc_ind_list = phantom.get_list_inds()
    seq_info = blcsim.store_pulseq_commands(seq)
    dfmap = phantom.dBmap

    # Multiprocessing simulation
    simulated_spins = [blcsim.sim_single_spingroup(loc_ind, dfmap[loc_ind], phantom, seq_info, 'T2Star', 0, num_spins, 'spingroup') for
                                  loc_ind in loc_ind_list]
    #spin_signals = [spin.get_avg_signal() for spin in simulated_spins]
    spin_signals = [spin.get_m_signal() for spin in simulated_spins]

    # -----------------------------------------------------------
    # Time the code: Toc
    print("Simulation complete!")
    print("Time used: %s seconds" % (time.time() - start_time))

    return spin_signals


def simulate_figure_2DMSI_ideal(seqs, rfbw, TE, phantom, num_spins):
    start_time = time.time()
    #
    dfmap = phantom.dBmap
    loc_ind_list = phantom.get_list_inds()
    all_spin_signals = []
    # Load RF displacements and bandwidths from seq file
    for seq in seqs: # Each seq has exactly 7 blocks and represents 1 spatial-spectral bin
        rf90 = seq.get_block(2).rf
        rf180 = seq.get_block(4).rf
        gz1 = np.max(seq.get_block(2).gz.waveform[0])
        gz2 = np.max(seq.get_block(4).gz.waveform[0])

        spin_signals = []

        for loc_ind in loc_ind_list:
            sgloc = phantom.get_location(loc_ind)

            isc = sg2.SpinGroupT2star(loc=sgloc, pdt1t2=phantom.get_params(loc_ind), df=dfmap[loc_ind],
                                      t2star=phantom.get_t2star(loc_ind),num_spins=num_spins)
            isc.apply_ideal_RF(rf_phase=rf90.phase_offset,
                               fa=np.pi/2, f_low=rf90.freq_offset-rfbw/2, f_high=rf90.freq_offset+rfbw/2,
                               gradients=np.array([0,0,gz1]))
            isc.delay(TE/2)
            isc.apply_ideal_RF(rf_phase=rf180.phase_offset,
                               fa=np.pi, f_low=rf180.freq_offset-rfbw/2, f_high=rf180.freq_offset+rfbw/2,
                               gradients=np.array([0,0,gz2]))
            isc.delay(TE/2)

            spin_signals.append(isc.get_m_signal())
        print(f'Another bin simulated')
        all_spin_signals.append(spin_signals)
    #
    print("Simulation complete!")
    print("Time used: %s seconds" % (time.time() - start_time))

    return all_spin_signals

def find_optimal_TE(T2,T2s):
    R2 = 1/T2
    R2s = 1/T2s
    TE_opt = np.log(R2/R2s)/(R2-R2s)
    return TE_opt

if __name__ == '__main__':
    bw = 1600

    # Make seq
    #myseq = make_mavric_RF_seq(nbins=5, bw=bw, TR=1)
    #myseq, thk = make_semac_RF_seq(nbins=5, bw_acq=2000, bw_grad=1.5*1500, pulse_bw_factor=1, fov_z=29e-3, TR=2)
    nb = 8
    #TE = 93.4e-3

    mypht = make_fig_phantom(bw=bw,vsize=1e-3,T2=110e-3,T2s=80e-3)
    TE = find_optimal_TE(T2=110e-3, T2s=80e-3)
    #TE = 93.4e-3  # Previously used for T2 = 110 ms, T2s = 80 ms

    seqs, sl_locs, bin_centers, gs_spr = make_2dmsi_RF_seq(TE=TE, nbins=nb, n_slices=1, thk=5e-3, gap=5e-3, bw=bw,
                                                   use_sigpy_90=False, use_sigpy_180=False)
    rfbw = bw/nb

    results = simulate_figure_2DMSI_ideal(seqs,rfbw,TE,mypht,num_spins=100)

    savemat('simulated_Data/msi2d_ideal_8bins_thk5mm_t2-110_t2s-80_n100.mat',{'signals': results})


    ############## NONIDEAL 2D MSI SIM ########################
    # u = 0
    # signals_dict = {}
    # signals_dict['bin_centers'] = bin_centers
    # for myseq in seqs:
    # #print(f"Slice thickness: {thk*1000} mm")
    #     mypht = make_fig_phantom(bw=bw)
    #     raw_signal = simulate_figure(myseq, mypht, num_spins=25)
    #     signals_dict[f'bin{u+1}'] = raw_signal
    #     u += 1
    #
    # savemat(f'simulated_Data/msi2d_all_{nb}bins_pulseq.mat', signals_dict)
    ############################################################
