# Simulate - replicate figure
import numpy as np
from scipy.io import loadmat, savemat
import bloch.phantom as pht
import time
import bloch.pulseq_blochsim_methods as blcsim
import multiprocessing as mp
from rf_sim.msi.msi_sim_make_rf_seqs import *
GAMMA_BAR = 42.58e6

# Make phantom
def make_fig_phantom(bw):
    df_data = loadmat('bmap_msi_fig.mat')
    b0 = df_data['b0']

    dBmap = np.zeros((b0.shape[1],1,b0.shape[0]))
    dBmap[:,0,:] = np.transpose((bw/2) * (b0 / np.max(b0)))


    T1map = np.zeros(dBmap.shape)
    T2map = 110e-3 * np.ones(dBmap.shape)
    PDmap = np.ones(dBmap.shape)
    T2starmap = 80e-3 * np.ones(dBmap.shape)


    fig_phantom = pht.Phantom(T1map=T1map,T2map=T2map,PDmap=PDmap,vsize=1e-3, T2star_map=T2starmap,
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
    spin_signals = [spin.get_avg_signal() for spin in simulated_spins]

    # -----------------------------------------------------------
    # Time the code: Toc
    print("Simulation complete!")
    print("Time used: %s seconds" % (time.time() - start_time))

    return spin_signals

if __name__ == '__main__':
    bw = 1500
    myseq = make_mavric_RF_seq(nbins=5, bw=bw, TR=1)
    mypht = make_fig_phantom(bw=bw)
    raw_signal = simulate_figure(myseq, mypht, num_spins=3)
    savemat('mavric.mat',{'raw_signal':raw_signal})
