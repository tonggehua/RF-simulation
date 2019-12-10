# Use im2projs to generate (32 x 32) PROJ_IM and PROJ_rf matrices from all imagenet 32 x 32 training data
import numpy as np
import multiprocessing as mp
import random
import rf_sim.im2projs_ as i2p
from skimage.transform import radon
from math import pi, ceil
import time
# Load data

def rf_sim_sample(spin, rf, grad, dt_rf, n_samples):
    return i2p.sample_rf_proj(spin.apply_rf_store(rf, grad, dt_rf)[0], dt_rf, n_samples)

if __name__ == '__main__':
    # Load single batch of 32 x 32 imagenet data
    nb = 1
    data = np.load('./rf_sim/imagenet32_converted_grayscale/imagenet32_batch_{:d}.npy'.format(nb))

    # Sample 5000 images form the batch
    random.seed(1210)
    n_im = 500
    im_inds = random.sample(list(np.arange(0, np.shape(data)[0])),n_im) # we want 5000 images
    print('Indices used: ')
    print(im_inds)
    N = 32
    n_proj = int(ceil(N*pi/2)) # 51

    # Define simulation parameters
    dt = 1e-6 # 1 us
    dur = 1.28e-3 # Tp = 1.28 ms for each projection
    n_sim = round(dur/dt)
    b1_mag = 1e-6 # 1 uT; corresponds to fr_deg_ms = 15.33 deg / ms
   # b1_mag = (fr_deg_ms * pi / 180) * 1e3 / GAMMA
    rf = b1_mag*np.ones(n_sim)
    grad_mag = 1e-3 # 1 mT/m # might be hard to implement on scanner
    grad = grad_mag*np.ones(n_sim)

    # Do it for all images!
    # Multiprocessing at the level of spins

    # Second run: 150 ~ 300
    ind_ranges = np.array([
           [480, 490],
           [490, 500]])
    
    # TODO EQ: uncomment this line and rerun! Thanks
    # ind_ranges = ind_ranges + 300

    for ind_range in ind_ranges:
        im_inds_sub = im_inds[ind_range[0]:ind_range[1]]

        # Create final arrays for storage
        P_im_all = np.zeros((N,n_proj,len(im_inds_sub)),dtype=complex)
        P_rf_all = np.zeros((N,n_proj,len(im_inds_sub)),dtype=complex)

        for u, ii in enumerate(im_inds[ind_range[0]:ind_range[1]]):
            im = data[ii,:,:]
            # Bin image
            n_tissues = 10
            image_binned = i2p.bin_image(im, n_tissues)

            # Assign PD, T1, T2 values
            # What is the range?
            # Image projections
            thetas_im = np.linspace(0, 180, n_proj + 1)[0:-1]
            P_im = radon(image_binned, theta=thetas_im)
           # print(np.shape(P_im))

            # RF projections
            tp = np.load('./rf_sim/tissue_params_info_sg_1021.npy', allow_pickle=True).all()
            tissue_params = tp['tissue_params']
            all_spins = i2p.make_spins_from_binned_image_2d(image_binned, tissue_params)

            thetas_rf = np.concatenate((thetas_im, thetas_im + 180))

            P_rf_radial = np.zeros((int(N/2)+1, len(thetas_rf)), dtype=complex)

            beg = time.time()

            for ind, theta in enumerate(thetas_rf):
                lasttime = time.time()
                q = 0 if ind < n_proj else 1
                n_samples=int(N/2) + q
                flip = theta * pi / 180

                grad_3d = np.zeros((3, len(grad)))
                grad_3d[0, :] = grad * np.cos(flip)  # Gx
                grad_3d[1, :] = grad * np.sin(flip)  # Gy

                pool = mp.Pool(mp.cpu_count())
                all_signals = pool.starmap_async(rf_sim_sample,
                                                 [(spin, rf, grad_3d, dt, n_samples) for spin in all_spins]).get()
                pool.close()

               # all_signals = np.array([rf_sim_sample(spin, rf, grad_3d, dt, n_samples) for spin in all_spins])
                P_rf_radial[0:int(N/2) + q, ind]= np.sum(all_signals, axis=0)

                print('Theta #{:d} simulated'.format(ind+1))
                print('Time taken: ', str(time.time()-lasttime))

            print("Time taken for one image: ", str(time.time()-beg))

            P_rf = np.zeros((N, n_proj), dtype=complex)
            for v in range(n_proj):
                rf_proj_k = np.concatenate((np.flip(P_rf_radial[1:, v + n_proj], axis=0), P_rf_radial[0:-1, v]))
                P_rf[:, v] = np.fft.fftshift(np.fft.ifft(rf_proj_k))

            P_im_all[:,:,u] = P_im
            P_rf_all[:,:,u] = P_rf

        # Save the data!
        timestamp = time.strftime("%Y%m%d%H%M")
        np.save('./rf_sim/training/projs_for_training_#{:d}to{:d}_'.format(ind_range[0],ind_range[1])+timestamp+'.npy',
                {'PROJ_IM':P_im_all,'PROJ_RF':P_rf_all,'IM_INDS_SUB': im_inds_sub,'N_BATCH':nb})



