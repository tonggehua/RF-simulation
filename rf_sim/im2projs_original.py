# GT, Oct 2019
#TODO : make this method by next Friday - Oct 18, 2019

import time
from skimage.io import imread
from scipy.io import savemat, loadmat
#from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import bloch.spingroup_ps as sg
from math import pi, ceil


GAMMA_BAR = 42.58e6
GAMMA = 2*pi*GAMMA_BAR

def image_to_projections(image, n_proj, rf, grad, dt):
    """Generates specified number of direct projections around image
        as well as STAR (confounded) projections using Bloch simulation

    Parameters
    ----------
    image : array_like
        2D real array of gray-scale values
    n_proj : int
        Number of projections
    rf : float or array_like
        Single value of constant complex B1 or length-m complex array of rf waveform
    grad : array_like
        (3 x 1) vector of constant 3D gradient or (3 x m) gradient waveform
    dt : float
        Time resolution for RF simulation


    Returns
    -------
    P_im : np.ndarray
        (n_proj x m) real array of direct projections
    P_rf : np.ndarray
        (n_proj x m) complex array of FT of simulated projection signals

    """
    # Bin image
    n_tissues = 10
    image_binned = bin_image(image,n_tissues)

    # Assign PD, T1, T2 values
    # What is the range?
    # Image projections
    thetas_im = np.linspace(0,180,n_proj+1)[0:-1]
    P_im = radon(image_binned, theta=thetas_im)
    print(np.shape(P_im))

    # RF projections
    tp = np.load('./rf_sim/tissue_info.npy', allow_pickle=True).all()
    tissue_params  = tp['tissue_params']
    all_spins = make_spins_from_binned_image_2d(image_binned, tissue_params)

    thetas_rf = np.concatenate((thetas_im, thetas_im+180))
    n_points = np.shape(P_im)[0]

    P_rf_radial = np.zeros((int(n_points/2)+1, len(thetas_rf)), dtype=complex)
    for ind, theta in enumerate(thetas_rf):
        q = 0 if ind < n_proj else 1
        P_rf_radial[0:int(n_points/2)+q,ind] = rf_sim_proj(all_spins, rf, grad, dt, theta, n_samples=int(n_points/2) + q)
        print('another theta simulated')

    # TODO convert rf from k domain to image domain ; two readouts make one projection!
    P_rf = np.zeros((n_points, n_proj), dtype=complex)
    for u in range(n_proj):
        P_rf[:,u] = np.concatenate((np.flip(P_rf_radial[1:,u+n_proj],axis=0),P_rf_radial[0:-1,u]))


    return P_im, P_rf

def rf_sim_proj(spin_list, rf, grad, dt, theta, n_samples):
    #  Simulate
    # TODO oblique gradients for projection purposes :)

    flip = theta * pi / 180

    grad_3d = np.zeros((3,len(grad)))
    grad_3d[0,:] =  grad * np.cos(flip)# Gx
    grad_3d[1,:] =  grad * np.sin(flip)# Gy

    all_signals = np.array([sample_rf_proj(spin.apply_rf_store(rf, grad_3d, dt)[0], dt, n_samples) \
                   for spin in spin_list])

    summed_signals = np.sum(all_signals, axis=0)

    return summed_signals

def sample_rf_proj(signal, dt, n_samples):
    # Sample the "projection" obtained from RF
    # interpolate
    sample_times = np.linspace(0, len(signal)*dt, n_samples)
    return np.interp(sample_times, dt*np.arange(len(signal)), signal, left=0, right=0)


def bin_image(image,n_bins):
    # normalize & bin gray-scale image into discrete values
    image= np.array(image)
    nm_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    binned_image = np.zeros(np.shape(image))
    dv = 1/n_bins

    val_map = np.arange(n_bins)

    for m in range(n_bins):
        vmin = dv*m
        vmax = dv*(m+1)
        binned_image += val_map[m]*np.where(nm_image > vmin, 1, 0)* np.where(nm_image <= vmax, 1, 0)

    print(binned_image[15,15])

    return binned_image

def make_spins_from_binned_image_2d(image, tissue_params):
    spin_list = []
    FOV = 0.256
    # Generate the correct locations (fov option!? )
    def get_loc_from_ind(ind1, ind2):
        S = np.shape(image)
        x = FOV * ind1/S[0] + (FOV/S[0])/2 - FOV/2
        y = FOV * ind2/S[1] + (FOV/S[1])/2 - FOV/2
        return (x,y,0)

    # Generate the spins
    for u in range(np.shape(image)[0]):
        for v in range(np.shape(image)[1]):
            spin_list.append(sg.SpinGroup(loc=get_loc_from_ind(u,v),
                                          pdt1t2=tissue_params[image[u,v]],
                                          df=0))

    return spin_list

if __name__ == '__main__':
    #image = loadmat('./rf_sim/phantom.mat')['a']

    # Load an ImageNet image
    #images = np.load('./rf_sim/imagenet32_converted_grayscale/imagenet32_batch_1.npy')
    images = loadmat('./rf_sim/simple_ims.mat')
    #im = images[0,:,:]
    im = images['im1']

    # Define simulation parameters
    dt = 1e-7
    dur = 1e-4
    n_sim = round(dur/dt)
    fr_deg_ms = 10 * 360
    b1_mag = (fr_deg_ms * pi / 180) * 1e3 / GAMMA
    rf = b1_mag*np.ones(n_sim)
    grad_mag = 10e-3
    grad = grad_mag*np.ones(n_sim)

    # Make both sets of projections
    #n_proj = int(ceil(32*pi/2))
    n_proj = 32

    beginning = time.time()

    Pim, Prf = image_to_projections(im, n_proj, rf, grad, dt=dt)

    print('Time taken: ', str(time.time()-beginning), ' seconds')
    savemat('./rf_sim/projs_point_obj.mat',{'PROJ_IM': Pim, 'PROJ_RF': Prf})
    # Inspect those projections
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(Pim)
    plt.gray()

    plt.subplot(1,2,2)
    plt.imshow(np.absolute(Prf))
    plt.gray()

    plt.show()

