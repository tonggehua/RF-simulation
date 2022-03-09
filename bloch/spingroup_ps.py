# Copyright of the Board of Trustees of Columbia University in the City of New York

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from bloch.util import combine_gradients
GAMMA = 2*42.58e6 * np.pi
GAMMA_BAR = 42.58e6

#from disimpy import gradients, simulations, utils as dgrad, dsim, dutils

class SpinGroup:
    """Basic magnetization unit for Bloch simulation

    A SpinGroup has a defined location, relaxation times, proton density, and off-resonance.

    Parameters
    ----------
    loc : tuple，optional
        (x,y,z)
        Location from isocenter in meters; default is (0,0,0)
    pdt1t2 : tuple, optional
        (PD,T1,T2)
        Proton density between 0 and 1
        T1, T2 in seconds; if zero it signifies no relaxation
        Default is PD = 1 and T1, T2 = 0
    df : float, optional
        Off-resonance in Hertz; default is 0

    Attributes
    ----------
    m : numpy.ndarray
        [[Mx],[My],[Mz]]
        3 x 1 array of magnetization
    PD : float
        Proton density between 0 and 1 that scales the signal
    T1 : float
        Longitudinal relaxation time in seconds
    T2 : float
        Transverse relaxation time in seconds
    loc : tuple
        (x,y,z)
        Location of spin group from isocenter in meters
    df : float
        Off-resonance in Hertz
    signal : array
        Complex signal produced from this spin group; only generated by self.readout()


    """

    def __init__(self, loc=(0,0,0), pdt1t2=(1,0,0), df=0):
        self.m = np.array([[0], [0], [1]])
        self.PD = pdt1t2[0]
        self.T1 = pdt1t2[1]
        self.T2 = pdt1t2[2]
        self.loc = loc
        self.df = df
        self.signal=[]

    def reset(self):
        self.signal=[]
        self.m = np.array([[0], [0], [1]])

    def get_m_signal(self):
        """Gets spin group's transverse magnetization

        Returns
        -------
        m_signal : numpy.ndarray
            Single complex number representing transverse magnetization
            Real part      = Mx
            Imaginary part = My

        """
        m_signal = np.squeeze(self.PD*(self.m[0] + 1j * self.m[1]))

        return m_signal

    def scale_m_signal(self, scale):
        """Scales the signal to simulate Rx effects

        Parameters
        ----------
        scale : complex
            Complex number B1_receive to multiply all signal samples by

        """

        self.signal = np.array(self.signal)*scale


    def fpwg(self,grad_area,t):
        """Apply only gradients to the spin group

        The spin group precesses with gradients applied.
        This function changes self.m and has no output.

        Parameters
        ----------
        grad_area : numpy.ndarray
            [Gx_area, Gy_area, Gz_area]
            Total area under Gx, Gy, and Gz in seconds*Tesla/meter
        t : float
            Total time of precession in seconds

        """
        x,y,z = self.loc
        phi = GAMMA*(x*grad_area[0]+y*grad_area[1]+z*grad_area[2])+2*np.pi*self.df*t
        C, S = np.cos(phi), np.sin(phi)
        E1 = 1 if self.T1 == 0 else np.exp(-t/self.T1)
        E2 = 1 if self.T2 == 0 else np.exp(-t/self.T2)
        A = np.array([[E2*C, E2*S, 0],
                      [-E2*S, E2*C, 0],
                      [0, 0, E1]])
        self.m = A@self.m + [[0],[0],[1 - E1]]

    def delay(self, t):
        """Applies a time passage to the spin group

        The spin group free-precesses with only T1, T2 effects (no RF, no gradients).
        This function changes self.m and has no output.

        Parameters
        ----------
        t : float
            Delay interval in seconds

        """
        self.T1 = max(0,self.T1)
        self.T2 = max(0,self.T2)
        E1 = 1 if self.T1 == 0 else np.exp(-t/self.T1)
        E2 = 1 if self.T2 == 0 else np.exp(-t/self.T2)
        phi = 2*np.pi*self.df*t
        C, S = np.cos(phi), np.sin(phi)

        A = np.array([[E2*C, E2*S, 0],
                      [-E2*S, E2*C, 0],
                      [0, 0, E1]])
        self.m = A@self.m + np.array([[0], [0], [1 - E1]])



    def apply_rf(self, pulse_shape, grads_shape, dt):
        """Applies an RF pulse

        Euler's method numerical integration of Bloch equation
        with both B1(RF) field and gradient field

        Parameters
        ----------
        pulse_shape :
            1 x n complex array (B1)[tesla]
        grads_shape :
            3 x n real array  [tesla/meter]
        dt:
            raster time for both shapes [seconds]

        """
        m = self.m
        dB = self.df/GAMMA_BAR
        T1_inv = 1/self.T1 if self.T1 > 0 else 0
        T2_inv = 1/self.T2 if self.T2 > 0 else 0

        x,y,z = self.loc
        for v in range(len(pulse_shape)):
            B1 = pulse_shape[v]
            B1x = np.real(B1)
            B1y = np.imag(B1)
            glocp = grads_shape[0,v]*x+grads_shape[1,v]*y+grads_shape[2,v]*z

           # A = np.array([[0, glocp, -B1y],
           #               [-glocp, 0, B1x],
           #               [B1y, -B1x, 0]])
          #  m = m + dt*GAMMA*A@m


            A = np.array([[-T2_inv, GAMMA*(dB + glocp), -GAMMA*B1y],
                          [-GAMMA*(dB + glocp), -T2_inv, GAMMA*B1x],
                          [GAMMA*B1y, -GAMMA*B1x, -T1_inv]])
            m = m + dt*(A@m + np.array([[0],[0],[T1_inv]]))

        self.m = m

    def apply_rf_store(self, pulse_shape, grads_shape, dt):
        """Applies an RF pulse and store magnetization at all time points

        Euler's method numerical integration of Bloch equation
        with both B1(RF) field and gradient field

        Parameters
        ----------
        pulse_shape :
            1 x n complex array (B1)[tesla]
        grads_shape :
            3 x n real array  [tesla/meter]
        dt:
            raster time for both shapes [seconds]

        Returns
        -------
        m_signal : 1 x n complex array (a.u.)
            Transverse magnetization in complex form
        magnetizations : 3 x n real array (a.u.)
            [Mx, My, Mz] magnetization over time

        """
        m = self.m

        T1_inv = 1/self.T1 if self.T1 > 0 else 0
        T2_inv = 1/self.T2 if self.T2 > 0 else 0

        m_signal = np.zeros(len(pulse_shape), dtype=complex)
        magnetizations = np.zeros((3, len(pulse_shape)+1))
        magnetizations[:,0] = np.squeeze(m)

        x,y,z = self.loc
        dB = self.df/GAMMA_BAR
        for v in range(len(pulse_shape)):
            B1 = pulse_shape[v]
            B1x = np.real(B1)
            B1y = np.imag(B1)
            glocp = grads_shape[0,v]*x+grads_shape[1,v]*y+grads_shape[2,v]*z
            A = np.array([[-T2_inv, GAMMA*(dB + glocp), -GAMMA*B1y],
                          [-GAMMA*(dB + glocp), -T2_inv, GAMMA*B1x],
                          [GAMMA*B1y, -GAMMA*B1x, -T1_inv]])
            #print(A)
            m = m + dt*(A@m + np.array([[0],[0],[T1_inv]]))
            magnetizations[:,v+1] = np.squeeze(m)
            self.m = m

            m_signal[v] = self.get_m_signal()

        return m_signal, magnetizations

    # TODO : incorporate scipy.integrate.solve_ivp for more accurate RF simulation
    # Motivated by the overflow problems generated by current simple method (step-and-add) when dealing with adiabatic pulses
    # def apply_rf_solveivp_store(self, pulse_func, grads_func, interval, dt):
    #     # Uses scipy ode integrator to simulate RF effects.
    #     # Preparation
    #     m = self.m
    #     T1_inv = 1/self.T1 if self.T1 > 0 else 0
    #     T2_inv = 1/self.T2 if self.T2 > 0 else 0
    #     x,y,z = self.loc
    #     dB = self.df/GAMMA_BAR
    #
    #     # Desired time points
    #     tmodel = np.linspace()
    #     tmodel = np.arange(0, len(pulse_shape)*dt, dt)
    #
    #     # Define ODE
    #     def bloch_fun(t,m):
    #         m = np.reshape(m, (3,1))
    #         B1 = pulse_shape[np.where(tmodel==t)]
    #         print(B1)
    #         B1x = np.real(B1)[0]
    #         B1y = np.imag(B1)[0]
    #
    #         grad = grads_shape[:,np.where(tmodel==t)[0]]
    #         glocp = grad[0,0]*x+grad[1,0]*y+grad[2,0]*z
    #
    #         A = np.array([[-T2_inv, GAMMA*(dB + glocp), -GAMMA*B1y],
    #                       [-GAMMA*(dB +glocp), -T2_inv, GAMMA*B1x],
    #                       [GAMMA*B1y, -GAMMA*B1x, -T1_inv]])
    #         print(A)
    #         return np.squeeze(np.matmul(A,m) + np.array([[0],[0],[T1_inv]])) # dm/dt
    #
    #
    #     # TODO test this?
    #     # Put into solve_ivp
    #     print(np.shape(m))
    #     sol = solve_ivp(fun=bloch_fun, t_span=(tmodel[0],tmodel[-1]), y0=np.squeeze(m), method='RK45',t_eval=tmodel)
    #
    #     all_ms = sol.y
    #     print(np.shape(all_ms))
    #
    #     t = sol.t
    #     print(np.shape(t))
    #
    #     self.m = all_ms[-1]
    #     return all_ms


    def _apply_rf_old(self, pulse_shape, grads_shape, dt):
        """Deprecated method for applying an RF pulse
        """

        b1 = pulse_shape
        gs = grads_shape
        loc = self.loc
        for k in range(len(b1)):
            bx = np.real(b1[k])
            by = np.imag(b1[k])
            bz = np.sum(np.multiply([gs[0, k], gs[1, k], gs[2, k]], loc)) + self.df / GAMMA_BAR
            be = np.array([bx, by, bz])
            self.m = anyrot(GAMMA * be * dt) @ self.m

    def _readout_old(self,dt,n,delay,grad,timing):
        """Deprecated method for sampling with gradients on
        """
        signal_1D = []
        self.fpwg(grad[:,0]*delay, delay)
        v = 1
        for q in range(1, len(timing)):
            if v <= n:
                signal_1D.append(self.get_m_signal())
            self.fpwg(grad[:, v]*dt,dt)
            v += 1

        self.signal.append(signal_1D)

    def readout_trapz(self,dwell,n,delay,grad,timing,phase):
        """ ADC sampling for single spin group

        Samples spin group's magnetization while playing an arbitrary gradient
        This data is then stored in self.signal

        Parameters
        ----------
        dwell : float
            Constant sampling interval in seconds
        n : int
            Number of samples
        delay : float
            Delay of the first point sampled relative to beginning of gradient waveform
        grad : numpy.ndarray
            2D array with shape 3 x m (i.e. m samples of the 3D gradient (Gx, Gy, Gz))
            Arbitrary gradient waveform in Tesla/meter
        timing : numpy.ndarray
            1D array with length m
            Timing of gradient waveform
        phase : float
            ADC phase in radians

        """

        signal_1D = []
        # ADC delay
        self.fpwg(np.trapz(y=grad[:,0:2], x=timing[0:2]), delay)
        for q in range(1, len(timing)):
            if q <= n:
                signal_1D.append(self.get_m_signal())
            self.fpwg(np.trapz(y=grad[:,q:q+2], dx=dwell), dwell)

        signal_1D_ref = np.array(signal_1D) * np.exp(-1j*phase)

        self.signal.append(signal_1D_ref)

    def readout(self,dwell,n,delay,grad,timing,phase):
        """ ADC sampling for single spin group

        Samples spin group's magnetization while playing an arbitrary gradient
        This data is then stored in self.signal

        Parameters
        ----------
        dwell : float
            Constant sampling interval in seconds
        n : int
            Number of samples
        delay : float
            Delay of the first point sampled relative to beginning of gradient waveform
        grad : numpy.ndarray
            2D array with shape 3 x m (i.e. m samples of the 3D gradient (Gx, Gy, Gz))
            Arbitrary gradient waveform in Tesla/meter
        timing : numpy.ndarray
            1D array with length m
            Timing of gradient waveform
        phase : float
            ADC phase in radians

        """

        signal_1D = []
        # ADC raster time assuming timing is uniformly spaced
        dt_adc = timing[1] - timing[0]

        N_delay = int(delay / dt_adc)
        delay_times = dt_adc * np.arange(N_delay)
        delay_grads = np.zeros((3,N_delay))
        delay_grads[0,:] = np.interp(delay_times, timing, grad[0,:])
        delay_grads[1,:] = np.interp(delay_times, timing, grad[1,:])
        delay_grads[2,:] = np.interp(delay_times, timing, grad[2,:])

        # ADC delay
        self.fpwg(np.trapz(y=delay_grads, x=delay_times), delay)

        # Readout
        adc_begin_time = delay
        N_dwell = int(dwell / dt_adc)
        for q in range(n):
            signal_1D.append(self.get_m_signal())
            dwell_times = adc_begin_time + np.linspace(0,dwell,N_dwell+1,endpoint=True)
            dwell_grads = np.zeros((3, N_dwell+1))
            dwell_grads[0,:] = np.interp(dwell_times, timing, grad[0,:])
            dwell_grads[1,:] = np.interp(dwell_times, timing, grad[1,:])
            dwell_grads[2,:] = np.interp(dwell_times, timing, grad[2,:])
            self.fpwg(np.trapz(y=dwell_grads, x=dwell_times), dwell)
            adc_begin_time += dwell

        signal_1D_ref = np.array(signal_1D) * np.exp(-1j*phase)

        self.signal.append(signal_1D_ref)


class NumSolverSpinGroup(SpinGroup):
    # TODO package the funtions to generate a final function that only takes in t and M and returns dM/dt

    @staticmethod
    def interpolate_waveforms(grads_shape, pulse_shape, dt):
        # Helper function to generate continuous waveforms
        gx_func = interp1d(x=dt*np.arange(len(pulse_shape)), y=grads_shape[0,:], bounds_error=False, fill_value=0)
        gy_func = interp1d(x=dt*np.arange(len(pulse_shape)), y=grads_shape[1,:], bounds_error=False, fill_value=0)
        gz_func = interp1d(x=dt*np.arange(len(pulse_shape)), y=grads_shape[2,:], bounds_error=False, fill_value=0)
        pulse_real_func = interp1d(x=dt*np.arange(len(pulse_shape)), y=np.real(pulse_shape),bounds_error=False, fill_value=0)
        pulse_imag_func = interp1d(x=dt*np.arange(len(pulse_shape)), y=np.imag(pulse_shape),bounds_error=False, fill_value=0)

        return gx_func, gy_func, gz_func, pulse_real_func, pulse_imag_func

    # TODO make this return the input diffEQ to solver
    def get_bloch_eqn(self, grads_shape, pulse_shape, dt):
        x, y, z = self.loc
        gx_func, gy_func, gz_func, pulse_real_func, pulse_imag_func = self.interpolate_waveforms(grads_shape, pulse_shape, dt)

        T1_inv = 1 / self.T1 if self.T1 > 0 else 0
        T2_inv = 1 / self.T2 if self.T2 > 0 else 0
        dB = self.df / GAMMA_BAR

        def bloch_eqn(t,m):
            gx, gy, gz = gx_func(t), gy_func(t), gz_func(t)
            B1x = pulse_real_func(t)
            B1y = pulse_imag_func(t)
            glocp = gx*x + gy*y + gz*z
            A = np.array([[-T2_inv, GAMMA * (dB + glocp), -GAMMA * B1y],
                          [-GAMMA * (dB + glocp), -T2_inv, GAMMA * B1x],
                          [GAMMA * B1y, -GAMMA * B1x, -T1_inv]])
            return A @ m + np.array([[0], [0], [T1_inv]])


        return bloch_eqn


    # Override RF method!
    # TODO fix problem with using this in sequence simulation
    # TODO override apply_rf
    def apply_rf_store(self, pulse_shape, grads_shape, dt):
        m = np.squeeze(self.m)

        ####
        magnetizations = np.zeros((3, len(pulse_shape) + 1))
        magnetizations[:, 0] = np.squeeze(m)
        ####

        # Set correct arguments to ivp solver ...
        results = solve_ivp(fun=self.get_bloch_eqn(grads_shape,pulse_shape,dt), t_span=[0,len(pulse_shape)*dt],
                            y0=m, method="RK45",t_eval=dt*np.arange(len(pulse_shape)), vectorized=True)

        m_signal = results.y[0,:] + 1j*results.y[1,:]
        magnetizations = results.y

        # Update spingroup magnetization
        self.m = np.reshape(magnetizations[:,-1], (3,1))

        return m_signal, magnetizations


class SpinGroupDiffusion(SpinGroup):
    # Add D parameter as attribute
    def __init__(self, loc=(0,0,0), pdt1t2=(1,0,0), df=0, D=0, b=0):
        # Diffusion coefficient in [(mm^2)/seconds]
        super().__init__(loc, pdt1t2, df)
        self.D = D
        self.b = b
        self.diff_att = np.exp(-self.b*self.D) # diffusion attenuation factor

    # Overwrite signal method
    def get_m_signal(self):
        """Gets spin group's transverse magnetization attenuated by pulse sequence diffusion effects
           Note: this is a simple version of diffusion simulation using the exp(-bD) factor.

        Returns
        -------
        m_signal : numpy.ndarray
            Single complex number representing transverse magnetization
            Real part      = Mx
            Imaginary part = My

        """
        m_signal = self.diff_att*np.squeeze(self.PD*(self.m[0] + 1j * self.m[1]))

        return m_signal

    # def apply_diffusion(self, gradient_blocks_list, dt, num_spins=1e4):
    #     """
    #     Use disimpy to simulate the amount of attenuation
    #
    #     Parameters
    #     ----------
    #     gradient_blocks_list : list
    #         List of Pypulseq gradient objects to be applied to randomly diffusing spins
    #
    #     Returns
    #     -------
    #     diff_att : float
    #         Net diffusion attenuation coefficient after applying this list of blocks
    #
    #     """
    #     # Reset diffusion attenuation to 1
    #     self.diff_att = 1
    #
    #     for block in gradient_blocks_list: # For each of the selected gradients, model diffusion effects
    #         # Compile gradient
    #         waveform = self.compile_gradients_for_diffusion_sim(block, dt)
    #         # Set up substrate
    #         substrate = {'type': 'free'}
    #         n_s = num_spins
    #         diffusivity = 1e-6 * self.D # Convert from [mm^2/s] to [m^2/s]
    #         signal = dsim.simulation(n_s, diffusivity, waveform, dt, substrate)
    #         self.diff_att = self.diff_att * float(signal)
    #
    #     return self.diff_att

    @staticmethod
    def compile_gradients_for_diffusion_sim(block, dt):
        # Shapes pulseq gradient block into (1, N_timepoints, 3) numpy array
        if 'rf' in block.__dict__.keys():
            raise TypeError("Blocks with diffusion gradients must not include an RF pulse")
        grad_waveform, _, _, _ = combine_gradients(blk=block,dt=dt,delay=0)
        N = grad_waveform.shape[1]
        grad_waveform_shaped = np.reshape(np.swapaxes(grad_waveform,0,1), (1,N,3))

        return grad_waveform_shaped




# Helpers
def anyrot(v):
    """ Helper method that generates rotational matrix from Rodrigues's formula

    Rodrigue's formula gives a 3 x 3 matrix for arbitrary rotational axis and angle

    Parameters
    ----------
    v : tuple
        (vx, vy, vz)
        3D angular rotation vector in Cartesian space
        whose direction is the rotational axis and whose length the angle rotated in radians

    Returns
    -------
    R : numpy.ndarray
        3 x 3 rotational matrix`1

    """
    vx = v[0]
    vy = v[1]
    vz = v[2]
    th = np.linalg.norm(v,2)
    C = np.cos(th)
    S = np.sin(th)

    if th != 0:
        R = (1/(th*th))*np.array([[vx*vx*(1-C)+th*th*C, vx*vy*(1-C)-th*vz*S, vx*vz*(1-C)+th*vy*S],
                                  [vx*vy*(1-C)+th*vz*S, vy*vy*(1-C)+th*th*C, vy*vz*(1-C)-th*vx*S],
                                  [vx*vz*(1-C)-th*vy*S, vy*vz*(1-C)+th*vx*S, vz*vz*(1-C)+th*th*C]])
    else:
        R = np.array([[1,0,0],[0,1,0],[0,0,1]])

    return R
