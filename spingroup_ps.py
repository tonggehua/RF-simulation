# Copyright of the Board of Trustees of Columbia University in the City of New York
# 08/2019 WIP

import numpy as np
from scipy.integrate import solve_ivp

GAMMA = 2*42.58e6 * np.pi
GAMMA_BAR = 42.58e6

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

        A = np.array([[E2, E2, 0],
                      [-E2, E2, 0],
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
        x,y,z = self.loc
        for v in range(len(pulse_shape)):
            B1 = pulse_shape[v]
            B1x = np.real(B1)
            B1y = np.imag(B1)
            glocp = grads_shape[0,v]*x+grads_shape[1,v]*y+grads_shape[2,v]*z
            A = np.array([[0, glocp, -B1y],
                          [-glocp, 0, B1x],
                          [B1y, -B1x, 0]])
            m = m + dt*GAMMA*A@m
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
                          [-GAMMA*(dB +glocp), -T2_inv, GAMMA*B1x],
                          [GAMMA*B1y, -GAMMA*B1x, -T1_inv]])

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

    def readout(self,dwell,n,delay,grad,timing):
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

        """

        signal_1D = []
        # ADC delay
        self.fpwg(np.trapz(y=grad[:,0:2], x=timing[0:2]), delay)
        for q in range(1, len(timing)):
            if q <= n:
                signal_1D.append(self.get_m_signal())
            self.fpwg(np.trapz(y=grad[:,q:q+2], dx=dwell), dwell)

        self.signal.append(signal_1D)

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
        3 x 3 rotational matrix

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
