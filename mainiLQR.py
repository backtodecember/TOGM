from robosimian_GM_simulation import robosimianSimulator
from ilqr import iLQR
from ilqr.dynamics import Dynamics
import numpy as np

class robosimianDyn(Dynamics):
    def __init__(self,dt):
        self.dt = dt
        self._state_size = 30
        self._action_size = 12
        self._has_hessians = False

        self.compare_eps = 1e-7
        self.current_x = np.zeros(self._state_size)
        self.current_u = np.zeros(self._action_size)
        self.current_J = np.zeros((self._state_size,self._state_size + self._action_size))
        super(Dynamics,self).__init__()

    def f(self,x,u,i):
        # if np.linalg,norm(self.current_x)


        return

    def f_x(self,x,u,i):
        return

    def f_u(self,x,u,i):
        return

    #these would not be implemented
    def f_xx(self,x,u,i):
        return
    
    def f_ux(self,x,u,i):
        return

    def f_uu(self,x,u,i):
        return

if __name__ == "__main__":
    robosimianDyn = robosimianDyn()
    robosimianDyn.f([],[],[])
    robosimianDyn.f_x([],[],[])