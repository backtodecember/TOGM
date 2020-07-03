from robosimian_GM_simulation import robosimianSimulator
from ilqr import iLQR
from ilqr.dynamics import Dynamics
from ilqr.cost import Cost, AutoDiffCost, QRCost
import theano.tensor as T

import numpy as np
from KlamptDiffNE import *
import pyDiffNE
import time
class robosimianDyn(Dynamics):
    def __init__(self,dt):
        self.dt = dt
        self._state_size = 30
        self._action_size = 12
        self._has_hessians = False

        self.compare_eps = 1e-7
        self.current_x = np.random.rand(self._state_size)
        self.current_u = np.random.rand(self._action_size)
        self.current_J = np.zeros((self._state_size,self._state_size + self._action_size))

        #initialize robot
        q_2D = np.array([0.0,1.02,0.0] + [0.6- 1.5708,0.0,-0.6]+[0.6+1.5708,0.0,-0.6]+[0.6-1.5708,0.0,-0.6] \
            +[0.6+1.5708,0.0,-0.6])[np.newaxis].T
        q_dot_2D = np.array([0.0]*15)[np.newaxis].T

        #Test 14+ should have extraploation set to be True
        self.robot = robosimianSimulator(q= q_2D,q_dot = q_dot_2D,dt = self.dt,dyn = 'diffne', augmented = True, extrapolation = True, integrate_dt = self.dt)
        super(Dynamics,self).__init__()

    @property
    def state_size(self):
        """State size."""
        return self._state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size

    @property
    def has_hessians(self):
        """Whether the second order derivatives are available."""
        return self._has_hessians

    ##f is used in forward pass, while f_x, and f_u are used in backward pass
    def f(self,x,u,i):
        _,x_next = self.robot.getDyn(x,u,continuous=True) 
        return x_next

    def f_x(self,x,u,i):
        if (np.linalg.norm(self.current_x - x) < self.compare_eps) and (np.linalg.norm(self.current_u - u) < self.compare_eps):
            return self.current_J[:,0:self._state_size]
        else:
            _,self.current_J = self.robot.getDynJac(x,u)
            self.current_x = x
            self.current_u = u
            return self.current_J[:,0:self._state_size]
        return

    def f_u(self,x,u,i):
        if (np.linalg.norm(self.current_x - x) < self.compare_eps) and (np.linalg.norm(self.current_u - u) < self.compare_eps):
            return self.current_J[:,self._state_size:self._action_size + self._state_size]
        else:
            _,self.current_J = self.robot.getDynJac(x,u)
            self.current_x = x
            self.current_u = u
            return self.current_J[:,self._state_size:self._action_size + self._state_size]
        return

    #these would not be implemented
    def f_xx(self,x,u,i):
        return
    
    def f_ux(self,x,u,i):
        return

    def f_uu(self,x,u,i):
        return

class robosimianCost(AutoDiffCost):
    def __init__(self,N):
        self.N = self.N

if __name__ == "__main__":
    #LQR setup
    robosimianDyn = robosimianDyn(dt = 0.01)
    Q = np.zeros((30,30)) 
    Q[15,15] = 2.0 #torso speed
    Q[1,1] = 1.0 #torso height
    x_goal = np.zeros(30)
    x_goal[15] = 0.3
    x_goal[1] = 0.9 
    R = np.eye(12,12)*0.01 #regularize the control a little bit
    cost = QRCost(Q = Q,R = R, x_goal = x_goal)

    #initial guess and other setup
    N = 2000 #10s with dt = 0.005
    x0 = np.concatenate((np.load('results/PID_trajectory/4/q_history.npy')[1],np.load('results/PID_trajectory/4/q_dot_history.npy')[1]))
    u0 = np.load('results/PID_trajectory/4/u_history.npy')[1:2001] #N*action_size
    ilqr = iLQR(robosimianDyn,cost,N)
    #callback function
    J_hist = []
    def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
        J_hist.append(J_opt)
        info = "converged" if converged else ("accepted" if accepted else "failed")
        print("iteration", iteration_count, info, J_opt, xs[-1][0])

    #start calculating
    start_time = time.time()
    total_iter = 500
    xs, us = ilqr.fit(x0,u0,n_iterations = total_iter,on_iteration = on_iteration)
    print('========================')
    print('Took',time.time() - start_time)

    np.save('temp_files/solution_x'+str(total_iter)+'.npy',xs)
    np.save('temp_files/solution_u'+str(total_iter)+'.npy',us)
