import pinocchio
from sys import argv
from os.path import dirname, join, abspath
import numpy as np
from robosimian_model_klampt import robosimian
# This path refers to Pinocchio source code but you can define your own directory here.
#pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")

# You should change here to set up your own URDF file or just pass it as an argument of this example.
#urdf_filename = pinocchio_model_dir + '/others/robots/ur_description/urdf/ur5_robot.urdf' if len(argv)<2 else argv[1]

urdf_filename = 'data/robosimian_caesar_new_pinnochio.urdf'

# Load the urdf model
model = pinocchio.buildModelFromUrdf(urdf_filename)
print('model name: ' + model.name)

# Create data required by the algorithms
data     = model.createData()


q = pinocchio.randomConfiguration(model) # joint configuration
v = np.matrix(np.random.rand(model.nv,1)) # joint velocity
tau = np.matrix(np.random.rand(model.nv,1)) # joint acceleration
#print(np.shape(q),np.shape(v),np.shape(tau))


#compute acceleration, without external force
#accel = pinocchio.aba(model,data,q,v,tau,[np.matrix(np.zeros((6,1))) for i in range(20)])

#computer inverse intertia 
Minv = pinocchio.computeMinverse(model,data,q)

#Get the Jacobian
joint_id_1 = 7
joint_id_2 = 11
joint_id_3 = 15
joint_id_4 = 19

# Perform the forward kinematics over the kinematic tree
pinocchio.computeJointJacobians(model,data,q)
J1 = pinocchio.getJointJacobian(model,data,joint_id_1,pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED) #pinocchio.ReferenceFrame.WORLD,
# J1 = J1[[0,2,4],:]
# J1 = J1[:,indices]
fext = J1.T@np.array([1,0,10,0,1,0])
fext = np.insert(fext, 0, 0.0, axis=0) #which position, insert 0.0
# fext = fext[np.newaxis].T
#vec = pinocchio.StdVec_Force()
vec = [pinocchio.Force(np.matrix(np.array([1,1,1,1,1,1])).T) for i in range(20)]
# for i in range(20):
# 	vec.append(force)
#compute dynamics jacobians
pinocchio.computeABADerivatives(model,data,q,v,tau,vec)#,np.matrix(np.zeros((6,1))))
ddq_dq = data.ddq_dq # Derivatives of the FD w.r.t. the joint config vector
ddq_dv = data.ddq_dv # Derivatives of the FD w.r.t. the joint velocity vector
ddq_dtau = data.Minv # Derivatives of the FD w.r.t. the joint acceleration vector

#pinocchio.computeForwardKinematicsDerivatives(model,data,q,v,tau)
#print(data)

indices = [0,1,2,3,4,5,7,8,9,11,12,13,15,16,17]





ddq_dq = data.ddq_dq # Derivatives of the FD w.r.t. the joint config vector
ddq_dv = data.ddq_dv # Derivatives of the FD w.r.t. the joint velocity vector
ddq_dtau = data.Minv # Derivatives of the FD w.r.t. the joint acceleration vector

#print(data.J)




# indices = [0,1,2,3,4,5,7,8,9,11,12,13,15,16,17]
# ###compare against Klampt
# rob= robosimian()
# rob.set_q_2D_(q[indices])
# rob.set_q_dot_2D_(v[indices])
# a = rob.get_Jacobians()
# print(J1)
# print(a)
# print(q)

