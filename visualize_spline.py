import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# n = 10
# # AddX = np.load('results/28/run9/solution_addx256.npy')
# AddX = np.load('temp_files22/solution_addx481.npy')
# control_pts = AddX[0,0:0+2*n]
# t = np.linspace(0,1,n)
# spx = CubicSpline(t,control_pts[0:n])
# spz = CubicSpline(t,control_pts[n:2*n])

# s = np.linspace(0,1,100)
# xs = spx(s)
# ys = spz(s)

# print(xs)
# print(ys)
# plt.plot(xs,ys)
# plt.plot(xs[0],ys[0],'ro')
# plt.legend(['spline'])
# plt.ylabel('y (m)')
# plt.xlabel('x (m)')
# plt.title('splines')
# plt.axis('equal')
# plt.show()



n = 10
# AddX = np.load('results/28/run9/solution_addx256.npy')
AddX = np.load('temp_files22/solution_addx481.npy')
control_pts = AddX[0,0:0+n]
print(control_pts)
t = np.linspace(0,1,n)
spx = CubicSpline(t,control_pts[0:n])
# spz = CubicSpline(t,control_pts[n:2*n])

s = np.linspace(0,1,100)
xs = spx(s)
# ys = spz(s)

print(xs)
print(ys)
plt.plot(xs,ys)
plt.plot(xs[0],ys[0],'ro')
plt.legend(['spline'])
plt.ylabel('y (m)')
plt.xlabel('x (m)')
plt.title('splines')
plt.axis('equal')
plt.show()

