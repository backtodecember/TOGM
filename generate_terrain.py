import numpy as np
import math
from scipy.interpolate import CubicSpline

pts = [(-3,0),(-2,0),(-1,0),(-0.5,0),(0,0),(0.25,0.1),(0.5,0),(0.75,-0.06),(1,0),(1.7,0.2),(2.0,0.16),(2.2,0.13),(2.7,0.25),(3.5,0.22),(6,0),(10,0)]
t = []
control_pts = []
for pt in pts:
    t.append(pt[0])
    control_pts.append(pt[1])
spz = CubicSpline(t,control_pts)

def fz(x,y,No):
    """
    No1: 10 degree slope
    No2: 20 degree slope
    No3: -10 degree slope
    No4: -20 degree slope 
    No5: wavy                                                                                
    """
    if No == 1:
        slope = math.tan(10/180*math.pi)
        return x*slope
    elif No == 2:
        slope = math.tan(20/180*math.pi)
        return x*slope
    elif No == 3:
        slope = math.tan(20/180*math.pi)
        return x*slope
    elif No == 4:
        slope = math.tan(20/180*math.pi)
        return x*slope
    elif No == 5:
        return spz(x)

start = -3
stop = 12
res = 0.01
xs = np.linspace(start,stop,int((stop-start)/res)+1)
ys = np.linspace(-1,1,int(2/0.2)+1)

for No in [5]:
    with open('terrain'+str(No)+'_lowres.txt','w') as f:
        for x in xs:
            for y in ys:
                f.write(str(x)+' '+str(y)+' '+str(fz(x,y,No))+'\n')

# pts = [(-3,0),(-2,0),(-1,0),(-0.5,0),(0,0),(0.25,0.1),(0.5,0),(0.75,-0.06),(1,0),(1.7,0.2),(2.0,0.16),(2.2,0.13),(2.7,0.25),(3.5,0.22),(6,0),(10,0)]
# t = []
# control_pts = []
# for pt in pts:
#     t.append(pt[0])
#     control_pts.append(pt[1])
# spz = CubicSpline(t,control_pts)

# import matplotlib.pyplot as plt
# s = np.linspace(-3,12,100)
# zs = spz(s)
# print(zs)
# # ys = spz(s)

# plt.plot(s,zs)
# plt.legend(['spline'])
# plt.ylabel('y (m)')
# plt.xlabel('x (m)')
# plt.title('splines')
# plt.axis('equal')
# plt.grid()
# plt.show()
