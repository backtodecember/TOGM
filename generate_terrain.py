import numpy as np
import math
from scipy.interpolate import CubicSpline

#this first one is a bit too steep
# pts = [(-3,0),(-2,0),(-1,0),(-0.5,0),(0,0.02),(0.25,0.1),(0.5,0.16),(1.7,-0.25),(2.0,-0.3),(2.7,-0.05),(3.5,0.4),(6,0),(10,0),(12,0)]
pts = [(-3,0),(-2,0),(-1,0),(-0.5,0),(0,0.02),(0.25,0.1),(0.5,0.16),(1.7,-0.15),(2.0,-0.2),(2.7,-0.05),(3.5,0.25),(6,0),(10,0),(12,0)]
t = []
control_pts = []
for pt in pts:
    t.append(pt[0])
    control_pts.append(pt[1])
spz = CubicSpline(t,control_pts)

spderiv= spz.derivative()


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
        slope = math.tan(-10/180*math.pi)
        return x*slope
    elif No == 4:
        slope = math.tan(-20/180*math.pi)
        return x*slope
    elif No == 5:
        return spz(x)
    elif No == 6:
        slope = math.tan(-5/180*math.pi)
        return x*slope
start = -3
stop = 12
res = 0.02
xs = np.linspace(start,stop,int((stop-start)/res)+1)
# for x in xs:
#     print(x,spderiv(x))
ys = np.linspace(-1,1,int(2/res)+1)

for No in [6]:
    with open('terrain'+str(No)+'.txt','w') as f:
        for x in xs:
            for y in ys:
                f.write(str(x)+' '+str(y)+' '+str(fz(x,y,No))+'\n')

    with open('terrain'+str(No)+'.xyz','w') as f:
        for x in xs:
            for y in ys:
                f.write(str(x)+' '+str(y)+' '+str(fz(x,y,No))+'\n')
                f.write(str(x)+' '+str(y)+' '+str(fz(x,y,No)-0.02)+'\n')

                
# pts = [(-3,0),(-2,0),(-1,0),(-0.5,0),(0,0.02),(0.25,0.1),(0.5,0.16),(1.7,-0.25),(2.0,-0.3),(2.7,-0.05),(3.5,0.4),(6,0),(10,0),(12,0)]
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
