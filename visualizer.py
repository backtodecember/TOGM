from klampt import WorldModel,Geometry3D,Simulator
from klampt.vis import GLProgram,camera,gldraw
from klampt.math import se3,so3
import math,utils as op

TEXT_COLOR=(0,0,0)

def empty_bb(dim=3):
    return ([1000]*dim,[-1000]*dim)

def union_bb(a,b):
    if isinstance(b,list):
        return union_bb(a,(b,b))
    else:
        return ([min(c[0],c[1]) for c in zip(a[0],b[0])],   \
                [max(c[0],c[1]) for c in zip(a[1],b[1])])

def expand_bb(bb,expand):
    return ([c-expand for c in bb[0]],   \
            [c+expand for c in bb[1]])

def expand_bb_eps(bb,eps):
    d=max(op.sub(bb[1],bb[0]))
    return expand_bb(bb,d*eps)

def contain_bb(bb,pt):
    for d in range(len(bb[0])):
        if pt[d]<bb[0][d] or pt[d]>bb[1][d]:
            return False
    return True

def compute_bb_tight(geom,Rt=None):
    if geom.type()=="Group":
        return geom.getBB()
        #Rtc=geom.getCurrentTransform() if Rt is None else se3.mul(Rt,geom.getCurrentTransform())
        #for i in range(geom.numElements()):
        #    e=geom.getElement(i)
        #    bb=union_bb(bb,compute_bb_tight(e,Rtc))
        #return bb
    elif geom.type()=="TriangleMesh":
        bb=empty_bb(3)
        m=geom.getTriangleMesh()
        for v in range(len(m.vertices)//3):
            vert=[m.vertices[v*3+d] for d in range(3)]
            if Rt is not None:
                vert=se3.apply(Rt,vert)
            bb=union_bb(bb,vert)
        return bb
    else: 
        return None

def get_robot_bb(robot,link0=None):
    bb=None
    if link0 is None:
        link0=0
    for i in range(link0,robot.numLinks()):
        if i==link0:
            bb=compute_bb_tight(robot.link(i).geometry(),robot.link(i).getTransform())
        else:
            bbI=compute_bb_tight(robot.link(i).geometry(),robot.link(i).getTransform())
            if bb is None:
                bb=bbI
            elif bbI is not None:
                bb=union_bb(bb,bbI)
    return bb

def get_object_bb(object):
    return compute_bb_tight(object.geometry())

class GLVisualizer(GLProgram):
    def __init__(self,world,table,robot,sim):
        GLProgram.__init__(self,"Visualizer")
        self.zeroZ=True
        self.world=world
        self.table=table
        self.robot=robot
        self.sim=sim
        self.doSim=True
        self.testDerivative=False
        self.qdq0=None
        self.dt=1/60.0
        self.dtSub=0.0025
        self.init_camera()

    def look_at(self,pos,tgt,scale=None):
        cam=self.view.camera
        if scale is not None:
            cam.dist=op.norm(op.sub(tgt,pos))*scale
        cam.rot=self.get_camera_rot(op.sub(pos,tgt))
        cam.tgt=tgt
        
    def get_camera_pos(self):
        cam=self.view.camera
        z=op.sin_safe(-cam.rot[1])
        x=op.sin_safe(cam.rot[2])*op.cos_safe(cam.rot[1])
        y=op.cos_safe(cam.rot[2])*op.cos_safe(cam.rot[1])
        pos=[x,y,z]
        return op.add(cam.tgt,op.mul(pos,cam.dist))
    
    def get_camera_rot(self,d):
        angz=op.atan2_safe(d[0],d[1])
        angy=op.atan2_safe(-d[2],op.sqrt_safe(d[0]*d[0]+d[1]*d[1]))
        return [0,angy,angz]

    def get_camera_dir(self,zeroZ=False):
        cam=self.view.camera
        dir=op.sub(cam.tgt,self.get_camera_pos())
        if zeroZ:
            dir=(dir[0],dir[1],0)
        if op.norm(dir)>1e-6:
            dir=op.mul(dir,1/op.norm(dir))
        dir[1]*=-1
        return dir

    def get_left_dir(self,zeroZ=False):
        dir=op.cross([0,0,1],self.get_camera_dir())
        if zeroZ:
            dir=(dir[0],dir[1],0)
        if op.norm(dir)>1e-6:
            dir=op.mul(dir,1/op.norm(dir))
        return dir

    def init_camera(self):
        bb_robot=get_robot_bb(self.robot)
        if self.table is not None:
            bb=union_bb(bb_robot,get_object_bb(self.table))
        else:
            bb=bb_robot
        pos=[bb[1][0],(bb[0][1]+bb[1][1])/2,bb_robot[1][2]]
        tgt=[bb[0][0],(bb[0][1]+bb[1][1])/2,bb_robot[0][2]]
        self.look_at(pos,tgt,2.0)
        self.moveSpd=0.005
        self.zoomSpd=1.03
        self.zoomMin=0.01
        self.zoomMax=100.0
        
        self.zoomInCam=False
        self.zoomOutCam=False
        self.forwardCam=False
        self.backCam=False
        self.leftCam=False
        self.rightCam=False
        self.raiseCam=False
        self.sinkCam=False
        return

    def keyboardfunc(self,c,x,y):
        if c==b'f':
            self.init_camera()
        elif c==b'z':
            self.zeroZ=not self.zeroZ
        elif c==b'q':
            self.zoomInCam=True
        elif c==b'e':
            self.zoomOutCam=True
        elif c==b'w':
            self.forwardCam=True
        elif c==b's':
            self.backCam=True
        elif c==b'a':
            self.leftCam=True
        elif c==b'd':
            self.rightCam=True
        elif c==b' ':
            self.raiseCam=True
        elif c==b'c':
            self.sinkCam=True
        elif c==b'r':
            self.doSim=not self.doSim
        elif c==b't':
            self.testDerivative=True
        elif c==b'y':
            if self.qdq0 is not None and self.sim is not None:
                self.sim.qdq=self.qdq0

    def keyboardupfunc(self,c,x,y):
        if c==b'q':
            self.zoomInCam=False
        elif c==b'e':
            self.zoomOutCam=False
        elif c==b'w':
            self.forwardCam=False
        elif c==b's':
            self.backCam=False
        elif c==b'a':
            self.leftCam=False
        elif c==b'd':
            self.rightCam=False
        elif c==b' ':
            self.raiseCam=False
        elif c==b'c':
            self.sinkCam=False
        elif c==b't':
            self.testDerivative=False

    def display_screen(self):
        gldraw.setcolor(TEXT_COLOR[0],TEXT_COLOR[1],TEXT_COLOR[2])
        self.draw_text((0,12),"Press r to start/stop simulation")
        self.draw_text((0,24),"Press t to test derivative")
        self.draw_text((0,36),"Press y to reset simulation")
        
    def display(self):
        if self.sim is not None:
            self.sim.update_world()
            self.sim.display()
        self.world.drawGL()

    def handle_camera(self):
        self.view.clippingplanes=(self.view.clippingplanes[0],self.zoomMax)
        cam=self.view.camera
        moveSpd=self.moveSpd*cam.dist
        if self.zoomInCam:
            cam.dist=max(cam.dist/self.zoomSpd,self.zoomMin)
        elif self.zoomOutCam:
            cam.dist=min(cam.dist*self.zoomSpd,self.zoomMax)
        elif self.forwardCam:
            delta=op.mul(self.get_camera_dir(self.zeroZ),moveSpd)
            self.look_at(op.add(self.get_camera_pos(),delta),op.add(cam.tgt,delta))
        elif self.backCam:
            delta=op.mul(self.get_camera_dir(self.zeroZ),-moveSpd)
            self.look_at(op.add(self.get_camera_pos(),delta),op.add(cam.tgt,delta))
        elif self.leftCam:
            delta=op.mul(self.get_left_dir(self.zeroZ),moveSpd)
            self.look_at(op.add(self.get_camera_pos(),delta),op.add(cam.tgt,delta))
        elif self.rightCam:
            delta=op.mul(self.get_left_dir(self.zeroZ),-moveSpd)
            self.look_at(op.add(self.get_camera_pos(),delta),op.add(cam.tgt,delta))
        elif self.raiseCam:
            delta=(0,0,moveSpd)
            self.look_at(op.add(self.get_camera_pos(),delta),op.add(cam.tgt,delta))
        elif self.sinkCam:
            delta=(0,0,-moveSpd)
            self.look_at(op.add(self.get_camera_pos(),delta),op.add(cam.tgt,delta))

    def idle(self):
        self.handle_camera()
        if self.doSim and self.sim is not None:
            if self.sim.sim is not None or self.sim.qdqSeq is not None:
                t=0.0
                while t<self.dt:
                    subStep=min(self.dt-t,self.dtSub)
                    self.sim.simulate(subStep)
                    if subStep<self.dtSub:
                        break
                    t+=subStep
        if self.testDerivative and self.sim is not None:
            self.sim.test_derivative()
        self.update_animation()
    
    def update_animation(self):
        try:
            import cv2
        except:
            return
        tmpPath="tmp.png"
        if not hasattr(self,"animation_fn") or self.animation_fn is None:
            return
        ret=self.animation_op()
        self.save_screen(tmpPath,multithreaded=False)
        self.animation_frms.append(cv2.imread(tmpPath))
        if not ret:
            height,width,layers=self.animation_frms[0].shape
            out=cv2.VideoWriter(self.animation_fn,cv2.VideoWriter_fourcc(*'DIVX'),1.0/self.animation_dur,(width,height))
            for f in self.animation_frms:
                out.write(f)
            out.release()
            if os.path.exists(tmpPath):
                os.remove(tmpPath)
            print("Finishing saving animation to %s!"%self.animation_fn)
            self.animation_fn=None
            self.animation_op=None
            self.animation_dur=None
            self.animation_frms=None
    
    def render_animation(self,fn,op,dur=1):
        if hasattr(self,"animation_fn") and self.animation_fn is not None:
            return
        self.animation_fn=fn
        self.animation_op=op
        self.animation_dur=dur
        self.animation_frms=[]