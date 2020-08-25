from visualizer import GLVisualizer
import transforms3d.euler as eul
import pyDiffNE as DNE
from OpenGL import GL
import numpy as np
import klampt,random,math,utils as op

TABLE_COLOR=(0.2,0.6,0.3)

class DOFMap:
    def __init__(self,type,klamptDOFId=None,diffNEDOFId=None,constant=None,coef=1):
        self.type=type
        self.klamptDOFId=klamptDOFId
        self.diffNEDOFId=diffNEDOFId
        self.constant=constant
        self.coef=coef

    def print_DMap(self,klampt,diffNE):
        kName=klampt.link(self.klamptDOFId).getName()
        if self.diffNEDOFId is None:
            print("Klampt joint: ",kName," mapped to constant: ",self.constant)
        else:
            dName=diffNE.joint(self.diffNEDOFId[0])._name
            print("Klampt joint: ",kName," mapped to diffNE name: ",dName)

    def map_to_klampt(self,diffNE,q,T):
        if self.constant is not None:
            return self.constant*self.coef
        elif self.type=="t" or self.type=="m":
            return q[self.diffNEDOFId[1]]*self.coef
        else:
            assert self.type=="r"
            J=diffNE.body.joint(self.diffNEDOFId[0])
            #i=self.diffNEDOFId[0]
            #R=[[T[0,i*4+0],T[1,i*4+0],T[2,i*4+0]],
            #   [T[0,i*4+1],T[1,i*4+1],T[2,i*4+1]],
            #   [T[0,i*4+2],T[1,i*4+2],T[2,i*4+2]]]
            #ang=eul.mat2euler(np.array(R).transpose(),axes='sxyz')
            #return ang[self.klamptDOFId%3]*self.coef
            if J._typeJoint==DNE.ROT_3D_XYZ:
                r=DNE.eulerX1Y3Z2(DNE.Vec3d(q[J._offDOF:J._offDOF+J.nrDOF()]))
                R=np.array(r.toList())
                ang=eul.mat2euler(R,axes='sxyz')
                return ang[self.klamptDOFId%3]*self.coef
            else:
                return q[self.diffNEDOFId[1]]*self.coef

class DiffNERobotModel:
    def __init__(self,world,urdf,use2DBase=False):
        self.world=world
        self.robot=world.loadRobot(urdf)
        self.body=DNE.ArticulatedLoader.readURDF(urdf,False,False)
        if use2DBase:
            self.use2DBase=True
            self.body.addBase(2,DNE.Vec3d([0,1,0]))     #add a 2D base
        else:
            self.use2DBase=False
            self.body.addBase(3,DNE.Vec3d())            #add a 3D base
        self.body.simplify(10)
        self.DMap=self.get_DMap()
        self.print_DMap()
        self.sim=None
        self.ops=None
        self.floor=None
        self.floorQuad=None
        #qdq
        self.qdq=DNE.Vecd([0.0 for i in range(self.num_DOF()*2)])
        self.qdqSeqId=0
        self.qdqSeq=None
        self.qdq0=None
        
    def get_DMap(self):
        if self.use2DBase:
            DMap=[DOFMap("t",0,(0,0)),
                  DOFMap("t",1,constant=0),
                  DOFMap("t",2,(0,1),coef=-1),
                  DOFMap("r",5,constant=0),
                  DOFMap("r",4,(1,2),coef=1),
                  DOFMap("r",3,constant=0)]
        else:
            DMap=[DOFMap("t",0,(0,0)),
                  DOFMap("t",1,(0,1)),
                  DOFMap("t",2,(0,2)),
                  DOFMap("r",5,(1,3)),
                  DOFMap("r",4,(1,4)),
                  DOFMap("r",3,(1,5))]
        for i in range(2,self.body.nrJ()):
            joint=self.body.joint(i)
            if joint._typeJoint==DNE.HINGE_JOINT:
                DMap.append(DOFMap("m",self.get_klampt_DOFId(joint._name),(i,joint._offDOF),coef=-1))
            else: 
                print("DiffNE-Klampt-Interface does not recognize joint type: ",joint._typeJoint)
                assert False
        return DMap
        
    def get_klampt_DOFId(self,name):
        if '+' in name:
            name=name.split('+')[0]
        for i in range(self.robot.numLinks()):
            if self.robot.link(i).getName()==name:
                return i
        assert False
        return -1
    
    def eliminate_joints(self,names,DOF=None):
        if DOF is None:
            DOF=[0.0 for i in range(self.num_DOF())]
        #clear DOF
        for i in range(6,len(self.DMap)):
            if self.DMap[i].diffNEDOFId is not None:
                self.DMap[i].constant=DOF[self.DMap[i].diffNEDOFId[1]]
                self.DMap[i].diffNEDOFId=None
        #eliminate
        self.body.eliminateJoint(names,DNE.Vecd(DOF),10)
        #remap DOF
        DMap=self.get_DMap()
        for i in range(6,len(DMap)):
            for j in range(6,len(self.DMap)):
                if self.DMap[j].klamptDOFId==DMap[i].klamptDOFId:
                    self.DMap[j].diffNEDOFId=DMap[i].diffNEDOFId
                    self.DMap[j].constant=None
        self.print_DMap()
        #qdq
        self.qdq=DNE.Vecd([0.0 for i in range(self.num_DOF()*2)])
        
    def set_qdq_sequence(self,qdqSeq):
        if hasattr(self,"VecType"):
            self.qdqSeq=[self.VecType(qdq) for qdq in qdqSeq]
        else: self.qdqSeq=[DNE.Vecd(qdq) for qdq in qdqSeq]
        
    def set_qdq(self,qdq):
        assert len(qdq)==self.num_DOF()*2
        if hasattr(self,"VecType"):
            self.qdq=self.VecType(qdq)
        else: self.qdq=DNE.Vecd(qdq)
        
    def set_floor(self,floor,ees=[],nrB=6,mu=0.7,strengthBasis=1000):
        if self.floor is not None:
            raise RuntimeError("Cannot call set_floor twice!")
        if self.granular:
            if DiffNERobotModel.is_plane(floor):
                self.floor=self.FloorType(self.body,DNE.Vec4d(floor),"inputs.txt","weights.txt")
            elif DiffNERobotModel.is_terrain_file(floor):
                self.floor=self.FloorType(self.body,floor[0],floor[1],floor[2],"inputs.txt","weights.txt")
            else: raise RuntimeError("Unsupported floor type!")
            self.floor.writeEndEffectorVTK("EndEffector.vtk",False)
            self.floor.writeEndEffectorVTK("TorqueCenter.vtk",True)
        else: 
            if DiffNERobotModel.is_plane(floor):
                self.floor=self.FloorType(self.body,DNE.Vec4d(floor),self.Vec3Type(self.g),nrB,mu,strengthBasis)
            elif DiffNERobotModel.is_terrain_file(floor):
                self.floor=self.FloorType(self.body,floor[0],floor[1],floor[2],self.Vec3Type(self.g),nrB,mu,strengthBasis)
            else: raise RuntimeError("Unsupported floor type!")
            self.floor._externalForces=ees
            self.floor.writeEndEffectorVTK("EndEffector.vtk")
        self.floor.writeVTK("MDPFloor.vtk")
        if self.sim is not None:
            if self.granular:
                self.sim.setC2GranularWrenchConstructor(self.floor)
            else: self.sim.setC2EnvWrenchConstructor(self.floor)
        self.setup_mesh()
            
    def setup_mesh(self):
        self.mesh=self.sim.getTerrainMesh()
        
        self.mesh_vss=[]
        for v in self.mesh.getV():
            for vi in [v[0],v[1],v[2]]:
                self.mesh_vss.append(vi)
                
        self.mesh_nss=[]
        for n in self.mesh.getN():
            for ni in [n[0],n[1],n[2]]:
                self.mesh_nss.append(ni)
                
        self.mesh_iss=[]
        for i in self.mesh.getI():
            for ii in [i[0],i[1],i[2]]:
                self.mesh_iss.append(ii)
        
    def set_PD_controller(self,PCoef=1000.0,DCoef=1.0):
        if self.sim is None:
            raise RuntimeError("PD Controller can only be set after calling create_simulator!")
        #setup PDController coef
        N=self.num_DOF()
        self.PDCoef=[0.0 for i in range(N*2)]
        for j in range(self.body.nrJ()):
            J=self.body.joint(j)
            if not J.isRoot(self.body):  #cannot apply torque to root joint
                for k in range(J._offDOF,J._offDOF+J.nrDOF()):
                    self.PDCoef[k]=PCoef
                    self.PDCoef[N+k]=DCoef
        self.qdq0=self.VecType(self.qdq.toList())
        
    def num_DOF(self):
        return self.body.nrDOF()
       
    def print_DMap(self):
        for dMap in self.DMap:
            dMap.print_DMap(self.robot,self.body)
        
    def get_klampt_DOF(self,diffNEq):
        T=self.body.getT(DNE.Vecd(diffNEq))
        return [m.map_to_klampt(self,diffNEq,T) for m in self.DMap]
        
    def get_klampt_link_pose(self,name):
        for i in range(self.robot.numLinks()):
            if self.robot.link(i).getName()==name:
                return self.robot.link(i).getTransform()
        return None

    def get_DiffNE_link_pose(self,name):
        for i in range(self.body.nrJ()):
            if self.body.joint(i)._name.startswith(name):
                T=self.body.getT(self.qdq.castToDouble())
                R=[T[0,i*4+0],T[1,i*4+0],T[2,i*4+0],
                   T[0,i*4+1],T[1,i*4+1],T[2,i*4+1],
                   T[0,i*4+2],T[1,i*4+2],T[2,i*4+2]]
                t=[T[0,i*4+3],T[1,i*4+3],T[2,i*4+3]]
                return R,t
        return None
        
    def create_simulator(self,g=[0,0,-9.81],mode=DNE.FORWARD_RK1F,accuracy=64,granular=False):
        self.g=g
        if self.sim is not None:
            raise RuntimeError("Cannot call create_simulator twice!")
        self.ops=DNE.Options()
        if accuracy==64:
            self.Vec3Type=DNE.Vec3d
            self.Vec4Type=DNE.Vec4d
            self.VecType=DNE.Vecd
            self.MatType=DNE.Matd
            self.granular=granular
            if granular:
                self.FloorType=DNE.C2GranularWrenchConstructord
            else: self.FloorType=DNE.C2EnvWrenchConstructord
            self.sim=DNE.MDPSimulatord(self.body,self.ops,self.Vec3Type(g),mode)
        elif accuracy==128:
            self.Vec3Type=DNE.Vec3q
            self.Vec4Type=DNE.Vec4q
            self.VecType=DNE.Vecq
            self.MatType=DNE.Matq
            self.granular=granular
            if granular:
                self.FloorType=DNE.C2GranularWrenchConstructorq
            else: self.FloorType=DNE.C2EnvWrenchConstructorq
            self.sim=DNE.MDPSimulatorq(self.body,self.ops,self.Vec3Type(g),mode)
        elif accuracy>128:
            DNE.mpreal.set_prec(accuracy)
            self.Vec3Type=DNE.Vec3m
            self.Vec4Type=DNE.Vec4m
            self.VecType=DNE.Vecm
            self.MatType=DNE.Matm
            self.granular=granular
            if granular:
                self.FloorType=DNE.C2GranularWrenchConstructorm
            else: self.FloorType=DNE.C2EnvWrenchConstructorm
            self.sim=DNE.MDPSimulatorm(self.body,self.ops,self.Vec3Type(g),mode)
        else:
            print("Unsupported accuracy: ",accuracy)
            assert False
        self.qdq=self.VecType(self.qdq.toList())
        self.set_option("callback",False)
        self.sim.reset(self.ops)
        if self.floor is not None:
            self.sim.setC2FloorWrenchConstructor(self.floor)
        
    def set_option(self,name,value):
        if self.ops is None:
            return
        self.ops.setOption(self.sim,name,value)
        
    def print_vars(self):
        self.ops.printVars()
        
    def simulate(self,dt,qdq=None,tau=None,Dqdq=True,Dtau=True):
        if self.sim is not None:
            if qdq is not None:
                qdq=self.VecType(qdq)
                update_self_qdq=False
            else: 
                qdq=self.qdq
                update_self_qdq=True
            if tau is None and self.qdq0 is not None:
                tau=[None for i in range(self.num_DOF())]
                #build PDController command
                N=self.num_DOF()
                for i in range(N):
                    tau[i] =(self.qdq0[i  ]-self.qdq[i  ])*self.PDCoef[i  ]
                    tau[i]+=(self.qdq0[N+i]-self.qdq[N+i])*self.PDCoef[N+i]
                tau=self.VecType(tau)
            else: tau=self.VecType(tau)
            qdq,Dqdqs,Dtaus=self.sim.step(dt,tau,qdq,Dqdq,Dtau)
            if update_self_qdq:
                self.qdq=qdq
            return  qdq.castToDouble().toList(),   \
                    Dqdqs.castToDouble().toList() if Dqdq else None,    \
                    Dtaus.castToDouble().toList() if Dtau else None
        elif self.qdqSeq is not None:
            self.qdq=self.qdqSeq[(self.qdqSeqId+1)%len(self.qdqSeq)]
            self.qdqSeqId=(self.qdqSeqId+1)%len(self.qdqSeq)
            return self.qdq.toList(),None,None
        else: 
            raise RuntimeError("Cannot simulate without calling create_simulator!")
            return None,None,None
    
    def simulate_batched(self,dt,qdqs,taus=None,Dqdq=True,Dtau=True,mode=None):
        if self.sim is not None:
            if mode is not None:
                tmpMode=self.sim.getMode()
                self.sim.setMode(mode)
            qdqs=[self.VecType(qdq) for qdq in qdqs]
            if taus is None and self.qdq0 is not None:
                taus=[[None for i in range(self.num_DOF())] for qdq in qdqs]
                #build PDController command
                N=self.num_DOF()
                for tau,qdq in zip(taus,qdqs):
                    for i in range(N):
                        tau[i] =(self.qdq0[i  ]-self.qdq[i  ])*self.PDCoef[i  ]
                        tau[i]+=(self.qdq0[N+i]-self.qdq[N+i])*self.PDCoef[N+i]
                taus=[self.VecType(tau) for tau in taus]
            elif taus is not None:
                taus=[self.VecType(tau) for tau in taus]
            qdqs,Dqdqs,Dtaus=self.sim.stepBatched(dt,taus,qdqs,Dqdq,Dtau)
            if mode is not None:
                self.sim.setMode(tmpMode)
            return  [qdq.castToDouble().toList() for qdq in qdqs],  \
                    [Dqdqi.castToDouble().toList() for Dqdqi in Dqdqs] if Dqdq else None,    \
                    [Dtaui.castToDouble().toList() for Dtaui in Dtaus] if Dtau else None
        else: 
            raise RuntimeError("Cannot simulate_batched without calling create_simulator!")
            return None,None,None
    
    def test_derivative(self,dt=0.001,mode=None):
        tmpMode=self.sim.getMode()
        if mode is not None:
            self.sim.setMode(mode)
            
        if mode==DNE.INVERSE_I or mode==DNE.INVERSE_LF or mode==DNE.INVERSE_BACKWARD_RK1F or mode==DNE.BACKWARD_RK1F:
            nrInput=self.num_DOF()*3
        else: nrInput=self.num_DOF()*2
            
        DELTA=1e-6
        assert self.sim is not None
        delta=[random.uniform(-1,1) for i in range(nrInput)]
        deltat=[random.uniform(-1,1) for i in range(self.num_DOF())]
        tau=[random.uniform(-1,1) for i in range(self.num_DOF())]
        if mode is None:
            qdq=self.qdq.castToDouble().toList()
        else: 
            qdq=[random.uniform(-1,1) for i in range(nrInput)]
        qdq2=op.add(qdq,op.mul(delta,DELTA))
        tau2=op.add(tau,op.mul(deltat,DELTA))
        
        o,DoDq,_=self.simulate(dt,qdq,tau)
        o2,_,_=self.simulate(dt,qdq2,tau,False,False)
        dirDeriv=op.matvecmul(DoDq,delta)
        numDeriv=op.mul(op.sub(o2,o),1/DELTA)
        print("DoDq        : ",op.norm(dirDeriv)," Err: ",op.norm(op.sub(dirDeriv,numDeriv)))
        o,_,DoDt=self.simulate(dt,qdq,tau)
        o2,_,_=self.simulate(dt,qdq,tau2,False,False)
        dirDeriv=op.matvecmul(DoDt,deltat)
        numDeriv=op.mul(op.sub(o2,o),1/DELTA)
        print("DoDt        : ",op.norm(dirDeriv)," Err: ",op.norm(op.sub(dirDeriv,numDeriv)))
    
        oss,DoDq,_=self.simulate_batched(dt,[qdq,qdq2],[tau,tau])
        dirDeriv=op.matvecmul(DoDq[0],delta)
        numDeriv=op.mul(op.sub(oss[1],oss[0]),1/DELTA)
        print("DoDq-Batched: ",op.norm(dirDeriv)," Err: ",op.norm(op.sub(dirDeriv,numDeriv)))
        oss,_,DoDt=self.simulate_batched(dt,[qdq,qdq],[tau,tau2])
        dirDeriv=op.matvecmul(DoDt[0],deltat)
        numDeriv=op.mul(op.sub(oss[1],oss[0]),1/DELTA)
        print("DoDq-Batched: ",op.norm(dirDeriv)," Err: ",op.norm(op.sub(dirDeriv,numDeriv)))
        if mode is not None:
            self.sim.setMode(tmpMode)

    def set_threads(self,N):
        DNE.setNumThreads(N)
    
    def set_joint_angle(self,name,value):
        for j in range(self.body.nrJ()):
            if self.body.joint(j)._name.startswith(name):
                self.qdq[self.body.joint(j)._offDOF]=value
                return
        print("Cannot find joint: ",name)
        assert False
    
    def update_world(self):
        qdq=self.qdq.castToDouble().toList()[0:self.num_DOF()]
        self.robot.setConfig(self.get_klampt_DOF(qdq))
        
    def display(self):
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK,GL.GL_LINE)
        klampt.vis.gldraw.setcolor(TABLE_COLOR[0],TABLE_COLOR[1],TABLE_COLOR[2])
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, self.mesh_vss)
        GL.glNormalPointer(GL.GL_FLOAT, 0, self.mesh_nss)
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.mesh_iss), GL.GL_UNSIGNED_INT, self.mesh_iss)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
        
    @staticmethod
    def fori(val):
        return isinstance(val,float) or isinstance(val,int)
        
    @staticmethod
    def is_plane(floor):
        return len(floor)==4 and DiffNERobotModel.fori(floor[0]) and DiffNERobotModel.fori(floor[1]) and DiffNERobotModel.fori(floor[2]) and DiffNERobotModel.fori(floor[3])
    
    @staticmethod
    def is_terrain_file(floor):
        return len(floor)==3 and isinstance(floor[0],str) and isinstance(floor[1],int) and DiffNERobotModel.fori(floor[2])