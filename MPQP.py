import ctypes as ct
import numpy as np

class MPQP:
    def __init__(self,PREFIX="."):
        self.lib=ct.CDLL(PREFIX+"/libMPQP.so")
        #LP
        self.lib.solveLP.argtypes=  \
        (ct.c_double,#mu
         ct.c_int,#n
         ct.POINTER(ct.c_double),#c
         ct.POINTER(ct.c_double),#w
         ct.POINTER(ct.c_double),#dwdc
         ct.POINTER(ct.c_double),#initw
         ct.c_int)#prec
        self.lib.solveQP.restype=ct.c_bool
        #QP
        self.lib.solveQP.argtypes=  \
        (ct.c_double,#mu
         ct.c_int,#n
         ct.POINTER(ct.c_double),#g
         ct.POINTER(ct.c_double),#H
         ct.POINTER(ct.c_double),#w
         ct.POINTER(ct.c_double),#dwdg
         ct.POINTER(ct.c_double),#dwdh
         ct.POINTER(ct.c_double),#initw
         ct.c_int)#prec
        
    def len(self,obj):
        if isinstance(obj,np.ndarray):
            return np.prod(obj.shape)
        else: 
            assert isinstance(obj,list)
            return len(obj)
        
    def to_pointer(self,obj):
        if obj is None:
            return ct.POINTER(ct.c_double)()
        elif isinstance(obj,np.ndarray):
            if obj.dtype!=np.float64:
                obj=obj.astype(np.float64).flatten().tolist()
                return (ct.c_double*len(obj))(*obj)
            else: return np.ctypeslib.as_ctypes(obj.flatten())
        else: 
            assert isinstance(obj,list)
            obj=np.array(obj,dtype=np.float64).tolist()
            return (ct.c_double*len(obj))(*obj)
        
    def solve_LP(self,c,initw=None,mu=0.001,prec=512):
        n=self.len(c)
        w=(ct.c_double*n)()
        dwdc=(ct.c_double*(n*n))()
        succ=self.lib.solveLP(mu,n,self.to_pointer(c),w,dwdc,self.to_pointer(initw),prec)
        return  np.array([w[i] for i in range(n)]), \
                np.array([[dwdc[i*n+j] for j in range(n)] for i in range(n)]).transpose()

    def solve_QP(self,g,H,initw=None,mu=0.0001,prec=512):
        n=self.len(g)
        assert n*n==self.len(H)
        w=(ct.c_double*n)()
        dwdg=(ct.c_double*(n*n))()
        dwdH=(ct.c_double*(n*n*n))()
        succ=self.lib.solveQP(mu,n,self.to_pointer(g),self.to_pointer(H),w,dwdg,dwdH,self.to_pointer(initw),prec)
        return  np.array([w[i] for i in range(n)]), \
                np.array([[dwdg[i*n+j] for j in range(n)] for i in range(n)]).transpose(),  \
                np.array([[dwdH[i*n+j] for j in range(n)] for i in range(n*n)]).transpose()

def example_solve_LP():
    mpqp=MPQP(".")
    c=np.array([1,2],dtype=int)

    w,dwdc=mpqp.solve_LP(c,prec=0)
    print("Solve LP Double: w=%s \ndwdc=\n%s\n"%(str(w),str(dwdc)))
    
    w2,dwdc2=mpqp.solve_LP(c,prec=512)
    print("Solve LP MPFR: w=%s \ndwdc=\n%s\n"%(str(w2),str(dwdc2)))
    
    initw=w+np.array([0.01,0.01])
    w3,dwdc3=mpqp.solve_LP(c,initw=initw,prec=512)
    print("Warm-Started Solve LP MPFR: w=%s initw=%s \ndwdc=\n%s\n"%(str(w3),str(initw),str(dwdc3)))
        
def example_solve_QP():
    mpqp=MPQP(".")
    g=np.array([1,2],dtype=int)
    H=np.array([[0.1,0.01],[0.01,0.1]],dtype=np.float32)

    w,dwdg,dwdh=mpqp.solve_QP(g,H)
    print("Solve QP Double: w=%s \ndwdg=\n%s \ndwdh=\n%s\n"%(str(w),str(dwdg),str(dwdh)))
    
    w2,dwdg2,dwdh2=mpqp.solve_QP(g,H,prec=512)
    print("Solve QP MPFR: w=%s \ndwdg=\n%s \ndwdh=\n%s\n"%(str(w2),str(dwdg2),str(dwdh2)))
    
    initw=w+np.array([0.01,0.01])
    w3,dwdg3,dwdh3=mpqp.solve_QP(g,H,initw=initw)
    print("Warm-Started Solve QP MPFR: w=%s initw=%s \ndwdg=\n%s \ndwdh=\n%s\n"%(str(w3),str(initw),str(dwdg3),str(dwdh3)))
        
if __name__=='__main__':
    example_solve_LP()
    example_solve_QP()