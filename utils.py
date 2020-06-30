import numpy as np
import math

def sqrt_safe(x):
    return math.sqrt(x)

def sin_safe(x):
    return math.sin(x)

def cos_safe(x):
    return math.cos(x)
    
def asin_safe(x):
    return math.asin(x)

def acos_safe(x):
    return math.acos(x)
    
def atan2_safe(y,x):
    return math.atan2(y,x)

def floor_safe(x):
    return math.floor(x)

def ceil_safe(x):
    return math.ceil(x)

def log_safe(x):
    return math.log(x)

def is_float(a):
    return  isinstance(a,float) or isinstance(a,np.float64) or  \
            isinstance(a,np.float128) or isinstance(a,int)

def is_scalar(a):
    return not isinstance(a,list) and not isinstance(a,tuple)

def cross(a,b):
    axb=[None,None,None]
    axb[0]=(a[1]*b[2])-(a[2]*b[1])
    axb[1]=(a[2]*b[0])-(a[0]*b[2])
    axb[2]=(a[0]*b[1])-(a[1]*b[0])
    return axb

def cross_mat(v):
    #    0,-v[2], v[1],
    # v[2],    0,-v[0],
    #-v[1],val v[0],    0;
    ret=[[0.0 for c in range(3)] for r in range(3)]
    ret[0][1]=-v[2]
    ret[0][2]= v[1]
    ret[1][0]= v[2]
    ret[1][2]=-v[0]
    ret[2][0]=-v[1]
    ret[2][1]= v[0]
    return ret

def abs_max(a):
    if is_scalar(a):
        return abs(a)
    ret=0.0
    for v in a:
        val=abs_max(v)
        if val>ret:
            ret=val
    return ret

def dot(a,b):
    if is_scalar(a):
        return a*b
    elif is_scalar(b):
        return a*b
    else:
        ret=0
        for ai,bi in zip(a,b):
            ret+=dot(ai,bi)
        return ret
    
def shape(m):
    return (len(m),len(m[0]))
    
def norm(a):
    return sqrt_safe(dot(a,a))
    
def mul(a,b):
    if a is None:
        return b
    elif b is None:
        return a
    elif is_scalar(a):
        if is_scalar(b):
            return a*b
        else: return [mul(a,bi) for bi in b]
    elif is_scalar(b):
        if is_scalar(a):
            return a*b
        else: return [mul(ai,b) for ai in a]
    else: return [mul(ai,bi) for ai,bi in zip(a,b)]
    
def matmul(a,b):
    assert len(a[0])==len(b)
    ret=[[0.0 for c in range(len(b[0]))] for r in range(len(a))]
    for r in range(len(a)):
        for c in range(len(b[0])):
            for k in range(len(a[0])):
                ret[r][c]+=a[r][k]*b[k][c]
    return ret

def catcol(a,b):
    assert len(a)==len(b)
    return [ai+bi for ai,bi in zip(a,b)]

def matTmul(a,b):
    assert len(a)==len(b)
    ret=[[0.0 for c in range(len(b[0]))] for r in range(len(a[0]))]
    for r in range(len(a[0])):
        for c in range(len(b[0])):
            for k in range(len(a)):
                ret[r][c]+=a[k][r]*b[k][c]
    return ret

def matvecmul(a,b):
    assert len(a[0])==len(b)
    ret=[0.0 for r in range(len(a))]
    for r in range(len(a)):
        for c in range(len(b)):
                ret[r]+=a[r][c]*b[c]
    return ret

def matTvecmul(a,b):
    assert len(a)==len(b)
    ret=[0.0 for r in range(len(a[0]))]
    for r in range(len(a[0])):
        for c in range(len(b)):
                ret[r]+=a[c][r]*b[c]
    return ret

def transpose(a):
    ret=[[0.0 for c in range(len(a))] for r in range(len(a[0]))]
    for r in range(len(a)):
        for c in range(len(a[0])):
            ret[c][r]=a[r][c]
    return ret

def muldiagAB(a,b):
    return [mul(a,b[i]) for i in range(b)]

def add(a,b):
    if a is None:
        return b
    elif b is None:
        return a
    elif is_scalar(a):
        if is_scalar(b):
            return a+b
        else: return [add(a,bi) for bi in b]
    elif is_scalar(b):
        if is_scalar(a):
            return a+b
        else: return [add(ai,b) for ai in a]
    else: return [add(ai,bi) for ai,bi in zip(a,b)]

def sub(a,b):
    if a is None:
        return b
    elif b is None:
        return a
    elif is_scalar(a):
        if is_scalar(b):
            return a-b
        else: return [sub(a,bi) for bi in b]
    elif is_scalar(b):
        if is_scalar(a):
            return a-b
        else: return [sub(ai,b) for ai in a]
    else: return [sub(ai,bi) for ai,bi in zip(a,b)]
    