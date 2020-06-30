import matplotlib.pyplot as plt
import utils as op

def read_force(filepath):
    with open(filepath) as fp:
        line=fp.readline()
        start=False
        time=[]
        foots=[]
        newFrm=True
        fid=0
        while line:
            if newFrm:
                time.append(float(line))
                newFrm=False
            elif len(line)<=1:
                newFrm=True
                fid=0
            else:
                f=line.split(',')
                while len(foots)<=fid:
                    foots.append([])
                foots[fid].append([float(f[0]),float(f[1]),float(f[2])])
                fid+=1
            line=fp.readline()
    return time,foots

def plot_force(filepath,offset=0):
    time,foots=read_force(filepath)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for idF,fF in enumerate(foots):
        ax.plot(time,[op.norm(ff) for ff in fF],label='foot'+str(idF))
    ax.set_xlabel('time')
    plt.legend(loc='center right')
    plt.title('Force on Foots')
    plt.savefig(filepath[0:len(filepath)-4]+'.pdf')
    plt.show()

if __name__=='__main__':
    plot_force("robosimian_bilevel_cmp/solutionBilevel/force.txt",5)
    plot_force("robosimian_bilevel_cmp/solutionBilevelNE/force.txt",5)
