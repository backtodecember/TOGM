import matplotlib.pyplot as plt

def read_convergence(filepath):
    with open(filepath) as fp:
        line=fp.readline()
        start=False
        obj=[]
        fea=[]
        while line:
            if start:
                if line.find('Starting feasible mode')>=0:
                    line=fp.readline()
                    continue
                while line.endswith('\n'):
                    line=line[0:len(line)-1]
                terms=[s for s in line.split(' ') if len(s)>0]
                if len(terms)>=3:
                    try:
                        obj.append(float(terms[1]))
                        fea.append(float(terms[2]))
                    except:
                        break
                else:
                    break
            elif line.find('FeasError')>=0:
                start=True
                line=fp.readline()
            line=fp.readline()
    return obj,fea

def plot_convergence(filepath,offset=0):
    obj,fea=read_convergence(filepath)
    obj=obj[offset:len(obj)]
    fea=fea[offset:len(fea)]
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot([i for i in range(len(obj))],obj,color='tab:blue',label='objective')
    ax.plot([i for i in range(len(fea))],fea,color='tab:orange',label='feasibility')
    ax.set_xlabel('#iter')
    plt.legend(loc='center right')
    plt.title('Convergence History - '+filepath[0:len(filepath)-4])
    plt.savefig(filepath[0:len(filepath)-4]+'.pdf')
    plt.show()

if __name__=='__main__':
    plot_convergence("NE.txt",5)
    plot_convergence("PBD.txt",5)
