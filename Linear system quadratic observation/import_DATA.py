import numpy as np
import matplotlib.pyplot as plt
import torch
import time as Time
import matplotlib
import sys
from SIR import SIR
from EnKF import EnKF

plt.close('all')


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=13)          # controls default text sizes
#plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
def relu(x):
    a = 0
    return np.maximum(x,a)

# Choose h(x) here, the observation rule
def h(x):
    return x[0].reshape(1,-1)*x[0].reshape(1,-1)



def A(x,t=0):
    return F@ (x)


load = np.load('DATA_file.npz',allow_pickle=True) # window 1,10,20,50


data = {}
for key in load:
    print(key)
    data[key] = load[key]
    
    
time = data['time']
X_true = data['X_true']
Y_true = data['Y_true']
X_OTPF = data['X_OTPF']
Window = data['Window']

X_OT_DDF_dic = data['X_OT_DDF_dic'].tolist()
Noise = data['Noise']
H = data['H']
F = data['F']
parameters = data['parameters'].tolist()
burnin = data['burnin']

AVG_SIM = X_OTPF.shape[0]
J = X_OTPF.shape[3]
SAMPLE_SIZE = X_OTPF.shape[3]
SAMPLE_SIZE = 500
L = X_true.shape[2]
tau = 1e-1 # timpe step 
[noise,sigmma,sigmma0,gamma,x0_amp] = Noise



X0 = X_OTPF[:,0,:,:]
X_SIR  = SIR(X_true,Y_true,X0,A,h,time,tau,Noise)
X_EnKF = EnKF(X_true,Y_true,X0,A,h,time,tau,Noise)


T = len(time)

true_particles = int(J*100/1)

sampled = int(1e4/1)
device = 'mps'

sigma = 1/(2*1**2)


#%%
def kernel(X,Y,method = 'linear', degree=2, offset=0.5, sigma=1, beta=1.5):
    return torch.exp(-sigma*torch.cdist(X.T,Y.T)*torch.cdist(X.T,Y.T))

def MMD(XY, XY_target, kernel,sigma):
    XY = XY.to(device)
    XY_target = XY_target.to(device)
    return torch.sqrt(kernel(XY,XY,sigma=sigma).mean() + kernel(XY_target,XY_target,sigma=sigma).mean() - 2*kernel(XY,XY_target,sigma=sigma).mean())


X0 = np.zeros((AVG_SIM,L,true_particles))
for k in range(AVG_SIM):    
    X0[k,] = x0_amp*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),true_particles))


X = SIR(X_true,Y_true,X0,A,h,time,tau,Noise)


mean =  X.mean(axis=3,keepdims=True)
std = X.std(axis=3,keepdims=True)


X = (X-mean)/std
X_EnKF = (X_EnKF-mean)/std
X_SIR = (X_SIR-mean)/std
X_OTPF = (X_OTPF-mean)/std

for window in Window:
    X_OT_DDF_dic[window.astype('str')] = (X_OT_DDF_dic[window.astype('str')]-mean)/std

print('done')   
X = torch.from_numpy(X).to(torch.float32)
X_EnKF = torch.from_numpy(X_EnKF).to(torch.float32)
X_SIR = torch.from_numpy(X_SIR).to(torch.float32)
X_OTPF = torch.from_numpy(X_OTPF).to(torch.float32)


mmd_EnKF = []
mmd_SIR = []
mmd_OTPF = []
mmd_OT_DDF = {}
for window in Window:
    mmd_OT_DDF[window.astype('str')] = []
    
start_time = Time.time()
    
for i in range(len(time)):
        print('dim : ',L, ' time :', i)
        result_enkf = 0
        result_sir = 0
        result_ot = 0
        result_ot_burnin = {}
        for window in Window:
            result_ot_burnin[window.astype('str')] = 0
            
        
        for j in range(AVG_SIM):
            x = X[j,i,:,torch.randint(0,X.shape[3],(sampled,))]
# =============================================================================
#             quantile = torch.quantile(torch.cdist(X[j,i,:,:J].T,X_OT[j,i].T).reshape(1,-1),q=0.25).item()
#             sigma = 1/(2*quantile**2)
#             print(quantile)
# =============================================================================
            
            result_enkf += MMD(x, X_EnKF[j,i], kernel,sigma)
            result_sir += MMD(x, X_SIR[j,i], kernel,sigma)
            result_ot += MMD(x, X_OTPF[j,i], kernel,sigma)
            
            for window in Window:
                x_ot_burnin = X_OT_DDF_dic[window.astype('str')]
                
                x_ot_burnin = torch.from_numpy(x_ot_burnin).to(torch.float32)
                
                result_ot_burnin[window.astype('str')] += MMD(x, x_ot_burnin[j,i], kernel,sigma)
        
        mmd_EnKF.append(result_enkf.item()/AVG_SIM)
        mmd_SIR.append(result_sir.item()/AVG_SIM)
        mmd_OTPF.append(result_ot.item()/AVG_SIM)
        
        for window in Window:
            mmd_OT_DDF[window.astype('str')].append(result_ot_burnin[window.astype('str')].item()/AVG_SIM)
        

mmd_OT_DDF_avg = []
for window in Window:
    mmd_OT_DDF_avg.append(np.mean(mmd_OT_DDF[window.astype('str')][100:]))
#%%
X_EnKF = np.array(X_EnKF)
X_SIR = np.array(X_SIR)
X_OTPF = np.array(X_OTPF)

#%%
plot_particle = 100#J
y_lim = 10.9
for k in range(1):
    k=0
    plt.figure(figsize=(20,7.2)) 
    grid = plt.GridSpec(8, 2, wspace =0.17, hspace = 0.15)

            
    for l in range(1,L):
        g1 = plt.subplot(grid[:2, l-1])
        plt.plot(time[burnin:],X_EnKF[k,burnin:,l,:plot_particle],'g',alpha = 0.1)
        plt.plot(time[burnin:],X_true[k,burnin:,l],'k--',label='True state',lw=2)
        plt.xlabel('time')

        plt.ylim([-y_lim,y_lim])
        if l==1:
            plt.ylabel('EnKF',fontsize=16)

        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        
        
    for l in range(1,L):
        g1 = plt.subplot(grid[2:4, l-1])
        plt.plot(time[burnin:],X_OTPF[k,burnin:,l,:plot_particle],'r',alpha = 0.1)
        plt.plot(time[burnin:],X_true[k,burnin:,l],'k--',lw=2)
        plt.xlabel('time',fontsize=16)
        plt.ylim([-y_lim,y_lim])
        if l==1:
            plt.ylabel('OTPF',fontsize=16)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        
        
    for l in range(1,L):
        g1 = plt.subplot(grid[4:6, l-1])
        plt.plot(time[burnin:],X_OT_DDF_dic['5'][k,burnin:,l,:plot_particle],'C0',alpha = 0.05)
        plt.plot(time[burnin:],X_true[k,burnin:,l],'k--',lw=2)
        plt.xlabel('time',fontsize=16)
        # plt.ylabel('OT X'+str(l+1))
        plt.ylim([-y_lim,y_lim])
        if l==1:
            plt.ylabel('OT-DDF',fontsize=16)
        # plt.show() 
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        
                    
    for l in range(1,L):
        g1 = plt.subplot(grid[6:8, l-1])
        plt.plot(time[burnin:],X_SIR[k,burnin:,l,:plot_particle],'b',alpha = 0.1)
        plt.plot(time[burnin:],X_true[k,burnin:,l],'k--',lw=2)
        plt.xlabel('time',fontsize=16)
        # plt.ylabel('SIR X'+str(l+1))
        #plt.legend()
        plt.ylim([-y_lim,y_lim])
        if l==1:
            plt.ylabel('SIR',fontsize=16)
        # plt.show()
    


Window_plot = np.array([1,5,20,50])
g1 = plt.subplot(grid[:4, 1:])

plt.semilogy(time[burnin:],mmd_EnKF[burnin:],'g--',label="EnKF",lw=2,alpha=0.7)
plt.semilogy(time[burnin:],mmd_OTPF[burnin:],'r-.',label="OTPF" ,lw=2,alpha=0.7)
plt.semilogy(time[burnin:],mmd_SIR[burnin:],'b:',label="SIR" ,lw=2,alpha=0.7)


for window in Window_plot:
    if window==5:
        plt.semilogy(time[burnin:], mmd_OT_DDF[window.astype('str')][burnin:],lw=2,label='OT-DDF ($w$=' + window.astype('str')+')',color='C0')
    elif window==1:
        plt.semilogy(time[burnin:], mmd_OT_DDF[window.astype('str')][burnin:],lw=2,label='OT-DDF ($w$=' + window.astype('str')+')',alpha=0.7,color='C1')
    elif window==20:
        plt.semilogy(time[burnin:], mmd_OT_DDF[window.astype('str')][burnin:],lw=2,label='OT-DDF ($w$=' + window.astype('str')+')',alpha=0.7,color='C5')
    else:
        plt.semilogy(time[burnin:], mmd_OT_DDF[window.astype('str')][burnin:],lw=2,label='OT-DDF ($w$=' + window.astype('str')+')',alpha=0.7,color='C4')

plt.xlabel('time',fontsize=16)
plt.ylabel('MMD',fontsize=15)
plt.legend(fontsize=12,bbox_to_anchor=(0.95, 0.8))
plt.show()

W_size = []
for window in Window:
    W_size.append(str(int(window)))
    


g1 = plt.subplot(grid[5:, 1:])
plt.axhline(y= np.mean(mmd_EnKF[burnin:]),color = 'g',linestyle='--',label="EnKF",lw=2)
plt.axhline(y= np.mean(mmd_OTPF[burnin:]),color ='r',linestyle = '-.',label="OTPF" ,lw=2)
plt.axhline(y = np.mean(mmd_SIR[burnin:]),color='b',linestyle =':',label="SIR" ,lw=2)
plt.plot(W_size,mmd_OT_DDF_avg,lw=2,label='OT-DDF',color='C0')
plt.xlabel('$w $',fontsize=16)
plt.ylabel('MMD',fontsize=16)
plt.yscale('log')
plt.legend(fontsize=12,bbox_to_anchor=(0.95, 0.8))
plt.show()
