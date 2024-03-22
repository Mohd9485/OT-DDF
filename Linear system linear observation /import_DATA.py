import numpy as np
import matplotlib.pyplot as plt
import torch
import time as Time

import matplotlib
import sys
from SIR import SIR
from KF import KF
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
    return H @ (x)


def A(x,t=0):
    return F@ (x)



load = np.load('DATA_file.npz',allow_pickle=True) # h(x) = x


data = {}
for key in load:
    print(key)
    data[key] = load[key]
    
    
time = data['time']
X_true = data['X_true']
Y_true = data['Y_true']
X_OTPF = data['X_OTPF']
X_OT_DDF_dic = data['X_OT_DDF_dic'].tolist()
Noise = data['Noise']
Window = data['Window']
H = data['H']
F = data['F']
burnin = data['burnin']
parameters = data['parameters'].tolist()

AVG_SIM = X_OTPF.shape[0]
J = X_OTPF.shape[3]
SAMPLE_SIZE = X_OTPF.shape[3]
SAMPLE_SIZE = 500
L = X_true.shape[2]
tau = 1e-1 # timpe step 
[noise,sigmma,sigmma0,gamma,x0_amp] = Noise


X0 = X_OTPF[:,0,:,:]
X_KF = KF(Y_true,X0.mean(axis=2,keepdims=True),F,H,time,tau,Noise)
X_SIR  = SIR(X_true,Y_true,X0,A,h,time,tau,Noise)
X_EnKF  = EnKF(X_true,Y_true,X0,A,h,time,tau,Noise)

T = len(time)

#%%

mse_EnKF = ((X_EnKF.mean(axis=3)-X_true)*(X_EnKF.mean(axis=3)-X_true)).mean(axis=(0,2))
mse_KF = ((X_KF.mean(axis=3)-X_true)*(X_KF.mean(axis=3)-X_true)).mean(axis=(0,2))
mse_OTPF = ((X_OTPF.mean(axis=3)-X_true)*(X_OTPF.mean(axis=3)-X_true)).mean(axis=(0,2))
mse_SIR = ((X_SIR.mean(axis=3)-X_true)*(X_SIR.mean(axis=3)-X_true)).mean(axis=(0,2))

mse_OT_DDF = {}
for window in Window:
    mse_OT_DDF[window.astype('str')] = (( X_OT_DDF_dic[window.astype('str')].mean(axis=3)-X_true)
                                        *( X_OT_DDF_dic[window.astype('str')].mean(axis=3)-X_true)).mean(axis=(0,2))


mse_OT_DDF_avg = []
for window in Window:
    mse_OT_DDF_avg.append(np.mean(mse_OT_DDF[window.astype('str')][100:]))



#%%


plot_particle = 100#J
for k in range(1):
    plt.figure(figsize=(20,7.2)) 
    grid = plt.GridSpec(10, 2, wspace =0.12, hspace = 0.15)
    
    for l in range(1,L):
        g1 = plt.subplot(grid[0:2, l-1])
        plt.plot(time[burnin:],X_KF[k,burnin:,l,],color='C9',lw=2)
        plt.plot(time[burnin:],X_true[k,burnin:,l],'k--',label='True state',lw=2)
        plt.xlabel('time')

        if l==1:
            plt.ylabel('KF',fontsize=16)
            plt.legend(loc=3,fontsize=14)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        
            
    for l in range(1,L):
        g1 = plt.subplot(grid[2:4, l-1])
        plt.plot(time[burnin:],X_EnKF[k,burnin:,l,:plot_particle],'g',alpha = 0.1)
        plt.plot(time[burnin:],X_true[k,burnin:,l],'k--',label='True state',lw=2)
        plt.xlabel('time')
        if l==1:
            plt.ylabel('EnKF',fontsize=16)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        
    
    # plt.figure()        
    for l in range(1,L):
        #for j in range(AVG_SIM): 
        g1 = plt.subplot(grid[4:6, l-1])
        plt.plot(time[burnin:],X_OTPF[k,burnin:,l,:plot_particle],'r',alpha = 0.1)
        plt.plot(time[burnin:],X_true[k,burnin:,l],'k--',lw=2)
        plt.xlabel('time',fontsize=16)
        # plt.ylabel('OT X'+str(l+1))
        if l==1:
            plt.ylabel('OTPF',fontsize=16)
        # plt.show() 
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
      
    for l in range(1,L):
        #for j in range(AVG_SIM):  
        g1 = plt.subplot(grid[6:8, l-1])
        plt.plot(time[burnin:],X_OT_DDF_dic['50'][k,burnin:,l,:plot_particle],'C0',alpha = 0.1)
        plt.plot(time[burnin:],X_true[k,burnin:,l],'k--',lw=2)
        plt.xlabel('time',fontsize=16)
        # plt.ylabel('OT X'+str(l+1))
        if l==1:
            plt.ylabel('OT-DDF',fontsize=16)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
                    
    # plt.figure()   
    for l in range(1,L):
        g1 = plt.subplot(grid[8:, l-1])
# =============================================================================
#         plt.subplot(4,L,l+10)
# =============================================================================
        plt.plot(time[burnin:],X_SIR[k,burnin:,l,:plot_particle],'b',alpha = 0.1)
        plt.plot(time[burnin:],X_true[k,burnin:,l],'k--',lw=2)
        plt.xlabel('time',fontsize=16)
        # plt.ylabel('SIR X'+str(l+1))
        #plt.legend()
        if l==1:
            plt.ylabel('SIR',fontsize=16)

Window_plot = np.array([1,10,50])
g1 = plt.subplot(grid[:5, 1:])
plt.semilogy(time[burnin:],mse_EnKF[burnin:],'g--',lw=2,label="EnKF",alpha=0.7)
plt.semilogy(time[burnin:],mse_OTPF[burnin:],'r-.',lw=2,label="OTPF" ,alpha=0.7)
plt.semilogy(time[burnin:],mse_SIR[burnin:],'b:',lw=2,label="SIR" ,alpha=0.7)

for window in Window_plot:
    if window==50:
        plt.semilogy(time[burnin:], mse_OT_DDF[window.astype('str')][burnin:],lw=2,label='OT-DDF ($w$=' + window.astype('str')+')',color='C0')
    elif window==1:
        plt.semilogy(time[burnin:], mse_OT_DDF[window.astype('str')][burnin:],lw=2,label='OT-DDF ($w$=' + window.astype('str')+')',alpha=0.7,color='C1')
    else:
        plt.semilogy(time[burnin:], mse_OT_DDF[window.astype('str')][burnin:],lw=2,label='OT-DDF ($w$=' + window.astype('str')+')',alpha=0.7,color='C4')

plt.xlabel('time',fontsize=16)
plt.ylabel('MSE',fontsize=15)
plt.legend(fontsize=12,bbox_to_anchor=(0.95, 0.8))


g1 = plt.subplot(grid[6:, 1:])
plt.axhline(y= np.mean(mse_EnKF[burnin:]),color = 'g',linestyle='--',label="EnKF",lw=2)
plt.axhline(y= np.mean(mse_OTPF[burnin:]),color ='r',linestyle = '-.',label="OTPF" ,lw=2)
plt.axhline(y = np.mean(mse_SIR[burnin:]),color='b',linestyle =':',label="SIR" ,lw=2)
plt.plot(Window,mse_OT_DDF_avg,lw=2,label='OT-DDF',color='C0')
plt.xlabel('$w $',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.yscale('log')
plt.legend(fontsize=14,bbox_to_anchor=(0.95, 0.8))
plt.show()