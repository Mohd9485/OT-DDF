"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib
from EnKF import EnKF
from SIR import SIR
from scipy.integrate import odeint

plt.close('all')

plt.rc('font', size=13)          # controls default text sizes
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
def relu(x):
    a = 0
    return np.maximum(x,a)

def h(x):
    return x[0,].reshape(1,-1)


def L63(x, t):
    """Lorenz 96 model"""
    # Setting up vector
    #L = 3
    d = np.zeros_like(x)
    sigma = 10
    r = 28
    b = 8/3
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    d[0] = sigma*(x[1]-x[0])
    d[1] = x[0]*(r-x[2])-x[1]
    d[2] = x[0]*x[1]-b*x[2]
    return d

def ML63(x, t , particles = 100):
    """Lorenz 63 model"""
    # Setting up vector
    #L = 3
    d = np.zeros_like(x)
    sigma = 10
    r = 28
    b = 8/3

    
    d[0,] = sigma*(x[1,]-x[0,])
    d[1,] = x[0,]*(r-x[2,])-x[1,]
    d[2,] = x[0,]*x[1,]-b*x[2,]
    return d


load = np.load('DATA_file.npz',allow_pickle=True)


data = {}
for key in load:
    print(key)
    data[key] = load[key]
    
    
time = data['time']
X_true = data['X_true']
Y_true = data['Y_true']

X_OT = data['X_OT']
X_OT_skip_dic = data['X_OT_skip_dic'].tolist()


MSE_OT = data['MSE_OT']

MSE_OT_skip_dic = data['MSE_OT_skip_dic'].tolist()



Noise = data['Noise']
Window = data['Window']
parameters = data['parameters']

skip = data['skip']
Odeint = data['Odeint']
#%%
AVG_SIM = X_OT.shape[0]
J = X_OT.shape[3]
SAMPLE_SIZE = X_OT.shape[3]
L = X_true.shape[2]

tau = time[1]-time[0] # timpe step 
[noise,sigmma,sigmma0,gamma,x0_amp] = Noise




X0 = X_OT[:,0,:,:]
X_SIR , MSE_SIR = SIR(X_true,Y_true,X0,ML63,h,time,tau,Noise,Odeint)
X_EnKF , MSE_EnKF = EnKF(X_true,Y_true,X0,ML63,h,time,tau,Noise,Odeint)


mse_OT_skip_avg = []
for window in Window:
    mse_OT_skip_avg.append(np.mean(MSE_OT_skip_dic[window.astype('str')][skip:]))



#%%

plot_particle = 100#J
plot_x = 2000
L=2
for k in range(1):
    plt.figure(figsize=(20,7.2)) 
    grid = plt.GridSpec(8, 2, wspace =0.12, hspace = 0.15)
    for l in range(1,L):
        g1 = plt.subplot(grid[0:2, l-1])
# =============================================================================
#         plt.subplot(4,L,l+1)
# =============================================================================
        plt.plot(time[plot_x:],X_EnKF[k,plot_x:,l,:plot_particle],'g',alpha = 0.1)
        plt.plot(time[plot_x:],X_true[k,plot_x:,l],'k--',label='True state',lw=2)
        plt.xlabel('time')
    # =============================================================================
    #     plt.ylabel('X'+str(l+1))
    # =============================================================================
        if l==1:
            plt.ylabel('EnKF',fontsize=16)
            plt.legend(loc=8,fontsize=14)
# =============================================================================
#         plt.title('X(%i)'%(l+1),fontsize=20)
# =============================================================================
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        
        if l==0:
            y_lim = 44
            plt.ylim([-y_lim,y_lim])
        elif l==1:
            y_lim = 35 
            plt.ylim([-y_lim,y_lim])
        elif l==2:
            y_lim = 88
            plt.ylim([-45,99])
        
    
    # plt.figure()        
    for l in range(1,L):
        #for j in range(AVG_SIM): 
        g1 = plt.subplot(grid[2:4, l-1])
# =============================================================================
#         plt.subplot(4,L,l+4)   
# =============================================================================
        plt.plot(time[plot_x:],X_OT[k,plot_x:,l,:plot_particle],'r',alpha = 0.1)
        plt.plot(time[plot_x:],X_true[k,plot_x:,l],'k--',lw=2)
        plt.xlabel('time',fontsize=16)
        # plt.ylabel('OT X'+str(l+1))
        if l==1:
            plt.ylabel('OTPF',fontsize=16)
        # plt.show() 
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        
        if l==0:
            y_lim = 44
            plt.ylim([-y_lim,y_lim])
        elif l==1:
            y_lim = 35 
            plt.ylim([-y_lim,y_lim])
        elif l==2:
            y_lim = 88
            plt.ylim([-45,99])
      
    for l in range(1,L):
        #for j in range(AVG_SIM):  
        g1 = plt.subplot(grid[4:6, l-1])
# =============================================================================
#         plt.subplot(4,L,l+7)   
# =============================================================================
        plt.plot(time[plot_x:],X_OT_skip_dic['50'][k,plot_x:,l,:plot_particle],'C0',alpha = 0.1)
        plt.plot(time[plot_x:],X_true[k,plot_x:,l],'k--',lw=2)
        plt.xlabel('time',fontsize=16)
        # plt.ylabel('OT X'+str(l+1))
        if l==1:
            plt.ylabel('OT-DDF',fontsize=16)
        # plt.show() 
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        
        if l==0:
            y_lim = 44
            plt.ylim([-y_lim,y_lim])
        elif l==1:
            y_lim = 35 
            plt.ylim([-y_lim,y_lim])
        elif l==2:
            y_lim = 88
            plt.ylim([-45,99]) 
                    
    # plt.figure()   
    for l in range(1,L):
        g1 = plt.subplot(grid[6:, l-1])
# =============================================================================
#         plt.subplot(4,L,l+10)
# =============================================================================
        plt.plot(time[plot_x:],X_SIR[k,plot_x:,l,:plot_particle],'b',alpha = 0.1)
        plt.plot(time[plot_x:],X_true[k,plot_x:,l],'k--',lw=2)
        plt.xlabel('time',fontsize=16)
        # plt.ylabel('SIR X'+str(l+1))
        #plt.legend()
        if l==1:
            plt.ylabel('SIR',fontsize=16)
        # plt.show()
    
        if l==0:
            y_lim = 44
            plt.ylim([-y_lim,y_lim])
        elif l==1:
            y_lim = 35 
            plt.ylim([-y_lim,y_lim])
        elif l==2:
            y_lim = 88
            plt.ylim([-45,99])




Window_plot = np.array([10,50,200])
g1 = plt.subplot(grid[:4, 1:])
plt.semilogy(time[skip:3000],MSE_EnKF[skip:3000],'g--',lw=2,label="EnKF",alpha=0.7)
plt.semilogy(time[skip:3000],MSE_OT[skip:3000],'r-.',lw=2,label="OTPF" ,alpha=0.7)
# =============================================================================
# plt.plot(time,mse_OT_without,'c-.',label="$OT_{without EnKF}$" )
# =============================================================================
plt.semilogy(time[skip:3000],MSE_SIR[skip:3000],'b:',lw=2,label="SIR" ,alpha=0.7)

for window in Window_plot:
    if window==50:
        plt.semilogy(time[skip:3000], MSE_OT_skip_dic[window.astype('str')][skip:3000],lw=2,label='OT-DDF ($w$=' + window.astype('str')+')',color='C0')
    elif window==10:
        plt.semilogy(time[skip:3000], MSE_OT_skip_dic[window.astype('str')][skip:3000],lw=2,label='OT-DDF ($w$=' + window.astype('str')+')',alpha=0.7,color='C1')
    else:
        plt.semilogy(time[skip:3000], MSE_OT_skip_dic[window.astype('str')][skip:3000],lw=2,label='OT-DDF ($w$=' + window.astype('str')+')',alpha=0.7,color='C4')

plt.xlabel('time',fontsize=16)
plt.ylabel('MSE',fontsize=15)
plt.legend(fontsize=12,bbox_to_anchor=(0.95, 0.8))


g1 = plt.subplot(grid[5:, 1:])
plt.axhline(y= np.mean(MSE_EnKF[skip:]),color = 'g',linestyle='--',label="EnKF",lw=2)
plt.axhline(y= np.mean(MSE_OT[skip:]),color ='r',linestyle = '-.',label="OTPF" ,lw=2)
plt.axhline(y = np.mean(MSE_SIR[skip:]),color='b',linestyle =':',label="SIR" ,lw=2)
plt.plot(Window*tau,mse_OT_skip_avg,lw=2,label='OT-DDF',color='C0')
plt.xlabel('w $*$ dt',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.yscale('log')
plt.legend(fontsize=14,bbox_to_anchor=(0.95, 0.8))
plt.show()
