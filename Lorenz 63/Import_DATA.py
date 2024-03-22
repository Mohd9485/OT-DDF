"""

@author: Mohammad Al-Jarrah
Ph.D. student, University of Washington - Seattle
email: mohd9485@uw.edu 

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

X_OTPF = data['X_OTPF']
X_OTDDF_dic = data['X_OTDDF_dic'].tolist()


MSE_OTPF = data['MSE_OTPF']

MSE_OTDDF_dic = data['MSE_OTDDF_dic'].tolist()



Noise = data['Noise']
Window = data['Window']
parameters = data['parameters']

burn_in = data['burn_in']

#%%
AVG_SIM = X_OTPF.shape[0]
J = X_OTPF.shape[3]
SAMPLE_SIZE = X_OTPF.shape[3]
n = X_true.shape[2]

tau = time[1]-time[0] # timpe step 
[noise,sigmma,sigmma0,gamma,x0_amp] = Noise




X0 = X_OTPF[:,0,:,:]
X_SIR , MSE_SIR = SIR(X_true,Y_true,X0,ML63,h,time,tau,Noise)
X_EnKF , MSE_EnKF = EnKF(X_true,Y_true,X0,ML63,h,time,tau,Noise)


mse_OTDDF_avg = []
for window in Window:
    mse_OTDDF_avg.append(np.mean(MSE_OTDDF_dic[window.astype('str')][burn_in:]))


time = time/tau

#%%

plot_particle = 100#J
L=2
for k in range(1):
    plt.figure(figsize=(20,7.2)) 
    grid = plt.GridSpec(8, 2, wspace =0.12, hspace = 0.15)
    for l in range(1,L):
        g1 = plt.subplot(grid[0:2, l-1])
# =============================================================================
#         plt.subplot(4,L,l+1)
# =============================================================================
        plt.plot(time[burn_in:],X_EnKF[k,burn_in:,l,:plot_particle],'g',alpha = 0.1)
        plt.plot(time[burn_in:],X_true[k,burn_in:,l],'k--',label='True state',lw=2)
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
        plt.plot(time[burn_in:],X_OTPF[k,burn_in:,l,:plot_particle],'r',alpha = 0.1)
        plt.plot(time[burn_in:],X_true[k,burn_in:,l],'k--',lw=2)
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
        plt.plot(time[burn_in:],X_OTDDF_dic['50'][k,burn_in:,l,:plot_particle],'C0',alpha = 0.1)
        plt.plot(time[burn_in:],X_true[k,burn_in:,l],'k--',lw=2)
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
        plt.plot(time[burn_in:],X_SIR[k,burn_in:,l,:plot_particle],'b',alpha = 0.1)
        plt.plot(time[burn_in:],X_true[k,burn_in:,l],'k--',lw=2)
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
plt.semilogy(time[burn_in:],MSE_EnKF[burn_in:],'g--',lw=2,label="EnKF",alpha=0.7)
plt.semilogy(time[burn_in:],MSE_OTPF[burn_in:],'r-.',lw=2,label="OTPF" ,alpha=0.7)
# =============================================================================
# plt.plot(time,mse_OT_without,'c-.',label="$OT_{without EnKF}$" )
# =============================================================================
plt.semilogy(time[burn_in:],MSE_SIR[burn_in:],'b:',lw=2,label="SIR" ,alpha=0.7)

for window in Window_plot:
    if window==50:
        plt.semilogy(time[burn_in:], MSE_OTDDF_dic[window.astype('str')][burn_in:],lw=2,label='OT-DDF ($w$=' + window.astype('str')+')',color='C0')
    elif window==10:
        plt.semilogy(time[burn_in:], MSE_OTDDF_dic[window.astype('str')][burn_in:],lw=2,label='OT-DDF ($w$=' + window.astype('str')+')',alpha=0.7,color='C1')
    else:
        plt.semilogy(time[burn_in:], MSE_OTDDF_dic[window.astype('str')][burn_in:],lw=2,label='OT-DDF ($w$=' + window.astype('str')+')',alpha=0.7,color='C4')

plt.xlabel('time',fontsize=16)
plt.ylabel('MSE',fontsize=15)
plt.legend(fontsize=12,bbox_to_anchor=(0.95, 0.8))


g1 = plt.subplot(grid[5:, 1:])
plt.axhline(y= np.mean(MSE_EnKF[burn_in:]),color = 'g',linestyle='--',label="EnKF",lw=2)
plt.axhline(y= np.mean(MSE_OTPF[burn_in:]),color ='r',linestyle = '-.',label="OTPF" ,lw=2)
plt.axhline(y = np.mean(MSE_SIR[burn_in:]),color='b',linestyle =':',label="SIR" ,lw=2)
plt.plot(Window*tau,mse_OTDDF_avg,lw=2,label='OT-DDF',color='C0')
plt.xlabel('w $*$ dt',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.yscale('log')
plt.legend(fontsize=14,bbox_to_anchor=(0.95, 0.8))
plt.show()
