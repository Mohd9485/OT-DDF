import numpy as np
import matplotlib.pyplot as plt
import torch, math, time
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR
import sys
from OT_DDF import OT_DDF
from OTPF import OTPF

#%matplotlib auto


plt.close('all')


# Choose h(x) here, the observation rule
def h(x):
    return x[0].reshape(1,-1)


def A(x,t=0):
    return F @ (x)




def Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau):
    Odeint = True*0
    sai = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),N)
    eta = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),N)
    
    x = np.zeros((N,L))
    y = np.zeros((N,dy))
    x0 = x0_amp*np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),1)
    x[0,] = x0

    
    for i in range(N-1):
        x[i+1,:] = A(x[i,:])  + sai[i,:] 
        y[i+1,] = h(x[i+1,]) + eta[i+1,]
        
    return x,y

#%%    
L = 2 # number of states
tau = 1e-1 # timpe step 
T = 40 # final time in seconds
N = int(T/tau) # number of time steps T = 20 s
dy = 1 # number of states observed
burn_in = 100
Window = np.array([1,2,5,10,20,50])


# dynmaical system
H = np.array([[1,0]]) 
alpha = 0.9
a = alpha
b= np.sqrt(1-alpha**2)
c = alpha
F = np.array([[a, -b],[b,c]]) 


noise = np.sqrt(1e-1) # noise level std
sigmma = noise # Noise in the hidden state
sigmma0 = 1#5*noise # Noise in the initial state distribution
gamma = noise # Noise in the observation
x0_amp = 1#/noise # Amplifiying the initial state 
Noise = [noise,sigmma,sigmma0,gamma,x0_amp]

J = int(1e3) # Number of ensembles EnKF
AVG_SIM = 2 # Number of Simulations to average over

# OT networks parameters
parameters = {}
parameters['normalization'] = 'None' #'MinMax' #'Mean' # Choose 'None' for nothing , 'Mean' for standard gaussian, 'MinMax' for d[0,1]
parameters['INPUT_DIM'] = [L,dy]
parameters['NUM_NEURON'] =  int(32*2) #64
parameters['SAMPLE_SIZE'] = int(J) 
parameters['BATCH_SIZE'] = int(64*1) #128
parameters['LearningRate'] = 1e-2/10
parameters['ITERATION'] = int(1024/1) 
parameters['Final_Number_ITERATION'] = int(64/1) #int(64) #ITERATION 
parameters['Time_step'] = N


t = np.arange(0.0, tau*N, tau)
SAVE_True_X = np.zeros((AVG_SIM,N,L))
SAVE_True_Y = np.zeros((AVG_SIM,N,dy))
X0 = np.zeros((AVG_SIM,L,J))
for k in range(AVG_SIM):    
    x,y = Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau)
    SAVE_True_X[k,] = x
    SAVE_True_Y[k,] = y
    X0[k,] = x0_amp*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))





# =============================================================================
# SAVE_X_KF  = KF(SAVE_True_Y,X0,F,H,t,tau,Noise)
# SAVE_X_SIR = SIR(SAVE_True_X,SAVE_True_Y,X0,A,h,t,tau,Noise)
# SAVE_X_EnKF = EnKF(SAVE_True_X,SAVE_True_Y,X0,A,h,t,tau,Noise)
# =============================================================================
SAVE_X_OTPF  = OTPF(SAVE_True_X,SAVE_True_Y,X0,parameters,A,h,t,tau,Noise)



X_OT_DDF_dic = {}
for window in Window:
    print('Window : ' + window.astype('str'))
    parameters['INPUT_DIM'] = [L,dy*window]
    parameters['ITERATION'] = int(1000*12) 
    SAVE_X_OT_DDF  = OT_DDF(SAVE_True_X,SAVE_True_Y,X0,parameters,A,h,t,tau,Noise,window,burn_in)

    X_OT_DDF_dic[window.astype('str')] = SAVE_X_OT_DDF
#%%
    
np.savez('DATA_file.npz',\
    time = t, Y_true = SAVE_True_Y,X_true = SAVE_True_X, Noise=Noise,Window = Window,\
     F= F, H = H, X_OTPF = SAVE_X_OTPF,\
         X_OT_DDF_dic =X_OT_DDF_dic,parameters=parameters, burnin=burn_in)
