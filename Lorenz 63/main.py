"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import matplotlib.pyplot as plt
import torch, math, time
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR
import sys
from EnKF import EnKF
from SIR import SIR
# =============================================================================
# from OT_new import OT
# =============================================================================
from OT import OT
from OT_skip import OT_skip
# =============================================================================
# from OT_skip_window import OT_skip
# =============================================================================
from scipy.integrate import odeint
#%matplotlib auto

plt.close('all')

# np.random.seed(101)
# torch.manual_seed(101)

# Choose h(x) here, the observation rule
def h(x):
    return x[0,].reshape(1,-1)


def L63(x, t=0):
    """Lorenz 96 model"""
    # Setting up vector
    #L = 3
    d = np.zeros_like(x)
    sigma = 10
    r = 28
    b = 8/3

    d[0] = sigma*(x[1]-x[0])
    d[1] = x[0]*(r-x[2])-x[1]
    d[2] = x[0]*x[1]-b*x[2]
    return d

def ML63(x, t=0 , particles = 100):
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

def Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau):
    sai = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),N)
    eta = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),N)
    
    x = np.zeros((N,L))
    y = np.zeros((N,dy))
    x0 = 0+x0_amp*np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),1)
    x[0,] = x0

    
    for i in range(N-1):
        #print(i)
        if Odeint:
            x[i+1,] = odeint(L63, x[i,], t[i:i+2])[1,] #+ sai[i,]   
        else:
            x[i+1,:] = x[i,:] + L63(x[i,:],t[i])*tau  #+ sai[i,:] 
        y[i+1,] = h(x[i+1,]) + eta[i+1,]
    
    return x,y

#%%    
L = 3 # number of states
tau = 1e-2 # timpe step 
T = 40 # final time in seconds
N = int(T/tau) # number of time steps T = 20 s
dy = 1 # number of states observed

Window = np.array([1,10,50,100,200])


noise = np.sqrt(1e1) # noise level std
sigmma = noise/10 # Noise in the hidden state
sigmma0 = noise/1 # Noise in the initial state distribution
gamma = noise/1 # Noise in the observation
x0_amp = 1 # Amplifiying the initial state 
Noise = [noise,sigmma,sigmma0,gamma,x0_amp]
Odeint = True*0

skip = 1000

J = 1000 # Number of ensembles EnKF
AVG_SIM = 1 # Number of Simulations to average over

# OT networks parameters
parameters = {}
parameters['normalization'] = 'None' #'MinMax' #'Mean' # Choose 'None' for nothing , 'Mean' for standard gaussian, 'MinMax' for d[0,1]
parameters['INPUT_DIM'] = [L,dy]
parameters['NUM_NEURON'] =  int(32*1)
parameters['SAMPLE_SIZE'] = int(J) 
parameters['BATCH_SIZE'] = int(64)
parameters['LearningRate'] = 1e-2
parameters['ITERATION'] = int(1024/1) 
parameters['Final_Number_ITERATION'] = int(64/4) #int(64) #ITERATION 
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



SAVE_X_EnKF , MSE_EnKF = EnKF(SAVE_True_X,SAVE_True_Y,X0,ML63,h,t,tau,Noise,Odeint)
SAVE_X_SIR , MSE_SIR = SIR(SAVE_True_X,SAVE_True_Y,X0,ML63,h,t,tau,Noise,Odeint)
SAVE_X_OT , MSE_OT = OT(SAVE_True_X,SAVE_True_Y,X0,parameters,L63,h,t,tau,Noise,Odeint)

#%%
X_OT_skip_dic = {}
MSE_OT_skip_dic = {}
for window in Window:
    print('Window : ' + window.astype('str'))
    parameters['INPUT_DIM'] = [L,dy*window]
    parameters['ITERATION'] = int(1024*15) 
    parameters['LearningRate'] = 1e-2

    SAVE_X_OT_skip, MSE_OT_skip  = OT_skip(SAVE_True_X,SAVE_True_Y,X0,parameters,L63,h,t,tau,Noise,Odeint,window,skip)
    
    X_OT_skip_dic[window.astype('str')] = SAVE_X_OT_skip
    MSE_OT_skip_dic[window.astype('str')] = MSE_OT_skip
    
    np.savez('DATA_file.npz',\
        time = t, Y_true = SAVE_True_Y,X_true = SAVE_True_X,Noise=Noise,\
         X_OT = SAVE_X_OT , MSE_OT=MSE_OT, Window = Window, 
         X_OT_skip_dic =X_OT_skip_dic,MSE_OT_skip_dic=MSE_OT_skip_dic,
            parameters=parameters, skip=skip,Odeint=Odeint)

#%%
np.savez('DATA_file.npz',\
    time = t, Y_true = SAVE_True_Y,X_true = SAVE_True_X,Noise=Noise,\
     X_OT = SAVE_X_OT , MSE_OT=MSE_OT, Window = Window, 
     X_OT_skip_dic =X_OT_skip_dic,MSE_OT_skip_dic=MSE_OT_skip_dic,
        parameters=parameters, skip=skip,Odeint=Odeint)
