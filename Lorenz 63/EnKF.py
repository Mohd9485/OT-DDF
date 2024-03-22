
"""

@author: Mohammad Al-Jarrah
Ph.D. student, University of Washington - Seattle
email: mohd9485@uw.edu 

"""

import numpy as np
import time
def EnKF(X,Y,X0,A,h,t,tau,Noise):
    AVG_SIM = X.shape[0]
    N = X.shape[1]
    L = X.shape[2]
    dy = Y.shape[2]
    J = X0.shape[2]
    noise = Noise[0]
    sigmma = Noise[1]# Noise in the hidden state
    sigmma0 = Noise[2] # Noise in the initial state distribution
    gamma = Noise[3] # Noise in the observation
    x0_amp = 1/noise # Amplifiying the initial state 
    start_time = time.time()
    SAVE_X_EnKF =  np.zeros((AVG_SIM,N,L,J))
    mse_EnKF =  np.zeros((N,AVG_SIM))
    for k in range(AVG_SIM):
 
        x = X[k,]
        y = Y[k,]
        
        x_EnKF  = np.zeros((N,L,J))
        #x_EnKF[0,] = x0_amp*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))
        x_EnKF[0,] = X0[k,] 
        x_hatEnKF = np.transpose(np.random.multivariate_normal(np.zeros(L),noise*noise * np.eye(L),J))
        y_hatEnKF = np.zeros((dy,J)) 
        
        m_hatEnKF = np.zeros(L)
        o_hatEnKF = np.zeros((N,dy))
        C_hat_vh = np.zeros((L,dy))
        C_hat_hh = np.zeros((dy,dy))
        
        K_EnKF = np.zeros((L,dy))
        
        SAVE_X_EnKF[k,0,:,:] = x_EnKF[0,]
        # EnKF & 3DVAR
        for i in range(N-1):

            sai_EnKF = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),J).transpose()
                
            x_hatEnKF = x_EnKF[i,] + tau*A(x_EnKF[i,],t[i]) + sai_EnKF
                
            eta_EnKF = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),J)    
            #y_hatEnKF = (x_hatEnKF**3).transpose() + eta_EnKF
            y_hatEnKF = (h(x_hatEnKF).reshape(dy,-1)).transpose() + eta_EnKF
            
            m_hatEnKF = x_hatEnKF.mean(axis=1)
            #o_hatEnKF = (x_hatEnKF**3).mean(axis=1)
            o_hatEnKF = (h(x_hatEnKF).reshape(dy,-1)).mean(axis=1)
            
            a = (x_hatEnKF.transpose() - m_hatEnKF)
            #b = ((x_hatEnKF**3).transpose() - o_hatEnKF)
            b = (h(x_hatEnKF).reshape(dy,-1).transpose() - o_hatEnKF)
            
 
            C_hat_vh = np.matmul(a.transpose(),b)/J
            C_hat_hh = np.matmul(b.transpose(),b)/J
            K_EnKF = np.matmul(C_hat_vh,np.linalg.inv(C_hat_hh + np.eye(dy)/10))
                
            x_EnKF[i+1,:,:] = x_hatEnKF + np.matmul( K_EnKF, (y[i+1,:] - y_hatEnKF).transpose())
            
            SAVE_X_EnKF[k,i+1,:,:] = x_EnKF[i+1,]
        mse_EnKF[:,k] = ((x_EnKF.mean(axis=2)-x)*(x_EnKF.mean(axis=2)-x)).mean(axis=1)
    MSE_EnKF = mse_EnKF.mean(axis=1)

    print("--- EnKF time : %s seconds ---" % (time.time() - start_time))
    return SAVE_X_EnKF , MSE_EnKF