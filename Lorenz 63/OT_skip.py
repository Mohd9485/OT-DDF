"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from torch.distributions.multivariate_normal import MultivariateNormal
# =============================================================================
# from scipy.integrate import  RK45
# =============================================================================

def OT_skip(X,Y,X0_const,parameters,A,h,t,tau,Noise,Odeint,window,skip):
    AVG_SIM = X.shape[0]
    N = X.shape[1]
    L = X.shape[2]
    dy = Y.shape[2]
    J = X0_const.shape[2]
    noise = Noise[0]
    sigmma = Noise[1]# Noise in the hidden state
    sigmma0 = Noise[2] # Noise in the initial state distribution
    gamma = Noise[3] # Noise in the observation
    x0_amp = Noise[4]
    T = N*tau
    
    # OT networks parameters
    normalization = parameters['normalization']
    NUM_NEURON = parameters['NUM_NEURON']
    INPUT_DIM = parameters['INPUT_DIM']
    SAMPLE_SIZE = parameters['SAMPLE_SIZE']
    BATCH_SIZE =  parameters['BATCH_SIZE']
    LearningRate = parameters['LearningRate']
    ITERATION = parameters['ITERATION']
    Final_Number_ITERATION = parameters['Final_Number_ITERATION']
# =============================================================================
#     Time_step = parameters['Time_step']
# =============================================================================
    
    #device = torch.device('mps' if torch.has_mps else 'cpu') # M1 Chip
    device = torch.device('cpu')
    # NN , initialization and training    
    class NeuralNet(nn.Module):
        
        def __init__(self, input_dim, hidden_dim):
            super(NeuralNet, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.activationSigmoid = nn.Sigmoid()
            self.activationReLu = nn.ReLU()
            self.layer_input = nn.Linear(self.input_dim[0]+self.input_dim[1], self.hidden_dim, bias=False)
            self.layer_1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer_out = nn.Linear(self.hidden_dim, 1, bias=True)
            
            self.layer_2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            
        # Input is of size
        def forward(self, x,y):
            h = self.layer_input(torch.concat((x,y),dim=1))
            
            h_temp = self.layer_1(self.activationReLu(h)) 
            
# =============================================================================
#             h_temp = self.layer_2(self.activationReLu(h_temp) + h) 
# =============================================================================
            
            z = self.layer_out(self.activationReLu(h_temp) + h)  #+ 0.01*(x*x).sum(dim=1)
            return z
        
        
# =============================================================================
#     class NeuralNet(nn.Module):
#             
#             def __init__(self, input_dim, hidden_dim):
#                 super(NeuralNet, self).__init__()
#                 self.input_dim = input_dim
#                 self.hidden_dim = hidden_dim
#                 self.activationSigmoid = nn.Sigmoid()
#                 self.activationReLu = nn.ReLU()
#                 self.activationNonLinear = nn.Sigmoid()
#                 self.layer_input = nn.Linear(self.input_dim[0]+self.input_dim[1], self.hidden_dim, bias=False)
#                 self.layer11 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
#                 self.layer12 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
#                 self.layer21 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
#                 self.layer22 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
#                 self.layerout = nn.Linear(self.hidden_dim, 1, bias=False)
#                 
#             # Input is of size
#             def forward(self, x, y):
#                 X = self.layer_input(torch.concat((x,y),dim=1))
#                 
#                 xy = self.layer11(X)
#                 xy = self.activationReLu(xy)
#                 xy = self.layer12 (xy)
#                 
#                 xy = self.activationReLu(xy)+X
# # =============================================================================
# #                 xy = xy+X
# #                 xy = self.activationReLu(xy)
# # =============================================================================
#                 
#                 xy = self.layer21(xy)
#                 xy = self.activationReLu(xy)
#                 xy = self.layer22 (xy)
#                 
#                 xy = self.layerout(self.activationReLu(xy)+X)
#                 return xy
# # =============================================================================
# #                 return 50*nn.Tanh()(xy)
# # =============================================================================
# =============================================================================

    class T_NeuralNet(nn.Module):
            
            def __init__(self, input_dim, hidden_dim):
                super(T_NeuralNet, self).__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.activationSigmoid = nn.Sigmoid()
                self.activationReLu = nn.ReLU()
                self.activationNonLinear = nn.Sigmoid()
                self.layer_input = nn.Linear(self.input_dim[0]+self.input_dim[1], self.hidden_dim, bias=False)
                self.layer11 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                self.layer12 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                self.layer21 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                self.layer22 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                self.layerout = nn.Linear(self.hidden_dim, input_dim[0], bias=False)
            
                
                
                self.dist = MultivariateNormal(torch.zeros(self.input_dim[1]),gamma*gamma * torch.eye(self.input_dim[1]))
            # Input is of size
            def forward(self, x, y):
                
                X = self.layer_input(torch.concat((x,y),dim=1))
                
                xy = self.layer11(X)
                xy = self.activationReLu(xy)
                xy = self.layer12 (xy)
                
                xy = self.activationReLu(xy)+X
# =============================================================================
#                 xy = xy+X
#                 xy = self.activationReLu(xy)
# =============================================================================
                
                xy = self.layer21(xy)
                xy = self.activationReLu(xy)
                xy = self.layer22 (xy)

                
                xy = self.layerout(self.activationReLu(xy)+X)
# =============================================================================
#                 xy = self.layerout(self.activationReLu(xy))+x
# =============================================================================
                return xy
    
        
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            #torch.nn.init.xavier_normal_(m.weight)
            #torch.nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu')
            #torch.nn.init.kaiming_uniform_(m.weight,mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.fill_(0.1)
                
    def train(f,T,X_Ref,X_Train,Y_Train, iterations,learning_rate,ts,Ts,batch_size,k,K):
        f.train()
        T.train()
        
        optimizer_T = torch.optim.Adam(T.parameters(), lr=learning_rate/2) 
        optimizer_f = torch.optim.Adam(f.parameters(), lr=learning_rate/10)
        scheduler_f = ExponentialLR(optimizer_f, gamma=0.999) #set LR = 1e-1
        scheduler_T = ExponentialLR(optimizer_T, gamma=0.999) #set LR = 1e-1
       
        inner_iterations = 10

        Y_Train_shuffled = Y_Train[torch.randperm(Y_Train.shape[0])].view(Y_Train.shape)
        for i in range(iterations):
            idx = torch.randperm(X1.shape[0])[:batch_size]
            #X_train = torch.tensor(X_Train[idx])
            #Y_train = torch.tensor(Y_Train[idx])
            X_train = X_Train[idx].clone().detach()
            Y_train = Y_Train[idx].clone().detach()
            X_ref = X_Ref[idx].clone().detach()
            
            #X_train.requires_grad = True
            Y_shuffled = Y_train[torch.randperm(Y_train.shape[0])].view(Y_train.shape)
            #Y_shuffled.requires_grad = True
            for j in range(inner_iterations):
                map_T = T.forward(X_ref,Y_shuffled)
                f_of_map_T= f.forward(map_T,Y_shuffled) 
                #grad_f_of_map_T = torch.autograd.grad(f_of_map_T.sum(),map_T,create_graph=True)[0]
                loss_T = - f_of_map_T.mean() + 0.5*((X_train-map_T)*(X_train-map_T)).sum(axis=1).mean()
# =============================================================================
#                 loss_T = f_of_map_T.mean() - (X_train*map_T).sum(axis=1).mean()
# =============================================================================
                optimizer_T.zero_grad()
                loss_T.backward()
                optimizer_T.step()
                
            f_of_xy = f.forward(X_train,Y_train) 
            map_T = T.forward(X_ref,Y_shuffled)
            f_of_map_T= f.forward(map_T,Y_shuffled) 
            loss_f = -f_of_xy.mean() + f_of_map_T.mean()
# =============================================================================
#             loss_f =f_of_xy.mean() - f_of_map_T.mean()
# =============================================================================
            optimizer_f.zero_grad()
            loss_f.backward()
            optimizer_f.step()
            if  (i+1)==iterations or i%1000==0:
                with torch.no_grad():
                    f_of_xy = f.forward(X_Train,Y_Train) 
                    map_T = T.forward(X_Ref,Y_Train_shuffled)
                    f_of_map_T = f.forward(map_T,Y_Train_shuffled) 
                    loss_f = f_of_xy.mean() - f_of_map_T.mean()
                    loss = f_of_xy.mean() - f_of_map_T.mean() + ((X_Train-map_T)*(X_Train-map_T)).sum(axis=1).mean()
    # =============================================================================
    #                 f.layer.weight = torch.nn.parameter.Parameter(nn.functional.relu(f.layer.weight))
    # =============================================================================
                    
                    #print(g.W.data)
                    print("Simu#%d/%d ,Time Step:%d/%d, Iteration: %d/%d, loss = %.4f" %(k+1,K,ts,Ts-1,i+1,iterations,loss.item()))
                      
                
            scheduler_f.step()
            scheduler_T.step()


            

    def Normalization(X,Type = 'None'):
        ''' Normalize Date with type 'MinMax' out data between [0,1] or 'Mean' for mean 0 and std 1 '''
        if Type == 'None':
            return 0,0,X
        elif Type == 'Mean':
            Mean_X_training_data = torch.mean(X)
            Std_X_training_data = torch.std(X)
            return Mean_X_training_data , Std_X_training_data , (X - Mean_X_training_data)/Std_X_training_data
        elif Type == 'MinMax':
            Min = torch.min(X) 
            Max = torch.max(X)
            return Min , Max , (X-Min)/(Max-Min)

            
    def Transfer(M,S,X,Type='None'):
        '''Trasfer test Data to normalized data using knowledge of training data
        M = Mean/Min , S = Std/Max , X is data , Type = Mean/Min-Max Normalization '''
        if Type == 'None':
            return X
        elif Type == 'Mean':
            return (X - M)/S
        elif Type == 'MinMax':
            return (X - M)/(S - M)
        
    def deTransfer(M,S,X , Type = 'None'):
        ''' Detransfer the normalized data to the origin set
         M = Mean/Min , S = Std/Max , X is data , Type = Mean/Min-Max Normalization'''  
        if Type == 'None':
            return X
        elif Type == 'Mean':
            return X*S + M
        elif Type == 'MinMax':
            return X*(S - M) + M
    #
    start_time = time.time()
    SAVE_all_X_OT = np.zeros((AVG_SIM,N,SAMPLE_SIZE,L))
    mse_OT = np.zeros((N,AVG_SIM))
    
    for k in range(AVG_SIM):
        
        x = X[k,]
        y = Y[k,]
        
        
        ITERS = ITERATION
        LR = LearningRate
        
        convex_f = NeuralNet(INPUT_DIM, NUM_NEURON)
        MAP_T = T_NeuralNet(INPUT_DIM, NUM_NEURON)
        
        convex_f.apply(init_weights)
        MAP_T.apply(init_weights)     

        #X0 = x0_amp*np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),SAMPLE_SIZE)
        X0 = X0_const[k,].T
        X1 = np.zeros((SAMPLE_SIZE,L))
        Y1 = np.zeros((SAMPLE_SIZE,dy))
        x_OT = np.zeros((N,L))
        x_OT[0,:] = X0.mean(axis=0)
        SAVE_all_X_OT[k,0,:,:] = X0
        #plt.figure()
        for i in range(N-1):
           
            sai_train = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),SAMPLE_SIZE)
            if Odeint:
                sai_train = sai_train.T
                X1 = ((odeint(A, (X0.T).reshape(-1), t[i:i+2])[1,]).reshape(L,J) + sai_train).T
            else: 
                X1 = X0 + (((A(X0.T,t[i]).T)*tau)  + sai_train)
            
            if i == skip - window:
                X_ref = X1
                X_ref = torch.from_numpy(X_ref)
                X_ref = X_ref.to(torch.float32).to(device)
            

            if i == skip: 
                w_multiple = int(1e2)
                T_sim = int(window*(w_multiple-1))+skip
                for j in range(T_sim):
                    #print(j)
                    train_sample = 100
                    if j==0:
                        X0_train = np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),train_sample)
                        X_all = X0_train

                    sai_train = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),train_sample)
                    if Odeint:
                        sai_train = sai_train.T
                        X1_train = ((odeint(A, (X0_train.T).reshape(-1), t[j:j+2])[1,]).reshape(L,train_sample) + sai_train).T
                    else: 
                        X1_train = X0_train + (A(X0_train.T).T)*tau # + sai_train

                    X0_train = X1_train
                    
                    eta_train = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),train_sample)
                    Y1_train = np.array(h(X1_train.T).T + eta_train)
                    if j==0:
                        Y_all = Y1_train
                    else:    
                        Y_all = np.concatenate((Y_all,Y1_train),axis=1)
                        X_all = np.concatenate((X_all,X1_train),axis=0)
                
                

                X_all = X_all[(skip-window)*train_sample:].reshape(w_multiple,window*train_sample,L)
                X_ref_train = X_all[:,:train_sample,:].reshape(w_multiple*train_sample,L)
                X_ref_train = torch.from_numpy(X_ref_train)
                X_ref_train = X_ref_train.to(torch.float32).to(device)
                
                
                X1_train = torch.from_numpy(X_all[:,-train_sample:,:].reshape(w_multiple*train_sample,L))
                X1_train = X1_train.to(torch.float32)
                X1_train = X1_train.to(device)
                
                Y_all =Y_all[:,(skip-window)*dy:].reshape(train_sample,w_multiple,window*dy).reshape(w_multiple*train_sample, window*dy,order='F')
                Y_all = torch.from_numpy(Y_all)
                Y_all = Y_all.to(torch.float32)
                
                

                ITERS = ITERATION
                LR = LearningRate
                convex_f = NeuralNet(INPUT_DIM, NUM_NEURON)
                MAP_T = T_NeuralNet(INPUT_DIM, NUM_NEURON)
                convex_f.apply(init_weights)
                MAP_T.apply(init_weights)   

                train(convex_f,MAP_T,X_ref_train,X1_train,Y_all[:,-window*dy:],ITERS,LR,i+1,N,BATCH_SIZE,k,AVG_SIM)
            

            if i==0:
                Y_true_all = y[i+1,:].reshape(1,-1)
            else:    
                Y_true_all = np.concatenate((Y_true_all,y[i+1,:].reshape(1,-1)),axis=1)
        
            Y_true_all = torch.from_numpy(Y_true_all)
            Y_true_all = Y_true_all.to(torch.float32)
            
            # Update X^(j) for the next time step
            X1_test = torch.from_numpy(X1).to(torch.float32).to(device)
            Y_true_all = Y_true_all.to(device)
            

            if i >= skip: 
                Y_true_all_vec = Y_true_all[:,-window*dy:]*torch.ones((X_ref.shape[0],dy*window))
                map_T = MAP_T.forward(X_ref, Y_true_all_vec)

            else:
                map_T = X1_test
                
            if device.type == 'mps':
                X0 = map_T.cpu().detach().numpy()
            else:
                X0 = map_T.detach().numpy()
            
            x_OT[i+1,:] = (torch.mean(map_T,dim=0)).detach().numpy()
            SAVE_all_X_OT[k,i+1,:,:] = map_T.detach().numpy()
            
            mse_OT[:,k] =  ((x - x_OT)*(x - x_OT)).mean(axis=1)
            
    SAVE_all_X_OT = SAVE_all_X_OT.transpose((0,1,3,2))       
    MSE_OT =  mse_OT.mean(axis=1)
    print("--- OT time : %s seconds ---" % (time.time() - start_time))
    return SAVE_all_X_OT,MSE_OT 