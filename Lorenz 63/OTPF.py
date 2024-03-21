"""
@author: Mohammad Al-Jarrah
"""
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR


def OTPF(X,Y,X0_const,parameters,A,h,t,tau,Noise):
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

            
        # Input is of size
        def forward(self, x,y):
            h = self.layer_input(torch.concat((x,y),dim=1))
            h_temp = self.layer_1(self.activationReLu(h)) 
            z = self.layer_out(self.activationReLu(h_temp) + h)  #+ 0.01*(x*x).sum(dim=1)
            return z
        
        

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
# =============================================================================
#                 return 50*nn.Tanh()(xy)
# =============================================================================
    
        
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            #torch.nn.init.xavier_normal_(m.weight)
            #torch.nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu')
            #torch.nn.init.kaiming_uniform_(m.weight,mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.fill_(0.1)

    def train(f,T,X_Train,Y_Train, iterations,learning_rate,ts,Ts,batch_size,k,K):
        f.train()
        T.train()
        optimizer_T = torch.optim.Adam(T.parameters(), lr=learning_rate/2) 
        optimizer_f = torch.optim.Adam(f.parameters(), lr=learning_rate/10)
        scheduler_f = ExponentialLR(optimizer_f, gamma=0.999) #set LR = 1e-1
        scheduler_T = ExponentialLR(optimizer_T, gamma=0.999) #set LR = 1e-1
# =============================================================================
       
        inner_iterations = 10
        Y_Train_shuffled = Y_Train[torch.randperm(Y_Train.shape[0])].view(Y_Train.shape)
        for i in range(iterations):
            idx = torch.randperm(X1.shape[0])[:batch_size]
            #X_train = torch.tensor(X_Train[idx])
            #Y_train = torch.tensor(Y_Train[idx])
            X_train = X_Train[idx].clone().detach()
            Y_train = Y_Train[idx].clone().detach()
            
            #X_train.requires_grad = True
            Y_shuffled = Y_train[torch.randperm(Y_train.shape[0])].view(Y_train.shape)
            #Y_shuffled.requires_grad = True
            for j in range(inner_iterations):
                map_T = T.forward(X_train,Y_shuffled)
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
            map_T = T.forward(X_train,Y_shuffled)
            f_of_map_T= f.forward(map_T,Y_shuffled) 
            loss_f = -f_of_xy.mean() + f_of_map_T.mean()
# =============================================================================
#             loss_f =f_of_xy.mean() - f_of_map_T.mean()
# =============================================================================
            optimizer_f.zero_grad()
            loss_f.backward()
            optimizer_f.step()
            if  (i+1)==iterations or i%100==0:
                            with torch.no_grad():
                                f_of_xy = f.forward(X_Train,Y_Train) 
                                map_T = T.forward(X_Train,Y_Train_shuffled)
                                f_of_map_T = f.forward(map_T,Y_Train_shuffled) 
                                loss_f = f_of_xy.mean() - f_of_map_T.mean()
                                loss = f_of_xy.mean() - f_of_map_T.mean() + ((X_Train-map_T)*(X_Train-map_T)).sum(axis=1).mean()
                # =============================================================================
                #                 f.layer.weight = torch.nn.parameter.Parameter(nn.functional.relu(f.layer.weight))
                # =============================================================================
                                
                                #print(g.W.data)
                                print("Simu#%d/%d ,Time Step:%d/%d, Iteration: %d/%d, loss = %.4f" %(k+1,K,ts,Ts-1,i+1,iterations,loss.item()))
            # if  (i+1)==iterations:
            #    print("Simu#%d/%d ,Time Step:%d/%d, Iteration: %d/%d" %(k+1,K,ts,Ts-1,i+1,iterations))       
            
             
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
            X1 = X0 + (((A(X0.T,t[i]).T)*tau)  + sai_train)
            
            eta_train = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),SAMPLE_SIZE)

            Y1 = np.array(h(X1.T).reshape(dy,-1).T + eta_train)

            X1_train = torch.from_numpy(X1)
            X1_train = X1_train.to(torch.float32)
            Y1_train = torch.from_numpy(Y1)
            Y1_train = Y1_train.to(torch.float32)
            X1_train = X1_train.to(device)
            Y1_train = Y1_train.to(device)

            
            train(convex_f,MAP_T,X1_train,Y1_train,ITERS,LR,i+1,N,BATCH_SIZE,k,AVG_SIM)

                
            if ITERS > Final_Number_ITERATION and i%1 == 0 :
                ITERS = int(ITERS/2)
                

                
            Y1_true = y[i+1,:]
            Y1_true = torch.from_numpy(Y1_true)
            Y1_true = Y1_true.to(torch.float32)
            
            # Update X^(j) for the next time step
            X1_test = torch.from_numpy(X1).to(torch.float32).to(device)
            Y1_true = Y1_true.to(device)
            
 
            
            map_T = MAP_T.forward(X1_test, Y1_true*torch.ones((X1_test.shape[0],dy)))
            
            
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