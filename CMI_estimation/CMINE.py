import time
import numpy as np
import pickle
from numpy.linalg import det

import CMINE_lib as CMINE



def estimate_CMI(config):
    #-----------------------------------------------------------------#    
    #--------------- Create the dataset ------------------------------#
    #-----------------------------------------------------------------#    
    dim = config.d # 1 # number of samples
    n = config.n # int(8e4)  # dimension of X,Y, and Z
    
    sigma_x = config.sigma_x # 10
    sigma_1 = config.sigma_y # 1
    sigma_2 = config.sigma_z # 5
    arrng = config.arrng # [[0],[1],[2]]       # I(arg0 ; arg1 | arg2)
    
    params = (sigma_x,sigma_1,sigma_2)
    
    if config.scenario == 0: #Estimate I(X;Y|Z)
        True_CMI = -dim*0.5*np.log(sigma_1**2 * (sigma_x**2+sigma_1**2 + sigma_2**2)/((sigma_x**2 + sigma_1**2)*(sigma_1**2 + sigma_2**2)))
    elif config.scenario == 1: #Estimate I(X;Z|Y)    
        True_CMI = 0
    
    K = config.k
    b_size = config.batch_size
    
    #----------------------------------------------------------------------#
    #------------------------Train the network-----------------------------#
    #----------------------------------------------------------------------#
    
    # Set up neural network paramters
    LR = config.lr
    EPOCH = config.e
    SEED = config.seed
    input_size = 3*dim
    hidden_size = 64
    num_classes = 2
    tau = config.tau  # 1e-4  clip the NN output [tau,1-tau]
    
    NN_params = (input_size,hidden_size,num_classes,tau) # 3, 64, 2, 1e-4
    EVAL = False
    
    #Monte Carlo param
    T = config.t # 20 number of trials
    S = config.s # 10 number of repretitions
    
    CMI_LDR = []
    CMI_DV = []
    CMI_NWJ = []
    
    for s in range(S):
        CMI_LDR_t = []
        CMI_DV_t = []
        CMI_NWJ_t = []
            
        #Create dataset
        dataset = CMINE.create_dataset(GenModel='Gaussian_nonZero', Params=params, Dim=dim, N=n)
        #  [array([[-15.0978431 , -14.95947252,  16.02306022,  -0.57511926,], dataset[0].shape: (800,5), len(dataset):3

        for t in range(T): 
            start_time = time.time()
            
            batch_train, target_train, joint_test, prod_test = CMINE.batch_construction(data=dataset, arrange=arrng, set_size=b_size, K_neighbor=K)     # datatset,[[0],[1],[2]], n//2, 2
            # batch_train.shape  torch.Size([800, 15]) torch.Size([800, 2]) jt torch.Size([400, 15]) pt torch.Size([400, 15])
            print('Duration of data preparation: ',time.time()-start_time,' seconds')
            
            CMI_LDR_Eval=[]
            CMI_DV_Eval=[]
            CMI_NWJ_Eval=[]

            start_time = time.time()
            #Train
            if EVAL:
                model, loss_e, CMI_LDR_e, CMI_DV_e, CMI_NWJ_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Seed=SEED, Eval=True, JointEval=joint_test, ProdEval=prod_test)
                CMI_LDR_Eval.append(CMI_LDR_e)
                CMI_DV_Eval.append(CMI_DV_e)    
                CMI_NWJ_Eval.append(CMI_NWJ_e)
            else:
                model, loss_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Seed=SEED)
            
            #Compute I(X;Y|Z)
            CMI_est = CMINE.estimate_CMI(model, joint_test, prod_test) # (8.025778863719374, 0.5846232668096443, -1695.6932725486215)

        
            print('Duration: ', time.time()-start_time, ' seconds')       
            
            print('LDR=',CMI_est[0])   
            print('DV=',CMI_est[1])   
            print('NWJ=',CMI_est[2]) 
            print('True=',True_CMI)
            
            CMI_LDR_t.append(CMI_est[0])
            CMI_DV_t.append(CMI_est[1])
            CMI_NWJ_t.append(CMI_est[2])
            
        CMI_LDR.append(np.mean(CMI_LDR_t))
        CMI_DV.append(np.mean(CMI_DV_t))
        CMI_NWJ.append(np.mean(CMI_NWJ_t))    
        
    file = open(config.directory+'/result_'+str(config.seed), 'wb')
    pickle.dump((True_CMI,CMI_LDR,CMI_DV,CMI_NWJ,CMI_LDR_Eval,CMI_DV_Eval,CMI_NWJ_Eval,n,dim,K,LR,EPOCH,loss_e), file)
    
    file.close()    
    

