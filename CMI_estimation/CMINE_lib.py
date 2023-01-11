import numpy as np
from sklearn.neighbors import NearestNeighbors

# MLP
import torch
from torch import autograd, nn, optim
import torch.nn.functional as F


def sample_batch(data, arrange=[[0],[1],[2]], batch_size=100, sample_mode='joint', K_neighbor=10, radius=1000):
    #data is represented as a tuple (x,y,u,w)
    #arrange is a triple that each determines what random variables are placed in what positions
    # I(X;Y|Z) where Z=(U,W)
    #(X,Y,Z)=data
    
    #X=data[arrange[0][0]]
    #Y=data[arrange[1][0]]
    X = np.concatenate([data[i] for i in arrange[0]],axis=1) #  (400, 5)
    Y = np.concatenate([data[i] for i in arrange[1]],axis=1)
    Z = np.concatenate([data[i] for i in arrange[2]],axis=1)

    N = X.shape[0]
    if sample_mode == 'joint':
        # sample according to p(x,y,z)
        index = np.random.choice(range(N), size=batch_size, replace=False)
        batch = np.concatenate((X[index], Y[index], Z[index]), axis=1)
    elif sample_mode == 'prod_iso_kNN':
        # In this case we first pick m=batch_size/K_neighbor x. Then we look for neighbors among the rest of samples
        # Note that in nearest neighbor we should not consider the point itself as neighbor
        m = batch_size // K_neighbor # 400//20
        index_yz = np.random.choice(range(N), size=m, replace=False)
        neigh = NearestNeighbors(n_neighbors=K_neighbor, radius=radius, metric='euclidean')
        # NearestNeighbors(metric='euclidean', n_neighbors=20, radius=1000)
        X2 = np.asarray([element for i, element in enumerate(X) if i not in index_yz])
        Y2 = np.asarray([element for i, element in enumerate(Y) if i not in index_yz])
        Z2 = np.asarray([element for i, element in enumerate(Z) if i not in index_yz])
        neigh.fit(Z2) #
        Neighbor_indices = neigh.kneighbors(Z[index_yz], return_distance=False)
        # Z2 데이터 중 Z[index_yz]에 가까운 20개
        index_x = []
        index_y = []
        index_z = []
        for n_i in Neighbor_indices:
            index_x = np.append(index_x, n_i).astype(int)
        for ind in index_yz:
            index_y = np.append(index_y, [ind] * K_neighbor).astype(int)
            index_z = np.append(index_z, [ind] * K_neighbor).astype(int)

        batch = np.column_stack((X2[index_x], Y[index_y], Z[index_z]))
    return batch

def batch_construction(data, arrange, set_size=100, K_neighbor=2, mode='memoryless', depth=0):   
    n = data[0].shape[0]
    if mode == 'memoryless':
        # n 800 set_size 400 k_nei 20 arrange [[0], [1], [2]]
        train_index = np.random.choice(range(n), size=set_size, replace=False) # size n//2
        test_index = [j for j in range(n) if j not in train_index]        
    
        Train_set = [data[i][train_index] for i in range(len(data))]
        Test_set = [data[i][test_index] for i in range(len(data))]
        # len(train_set) 3, train_set[0].shape (400, 5)

        joint_target = np.repeat([[1,0]],set_size,axis=0)
        # array([[1, 0],
        #         ...,
        #        [1, 0]])
        prod_target = np.repeat([[0,1]],set_size,axis=0)
        target_train = np.concatenate((joint_target,prod_target),axis=0)
        target_train = autograd.Variable(torch.tensor(target_train).float())
        
        joint_train = sample_batch(Train_set, arrange, batch_size=set_size,sample_mode='joint',K_neighbor=K_neighbor)
        prod_train = sample_batch(Train_set, arrange, batch_size=set_size,sample_mode='prod_iso_kNN',K_neighbor=K_neighbor)
        batch_train = autograd.Variable(torch.tensor(np.concatenate((joint_train, prod_train))).float())
        
        joint_test = sample_batch(Test_set, arrange, batch_size=set_size,sample_mode='joint',K_neighbor=K_neighbor)
        joint_test = autograd.Variable(torch.tensor(joint_test).float())
        prod_test = sample_batch(Test_set, arrange, batch_size=set_size,sample_mode='prod_iso_kNN',K_neighbor=K_neighbor)
        prod_test = autograd.Variable(torch.tensor(prod_test).float())
    else:
        print('lib 119')
    return batch_train, target_train, joint_test, prod_test
    # torch.Size([800, 15]) torch.Size([800, 2]) torch.Size([400, 15]) torch.Size([400, 15])

    

# Classifier
class ClassifierModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,tau):
        super().__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        self.h3 = nn.Linear(hidden_size, num_classes)
        self.Tau = tau

    def forward(self, x):
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        x = self.h3(x)
        x = F.softmax(x, dim=1)
        hardT = nn.Hardtanh(self.Tau, 1-self.Tau)
        x = hardT(x)
        return x  


def estimate_CMI(Model, JointBatch, ProdBatch):
    gamma_joint = Model(JointBatch).detach().numpy()[:,0]
    gamma_prod = Model(ProdBatch).detach().numpy()[:,0]
    
    b  = JointBatch.shape[0]
    b_ = ProdBatch.shape[0]
    sum1 = 0
    for j in range(b):  
        sum1 += np.log(gamma_joint[j]/(1-gamma_joint[j]))

    sum2=0
    for j in range(b_):
        sum2 += gamma_prod[j]/(1-gamma_prod[j])
            
    CMI_LDR = (1/b)*sum1
    CMI_DV = (1/b)*sum1 - np.log((1/b_)*sum2)
    CMI_NWJ = 1+(1/b)*sum1 - (1/b_)*sum2        
    return CMI_LDR, CMI_DV, CMI_NWJ
    
def train_classifier(BatchTrain, TargetTrain, Params, Epoch, Lr, Seed, Epsilon=1e-7, Eval=False, JointEval=[], ProdEval=[]):
    loss_e = []
    last_loss = 1000
    CMI_LDR_e = []
    CMI_DV_e = []
    CMI_NWJ_e = []
    #Set up the model
    torch.manual_seed(Seed)
    (input_size, hidden_size, num_classes, tau) = Params # (15, 64, 2, 0.0001)

    model = ClassifierModel(input_size, hidden_size, num_classes, tau)    
    opt = optim.Adam(params=model.parameters(), lr=Lr)

    for epoch in range(int(Epoch)):
        out = model(BatchTrain)
        _, pred = out.max(1)
        loss = F.binary_cross_entropy(out, TargetTrain) 
        loss_e.append(loss.detach().numpy())
        
        if Eval:
            CMI_eval = estimate_CMI(model,JointEval,ProdEval)
            print('epoch: ',epoch,'  ,', CMI_eval[1], ' loss: ',loss_e[-1])        
            CMI_LDR_e.append(CMI_eval[0])
            CMI_DV_e.append(CMI_eval[1])
            CMI_NWJ_e.append(CMI_eval[2])        
        
        if abs(loss-last_loss) < Epsilon and epoch > 50:
            print('epoch=',epoch)
            break

        last_loss = loss
        model.zero_grad()
        loss.backward()
        opt.step()
    if Eval:    
        return model, loss_e, CMI_LDR_e, CMI_DV_e, CMI_NWJ_e
    else:
        return model, loss_e


def create_dataset(GenModel, Params, Dim, N): # 'Gaussian_nonZero', (10,1,5), 1, 8e4)
    # If the dataset=(x,y,z) we compute I(X;Y|Z)    
    if GenModel == 'Gaussian_nonZero':
        (sigma_x, sigma_1, sigma_2) = Params
        x = np.random.multivariate_normal(mean=[0]*Dim,
                                     cov=sigma_x**2*np.eye(Dim),
                                     size=N) 

        y = x + np.random.multivariate_normal(mean=[0]*Dim,
                                     cov=sigma_1**2*np.eye(Dim),
                                     size=N)     

        z = y + np.random.multivariate_normal(mean=[0]*Dim,
                                     cov=sigma_2**2*np.eye(Dim),
                                     size=N)     
        dataset = [x,y,z]

    else:
        print('lib 223')

    return dataset


def prepare_dataset(dataset, Mode='max',S=4):
    if Mode == 'max':
        #map the data between [-S,S]
        #scaling the whole dataset with one factor to be able to compute the true CMI   
        for i in range(len(dataset)):  
            MAX = np.max(dataset[i])
            MIN = np.min(dataset[i])
            
            dataset[i] = 2*S*(dataset[i]-MIN)/(MAX-MIN)-S
        return dataset
    elif Mode == 'norm':
        #Make each column zero mean and devided to SD
        for i in range(len(dataset)):  
            MEAN = dataset[i].mean()
            STD = dataset[i].std()
            dataset[i] = 2*(dataset[i]-MEAN)/STD
        return dataset        
    