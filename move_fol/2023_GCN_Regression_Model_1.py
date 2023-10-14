#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import argparse
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from torch import nn
from torch import optim

import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool
from torch_geometric.data import DataLoader, Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.Polypeptide import three_to_one
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.database.rcsb as rcsb
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm.notebook import tqdm
import parmap
import time

import re
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import ndtr
from modAL.utils.selection import multi_argmax
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit
import warnings
warnings.filterwarnings(action='ignore')
from torch_geometric.data import Batch


# In[31]:


class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint_1.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# In[48]:


paser = argparse.ArgumentParser()
args = paser.parse_args("")
args.seed = 123
args.test_size = 0.2
args.shuffle = True
    
g = torch.Generator()
g.manual_seed(0)

pdb_path = './Data/pdb/'
graph_path = './Data/graph/'


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(args.seed)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# In[33]:


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

AA = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
def aa_features(x):
    return np.array(one_of_k_encoding(x, AA))

def adjacency2edgeindex(adjacency):
    start = []
    end = []
    adjacency = adjacency - np.eye(adjacency.shape[0], dtype=int)
    for x in range(adjacency.shape[1]):
        for y in range(adjacency.shape[0]):
            if adjacency[x, y] == 1:
                start.append(x)
                end.append(y)

    edge_index = np.asarray([start, end])
    return edge_index

AMINOS =  ['CYS', 'ASP', 'SER', 'GLN', 'LYS', 'ILE', 'PRO', 'THR', 'PHE', 'ASN', 
           'GLY', 'HIS', 'LEU', 'ARG', 'TRP', 'ALA', 'VAL', 'GLU', 'TYR', 'MET']
def filter_20_amino_acids(array):
    return ( np.in1d(array.res_name, AMINOS) & (array.res_id != -1) )

def protein_analysis(pdb_id):
    pdb_name=os.listdir(pdb_path)
    protein_name=[re.sub('.pdb', '', i) for i in pdb_name]
    
    if pdb_id not in protein_name:
        file_name = rcsb.fetch(pdb_id, "pdb", pdb_path)
        array = strucio.load_structure(file_name)
        protein_mask = filter_20_amino_acids(array)
        try:
            array = array[protein_mask]
        except:
            array = array[0]
            array = array[protein_mask]
        try:
            ca = array[array.atom_name == "CA"]
        except:
            array = array[0]
            ca = array[array.atom_name == "CA"]
        seq = ''.join([three_to_one(str(i).split(' CA')[0][-3:]) for i in ca])
        threshold = 7
        cell_list = struc.CellList(ca, cell_size=threshold)
        A = cell_list.create_adjacency_matrix(threshold)
        A = np.where(A == True, 1, A)
        return [aa_features(aa) for aa in seq], adjacency2edgeindex(A)
    
    if pdb_id in protein_name:
        array = strucio.load_structure('./Data/pdb/'+pdb_id+".pdb")
        protein_mask = filter_20_amino_acids(array)
        try:
            array = array[protein_mask]
        except:
            array = array[0]
            array = array[protein_mask]
        try:
            ca = array[array.atom_name == "CA"]
        except:
            array = array[0]
            ca = array[array.atom_name == "CA"]
        seq = ''.join([three_to_one(str(i).split(' CA')[0][-3:]) for i in ca])
        threshold = 7
        cell_list = struc.CellList(ca, cell_size=threshold)
        A = cell_list.create_adjacency_matrix(threshold)
        A = np.where(A == True, 1, A)
        return [aa_features(aa) for aa in seq], adjacency2edgeindex(A)

def pro2vec(pdb_id):
    node_f, edge_index = protein_analysis(pdb_id)
    data = Data(x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long))
    return data


def make_pro(df, target):
    pro_key = []
    pro_value = []
    for i in range(df.shape[0]):
        pro_key.append(df['PDB'].iloc[i])
        pro_value.append(df[target].iloc[i])
    return pro_key, pro_value

    
def save_graph(graph_path, pdb_id):
    vec = pro2vec(pdb_id)
    np.save(graph_path+pdb_id+'_e.npy', vec.edge_index)
    np.save(graph_path+pdb_id+'_n.npy', vec.x)
    
def load_graph(graph_path, pdb_id):
    n = np.load(graph_path+str(pdb_id)+'_n.npy')
    e = np.load(graph_path+str(pdb_id)+'_e.npy')
    N = torch.tensor(n, dtype=torch.float)
    E = torch.tensor(e, dtype=torch.long)
    data = Data(x=N, edge_index=E)
    return data

def make_vec(pro, value, graph_path):
    X = []
    Y = []
    for i in range(len(pro)):
        m = pro[i]
        y = value[i]
        v = load_graph(graph_path, m)
        if v.x.shape[0] < 100000:
            X.append(v)
            Y.append(y)
            
    for i, data in enumerate(X):
        y = Y[i]
        #data.y = torch.tensor([y], dtype=torch.long)
        data.y = torch.tensor([y], dtype=torch.float)#flaot
    return X

def make_vec_test(pro, values, graph_path):
    index = test_pro_key.index(pro)
    y = values[index]
    v = load_graph(graph_path, pro)
    if v.x.shape[0] < 100000:
        v.y = torch.tensor([y], dtype=torch.float)
    return v



def generate_graph(pdb, graph_path):
    done = 0
    while done == 0:
        graph_dirs = list(set([d[:-6] for d in os.listdir(graph_path)]))
        if pdb not in graph_dirs:
            try:
                save_graph(graph_path,pdb)
                done = 1
                return 1
            except:
                done = 1
                return 0
        else:
            done = 1
            return 1


# In[34]:


def save_checkpoint(epoch, model, optimizer, filename):
    state = {'Epoch': epoch, 
             'State_dict': model.state_dict(), 
             'optimizer': optimizer.state_dict()}
    torch.save(state, filename)

def train(model, device, optimizer, train_loader, criterion, args):
    epoch_train_loss = 0
    for i, pro in enumerate(train_loader):
        pro, labels = pro.to(device), pro.y.to(device)  # Modify pro and labels extraction
        optimizer.zero_grad()
        outputs = model(pro)
        outputs.require_grad = False
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs.flatten(), labels)
        epoch_train_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_train_loss /= len(train_loader)
    return model, epoch_train_loss


def test(model, device, test_loader, criterion, args):
    model.eval()
    data_total = []
    pred_data_total = []
    epoch_test_loss = 0
    with torch.no_grad():
        for i, pro in enumerate(test_loader):
            pro, labels= pro.to(device), pro.y.to(device)
            data_total += pro.y.tolist()
            outputs = model(pro)
            pred_data_total += outputs.view(-1).tolist()
            loss = criterion(outputs.flatten(), labels)
            epoch_test_loss += loss.item()
    epoch_test_loss /= len(test_loader)
    return data_total, pred_data_total, epoch_test_loss

def experiment(model, train_loader, test_loader, device, args):
    time_start = time.time()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.step_size,
                                          gamma=args.gamma)

    list_train_loss = []
    list_test_loss = []
    print('[Train]')
    
    early_stopping = EarlyStopping(patience = patience, verbose = True)    
    for epoch in range(args.epoch):
        scheduler.step()
        model, train_loss = train(model, device, optimizer, train_loader, criterion, args)
        _, _, test_loss = test(model, device, test_loader, criterion, args)
        list_train_loss.append(train_loss)
        list_test_loss.append(test_loss)
        
        if epoch > 20:
            early_stopping(test_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            model.load_state_dict(torch.load('checkpoint_1.pt'))
        #model = early_stopper.load(model)    
        print('- Epoch: {0}, Train Loss: {1:0.6f}, Test Loss: {2:0.6f}'.
              format(epoch + 1, train_loss, test_loss))

    print()
    print('[Test]')
    data_total, pred_data_total, _ = test(model, device, test_loader, criterion, args)
    print('- R2: {0:0.4f}'.format(r2_score(data_total, pred_data_total)))
    solution = pd.DataFrame(data_total, columns=["test"])
    answer = pd.DataFrame(pred_data_total, columns=["answer"])
    csv = pd.concat([solution, answer], axis=1)
    args.csv = csv
    time_end = time.time()
    time_required = time_end - time_start

    args.list_train_loss = list_train_loss
    args.list_test_loss = list_test_loss
    args.data_total = data_total
    args.pred_data_total = pred_data_total

    save_checkpoint(args.epoch, model, optimizer, './mymodel_1.pt')  # Pass epoch instead of epoch variable

    return args


# In[35]:


class GCNlayer(nn.Module):
    def __init__(self, n_features, conv_dim1, conv_dim2,
                 conv_dim3, conv_dim4, conv_dim5, concat_dim, dropout):
        
        super(GCNlayer, self).__init__()
        self.n_features = n_features
        self.conv_dim1 = conv_dim1
        self.conv_dim2 = conv_dim2
        self.conv_dim3 = conv_dim3
        self.conv_dim4 = conv_dim4
        self.conv_dim5 = conv_dim5
        self.concat_dim = concat_dim
        self.dropout = dropout
        
        self.conv1 = GCNConv(self.n_features, self.conv_dim1)
        self.bn1 = BatchNorm1d(self.conv_dim1)
        self.conv2 = GCNConv(self.conv_dim1, self.conv_dim2)
        self.bn2 = BatchNorm1d(self.conv_dim2)
        self.conv3 = GCNConv(self.conv_dim2, self.conv_dim3)
        self.bn3 = BatchNorm1d(self.conv_dim3)
        self.conv4 = GCNConv(self.conv_dim3, self.conv_dim4)
        self.bn4 = BatchNorm1d(self.conv_dim4)
        self.conv5 = GCNConv(self.conv_dim4, self.conv_dim5)
        self.bn5 = BatchNorm1d(self.conv_dim5)
        self.conv6 = GCNConv(self.conv_dim5, self.concat_dim)
        
    def forward(self, data):
        edge_index, x = data.edge_index, data.x
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = F.relu(self.conv6(x, edge_index))
        x = global_add_pool(x, data.batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
class FClayer(nn.Module):
    def __init__(self, concat_dim, pred_dim1, out_dim, dropout):
        super(FClayer, self).__init__()
        self.concat_dim = concat_dim
        self.pred_dim1 = pred_dim1  
        self.out_dim = out_dim
        self.dropout = dropout

        self.fc1 = Linear(self.concat_dim, self.pred_dim1)
        self.fc2 = Linear(self.pred_dim1, self.out_dim)
    
    def forward(self, data):
        x = self.fc1(data)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x
    
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = GCNlayer(args.n_features, 
                              args.conv_dim1, 
                              args.conv_dim2, 
                              args.conv_dim3,
                              args.conv_dim4,
                              args.conv_dim5,
                              args.concat_dim, 
                              args.dropout)
        
        self.fc = FClayer(args.concat_dim, 
                          args.pred_dim1, 
                          args.out_dim, 
                          args.dropout)
        
    def forward(self, pro):
        x = self.conv1(pro)
        x = self.fc(x)
        return x


# In[36]:


class MyDataset(torch.utils.data.IterableDataset):
    def __init__(self, pro_list, value_list, graph_path):
        self.pro_list = pro_list
        self.value_list = value_list
        self.graph_path = graph_path
        self.length = len(pro_list)  # Set the length to the number of samples in your dataset

    def __iter__(self):
        for pro, value in zip(self.pro_list, self.value_list):
            v = load_graph(self.graph_path, pro)
            v.y = torch.tensor([value], dtype=torch.float)
            yield v
#
#
    def __len__(self):
        return self.length


def make_vec(pro_list, value_list, graph_path):
    #print(pro_list)
    dataset = MyDataset(pro_list, value_list, graph_path)
    return dataset


# In[37]:


def custom_collate(batch):
    # Implement your custom collation logic here
    # This function should take a list of batch elements and return a batch tensor
    
    # Example implementation for concatenating the graphs in the batch:
    batched_graph = Batch.from_data_list(batch)
    return batched_graph

# In[50]:

patience = 10

args.epoch = 300
args.lr = 0.0001
dim = 32
args.batch_size = 128
args.dropout = 0.2
args.optim = 'Adam'
args.step_size = 10
args.gamma = 0.9
args.n_features = 20
args.conv_dim1 = dim
args.conv_dim2 = 4*dim
args.conv_dim3 = 8*dim
args.conv_dim4 = 16*dim
args.conv_dim5 = 32*dim
args.concat_dim = dim
args.pred_dim1 = dim
args.out_dim = 1
args.exp_name = 'Protein Aggragation'

r2_list=[]
mae_list=[]
train_loss=[]
test_loss=[]

# In[7]:

df = pd.read_csv('train.csv')
#df = df[0:100]
df = df[["PDB", "y_dynamic"]]


test_df = pd.read_csv('test.csv')
#test_df=test_df[0:100]
test_df=test_df[["PDB", "y_dynamic"]]


# In[41]:

train_pro_key, train_pro_value = make_pro(df, "y_dynamic")
test_pro_key, test_pro_value = make_pro(test_df, "y_dynamic")

X_train_key, X_val_key, y_train_value, y_val_value = train_test_split(train_pro_key,
                                                                       train_pro_value,
                                                                       test_size=0.2,
                                                                       random_state=0)

pro_list_train = X_train_key  
values_list_train = y_train_value
pro_list_val = X_val_key  
values_list_val = y_val_value 

graph_paths =  './Data/graph/'


# In[63]:

dataset_train = make_vec(pro_list_train, values_list_train, graph_paths)
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                               collate_fn=custom_collate, num_workers=4)

dataset_val = make_vec(pro_list_val, values_list_val, graph_paths)
dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, 
                                             collate_fn=custom_collate, num_workers=4)

dataset_test = make_vec(test_pro_key, test_pro_value, graph_paths)
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, 
                                             collate_fn=custom_collate, num_workers=4)

model = Net(args)
model = model.to(device)

dict_result = dict()
args.exp_name = 'Protein Aggragation'

result = vars(experiment(model,
                         dataloader_train,
                         dataloader_val,
                         device,
                         args))

dict_result[args.exp_name] = result
torch.cuda.empty_cache()
new_result=result['csv']
train_=result["list_train_loss"]
test_=result["list_test_loss"]
r2=r2_score(new_result["test"], new_result["answer"])
mae=mean_absolute_error(new_result["test"], new_result["answer"])

res=pd.concat((pd.DataFrame(train_, columns=['train_loss']),
           pd.DataFrame(test_, columns=['test_loss']), new_result), axis=1)
#res.to_csv('./result.csv')


# In[67]:

model.eval()
data_total = []
pred_data_total = []
epoch_test_loss = 0
criterion = nn.MSELoss()

with torch.no_grad():
    for i, pro in enumerate(dataloader_test):
        pro, labels= pro.to(device), pro.y.to(device)
        data_total += pro.y.tolist()
        outputs = model(pro)
        pred_data_total += outputs.view(-1).tolist()
        loss = criterion(outputs.flatten(), labels)
        epoch_test_loss += loss.item()
    epoch_test_loss /= len(dataloader_test)
print()
print('[alphafold]')
print('- R2 : {0:0.4f}'.format(r2_score(data_total, pred_data_total)))
# In[70]:

pred_data_total = pd.DataFrame(pred_data_total, columns=['1']) 
pred_data_total = pd.concat([test_df[test_df.columns[0:2]], pred_data_total],axis=1)

iter_num = "Iter_3"
res.to_csv(f'./Result/{iter_num}/result_1.csv',index=False)
pred_data_total.to_csv(f"./Result/{iter_num}/alphafold_prediction_1.csv",index=False)


