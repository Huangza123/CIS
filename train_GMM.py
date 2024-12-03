# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:27:02 2022

@author: zhan ao huang

"""
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score 
import copy
import read_data as rd
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import os
from sklearn.mixture import GaussianMixture
# import warnings
# warnings.filterwarnings("ignore")
# os.environ["OMP_NUM_THREADS"] = '1'
class Net(nn.Module):
    def __init__(self,attr_count):
        super(Net,self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(attr_count,10),
            nn.ReLU(),
            nn.Linear(10,2),
            nn.ReLU(),
            )
    def forward(self,x):
        return self.layer(x.view(x.size(0),-1))
    
def test_confusion_matrix(net,testloader,attr_count,device):
    true_labels=[]
    predict=[]

    for i,data in enumerate(testloader):
        inputs,labels=data
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=torch.softmax(torch.tanh(net(inputs.view(-1,1,attr_count))),dim=1)
       
        _,_predict=torch.max(outputs,1)
        
        if _predict==0:
            tmp_predict=0
        else:
            tmp_predict=1
        
        true_labels.append(labels[0].item())
        predict.append(tmp_predict)
    cmatrix=confusion_matrix(true_labels,predict)
    f1=f1_score(true_labels,predict)
    TN,FP,FN,TP=cmatrix.ravel()
    
    TPR,TNR=0,0
    if TP+FN>0:
        TPR=TP/(TP+FN)
    if FP+TN>0:
        TNR=TN/(FP+TN)
    macc=(TPR+TNR)/2
    g_mean=(TPR*TNR)**0.5
    return TPR,TNR,macc,g_mean,f1


def cv_train_au(net,cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count,path):
    init_net=copy.deepcopy(net)
    print(cv_neg_count,attr_count)
    for i in range(len(cv_trainloader)):
        print('%d fold cross validation:'%(i+1))
        train_gmm(net,cv_trainloader[i],cv_testloader[i],cv_neg_count[i],cv_pos_count[i],attr_count,i+1,path)
        
        net=init_net

    
def train_gmm(net,trainloader,testloader,cv_neg_count,cv_pos_count,attr_count,fold,path):
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:1')
    net.to(device)
    
    """train network"""
    EPOCH=10000
    optimizer=torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    loss_func=nn.CrossEntropyLoss()
    update_center_duration=100

    neg_data=torch.zeros((cv_neg_count,1,attr_count)).to(device)
    pos_data=torch.zeros((cv_pos_count,1,attr_count)).to(device)
    neg_count=0
    pos_count=0
    for i, data in enumerate(trainloader):
        inputs,labels=data
        if labels.item()==0:
            neg_data[neg_count]=inputs
            neg_count+=1
        else:
            pos_data[pos_count]=inputs
            pos_count+=1
    tensor_original_neg_data=(neg_data).float()
    tensor_neg_label=torch.zeros(len(neg_data)).long()
    tensor_original_pos_data=(pos_data).float()
    tensor_pos_label=torch.ones(len(pos_data)).long()
    #svdd
    
    n_components,l_best,l=1,0,0
    gmm_data_pos=(pos_data).numpy().reshape(-1,attr_count)
    gmm_data_neg=(neg_data).numpy().reshape(-1,attr_count)
    
    gmm_data=np.concatenate((gmm_data_pos,gmm_data_neg), axis=0)
    gmm_label=np.concatenate((np.ones(len(pos_data)),np.zeros(len(neg_data))),axis=0)
   
    gmm = GaussianMixture(n_components=n_components,random_state=0).fit(gmm_data_pos)
    l=gmm.score(gmm_data_pos)
    l_best=l
    while True:
        n_components+=1
        if n_components>=len(gmm_data_pos):
            break
        gmm = GaussianMixture(n_components=n_components,random_state=0).fit(gmm_data_pos)
        l=gmm.score(gmm_data)
        
        if l_best<l:
            break
        l_best=l
        
    gmm = GaussianMixture(n_components=n_components-1,random_state=0).fit(gmm_data)  
    Q=np.zeros(n_components-1)
    
    for k in range(n_components-1):
        
        #safe level computation
        neighbor_k=5
        
        for pos in gmm_data_pos:
            p_k=gmm.predict_proba(pos.reshape(-1,attr_count))[:,k]
            
            dis_pos=(pos-gmm_data_pos).sum(axis=1)
            dis_neg=(pos-gmm_data_neg).sum(axis=1)
            dis=np.concatenate((dis_pos,dis_neg),axis=0)
            neighbor_label=[]
            for _ in range(neighbor_k):
                idx=np.argmin(dis)
                dis[idx]=max(dis)
                neighbor_label.append(gmm_label[idx])
            
            neighbor_label=np.array(neighbor_label)
            n0=np.sum(neighbor_label==0)
            n1=np.sum(neighbor_label==1)
            u11=1.
            u10=len(gmm_data_pos)/len(gmm_data_neg)
            save_level=(n0*u10+n1*u11)/neighbor_k
            
            Q[k]+=(1-save_level)*p_k
    Q=Q/np.sum(Q)
    gmm_sample_data,gmm_sample_label=gmm.sample(len(gmm_data_neg)*n_components)
    sample_num=len(neg_data)-len(pos_data)
    Q=Q*sample_num
    sample_pos_set=[]
    
    for q in range(len(Q)):
        if int(Q[q])>0:
            _data=gmm_sample_data[gmm_sample_label==q]
            idx=np.random.randint(0,len(_data),int(Q[q]))
            for data in _data[idx]:
                sample_pos_set.append(data)
    sample_pos_set=np.array(sample_pos_set).reshape(-1,attr_count)
    tensor_sample_pos_data=torch.from_numpy(sample_pos_set).float().reshape(-1,1,attr_count)
    tensor_sample_pos_label=torch.ones(len(sample_pos_set)).long()
    tensor_pos_data=torch.concat((tensor_sample_pos_data,tensor_original_pos_data),axis=0)
    tensor_pos_label=torch.concat((tensor_sample_pos_label,tensor_pos_label),axis=0)
    
   
    for epoch in range(EPOCH):
        neg_outputs=net(tensor_original_neg_data)
        pos_outputs=net(tensor_pos_data)
        
        loss_neg=loss_func(neg_outputs,tensor_neg_label)
        loss_pos=loss_func(pos_outputs,tensor_pos_label)
        loss=loss_neg+loss_pos
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if epoch % update_center_duration==0:
            f=open(path+'/log_GMM/log_GMM.txt','a')
            TPR,TNR,macc,g_mean,f1=test_confusion_matrix(net,testloader,attr_count,device)
            print('[%d,%d]'%(epoch,EPOCH),'loss=%.3f'%loss.item(),
                  'F1=%.3f'%f1,
                  'MACC=%.3f'%macc,
                  'G-MEAN=%.3f'%g_mean,
                  'TPR=%.3f'%TPR,
                  'TNR=%.3f'%TNR,
                  )
            
            f.write(str(['[%d,%d]'%(epoch,EPOCH),'loss=%.3f'%loss.item(),
                  'F1=%.3f'%f1,
                  'MACC=%.3f'%macc,
                  'G-MEAN=%.3f'%g_mean,
                  'TPR=%.3f'%TPR,
                  'TNR=%.3f'%TNR]))      
            f.write('\n')
            f.close()
            if (f1*macc*g_mean)==1.0:
                break

def train(path):
    if os.path.exists(path+'/log_GMM/log_GMM.txt'):
        print('Remove log_GMM.txt')
        os.remove(path+'/log_GMM/log_GMM.txt') 
    elif os.path.exists(path+'/log_GMM'):
        pass
    else:
        os.mkdir(path+'log_GMM')
    train_data_all,test_data_all=rd.read_data(path)
    cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count=rd.construct_cv_trainloader(train_data_all,test_data_all)
    
    torch.manual_seed(0)
    print(attr_count)
    net=Net(attr_count)
    cv_train_au(net,cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count,path)
if __name__=='__main__':
    path='../abalone_3_vs_11_5_fold/'
    train(path)
