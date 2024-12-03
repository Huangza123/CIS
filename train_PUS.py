# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 15:42:30 2021
@author: Zhan ao Huang
@email: huangzhan_ao@outlook.com
"""

import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score 
import copy
import read_data as rd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import os

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
        train_neighbor_based_undersampling(net,cv_trainloader[i],cv_testloader[i],cv_neg_count[i],cv_pos_count[i],attr_count,i+1,path)
        net=init_net

def train_neighbor_based_undersampling(net,trainloader,testloader,cv_neg_count,cv_pos_count,attr_count,fold,path):
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:1')
    net.to(device)
    
    """train network"""
    EPOCH=10000
    update_center_duration=100
    #optimizer=torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer=torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    loss_func=nn.CrossEntropyLoss()
    
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
    
    #neighbor based undersampling
    #compute density for each instance
    density=torch.zeros(cv_neg_count)
    for i in range(len(neg_data)):
        for j in range(len(neg_data)):
            if j!=i:
                _dis=dis_compute(neg_data[i],neg_data[j])
                density[i]+=np.e**(-_dis**2)
    
    #compute distance for each instance
    distance=torch.zeros(cv_neg_count)
    for i in range(len(neg_data)):
        if density[i]==max(density):
            tmp_dis=[]
            for j in range(len(neg_data)):
                if j!=i:
                    _dis=dis_compute(neg_data[i],neg_data[j])
                    tmp_dis.append(_dis)
            distance[i]=max(tmp_dis)
        else:
            tmp_dis=[]
            for j in range(len(neg_data)):
                if density[i]<density[j] and j!=i:
                    _dis=dis_compute(neg_data[i],neg_data[j])
                    tmp_dis.append(_dis)
            distance[i]=min(tmp_dis)
    factor=(density**2)*distance
    new_neg_data=torch.zeros((cv_pos_count,1,attr_count)).to(device)
    for k in range(len(new_neg_data)):
        max_idx=torch.argmax(factor)
        new_neg_data[k]=neg_data[max_idx]
        factor[max_idx]=torch.min(factor)
    
    new_neg_label=torch.zeros(len(new_neg_data))
    pos_label=torch.ones(len(pos_data))
    train_data2=torch.cat((new_neg_data,pos_data),axis=0)
    train_label2=torch.cat((new_neg_label,pos_label),axis=0)
    
    torch_train_dataset=Data.TensorDataset(train_data2,train_label2)
    trainloader2=Data.DataLoader(dataset=torch_train_dataset,
                                batch_size=1,
                                shuffle=True)
    
    for epoch in range(EPOCH):
        for i,data in enumerate(trainloader2):
            inputs,labels=data
            inputs=inputs.to(device)
            labels=labels.to(device)
            
            outputs=net(inputs.view(-1,1,attr_count))
            loss=loss_func(outputs.view(-1,2),labels.view(-1).long())        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % update_center_duration==0:
            f=open(path+'/log_PUS/log_PUS.txt','a')
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



def dis_compute(data0,data1):
    return torch.sqrt(((data0-data1)**2).sum())   
def train3(path):
    if os.path.exists(path+'/log_PUS/log_PUS.txt'):
        print('Remove log_PUS.txt')
        os.remove(path+'/log_PUS/log_PUS.txt') 
    elif os.path.exists(path+'/log_PUS'):
        pass
    else:
        os.mkdir(path+'log_PUS')
    train_data_all,test_data_all=rd.read_data(path)
    cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count=rd.construct_cv_trainloader(train_data_all,test_data_all)
    
    torch.manual_seed(0)
    net=Net(attr_count)
    cv_train_au(net,cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count,path)
if __name__=='__main__':
    path='../abalone_3_vs_11_5_fold/'
    train3(path)