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
        train_dwbl(net,cv_trainloader[i],cv_testloader[i],cv_neg_count[i],cv_pos_count[i],attr_count,i+1,path)
        
        net=init_net

class DWBLoss(nn.Module):
    def __init__(self,weight):
        super(DWBLoss, self).__init__()
        self.weight=weight
    def forward(self,outputs,labels):
        sft_outputs=torch.softmax(outputs,dim=1)
        loss0_set=-self.weight[0][0]**(1-sft_outputs[:,0][labels==0])*torch.log(sft_outputs[:,0][labels==0])-sft_outputs[:,0][labels==0]*(1-sft_outputs[:,0][labels==0])
        loss1_set=-self.weight[0][1]**(1-sft_outputs[:,1][labels==1])*torch.log(sft_outputs[:,1][labels==1])-sft_outputs[:,1][labels==1]*(1-sft_outputs[:,1][labels==1])
        loss=torch.mean(torch.cat((loss0_set,loss1_set),axis=0))
       
        return loss
    
    
def train_dwbl(net,trainloader,testloader,cv_neg_count,cv_pos_count,attr_count,fold,path):
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:1')
    net.to(device)
    
    """train network"""
    EPOCH=10000
    optimizer=torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    
    nn.CrossEntropyLoss()
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
    tensor_original_pos_data=(pos_data).float()
    tensor_neg_label=torch.zeros(len(neg_data))
    tensor_pos_label=torch.ones(len(pos_data))
    labels=torch.cat((tensor_neg_label,tensor_pos_label),axis=0)
    #compute dynamical weight for dwl
    weights=torch.zeros((1,2))
    weights[0][0]=1
    
    weights[0][1]=np.log(neg_data.shape[0]/pos_data.shape[0])+1
    #print(neg_data.shape,pos_data.shape,weights,neg_data.shape[0]/pos_data.shape[0])
    loss_func=DWBLoss(weights)
    for epoch in range(EPOCH):
        neg_outputs=net(tensor_original_neg_data)
        pos_outputs=net(tensor_original_pos_data)
        outputs=torch.cat((neg_outputs,pos_outputs),axis=0)
        loss=loss_func(outputs,labels)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if epoch % update_center_duration==0:
            f=open(path+'/log_DWBL/log_DWBL.txt','a')
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
    if os.path.exists(path+'/log_DWBL/log_DWBL.txt'):
        print('Remove log_DWBL.txt')
        os.remove(path+'/log_DWBL/log_DWBL.txt') 
    elif os.path.exists(path+'/log_DWBL'):
        pass
    else:
        os.mkdir(path+'log_DWBL')
    train_data_all,test_data_all=rd.read_data(path)
    cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count=rd.construct_cv_trainloader(train_data_all,test_data_all)
    
    torch.manual_seed(0)
    print(attr_count)
    net=Net(attr_count)
    cv_train_au(net,cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count,path)
if __name__=='__main__':
    path='../abalone_3_vs_11_5_fold/'
    train(path)
