# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:27:02 2022

@author: Zhan ao Huang
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
import math
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
        outputs=torch.tanh(net(inputs.view(-1,1,attr_count)))
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

def generate(neg_data,size,attr_count):
    select_neg=np.zeros((size,attr_count))
    for i in range(size):
        idx=np.random.randint(0,len(neg_data))
        select_neg[i]=neg_data[idx]
    return select_neg
def ip(x,sigmma):
    n=len(x)
    rvalue=0
    for i in range(n):
        for j in range(n):
              rvalue+=math.e**(-((x[i]-x[j])**2).sum()/(4*sigmma**2))
    return rvalue/(n**2)
def cip(x,y,sigmma):
    rvalue=0
    for i in range(len(x)):
        for j in range(len(y)):
            rvalue+=math.e**(-((x[i]-y[j])**2).sum()/(4*sigmma**2))
    return rvalue/(len(x)*len(y))
def ris(x,y,sigmma,lam):#x \in y
    C=len(y)*cip(x,y,sigmma)/(len(x)*ip(x,sigmma))
    x_t=np.zeros(x.shape)
    for i in range(len(x)):
        f1,f2,f3,f4,f5,f6=0,0,0,0,0,0
        for j in range(len(y)):
            f2+=math.e**(-((x[i]-y[j])**2).sum()/(4*sigmma**2))
            f3+=(math.e**(-((x[i]-y[j])**2).sum()/(4*sigmma**2))*y[j])
        for k in range(len(x)):
            f1+=(math.e**(-((x[i]-x[k])**2).sum()/(4*sigmma**2))*x[k])
            f5+=math.e**(-((x[i]-x[k])**2).sum()/(4*sigmma**2))
        f4=f2
        f6=f2
        #print(f2,f4,f6,f5)
        x_t[i]=C*((1-lam)/lam)*f1/f2+f3/f4-C*((1-lam)/lam)*f5/f6*x[i]
    
    return x_t


def ris_undersampling(neg_data,attr_count,pos_count):
    select_neg=generate(neg_data,pos_count,attr_count)
    epoch=0
    while True:
        sigmma,lam=5,1000
        sigmma/=(epoch+1)
        select_neg2=ris(select_neg,neg_data,sigmma,lam)
        diff=((select_neg2-select_neg)**2).sum()
        print('ris undersampling: [%d,%.5f]'%(epoch,diff))
        select_neg=select_neg2
        if diff<=5e-5:
            break
        epoch+=1
    return select_neg

def cv_train_au(net,cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count,path):
    init_net=copy.deepcopy(net)
    print(cv_neg_count,attr_count)
    for i in range(len(cv_trainloader)):
        print('%d fold cross validation:'%(i+1))
        train_rius(net,cv_trainloader[i],cv_testloader[i],cv_neg_count[i],cv_pos_count[i],attr_count,i+1,path)
        
        net=init_net



def train_rius(net,trainloader,testloader,cv_neg_count,cv_pos_count,attr_count,fold,path):
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
    
    new_neg_data=ris_undersampling(neg_data.numpy(),attr_count,pos_count)
    new_neg_data=torch.from_numpy(new_neg_data).view(-1,1,attr_count).float()
       
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
            f=open(path+'/log_RIUS/log_RIUS.txt','a')
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
    if os.path.exists(path+'/log_RIUS/log_RIUS.txt'):
        print('Remove log_RIUS.txt')
        os.remove(path+'/log_RIUS/log_RIUS.txt') 
    else:
        os.mkdir(path+'log_RIUS')
    train_data_all,test_data_all=rd.read_data(path)
    cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count=rd.construct_cv_trainloader(train_data_all,test_data_all)
    
    torch.manual_seed(0)
    print(attr_count)
    net=Net(attr_count)
    cv_train_au(net,cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count,path)
if __name__=='__main__':
    path='../abalone_3_vs_11_5_fold/'
    train(path)
