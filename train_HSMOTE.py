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
        train_hsmote(net,cv_trainloader[i],cv_testloader[i],cv_neg_count[i],cv_pos_count[i],attr_count,i+1,path)
        
        net=init_net

def hsmote(pos_data,target_num):
    generate_pos_data=[]
    position=torch.ones(len(pos_data))
    for i in range(pos_data.shape[0]):
        for j in range(pos_data.shape[1]):
            position[i]*=pos_data[i,j]
    
    position_max,position_min=torch.max(position),torch.min(position)
    c=30
    bin_width=(position_max-position_min)/30
    if bin_width==0:
        return pos_data[torch.randint(0,len(pos_data),(target_num,1))]
    intervels=torch.arange(position_min, position_max, bin_width)
    bins=torch.zeros(c)
    bins_avaliable=[]
    for i in range(c):
        bins_avaliable.append([])
    
    for j in range(len(position)):
        for i in range(c-1):
            if position[j]>=intervels[i] and position[j]<intervels[i+1]:
                bins[i]+=1
                bins_avaliable[i].append(pos_data[j])
    phi=torch.round(target_num/(bins>0).sum())
    #print(phi,(bins>0).sum())
    sizeofexceed=0
    
    for i in range(c):
        if bins[i]>phi:
            sizeofexceed+=bins[i]
    phi_new=torch.round((target_num-sizeofexceed)/(bins>0).sum())
    
    #smote in bins_avaliable
    for i in range(len(bins_avaliable)):
        if bins[i]>0 and bins[i]<phi:
            sampling_num=torch.ceil(phi_new-bins[i])
            data=bins_avaliable[i]
            
            while sampling_num:
                idx=torch.randint(0,int(bins[i]),(1,))
                dis=torch.zeros(len(data))
                for j in range(len(data)):
                    dis[j]=torch.sum((data[idx]-data[j])**2)
                
                if len(dis)==1:
                    generate_pos_data.append(data[0].numpy())
                    sampling_num-=1
                    if sampling_num==0:
                        break
                elif 1<len(dis) and len(dis)<5:
                    dis[torch.argmin(dis)]=torch.max(dis)
                    for k in range(len(dis)):
                        if dis[k]!=0:
                            generate_pos_data.append((data[idx]+(data[k]-data[idx])*torch.rand(1)).numpy())
                            sampling_num-=1
                            if sampling_num==0:
                                break
                    else:
                        generate_pos_data.append((data[idx].numpy()))
                        sampling_num-=1
                        if sampling_num==0:
                            break
                else:
                    dis[torch.argmin(dis)]=torch.max(dis)
                    neighbor=[]
                    for k in range(5):
                        generate_pos_data.append((data[idx]+(data[torch.argmin(dis)]-data[idx])*torch.rand(1)).numpy())
                        dis[torch.argmin(dis)]=torch.max(dis)
                        sampling_num-=1
                        if sampling_num==0:
                            break
    generate_pos_data=torch.from_numpy(np.array(generate_pos_data))
    new_pos_data=torch.concatenate((generate_pos_data,pos_data),axis=0)
    return new_pos_data
    
def train_hsmote(net,trainloader,testloader,cv_neg_count,cv_pos_count,attr_count,fold,path):
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
    tensor_original_pos_data=(pos_data).float()
    tensor_neg_label=torch.zeros(len(neg_data)).long()
    
    # hsmote
    pos_data=hsmote(pos_data.reshape(-1,attr_count),len(neg_data)).reshape(-1,1,attr_count)
    tensor_pos_label=torch.ones(len(pos_data)).long()
    # print(pos_data.shape,tensor_original_neg_data.shape)
    for epoch in range(EPOCH):
        neg_outputs=net(tensor_original_neg_data)
        pos_outputs=net(pos_data)
        outputs=torch.concatenate((neg_outputs,pos_outputs),axis=0)
        labels=torch.concatenate((tensor_neg_label,tensor_pos_label),axis=0)
        
        loss=loss_func(outputs,labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if epoch % update_center_duration==0:
            f=open(path+'/log_HSMOTE/log_HSMOTE.txt','a')
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
    if os.path.exists(path+'/log_HSMOTE/log_HSMOTE.txt'):
        print('Remove log_HSMOTE.txt')
        os.remove(path+'/log_HSMOTE/log_HSMOTE.txt') 
    elif os.path.exists(path+'/log_HSMOTE'):
        pass
    else:
        os.mkdir(path+'log_HSMOTE')
    train_data_all,test_data_all=rd.read_data(path)
    cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count=rd.construct_cv_trainloader(train_data_all,test_data_all)
    
    torch.manual_seed(0)
    print(attr_count)
    net=Net(attr_count)
    cv_train_au(net,cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count,path)
if __name__=='__main__':
    path='../abalone_3_vs_11_5_fold/'
    train(path)
