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
        train_cis(net,cv_trainloader[i],cv_testloader[i],cv_neg_count[i],cv_pos_count[i],attr_count,i+1,path)
        
        net=init_net

    
def train_cis(net,trainloader,testloader,cv_neg_count,cv_pos_count,attr_count,fold,path):
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
    
    n_clusters=15  #b_0
    
    dis_factor,kmeans_neg_set,cluster_center_neg=important_subcluster_position(neg_data.numpy(),pos_data.numpy(),n_clusters,attr_count)

    tensor_pos_data=pos_data
    tensor_pos_label=torch.ones(len(tensor_pos_data)).long()
    tensor_cluster_center_neg=torch.from_numpy(np.array(cluster_center_neg)).float()
    

    top_count_index=4  #m
    eta=0.9  #lambda
    for epoch in range(EPOCH):
        pos_output=net(tensor_pos_data.view(-1,attr_count))
        pos_loss=loss_func(pos_output,tensor_pos_label)

        cluster_center_neg_output=net(tensor_cluster_center_neg)
        loss_factor=1-torch.exp(torch.log(torch.softmax(cluster_center_neg_output,axis=1)[:,0]))
        cif_factor=torch.from_numpy(dis_factor)+eta*loss_factor
        stratified_importance_neg_data_all=[]
        
        selected_index=[]
        for i in range(top_count_index):
            max_cif_factor=torch.max(cif_factor)
            for j in range(len(cif_factor)):
                if max_cif_factor==cif_factor[j]:
                    stratified_importance_neg_data_all.append(kmeans_neg_set[j])
                    cif_factor[j]=torch.min(cif_factor)
                    selected_index.append(j)
                    break
        
        stratified_importance_neg_data_array=[]
        for i in range(len(stratified_importance_neg_data_all)):
            for j in range(len(stratified_importance_neg_data_all[i])):
                stratified_importance_neg_data_array.append(stratified_importance_neg_data_all[i][j])

        remained_neg_data_all=[]
        for i in range(len(kmeans_neg_set)):
            if i not in selected_index:
                for j in range(len(kmeans_neg_set[i])):
                    remained_neg_data_all.append(kmeans_neg_set[i][j])
        remained_sampling_count=len(stratified_importance_neg_data_array)//2 #delta=2
        
        for i in range(remained_sampling_count):
            idx=np.random.randint(0,len(remained_neg_data_all))
            stratified_importance_neg_data_array.append(remained_neg_data_all[idx])


        stratified_importance_neg_data_array=np.array(stratified_importance_neg_data_array)
        tensor_neg_data=torch.from_numpy(stratified_importance_neg_data_array).float()
        tensor_neg_label=torch.zeros(len(tensor_neg_data)).long()
        neg_output=net(tensor_neg_data)
        neg_loss=loss_func(neg_output,tensor_neg_label)

        loss=pos_loss+neg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if epoch % update_center_duration==0:
            f=open(path+'/log_CIS/log_CIS.txt','a')
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
def important_subcluster_position(neg_data,pos_data,n_clusters,attr_count):
    neg_data=neg_data.reshape(-1,attr_count)
    pos_data=pos_data.reshape(-1,attr_count)
    kmeans_neg=KMeans(n_clusters=n_clusters).fit(neg_data)
    
    kmeans_neg_set=[]
    for i in range(n_clusters):
        kmeans_neg_set.append([])
        
    for i in range(len(neg_data)):
        kmeans_neg_set[kmeans_neg.labels_[i]].append(neg_data[i])
    
    cluster_centers_neg=[]
    for i in range(n_clusters):
        cluster_centers_neg.append(np.array(kmeans_neg_set[i]).mean(axis=0))
    
    # center_pos=np.mean(pos_data,axis=0)
    # dis_factor=np.zeros(n_clusters)
    # for i in range(n_clusters):
    #     dis_factor[i]=np.sqrt(np.sum((cluster_centers_neg[i]-center_pos)**2))
        
    dis_factor=np.zeros(n_clusters)
    tmp_pos_dis=np.zeros(len(pos_data))
    for i in range(n_clusters):
        for j in range(len(pos_data)):
            tmp_pos_dis[j]=np.sqrt(np.sum((cluster_centers_neg[i]-pos_data[j])**2))
        dis_factor[i]=min(tmp_pos_dis)
    return np.exp(-dis_factor),kmeans_neg_set,cluster_centers_neg

def train(path):
    if os.path.exists(path+'/log_CIS/log_CIS.txt'):
        print('Remove log_CIS.txt')
        os.remove(path+'/log_CIS/log_CIS.txt') 
    elif os.path.exists(path+'/log_CIS'):
        pass
    else:
        os.mkdir(path+'log_CIS')
    train_data_all,test_data_all=rd.read_data(path)
    cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count=rd.construct_cv_trainloader(train_data_all,test_data_all)
    
    torch.manual_seed(0)
    print(attr_count)
    net=Net(attr_count)
    cv_train_au(net,cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count,path)
if __name__=='__main__':
    path='../abalone_3_vs_11_5_fold/'
    train(path)
