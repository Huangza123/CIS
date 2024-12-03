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
        train_dros(net,cv_trainloader[i],cv_testloader[i],cv_neg_count[i],cv_pos_count[i],attr_count,i+1,path)
        
        net=init_net

    
def train_dros(net,trainloader,testloader,cv_neg_count,cv_pos_count,attr_count,fold,path):
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
    
    new_pos_data=DROS(pos_data.numpy().reshape(-1,attr_count),neg_data.numpy().reshape(-1,attr_count))
   
    pos_data=torch.from_numpy(np.concatenate((pos_data,new_pos_data.reshape(-1,1,attr_count)),axis=0))
    pos_label=torch.ones(len(pos_data))
    neg_label=torch.zeros(len(neg_data))
    label=torch.cat((pos_label,neg_label),axis=0).long()
    data=torch.cat((pos_data,neg_data),axis=0).float()

    for epoch in range(EPOCH):
        output=net(data)
        loss=loss_func(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if epoch % update_center_duration==0:
            f=open(path+'/log_DROS/log_DROS.txt','a')
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
def compute_base_unit_vector(x_pos,neg_data,k=7):
    #compute the base unit vector
    # x_pos--minority class sample
    # neg_data--majority class dataset
    # k -- the number of nearest neightbor of x_pos
    dis=[]
    for i in range(len(neg_data)):
        dis.append(np.sqrt(np.sum((x_pos-neg_data[i])**2)))
    k_nearest_majority_class=[]
    for i in range(k):
        min_dis_idx=dis.index(min(dis))
        dis[min_dis_idx]=max(dis)
        k_nearest_majority_class.append(neg_data[min_dis_idx])

    center=np.array(k_nearest_majority_class).mean(axis=0)
    base_unit_vector=(x_pos-center)/np.sqrt(1e-6+np.sum((x_pos-center)**2)) #base_unit_vector=c, a=-c
    return base_unit_vector

def direct_interlinked(x0_pos,x1_pos,neg_data,delta=-0.7660):
    D_k=np.zeros(len(neg_data))
    
    for i in range(len(neg_data)):
        D_k[i]=np.dot((x0_pos-neg_data[i])/np.sqrt(1e-6+np.sum((x0_pos-neg_data[i])**2)),(x1_pos-neg_data[i])/np.sqrt(1e-6+np.sum((x1_pos-neg_data[i])**2)))
    M=min(D_k)
    if M>delta:
        return True
    else:
        return False
def compute_vertex(x_pos,pos_data,neg_data,c,delta=-0.7660):
    d=[]
    for i in range(len(pos_data)):
        if np.sum(x_pos-pos_data[i])!=0:
            if direct_interlinked(x_pos,pos_data[i],neg_data,delta):
                d.append(pos_data[i]-x_pos)
    if len(d)>0:
        p=np.zeros(len(d))
        for i in range(len(d)):
            if np.dot(d[i],c)>0: #J(p_i)
                p[i]=np.dot(d[i],c)
        vertex_vector=np.mean(p)*c+x_pos
        return vertex_vector
    else:
        return np.ones(2)*100
    
    
def compute_radius(x_pos,neg_data,vertex_vector_v,rho=0.5):
    illuminated_neg_data=[]  
    for i in range(len(neg_data)):
        if np.dot((neg_data[i]-vertex_vector_v)/np.sqrt(1e-6+np.sum((neg_data[i]-vertex_vector_v)**2)),(x_pos-vertex_vector_v)/np.sqrt(1e-6+np.sum((x_pos-vertex_vector_v)**2)))>=rho:
            illuminated_neg_data.append(neg_data[i])
    
    illuminated_neg_data=np.array(illuminated_neg_data).reshape(-1,len(x_pos))
    # plt.scatter(x_pos[0],x_pos[1],color='r')
    # plt.scatter(neg_data[:,0],neg_data[:,1],color='white',edgecolor='g')
    # plt.scatter(illuminated_neg_data[:,0],illuminated_neg_data[:,1],color='pink',marker='*')
    # plt.show()
    dis_illuminated=[]
    if len(illuminated_neg_data)>0:
        for i in range(len(illuminated_neg_data)):
            dis_illuminated.append(np.sqrt(1e-6+np.sum((x_pos-illuminated_neg_data[i])**2)))
        min_dis_idx=dis_illuminated.index(min(dis_illuminated))
        
        g=illuminated_neg_data[min_dis_idx]
        dis_v_x=np.sqrt(1e-6+np.sum((x_pos-vertex_vector_v)**2))
        dis_v_g=np.sqrt(1e-6+np.sum((g-vertex_vector_v)**2))
        r=min(dis_v_x,dis_v_g)+0.5*(dis_v_g-dis_v_x)
        return r
    else:
        return 0

def light_cone_data_generation(x_pos,radius,vertex_vector,rho=0.5):
    a=(vertex_vector-x_pos)/np.sqrt(np.sum((vertex_vector-x_pos)**2))
    random_vector=np.random.rand(len(a))
    random_unit_vector=random_vector/np.sqrt(1e-6+np.sum(random_vector**2))
    d=a+random_unit_vector*np.random.uniform(-1,1)
    d=d/np.sqrt(np.sum(d**2))
    while True:
        if np.dot(d,a)>=rho:
            break
        random_vector=np.random.rand(len(a))
        random_unit_vector=random_vector/np.sqrt(np.sum(random_vector**2))
        d=a+random_unit_vector*np.random.uniform(-1,1)
        d=d/np.sqrt(np.sum(d**2))
    g=np.random.rand()#hyper parameter for user-defined, g\in[0,1]
    xi=g+ np.random.rand()*(1-g)
    new_data=vertex_vector+xi*radius*d
    return new_data
def DROS(pos_data,neg_data):
    sampling_count=len(neg_data)-len(pos_data)
    new_pos_data=[]
    while True:
        x_pos=pos_data[np.random.randint(len(pos_data))]
        base_unit_vector_c=compute_base_unit_vector(x_pos,neg_data,7)
        vertex_vector_v=compute_vertex(x_pos,pos_data,neg_data,base_unit_vector_c,-0.776)
        if np.sum(vertex_vector_v)!=200:
            radius=compute_radius(x_pos,neg_data,vertex_vector_v,rho=0.5)
            if radius>0:
                new_data=light_cone_data_generation(x_pos,radius,vertex_vector_v,0.5)
                new_pos_data.append(new_data)
                sampling_count-=1
                print('sampling_count:%d,radius=%.3f'%(sampling_count,radius))
                if sampling_count==0:
                    break
    return np.array(new_pos_data).reshape(-1,pos_data.shape[1])
def train(path):
    if os.path.exists(path+'/log_DROS/log_DROS.txt'):
        print('Remove log_DROS.txt')
        os.remove(path+'/log_DROS/log_DROS.txt') 
    elif os.path.exists(path+'/log_DROS'):
        pass
    else:
        os.mkdir(path+'log_DROS')
    train_data_all,test_data_all=rd.read_data(path)
    cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count=rd.construct_cv_trainloader(train_data_all,test_data_all)
    
    torch.manual_seed(0)
    print(attr_count)
    net=Net(attr_count)
    cv_train_au(net,cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count,path)
if __name__=='__main__':
    path='../abalone_3_vs_11_5_fold/'
    train(path)
