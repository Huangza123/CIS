# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:31:19 2021
@author: Zhan ao Huang
@email: huangzhan_ao@outlook.com
"""
import numpy as np
import os
import torch.utils.data as Data
import torch

def read_data(path):
    train_data=[[],[],[],[],[]]
    test_data=[[],[],[],[],[]]
    #attr_params=['M','F','I','negative','positive']
    #attr_params_dict={'M':'0.1','F':'0.3','I':'0.5','negative':'0','positive':'1'}
    
    for name in os.listdir(path):
        if 'abalone' in name:
            #train file
            fold=int(name[-8])
            if 'tra' in name:    
                f=open(path+name)
                line=f.readline()
                while line:
                    if '@data' in line:
                        line=f.readline()
                        while line:
                            line=line.replace('M','0.1')
                            line=line.replace('F','0.3')
                            line=line.replace('I','0.5')
                            line=line.replace('negative','0')
                            line=line.replace('positive','1')
                            train_data[fold-1].append(line.strip('\n').split(','))
                            line=f.readline()
                    line=f.readline()
                f.close()
            elif 'tst' in name:
                f=open(path+name)
                line=f.readline()
                while line:
                    if '@data' in line:
                        line=f.readline()
                        while line:
                            line=line.replace('M','0.1')
                            line=line.replace('F','0.3')
                            line=line.replace('I','0.5')
                            line=line.replace('negative','0')
                            line=line.replace('positive','1')
                            test_data[fold-1].append(line.strip('\n').split(','))
                            line=f.readline()
                    line=f.readline()
                f.close()
    for i in range(5):
        for j in range(len(train_data[i])):
            for k in range(len(train_data[i][j])):
                train_data[i][j][k]=float(train_data[i][j][k])
    for i in range(5):
        for j in range(len(test_data[i])):
            for k in range(len(test_data[i][j])):
                test_data[i][j][k]=float(test_data[i][j][k])
    return train_data,test_data
def construct_cv_trainloader(train_data_all,test_data_all):
    cv_trainloader=[]
    cv_testloader=[]
    cv_pos_count,cv_neg_count=[],[]
    for i in range(5):
        train_data,train_label=[],[]
        test_data,test_label=[],[]
        pos_count,neg_count=0,0
        for j in range(len(train_data_all[i])):
            train_label.append(train_data_all[i][j][-1])
            train_data.append(train_data_all[i][j][0:-1])
            if train_data_all[i][j][-1]==1:
                pos_count+=1
            else:
                neg_count+=1
        for k in range(len(test_data_all[i])):
            test_data.append(test_data_all[i][k][0:-1])
            test_label.append(test_data_all[i][k][-1])
        cv_pos_count.append(pos_count)
        cv_neg_count.append(neg_count)
        #trainloader
        tensor_train_data=torch.from_numpy(np.array(train_data).reshape(-1,len(train_data[0]))).float()
        mean=torch.mean(tensor_train_data)
        std=torch.std(tensor_train_data)
        tensor_train_data=(tensor_train_data-mean)/std
        tensor_train_label=torch.from_numpy(np.array(train_label).reshape(-1,1))
        torch_train_dataset=Data.TensorDataset(tensor_train_data,tensor_train_label)
        trainloader=Data.DataLoader(dataset=torch_train_dataset,
                                    batch_size=1,
                                    shuffle=True)
        #testloader
        tensor_test_data=torch.from_numpy(np.array(test_data).reshape(-1,len(test_data[0]))).float()
        tensor_test_data=(tensor_test_data-mean)/std
        tensor_test_label=torch.from_numpy(np.array(test_label).reshape(-1,1))
        torch_test_dataset=Data.TensorDataset(tensor_test_data,tensor_test_label)
        testloader=Data.DataLoader(dataset=torch_test_dataset,
                                    batch_size=1,
                                    shuffle=False)
        cv_trainloader.append(trainloader)
        cv_testloader.append(testloader)
    attr_count=len(test_data[0])
    #print(attr_count)
    print('pos count:',cv_pos_count,',neg count:',cv_neg_count)
    return cv_trainloader,cv_testloader,cv_pos_count,cv_neg_count,attr_count
if __name__=='__main__':
    train_data_all,test_data_all=read_data()
    cv_trainloader,cv_testloader=construct_cv_trainloader(train_data_all,test_data_all)
    #print(cv_trainloader,cv_testloader)