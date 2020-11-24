import numpy as np 
import torch
import pickle
def label2binlabel():
    bins = 5
    num_classes=1001
    #1000 means background
    
    data = np.zeros((bins, num_classes), dtype=np.int64)
    data[0]=0
    data[0,-1]=1
    data[1]=0
    data[1,:36] = np.arange(1, 36+1,dtype=np.int64).reshape(-1)
    data[2]=0
    data[2,36:36+164] = np.arange(1,164+1,dtype=np.int64).reshape(-1)
    data[3]=0
    data[3,36+164:36+164+330] = np.arange(1,330+1,dtype=np.int64).reshape(-1)
    data[4]=0
    data[4,36+164+330:36+164+330+470]=np.arange(1,470+1,dtype=np.int64).reshape(-1)
    data = torch.from_numpy(data)
    torch.save(data, "vg_label2binlabel.pt")
def pred_slice_with0():
    data=np.array([[0,2],
                   [2, 37],
                   [39,165],
                   [204,331],
                   [535,471]],dtype=np.int64)
    data = torch.from_numpy(data)
    torch.save(data, "vg_pred_slice_with0.pt")
def valsplit():
    data = {}
    data['(10000,~)'] = np.arange(0,36,dtype=np.int64).reshape(-1)
    data['(2000,10000)']=np.arange(36, 36+164,dtype=np.int64).reshape(-1)
    data['(500,2000)'] = np.arange(200,530,dtype=np.int64).reshape(-1)
    data['(0,500)']=np.arange(530, 1000,dtype=np.int64).reshape(-1)
    with open("vgsplit.pkl","wb") as f:
        pickle.dump(data, f)
valsplit()
pred_slice_with0()
label2binlabel()