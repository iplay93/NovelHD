
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
from ..data_preprocessing.dataloader import count_label_labellist

import torch
import matplotlib.pyplot as plt
import pandas as pd

def tsne_visualization(data_list, label_list, dataset, dim):

    label_list = torch.tensor(label_list).tolist()
    num_classes, _ = count_label_labellist(label_list)
    
    #print('Shaped x', data_list.shape) #Shaped x torch.Size([676, 87, 7])
    #print(label_list) #[1,1,1,...]
    #print(type(data_list)) #Reshaped x torch.Size([676, 609])
    if(dim>=3):
        x_rs = np.reshape(data_list, [data_list.shape[0], data_list.shape[1]*data_list.shape[2]])
    else:
        label_list = (np.array(label_list)+1).tolist()
        x_rs = data_list.detach().numpy() 
    print('Reshaped x', x_rs.shape)
    print(type(x_rs))

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(x_rs)
    df = pd.DataFrame()
    df["y"] = label_list    
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    splot  = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls",len(num_classes)), data=df).set(title= dataset+" data T-SNE projection")
    plt.savefig('figure/'+ dataset +'-tsne.png', dpi=300)
    # clear the plot 
    plt.cla() 