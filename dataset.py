#%%
import os.path as osp
from os import listdir

import torch
from torch_geometric.data import Dataset, download_url




#%%
class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        datafiles = [f for f in listdir(self.processed_dir) if osp.isfile(osp.join(self.processed_dir, f))]
        return datafiles

    @property
    def processed_file_names(self):

        datafiles = [f for f in listdir(self.processed_dir) if osp.isfile(osp.join(self.processed_dir, f))]
        return datafiles

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))       
        return data


def get_big_matrix(dataset, num_atributes):
    train_len = 400
    big_matrix_x = torch.empty((0, num_atributes), dtype=torch.double)
    #big_matrix_y = torch.empty((0), dtype=torch.double)
    for graph in range(train_len):
        big_matrix_x = torch.cat((big_matrix_x, dataset[graph].x), 0)
        #big_matrix_y = torch.cat((big_matrix_y, dataset[graph].y), 0)
    return big_matrix_x #, big_matrix_y

'''
def normalize_self(dataset, std, mean, std_y, mean_y):
    for graph in range(len(dataset)):
        #print("N X: " +str(dataset[graph].x))

        for row in range(dataset[graph].x.size()[0]):
            for col in range(dataset[graph].x.size()[1]):
                dataset[graph].x[row][col] = ((dataset[graph].x[row][col] - mean[0][col]) / std[0][col])

                #torch.save(data, 'graph_stuff/graphs/normalized/data_' + str(pic_number) + '.pt' )
            dataset[graph].y[row] = (dataset[graph].y[row] - mean_y) / std_y
        #print("N X: " +str(dataset[graph].x))
        #print("N Y: " +str(dataset[graph].y))
        print("Graph #" +str(graph) + " done!")
    
    print("Finished")
    return dataset
''' 
