import hydra
import os
from os.path import join
import shutil
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
import torch_geometric
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pygame as pg
from graph_stuff.compute_graph import set_note_features, set_edge_list, show_graph_with_position, set_edge_list_selfconnected
from torchvision import transforms
from omegaconf import DictConfig
from pathlib import Path
from sklearn.model_selection import train_test_split
from joblib import dump
from metaseg.utils import get_dataset, get_metaseg_data_per_image
from metaseg.compute import prepare_meta_training_data
from metaseg.train_utils import kfold_validation, train_full, validate
from metaseg.statistics_and_plots import scatterplot, confusion_matr, roc_fromclassifier, roc_fromprobs, feature_importances
import sys
from torch_geometric.utils import sort_edge_index
#np.set_printoptions(threshold=sys.maxsize)


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    dataset_name = cfg.dataset
    print(f"Start fitting {cfg.meta_model} meta model on {dataset_name}!")
    metaseg_data_directory = Path(cfg.save_roots.metaseg_data)
    
    """set up plot directory and dump cfg there"""
    statistics_directory = join(cfg.save_roots.stats_and_plots,cfg.meta_model)
    if os.path.exists(statistics_directory):
        shutil.rmtree(statistics_directory)
    os.makedirs(statistics_directory)
    with open(join(statistics_directory,'config.yaml'), 'w') as f: 
        OmegaConf.save(cfg, f)

    """load metaseg data for the given dataset"""

    dataset = get_dataset(cfg[dataset_name])
    metaseg_data = get_metaseg_data_per_image(dataset, metaseg_data_directory, cfg.worker)
    feature_names = [f for f in list(metaseg_data[0]["metrics"].keys()) if f not in ['class','iou','iou0']]
    
    """prepare data to fit meta model on"""
    x, y, mu, sigma = prepare_meta_training_data(metaseg_data, target_key=cfg.meta_target)
    
        
    
    #####################################################################################################
    print("Data to Graph:")
    #Parameter
    num_atributes = 57
    num_classes = 1
    plotten = False
    normale_kanten = True  # normaler Graph/ bei False nur Kanten zu sich selbst (Baseline)
    set_edge_weights = False
    save = True
    y_type = 'iou'
    normalize = True
    
    
    #Hier genutzt über Train Datensatz
    if normalize:
        mean_std = torch.load('DV3+_RN18_on_cityscapes_mIoU46/graphs/mean_std.pt')
    
    for pic_number in tqdm(range(len(metaseg_data))): # ersetze durch len(metaseg_data)
        #print(metaseg_data[pic_number].metrics.keys())
           # Erstelle Note-Features-Matrix
        
        
        #Erzeuge Torch-Graph
        if normale_kanten:
            
            x, num_segments, y = set_note_features(metaseg_data, num_atributes, pic_number, y_type=y_type) 
            e, index1, index2, degree = set_edge_list(metaseg_data, num_segments, pic_number)

            #degree added
            #for i in range(num_segments):
                #x[i][57] = degree[i]
            edge_index = np.array([index1, index2])
            edge_index = torch.tensor(edge_index)
            edge_index = sort_edge_index(edge_index, sort_by_row=False)
            if y_type == 'iou':
                save_dir = f'DV3+_RN18_on_cityscapes_mIoU46/graphs/yType_{y_type}_norm_{normalize}_nAttr{num_atributes}/processed/data_'
            elif y_type == 'iou0':
                save_dir = f'DV3+_RN18_on_cityscapes_mIoU46/graphs/yType_{y_type}_norm_{normalize}_nAttr{num_atributes}/processed/data_'          
            edge_weight = np.zeros(len(index1))
            for edges in index1:
                edge_weight[edges] = e[index1[edges]][index2[edges]]
            edge_weight = torch.tensor(edge_weight)
        else: #keine Kanten
            
            x, num_segments, y = set_note_features(metaseg_data, num_atributes, pic_number, y_type=y_type)
            
            #index1, index2 = set_edge_list_selfconnected(metaseg_data, num_atributes, pic_number)
            #edge_index = np.array([index1, index2])
            #edge_index = torch.tensor(edge_index)
            #edge_index = sort_edge_index(edge_index)
            
            index1, index2 = [],[]
            edge_index = np.array([index1, index2])
            edge_index = torch.tensor(edge_index,dtype=torch.int)
            edge_index = sort_edge_index(edge_index)
            
            if y_type == 'iou':
                save_dir = 'graph_stuff/graphs/processed/normalized_onlyselfconected/processed/data_'
            elif y_type == 'iou0':
                save_dir = f'graph_stuff/graphs/JustSelfconnected_yType_{y_type}_norm_{normalize}_nAttr{num_atributes}/processed/data_'
        if (plotten):
            labels_for_plot = x[:,1]
            show_graph_with_position(x, index1, index2, metaseg_data[pic_number].basename)                                     # Plotte Graph
        
        x = np.array(x)
        x = torch.tensor(x, dtype=torch.double)
        y = torch.tensor(y, dtype=torch.double)
        
        #Normalize
        if normalize:
            x = torch.sub(x, mean_std[0])
            x = torch.div(x, mean_std[1])
        
        
        if set_edge_weights:
            data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)  
            save_dir = 'graph_stuff/graphs/processed/normalized_edge-weight/processed/data_'
        else:
            data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)                  
        if save:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(data, save_dir + str(pic_number) + '.pt' )
        #print(str(pic_number + 1) + "/" + str(len(metaseg_data))) 
        
        
        #fur speichern mit namen 
        #torch.save(data, 'graph_stuff/graphs/torch_set1/' + str(metaseg_data[pic_number].basename) + '.pt' )   # Save Graph für spätere Nutzung
        
        #####print graph with nx
        #g = torch_geometric.utils.to_networkx(data, to_undirected=True)
        #nx.draw(g, with_labels=True, node_color=labels_for_plot, pos=nx.fruchterman_reingold_layout(g))
        #plt.show()
        
        
    #####################################################################################################





if __name__ == "__main__":

    main()

# %%
