from torch_geometric.loader import DataLoader
from torch_geometric.data import Data 
import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GATConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from dataset import GraphDataset , get_big_matrix
from model_gnn import myGNN
from torch_geometric.loader import DataLoader
from torchmetrics.functional import r2_score
from graph_stuff.train_output import print_status, set_title, plot_sd, plot_acc, print_hyperpara

from ConfigSpace import Configuration, ConfigurationSpace, Categorical, EqualsCondition, Float, InCondition, Integer
from smac import Scenario, HyperparameterOptimizationFacade, MultiFidelityFacade
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving
import numpy as np
import os
import json
from tqdm import tqdm

#Settings
meta_data_source = 'DV3+_RN18_on_cityscapes_mIoU46'

with_edges = True
with_edge_weights = False
load_meanstd = False
calc_meanstd = False
save_meanstd = False
epochen = 100

num_atributes = 57
edge_drop_prob = 0.0

tune=False
#Ursprungsdatensatz
#dataset = GraphDataset(root='graph_stuff/graphs/')
if meta_data_source == 'DV3+_WRN38_on_cityscapes':
    #Normalisierter Datensatz
    dataset_original = GraphDataset(root='graph_stuff/graphs/')
    #if with_edges:
    #    if with_edge_weights:
    #        dataset = GraphDataset(root='graph_stuff/graphs/processed/normalized_edge-weight/')
    #    elif not with_edge_weights:
    #        dataset = GraphDataset(root='graph_stuff/graphs/processed/normalized/')
    #elif with_edges == False:
    #    dataset = GraphDataset(root='graph_stuff/graphs/processed/normalized_onlyselfconected/')

    dataset = GraphDataset(root='graph_stuff/graphs/processed/normalized/')
    #print("x: " + str(dataset[0].x))    
    #print("edges: " + str(dataset[0].edge_index))    
    #print("y: " + str(dataset[0].y)) 
elif meta_data_source == 'DV3+_RN18_on_cityscapes_mIoU46':
    dataset_original = GraphDataset(root='DV3+_RN18_on_cityscapes_mIoU46/graphs/yType_iou_norm_False_nAttr57/')
    dataset = GraphDataset(root='DV3+_RN18_on_cityscapes_mIoU46/graphs/yType_iou_norm_True_nAttr57/')


if load_meanstd:
    mean_std = torch.load('graph_stuff/graphs/mean_std.pt')
    mean_std_y = torch.load('graph_stuff/graphs/mean_std_y.pt')

#Mean & STD über ganzen Datensatz oder getrennt für Train und Test
if calc_meanstd:
    big_matrix_x = get_big_matrix(dataset_original, num_atributes)
    std, mean = torch.std_mean(big_matrix_x, dim=0, keepdim=True)


#Daten für normalisierung
if save_meanstd:
    torch.save([mean, std], 'DV3+_RN18_on_cityscapes_mIoU46/graphs/mean_std.pt')
    print("Saved")

#Set Train and Test Data
train_dataset = dataset[len(dataset) // 5:]
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_dataset = dataset[: len(dataset)// 5]
test_loader = DataLoader(test_dataset, batch_size=10) 

'''
print(train_dataset[0])
train_dataset_test = train_dataset[0]
train_dataset_test  = train_dataset_test.sort()
print("x: " + str(train_dataset[0].x))    
print("edges: " + str(train_dataset[0].edge_index))    
print("y: " + str(train_dataset[0].y)) 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = myGNN().to(device).double()
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
#, weight_decay=5e-4
'''


def train(model, data, optimizer,device):
    model.train()
    out_concat = torch.empty((0), dtype=torch.double)
    out_concat = out_concat.to(device)
    y_concat = torch.empty((0), dtype=torch.double)
    y_concat = y_concat.to(device)
    
    for data in train_loader:
        #reduce the number of features in an online manner
        data.x=data.x[:,:num_atributes]
        #remove edges in an online manner
        if not with_edges:
            data.edge_index=torch.empty((2, 0), dtype=torch.long)
        
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        out = torch.reshape(out, (-1,)) 
        loss = F.mse_loss(out, data.y)  
        loss.backward()
        optimizer.step()
        
        out_concat = torch.cat((out_concat, out), 0)
        y_concat = torch.cat((y_concat, data.y), 0)
        
    r2metric = r2_score(out_concat, y_concat)    
    return r2metric, out, data.y  



#%%
@torch.no_grad()
def test(loader, model, data, device):
    model.eval()
    out_concat = torch.empty((0), dtype=torch.double)
    out_concat = out_concat.to(device)
    y_concat = torch.empty((0), dtype=torch.double)
    y_concat = y_concat.to(device)
    
    for data in loader:
        #reduce the number of features in an online manner
        data.x=data.x[:,:num_atributes]
        #remove edges in an online manner
        if not with_edges:
            data.edge_index=torch.empty((2, 0), dtype=torch.long)
        
        data = data.to(device)
        out = model(data)
        out = torch.reshape(out, (-1,))   
        mse = F.mse_loss(out, data.y)
        
        out_concat = torch.cat((out_concat, out), 0)
        y_concat = torch.cat((y_concat, data.y), 0)

           
    r2metric = r2_score(out_concat, y_concat) 
    return r2metric , mse.item() , out, data.y 




def train_model(agg, first_layer, second_layer, third_layer, forth_layer, num_neurons1, num_neurons2, num_neurons3, lr, epochs, 
                active_layers=4, edge_drop_prob=0.0, workdir=None):
    #torch.manual_seed(0)
    #int werde für seed 100er oder 1000er schritte
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = myGNN(first_layer, second_layer, third_layer, forth_layer, agg, num_neurons1, num_neurons2,
                            num_neurons3, active_layers=active_layers,num_atributes=num_atributes, edge_drop_prob=edge_drop_prob).to(device).double()
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #weight_decay=5e-4
    
    #Track Metric and MSE
    y_val_train = []
    y_val_test = []
    mse_y_train = []
    mse_y_test = []
    mse_x = []
    train_max = -10000000#0.0
    test_max = -10000000#0.0
    out_max = 0.0
    y_max = 0.0
    for epoch in tqdm(range(1, epochs)):
    
        loss = train(model, data, optimizer, device)
        train_acc, mse_train, no1, no2 = test(train_loader, model, data, device)
        test_acc, mse_test, out_last, y_last = test(test_loader, model, data, device)
        
        mse_y_train.append(mse_train)
        mse_y_test.append(mse_test)
        mse_x.append(epoch)
        y_val_train.append( train_acc.item())
        y_val_test.append(test_acc.item())

        if train_acc.item() > train_max:
            train_max = train_acc.item()
        if test_acc.item() >= test_max:
            test_max = test_acc.item()
            out_max = out_last
            y_max = y_last
            
            if workdir != None:
                os.makedirs(workdir, exist_ok=True)
                torch.save(model.state_dict(),os.path.join(workdir,'model_state_dict_maxaurox.pth'))
                test_params = {'max_test_r2':float(test_acc.item()), 'epoch_max_auroc':int(epoch), 'train_2_to_max_test_r2':train_acc.item()}
                with open(os.path.join(workdir,'testparams_maxauroc.json'), 'w') as jf:
                    json.dump(test_params, jf, indent=4)
        
        
        #if epoch % 10 == 0:
            #print_status(epoch, train_acc, test_acc, train_max, test_max, mse_train)
    print_hyperpara(agg, first_layer, second_layer, third_layer, forth_layer, num_neurons1, num_neurons2, num_neurons3, lr, epochs, train_max, test_max)

    title = set_title(model, with_edges, with_edge_weights)
        
    #Plot
    #plot_sd(out_max, y_max, title)
    #plot_acc(mse_x, y_val_train, y_val_test, mse_y_train, mse_y_test, title)
    
    return test_max


class Demo:
    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges.
        # To illustrate different parameter types, we use continuous, integer and categorical parameters.
        cs = ConfigurationSpace()

        #active_layers = Integer("active_layers", (2, 4))
        num_neurons1 = Integer("num_neurons1", (10, 400))
        num_neurons2 = Integer("num_neurons2", (10, 400))
        num_neurons3 = Integer("num_neurons3", (10, 200))
        #heads = Integer("heads", (1, 3))
        
        agg = Categorical("agg", ["mean","max"], default="mean")
        
        first_layer = Categorical("first_layer", ["LIN", "SAGE","GAT"], default="GAT")
        second_layer = Categorical("second_layer", ["LIN", "SAGE","GAT"], default="GAT")
        third_layer = Categorical("third_layer", ["LIN", "SAGE","GAT"], default="GAT")
        forth_layer = Categorical("forth_layer", ["LIN", "SAGE","GAT"], default="GAT")
        lr= Float("lr", (0.001, 0.2))
        

        # Add all hyperparameters at once:
        

        cs.add_hyperparameters([agg, num_neurons1, num_neurons2, num_neurons3, first_layer, second_layer, third_layer, forth_layer, lr])
        
        return cs
    
    def train(self, config: Configuration, seed: int = 0, budget: int = 25) -> float:
        
        epochs=int(np.ceil(budget))
  
        result=train_model(**config, epochs=epochs)
        return 1 - result  #da SMAC minimiert

if tune:
    print("Start")
    demo = Demo()
    facades: list[AbstractFacade] = []
    for intensifier_object in [Hyperband]:
        # Define our environment variables
        scenario = Scenario(
            demo.configspace,
            walltime_limit=3600*8, 
            n_trials=500,  # Evaluate max 500 different trials
            min_budget=50,  # Train the MLP using a hyperparameter configuration for at least 10 epochs
            max_budget=200,  # Train the MLP using a hyperparameter configuration for at most 200 epochs
            n_workers=1,
            objectives="quality"
        )

        # We want to run five random configurations before starting the optimization.
        initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs=5)

        # Create our intensifier
        intensifier = intensifier_object(scenario, incumbent_selection="any_budget",eta=2)

        # Create our SMAC object and pass the scenario and the train method
        smac = MultiFidelityFacade(
            scenario,
            demo.train,
            initial_design=initial_design,
            intensifier=intensifier,
            overwrite=True,
        )

        # Let's optimize
        incumbent = smac.optimize()
        print("Incumbent",incumbent)
        print("Training Incumbent")
        print("Incumbent Cost:", smac.validate(incumbent))
else:
    #room for probe trainings
    agg = "mean"
    
    #LL
    #first_layer, second_layer, third_layer, forth_layer = "LIN", "LIN", "GAT", "LIN"
    #num_neurons1, num_neurons2, num_neurons3 = 317, 53, 127
    #lr, epochs = 0.0011498124235929928, 200
    #LS
    #first_layer, second_layer, third_layer, forth_layer = "LIN", "SAGE", "LIN", "GAT"
    #num_neurons1, num_neurons2, num_neurons3 = 314, 53, 129
    #lr, epochs = 0.0011665284259129763, 200
    #LLSL
    #first_layer, second_layer, third_layer, forth_layer = "LIN", "LIN", "SAGE", "LIN"
    #num_neurons1, num_neurons2, num_neurons3 = 337, 292, 99
    #lr, epochs = 0.002908610393805402, 200
    #LLS
    #first_layer, second_layer, third_layer, forth_layer = "LIN", "LIN", "SAGE", "LIN"
    #num_neurons1, num_neurons2, num_neurons3 = 145, 103, 94
    #lr, epochs = 0.00623782639669276, 200    
    #LSS
    #first_layer, second_layer, third_layer, forth_layer = "LIN", "SAGE", "SAGE", "LIN"
    #num_neurons1, num_neurons2, num_neurons3 = 142, 137, 10
    #lr, epochs = 0.008798370658609816, 200 
    
    #GLG
    #first_layer, second_layer, third_layer, forth_layer = "GAT", "LIN", "GAT", "GAT"
    #num_neurons1, num_neurons2, num_neurons3 = 145, 135, 95
    #lr, epochs = 0.0074647240582723755, 200 
    
    #LGS
    first_layer, second_layer, third_layer, forth_layer = "LIN", "GAT", "SAGE", "LIN"
    num_neurons1, num_neurons2, num_neurons3 = 145, 139, 10
    lr, epochs = 0.0019205353926687696, 200 
    
    for i in range(1,21):
        train_model(agg, first_layer, second_layer, third_layer, forth_layer, num_neurons1, num_neurons2, num_neurons3, lr, epochs, active_layers=3,
            edge_drop_prob=edge_drop_prob, 
            workdir=f'DV3+_RN18_on_cityscapes_mIoU46/runs/gnns/IoU/LIN_GAT_SAGE_WithEdges_EDrop_{edge_drop_prob}_nAt_{num_atributes}/{i}')
            #workdir=f'DV3+_RN18_on_cityscapes_mIoU46/runs/gnns/IoU/LIN_LIN_nAt_{num_atributes}/{i}')
