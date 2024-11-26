
import torch 
from torch.nn import Linear, Conv1d, Linear, MaxPool2d
import torch.nn.functional as F

#from torch_geometric.utils import scatter
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GATv2Conv
from torch_geometric.utils import dropout_edge


#TODO Tannenbaum-Arch.  // Heads (concat bis zum letzten layer, dann MW)
#TODO Andere Layer-Arten (ggf. ohne Kanten mit nur eigenen Verbindungen)
#TODO nur ein GATConv und 1-2 lineare

class myGNN(torch.nn.Module):
    def __init__(self, first_layer, second_layer, third_layer, forth_layer, agg, num_neurons1, num_neurons2, num_neurons3, 
                 active_layers = 2, seed = None, num_atributes=57, edge_drop_prob=0.0):
        super().__init__()
        if seed:
            torch.manual_seed(seed)
        self.num_atributes = num_atributes
        self.first_layer = first_layer
        self.second_layer = second_layer
        self.third_layer = third_layer
        self.forth_layer = forth_layer
        
        self.active_layers = active_layers
        self.num_neurons1 = num_neurons1
        self.num_neurons2 = num_neurons2
        self.num_neurons3 = num_neurons3
        self.agg = agg
        num_classes = 1
        self.heads= 1
        
        self.edge_drop_prob = edge_drop_prob
        
        if self.first_layer == "GAT":
            self.layer1 = GATConv(num_atributes, self.num_neurons1, heads=self.heads)
        if self.first_layer == "SAGE":
            self.layer1 = SAGEConv(num_atributes, self.num_neurons1, aggr=self.agg)
        if self.first_layer == "LIN":
            self.layer1 = Linear(num_atributes, self.num_neurons1)
        if self.second_layer == "GAT":
            if self.first_layer == "GAT":
                self.layer1 = GATConv(num_atributes, self.num_neurons1*self.heads, heads=self.heads)
            if self.first_layer == "SAGE":
                self.layer1 = SAGEConv(num_atributes, self.num_neurons1*self.heads, aggr=self.agg)
            if self.first_layer == "LIN":
                self.layer1 = Linear(num_atributes, self.num_neurons1*self.heads)
        if self.second_layer == "GAT":
            self.layer2 = GATConv(self.num_neurons1*self.heads, self.num_neurons2, heads=self.heads)
        if self.second_layer == "SAGE":
            self.layer2 = SAGEConv(self.num_neurons1*self.heads, self.num_neurons2, aggr=self.agg)
        if self.second_layer == "LIN":
            self.layer2 = Linear(self.num_neurons1*self.heads, self.num_neurons2)
        if self.third_layer == "GAT":
            if self.second_layer == "GAT":
                self.layer2 = GATConv(self.num_neurons1*self.heads, self.num_neurons2*self.heads, heads=self.heads)
            if self.second_layer == "SAGE":
                self.layer2 = SAGEConv(self.num_neurons1*self.heads, self.num_neurons2*self.heads, aggr=self.agg)
            if self.second_layer == "LIN":
                self.layer2 = Linear(self.num_neurons1*self.heads, self.num_neurons2*self.heads)
        if self.active_layers == 2:
            if self.second_layer == "GAT":
                self.layer2 = GATConv(self.num_neurons1*self.heads, num_classes, heads=self.heads)
            if self.second_layer == "SAGE":
                self.layer2 = SAGEConv(self.num_neurons1*self.heads, num_classes, aggr=self.agg)
            if self.second_layer == "LIN":
                self.layer2 = Linear(self.num_neurons1*self.heads, num_classes)
        if self.active_layers > 2:     
            if self.third_layer == "GAT":
                self.layer3 = GATConv(self.num_neurons2*self.heads, self.num_neurons3, heads=self.heads)
            if self.third_layer == "LIN":
                self.layer3 = Linear(self.num_neurons2*self.heads, self.num_neurons3)
            if self.third_layer == "SAGE":
                self.layer3 = SAGEConv(self.num_neurons2*self.heads, self.num_neurons3, aggr=self.agg)
            if self.forth_layer == "GAT":
                if self.third_layer == "GAT":
                    self.layer3 = GATConv(self.num_neurons2*self.heads, self.num_neurons3*self.heads, heads=self.heads)
                if self.third_layer == "LIN":
                    self.layer3 = Linear(self.num_neurons2*self.heads, self.num_neurons3*self.heads)
                if self.third_layer == "SAGE":
                    self.layer3 = SAGEConv(self.num_neurons2*self.heads, self.num_neurons3*self.heads, aggr=self.agg)
            if self.active_layers == 3:
                if self.third_layer == "GAT":
                    self.layer3 = GATConv(self.num_neurons2*self.heads, num_classes, heads=self.heads)
                if self.third_layer == "LIN":
                    self.layer3 = Linear(self.num_neurons2*self.heads, num_classes)
                if self.third_layer == "SAGE":
                    self.layer3 = SAGEConv(self.num_neurons2*self.heads, num_classes, aggr=self.agg)
            if self.active_layers > 3:
                if self.forth_layer == "GAT":
                    self.layer4 = GATConv(self.num_neurons3*self.heads, num_classes, heads=self.heads)
                if self.forth_layer == "LIN":
                    self.layer4 = Linear(self.num_neurons3*self.heads, num_classes)
                if self.forth_layer == "SAGE":
                    self.layer4 = SAGEConv(self.num_neurons3*self.heads, num_classes, aggr=self.agg)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #print(edge_index)
        if self.edge_drop_prob > 0:
            #edge_index, _ = dropout_adj(edge_index, p=self.edge_drop_prob)
            edge_index, _ = dropout_edge(data.edge_index, p=self.edge_drop_prob, force_undirected=True)
        
        if self.first_layer == "GAT":
            x = self.layer1(x, edge_index)
        if self.first_layer == "SAGE":
            x = self.layer1(x, edge_index)
        if self.first_layer == "LIN":
            x = self.layer1(x)
        x = F.relu(x)
        
        if self.second_layer == "GAT":
            x = self.layer2(x, edge_index)
        if self.second_layer == "SAGE":
            x = self.layer2(x, edge_index)
        if self.second_layer == "LIN":
            x = self.layer2(x)
            
        if self.active_layers > 2:
            x = F.relu(x)
            if self.third_layer == "SAGE":
                x = self.layer3(x, edge_index)
            if self.third_layer == "GAT":
                x = self.layer3(x, edge_index)
            if self.third_layer == "LIN":
                x = self.layer3(x)
            
            if self.active_layers > 3:
                x = F.relu(x)
                if self.forth_layer == "SAGE":
                    x = self.layer4(x, edge_index)
                if self.forth_layer == "GAT":
                    x = self.layer4(x, edge_index)
                if self.forth_layer == "LIN":
                    x = self.layer4(x)
        return torch.sigmoid(x)

# %%


class myBaselineCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_atributes = 57
        num_neurons = 128
        num_classes = 1
        
        self.lin1 = Linear(num_atributes, num_neurons)
        self.lin2 = Linear(num_neurons, num_neurons)
        self.lin3 = Linear(num_neurons, num_classes)
    
    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return torch.sigmoid(x)
    
    
class MetaNN(torch.nn.Module):
    def __init__(self):
        super(MetaNN, self).__init__()
        num_atributes = 57
        self.act = torch.nn.ReLU()
        self.first_layer = "LIN"
        self.second_layer = "LIN"
        self.third_layer = "LIN"
        self.forth_layer = "LIN"
        self.fifth_layer = "LIN"
        self.sixs_layer = "LIN"
        self.num_neurons1 = 50
        self.num_neurons2 = 40
        self.num_neurons3 = 30
        self.num_neurons4 = 20
        self.num_neurons5 = 10
        self.layers = torch.nn.Sequential(torch.nn.Linear(num_atributes, self.num_neurons1),
                                    self.act,
                                    torch.nn.Linear(self.num_neurons1, self.num_neurons2),
                                    self.act,
                                    torch.nn.Linear(self.num_neurons2, self.num_neurons3),
                                    self.act,
                                    torch.nn.Linear(self.num_neurons3, self.num_neurons4),
                                    self.act,
                                    torch.nn.Linear(self.num_neurons4, self.num_neurons5),
                                    self.act,
                                    torch.nn.Linear(self.num_neurons5, 1))

    def forward(self, x):
        return self.layers(x).view(x.shape[0], -1)
