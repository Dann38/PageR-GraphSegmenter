import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch.nn.functional import relu, sigmoid, binary_cross_entropy

class GNN(torch.nn.Module):
    def __init__(self,  layers):
        super(GNN, self).__init__()
        convs = []
        Bs = []
        for l_in, l_out in zip(layers[:-1], layers[1:]):
            convs.append(GCNConv(l_in, l_out, bias=False))
            torch.nn.init.normal_(convs[-1].lin.weight,mean=0.01, std=0.3)
            Bs.append(torch.nn.Linear(l_in, l_out, bias=False))
            torch.nn.init.normal_(Bs[-1].weight, mean=0.5, std=0.3)
        self.convs = torch.nn.ModuleList(convs)
        self.Bs = torch.nn.ModuleList(Bs)

    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, B in zip(self.convs, self.Bs):
            x = conv(x, edge_index) -  B(x)
            x = relu(x)
        return x

class EdgesMLP(torch.nn.Module):
    def __init__(self, layers):
        super(EdgesMLP, self).__init__()
        linears = []
        for l_in, l_out in zip(layers[:-1], layers[1:]):
            linears.append(Linear(l_in, l_out, bias=False))
            torch.nn.init.normal_(linears[-1].weight, mean=0.5, std=0.3)
        self.linears = linears

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear in self.linears:
            x = linear(x)
            x = sigmoid(x)
        return torch.squeeze(x, 1)

def get_models(params):
    layers_gnn = params["count_neuron_layers_gnn"]
    layers_edge = params["count_neuron_layers_edge"]
    node_gnn = GNN(layers_gnn)
    edge_linear = EdgesMLP(layers_edge)
    return node_gnn, edge_linear