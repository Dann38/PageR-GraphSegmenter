import torch
from torch.nn import Linear, BCELoss, CrossEntropyLoss
from torch.nn.functional import relu, sigmoid, binary_cross_entropy
from torch_geometric.nn import BatchNorm, TAGConv
import numpy as np
import json

class NodeGLAM(torch.nn.Module):
    def __init__(self,  input_, h, output_):
        super(NodeGLAM, self).__init__()
        self.batch_norm1 = BatchNorm(input_)
        self.linear1 = Linear(input_, h[0]) 
        self.tag1 = TAGConv(h[0], h[1])
        self.linear2 = Linear(h[1], h[2]) 
        self.tag2 = TAGConv(h[2], h[3])
        self.linear3 = Linear(h[3]+input_, h[4])
        self.linear4 =Linear(h[4], output_)

    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm1(x)
        h = self.linear1(x)
        h = relu(h)
        h = self.tag1(h, edge_index)
        h = relu(h)
        
        h = self.linear2(h)
        h = relu(h)
        h = self.tag2(h, edge_index)
        h = relu(h)
        a = torch.cat([x, h], dim=1)
        a = self.linear3(a)
        a = relu(a)
        a = self.linear4(a)
        return torch.softmax(a, dim=-1)

class EdgeGLAM(torch.nn.Module):
    def __init__(self, input_, h, output_):
        super(EdgeGLAM, self).__init__()
        self.batch_norm2 = BatchNorm(input_, output_)
        self.linear1 = Linear(input_, h[0]) 
        self.linear2 = Linear(h[0], output_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm2(x)
        h = self.linear1(x)
        h = relu(h)
        h = self.linear2(h)
        h = torch.sigmoid(h)
        return torch.squeeze(h, 1)

class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        
        self.bce = BCELoss()
        self.ce = CrossEntropyLoss()

    def forward(self, n_pred, n_true, e_pred, e_true):
        loss = self.ce(n_pred, n_true) + 4*self.bce(e_pred, e_true)
        return loss

def list_batchs(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size]

def get_tensor_from_graph(graph):
    def class_node(n):
        rez = [0, 0, 0, 0, 0]
        if n!= -1:
            rez[n] = 1
        return rez
    def rev_dist(a):
        if a==0:
            return 0
        else:
            return 1/a
        
    i = graph["A"]
    v_in = [rev_dist(e) for e in graph["edges_feature"]]
    v_true = graph["true_edges"]
    n_true = [class_node(n) for n in graph["true_nodes"]]
    x = graph["nodes_feature"]
    N = len(x)
    
    X = torch.tensor(data=x, dtype=torch.float32)
    sp_A = torch.sparse_coo_tensor(indices=i, values=v_in, size=(N, N), dtype=torch.float32)
    E_true = torch.tensor(data=v_true, dtype=torch.float32)
    N_true = torch.tensor(data=n_true, dtype=torch.float32)
    return X, sp_A, E_true, N_true, i

def validation(models, dataset, criterion):
    my_loss_list = []
    for j, graph in enumerate(dataset):
        X, sp_A, E_true, N_true, i = get_tensor_from_graph(graph)
        Node_emb = models[0](X, sp_A)
        Omega = torch.cat([Node_emb[i[0]],Node_emb[i[1]], X[i[0]], X[i[1]]],dim=1)
        E_pred = models[1](Omega)
        loss = criterion(Node_emb, N_true, E_pred, E_true)
        my_loss_list.append(loss.item())
        print(f"{(j+1)/len(dataset)*100:.2f} % loss = {my_loss_list[-1]:.5f} {' '*30}", end='\r')
    return np.mean(my_loss_list)

def split_train_val(dataset, val_split=0.2, shuffle=True, seed=1234):
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(dataset)
    train_size = int(len(dataset) * (1 - val_split))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    return train_dataset, val_dataset

def train_step(models, batch, optimizer, criterion):
    optimizer.zero_grad()
    my_loss_list = []
   
    for j, graph in enumerate(batch):
        X, sp_A, E_true, N_true, i = get_tensor_from_graph(graph)
        Node_emb = models[0](X, sp_A)
        Omega = torch.cat([Node_emb[i[0]],Node_emb[i[1]], X[i[0]], X[i[1]]],dim=1)
        E_pred = models[1](Omega)
        loss = criterion(Node_emb, N_true, E_pred, E_true)
        my_loss_list.append(loss.item())
        print(f"Batch loss={my_loss_list[-1]:.4f}" + " "*40, end="\r")
        loss.backward()
    optimizer.step()
    return np.mean(my_loss_list)

def train_model(params, models, dataset, path_save, save_frequency=5, restart=False):  
    optimizer = torch.optim.Adam(
    list(models[0].parameters()) + list(models[1].parameters()),
    lr=params["learning_rate"],
    )
    criterion = CustomLoss()
    loss_list = []
    with open('log.txt', 'a') as f:
        for key, val in params.items():
            f.write(f"{key}:\t{val}\n")
    train_dataset, val_dataset = split_train_val(dataset, val_split=0.1)
    for k in range(params["epochs"]):
        my_loss_list = []
        
        for l, batch in enumerate(list_batchs(train_dataset, params["batch_size"])):
            batch_loss = train_step(models, batch, optimizer, criterion)
            my_loss_list.append(batch_loss)
            print(f"Batch # {l+1} loss={my_loss_list[-1]:.4f}" + " "*40)
        train_val = np.mean(my_loss_list)
        loss_list.append(train_val)
        validation_val = validation(models, val_dataset, criterion)
        print("="*10, f"EPOCH #{k+1}","="*10, f"({train_val:.4f}/{validation_val:.4f})")
        
        # TODO: DELETE RESTART
        if restart and k>=2 and abs(loss_list[k] - loss_list[k-1]) < 0.001:
            return True
            
            
        with open('log.txt', 'a') as f:
            f.write(f"EPOCH #{k}\t {train_val:.8f} (VAL: {validation_val:.8f})\n")  
        if (k+1) % save_frequency == 0:
            num = k//save_frequency
            torch.save(models[0].state_dict(), path_save+f"_node_gnn_{num}")
            torch.save(models[1].state_dict(), path_save+f"_edge_linear_{num}")
    torch.save(models[0].state_dict(), path_save+f"_node_gnn_end")
    torch.save(models[1].state_dict(), path_save+f"_edge_linear_end")
    return False # For restart



if __name__ == "__main__":
    with open("rez.json", "r") as f:
        dataset = json.load(f)['dataset']

    dataset = [d for d in dataset if len(d["nodes_feature"]) > 1] # Удалить те графы где один node

    str_ = f"""DATASET INFO:
count row: {len(dataset)}
first: {dataset[0].keys()}
\t A:{np.shape(dataset[0]["A"])}
\t nodes_feature:{np.shape(dataset[0]["nodes_feature"])}
\t edges_feature:{np.shape(dataset[0]["edges_feature"])}
\t true_edges:{np.shape(dataset[0]["true_edges"])}
\t true_nodes:{np.shape(dataset[0]["true_nodes"])}
end:{dataset[-1].keys()}
\t A{np.shape(dataset[-1]["A"])}
\t nodes_feature:{np.shape(dataset[-1]["nodes_feature"])}
\t edges_feature:{np.shape(dataset[-1]["edges_feature"])}
\t true_edges:"{np.shape(dataset[-1]["true_edges"])}
\t true_nodes:{np.shape(dataset[-1]["true_nodes"])}

"""

    print(str_)
    with open('log.txt', 'a') as f:    
        f.write(str_)
    params = {
        "epochs": 100,
        "batch_size": 500,
        "learning_rate": 0.05,
        "H1": [64, 64, 64, 64, 64],
        "H2": [64]
    }
    node_glam = NodeGLAM(37, params["H1"], 5)
    edge_glam = EdgeGLAM(2*37+2*5, params["H2"],1)
    path_save = "glam"
    train_model(params, [node_glam, edge_glam], dataset, path_save, save_frequency=10, restart=False)