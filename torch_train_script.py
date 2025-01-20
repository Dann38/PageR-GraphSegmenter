import torch
import argparse
from torch_model import get_models
from torch_utils import get_tensor_from_graph
import numpy as np
import json

START_ROW = f"{'='*10}LEANING{'='*10}\n"

def list_batchs(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size]

def validation(models, dataset, criterion):
    my_loss_list = []
    for j, graph in enumerate(dataset):
        X, sp_A, E_true, i = get_tensor_from_graph(graph)
        H_end = models[0](X, sp_A)
        Omega = torch.cat([H_end[i[0]], H_end[i[1]]],dim=1)
        E_pred = models[1](Omega)
        loss = criterion(E_pred, E_true)
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
        X, sp_A, E_true, i = get_tensor_from_graph(graph)
        H_end = models[0](X, sp_A)
        Omega = torch.cat([H_end[i[0]], H_end[i[1]]],dim=1)
        E_pred = models[1](Omega)
        loss = criterion(E_pred, E_true)
        my_loss_list.append(loss.item())
        print(f"Batch loss={my_loss_list[-1]:.4f}" + " "*40, end="\r")
        loss.backward()
    optimizer.step()
    return np.mean(my_loss_list)

def train_model(params, models, dataset, path_save, save_frequency=5):  
    optimizer = torch.optim.Adam(
    list(models[0].parameters()) + list(models[1].parameters()),
    lr=params["learning_rate"],
    )
    criterion = torch.nn.BCELoss()
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
            
            
        with open('log.txt', 'a') as f:
            f.write(f"EPOCH #{k}\t {train_val:.8f} (VAL: {validation_val:.8f})\n")  
        if (k+1) % save_frequency == 0:
            num = k//save_frequency
            torch.save(models[0].state_dict(), path_save+f"_node_gnn_{num}")
            torch.save(models[1].state_dict(), path_save+f"_edge_linear_{num}")
    torch.save(models[0].state_dict(), path_save+f"_node_gnn_end")
    torch.save(models[1].state_dict(), path_save+f"_edge_linear_end")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train dataset')
    parser.add_argument('--epochs', type=int, nargs='?', required=True)
    parser.add_argument('--batch_size', type=int, nargs='?', required=True) 
    parser.add_argument('--learning_rate', type=float, nargs='?', required=True) 
    parser.add_argument('--path_dataset', type=str, nargs='?', required=True)
    parser.add_argument('--name_model', type=str, nargs='?', required=True)
    parser.add_argument('--fsave', type=int, nargs='?', required=True)
    args = parser.parse_args()
    params = {
        "count_neuron_layers_gnn": [9, 27, 18],
        "count_neuron_layers_edge": [18*2, 1],
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }
    with open(args.path_dataset, "r") as f:
        dataset = json.load(f)['dataset']
    with open('log.txt', 'a') as f:
        for i, p in params.items():
            f.write(f'{i}:\t{p}\n')
        f.write(f'count_graphs:\t{len(dataset)}\n')
        f.write(START_ROW)
    print("DATASET INFO:")
    print("count row:", len(dataset))
    print("first:", dataset[0].keys())
    print(f"\t A:", np.shape(dataset[0]["A"]))
    print(f"\t nodes_feature:", np.shape(dataset[0]["nodes_feature"]))
    print(f"\t edges_feature:", np.shape(dataset[0]["edges_feature"]))
    print(f"\t true_edges:", np.shape(dataset[0]["true_edges"]))
    print("end:", dataset[-1].keys())
    print(f"\t A:", np.shape(dataset[-1]["A"]))
    print(f"\t nodes_feature:", np.shape(dataset[-1]["nodes_feature"]))
    print(f"\t edges_feature:", np.shape(dataset[-1]["edges_feature"]))
    print(f"\t true_edges:", np.shape(dataset[-1]["true_edges"]))
    
    models = get_models(params)
    train_model(params, models, dataset,  args.name_model, args.fsave)
