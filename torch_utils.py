import torch

def rev_dist(a):
    if a==0:
        return 0
    else:
        return 1/a

def get_tensor_from_graph(graph):
    i = graph["A"]
    v_in = [rev_dist(e) for e in graph["edges_feature"]]
    v_true = graph["true_edges"]
    x = graph["nodes_feature"]
    N = len(x)
    
    X = torch.tensor(data=x, dtype=torch.float32)
    sp_A = torch.sparse_coo_tensor(indices=i, values=v_in, size=(N, N), dtype=torch.float32)
    E_true = torch.tensor(data=v_true, dtype=torch.float32)
    return X, sp_A, E_true, i