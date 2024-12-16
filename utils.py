import numpy as np
import tensorflow as tf

def value_matrix(v, index):
    if index[0] == index[1] or v == 0:
        return 0.0
    else: 
        return 1.0/v 

def get_need_model(graph):
    N = len(graph['nodes_feature'])#max(max(graph["A"][0])+1, max(graph["A"][1])+1)
    indices_A = np.transpose(graph["A"])
    
    v = np.array(graph['nodes_feature'], dtype=np.float32)
    max_ = np.max(v, axis=0)
    min_ = np.min(v, axis=0)
    delta_ = max_- min_
    
    # for i in range(len(v[0])):
    for i in range(2):
        v[:, i] = (max_[i] - v[:, i])/delta_[i] if delta_[i] != 0 else v[:, i]
    H0 = tf.constant(v[:, :])
    A = tf.SparseTensor(indices=indices_A,
                    values=np.array([value_matrix(v, index) for v, index in zip(graph["edges_feature"], np.transpose(graph["A"]))], dtype=np.float32), 
                    dense_shape=[N, N])
    d = tf.sparse.reduce_sum(A, axis=1) + 1.0
    Mtrx = tf.SparseTensor(indices=indices_A,
                    values=np.array([val/d[index[0]] for index, val in zip(A.indices, A.values)], dtype=np.float32), 
                    dense_shape=[N, N])
    
    s1 = tf.SparseTensor(indices=[[i, e0] for i, e0 in enumerate(graph["A"][0])],
                    values=np.ones_like(graph["A"][0], dtype=np.float32), 
                    dense_shape=[len(graph["A"][0]), N])
    s2 = tf.SparseTensor(indices=[[i, e1] for i, e1 in enumerate(graph["A"][1])],
                    values=np.ones_like(graph["A"][1], dtype=np.float32), 
                    dense_shape=[len(graph["A"][1]), N])
    return Mtrx, H0, s1, s2
    

def get_Mtrxs(graph):
    Mtrx, H0, s1, s2 = get_need_model(graph)
    true_edges = tf.transpose(tf.constant([graph["true_edges"]], dtype=tf.float32))
    return Mtrx, H0, s1, s2, true_edges


def classification_edges(model, graph, k=0.51):
    Mtrx, H0, s1, s2 = get_need_model(graph)
    h1 =  model(Mtrx, H0, s1, s2)
    a = np.zeros_like(h1)
    a[h1>k] = 1.0
    return a
