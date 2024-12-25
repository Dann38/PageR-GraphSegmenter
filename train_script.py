from utils import get_Mtrxs, classification_edges
from model import get_model, my_loss, params as model_params 
import tensorflow as tf
import json
import numpy as np
import argparse
CUDA_VISIBLE_DEVICES=""
def list_batchs(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size]

def split_train_val(dataset, val_split=0.2, shuffle=True, seed=1234):
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(dataset)
    train_size = int(len(dataset) * (1 - val_split))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    return train_dataset, val_dataset

def validation(model, dataset):
    my_loss_list = []
    for i, graph in enumerate(dataset):
        A, H0, s1, s2, true_edges = get_Mtrxs(graph)
        H_end = model(A, H0, s1, s2)
        loss = my_loss(H_end, true_edges)
        my_loss_list.append(loss[0].numpy())
        print(f"{(i+1)/len(dataset)*100:.2f} % loss = {loss[0].numpy():.5f} {' '*30}", end='\r')
    return np.mean(my_loss_list)

def train_one_step(model, batch, opt, loss_list):
    my_loss_list = []
    count_matrix_var = len(model.trainable_variables)
    dw_array = [[] for _ in range(count_matrix_var)]
    for i, graph in enumerate(batch):
        A, H0, s1, s2, true_edges = get_Mtrxs(graph)
        with tf.GradientTape() as tape:
            H_end = model(A, H0, s1, s2)
            loss = my_loss(H_end, true_edges)
        dW = tape.gradient(loss, model.trainable_variables)
        del tape
        my_loss_list.append(loss[0].numpy())
        print(f"{(i+1)/len(batch)*100:.2f} % loss = {loss[0].numpy():.5f} {' '*30}", end='\r')
 
        for i, dw in enumerate(dW):
            dw_array[i].append(dw)
    dW = []
    for idw_array in dw_array:
        dW.append(tf.reduce_mean(idw_array, axis=0))
    opt.apply_gradients(zip(dW, model.trainable_variables))
    loss_list.append(np.mean(my_loss_list))

def train_model(params, model, dataset, path_save, save_frequency=5):  
    opt = tf.optimizers.legacy.Adam(learning_rate=params["learning_rate"])
    train_dataset, val_dataset = split_train_val(dataset, val_split=0.1)
    for i in range(params["epochs"]):
        my_loss_list = []
        print("="*10, f"EPOCH #{i+1}","="*10)
        for j, batch in enumerate(list_batchs(train_dataset, params["batch_size"])):
            train_one_step(model, batch, opt, my_loss_list)
            print(f"\nBatch # {j+1} loss={my_loss_list[-1]:.4f}" + " "*40)
        train_val = np.mean(my_loss_list)
        validation_val = validation(model, val_dataset)
        with open('log.txt', 'a') as f:
            f.write(f"EPOCH #{i}\t {train_val:.4f} (VAL: {validation_val:.4f})\n")  
        if i % save_frequency == 0:
            model.save(path_save+f"_{i//save_frequency}")
    model.save(path_save)


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
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }
    with open(args.path_dataset, "r") as f:
        dataset = json.load(f)['dataset']
    with open('log.txt', 'a') as f:
        for i, p in params.items():
            f.write(f'{i}:\t{p}\n')
        for i, p in model_params.items():
            f.write(f'{i}:\t{p}\n')
        f.write(f'count_graphs:\t{len(dataset)}\n')
        f.write(f"{'='*10}LEANING{'='*10}\n")
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
    model = get_model(args.name_model)
    train_model(params, model, dataset, args.name_model, args.fsave)


