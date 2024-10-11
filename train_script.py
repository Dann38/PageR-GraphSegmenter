from utils import get_Mtrxs, classification_edges
from model import get_model, my_loss
import tensorflow as tf
import json
import numpy as np
import argparse


def list_batchs(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size]

def train_one_step(model, batch, opt):
    my_loss_list = []
    count_matrix_var = len(model.trainable_variables)
    dw_array = [[] for _ in range(count_matrix_var)]
    for i, graph in enumerate(batch):
        try:
            A, H0, s1, s2, true_edges = get_Mtrxs(graph)
            with tf.GradientTape() as tape:
                H_end = model(A, H0, s1, s2)
                loss = my_loss(H_end, true_edges)
                my_loss_list.append(loss)
                print(f"{(i+1)/len(batch)*100:.2f} % loss = {loss.numpy():.5f} {' '*30}",  end="\r")
            dW = tape.gradient(loss, model.trainable_variables)
            for i, dw in enumerate(dW):
                dw_array[i].append(dw)
        except:
            pass
    dW = []
    for idw_array in dw_array:
        dW.append(tf.reduce_mean(idw_array, axis=0))
    opt.apply_gradients(zip(dW, model.trainable_variables))
    return np.mean(my_loss_list)


def train_model(params, model, dataset, path_save, save_frequency=5):  
    opt = tf.optimizers.SGD(learning_rate=params["learning_rate"])
    for i in range(params["epochs"]):
        my_loss_list = []
        print("="*10, f"EPOCH #{i+1}","="*10)
        for j, batch in enumerate(list_batchs(dataset, params["batch_size"])):
            loss_ = train_one_step(model, batch, opt)
            my_loss_list.append(loss_)
            print(f"Batch # {j} \t {loss_:.4f}", " "*40)
        with open('log.txt', 'a') as f:
            f.write(f"EPOCH #{i}\t {np.mean(my_loss_list)}\n")  
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
    model = get_model()
    train_model(params, model, dataset, args.name_model, args.fsave)