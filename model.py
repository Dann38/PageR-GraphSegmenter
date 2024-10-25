import tensorflow as tf

params = {
    "start_w": 0.01,
    "start_b": 0.1,
    "count_neuron_layer_1": 9,
    "count_neuron_layer_2": 27,
    "count_neuron_layer_end": 18,
}

def get_model(path=None):
    l1 = params["count_neuron_layer_1"]
    l2 = params["count_neuron_layer_2"]
    l3 = params["count_neuron_layer_end"]
    model = GraphSegmenter(l1, l2, l3)
    if path is None:
        return model
    load_model = tf.saved_model.load(path)
   
    model.conv1.W = load_model.conv1.W
    model.conv1.B = load_model.conv1.B
    model.conv2.W = load_model.conv2.W
    model.conv2.B = load_model.conv2.B
    return model


class MyEndLayer(tf.Module):
    def __call__(self, s1, s2, h): 
        left_ = tf.sparse.sparse_dense_matmul(s1, h)
        right_ = tf.sparse.sparse_dense_matmul(s2, h)
        # return tf.reduce_sum(left_norm * right_norm, axis=1)
        return 0.5*(1.0-tf.losses.cosine_similarity(left_ , right_)) 
        
        

class MyGraphConv(tf.Module):
    def __init__(self, input_size, outpu_size, activation_fun):
        self.W = tf.Variable(tf.random.normal(mean=params["start_w"], stddev=1.0, shape=[input_size, outpu_size]))
        self.B = tf.Variable(tf.random.normal(mean=params["start_b"], stddev=1.0, shape=[input_size, outpu_size]))
        self.activation = activation_fun

    def __call__(self, A, H0):
        H1 = self.activation(
            tf.matmul(tf.sparse.sparse_dense_matmul(A, H0), self.W) - tf.matmul(H0, self.B)
        )
        return H1

class GraphSegmenter(tf.Module):
    def __init__(self, l1, l2, l3):
        self.conv1 = MyGraphConv(l1, l2, tf.nn.relu)
        self.conv2 = MyGraphConv(l2, l3, tf.nn.relu)
        self.end_layer = MyEndLayer()

    @tf.function
    def __call__(self, A, H0, s1, s2):
        H1 = self.conv1(A, H0)
        H2 = self.conv2(A, H1)
        return self.end_layer(s1, s2, H2)

    def save(self, path):
        tf.saved_model.save(self, path)

    
@tf.function        
def fun_loss(y_true, y_pred):
    return tf.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred)
    # return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=params["weight_loss"]))

@tf.function    
def my_loss(edges_pred, true_edges):
    return fun_loss(true_edges, edges_pred)
