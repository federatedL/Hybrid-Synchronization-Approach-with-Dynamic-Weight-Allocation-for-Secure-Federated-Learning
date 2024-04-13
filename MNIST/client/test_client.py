import os
from google.cloud import storage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'cloud.json'

import math
import random
import subprocess
import pickle
import argparse
import tensorflow as tf
import numpy as np
from numpy.polynomial import polynomial as poly
import phe
import json
from phe import paillier
from sklearn import datasets
from sklearn.model_selection import train_test_split

from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

'''
Arguments:
 loss, metrics, (lr, decay, momentum): SGD, (shape, classes_count): SimpleMLP, global_weights, epochs, verbose, client_batched_data

 #Careful:
   - metrics can be an array

Return:
 gradients received from local_model
'''

PK = None
SK = None
p = 58727 # both the primes are 15 bits
q = 65063
n = None
g = None
lmbda = None
mu = None
client_no = 1
X_train = None
y_train = None
agg_gradients = None

bucket_name = 'fl-data'
storage_client = storage.Client()
SERVER_FILE = 'agg_gradients.pkl'
batch_size = 32
loss_object = SparseCategoricalCrossentropy()

n = 2**4
q = 2**511
t = 2**8
poly_mod = np.array([1] + [0] * (n - 1) + [1])

num_bits = 8  # Number of bits to represent each value
quantization_range = None
quant_B = 8
quant_lower = -2^quant_B
quant_upper = 2^(quant_B-1)-1

######################## ARG LIST ########################
loss = None
metrics = None
lr = None
decay = None
momentum = None
shape = None
classes_count = None
global_weights = None
epochs = None
verbose = None
client_batched_data = None
#########################################################

def upload_to_cloud(file_name):
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_name)
    except Exception as e:
        return

def download_from_cloud(file_name):
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)
        data = blob.download_as_string()
        res = pickle.loads(data)
        return res
    except Exception as e:
        return

def gen_uniform_poly(size, modulus):
    return np.random.uniform(0, modulus, size)

def gen_binary_poly(size):
    return np.random.randint(0, 2, size)

def gen_normal_poly(size):
    return np.random.normal(0, 2, size=size)

def polymul(x, y, modulus, poly_mod):
    return poly.polydiv(poly.polymul(x, y) % modulus, poly_mod)[1] % modulus

def polyadd(x, y, modulus, poly_mod):
    return poly.polydiv(poly.polyadd(x, y) % modulus, poly_mod)[1] % modulus

def DECRYPT(sk, size, q, t, poly_mod, ct):
    scaled_pt = polyadd(polymul(ct[1], sk, q, poly_mod), ct[0], q, poly_mod)
    decrypted_poly = scaled_pt * t / q % t
    return float(decrypted_poly[0])

def ENCRYPT(pk, size, q, t, poly_mod, pt, b_cap):
    m = np.array([pt] + [0] * (size - 1)) % t
    delta = q / t
    scaled_m = delta * m % q
    e1 = gen_normal_poly(size)
    e2 = gen_normal_poly(size)
    u = gen_binary_poly(size)
    ct0 = polyadd(polyadd(polymul(b_cap, u, q, poly_mod), e1, q, poly_mod), scaled_m, q, poly_mod)
    ct1 = polyadd(polymul(pk[1], u, q, poly_mod), e2, q, poly_mod)
    return (ct0, ct1)

def get_compiled_model():
    '''
        returns compiled keras model
    '''

    inputs = Input(shape=(784,), name="digits")
    x1 = Dense(64, activation="relu")(inputs)
    x2 = Dense(64, activation="relu")(x1)
    outputs = Dense(10, name="predictions")(x2)
    model = Model(inputs=inputs, outputs=outputs)

    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    optimizer = Adam(learning_rate=1e-3)

    return model, loss_fn, optimizer

def read_client_batched_data(bs=32):
    global client_batched_data
    client_batched_data_list = download_from_cloud(f'client_batched_data_{client_no}.pkl')
    # with open('client_batched_data_{}.pkl'.format(client_no), 'rb') as pickle_file:
    #     client_batched_data_list = pickle.load(pickle_file)

    # Convert list of numpy array tuples back to TensorFlow _BatchDataset
    client_batched_data = None
    for data, labels in client_batched_data_list:
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        if client_batched_data is None:
            client_batched_data = dataset
        else:
            client_batched_data = client_batched_data.concatenate(dataset)

def read_b_cap():
    b_cap = download_from_cloud(f'b_cap.pkl')
    # with open('b_cap.pkl', 'rb') as pf:
    #     b_cap = pickle.load(pf)

    return b_cap

def read_client_dataset():
    global X_train, y_train, X_test, y_test
    client_dataset = download_from_cloud(f'client_dataset_{client_no}.pkl')
    # with open('client_dataset_{}.pkl'.format(client_no), 'rb') as pickle_file:
    #     client_dataset = pickle.load(pickle_file)

    Xt, yt = zip(*client_dataset)
    X_train = np.array(Xt, dtype=np.float32)
    y_train = np.array(yt)

def read_paillier_keys():
    with open('paillier_keys.pkl', 'rb') as pk:
        paillier_keys = pickle.load(pk)

    n = paillier_keys['public_key']['n']
    g = paillier_keys['public_key']['g']
    p = paillier_keys['private_key']['p']
    q = paillier_keys['private_key']['q']

    PK = paillier.PaillierPublicKey(n=n)
    SK = paillier.PaillierPrivateKey(PK, p, q)

    return PK, SK

def read_creds():
    #creds = download_from_cloud(f'creds.pkl')
    with open('creds.pkl', 'rb') as c:
         creds = pickle.load(c)

    PK = creds['keys']['pk']
    SK = creds['keys']['sk']

    return PK, SK

def get_public_key():
    n = p*q
    g = n + 1

    return n, g

def get_private_key():
    phi = (p-1)*(q-1)
    lmbda = phi * 1
    mu = pow(lmbda, -1, n)

    return lmbda, mu

def read_public_key():
    with open('public_key.pkl', 'rb') as pk:
        public_key = pickle.load(pk)

    return public_key['n'], public_key['g']

def write_public_key(n, g):
    public_key = {'n': n, 'g': g}
    with open('public_key.pkl', 'wb') as pk:
        pickle.dump(public_key, pk)

    return

def lx(x):
    y = (x-1)/n
    # assert y - int(y) == 0
    return int(y)

def encrypt(m, r):
    # assert math.gcd(r, n) == 1
    m = int(m)
    c = ( pow(g, m, n*n) * pow(r, n, n*n) ) % (n*n)
    return c

def decrypt(c):
    c = int(c)
    p = ( lx(pow(c, lmbda, n*n)) * mu ) % n
    return p

def encrypt_gradients(grads):
    enc_grads = []
    for grad in grads:
        if len(grad.shape) == 2:
            a, b = grad.shape
            empty_arr = np.empty((a, b), dtype=np.float32)
            for i in range(len(grad)):
                for j in range(len(grad[i])):
                    r = random.randint(0, n)
                    enc_val = encrypt(grad[i, j], r)*1.0
                    empty_arr[i, j] = enc_val
            aa = tf.stack(empty_arr)
            aa = tf.cast(aa, tf.float64)
            enc_grads.append(aa)
        elif len(grad.shape) == 1:
            a, = grad.shape
            empty_arr = np.empty((a), dtype=np.float32)
            for i in range(len(grad)):
               r = random.randint(0, n)
               enc_val = encrypt(grad[i], r)*1.0
               empty_arr[i] = enc_val
            aa = tf.cast(aa, tf.float64)
            aa = tf.stack(empty_arr)
            enc_grads.append(aa)

    return enc_grads

def decrypt_gradients(grads):
    dec_grads = []
    for grad in grads:
        if len(grad.shape) == 2:
            a, b = grad.shape
            empty_arr = np.empty((a, b), dtype=np.float32)
            for i in range(len(grad)):
                for j in range(len(grad[i])):
                    enc_val = decrypt(grad[i, j])*1.0
                    empty_arr[i, j] = enc_val
            aa = tf.stack(empty_arr)
            aa = tf.cast(aa, tf.float64)
            dec_grads.append(aa)
        elif len(grad.shape) == 1:
            a, = grad.shape
            empty_arr = np.empty((a), dtype=np.float32)
            for i in range(len(grad)):
               enc_val = decrypt(grad[i])*1.0
               empty_arr[i] = enc_val
            aa = tf.cast(aa, tf.float64)
            aa = tf.stack(empty_arr)
            dec_grads.append(aa)

    return dec_grads

def DEC(grads):
    dec_grads = []
    for grad in grads:
        if len(grad.shape) == 2:
            a, b = grad.shape
            empty_arr = np.empty((a, b), dtype=np.float32)
            for i in range(len(grad)):
                for j in range(len(grad[i])):
                    val = DECRYPT(SK, n, q, t, poly_mod, grad[i, j])
                    empty_arr[i, j] = val
            res = tf.stack(empty_arr)
            res = tf.cast(res, tf.float32)
            dec_grads.append(res)
        elif len(grad.shape) == 4:
            a, b, c, d = grad.shape
            empty_arr = np.empty((a, b, c, d), dtype=np.float32)
            for a_i in range(a):
                for b_i in range(b):
                    for c_i in range(c):
                        for d_i in range(d):
                            val = DECRYPT(SK, n, q, t, poly_mod, grad[a_i, b_i, c_i, d_i])
                            empty_arr[a_i, b_i, c_i, d_i] = val
            res = tf.stack(empty_arr)
            res = tf.cast(res, tf.float32)
            dec_grads.append(res)
        elif len(grad.shape) == 1:
            a, = grad.shape
            empty_arr = np.empty((a), dtype=np.float32)
            for i in range(len(grad)):
                val = DECRYPT(SK, n, q, t, poly_mod, grad[i])
                empty_arr[i] = val
            res = tf.stack(empty_arr)
            res = tf.cast(res, tf.float32)
            dec_grads.append(res)

    return dec_grads

def dequant_helper(x, s, z):
    return (x-z)/s

def read_quant_info(i):
    with open(f'quant_info_{i}.pkl', 'rb') as pk:
        quant_info = pickle.load(pk)

    return quant_info['s'], quant_info['z']

def dequantize(q_grads):
    grads = []
    for i in range(len(q_grads)):
        s, z = read_quant_info(i)
        dequantized_tensor = dequant_helper(q_grads[i], s, z)
        grads.append(dequantized_tensor)

    return grads

def tryDownloadingAgg():
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob('agg_gradients.pkl')
        data = blob.download_as_string()
        globals()['agg_gradients'] = pickle.loads(data)
        return True
    except Exception as e:
        return False

def read_updated_gradients():
    # global agg_gradients
    updated_gradients = download_from_cloud(SERVER_FILE)
    # with open(SERVER_FILE, 'rb') as pickle_file:
    #     updated_gradients = pickle.load(pickle_file)

    q_agg_gradients = DEC(updated_gradients)
    globals()['agg_gradients'] = dequantize(q_agg_gradients)

    return

def local_train(model, loss_fn):
    grads = None

    # Iterate through a specified number of training epochs
    for epoch in range(epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            epoch_accuracy.update_state(y_batch_train, model(x_batch_train, training=True))

    return grads

def loss_eval(model, x, y, training):
    y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss_eval(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_weights)

def write_local_accuracy(acc):
    with open(f'local_accuracy_{client_no}.pkl', 'wb') as f:
        pickle.dump(acc, f)

    upload_to_cloud(f'local_accuracy_{client_no}.pkl')

    return

def new_local_training(model, optimizer, ckpt, manager):
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    grads = None
    ckpt.restore(manager.latest_checkpoint)

    for epoch in range(epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            epoch_accuracy.update_state(y_batch_train, model(x_batch_train, training=True))
            ckpt.step.assign_add(1)

            if int(ckpt.step) % 100 == 0:
                save_path = manager.save()

    epoch_acc = epoch_accuracy.result().numpy().item()
    write_local_accuracy(epoch_acc)

    return grads

def ENC(grads, pk, b_cap):
    local_grads_file_exists = os.path.isfile(f'local_gradients_{client_no}.pkl')
    if local_grads_file_exists:
        with open(f'local_gradients_{client_no}.pkl', 'rb') as pf:
            prev_enc_grads = pickle.load(pf)

    enc_grads = []
    for idx, grad in enumerate(grads):
        if len(grad.shape) == 2:
            a, b = grad.shape
            empty_arr = np.empty((a, b), dtype=tf.Tensor)
            for i in range(len(grad)):
                for j in range(len(grad[i])):
                    n_to_t = float(grad[i, j].numpy())
                    if not local_grads_file_exists or n_to_t != 0:
                        enc_val = ENCRYPT(PK, n, q, t, poly_mod, n_to_t, b_cap)
                        empty_arr[i, j] = enc_val
                    else:
                        empty_arr[i, j] = prev_enc_grads[idx][i, j]
            enc_grads.append(empty_arr)
        elif len(grad.shape) == 4:
            a, b, c, d = grad.shape
            empty_arr = np.empty((a, b, c, d), dtype=tf.Tensor)
            for a_i in range(a):
                for b_i in range(b):
                    for c_i in range(c):
                        for d_i in range(d):
                            n_to_t = float(grad[a_i, b_i, c_i, d_i].numpy())
                            if not local_grads_file_exists or n_to_t != 0:
                                enc_val = ENCRYPT(PK, n, q, t, poly_mod, n_to_t, b_cap)
                                empty_arr[a_i, b_i, c_i, d_i] = enc_val
                            else:
                                empty_arr[a_i, b_i, c_i, d_i] = prev_enc_grads[idx][a_i, b_i, c_i, d_i]
            enc_grads.append(empty_arr)
        elif len(grad.shape) == 1:
            a, = grad.shape
            empty_arr = np.empty((a), dtype=tf.Tensor)
            for i in range(len(grad)):
                n_to_t = float(grad[i].numpy())
                if not local_grads_file_exists or n_to_t != 0:
                    enc_val = ENCRYPT(PK, n, q, t, poly_mod, n_to_t, b_cap)
                    empty_arr[i] = enc_val
                else:
                    empty_arr[i] = prev_enc_grads[idx][i]
            enc_grads.append(empty_arr)

    return enc_grads

def clip(x):
    return tf.maximum(tf.minimum(x, quant_upper), quant_lower)

def quant_helper(x, s, z):
    return clip(s*x+z)

def write_quant_info(i, s, z):
    quant_info = {'s': s, 'z': z}
    with open(f'quant_info_{i}.pkl', 'wb') as pk:
        pickle.dump(quant_info, pk)

    return

def quantize(grads):
    q_grads = []
    for i in range(len(grads)):
        A1 = tf.reduce_max(grads[i]).numpy()
        A2 = tf.reduce_min(grads[i]).numpy()
        s = (2^quant_B + 1)/(A1-A2)
        z = -(round(A2 * s)) - 2^(quant_B-1)
        quantized_tensor = quant_helper(grads[i], s, z)
        q_grads.append(quantized_tensor)

        write_quant_info(i, s, z)

    return q_grads

def normalize():
    # global agg_gradients
    with open('grads.pkl', 'rb') as pickle_file:
        grads = pickle.load(pickle_file)

    if globals()['agg_gradients'] is None: print('Agg...')
    if grads is None: print('Grads...')

    updated_agg = [tf.where(tf.equal(agg_value, 0), prev_value, agg_value) for agg_value, prev_value in zip(globals()['agg_gradients'], grads)]

    difference_tensors = [new_tensor - old_tensor for new_tensor, old_tensor in zip(updated_agg, grads)]
    n_grads = [tf.math.divide_no_nan(tensor - tf.reduce_min(tensor), tf.reduce_max(tensor) - tf.reduce_min(tensor)) for tensor in difference_tensors]

    return n_grads

def compress(n_grads, q_grads):
    c_grads = [tf.where(normalized_tensor >= 0.5, q_grads_value, 0.0) for normalized_tensor, q_grads_value in zip(n_grads, q_grads)]
    total_entries = tf.reduce_sum([tf.size(tensor) for tensor in n_grads])
    entries_less_than_0_5 = tf.reduce_sum([tf.reduce_sum(tf.cast(tensor < 0.5, tf.int32)) for tensor in n_grads])

    R = entries_less_than_0_5 / total_entries
    R = R.numpy()

    with open(f'R_{client_no}.pkl', 'wb') as f:
        pickle.dump(R, f)

    upload_to_cloud(f'R_{client_no}.pkl')

    return c_grads

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('loss', type=str, help="Required argument for the loss method to be used")
    parser.add_argument('metrics', type=str, help="Required argument for the metrics to be used")
    parser.add_argument('lr', type=float, help="Required argument for lr value")
    parser.add_argument('decay', type=float, help="Required argument for ratio of lr & comms_round")
    parser.add_argument('momentum', type=float, help="Required argument for ratio optimizer momentum")
    parser.add_argument('shape', type=int, help="Required argument for shape of dataset")
    parser.add_argument('classes_count', type=int, help="Required argument for total number of classes")
    parser.add_argument('epoch', type=int, help="Required argument for number of epochs")
    parser.add_argument('verbose', type=int, help="Required argument for verbose")

    args = parser.parse_args()

    loss = args.loss
    metrics = args.metrics
    lr = args.lr
    decay = args.decay
    momentum = args.momentum
    shape = args.shape
    classes_count = args.classes_count
    epochs = args.epoch
    verbose = args.verbose

    model, loss_fn, optimizer = get_compiled_model()
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

    # PK, SK = read_paillier_keys()
    PK, SK = read_creds()

    #server_file_exists = os.path.isfile(SERVER_FILE)
    server_file_exists = tryDownloadingAgg()

    if server_file_exists:
        read_updated_gradients()
        globals()['agg_gradients'] = [tf.cast(grad, tf.float32) for grad in globals()['agg_gradients']]
        optimizer.apply_gradients(zip(globals()['agg_gradients'], model.trainable_variables))

    read_client_dataset()
    b_cap = read_b_cap()

    grads = new_local_training(model, optimizer, ckpt, manager)
    q_grads = quantize(grads)
    q_grads_file_exists = os.path.isfile('grads.pkl')

    enc_grads = None
    if q_grads_file_exists:
        n_grads = normalize()
        c_grads = compress(n_grads, q_grads)
        enc_grads = ENC(c_grads, PK, b_cap)
    else:
        # problem HERE!!!!
        enc_grads = ENC(q_grads, PK, b_cap)

    with open('local_gradients_{}.pkl'.format(client_no), 'wb') as pickle_file:
        pickle.dump(enc_grads, pickle_file)

    upload_to_cloud(f'local_gradients_{client_no}.pkl')

    with open('grads.pkl', 'wb') as pf:
        pickle.dump(grads, pf)

    model.save('iris.keras')
    print(client_no)