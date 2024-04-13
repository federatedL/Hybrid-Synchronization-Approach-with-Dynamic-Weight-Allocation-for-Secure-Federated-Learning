import os
import traceback
import glob
import math
from google.cloud import storage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'cloud.json'

import random
import subprocess
import pickle
import phe
from datetime import datetime
from phe import paillier
import tensorflow as tf
import numpy as np
from numpy.polynomial import polynomial as poly
from tqdm import tqdm
import asyncssh
import asyncio
import threading

from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import backend as K

# Define hyperparameters
SYNC_MODE = True
lr = 0.01
comms_round = 32
percentile = 6.25 # For Sync-Async switch
loss='sparse_categorical_crossentropy'
metrics = ['sparse_categorical_accuracy'] #['accuracy']
client_epoch = 1
client_verbose = 0
decay = lr / comms_round #learning rate decay
momentum = 0.9
classes_count = 3
shape = None

bucket_name = 'fl-data'
storage_client = storage.Client()
data_path = './swarm_aligned'
clients = None
clients_batched = None
PK = None
ASYNC_TIMEOUT = 120 # 2 minutes
lock = threading.Lock()
comp_time = 0

n = 2**4
q = 2**511
t = 2**63
poly_mod = np.array([1] + [0] * (n - 1) + [1])

# Data Structures
client_private_ips = [
    '10.128.0.3',
    '10.128.0.4',
    '10.128.0.5',
    '10.128.0.6'
]

client_public_ips = [
    '104.197.89.108',
    '34.41.69.242',
    '34.30.32.19',
    '34.27.85.124'
]

global_gradient = None

latest_client_round = {
    1: 0, # client_no: latest_round
    2: 0,
    3: 0,
    4: 0
}

latest_client_grad = {
    1: None, # client_no: latest_grad
    2: None,
    3: None,
    4: None
}

ratios = [ 0.25, 0.25, 0.25, 0.25 ]

def upload_to_cloud(file_name):
    start = datetime.now()
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_name)
        end = datetime.now()
        globals()['comp_time'] += (end-start).total_seconds()
    except Exception as e:
        end = datetime.now()
        globals()['comp_time'] += (end-start).total_seconds()
        return

def download_from_cloud(file_name):
    start = datetime.now()
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)
        data = blob.download_as_string()
        res = pickle.loads(data)
        end = datetime.now()
        globals()['comp_time'] += (end-start).total_seconds()
        return res
    except Exception as e:
        end = datetime.now()
        globals()['comp_time'] += (end-start).total_seconds()
        return None

def load_dataset():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    X_train = np.reshape(X_train, (-1, 784))
    X_test = np.reshape(X_test, (-1, 784))

    return X_train, X_test, y_train, y_test

def create_clients(data_list, label_list, num_clients=10, initial='clients'):
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]
    data = list(zip(data_list, label_list))
    random.shuffle(data)

    # Define unique ratios for each client
    #ratios = np.random.dirichlet(np.ones(num_clients), size=1).tolist()[0]
    print('-------------------------------------')
    print(f"Ratios of divided datasets: {globals()['ratios']}")
    print('-------------------------------------')

    sizes = []
    total_samples = len(data)
    remaining_samples = total_samples

    for i in range(num_clients - 1):
        size = int(globals()['ratios'][i] * total_samples)
        sizes.append(size)
        remaining_samples -= size

    sizes.append(remaining_samples)

    shards = [data[i:i + sizes[idx]] for idx, i in enumerate(np.cumsum([0] + sizes[:-1]))]

    return {client_names[i]: shards[i] for i in range(len(client_names))}

def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

def save_client_batched_data(client_no, client_batched_data):
    # Convert TensorFlow _BatchDataset to list of numpy arrays
    client_batched_data_list = []
    for data, labels in client_batched_data:
        client_batched_data_list.append((data.numpy(), labels.numpy()))

    with open('client_batched_data_{}.pkl'.format(client_no), 'wb') as pickle_file:
        pickle.dump(client_batched_data_list, pickle_file)

    return

def save_client_dataset(client_no, client_dataset):
    with open('client_dataset_{}.pkl'.format(client_no), 'wb') as pickle_file:
        pickle.dump(client_dataset, pickle_file)

    return

def save_agg_gradients(agg_gradients):
    with open('agg_gradients.pkl', 'wb') as pickle_file:
        pickle.dump(agg_gradients, pickle_file)

    return

def dispatch_data():
    '''
        sends data to clients
    '''

    # Save dataset files
    client_no = 1
    for client in clients:
        client_dataset = clients[client]
        save_client_dataset(client_no, client_dataset)
        client_no += 1

    # Send dataset files to clients
    for i in range(len(client_public_ips)):
        upload_to_cloud(f'client_dataset_{i+1}.pkl')
        # public_ip = client_public_ips[i]
        # scp_command = f'scp -i ~/.ssh/id_rsa ./client_dataset_{i+1}.pkl g1805021@{public_ip}:~'
        # subprocess.check_output(scp_command, shell=True, text=True).strip()

    return

def get_client_gradients(client_no, public_ip):
    # scp_command = f'scp -i ~/.ssh/id_rsa g1805021@{public_ip}:/home/g1805021/local_gradients_{client_no}.pkl /home/g1805021/FedServer/'
    # subprocess.check_output(scp_command, shell=True, text=True).strip()
    local_gradients = download_from_cloud(f'local_gradients_{client_no}.pkl')

    # with open('local_gradients_{}.pkl'.format(client_no), 'rb') as pickle_file:
    #     local_gradients = pickle.load(pickle_file)

    return local_gradients

def get_client_local_accuracy(client_no, public_ip):
    # scp_command = f'scp -i ~/.ssh/id_rsa g1805021@{public_ip}:/home/g1805021/local_accuracy_{client_no}.pkl /home/g1805021/FedServer/LA'
    # subprocess.check_output(scp_command, shell=True, text=True).strip()
    local_accuracy = download_from_cloud(f'local_accuracy_{client_no}.pkl')

    # with open('LA/local_accuracy_{}.pkl'.format(client_no), 'rb') as pickle_file:
    #     local_accuracy = pickle.load(pickle_file)

    return local_accuracy

def get_public_key(public_ip):
    scp_command = f'scp -i ~/.ssh/id_rsa ubuntu@{public_ip}:/home/g1805021/public_key.pkl /home/g1805021/FedServer/'
    subprocess.check_output(scp_command, shell=True, text=True).strip()

    with open('public_key.pkl', 'rb') as pickle_file:
        public_key = pickle.load(pickle_file)

    return public_key['n']

def send_public_key(public_ip):
    scp_command = f'scp -i ~/.ssh/id_rsa ./public_key.pkl g1805021@{public_ip}:~'
    subprocess.check_output(scp_command, shell=True, text=True).strip()

    return

def polymul(x, y, modulus, poly_mod):
    return poly.polydiv(poly.polymul(x, y) % modulus, poly_mod)[1] % modulus

def polyadd(x, y, modulus, poly_mod):
    return poly.polydiv(poly.polyadd(x, y) % modulus, poly_mod)[1] % modulus

def add_ciphers(ct1, ct2, q, poly_mod):
    new_ct0 = polyadd(ct1[0], ct2[0], q, poly_mod)
    new_ct1 = polyadd(ct1[1], ct2[1], q, poly_mod)

    return (new_ct0, new_ct1)

def mul_plain(ct, pt, q, t, poly_mod):
    size = len(poly_mod) - 1
    m = np.array([pt], dtype=np.float32) % t
    new_c0 = polymul(ct[0], m, q, poly_mod)
    new_c1 = polymul(ct[1], m, q, poly_mod)

    return (new_c0, new_c1)

def ADD(lst):
    res = None
    if len(lst[0].shape) == 2:
        a, b = lst[0].shape
        empty_arr = np.zeros((a, b), dtype=tf.Tensor)
        for k in range(len(lst)):
            for i in range(a):
                for j in range(b):
                    if k == 0:
                        empty_arr[i, j] = lst[k][i, j]
                    else:
                        arr = lst[k][i, j]
                        empty_arr[i, j] = add_ciphers(arr, empty_arr[i, j], q, poly_mod)
        res = empty_arr
    elif len(lst[0].shape) == 4:
        a, b, c, d = lst[0].shape
        empty_arr = np.zeros((a, b, c, d), dtype=tf.Tensor)
        for k in range(len(lst)):
            for a_i in range(a):
                for b_i in range(b):
                    for c_i in range(c):
                        for d_i in range(d):
                            if k == 0:
                                empty_arr[a_i, b_i, c_i, d_i] = lst[k][a_i, b_i, c_i, d_i]
                            else:
                                arr = lst[k][a_i, b_i, c_i, d_i]
                                empty_arr[a_i, b_i, c_i, d_i] = add_ciphers(arr, empty_arr[a_i, b_i, c_i, d_i], q, poly_mod)
        res = empty_arr
    elif len(lst[0].shape) == 1:
        a, = lst[0].shape
        empty_arr = np.zeros((a), dtype=tf.Tensor)
        for k in range(len(lst)):
            for i in range(a):
                if k == 0:
                    empty_arr[i] = lst[k][i]
                else:
                    arr = lst[k][i]
                    empty_arr[i] = add_ciphers(arr, empty_arr[i], q, poly_mod)
        res = empty_arr

    return res

def FILTER_CSUM1(ret):
    res = None
    if len(ret.shape) == 2:
        a, b = ret.shape
        empty_arr = np.zeros((a, b), dtype=np.ndarray)
        for i in range(a):
            for j in range(b):
                _, csum1 = ret[i, j]
                empty_arr[i, j] = csum1
        res = empty_arr
    elif len(ret.shape) == 4:
        a, b, c, d = ret.shape
        empty_arr = np.zeros((a, b, c, d), dtype=np.ndarray)
        for a_i in range(a):
            for b_i in range(b):
                for c_i in range(c):
                    for d_i in range(d):
                        _, csum1 = ret[a_i, b_i, c_i, d_i]
                        empty_arr[a_i, b_i, c_i, d_i] = csum1
        res = empty_arr
    elif len(ret.shape) == 1:
        a,  = ret.shape
        empty_arr = np.zeros((a), dtype=np.ndarray)
        for i in range(a):
            _, csum1 = ret[i]
            empty_arr[i] = csum1
        res = empty_arr

    return res

def read_R(client_no):
    with open(f'R_{client_no}.pkl', 'rb') as r:
        R = pickle.load(r)

    return R

def R_values():
    client_no = 1
    R_vals = []
    for i in range(len(client_public_ips)):
        R = download_from_cloud(f'R_{client_no}.pkl')
        # public_ip = client_public_ips[i]
        # scp_command = f'scp -i ~/.ssh/id_rsa g1805021@{public_ip}:/home/g1805021/R_{client_no}.pkl /home/g1805021/FedServer/'
        # subprocess.check_output(scp_command, shell=True, text=True).strip()

        # R = read_R(client_no)
        R_vals.append(R)
        client_no += 1

    print(f'R_values are: {R_vals}')

    return R_vals

def calculate_client_weights(cur_round):
    R_vals = R_values()
    client_weights = {}
    max_round = max(latest_client_round.values())
    for client_no in latest_client_round.keys():
        if R_vals[client_no-1] is None:
          client_weights[client_no] = 1
        else:
          delay = max_round - latest_client_round[client_no]
          w = (globals()['ratios'][client_no-1]/(delay+1)) / R_vals[client_no-1]
          w = round(w, 2)
          w = round(w*100)
          client_weights[client_no] = w

    # normalizing weights
    values = {key: value / 100.0 for key, value in client_weights.items()}
    int_min_value = 0
    int_max_value = 100
    normalized_weights = {key: int((value * (int_max_value - int_min_value)) + int_min_value) for key, value in values.items()}

    return normalized_weights

def calcualte_sync_client_weights():
    R_vals = R_values()
    client_weights = {}

    for client_no in range(1, 5):
         w = globals()['ratios'][client_no-1]/R_vals[client_no-1]
         w = round(w, 2)
         w = round(w*100)
         client_weights[client_no] = w

    values = {key: value / 100.0 for key, value in client_weights.items()}
    int_min_value = 0
    int_max_value = 100
    normalized_weights = {key: int((value * (int_max_value - int_min_value)) + int_min_value) for key, value in values.items()}

    return normalized_weights

def aggregate_client_gradients(client_grads_list, comm_round):
    '''
        returns aggregated gradients
    '''
    print('In aggregate gradients...')
    agg_gradients = []
    steps = len(client_grads_list[0])

    client_weights = None
    if comm_round != 0:
        client_weights = calcualte_sync_client_weights()

    for i in range(steps):
        if comm_round != 0:
            agg_gradients.append(ADD([WEIGHT_GRAD(client_grads[i], client_weights[idx+1]) for idx, client_grads in enumerate(client_grads_list)]))
        else:
            agg_gradients.append(ADD([client_grads[i] for client_grads in client_grads_list]))


    csum1 = []
    for ag in agg_gradients:
        cs1 = FILTER_CSUM1(ag)
        csum1.append(cs1)

    return agg_gradients, csum1

def WEIGHT_GRAD(client_grad, client_weight):
    res = None
    if len(client_grad.shape) == 2:
        a, b = client_grad.shape
        empty_arr = np.empty((a, b), dtype=tf.Tensor)
        for i in range(a):
            for j in range(b):
                weighted_val = mul_plain(client_grad[i, j], client_weight, q, t, poly_mod)
                empty_arr[i, j] = weighted_val
        res = empty_arr
    elif len(client_grad.shape) == 4:
        a, b, c, d = client_grad.shape
        empty_arr = np.empty((a, b, c, d), dtype=tf.Tensor)
        for a_i in range(a):
            for b_i in range(b):
                for c_i in range(c):
                    for d_i in range(d):
                        weighted_val = mul_plain(client_grad[a_i, b_i, c_i, d_i], client_weight, q, t, poly_mod)
                        empty_arr[a_i, b_i, c_i, d_i] = weighted_val
        res = empty_arr
    elif len(client_grad.shape) == 1:
        a, = client_grad.shape
        empty_arr = np.empty((a), dtype=tf.Tensor)
        for i in range(a):
            weighted_val = mul_plain(client_grad[i], client_weight, q, t, poly_mod)
            empty_arr[i] = weighted_val
        res = empty_arr

    return res

def async_aggregate_client_gradients(client_grads_list, cur_round):
    '''
        returns aggregated gradients for asynchronous training
    '''
    agg_gradients = []
    steps = len(client_grads_list[0])
    client_weights = calculate_client_weights(cur_round)

    for i in range(steps):
        lst = []
        for client_no in latest_client_grad.keys():
            client_grads = latest_client_grad[client_no]
            if client_grads is not None:
                client_grad = client_grads[i]
                weighted_grad = WEIGHT_GRAD(client_grad, client_weights[client_no])
                lst.append(weighted_grad)

        agg_gradients.append(ADD(lst))
        # agg_gradients.append(ADD([client_grads[i] for client_grads in client_grads_list]))

    csum1 = []
    for ag in agg_gradients:
        cs1 = FILTER_CSUM1(ag)
        csum1.append(cs1)

    return agg_gradients, csum1

def send_updated_gradients(grads):
    save_agg_gradients(grads)
    upload_to_cloud(f'agg_gradients.pkl')

    # for i in range(len(client_public_ips)):
        # public_ip = client_public_ips[i]
        # scp_command = f'scp -i ~/.ssh/id_rsa ./agg_gradients.pkl g1805021@{public_ip}:~'
        # subprocess.check_output(scp_command, shell=True, text=True).strip()
    return

def async_cleanup():
    pattern1 = 'local_gradients_*.pkl'
    pattern2 = 'client_acc_*.pkl'

    files1 = glob.glob(pattern1)
    files2 = glob.glob(pattern2)
    files = files1 + files2

    for file in files:
        try:
            os.remove(file)
            print(f"File '{file}' removed successfully")
        except OSError as e:
            print(f"Error: {file} - {e.strerror}")

    return

async def run_client(client_host):
    try:
        async with asyncssh.connect(client_host) as conn:
            # Execute the 'test.py' file (replace with your actual command)
            return await conn.run(f"python3 test_client.py {loss} {metrics} {lr} {decay} {momentum} {shape} {classes_count} {client_epoch} {client_verbose}", check=True)
    except (OSError, asyncssh.Error) as exc:
        return exc
        # return f"{exc}\n{traceback.format_exc()}"

async def run_multiple_clients(lst):
    async_cleanup()

    tasks = [run_client(host) for host in client_public_ips]
    done, pending = await asyncio.wait(tasks, timeout=ASYNC_TIMEOUT)
    # done, pending = await asyncio.wait(tasks)

    for task in done:
        try:
            result = task.result()
            if isinstance(result, Exception):
                print(f'Task failed: {result.stderr}')
            else:
                ret = int(result.stdout)
                lst.append(ret)
        except asyncio.CancelledError:
            pass

    for task in pending:
        task.cancel()

    return

def polyadd(x, y, modulus, poly_mod):
    return poly.polydiv(poly.polyadd(x, y) % modulus, poly_mod)[1] % modulus

def agg_pub_key(lst, modulus, poly_mod):
  if len(lst) <= 1:
    return lst[0]
  ans = polyadd(lst[0], lst[1], modulus, poly_mod)
  if len(lst) == 2:
    return ans
  for i in range(2, len(lst)):
    ans = polyadd(ans, lst[i], modulus, poly_mod)

  return ans

def read_coeff_b(client_no):
    with open(f'coeff_b_{client_no}.pkl', 'rb') as c:
        coeff = pickle.load(c)

    b = coeff['coeff']['b']

    return b

def client_init():
    client_no = 1
    for private_ip in client_private_ips:
        ssh_command = f'ssh uncleroger@{private_ip} "python3 test_creds.py"'
        result = subprocess.check_output(ssh_command, shell=True, text=True).strip()

        client_no += 1

    print('-------------------------------')
    print('Created client-side credentials')
    print('-------------------------------\n')

    client_no = 1
    b_cap = []
    for public_ip in client_public_ips:
        coeff = download_from_cloud(f'coeff_b_{client_no}.pkl')
        # scp_command = f'scp -i ~/.ssh/id_rsa g1805021@{public_ip}:/home/g1805021/coeff_b_{client_no}.pkl /home/g1805021/FedServer/'
        # subprocess.check_output(scp_command, shell=True, text=True).strip()

        # b = read_coeff_b(client_no)
        b = coeff['coeff']['b']
        b_cap.append(b)
        client_no += 1

    res = agg_pub_key(b_cap, q, poly_mod)

    print('-------------------------------')
    print('Computed b_cap')
    print('-------------------------------\n')

    with open('b_cap.pkl', 'wb') as pk:
        pickle.dump(res, pk)

    client_no = 1
    upload_to_cloud(f'b_cap.pkl')
    # for public_ip in client_public_ips:
        # scp_command = f'scp -i ~/.ssh/id_rsa ./b_cap.pkl g1805021@{public_ip}:~'
        # subprocess.check_output(scp_command, shell=True, text=True).strip()
        # client_no += 1

    print('-------------------------------')
    print('Transferred b_cap data')
    print('-------------------------------\n')

    return

def send_csum1_to_clients(csum1):
    with open('csum1.pkl', 'wb') as pickle_file:
        pickle.dump(csum1, pickle_file)

    upload_to_cloud(f'csum1.pkl')

    # for public_ip in client_public_ips:
    #     scp_command = f'scp -i ~/.ssh/id_rsa ./csum1.pkl g1805021@{public_ip}:~'
    #     subprocess.check_output(scp_command, shell=True, text=True).strip()

    return

def calculate_ds_at_clients():
    client_no = 1
    for private_ip in client_private_ips:
        ssh_command = f'ssh uncleroger@{private_ip} "python3 test_ds.py"'
        result = subprocess.check_output(ssh_command, shell=True, text=True).strip()

        client_no += 1

    return

def ADD_DS(lst):
    D = None
    if len(lst[0].shape) == 2:
        a, b = lst[0].shape
        empty_arr = np.zeros((a, b), dtype=np.ndarray)
        for k in range(len(lst)):
            for i in range(a):
                for j in range(b):
                    if k == 0:
                        empty_arr[i, j] = lst[k][i, j]
                    else:
                        empty_arr[i, j] = polyadd(lst[k][i, j], empty_arr[i, j], q, poly_mod)
        D = empty_arr
    elif len(lst[0].shape) == 4:
        a, b, c, d = lst[0].shape
        empty_arr = np.zeros((a, b, c, d), dtype=np.ndarray)
        for k in range(len(lst)):
            for a_i in range(a):
                for b_i in range(b):
                    for c_i in range(c):
                        for d_i in range(d):
                            if k == 0:
                                empty_arr[a_i, b_i, c_i, d_i] = lst[k][a_i, b_i, c_i, d_i]
                            else:
                                empty_arr[a_i, b_i, c_i, d_i] = polyadd(lst[k][a_i, b_i, c_i, d_i], empty_arr[a_i, b_i, c_i, d_i], q, poly_mod)
        D = empty_arr
    elif len(lst[0].shape) == 1:
        a, = lst[0].shape
        empty_arr = np.zeros((a), dtype=np.ndarray)
        for k in range(len(lst)):
            for i in range(a):
                if k == 0:
                    empty_arr[i] = lst[k][i]
                else:
                    empty_arr[i] = polyadd(lst[k][i], empty_arr[i], q, poly_mod)
        D = empty_arr

    return D

def combine_ds():
    client_no = 1
    dec_shares = []
    for public_ip in client_public_ips:
        ds = download_from_cloud(f'ds_{client_no}.pkl')
        # scp_command = f'scp -i ~/.ssh/id_rsa g1805021@{public_ip}:/home/g1805021/ds_{client_no}.pkl /home/g1805021/FedServer/'
        # subprocess.check_output(scp_command, shell=True, text=True).strip()

        with open(f'ds_{client_no}.pkl', 'wb') as f:
            pickle.dump(ds, f)

        # with open(f'ds_{client_no}.pkl', 'rb') as pf:
        #     ds = pickle.load(pf)

        dec_shares.append(ds)
        client_no += 1

    print('read ds from clients')

    added_ds = []
    steps = len(dec_shares[0])
    for i in range(steps):
        added_ds.append(ADD_DS([ds[i] for ds in dec_shares]))

    return added_ds

def FINAL_AGG(agg, D):
    res = None
    if len(agg.shape) == 2:
        a, b = agg.shape
        empty_arr = np.zeros((a, b), dtype=np.ndarray)
        for i in range(a):
            for j in range(b):
                csum0, _ = agg[i, j]
                #if not SYNC_MODE:
                    # print(f'Shape of csum0 is {csum0.shape} and shape of D is {D.shape}')
                empty_arr[i, j] = csum0 + D[i, j]
        res = empty_arr
    elif len(agg.shape) == 4:
        a, b, c, d = agg.shape
        empty_arr = np.zeros((a, b, c, d), dtype=np.ndarray)
        for a_i in range(a):
            for b_i in range(b):
                for c_i in range(c):
                    for d_i in range(d):
                        csum0, _ = agg[a_i, b_i, c_i, d_i]
                        empty_arr[a_i, b_i, c_i, d_i] = csum0 + D[a_i, b_i, c_i, d_i]
        res = empty_arr
    elif len(agg.shape) == 1:
        a,  = agg.shape
        empty_arr = np.zeros((a), dtype=np.ndarray)
        for i in range(a):
            csum0, _ = agg[i]
            empty_arr[i] = csum0 + D[i]
        res = empty_arr

    return res

def aggregate_grads_ds(updated_grads, added_ds):
    res = []
    # print(f'Length of updated_grads in aggregate_grads_ds: {len(updated_grads)}')
    for i in range(len(updated_grads)):
        grad = updated_grads[i]
        ds = added_ds[i]
        agg = FINAL_AGG(grad, ds)
        total_clients = 4
        agg /= total_clients
        res.append(agg)

    return res

def combine_threading_ds(client_no):
    dec_shares = []
    public_ip = client_public_ips[client_no-1]
    ds = download_from_cloud(f'ds_{client_no}.pkl')
    # scp_command = f'scp -i ~/.ssh/id_rsa g1805021@{public_ip}:/home/g1805021/ds_{client_no}.pkl /home/g1805021/FedServer/'
    # subprocess.check_output(scp_command, shell=True, text=True).strip()

    with open(f'ds_{client_no}.pkl', 'wb') as f:
            pickle.dump(ds, f)

    for cn in range(1, len(client_private_ips)+1):
        if os.path.isfile(f'ds_{cn}.pkl'):
            with open(f'ds_{cn}.pkl', 'rb') as pf:
                ds = pickle.load(pf)
            dec_shares.append(ds)

    print(f'read ds for client {client_no}')

    added_ds = []
    steps = len(dec_shares[0])
    for i in range(steps):
        added_ds.append(ADD_DS([ds[i] for ds in dec_shares]))

    return added_ds

def calculate_ds_at_threading_client(client_no):
    private_ip = client_private_ips[client_no-1]
    ssh_command = f'ssh uncleroger@{private_ip} "python3 test_ds.py"'
    result = subprocess.check_output(ssh_command, shell=True, text=True).strip()

    return

def send_csum1_to_threading_clients(csum1, client_no):
    with open('csum1.pkl', 'wb') as pickle_file:
        pickle.dump(csum1, pickle_file)

    # public_ip = client_public_ips[client_no-1]
    # scp_command = f'scp -i ~/.ssh/id_rsa ./csum1.pkl g1805021@{public_ip}:~'
    # subprocess.check_output(scp_command, shell=True, text=True).strip()
    upload_to_cloud(f'csum1.pkl')

    return

def threading_aggregate_client_gradients(cn, grads, cur_round):
    '''
        returns aggregated gradients for asynchronous training
    '''

    latest_client_round[cn] = cur_round+1
    agg_gradients = []
    steps = len(grads)
    client_weights = calculate_client_weights(cur_round+1)

    for i in range(steps):
        lst = []
        for client_no in latest_client_grad.keys():
            client_grads = latest_client_grad[client_no]
            if client_grads is not None:
                client_grad = client_grads[i]
                weighted_grad = WEIGHT_GRAD(client_grad, client_weights[client_no])
                lst.append(weighted_grad)

        agg_gradients.append(ADD(lst))
        # agg_gradients.append(ADD([client_grads[i] for client_grads in client_grads_list]))

    csum1 = []
    for ag in agg_gradients:
        cs1 = FILTER_CSUM1(ag)
        csum1.append(cs1)

    return agg_gradients, csum1

def send_threading_updated_gradients(client_no, grads):
    save_agg_gradients(grads)

    # public_ip = client_public_ips[client_no-1]
    # scp_command = f'scp -i ~/.ssh/id_rsa ./agg_gradients.pkl g1805021@{public_ip}:~'
    # subprocess.check_output(scp_command, shell=True, text=True).strip()
    upload_to_cloud(f'agg_gradients.pkl')

    return

def threading_helper(client_no, r):
    start = datetime.now()
    public_ip = client_public_ips[client_no-1]
    with lock:
        client_grads = get_client_gradients(client_no, public_ip)
        local_accuracy = get_client_local_accuracy(client_no, public_ip)

        latest_client_grad[client_no] = client_grads
        updated_grads, csum1 = threading_aggregate_client_gradients(client_no, client_grads, r)
        send_csum1_to_threading_clients(csum1, client_no)
        print(f'csum1 transferred to client {client_no}')
        calculate_ds_at_threading_client(client_no)
        print(f'calculated ds at client {client_no}')
        added_ds = combine_threading_ds(client_no)
        print(f'combined ds for client {client_no}')
        res = aggregate_grads_ds(updated_grads, added_ds)
        
        # Because the weights were multiplied by 100 for convenience
        for i, arr in enumerate(res):
            res[i] = arr/100

        send_threading_updated_gradients(client_no, res)

        end = datetime.now()
        globals()['comp_time'] += (end-start).total_seconds()

        print(f'Client {client_no} has completed training round {r+1}')

    return

def handle_async_client(client_no, limit):
    private_ip = client_private_ips[client_no-1]
    for r in range(limit, comms_round):
        # ssh to call test_client.py
        ssh_command = f'ssh uncleroger@{private_ip} "python3 test_client.py {loss} {metrics} {lr} {decay} {momentum} {shape} {classes_count} {client_epoch} {client_verbose}"'
        result = subprocess.check_output(ssh_command, shell=True, text=True).strip() #the result contains the local model weights

        # scp the local gradient and accuracy file
        # aggregate weighted grads from a function with lock
        threading_helper(client_no, r)

    print(f'Training completed for client {client_no}')

    return

def new_global_training():
    '''
        returns global model trained with client models
    '''

    global SYNC_MODE
    global ASYNC_TIMEOUT
    global global_gradient

    # Send the clients their respective data shards
    start = datetime.now()
    dispatch_data()
    end = datetime.now()
    print(f'Dataset distribution time: {(end-start).total_seconds()} seconds')
    globals()['comp_time'] += (end-start).total_seconds()

    start = datetime.now()
    client_init()
    end = datetime.now()
    print(f'Credentials time: {(end-start).total_seconds()} seconds')
    globals()['comp_time'] += (end-start).total_seconds()

    limit = 0

    # Asynchronous rounds
    print('Asynchronous training started...')
    threads = []
    for client_id in range(len(client_private_ips)):
        thread_name = f'Thread client {client_id+1}'
        thread = threading.Thread(name=thread_name, target=handle_async_client, args=(client_id+1, limit,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print('Asynchronous training ended...')

    return

def save_test_data(X_test, y_test):
    with open('test_data_X.pkl', 'wb') as pickle_file:
        pickle.dump(X_test, pickle_file)

    with open('test_data_y.pkl', 'wb') as pickle_file:
        pickle.dump(y_test, pickle_file)

    return

def send_test_data(public_ip):
    # scp_command = f'scp -i ~/.ssh/id_rsa ./test_data_X.pkl g1805021@{public_ip}:~'
    # subprocess.check_output(scp_command, shell=True, text=True).strip()
    # scp_command = f'scp -i ~/.ssh/id_rsa ./test_data_y.pkl g1805021@{public_ip}:~'
    # subprocess.check_output(scp_command, shell=True, text=True).strip()
    upload_to_cloud(f'test_data_X.pkl')
    upload_to_cloud(f'test_data_y.pkl')

    return

def get_client_loss_accuracy(client_no, public_ip):
    result = download_from_cloud(f'client_acc_{client_no}.pkl')
    # scp_command = f'scp -i ~/.ssh/id_rsa g1805021@{public_ip}:/home/g1805021/client_acc_{client_no}.pkl /home/g1805021/FedServer/'
    # subprocess.check_output(scp_command, shell=True, text=True).strip()

    # with open('client_acc_{}.pkl'.format(client_no), 'rb') as client_acc:
    #     result = pickle.load(client_acc)

    #return combined_data['loss'], combined_data['acc']
    return result

def test_accuracy(X_test, y_test):
    client_no = 1
    loss_arr = []
    acc_arr = []
    client_names = list(clients.keys())
    random.shuffle(client_names)

    save_test_data(X_test, y_test)
    start = datetime.now()

    for client in tqdm(client_names, desc='Progress Bar'):
        private_ip = client_private_ips[client_no - 1]
        public_ip = client_public_ips[client_no - 1]

        send_test_data(public_ip)

        ssh_command = f'ssh uncleroger@{private_ip} "python3 client_accuracy.py {client_no}"'
        subprocess.check_output(ssh_command, shell=True, text=True).strip()

        acc = get_client_loss_accuracy(client_no, public_ip)
        #loss_arr.append(loss)
        acc_arr.append(acc)
        client_no += 1

        K.clear_session()

    #avg_loss = sum(loss_arr)/len(loss_arr)
    end = datetime.now()
    globals()['comp_time'] += (end-start).total_seconds()
    avg_acc = sum(acc_arr)/len(acc_arr)
    testing_time = (end-start).total_seconds()

    with open('result_avg_acc.txt', 'w') as f:
        f.write(str(avg_acc) + '\n')

    with open('result_testing_time.txt', 'w') as t:
        t.write(str(testing_time) + '\n')

    with open('computational_overhead.txt', 'w') as c:
        c.write(str(globals()['comp_time']) + '\n')

    print('\n-------------------------------------------------------------')
    print(f'Average accuracy is {avg_acc}')
    print(f'Testing time taken: {testing_time}')
    print('-------------------------------------------------------------')

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_dataset()
    clients = create_clients(X_train, y_train, num_clients=4, initial='client')

    shape = X_train.shape[1]

    start = datetime.now()
    new_global_training()
    end = datetime.now()

    training_time = (end-start).total_seconds()

    with open('result_training_time.txt', 'w') as f:
        f.write(str(training_time) + '\n')

    print('----------------------------')
    print('Global training completed...')
    print(f'Training time taken: {training_time}')
    print('----------------------------')
    print('Testing accuracy')
    print('----------------------------')

    test_accuracy(X_test, y_test)