'''
    saved in only the first client (client_no = 1)
'''

import pickle
import subprocess
import numpy as np
from numpy.polynomial import polynomial as poly

client_no = 1

n = 2**4
# ciphertext modulus
q = 2**511
# plaintext modulus
t = 2**8
# polynomial modulus
poly_mod = np.array([1] + [0] * (n - 1) + [1])

client_private_ips = ['10.128.0.4', '10.128.0.5', '10.128.0.6']

def write_creds(pk, sk, a, b):
    creds = {}
    creds['keys'] = { 'pk': pk, 'sk': sk}

    with open('creds.pkl', 'wb') as pk:
        pickle.dump(creds, pk)

    cred = {}
    creds['coeff'] = {'a': a}

    with open('coeff_a.pkl', 'wb') as pk:
        pickle.dump(creds, pk)

    cred = {}
    creds['coeff'] = {'b': b}

    with open(f'coeff_b_{client_no}.pkl', 'wb') as pk:
        pickle.dump(creds, pk)

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

def keygen(size, modulus, poly_mod, a):
    sk = gen_binary_poly(size)
    e = gen_normal_poly(size)
    b = polyadd(polymul(-a, sk, modulus, poly_mod), e, modulus, poly_mod)
    return (b, a), sk

if __name__ == "__main__":
    a = gen_uniform_poly(n, q)
    pk, sk = keygen(n, q, poly_mod, a)
    b = pk[0]

    write_creds(pk, sk, a, b)

    for i in range(len(client_private_ips)):
        private_ip = client_private_ips[i]
        scp_command = f'scp coeff_a.pkl g1805021@{private_ip}:/home/g1805021'
        subprocess.check_output(scp_command, shell=True, text=True).strip()