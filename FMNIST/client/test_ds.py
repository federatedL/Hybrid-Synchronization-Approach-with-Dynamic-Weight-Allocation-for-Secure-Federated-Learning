# For chief coordinator

import pickle
import numpy as np
from numpy.polynomial import polynomial as poly

client_no = 1
n = 2**4
q = 2**511
poly_mod = np.array([1] + [0] * (n - 1) + [1])

def read_sk():
    with open('creds.pkl', 'rb') as c:
        creds = pickle.load(c)

    sk = creds['keys']['sk']

    return sk
    
def read_csum1():
    with open('csum1.pkl', 'rb') as c:
        csum1 = pickle.load(c)
        
    return csum1

def polymul(x, y, modulus, poly_mod):
    return poly.polydiv(poly.polymul(x, y) % modulus, poly_mod)[1] % modulus

def polyadd(x, y, modulus, poly_mod):
    return poly.polydiv(poly.polyadd(x, y) % modulus, poly_mod)[1] % modulus

def gen_normal_poly(size):
    return np.random.normal(0, 2, size=size)

def calc_dec_share(sk, csum1, size, modulus, poly_mod):
    e = gen_normal_poly(size)
    
    return polyadd(polymul(sk, csum1, modulus, poly_mod), e, modulus, poly_mod)

def DEC_SHARE(csum1, sk):
    res = None
    if len(csum1.shape) == 2:
        a, b = csum1.shape
        empty_arr = np.empty((a, b), dtype=np.ndarray)
        for i in range(len(csum1)):
            for j in range(len(csum1[i])):
                ds = calc_dec_share(sk, csum1[i, j], n, q, poly_mod)
                empty_arr[i, j] = ds
        res = empty_arr
    elif len(csum.shape) == 4:
        a, b, c, d = csum1.shape
        empty_arr = np.empty((a, b, c, d), dtype=np.ndarray)
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    for l in range(d):
                        ds = calc_dec_share(sk, csum1[i, j, k, l], n, q, poly_mod)
                        empty_arr[i, j, k, l] = ds
        res = empty_arr
    elif len(csum1.shape) == 1:
        a,  = csum1.shape
        empty_arr = np.empty((a), dtype=np.ndarray)
        for i in range(len(csum1)):
            ds = calc_dec_share(sk, csum1[i], n, q, poly_mod)
            empty_arr[i] = ds
        res = empty_arr

    return res
  
def ds_helper(csum1, sk):
    dec_shares = []
    for cs in csum1:
        ds = DEC_SHARE(cs, sk)
        dec_shares.append(ds)
        
    with open(f'ds_{client_no}.pkl', 'wb') as pk:
        pickle.dump(dec_shares, pk)
        
    return
  
if __name__ == "__main__":
    sk = read_sk()
    csum1 = read_csum1()
    ds_helper(csum1, sk)