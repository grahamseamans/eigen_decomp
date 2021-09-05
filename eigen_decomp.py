import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)
n = 4
high = 100
A = np.random.randint(1, high, n ** 2)
A = np.reshape(A, (n, n))

w, v = np.linalg.eig(A)
w = np.real(w)
v = np.real(v)
idx = np.flip(np.argsort(w))
w = w[idx]
v = v[:, idx]
w = np.diag(w)
v_inv = np.linalg.inv(v)
composed = v @ w @ v_inv
assert np.allclose(A, composed)

test = np.random.randint(1, high, n)

comp_level = 3
v_comp = v[:, :comp_level]
w_comp = w[:comp_level, :comp_level]
v_inv_comp = v_inv[:comp_level, :]
compressed = v_comp @ w_comp @ v_inv_comp
compressed = compressed.astype(int)

print("original")
print(A)
print("compressed")
print(compressed.astype(int))

print("test vector to multiply", test)
transformed_test = A @ test
print("uncompressed transform", transformed_test)
comp_transormed_test = compressed @ test
print("compressed transform", comp_transormed_test)
print("diff", transformed_test - comp_transormed_test)
