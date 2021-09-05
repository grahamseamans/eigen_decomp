import numpy as np
import matplotlib.pyplot as plt

# makes our square matrix
np.random.seed(1234)
n = 4
high = 100
A = np.random.randint(1, high, n ** 2)
A = np.reshape(A, (n, n))

# get the eigenvectors and eigenvalues
w, v = np.linalg.eig(A)
# real only
w = np.real(w)
v = np.real(v)
# sorts them in descending order
idx = np.flip(np.argsort(w))
w = w[idx]
v = v[:, idx]
# makes the eigen values into a diagonal
w = np.diag(w)
# gets inverse of our eigenvectors (after sorting)
v_inv = np.linalg.inv(v)
# tesing...
composed = v @ w @ v_inv
assert np.allclose(A, composed)

# gets our 'compressed' eigendecomp
comp_level = 1
assert comp_level > 0 and comp_level <= n
# ((n,c) @ (c,c)) @ (c,n)
# (n,c) @ (c,n)
# (n,n)
v_comp = v[:, :comp_level]
w_comp = w[:comp_level, :comp_level]
v_inv_comp = v_inv[:comp_level, :]
compressed = v_comp @ w_comp @ v_inv_comp
# easier to read test results
compressed = compressed.astype(int)

print("original")
print(A)
print("compressed")
print(compressed)

test = np.random.randint(1, high, n)
print("test vector to multiply", test)
transformed_test = A @ test
print("uncompressed transform", transformed_test)
comp_transormed_test = compressed @ test
print("compressed transform", comp_transormed_test)
print("diff", transformed_test - comp_transormed_test)
