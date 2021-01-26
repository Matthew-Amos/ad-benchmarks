import jax.numpy as np
from numpy.random import rand
from jax import grad, jit
import timeit

@jit
def fn_scalar(a):
    return 3*a**10 - 5

@jit
def fn_vector(a):
    return np.sum(3*a**10 - 5)

@jit
def fn_vector_vector(a):
    return np.sum(a * a)

@jit
def fn_matrix(a):
    return np.sum(3*a**10 - 5)

@jit
def fn_matrix_matrix(a):
    return np.sum(np.matmul(a, a.transpose()))

val_scalar = 100.
val_vector = rand(100)
val_matrix = rand(10, 10)

g_scalar = grad(fn_scalar)
g_vector = grad(fn_vector)
g_vector_vector = grad(fn_vector_vector)
g_matrix = grad(fn_matrix)
g_matrix_matrix = grad(fn_matrix_matrix)

# Initial pass
g_scalar(val_scalar)
g_vector(val_vector)
g_vector_vector(val_vector)
g_matrix(val_matrix)
g_matrix_matrix(val_matrix)

# Benchmark
res_1 = timeit.timeit("g_scalar(val_scalar)", setup="from __main__ import g_scalar, val_scalar", number=10000)
res_2 = timeit.timeit("g_vector(val_vector)", setup="from __main__ import g_vector, val_vector", number=10000)
res_3 = timeit.timeit("g_vector_vector(val_vector)", setup="from __main__ import g_vector_vector, val_vector", number=10000)
res_4 = timeit.timeit("g_matrix(val_matrix)", setup="from __main__ import g_matrix, val_matrix", number=10000)
res_5 = timeit.timeit("g_matrix_matrix(val_matrix)", setup="from __main__ import g_matrix_matrix, val_matrix", number=10000)

with open("jax_jit.txt", "w") as f:
    f.write(f"Scalar: {res_1}\n")
    f.write(f"Vector: {res_2}\n")
    f.write(f"Vector Vector: {res_3}\n")
    f.write(f"Matrix: {res_4}\n")
    f.write(f"Matrix Matrix: {res_5}\n")
