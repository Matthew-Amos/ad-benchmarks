using Zygote
using BenchmarkTools

fn_scalar(a) = 3a ^ 10 - 5
fn_vector(a) = sum(3 .* a .^ 10 .- 5)
fn_vector_vector(a) = sum(a .* a)
fn_matrix(a) = sum(3 .* a .^ 10 .- 5)
fn_matrix_matrix(a) = sum(a * a')

val_scalar = 100.
val_vector = rand(100)
val_matrix = rand(10, 10)

# Precompile
fn_scalar(val_scalar)
fn_vector(val_vector)
fn_vector_vector(val_vector)
fn_matrix(val_matrix)
fn_matrix_matrix(val_matrix)

gradient(fn_scalar, val_scalar)
gradient(fn_vector, val_vector)
gradient(fn_vector_vector, val_vector)
gradient(fn_matrix, val_matrix)
gradient(fn_matrix_matrix, val_matrix)

# Benchmark
res_1 = @benchmark gradient(fn_scalar, val_scalar)
res_2 = @benchmark gradient(fn_vector, val_vector)
res_3 = @benchmark gradient(fn_vector_vector, val_vector)
res_4 = @benchmark gradient(fn_matrix, val_matrix)
res_5 = @benchmark gradient(fn_matrix_matrix, val_matrix)

open("zygote.txt", "w") do file
    write(file, string("Scalar: ", res_1, "\n"))
    write(file, string("Vector: ", res_2, "\n"))
    write(file, string("Vector Vector: ", res_3, "\n"))
    write(file, string("Matrix: ", res_4, "\n"))
    write(file, string("Matrix Matrix: ", res_5, "\n"))
end
