import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch

x_tf = tf.Variable(25, dtype=tf.int16)
y_tf = tf.Variable(3, dtype=tf.int16)
print(x_tf)
print(x_tf.shape)
print(tf.add(x_tf,y_tf))

# Suppose you want to represent this sparse matrix:
# [[0, 0, 1],
#  [0, 2, 0]]

sparse_tensor = tf.SparseTensor(
    indices=[[0, 2], [1, 1]],    # positions of non-zero entries
    values=[1, 2],               # values at those positions
    dense_shape=[2, 3]           # shape of the full matrix
)

# Convert to dense for visualization
dense = tf.sparse.to_dense(sparse_tensor)
print(dense.numpy())
"""tf.function converts a Python function into a TensorFlow graph, 
which makes it faster and more optimized for repeated execution."""
@tf.function
def multiply_tensors(x, y):
    return x * y

# Define some input tensors
a = tf.constant([2.0, 3.0, 4.0])
b = tf.constant([5.0, 6.0, 7.0])

# Call the function
result = multiply_tensors(a, b)

print(result)
# L2 Norms = unit vector => L2 norm = 1 =>||x|| = 1
norm_array = np.array([25,2,5])
print(np.linalg.norm(norm_array))

# L1 Norms
l1_norm = np.array([25,2,5])
print(np.abs(l1_norm[0])+np.abs(l1_norm[1])+np.abs(l1_norm[2]))

#The maximum norm, also known as the L-infinity norm or ∞-norm, is a way of measuring the "size" or "length" of a vector by taking the largest absolute value among its components.
#torch.norm(X_pt),tf.norm(X_tf)
x = np.array([3, -7, 2])
max_norm = np.linalg.norm(x, ord=np.inf) 
print(max_norm)  # Output: 7.0

X = np.array([[25, 2], [5, 26], [3, 7]])
print(X.shape)
# Select left column of matrix X (zero-indexed)
print(X[:,0])
# Select middle row of matrix X:
print(X[1,:])
# Another slicing-by-index example:
print(X[0:1, 0:2])
print(X.sum())
print(tf.reduce_sum(X))

# Can also be done along one specific axis alone, e.g.:
print(X.sum(axis=0)) # summing over all rows (i.e., along columns) torch.sum(X_pt, 0)
print(X.sum(axis=1)) # summing over all columns (i.e., along rows) tf.reduce_sum(X_tf, 1)

#Solving Linear Systems 2x-3y = 15,4x+10y =15
y1 = -5 + (2*x)/3
y2 = (7-2*x)/5
fig,ax = plt.subplots()
plt.xlabel('x')
plt.ylabel('y')
# Add x and y axes:
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')
ax.set_xlim([-2, 10])
ax.set_ylim([-6, 4])
ax.plot(x, y1, c='green')
ax.plot(x, y2, c='brown')
plt.axvline(x=6, color='purple', linestyle='--')
_ = plt.axhline(y=-1, color='purple', linestyle='--')


#The Frobenius norm is a way to measure the “size” or “magnitude” of a matrix — similar to how you measure the length of a vector.
mul =  np.array([[1, 2], [3, 4]])
np.linalg.norm(mul) # same function as for vector L2 norm
X_pt = torch.tensor([[1, 2], [3, 4.]]) # torch.norm() supports floats only
torch.norm(X_pt)
X_tf = tf.Variable([[1, 2], [3, 4.]]) # tf.norm() also supports floats only
tf.norm(X_tf)

#Matrix Multiplication (with a Vector)
A = np.array([[3, 4], [5, 6], [7, 8]])
b = np.array([1, 2])
np.dot(A, b) # even though technically dot products are between vectors only

A_pt = torch.tensor([[3, 4], [5, 6], [7, 8]])
b_pt = torch.tensor([1, 2])
#np.dot(A_pt, b_pt),tf.linalg.matvec(A_tf, b_tf)
print(torch.matmul(A_pt, b_pt))

A_tf = tf.Variable([[3, 4], [5, 6], [7, 8]])
b_tf = tf.Variable([1, 2])
tf.linalg.matvec(A_tf, b_tf)

#Matrix Multiplication (with Two Matrices)
B = np.array([[1, 9], [2, 0]])
np.dot(A, B)

B_pt = torch.from_numpy(B) # much cleaner than TF conversion
# another neat way to create the same tensor with transposition:
# B_pt = torch.tensor([[1, 2], [9, 0]]).T
torch.matmul(A_pt, B_pt) # no need to change functions, unlike in TF
B_tf = tf.convert_to_tensor(B, dtype=tf.int32)
tf.matmul(A_tf, B_tf)

#Symmetric Matrices
X_sym = np.array([[0, 1, 2], [1, 7, 8], [2, 8, 9]])
print(X_sym.T)
X_sym.T == X_sym

#Identity Matrices
I = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
x_pt = torch.tensor([25, 2, 5])
torch.matmul(I, x_pt)

#Matrix Inversion XX-1 = I,y= Xw (w is the vector of weights),w = X-1 y
X = np.array([[4, 2], [-5, -3]])
Xinv = np.linalg.inv(X)
#solving for the unknowns in  𝑤 :
y = np.array([4, -7])
w = np.dot(Xinv, y)
#In PyTorch and TensorFlow:
torch.inverse(torch.tensor([[4, 2], [-5, -3.]])) # float type
tf.linalg.inv(tf.Variable([[4, 2], [-5, -3.]])) # also float

#matrix  𝐼3  has mutually orthogonal columns, we show that the dot product of any pair of columns is zero:
I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
column_1 = I[:,0]
column_2 = I[:,1]
column_3 = I[:,2]
np.dot(column_1, column_2) #0
np.dot(column_1, column_3)
np.dot(column_2, column_3)
#each column of  𝐼3  has unit norm
np.linalg.norm(column_1)#1

# Since the matrix  𝐼3  has mutually orthogonal columns and each column has unit norm, the column vectors of  𝐼3  are orthonormal. Since  𝐼𝑇3=𝐼3 , this means that the rows of  𝐼3  must also be orthonormal.

# Since the columns and rows of  𝐼3  are orthonormal,  𝐼3  is an orthogonal matrix.
#Instead of manually inspecting the matrix, use:
K = torch.tensor([[2/3, 1/3, 2/3], [-2/3, 2/3, 1/3], [1/3, 2/3, -2/3]])
torch.allclose(K.T @ K, torch.eye(3), atol=1e-6)
# @ - Matrix multiplication (same as .matmul() in NumPy or PyTorch)
# * - Element-wise multiplication (each element is multiplied individually)
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
A @ B  # Matrix multiplication
A * B  # Element-wise multiplication
