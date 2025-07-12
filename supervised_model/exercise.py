import tensorflow as tf
import numpy as np

I3 = tf.Variable([[1,0,0],[0,1,0],[0,0,1]])
print(I3)

u = tf.Variable([2,5,-3])
u2 = tf.Variable([0,-4,6])
B = tf.Variable([[2,0,-1],[-2,3,1],[0,4,-1]])

print(tf.linalg.matvec(I3,u))
print(tf.linalg.matvec(B,u))

u_col = tf.reshape(u, [-1, 1])   # shape (3, 1) -1 means “infer this dimension automatically based on the number of elements.”
u2_col = tf.reshape(u2, [-1, 1]) # shape (3, 1)
matrix = tf.concat([u_col, u2_col], axis=1)
print(matrix)
print (tf.matmul(B,matrix))