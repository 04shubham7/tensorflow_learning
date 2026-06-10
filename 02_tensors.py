#nd-array
#GPU support
#computational graph/Backpropagation

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #to hide warnings


import tensorflow as tf

# x=tf.constant(4,shape=(1,1),dtype=tf.float32)

#rank-1 Tensor
#x=tf.constant([1,2,3],dtype=tf.float32)

#rank-2 Tensor
#x=tf.constant([[1,2,3],[4,5,6]],dtype=tf.float32)

#x=tf.ones((3,3))
#x=tf.zeros((3,3))

#x=tf.eye(3) #identity matrix

#x=tf.random.normal((3,3),mean=0,stddev=1) #normal distribution
#x=tf.random.uniform((3,3),minval=0,maxval=1) #uniform distribution

#x=tf.range(10)

#CAST
#x=tf.cast(tf.range(10),dtype=tf.float32)
#print(x)

#all operations are elementwise

# x=tf.constant([1,2,3])
# y=tf.constant([4,5,6])
# #z=tf.add(x,y) #z=x+y/tf.subtract(x,y) #z=x-y/tf.multiply(x,y) #z=x*y/tf.divide(x,y) #z=x/y
# #z=tf.tensordot(x,y, axes=1) #dot product

# z=x**3


# #matrix multiplication(no of coloumns of 1st matrix should be equal to no of rows of 2nd matrix)
# x=tf.random.normal((2,2))
# y=tf.random.normal((2,2))
# z=tf.matmul(x,y)
# z=x@y #shortcut for matrix multiplication
# print(z)

#Slicing And Indexing
# x=tf.constant([[1,2,3,4],[5,6,7,8]])
# print(x[0]) #output: [1 2 3 4] prints first row
# print(x[:,0]) #output: [1 5] prints first column
# print(x[0,:]) #output: [1 2 3 4] prints first row
# print(x[0,0]) #output: 1 prints first element of first row
# print(x[0,1]) #output: 2 prints second element of first row
# print(x[0,1:3]) #output: [2 3] prints elements at indices 1 and 2 of first row

#Reshaping of Tensors
# x=tf.random.normal((2,3))
# print(x)

# x=tf.reshape(x,(3,2))
# print(x)

#Converting to Numpy array
# x=tf.random.normal((2,3))
# print(x)
# x=x.numpy()
# print(x)
# print(type(x))
# x=tf.convert_to_tensor(x)
# print(type(x))

#we can also use string tensors
# x=tf.constant(["Hello World","Shubham"])
# print(x)

#variable
# x=tf.Variable([1,2,3],dtype=tf.float32)
# print(x)