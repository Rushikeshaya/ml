import numpy as np
# Scalar
x = np.array(6)
print ("x: ", x)
print ("x ndim: ", x.ndim) # number of dimensions
print ("x shape:", x.shape) # dimensions
print ("x size: ", x.size) # size of elements
print ("x dtype: ", x.dtype) # data type
# Vector
x = np.array([1.3 , 2.2 , 1.7])
print ("x: ", x)
print ("x ndim: ", x.ndim)
print ("x shape:", x.shape)
print ("x size: ", x.size)
print ("x dtype: ", x.dtype) # notice the float datatype
# Matrix
x = np.array([[1,2], [3,4]])
print ("x:\n", x)
print ("x ndim: ", x.ndim)
print ("x shape:", x.shape)
print ("x size: ", x.size)
print ("x dtype: ", x.dtype)
# 3-D Tensor
x = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print ("x:\n", x)
print ("x ndim: ", x.ndim)
print ("x shape:", x.shape)
print ("x size: ", x.size)
print ("x dtype: ", x.dtype)
# Functions
print ("np.zeros((2,2)):\n", np.zeros((2,2)))
print ("np.ones((2,2)):\n", np.ones((2,2)))
print ("np.eye((2)):\n", np.eye((2))) # identity matrix
print ("np.random.random((2,2)):\n", np.random.random((2,2)))
# Indexing
x = np.array([1, 2, 3])
print ("x: ", x)
print ("x[0]: ", x[0])
x[0] = 0
print ("x: ", x)
# Slicing
x = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print (x)
print ("x column 1: ", x[:, 1])
print ("x row 0: ", x[0, :])
print ("x rows 0,1 & cols 1,2: \n", x[0:2, 1:3])
# Integer array indexing
print (x)
rows_to_get = np.array([0, 1, 2])
print ("rows_to_get: ", rows_to_get)
cols_to_get = np.array([0, 2, 1])
print ("cols_to_get: ", cols_to_get)
# Combine sequences above to get values to get
print ("indexed values: ", x[rows_to_get, cols_to_get]) # (0, 0), (1, 2), (2, 1)
# Boolean array indexing
x = np.array([[1, 2], [3, 4], [5, 6]])
print ("x:\n", x)
print ("x > 2:\n", x > 2)
print ("x[x > 2]:\n", x[x > 2])
print(np.mean(x[x>2]));#calculate mean of no's above 2
# Basic math
x = np.array([[1,2], [3,4]], dtype=np.float64)
y = np.array([[1,2], [3,4]], dtype=np.float64)
print ("x + y:\n", np.add(x, y)) # or x + y
print ("x - y:\n", np.subtract(x, y)) # or x - y
print ("x * y:\n", np.multiply(x, y)) # or x * y # simple multiplication
# Dot product
a = np.array([[1,2,3], [4,5,6]], dtype=np.float64) # we can specify dtype
b = np.array([[7,8], [9,10], [11, 12]], dtype=np.float64)
c = a.dot(b)
print (f"{a.shape} · {b.shape} = {c.shape}")
print (c)

# Sum across a dimension
x = np.array([[1,2],[3,4]])
print (x)
print ("sum all: ", np.sum(x)) # adds all elements
print ("sum axis=0: ", np.sum(x, axis=0)) # sum across rows
print ("sum axis=1: ", np.sum(x, axis=1)) # sum across columns

# Min/max
x = np.array([[1,2,3], [4,5,6]])
print ("min: ", x.min())
print ("max: ", x.max())
print ("min axis=0: ", x.min(axis=0))
print("------")
print ("min axis=1: ", x.min(axis=1))
print("Filtered rows:\n", x[x.min(axis=1) > 2])
print("------")
print(x[x.min(axis=1) > 2].min())
# Broadcasting
x = np.array([1,2]) # vector
y = np.array(3) # scalar
z = x + y
print ("z:\n", z)
# 1. Creates a 1D array
a = np.array((3, 4, 5))  
# a.shape is (3,)

# 2. Creates a NEW 2D array by adding a dimension to 'a'
b = np.expand_dims(a, axis=1)  
# b.shape is (3, 1) -> This leaves 'a' completely unchanged.

# 3. Triggers NumPy broadcasting
c = a + b  
# c.shape is (3, 3)
a.shape # (3,)
b.shape # (3, 1)
c.shape # (3, 3)
print (c)
a = a.reshape(-1, 1)
a.shape # (3, 1)
c = a + b
c.shape # (3, 1)
print (c)
a = np.array([3, 4, 5])
a.shape # (3,)
a = a.reshape(-1, 1)
a.shape # (3, 1)

# Transposing
x = np.array([[1,2,3], [4,5,6]])
print ("x:\n", x)
print ("x.shape: ", x.shape)
y = np.transpose(x, (1,0)) # flip dimensions at index 0 and 1//(row=0, col = 1) swapping row with col  --- in 3d (0-> pages,1-> rows,2-> col)
print ("y:\n", y)
print ("y.shape: ", y.shape)

# Reshaping
x = np.array([[1,2,3,4,5,6]])
print (x)
print ("x.shape: ", x.shape)
y = np.reshape(x, (2, 3))
print ("y: \n", y)
print ("y.shape: ", y.shape)
z = np.reshape(x, (2, -1))
print ("z: \n", z)
print ("z.shape: ", z.shape)

# Reshaping
x = np.array([[1,2,3,4,5,6]])
print (x)
print ("x.shape: ", x.shape)
y = np.reshape(x, (2, 3))
print ("y: \n", y)
print ("y.shape: ", y.shape)
z = np.reshape(x, (2, -1))
print ("z: \n", z)
print ("z.shape: ", z.shape)

x = np.random.random((2, 3))
print (x)
print (x.shape)
# Concatenation
y = np.concatenate([x, x], axis=0) # concat on a specified axis
print (y)
print (y.shape)
print('-----')
y = np.concatenate([x, x], axis=1) # concat on a specified axis
print (y)
print (y.shape)

# Stacking
z = np.stack([x, x], axis=0) # stack on new axis
print (z)
print (z.shape)
print("---")
z = np.stack([x, x], axis=1) # stack on new axis
print (z)
print (z.shape)