import numpy as np

def np_map(f,x):
  return np.array(map(f, np.ndarray.tolist(x)))

def np_tensor_map(f,x,y):
  m           = len(x)
  n           = len(y)
  eval_matrix = np.zeros((m,n))
  for i in range(0,m):
      for j in range(0,n):
          eval_matrix[i,j] = f(x[i],y[j])
  return eval_matrix


