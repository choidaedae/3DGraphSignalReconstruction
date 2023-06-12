import scipy.io
import numpy as np
import itertools
from scipy.sparse.csgraph import laplacian
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# load files
mat = scipy.io.loadmat('A.mat')
y_100 = scipy.io.loadmat('y_100.mat')
y_500 = scipy.io.loadmat('y_500.mat')
y_1000 =scipy.io.loadmat('y_1000.mat')

# transform .mat to np.array 
def mattoA(mat):
    A = mat['A'].astype(np.int16)
    return A

def ytow(y):
    w1 = y['w1']
    w2 = y['w2']
    w3 = y['w3']
    return w1, w2, w3

def ytoy(y):
    y1 = y['y1']
    y2 = y['y2']
    y3 = y['y3']
    return y1, y2, y3

# 1. Construct Degree Matrix & Graph Laplacian 
def AtoL(A, N):
    #return laplacian(A)
    D = np.zeros((N, N))
    d = np.zeros((N, N))
    D = np.diag(np.sum(A, axis=1))
    d = np.sqrt(D)
    L = laplacian(A)
    L = np.matmul(np.matmul(d, L), d)
    return L    

# 2. set hyperparameters
gamma = 10
N = 3400

# 3. implements
def sampling_matrix(w, N):
    print(len(w))
    M = []
    for j in range(len(w)):
        temp = []
        for i in range(N):
            if (i == w[j]): 
                temp.append(1)
            else:
                temp.append(0)
            M.append(temp)
    #print(M.shape)
    return M

def recovery(y, gamma, M, L):
    g = np.matmul(np.linalg.inv(np.matmul(np.transpose(M), M) + gamma*L), (np.matmul(np.transpose(M), y)))
    return g 

# 4. function call
A = mattoA(mat)
L = AtoL(A, N)
w1, w2, w3 = ytow(y_100) 
y1, y2, y3 = ytoy(y_100)
M1 = sampling_matrix(w1, N)
M2 = sampling_matrix(w2, N)
M3 = sampling_matrix(w3, N)
gx = recovery(y1, gamma, M1, L)
gy = recovery(y2, gamma, M2, L)
gz = recovery(y3, gamma, M3, L)

# 5. 3D Scatter plot
#fig = plt.figure(figsize=(6, 6))
#recovered = fig.add_subplot(111, projection='3d')
#plt.scatter(gx, gy, gz)
#plt.title('Recovered Signal')
#plt.show()
    
