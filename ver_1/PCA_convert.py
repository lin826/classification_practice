import numpy as np
from bmp_readin import get_data,save_item,get_item
from numpy import linalg

DATA = get_data()
NUM_CLASS = len(DATA)
NUM_TRAIN = [len(DATA[i]) for i in range(NUM_CLASS)]
TARGET_DIM = 2
def convert_to_X(data):
    tmp = list()
    for i in DATA: # Beyond Class
        for j in i: # get all pictures
            tmp.append([[a] for a in j])
    return np.array(tmp)

def get_covariance_matrix(X):
    N = len(X)
    # Get mean of x
    mean = X.mean()
    # Get covariance matrix
    S = (X[0]-mean)*np.transpose(X[0]-mean)
    for i in range(1,N):
        S = np.add((X[i]-mean)*np.transpose((X[i]-mean)), S)
    return S/N

def get_X():
    # X = convert_to_X(DATA)
    # save_item(X,"../tmp/X.npy")
    X = get_item("../tmp/X.npy")
    return X

def get_S():
    # S = get_covariance_matrix(X)
    # save_item(S,"../tmp/S_matrix.npy")
    S = get_item("../tmp/S_matrix.npy")
    return S

def get_eigenvector():
    # Get eigenvalues and eigenvectors
    # w, v = linalg.eig(S)
    # save_item(w,"../tmp/w_eigenvalue.npy")
    # save_item(v,"../tmp/v_eigenvactors.npy")
    w = get_item("../tmp/w_eigenvalue.npy")
    v = get_item("../tmp/v_eigenvactors.npy")
    # Get the most important two
    x_pair = [(np.abs(w[i]), v[:,i]) for i in range(len(w))]
    x_pair.sort(key=lambda x: x[0], reverse=True)
    x_vector = np.array([x_pair[0][1],x_pair[1][1]]).transpose()
    return x_vector

def reduce_dim(i,ev):
    i = np.array(i)
    output = [0 for k in range(TARGET_DIM)]
    for e in range(TARGET_DIM):
        eigen_vec = np.array([ev[:,e]]).transpose()
        output[e] = np.dot(i,eigen_vec)
    output = normalize(output)
    return output.flatten()

def normalize(v):
    norm=np.linalg.norm(v)
    std_dev = np.std(v, axis=0)
    zero_std_mask = std_dev == 0
    if zero_std_mask.any():
        std_dev[zero_std_mask] = 1.0
        warnings.warn("Some columns have standard deviation zero. "
                      "The values of these columns will not change.",
                      RuntimeWarning)
    return v / std_dev

def reduce_simpliest_PCA():
    ev = get_eigenvector()
    a = np.array([[[0.0 for k in range(TARGET_DIM)] for j in range(len(DATA[i]))] for i in range(len(DATA))])
    for i in range(len(DATA)): # Beyond Class
        for j in range(len(DATA[i])): # get all pictures
            a[i][j] = reduce_dim(DATA[i][j],ev)
    print(a.shape)
    save_item(a,"../tmp/DATA_reduced.npy")

def formulate_data(data):
    target_list = list()
    based_list = list()
    for a in range(len(data)):
        target = np.array([0,0,0])
        target[a] = 1
        for item in data[a]:
            target_list.append(target)
            based_list.append([i for i in item])
    based_list = np.array(based_list)
    target_list = np.array(target_list)
    save_item(based_list,'../tmp/data_based.npy')
    save_item(target_list,'../tmp/data_target.npy')

# reduce_simpliest_PCA()
# DATA = get_item("../tmp/DATA_reduced.npy")
# formulate_data(DATA)
