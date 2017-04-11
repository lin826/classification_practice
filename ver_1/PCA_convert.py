import numpy as np
from bmp_readin import get_data,save_item,get_item
from numpy import linalg


def reduce_simpliest_PCA(i):
    global DATA, TARGET_DIM
    DATA = get_data()
    TARGET_DIM = int(i)
    print("Start PCA...")
    # formulate_target()
    get_eigenvector()

def get_eigenvector():
    # # Get data X
    # ori_dim = DATA.shape[2]
    # X = DATA.reshape([-1,ori_dim])
    #
    # # Get covariance matrix
    # S = np.cov(X.T)
    #
    # # Get eigenvalues and eigenvectors
    # w, v = linalg.eig(S)
    # save_item(w,"../tmp/w_eigenvalue.npy")
    # save_item(v,"../tmp/v_eigenvactors.npy")
    w = get_item("../tmp/w_eigenvalue.npy")
    v = get_item("../tmp/v_eigenvactors.npy")

    # Get the most important ev
    x_pair = [(np.abs(w[i]), v[:,i]) for i in range(len(w))]
    x_pair.sort(key=lambda x: x[0], reverse=True)
    x_vector = np.array([x_pair[i][1] for i in range(TARGET_DIM)]).transpose()
    save_item(x_vector,"../tmp/x_vector.npy")

    # # Normalize the result
    # X = X.dot(x_vector)
    # X = normalize(X)
    # save_item(X,'../tmp/data_based.npy')
def normalize(v):
    norm=np.linalg.norm(v)
    std_dev = np.std(v, axis=0)
    zero_std_mask = std_dev == 0
    if zero_std_mask.any():
        std_dev[zero_std_mask] = 1.0
        # warnings.warn("Some columns have standard deviation zero. "
        #               "The values of these columns will not change.",
        #               RuntimeWarning)
    return v / std_dev

def formulate_target():
    N = DATA.shape[0]* DATA.shape[1]
    target_list = list()
    for a in range(len(DATA)):
        target = np.array([0,0,0])
        for item in DATA[a]:
            target_list.append(target)
    target_list = np.array(target_list)
    save_item(target_list,'../tmp/data_target.npy')

if __name__ == '__main__':
    reduce_simpliest_PCA(sys.argv[1])
