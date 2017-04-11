import numpy as np
from model import PDM, PGM
from bmp_readin import save_item,get_item
from PCA_convert import reduce_simpliest_PCA
from draw import draw_decision_region

# data_size upto 3000
s = {"map_size":1081,"dim":2,"data_size":2400,
    "batch_size":1,"iter":10,"k_folder":0,"PCA_reset":1,
    "x_train":"../tmp/data_based.npy","t_train":"../tmp/data_target.npy"}

def model_init():
    global Ground_x,Ground_t,settings
    Ground_x = get_item(s["x_train"])
    Ground_t = get_item(s["t_train"])
    Ground_x = np.array(Ground_x)
    Ground_t = np.array(Ground_t)
    settings = s
    print('Finish initializing')

def model_setting(k):
    global Ground_x,Ground_t,Train_x,Train_t,Test_x,Test_t
    n = len(Ground_x)
    indices = np.asarray(range(n), dtype=np.int32)
    n_train = s['data_size']
    np.random.shuffle(indices)
    if s['k_folder'] == 0:
        Train_x = Ground_x[indices[:n_train]]
        Train_t = Ground_t[indices[:n_train]]
        Test_x = Ground_x[indices[n_train:]]
        Test_t = Ground_t[indices[n_train:]]
        print('test data size:',Test_t.shape)
    else:
        if(k>s['k_folder']):
            return -1
        k_size = int(s['data_size'] / s['k_folder'])
        Train_x = Ground_x[k_size*(k-1) :k_size*(k)]
        Train_t = Ground_t[k_size*(k-1) :k_size*(k)]
        if(k==s['k_folder']):
            Test_x = Ground_x[0:k_size*(1)]
            Test_t = Ground_t[0:k_size*(1)]
        else:
            Test_x = Ground_x[k_size*(k):k_size*(k+1)]
            Test_t = Ground_t[k_size*(k):k_size*(k+1)]
        print(k,':',Train_t.shape,Test_t.shape)
    print('Train_x',Train_x.shape)
    print('Train_t',Train_t.shape)
    return 0

if(s["PCA_reset"]> 0):
    reduce_simpliest_PCA(s["dim"])

model_init()

if(s['k_folder']==0):
    model_setting(0)

    pdm = PDM()
    pdm.run(Train_x, Train_t,s['iter'])
    e1=pdm.eval(Test_x, Test_t)

    pgm = PGM()
    pgm.run(Train_x, Train_t)
    e2=pgm.eval(Test_x, Test_t)

    draw_decision_region(Test_x,Test_t,[pdm,pgm],[e1,e2])
# else:
#     print('K-folder is not implemented')
