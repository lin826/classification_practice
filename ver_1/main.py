import sys
import csv
import numpy as np
from model import PDM, PGM
from bmp_readin import save_item,get_item
from PCA_convert import reduce_simpliest_PCA,normalize
from draw import draw_decision_region

# data_size upto 3000
s = {"map_size":1081,"dim":2,"data_size":1500,"class":3,
    "batch_size":1,"iter":10,"unbalanced":1,"PCA_reset":1,
    "x_train":"../tmp/data_based.npy","t_train":"../tmp/data_target.npy",
    "x_demo":"../tmp/data_demo.npy"}

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
    if s['unbalanced'] == 0:
        n = len(Ground_x)
        indices = np.asarray(range(n), dtype=np.int32)
        n_train = s['data_size']
        np.random.shuffle(indices)
        Train_x = Ground_x[indices[:n_train]]
        Train_t = Ground_t[indices[:n_train]]
        Test_x = Ground_x[indices[n_train:]]
        Test_t = Ground_t[indices[n_train:]]
        print('test data size:',Test_t.shape)
    else: # unbalanced data set
        m = int(len(Ground_x)/s['class']) # 1000
        indices = np.asarray(range(m), dtype=np.int32)
        n_train = [800,350,350]
        np.random.shuffle(indices)
        Train_x = Ground_x[indices[:n_train[0]]]
        Train_t = Ground_t[indices[:n_train[0]]]
        Test_x = Ground_x[indices[n_train[0]:m]]
        Test_t = Ground_t[indices[n_train[0]:m]]
        for i in range(1,s['class']):
            np.random.shuffle(indices)
            Train_x = np.vstack((Ground_x[i*m+indices[:n_train[i]]],Train_x))
            Train_t = np.vstack((Ground_t[i*m+indices[:n_train[i]]],Train_t))
            Test_x = np.vstack((Ground_x[i*m+indices[n_train[i]:]],Test_x))
            Test_t = np.vstack((Ground_t[i*m+indices[n_train[i]:]],Test_t))
        print('test data size:',Test_t.shape)


# Main() where the start of program
if __name__ == "__main__":
    arg = sys.argv[1:]
    for i in range(len(arg)-1):
        if(arg[i].startswith("--")):
            opt = arg[i][2:]
            main_opt(opt,arg[i+1])

    if s["PCA_reset"]>0:
        reduce_simpliest_PCA(s['dim'])
    model_init()

    model_setting(0)

    pdm = PDM()
    pdm.run(Train_x, Train_t,s['iter'])
    e1=pdm.eval(Test_x, Test_t)

    pgm = PGM()
    pgm.run(Train_x, Train_t)
    e2=pgm.eval(Test_x, Test_t)

    x = get_item(s["x_demo"])
    print(x.shape)
    print(get_item("../tmp/x_vector.npy").shape)
    x = x.dot(get_item("../tmp/x_vector.npy"))
    x = normalize(x)
    print(x.shape)
    t = pgm.predict(x)
    with open('DemoTarget.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(len(t)):
            it = [0,0,0]
            it[t[i].argmax()] = 1
            writer.writerow(it)

    # draw_decision_region(Test_x,Test_t,[pdm,pgm],[e1,e2])
