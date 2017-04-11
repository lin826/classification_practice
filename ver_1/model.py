import numpy as np

def cal_softmax(a):
    e = np.exp(a)
    return e / e.sum(axis=1)[:, np.newaxis]

class PGM:
    def __init__(self,x,t,test_x,test_t):
        m = x.shape[1] # 2
        k = t.shape[1] # 3
        n = x.shape[0]
        mean = np.zeros([k,m])
        sig = np.zeros([m, m])
        p = np.zeros([k,1])

        for j in range(k):
            qualified_data = x[t[:, j] == 1]
            p[j] = len(qualified_data) / n
            mean[j] = np.mean(qualified_data, axis=0)
            dif = qualified_data - mean[j]
            sig += p[j] * dif.T.dot(dif) / len(qualified_data)

        sig_ = np.linalg.pinv(sig)

        self.w = np.zeros([k,m])
        self.om =np.zeros([k,1])
        for i in range(k):
            self.w[i] = np.dot(sig_,mean[i])
            self.om[i] = (-1.0/2)*mean[i].T.dot(sig_.dot(mean[i])) + np.log(p[i])[0]
        self.get_result(x,t)

        y = cal_softmax(test_x.dot(self.w.T) + np.tile(self.om.T,(len(test_x),1)))
        acc = float(np.equal(y.argmax(axis=1), test_t.argmax(axis=1)).sum()) / len(test_t)
        print ('Accuracy: ', acc)

    def get_result(self,x,t):
        n = len(x)
        k = t.shape[1]
        # y = np.zeros([n,k])
        y = cal_softmax(x.dot(self.w.T) + np.tile(self.om.T,(n,1)))
        acc = float(np.equal(y.argmax(axis=1), t.argmax(axis=1)).sum()) / len(t)
        print ('acc', acc)

class PDM_basic:
    def __init__(self):
        pass

    def cal_activations(self, p):
        a = p.dot(self.W.T)
        return a

    def get_phi(self,x):
        bias = np.ones([len(x), 1])
        return np.hstack((bias, x))
        #return x

    def update(self):
        M = self.M
        K = self.K
        N = len(self.phi)
        x = self.phi
        # Set y
        self.y = cal_softmax(self.phi.dot(self.W.T))

        # # Calculate gradient
        # grad = np.zeros([K, M])
        # for j in range(K):
        #     for n in range(N):
        #         grad[j, :] += (self.y[n,j] - self.t[n,j]) * x[n]
        # self.gradient = grad.flatten()
        #
        # # Calculate Hessian matrix
        # I = np.identity(K)
        # H = np.zeros([K*M, K*M])
        # for j in range(K):
        #     for k in range(K):
        #         D_wjk = np.zeros([M, M])
        #         for n in range(N):
        #             D_wjk += self.y[n,k] * (I[k,j] - self.y[n,j]) * x.T.dot(x)
        #         H[j*(M):(j+1)*M, (k)*M:(k+1)*M] = D_wjk
        # self.hessian = H

        # Set gradient
        E = np.zeros((K, M))
        for j in range(K):
            for i in range(len(self.phi)):
                dis = (self.y[i,j] - self.t[i,j]) * self.phi[i]
                E[:,j] += dis
        self.gradient = E.T.flatten()

        # Set Hessian
        H = np.zeros([M*K,M*K])
        I = np.identity(K)
        for j in range(K):
            for k in range(K):
                item = np.zeros([M, M])
                for n in range(len(self.phi)):
                    item += self.y[n,k]* (I[k,j]- self.y[n,j]) * x.T.dot(x)
                H[ j*M: (j+1)*M , k*M: (k+1)*M] = item
        self.hessian = H

        H_ = np.linalg.pinv(self.hessian)
        w_old = self.W.flatten()
        w_new = w_old - np.dot(H_,self.gradient)
        self.W = w_new.reshape([self.K, self.M])

    def do_IRLS(self,times,x,t):
        self.M = len(x[0])+1
        self.K = len(t[0])
        self.W = np.zeros((self.K , self.M))
        N = len(x)

        batch_size = 256
        num_each = int(N/batch_size)
        indices = [i for i in range(N)]
        for i in range(times):
            np.random.shuffle(indices)
            for begin in range(0, N, batch_size):
                end = min(N, begin + batch_size)
                self.phi = self.get_phi(x[indices[begin:end]])
                self.t = t[indices[begin:end]]
                self.update()

            phi = self.get_phi(x)
            y = cal_softmax(phi.dot(self.W.T))
            acc = float(np.equal(y.argmax(axis=1), t.argmax(axis=1)).sum()) / len(t)
            print ('acc', acc)

class PDM:
    def __init__(self,x,t,test_x,test_t):
        self.basic = PDM_basic()
        self.basic.do_IRLS(10,x,t)

        phi = self.get_phi(test_x)
        y = cal_softmax(phi.dot(self.W.T))
        acc = float(np.equal(y.argmax(axis=1), test_t.argmax(axis=1)).sum()) / len(t)
        print ('Accuracy: ', acc)
