import numpy as np
from scipy.special import expit
import sys

class NeuralNetMLP():
    """
    pass
    """
    def __init__(self, n_output, n_features, n_hidden=30,
                 l1=0.0, l2=0.0, epochs=500, eta=0.001,
                 alpha=0.0, decrease_const=0.0, shuffle=True,
                 minibatches=1, random_state=None):
        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.shuffle = shuffle
        self.minibatches = minibatches
        self.decrease_const = decrease_const

    def _encode_labels(self, y, k):
        # 将list y 转换为独热编码的数据矩阵
        onehot = np.zeros((k, y.shape[0])) # 每一列为一个数据
        for idx, value in enumerate(y):
            onehot[value, idx] = 1.0 # 将数字对应的行设置为1, idx对应列
        return onehot

    def _initialize_weights(self):
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden*(self.n_features+1))
        w1 = w1.reshape(self.n_hidden, self.n_features+1) # 第一层的权重矩阵(每个隐层单元对应f+1个权重)
        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output*(self.n_hidden+1))
        w2 = w2.reshape(self.n_output, self.n_hidden+1)
        return w1, w2

    def _sigmoid(self, z):
        # expit is equivalent to 1.0/(1.0 + np.exp(-z))
        return expit(z)

    def _sigmoid_gradient(self, z):
        # use ???
        sg = self._sigmoid(z)
        return sg * (1 - sg)

    def _add_bias_unit(self, X, how='column'):
        # 原始数据加上偏置
        if how == 'column': # 加一列偏置项（均为1）
            X_new = np.ones((X.shape[0], X.shape[1]+1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0]+1, X.shape[1]))
            X_new[1:,:] = X
        else:
            raise AttributeError()
        return X_new

    def _feedforward(self, X, w1, w2):
        a1 = self._add_bias_unit(X)
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3

    def _L2_reg(self, lambda_, w1, w2):
        # L2正则化
        return (lambda_/2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:,1:] ** 2))

    def _L1_reg(self, lambda_, w1, w2):
        # L1正则化
        return (lambda_/2.0) * (np.abs(w1[:, 1:]).sum() + np.abs(w2[:,1:]).sum())

    def _get_cost(self, y_enc, output, w1, w2):
        # 损失函数 -y*log(y_pred) - (1-y)*log(1-y_pred)
        term1 = -y_enc * (np.log(output))
        term2 = (1-y_enc) * np.log(1-output)
        cost = np.sum(term1-term2)
        L1_term = self._L1_reg(self.l1, w1, w2)
        L2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L1_term + L2_term
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        # ???
        # backpropagation(反向传播)
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:,:]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)

        # 标准化
        grad1[:, 1:] += (w1[:,1:] * (self.l1 + self.l2))
        grad2[:, 1:] += (w2[:,1:] * (self.l1 + self.l2))

        return grad1, grad2

    def predict(self, X):
        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)
        return y_pred

    def fit(self, X, y, print_progress=False):
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)
        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)
        for i in range(self.epochs):
            # adapative log
            self.eta /= (1+self.decrease_const*i)
            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                sys.stderr.flush()
            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_data = X_data[idx], y_data[idx]
            mini = np.array_split(range(
                y_data.shape[0]), self.minibatches
            ) # 将array的索引随机划分为几块
            for idx in mini:
                a1, z2, a2, z3, a3 = self._feedforward(X[idx], self.w1, self.w2)
                cost = self._get_cost(y_enc=y_enc[:,idx],
                                      output=a3,
                                      w1=self.w1,
                                      w2=self.w2)
                self.cost_.append(cost)
                # compute gradient via backpropagation
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2, a3=a3, z2=z2,
                                                  y_enc=y_enc[:, idx],
                                                  w1=self.w1, w2=self.w2)
                # update weights
                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2
        return  self

if __name__ == '__main__':
    nn = NeuralNetMLP(n_output=10,
                      n_features=784,
                      n_hidden=50,
                      l2=0.1,
                      l1=0.0,
                      epochs=1000,
                      eta=0.001,
                      alpha=0.001,
                      decrease_const=0.00001,
                      shuffle=True,
                      minibatches=50,
                      random_state=1)











