# PCA
import numpy as np
from sklearn.decomposition import PCA

class PCA_user():
    """
    基于PCA理论一步一步推到
    """
    def __init__(self, K):
        '''
        K:目标维度
        '''
        self.K = K

    def do_PCA(self,X):
        num = X.shape[0]
        X0 = X - X.mean(axis = 0)  # 数据0均值中心化
        self.cov = np.dot(X0.T,X0)/num  # 协方差
        eig_values,eig_vectors = np.linalg.eig(self.cov)    # 特征值，特征向量
        index = np.argsort(-eig_values) # 排序
        self.trans = eig_vectors[:,index[:self.K]]
        return np.dot(X0,self.trans)

X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4

pca_user = PCA_user(2)
ret=pca_user.do_PCA(X)
print(ret)
print('='*30)

# 调用sklearn封装好的方法
pca = PCA(n_components=2)   #降到2维
pca.fit(X)                  #训练
newX=pca.fit_transform(X)   #降维后的数据
print(pca.explained_variance_ratio_)  #输出贡献率
print(newX)                  #输出降维后的数据
"""
比较结果，为什么有一组成分差一个负号？
结合PCA原理，确保方差最大化，一组成分差一个负号，方差一样，应该是特征值差一个负号引起（特征值可以不唯一）
"""
