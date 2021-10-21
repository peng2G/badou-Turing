import numpy as np
import scipy.linalg as sl
import scipy as sp


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    输入：
        data - 样本点
        model - 假设模型：事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        threshold - 阈值
        min_num_data - 拟合较好时，需要的样本点最少个数，阈值

    输出：
        bestfit - 最优拟合解,有可能找不到
    """
    iterations = 0  # 实际迭代次数
    bestfit = None  # 当前迭代次数中最好的模型
    besterr = np.inf  # 当前迭代次数中最小误差
    best_inlier_idxs = None  # 局内点

    while iterations < k:
        if iterations % 500 == 0:
            print("第{}次迭代".format(iterations))
        # 随机抽取内群点
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybe_inliers = data[maybe_idxs, :]
        test_points = data[test_idxs, :]
        maybe_model = model.fit(maybe_inliers)
        test_err = model.get_errs(test_points, maybe_model)
        also_idxs = test_idxs[test_err < t]
        also_inliers = data[also_idxs, :]
        if debug:
            # 输出每次迭代信息
            print("test_err.min", test_err.min())

        if (len(also_inliers) > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            bettermodel = model.fit(betterdata)
            bettererrs = model.get_errs(betterdata, bettermodel)
            thiserr = np.mean(bettererrs)
            if thiserr < besterr:
                besterr = thiserr
                bestfit = bettermodel
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
        iterations += 1

    if bestfit is None:
        raise ValueError("Can not fit the criteria")

    if return_all:
        return bestfit, best_inlier_idxs
    else:
        return bestfit


def random_partition(n_choose, num_data):
    all_idxs = np.arange(num_data)
    np.random.shuffle(all_idxs)
    idxs_choose = all_idxs[:n_choose]
    idxs_left = all_idxs[n_choose:]
    return idxs_choose, idxs_left


class LinearLeastSquareModel:

    def __init__(self, input_columns, output_columms, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columms
        self.debug = debug

    def fit(self, data):
        A = data[:, [x for x in self.input_columns]]
        B = data[:, [x for x in self.output_columns]]

        x, resids, rank, s = sl.lstsq(A, B)  # B=A*M  x为拟合矩阵M
        return x  # 拟合矩阵

    def get_errs(self, data, model):
        # model: 模型拟合系数
        A = data[:, [x for x in self.input_columns]]
        B = data[:, [x for x in self.output_columns]]
        B_fit = np.dot(A, model)
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point


def test():
    n_sample = 500
    n_inputs = 1
    n_outputs = 1

    A_exact = 20 * np.random.random((n_sample, n_inputs))
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    B_exact = np.dot(A_exact, perfect_fit)
    # 噪声
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)
    if True:
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])
        np.random.shuffle(all_idxs)
        outlier_idxs = all_idxs[:n_outliers]
        A_noisy[outlier_idxs] = 20 * np.random.random(size=(n_outliers, n_inputs))
        B_noisy[outlier_idxs] = 80 * np.random.normal(size=(n_outliers, n_outputs))

    all_data = np.hstack((A_noisy, B_noisy))
    input_columns = range(n_inputs)
    output_columns = [n_inputs + i for i in range(n_outputs)]

    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)
    #  纯粹最小二乘算法
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # ransac 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 2000, 9e3, 300, debug=debug, return_all=True)

    if True:
        import pylab
        sort_idxs = np.argsort(A_exact[:, 0])
        A_colo_sorted = A_exact[sort_idxs]

        if True:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
            pylab.plot(A_noisy[ransac_data, 0], B_noisy[ransac_data, 0], 'bx',
                       label="ransac data")
        pylab.plot(A_colo_sorted[:, 0], np.dot(A_colo_sorted, ransac_fit)[:, 0], label='ransac fit')
        pylab.plot(A_colo_sorted[:, 0], np.dot(A_colo_sorted, perfect_fit)[:, 0], label='exact system')
        pylab.plot(A_colo_sorted[:, 0], np.dot(A_colo_sorted, linear_fit)[:, 0], label='linear fit')
        pylab.legend()
        pylab.show()


if __name__ == "__main__":
    test()
