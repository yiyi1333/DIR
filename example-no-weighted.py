import numpy as np
from sklearn.metrics.pairwise import cosine_distances

x = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
)
a = np.array(
    [
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
    ]
)


def trad(x, a):
    # 创建一个与x相同规模的矩阵,数据用float64类型
    y = np.zeros((10, 10))
    r = np.zeros(10)
    for i in range(x.shape[0]):
        sum_dist = 0
        count = 0
        for j in range(x.shape[1]):
            if a[i, j] == 1:
                count += 1
                y[i] += x[j]
                # 计算x[i]与x[j]的余弦距离
                sum_dist += cosine_distances([x[i]], [x[j]])
        if count != 0:
            y[i] /= count
            r[i] = sum_dist / count
    return y, r


def parallel(x, a):
    y = np.zeros((10, 10))
    r = np.zeros(10)
    # 计算邻接矩阵a的2阶矩阵
    a2 = np.dot(a, a)
    a2 = np.minimum(a2, 1)
    a2 = a2 - a
    a2 = np.maximum(a2, 0)

    for i in range(x.shape[0]):
        count = 0.0
        for j in range(x.shape[1]):
            if a[i, j] == 1:
                y[i] += x[j] * 0.9
                r[i] += cosine_distances([x[i]], [x[j]]) * 0.9
                count += 0.9
            elif a2[i, j] == 1:
                y[i] += x[j] * 0.1
                r[i] += cosine_distances([x[i]], [x[j]]) * 0.1
                count += 0.1
        y[i] = y[i] / count
        r[i] = r[i] / count
    return y, r


if __name__ == "__main__":
    y, r = trad(x, a)
    # 保留小数点后两位
    np.set_printoptions(precision=2)
    print(y)
    print(1 - r.mean())
    y, r = trad(y, a)
    print(y)
    print(1 - r.mean())
    y, r = trad(y, a)
    print(y)
    print(1 - r.mean())
    # y, r = parallel(x, a)
    # np.set_printoptions(precision=2)
    # print(y)
    # print(1 - r.mean())
