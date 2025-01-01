import struct
import numpy as np
import gzip

try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    # 打开并读取图像文件
    with gzip.open(image_filename, "rb") as f_images:
        f_images.read(4)
        num_images = struct.unpack(">I", f_images.read(4))[0]  # 图像数量
        rows = struct.unpack(">I", f_images.read(4))[0]  # 行数
        cols = struct.unpack(">I", f_images.read(4))[0]  # 列数 (28)

        # 读取图像数据
        image_data = np.frombuffer(f_images.read(), dtype=np.uint8)
        X = image_data.reshape(num_images, rows * cols)  # 重塑为 (num_images, 784)
        X = X.astype(np.float32) / 255.0  # 归一化到 [0, 1]

    # 打开并读取标签文件
    with gzip.open(label_filename, "rb") as f_labels:
        f_labels.read(4)  # 跳过 magic number
        num_labels = struct.unpack(">I", f_labels.read(4))[0]  # 标签数量

        # 读取标签数据
        label_data = np.frombuffer(f_labels.read(), dtype=np.uint8)
        y = label_data.astype(np.uint8)

    # 返回包含图像和标签的元组
    return X, y


def softmax_loss(Z, y):
    """
    Compute the softmax loss.

    Args:
        Z (np.ndarray): Logits of shape (batch_size, num_classes).
        y (np.ndarray): True labels of shape (batch_size,).

    Returns:
        float: The average softmax loss.
    """
    # 计算每个样本的log-sum-exp值
    log_sum_exp = np.log(np.sum(np.exp(Z), axis=1))  # log(sum(exp(z_i)))

    # 选择正确类别的logit并计算损失
    correct_logits = Z[np.arange(Z.shape[0]), y]  # 选择每个样本的真实类别logit

    # 计算平均softmax损失
    loss = np.mean(log_sum_exp - correct_logits)

    return loss


def softmax_regression_epoch(X, y, theta, lr=0.1, batch=100):
    """Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    num_classes = theta.shape[1]
    # 批量遍历所有样本
    for i in range(0, num_examples, batch):
        # 获取当前批次 (X_batch, y_batch)
        X_batch = X[i : i + batch]
        y_batch = y[i : i + batch]

        # 计算 logits (Z = X * theta)
        logits = np.dot(X_batch, theta)  # shape: (batch, num_classes)

        # 计算 softmax 概率 (Z_softmax)
        Z_exp = np.exp(logits)
        Z_softmax = Z_exp / np.sum(Z_exp, axis=1, keepdims=True)  # 按行进行 softmax

        # 创建真实标签的 one-hot 编码矩阵
        I_y = np.zeros_like(Z_softmax)
        I_y[np.arange(batch), y_batch] = 1

        # 计算梯度
        gradient = np.dot(X_batch.T, (Z_softmax - I_y)) / batch

        # 更新参数 theta
        theta -= lr * gradient
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network with ReLU activation.

    Args:
        X (np.ndarray): Input data of shape (num_examples, input_dim).
        y (np.ndarray): 1D array of class labels of shape (num_examples,).
        W1 (np.ndarray): First layer weights of shape (input_dim, hidden_dim).
        W2 (np.ndarray): Second layer weights of shape (hidden_dim, num_classes).
        lr (float): Learning rate for SGD.
        batch (int): Size of minibatch for SGD.

    Returns:
        None
    """
    num_examples = X.shape[0]
    num_classes = W2.shape[1]
    d = W1.shape[1]  # 隐藏层大小

    # 按批次遍历所有样本
    for i in range(0, num_examples, batch):
        # 获取当前批次
        X_batch = X[i : i + batch]
        y_batch = y[i : i + batch]

        # 前向传播：计算隐藏层 Z1，以及 logits = Z1 * W2
        Z1 = np.maximum(0, np.dot(X_batch, W1))  # ReLU 激活函数
        logits = np.dot(Z1, W2)  # shape: (batch, num_classes)

        # Softmax：计算概率分布
        Z_exp = np.exp(logits)
        Z_softmax = Z_exp / np.sum(Z_exp, axis=1, keepdims=True)

        # 真实标签的 one-hot 编码
        I_y = np.zeros_like(Z_softmax)
        I_y[np.arange(batch), y_batch] = 1

        # 反向传播：计算梯度
        G2 = Z_softmax - I_y  # 对 W2 的梯度
        G1 = (Z1 > 0) * np.dot(G2, W2.T)  # 对 W1 的梯度 (ReLU 导数)

        # 计算 W1 和 W2 的梯度
        grad_W2 = np.dot(Z1.T, G2) / batch
        grad_W1 = np.dot(X_batch.T, G1) / batch

        # 使用 SGD 更新权重
        W1 -= lr * grad_W1
        W2 -= lr * grad_W2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper funciton to compute both loss and error"""
    return softmax_loss(h, y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100, cpp=False):
    """Example function to fully train a softmax regression classifier"""
    theta = np.zeros((X_tr.shape[1], y_tr.max() + 1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print(
            "|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |".format(
                epoch, train_loss, train_err, test_loss, test_err
            )
        )


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=500, epochs=10, lr=0.5, batch=100):
    """Example function to train two layer neural network"""
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr @ W1, 0) @ W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te @ W1, 0) @ W2, y_te)
        print(
            "|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |".format(
                epoch, train_loss, train_err, test_loss, test_err
            )
        )


if __name__ == "__main__":
    X_tr, y_tr = parse_mnist(
        "data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz"
    )
    X_te, y_te = parse_mnist(
        "data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz"
    )

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr=0.2)
