"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


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


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR CODE
    batch_size = Z.shape[0]
    log_sum_exp = ndl.log(ndl.summation(ndl.exp(Z), axes=(1,)))
    correct_class_logits = ndl.summation(Z * y_one_hot, axes=(1,))
    loss = ndl.summation(-correct_class_logits + log_sum_exp, axes=(0,))
    return loss / batch_size
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    n_batch = (X.shape[0] + batch - 1) // batch  # 计算batch的数量
    one_hot = np.eye(W2.shape[1])[y]  # 生成one-hot编码
    for i in range(n_batch):  # 遍历所有batch
        X_batch = X[i * batch : (i + 1) * batch]  # 获取当前batch的数据
        y_batch = y[i * batch : (i + 1) * batch]  # 获取当前batch的标签
        one_hot_batch = one_hot[
            i * batch : (i + 1) * batch
        ]  # 获取当前batch的one-hot编码
        X_tensor = ndl.Tensor(X_batch)  # 转换为ndl.Tensor
        y_tensor = ndl.Tensor(y_batch)
        one_hot_tensor = ndl.Tensor(one_hot_batch)  # 转换为ndl.Tensor
        Z = ndl.matmul(ndl.relu(ndl.matmul(X_tensor, W1)), W2)  # 计算logits
        loss = softmax_loss(Z, one_hot_tensor)  # 计算损失
        loss.backward()  # 反向传播
        W1 = (W1 - lr * W1.grad).detach()  # 更新W1
        W2 = (W2 - lr * W2.grad).detach()  # 更新W2
    return W1, W2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
