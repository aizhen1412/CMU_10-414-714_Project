"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node.inputs[1] * node.inputs[0] ** (
            node.inputs[1] - 1
        ), out_grad * node.inputs[0] ** node.inputs[1] * log(node.inputs[0])
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        grad = out_grad * self.scalar * node.inputs[0] ** (self.scalar - 1)
        return grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = out_grad / b
        grad_b = -1 * out_grad * a / (b**2)
        return grad_a, grad_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        grad = out_grad / self.scalar
        return grad
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes=None):
        self.axes = axes

    def compute(self, a):
        shape = a.shape
        if self.axes is None:
            # 生成正序的 axes
            new_axes = tuple(range(len(shape)))
            # 交换最后两个维度
            new_axes = new_axes[:-2] + (new_axes[-1], new_axes[-2])
            return array_api.transpose(a, new_axes)
        else:
            # 获取输入张量的维度数
            temp = tuple(range(len(shape)))  # 原始的维度顺序
            # 创建一个新的列表来交换 axes[0] 和 axes[1] 对应的维度
            new_axes = list(temp)
            new_axes[self.axes[0]], new_axes[self.axes[1]] = (
                new_axes[self.axes[1]],
                new_axes[self.axes[0]],
            )

            return array_api.transpose(a, tuple(new_axes))

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTIOaN


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        output_shape = out_grad.shape
        grad = out_grad
        for i in range(len(output_shape) - len(input_shape)):
            grad = summation(grad, axes=0)
        for i, dim in enumerate(input_shape):
            if dim == 1 and self.shape[i] != 1:
                grad = summation(grad, axes=i)
        return reshape(grad, input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        expected_reshape = list(node.inputs[0].shape)  # 期望的梯度形状
        if self.axes is not None:  # 如果指定了 axes
            for i in self.axes:  # 遍历 axes
                expected_reshape[i] = 1  # 将指定的维度设置为 1
        else:  # 如果没有指定 axes
            expected_reshape = [
                1 for _ in range(len(expected_reshape))
            ]  # 将所有维度设置为 1
        return broadcast_to(
            reshape(out_grad, expected_reshape), node.inputs[0].shape
        )  # 广播梯度


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # 计算对 A 的梯度：out_grad * B^T
        grad_a = matmul(out_grad, array_api.transpose(node.inputs[1]))

        # 计算对 B 的梯度：A^T * out_grad
        grad_b = matmul(array_api.transpose(node.inputs[0]), out_grad)

        # 确保梯度的形状与输入张量一致
        if grad_a.shape != node.inputs[0].shape:
            grad_a = summation(
                grad_a, tuple(range(len(grad_a.shape) - len(node.inputs[0].shape)))
            )
        if grad_b.shape != node.inputs[1].shape:
            grad_b = summation(
                grad_b, tuple(range(len(grad_b.shape) - len(node.inputs[1].shape)))
            )

        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        relu_mask = Tensor((node.inputs[0].cached_data > 0).astype(array_api.float32))
        return out_grad * relu_mask


def relu(a):
    return ReLU()(a)
