#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t num_samples, size_t num_dim, size_t num_classes,
                                  float lr, size_t batch)
{
    // 遍历数据集，按 batch 大小进行处理
    for (size_t i = 0; i < num_samples; i += batch)
    {
        size_t current_batch = std::min(batch, num_samples - i);

        // 当前批次数据及标签指针
        const float *X_batch = X + i * num_dim;
        const unsigned char *y_batch = y + i;

        // 计算 logits：X_batch * theta
        float *logits = new float[current_batch * num_classes]();
        // logits[j, c] = Σ_i X_batch[j, d] * theta[d, c]
        for (size_t j = 0; j < current_batch; j++)
        {
            for (size_t c = 0; c < num_classes; c++)
            {
                for (size_t d = 0; d < num_dim; d++)
                {
                    logits[j * num_classes + c] += X_batch[j * num_dim + d] * theta[d * num_classes + c];
                }
            }
        }

        // 计算 softmax
        float *Z_exp = new float[current_batch * num_classes];
        float *Z_softmax = new float[current_batch * num_classes];
        for (size_t j = 0; j < current_batch; j++)
        {
            // 为数值稳定性，先减去当前样本 logits 的最大值
            float max_logit = *std::max_element(logits + j * num_classes,
                                                logits + (j + 1) * num_classes);
            float Z_exp_sum = 0;
            // Z_exp[j, c] = exp(logits[j, c] - max_logit)
            for (size_t c = 0; c < num_classes; c++)
            {
                Z_exp[j * num_classes + c] = exp(logits[j * num_classes + c] - max_logit);
                Z_exp_sum += Z_exp[j * num_classes + c];
            }
            // Z_softmax[j, c] = Z_exp[j, c] / Σ_c Z_exp[j, c]
            for (size_t c = 0; c < num_classes; c++)
            {
                Z_softmax[j * num_classes + c] = Z_exp[j * num_classes + c] / Z_exp_sum;
            }
        }

        // 生成当前批次的 one-hot 编码
        float *I_y = new float[current_batch * num_classes]();
        for (size_t j = 0; j < current_batch; j++)
        {
            I_y[j * num_classes + y_batch[j]] = 1.0f;
        }

        // 计算梯度：∇θ = X_batch^T * (Z_softmax - I_y) / batch
        float *gradient = new float[num_dim * num_classes]();
        // gradient[d, c] = Σ_j X_batch[j, d] * (Z_softmax[j, c] - I_y[j, c]) / batch
        for (size_t d = 0; d < num_dim; d++)
        {
            for (size_t c = 0; c < num_classes; c++)
            {
                for (size_t j = 0; j < current_batch; j++)
                {
                    gradient[d * num_classes + c] +=
                        X_batch[j * num_dim + d] *
                        (Z_softmax[j * num_classes + c] - I_y[j * num_classes + c]);
                }
                gradient[d * num_classes + c] /= current_batch;
            }
        }

        // 根据计算到的梯度更新参数 theta
        for (size_t d = 0; d < num_dim; d++)
        {
            for (size_t c = 0; c < num_classes; c++)
            {
                theta[d * num_classes + c] -= lr * gradient[d * num_classes + c];
            }
        }

        // 释放临时分配的内存
        delete[] logits;
        delete[] Z_exp;
        delete[] Z_softmax;
        delete[] I_y;
        delete[] gradient;
    }
}



/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
