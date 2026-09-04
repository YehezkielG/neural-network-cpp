#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <set>
#include <memory>
#include <functional>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;

struct TensorImpl;
// 1. Data & Graph disimpan di TensorImpl (Heap)
struct TensorImpl : public enable_shared_from_this<TensorImpl>{
    vector<double> data;
    shared_ptr<TensorImpl> grads = nullptr;
    set<shared_ptr<TensorImpl>> _prev;
    function<void(const TensorImpl&)> _backward;
    string name;
    int rows = 0;
    int cols = 0;
    bool isOperation = false;

    // 1. Constructor Default
    TensorImpl() : rows(0), cols(0), isOperation(false) {}

    // 2. Constructor dari 1D vector<double>
    TensorImpl(const vector<double>& d, int r = 1, int c = -1) : data(d), rows(r) {
        cols = (c == -1) ? d.size() : c;
        name = "init_" + to_string(rows) + "x" + to_string(cols);
    }

    TensorImpl(const vector<vector<double>>& arr) {
        rows = arr.size();
        cols = arr.empty() ? 0 : arr[0].size();
        name = "init_" + to_string(rows) + "x" + to_string(cols);
        data.reserve(rows * cols);
        for (const auto& row : arr) {
            data.insert(data.end(), row.begin(), row.end());
        }
    }

    TensorImpl(vector<double>&& d, int r, int c, string op) 
        : data(move(d)), rows(r), cols(c), name(op + " (" + to_string(r) + "x" + to_string(c) + ")"), isOperation(true) {}

    static shared_ptr<TensorImpl> create(const vector<vector<double>>& arr) {
        auto impl = make_shared<TensorImpl>(arr);
        impl->grads = make_shared<TensorImpl>(vector<double>(impl->rows * impl->cols, 0.0), impl->rows, impl->cols);
        return impl;
    }

    static shared_ptr<TensorImpl> create(const vector<double>& d, int r = 1, int c = -1) {
        auto impl = make_shared<TensorImpl>(d, r, c);
        impl->grads = make_shared<TensorImpl>(vector<double>(impl->data.size(), 0.0), impl->rows, impl->cols);
        return impl;
    }

        // Static Math Operations
    static shared_ptr<TensorImpl> add(shared_ptr<TensorImpl> left, shared_ptr<TensorImpl> right) {
        vector<double> res(left->data.size());
        int irow = 0;
        if ((left->cols % right->cols == 0) && (left->rows % right->rows == 0)) {
            int k = 0;
            for (size_t i = 0; i < left->data.size(); i++) {
                k = ((irow % right->rows) * right->cols) + (i % right->cols);
                if ((i + 1) % left->cols == 0) irow++;
                res[i] = left->data[i] + right->data[k];
            }
        } else {
            throw invalid_argument("Incompatible dimensions for add");
        } 
        
        auto out = make_shared<TensorImpl>(move(res), left->rows, left->cols, "+");
        out->grads = make_shared<TensorImpl>(vector<double>(out->rows * out->cols, 0.0), out->rows, out->cols);
        out->_prev.insert(left);
        out->_prev.insert(right);

        out->_backward = [left, right](const TensorImpl& grad) {
            if (left->grads) {
                for (size_t i = 0; i < left->grads->data.size(); ++i) {
                    left->grads->data[i] += grad.data[i];
                }
            }
            if (right->grads) {
                if (right->rows == grad.rows && right->cols == grad.cols) {
                    for (size_t i = 0; i < right->grads->data.size(); ++i) {
                        right->grads->data[i] += grad.data[i];
                    }
                } else {
                    for (size_t i = 0; i < grad.rows; ++i) {
                        for (size_t j = 0; j < right->data.size(); ++j) {
                            right->grads->data[j] += grad.data[grad.cols * i + j];
                        }
                    }
                }
            }
        };
        return out;
    }

    static shared_ptr<TensorImpl> minus(shared_ptr<TensorImpl> left, shared_ptr<TensorImpl> right) {
        vector<double> res(left->data.size());
        int irow = 0;
        if ((left->cols % right->cols == 0) && (left->rows % right->rows == 0)) {
            int k = 0;
            for (size_t i = 0; i < left->data.size(); i++) {
                k = ((irow % right->rows) * right->cols) + (i % right->cols);
                if ((i + 1) % left->cols == 0) irow++;
                res[i] = left->data[i] - right->data[k];
            }
        } else {
            throw invalid_argument("Incompatible dimensions for minus");
        } 
        
        auto out = make_shared<TensorImpl>(move(res), left->rows, left->cols, "-");
        out->grads = make_shared<TensorImpl>(vector<double>(out->rows * out->cols, 0.0), out->rows, out->cols);
        out->_prev.insert(left);
        out->_prev.insert(right);

        out->_backward = [left, right](const TensorImpl& grad) {
            if (left->grads) {
                for (size_t i = 0; i < left->grads->data.size(); ++i) {
                    left->grads->data[i] += grad.data[i];
                }
            }
            if (right->grads) {
                if (right->rows == grad.rows && right->cols == grad.cols) {
                    for (size_t i = 0; i < right->grads->data.size(); ++i) {
                        right->grads->data[i] -= grad.data[i];
                    }
                } else {
                    for (size_t i = 0; i < grad.rows; ++i) {
                        for (size_t j = 0; j < right->data.size(); ++j) {
                            right->grads->data[j] -= grad.data[grad.cols * i + j];
                        }
                    }
                }
            }
        };
        return out;
    }

    static shared_ptr<TensorImpl> multiply(shared_ptr<TensorImpl> left, shared_ptr<TensorImpl> right) {
        vector<double> res(left->data.size());
        int irow = 0;
        if ((left->cols % right->cols == 0) && (left->rows % right->rows == 0)) {
            int k = 0;
            for (size_t i = 0; i < left->data.size(); i++) {
                k = ((irow % right->rows) * right->cols) + (i % right->cols);
                if ((i + 1) % left->cols == 0) irow++;
                res[i] = left->data[i] * right->data[k];
            }
        } else {
            throw invalid_argument("Incompatible dimensions for multiply");
        } 

        auto out = make_shared<TensorImpl>(move(res), left->rows, left->cols, "*");
        out->grads = make_shared<TensorImpl>(vector<double>(out->rows * out->cols, 0.0), out->rows, out->cols);
        out->_prev.insert(left);
        out->_prev.insert(right);

        out->_backward = [left, right](const TensorImpl& grad) {
            if (left->grads) {
                for (size_t i = 0; i < left->grads->data.size(); ++i) {
                    left->grads->data[i] += grad.data[i] * right->data[i];
                }
            }
            if (right->grads) {
                for (size_t i = 0; i < right->grads->data.size(); ++i) {
                    right->grads->data[i] += grad.data[i] * left->data[i];
                }
            }
        };
        return out;
    }

    static shared_ptr<TensorImpl> devide(shared_ptr<TensorImpl> left, shared_ptr<TensorImpl> right) {
        vector<double> res(left->data.size());
        int irow = 0;
        if ((left->cols % right->cols == 0) && (left->rows % right->rows == 0)) {
            int k = 0;
            for (size_t i = 0; i < left->data.size(); i++) {
                k = ((irow % right->rows) * right->cols) + (i % right->cols);
                if ((i + 1) % left->cols == 0) irow++;
                res[i] = left->data[i] / right->data[k];
            }
        } else {
            throw invalid_argument("Incompatible dimensions for divide");
        } 
        
        auto out = make_shared<TensorImpl>(move(res), left->rows, left->cols, "/");
        out->grads = make_shared<TensorImpl>(vector<double>(out->rows * out->cols, 0.0), out->rows, out->cols);
        out->_prev.insert(left);
        out->_prev.insert(right);

        out->_backward = [left, right](const TensorImpl& grad) {
            if (left->grads) {
                for (size_t i = 0; i < grad.data.size(); ++i) {
                    left->grads->data[i] += grad.data[i] / right->data[i];
                }
            }
            if (right->grads) {
                for (size_t i = 0; i < grad.data.size(); ++i) {
                    right->grads->data[i] -= grad.data[i] * left->data[i] / (right->data[i] * right->data[i]);
                }
            }
        };
        return out;
    }

    static shared_ptr<TensorImpl> mn(shared_ptr<TensorImpl> left, shared_ptr<TensorImpl> right) {
        if (left->cols != right->rows) {
            throw invalid_argument("Incompatible dimensions for matrix multiplication");
        }
        vector<double> res(left->rows * right->cols, 0.0); 

        for (size_t i = 0; i < left->rows; ++i) {
            for (size_t j = 0; j < right->cols; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < left->cols; ++k) {
                    sum += left->data[i * left->cols + k] * right->data[k * right->cols + j];
                }
                res[i * right->cols + j] = sum;
            }
        }
        
        auto out = make_shared<TensorImpl>(move(res), left->rows, right->cols, "matmul");

        out->grads = make_shared<TensorImpl>(vector<double>(out->rows * out->cols, 0.0), out->rows, out->cols);

        out->_prev.insert(left);
        out->_prev.insert(right);

        out->_backward = [left, right](const TensorImpl& grad) {        
            auto grad_ptr = make_shared<TensorImpl>(grad.data, grad.rows, grad.cols);
            
            auto right_T = TensorImpl::transpose(right);
            auto grad_left = TensorImpl::mn(grad_ptr, right_T);

            auto left_T = TensorImpl::transpose(left);
            auto grad_right = TensorImpl::mn(left_T, grad_ptr);

            for (size_t i = 0; i < left->grads->data.size(); ++i) {
                left->grads->data[i] += grad_left->data[i];
            }

            for (size_t i = 0; i < right->grads->data.size(); ++i) {
                right->grads->data[i] += grad_right->data[i];
            }
        };

        return out;
    }

    static shared_ptr<TensorImpl> sigmoid(shared_ptr<TensorImpl> self) {
        vector<double> result;
        result.reserve(self->data.size());

        for (size_t i = 0; i < self->data.size(); ++i) {
            result.push_back(1.0 / (1.0 + exp(-self->data[i])));
        }

        auto out = make_shared<TensorImpl>(move(result), self->rows, self->cols, "sigmoid");
        out->grads = make_shared<TensorImpl>(vector<double>(out->rows * out->cols, 0.0), out->rows, out->cols);
        
        out->_prev.insert(self);

        // 4. Lambda Backward Pass
        out->_backward = [self, out](const TensorImpl& grad) {
            if (!self->grads) return;
            for (size_t i = 0; i < self->data.size(); ++i) {
                double y = out->data[i];
                self->grads->data[i] += grad.data[i] * y * (1.0 - y);
            }
        };

        return out;
    }


    static shared_ptr<TensorImpl> ReLU(shared_ptr<TensorImpl> self) {
        vector<double> result;
        result.reserve(self->data.size());

        for (size_t i = 0; i < self->data.size(); ++i) {
            result.push_back(max(0.0, self->data[i]));
        }

        auto out = make_shared<TensorImpl>(move(result), self->rows, self->cols, "sigmoid");
        out->grads = make_shared<TensorImpl>(vector<double>(out->rows * out->cols, 0.0), out->rows, out->cols);
        
        out->_prev.insert(self);

        out->_backward = [self, out](const TensorImpl& grad) {
            if (!self->grads) return;
            for (size_t i = 0; i < self->data.size(); ++i) {
                self->grads->data[i] += grad.data[i] * (self->data[i] > 0.0 ? 1.0 : 0.0);
            }
        };

        return out;
    }

    static shared_ptr<TensorImpl> Softmax(shared_ptr<TensorImpl> self) {
        vector<double> result;
        result.reserve(self->data.size());

        //$$S_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$

        
        for (size_t i = 0; i < self->rows; ++i) {
            for(size_t j = 0; j < self->cols; ++j) {
                double sum_exp = 0.0;
                for (size_t k = 0; k < self->cols; ++k) {
                    sum_exp += exp(self->data[i * self->cols + k]);
                }
                result.push_back(exp(self->data[i * self->cols + j]) / sum_exp);
            }
        }

        auto out = make_shared<TensorImpl>(move(result), self->rows, self->cols, "softmax");
        out->grads = make_shared<TensorImpl>(vector<double>(out->rows * out->cols, 0.0), out->rows, out->cols);
        
        out->_prev.insert(self);

        out->_backward = [self, out](const TensorImpl& grad) {
            if (!self->grads) return;
            for(size_t i = 0; i < self->rows; ++i) {
                for(size_t j = 0; j < self->cols; ++j) {
                    double sum = 0.0;
                    for(size_t k = 0; k < self->cols; ++k) {
                        double delta = (j == k) ? 1.0 : 0.0;
                        sum += out->data[i * self->cols + k] * (delta - out->data[i * self->cols + j]) * grad.data[i * self->cols + k];
                    }
                    self->grads->data[i * self->cols + j] += sum;
                }
            }
        };

        return out;
    }

    
    void transpose_inplace() {
        vector<double> temp(rows * cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                temp[j * rows + i] = data[i * cols + j];
            }
        }
        swap(rows, cols);
        data = move(temp);
    }

    static shared_ptr<TensorImpl> transpose(shared_ptr<TensorImpl> self) {
        vector<double> temp(self->rows * self->cols);
        for (int i = 0; i < self->rows; ++i) {
            for (int j = 0; j < self->cols; ++j) {
                temp[j * self->rows + i] = self->data[i * self->cols + j];
            }
        }

        auto out = make_shared<TensorImpl>(move(temp), self->cols, self->rows, "transpose");
        out->grads = make_shared<TensorImpl>(vector<double>(out->rows * out->cols, 0.0), out->rows, out->cols);
        out->_prev.insert(self);

        out->_backward = [self](const TensorImpl& grad) {
            if (!self->grads) return;
            for (int i = 0; i < grad.rows; ++i) {
                for (int j = 0; j < grad.cols; ++j) {
                    self->grads->data[j * self->cols + i] += grad.data[i * grad.cols + j];
                }
            }
        };

        return out;
    }

    
    static shared_ptr<TensorImpl> BinaryCrossEntropy (shared_ptr<TensorImpl> Y, shared_ptr<TensorImpl> y_pred){
        vector<double> result_data(1);

        const double eps = 1e-7;

        for (size_t i = 0; i < Y->data.size(); i++) {
            double p = y_pred->data[i];
            p = max(eps, min(p, 1.0 - eps));
            result_data[0] += Y->data[i] * log(p) + (1.0 - Y->data[i]) * log(1.0 - p);
        }

        result_data[0] *= -1.0 / Y->data.size();

        auto out = make_shared<TensorImpl>(move(result_data), 1, 1, "BCE");
        out->grads = make_shared<TensorImpl>(vector<double>(1, 1), 1, 1);

        out->_prev.insert(Y);
        out->_prev.insert(y_pred);

        out->_backward = [Y, y_pred](const TensorImpl& grad) {
            double N = y_pred->data.size();
            const double eps = 1e-7;

            for (size_t i = 0; i < N; ++i) {
                double p_val = max(
                    eps,
                    min(y_pred->data[i], 1.0 - eps)
                );

                y_pred->grads->data[i] +=
                    (
                        (1.0 - Y->data[i]) / (1.0 - p_val)
                        - Y->data[i] / p_val
                    ) / N * grad.data[0];
            }
        };

        return out;
    }

    static shared_ptr<TensorImpl> MSE (shared_ptr<TensorImpl> Y, shared_ptr<TensorImpl> y_pred){
        vector<double> result_data(1);

        if(Y->data.size() == y_pred->data.size()){
            for(size_t i = 0; i < Y->data.size(); i++){
                result_data[0] += pow((Y->data[i] - y_pred->data[i]), 2);
            }
            result_data[0] *= (1.0 / Y->data.size());
        }else{
            throw invalid_argument("Incompatible dimensions");
        }

        auto out = make_shared<TensorImpl>(move(result_data), 1, 1, "MSE");
        out->grads = make_shared<TensorImpl>(vector<double>(1, 1), 1, 1);

        out->_prev.insert(Y);
        out->_prev.insert(y_pred);

        out->_backward = [Y, y_pred](const TensorImpl& grad) {
            for (size_t i = 0; i < Y->data.size(); ++i) {
                Y->grads->data[i] += grad.data[0] * (2.0 / Y->data.size()) * (Y->data[i] - y_pred->data[i]);
            }

            for (size_t i = 0; i < y_pred->data.size(); ++i) {
                y_pred->grads->data[i] += grad.data[0] * (2.0 / y_pred->data.size()) * (y_pred->data[i] - Y->data[i]);
            }
        };

        return out;
    }

    static shared_ptr<TensorImpl> CrossEntropy(shared_ptr<TensorImpl> Y, shared_ptr<TensorImpl> y_pred){
        vector<double> result_data(1);

        if(Y->data.size() == y_pred->data.size()){
            for(size_t i = 0; i < Y->data.size(); i++){
                result_data[0] += Y->data[i] * log(y_pred->data[i]);
            }
            result_data[0] = -(result_data[0]);
        }else{
            throw invalid_argument("Incompatible dimensions");
        }

        auto out = make_shared<TensorImpl>(move(result_data), 1, 1, "CE");
        out->grads = make_shared<TensorImpl>(vector<double>(1, 1), 1, 1);

        out->_prev.insert(Y);
        out->_prev.insert(y_pred);

        out->_backward = [Y, y_pred](const TensorImpl& grad) {
            for (size_t i = 0; i < y_pred->data.size(); ++i) {
                y_pred->grads->data[i] += -Y->data[i] / (y_pred->data[i]) * grad.data[0];
            }
        };

        return out;
    }

    void backward() {
        vector<shared_ptr<TensorImpl>> topo;
        set<shared_ptr<TensorImpl>> visited;

        function<void(shared_ptr<TensorImpl>)> build_topo = [&](shared_ptr<TensorImpl> node) {
            if (!node || visited.count(node)) return;
            visited.insert(node);
            for (auto prev : node->_prev) {
                build_topo(prev);
            }
            topo.push_back(node);
        };

        build_topo(shared_from_this());
        reverse(topo.begin(), topo.end());

        if (grads && !grads->data.empty()) {
            grads->data = vector<double>(data.size(), 1.0);
        }

        for (auto node : topo) {
            if (node->_backward && node->grads) {
                node->_backward(*node->grads);
            }
        }
    }

    void step(double lr) {
        if (!grads) return;
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] -= lr * grads->data[i];
        }
    }    

};

class Tensor {
public:
    shared_ptr<TensorImpl> impl;
    // Constructors
    Tensor() : impl(make_shared<TensorImpl>()) {}
    Tensor(shared_ptr<TensorImpl> p) : impl(p) {}
    Tensor(const vector<vector<double>>& arr) : impl(TensorImpl::create(arr)) {}
    Tensor(const vector<double>& d, int r, int c) : impl(TensorImpl::create(d, r, c)) {}

    // Operators
    Tensor operator+(const Tensor& other) const { return Tensor(TensorImpl::add(this->impl, other.impl)); }
    Tensor operator-(const Tensor& other) const { return Tensor(TensorImpl::minus(this->impl, other.impl)); }
    Tensor operator*(const Tensor& other) const { return Tensor(TensorImpl::multiply(this->impl, other.impl)); }
    Tensor operator/(const Tensor& other) const { return Tensor(TensorImpl::devide(this->impl, other.impl)); }

    // Methods
    Tensor mn(const Tensor& other) const { return Tensor(TensorImpl::mn(this->impl, other.impl)); }
    Tensor matmul(const Tensor& other) const { return mn(other); }
    Tensor transpose() const { return Tensor(TensorImpl::transpose(this->impl)); }
    Tensor t() const { return transpose(); }

    // Activation Functions
    Tensor sigmoid() const { return Tensor(TensorImpl::sigmoid(this->impl)); }
    Tensor ReLU() const { return Tensor(TensorImpl::ReLU(this->impl)); }
    Tensor softmax() const { return Tensor(TensorImpl::Softmax(this->impl)); }

    
    void backward() {
        if (impl) impl->backward();
    }

    void print() const {
        cout << "[";
        for (size_t i = 0; i < impl->rows; ++i) {
            cout << " [";
            for (size_t j = 0; j < impl->cols; ++j) {
                cout << impl->data[i * impl->cols + j];
                if (j < impl->cols - 1) cout << ", ";
            }
            cout << "]";
            if (i < impl->rows - 1) cout << endl << " ";
        }
        cout << " ]" << endl;
    }
};

inline Tensor BinaryCrossEntropy(const Tensor& Y, const Tensor& y_pred) {
    return Tensor(TensorImpl::BinaryCrossEntropy(Y.impl, y_pred.impl));
}

inline Tensor MSE(const Tensor& Y, const Tensor& y_pred) {
    return Tensor(TensorImpl::MSE(Y.impl, y_pred.impl));
}

inline Tensor CategoricalCrossEntropy(const Tensor& Y, const Tensor& y_pred) {
    return Tensor(TensorImpl::CrossEntropy(Y.impl, y_pred.impl));
}

#endif 