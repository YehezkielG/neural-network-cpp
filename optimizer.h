#include "tensor.h"
#include <memory>

class optim {
private:
    string type;
    size_t t = 0;
    unordered_map<TensorImpl*, vector<double>> m, v;

public:
    optim(string optimizer_type) : type(optimizer_type) {}

    void step_sgd(shared_ptr<TensorImpl> loss_impl, double lr) {
        for (auto& node : loss_impl->topo) {
            if (node->_prev.empty() && node->grads) {
                for (size_t i = 0; i < node->data.size(); ++i)
                    node->data[i] -= lr * node->grads->data[i];
            }
        }
    }

    void step_adam(shared_ptr<TensorImpl> loss_impl, double lr) {
        ++t;
        double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;

        for (auto& node : loss_impl->topo) {
            if (node->_prev.empty() && node->grads) {
                if (m.find(node.get()) == m.end()) { 
                    m[node.get()] = vector<double>(node->data.size(), 0.0);
                    v[node.get()] = vector<double>(node->data.size(), 0.0);
                }

                for (size_t i = 0; i < node->data.size(); ++i) {
                    double gt = node->grads->data[i];

                    m[node.get()][i] = beta1 * m[node.get()][i] + (1.0 - beta1) * gt;
                    v[node.get()][i] = beta2 * v[node.get()][i] + (1.0 - beta2) * gt * gt;

                    double mt_hat = m[node.get()][i] / (1.0 - pow(beta1, t));
                    double vt_hat = v[node.get()][i] / (1.0 - pow(beta2, t));

                    node->data[i] -= lr * mt_hat / (sqrt(vt_hat) + eps);
                }
            }
        }
    }

    void step(Tensor loss, double lr) {
        if (type == "SGD")
            step_sgd(loss.impl, lr);
        else if (type == "adam")
            step_adam(loss.impl, lr);
        else
            throw invalid_argument("Unsupported optimizer type: " + type);
    }

    vector<double>& momentum(shared_ptr<TensorImpl> node) {
        return m[node.get()];
    }

    vector<double>& variance(shared_ptr<TensorImpl> node) {
        return v[node.get()];
    }
};
