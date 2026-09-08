#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>

#include "tensor.h"
#include "neuron.h"
#include "optimizer.h"

using namespace std;

uint32_t read_uint32(ifstream& file) {
    uint8_t bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);

    return (uint32_t(bytes[0]) << 24) |
           (uint32_t(bytes[1]) << 16) |
           (uint32_t(bytes[2]) << 8) |
           uint32_t(bytes[3]);
}

vector<vector<double>> load_images(const string& path) {
    ifstream file(path, ios::binary);

    if (!file)
        throw runtime_error("Cannot open: " + path);

    uint32_t magic = read_uint32(file);
    uint32_t count = read_uint32(file);
    uint32_t rows = read_uint32(file);
    uint32_t cols = read_uint32(file);

    if (magic != 2051)
        throw runtime_error("Invalid MNIST image file");

    vector<vector<double>> images(count, vector<double>(rows * cols));

    for (uint32_t i = 0; i < count; ++i) {
        for (uint32_t j = 0; j < rows * cols; ++j) {
            uint8_t pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            images[i][j] = static_cast<double>(pixel) / 255.0;
        }
    }

    return images;
}

vector<uint8_t> load_labels(const string& path) {
    ifstream file(path, ios::binary);

    if (!file)
        throw runtime_error("Cannot open: " + path);

    uint32_t magic = read_uint32(file);
    uint32_t count = read_uint32(file);

    if (magic != 2049)
        throw runtime_error("Invalid MNIST label file");

    vector<uint8_t> labels(count);
    file.read(reinterpret_cast<char*>(labels.data()), count);

    return labels;
}

int main() {
    auto images = load_images("mnist dataset/train-images.idx3-ubyte");
    auto labels = load_labels("mnist dataset/train-labels.idx1-ubyte");

    cout << "Images: " << images.size() << endl;
    cout << "Pixels: " << images[0].size() << endl;
    cout << "Labels: " << labels.size() << endl;

    nn::linear layer1(784, 128);
    nn::linear layer2(128, 10);

    optim optimizer("adam");

    int epochs = 3;
    size_t train_size = 10000;
    double lr = 0.001;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        for (size_t i = 0; i < train_size; ++i) {
            Tensor X({images[i]});

            vector<double> target(10, 0.0);
            target[labels[i]] = 1.0;
            Tensor Y({target});

            Tensor hidden = layer1(X);
            hidden = hidden.ReLU();

            Tensor logits = layer2(hidden);
            Tensor probabilities = logits.softmax();

            Tensor loss = CategoricalCrossEntropy(Y, probabilities);

            loss.backward();

            total_loss += loss.impl->data[0];

            optimizer.step(loss, lr);

            loss.zero_grad();

            if ((i + 1) % 100 == 0) {
                cout << "Epoch: " << epoch + 1
                     << " Step: " << i + 1
                     << " Loss: " << total_loss / (i + 1)
                     << endl;
            }
        }

        cout << "Epoch " << epoch + 1
             << " Loss: " << total_loss / train_size
             << endl;
    }

    auto test_images = load_images("mnist dataset/t10k-images.idx3-ubyte");
    auto test_labels = load_labels("mnist dataset/t10k-labels.idx1-ubyte");

    size_t correct = 0;
    size_t test_size = 1000;

    for (size_t i = 0; i < test_size; ++i) {
        Tensor X({test_images[i]});

        Tensor hidden = layer1(X);
        hidden = hidden.ReLU();

        Tensor logits = layer2(hidden);
        Tensor probabilities = logits.softmax();

        size_t prediction = 0;

        for (size_t j = 1; j < probabilities.impl->data.size(); ++j) {
            if (probabilities.impl->data[j] >
                probabilities.impl->data[prediction]) {
                prediction = j;
            }
        }

        if (prediction == test_labels[i])
            ++correct;
    }

    cout << "Accuracy: "
         << 100.0 * correct / test_size
         << "%" << endl;

    return 0;
}