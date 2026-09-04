#include <iostream>
#include <vector>
#include "tensor.h"
#include "neuron.h"

using namespace std;
using namespace nn;

void gradient_descent(Tensor& parameter, double lr) {
    if (!parameter.impl || !parameter.impl->grads) return;
    for (size_t i = 0; i < parameter.impl->data.size(); ++i) {
        parameter.impl->data[i] -= lr * parameter.impl->grads->data[i];
    }
}

void zero_grad(Tensor& parameter) {
    if (!parameter.impl || !parameter.impl->grads) return;
    for (size_t i = 0; i < parameter.impl->grads->data.size(); ++i) {
        parameter.impl->grads->data[i] = 0.0;
    }
}

// Fungsi bantuan untuk mencari indeks nilai tertinggi (Argmax) dalam satu baris
int argmax(const vector<double>& data, int start_idx, int cols) {
    int max_idx = 0;
    double max_val = data[start_idx];
    for (int i = 1; i < cols; ++i) {
        if (data[start_idx + i] > max_val) {
            max_val = data[start_idx + i];
            max_idx = i;
        }
    }
    return max_idx;
}

int main() {
    cout << "=== Training Multiclass Classification ===" << endl;

    // Dataset Input X (4 sampel, 2 fitur)
    Tensor X({
        {1.0, 2.0},
        {2.0, 1.0},
        {5.0, 6.0},
        {6.0, 5.0}
    });

    // Label Target Y (One-hot encoding untuk 3 kelas)
    Tensor Y({
        {1.0, 0.0, 0.0}, // Kelas 0
        {1.0, 0.0, 0.0}, // Kelas 0
        {0.0, 1.0, 0.0}, // Kelas 1
        {0.0, 0.0, 1.0}  // Kelas 2
    });

    // Model Architecture: Input 2 -> Hidden 4 -> Output 3
    linear layer1(2, 4); 
    linear layer2(4, 3); 

    // Pakai learning rate yang lebih stabil karena kita udah pakai rata-rata (mean) loss
    double learning_rate = 0.05; 
    int total_epochs = 2000;

    // --- SESI TRAINING ---
    for (int step = 1; step <= total_epochs; ++step) {
        Tensor y_pred = layer2(layer1(X).ReLU()).softmax();
        Tensor loss = CategoricalCrossEntropy(Y, y_pred);
        loss.backward();

        gradient_descent(layer1.weight, learning_rate);
        gradient_descent(layer1.biases, learning_rate);
        gradient_descent(layer2.weight, learning_rate);
        gradient_descent(layer2.biases, learning_rate);

        zero_grad(layer1.weight);
        zero_grad(layer1.biases);
        zero_grad(layer2.weight);
        zero_grad(layer2.biases);
    }
    
    cout << "Training selesai!" << endl << endl;

    // --- SESI EVALUASI (Melihat Tebakan Asli) ---
    cout << "=== Evaluasi Hasil Tebakan ===" << endl;
    
    // Lakukan satu kali forward pass terakhir tanpa backward
    Tensor final_pred = layer2(layer1(X).ReLU()).softmax();
    
    int correct_predictions = 0;
    int num_samples = X.impl->rows;
    int num_classes = final_pred.impl->cols;

    for (int i = 0; i < num_samples; ++i) {
        int start_idx = i * num_classes;
        
        // Cari tebakan model (indeks probabilitas tertinggi)
        int predicted_class = argmax(final_pred.impl->data, start_idx, num_classes);
        
        // Cari target asli dari data Y
        int actual_class = argmax(Y.impl->data, start_idx, num_classes);
        
        // Cek kebenaran
        bool is_correct = (predicted_class == actual_class);
        if (is_correct) correct_predictions++;

        // Print rincian per baris
        cout << "Sampel " << i + 1 << ":" << endl;
        cout << "  Probabilitas : [ ";
        for(int j = 0; j < num_classes; ++j) {
            cout << final_pred.impl->data[start_idx + j] << " ";
        }
        cout << "]" << endl;
        cout << "  Tebakan Model: Kelas " << predicted_class << endl;
        cout << "  Target Asli  : Kelas " << actual_class;
        cout << (is_correct ? "  (BENAR)" : "  (SALAH)") << endl << endl;
    }

    // Hitung Akurasi
    double accuracy = (double)correct_predictions / num_samples * 100.0;
    cout << "Akurasi Total: " << accuracy << "% (" << correct_predictions << "/" << num_samples << ")" << endl;

    return 0;
}