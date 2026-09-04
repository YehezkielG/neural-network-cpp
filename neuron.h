#include <iostream>
#include <vector>
#include <random>    
#include <algorithm> 
#include <array>

using namespace std;

namespace nn {
    inline static random_device rd;
    inline static mt19937 generator{rd()};
    inline static normal_distribution<double> distribution{0.0, 1.0}; 
    
    class linear{       
        public: 
            Tensor weight;
            Tensor biases;
            Tensor y;
            bool set_bias = true;
        
        linear(int in, int out, bool bias = true){
            set_bias = bias;
            vector<vector<double>> init_weight;
            vector<vector<double>> init_biases;

            if (bias) init_biases.resize(1, vector<double>(out));

            init_weight.resize(in,  vector<double>(out));

            for (int i = 0; i < in; i++) {
                for (int j = 0; j < out; j++) {
                    init_weight[i][j] = distribution(generator);
                }
            }

            if(bias){
                for (int j = 0; j < out; j++) {
                    init_biases[0][j] = distribution(generator);
                }
                
            this->biases = Tensor(init_biases);
            }
        this->weight = Tensor(init_weight);
        }
        Tensor operator()(const Tensor& input){
            this->y = input.matmul(weight);
            if (set_bias){
                return this->y + biases;
            }
            return this->y;
        }
    };
};
