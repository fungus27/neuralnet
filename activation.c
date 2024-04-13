#include "activation.h"

#define ABS(x) ( (x) > 0 ? (x) : -(x) )

float relu(float x) {
    return x >= 0.0f ? x : 0.0f;
}

float relu_der(float fx) {
    return fx > 0.0f ? 1.0f : 0.0f;
}

float alg_sigmoid(float x) {
    return (x / (1.0f + ABS(x)) + 1.0f) / 2.0f;
}

float alg_sigmoid_der(float fx) {
    float t = ABS(2.0f * fx - 1.0f);
    return (1.0f - 2.0f * t + t*t) / 2.0f;
}

activation_function get_activation(unsigned int activation_id) {
    activation_function a_relu = {NN_ACTIVATION_RELU, &relu, &relu_der};
    activation_function a_alg_sigmoid = {NN_ACTIVATION_ALG_SIGMOID, &alg_sigmoid, &alg_sigmoid_der};

    switch (activation_id) {
        case NN_ACTIVATION_RELU:
            return a_relu;
        case NN_ACTIVATION_ALG_SIGMOID:
            return a_alg_sigmoid;
    }
}
