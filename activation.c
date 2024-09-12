#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <blis.h>

#include "activation.h"

#define ABS(x) ( (x) > 0 ? (x) : -(x) )

void relu(float *x, unsigned int count, float *output) {
    for (unsigned int i = 0; i < count; ++i)
        output[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
}

void relu_der(float *fx, unsigned int count, float *jacobian) {
    memset(jacobian, 0, count*count*sizeof(float));
    for (unsigned int i = 0; i < count; ++i)
        jacobian[i * count + i] = (fx[i] > 0.0f) ? 1.0f : 0.0f;
}

void alg_sigmoid(float *x, unsigned int count, float *output) {
    for (unsigned int i = 0; i < count; ++i)
        output[i] = (x[i] / (1.0f + ABS(x[i])) + 1.0f) / 2.0f;
}

void alg_sigmoid_der(float *fx, unsigned int count, float *jacobian) {
    memset(jacobian, 0, count*count*sizeof(float));
    float t;
    for (unsigned int i = 0; i < count; ++i) {
        t = ABS(2.0f * fx[i] - 1.0f);
        jacobian[i * count + i] = (1.0f - 2.0f * t + t*t) / 2.0f;
    }
}

void softmax(float *input, unsigned int count, float *output) {
#ifdef NN_SOFTMAX_UNSTABLE
    float sum = 0.0f;

    for (unsigned int i = 0; i < count; ++i) {
        output[i] = expf(input[i]);
        sum += output[i];
    }

    bli_sinvscalv(BLIS_NO_CONJUGATE, count, &sum, output, 1);
#else
    float max = input[0];
    for (unsigned int i = 1; i < count; ++i)
        if (input[i] > max)
            max = input[i];

    float sum = 0.0f;

    for (unsigned int i = 0; i < count; ++i) {
        output[i] = expf(input[i] - max);
        sum += output[i];
    }
    bli_sinvscalv(BLIS_NO_CONJUGATE, count, &sum, output, 1);
#endif
}

void softmax_der(float *fx, unsigned int count, float *jacobian) {
    for (unsigned int i = 0; i < count; ++i)
        for (unsigned int j = 0; j < count; ++j)
            jacobian[j * count + i] = fx[i] * ( (float)(i == j) * 1.0f - fx[j]);
}

activation_function get_activation(unsigned int activation_id) {
    switch (activation_id) {
        case NN_ACTIVATION_RELU:
            return (activation_function){NN_ACTIVATION_RELU, &relu, &relu_der};
        case NN_ACTIVATION_ALG_SIGMOID:
            return (activation_function){NN_ACTIVATION_ALG_SIGMOID, &alg_sigmoid, &alg_sigmoid_der};
        case NN_ACTIVATION_SOFTMAX:
            return (activation_function){NN_ACTIVATION_SOFTMAX, &softmax, &softmax_der};
    }
}
