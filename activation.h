#ifndef ACTIVATION_H
#define ACTIVATION_H

enum activation_ids {
    NN_ACTIVATION_RELU = 0,
    NN_ACTIVATION_ALG_SIGMOID,
    NN_ACTIVATION_SOFTMAX
};

typedef struct {
    unsigned int id;
    void (*activation)(float *x, unsigned int count, float *output);
    // w.r.t f(x)
    void (*derivative)(float *fx, unsigned int count, float *jacobian);
} activation_function;

activation_function get_activation(unsigned int activation_id);

void softmax_der(float *fx, unsigned int count, float *jacobian);
void softmax(float *input, unsigned int count, float *output);
void alg_sigmoid(float *x, unsigned int count, float *output);
void alg_sigmoid_der(float *fx, unsigned int count, float *jacobian);

#endif // ACTIVATION_H
