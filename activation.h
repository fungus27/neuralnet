#ifndef ACTIVATION_H
#define ACTIVATION_H

enum activation_ids {
    NN_ACTIVATION_RELU = 0,
    NN_ACTIVATION_ALG_SIGMOID
};

typedef struct {
    unsigned int id;
    float (*activation)(float x);
    // w.r.t f(x)
    float (*derivative)(float fx);
} activation_function;

activation_function get_activation(unsigned int activation_id);

#endif // ACTIVATION_H
