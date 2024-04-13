#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <stdio.h>

#include "neural_network.h"

enum optimizer_ids {
    NN_OPTIMIZER_SGD = 0,
    NN_OPTIMIZER_ADAM
};

typedef struct {
    float (*train)(neural_network *nn, float *inputs, float *expected_outputs, unsigned int minibatch_size, void *state);

    // NOTE: the number of parameters and their types are per optimizer
    void (*initialize_state)(void **state, neural_network *nn, ...);
    //void (*reset_state)(void *state);
    void (*free_state)(void *state);

    //void (*save_state)(void *state, FILE *file);
    //void (*load_state)(void **state, FILE *file);
    // TODO: add more training outputs (accuracy etc.)
} optimizer;

optimizer get_optimizer(unsigned int optimizer_id);

#endif // OPTIMIZER_H
