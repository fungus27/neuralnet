#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>

#include "activation.h"

typedef struct {
    unsigned char layer_count; // must be at least 2
    unsigned int *layer_sizes;
    unsigned int max_layer_size; // NOTE: without the input layer
    float **layers;
    float **weights; // array of matrices
    float **biases;
    unsigned int node_count;
    unsigned int weight_count;
    unsigned int bias_count;
    activation_function *activations; // per layer 
} neural_network;

void initialize_nn(neural_network *nn, unsigned int layer_count, unsigned int *layer_sizes, unsigned int *activation_ids);
void free_nn(neural_network *nn);

void feed_forward(neural_network *nn);
void evaluate_nn(neural_network *nn, float *input, float *output);

void print_nn_params(neural_network *nn);

void save_nn(neural_network *nn, FILE *file);
void load_nn(neural_network *nn, FILE *file);

#endif // NEURAL_NETWORK_H
