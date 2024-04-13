#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>

#include "loss.h"
#include "activation.h"

typedef struct {
    unsigned char layer_count; // must be at least 2
    unsigned int *layer_sizes;
    unsigned int max_layer_size;
    float **layers;
    float **weights; // array of matrices
    float **biases;
    unsigned int node_count;
    unsigned int weight_count;
    unsigned int bias_count;
    loss_function loss; 
    activation_function *activations; // per layer 
} neural_network;

void initialize_nn(neural_network *nn, unsigned int layer_count, unsigned int *layer_sizes, unsigned int loss_id, unsigned int *activation_ids);
void free_nn(neural_network *nn);

void feed_forward(neural_network *nn);
void evaluate_nn(neural_network *nn, float *input, float *output);
float test_class_nn(neural_network *nn, float *test_inputs, unsigned char *test_outputs, unsigned int set_count, float *accuracy);

float calculate_gradients(neural_network *nn, float *expected_values, float **w_gradients, float **b_gradients);
void apply_gradients(neural_network *nn, float **w_gradients, float **b_gradients, float eta);

void print_nn_params(neural_network *nn);

void save_nn(neural_network *nn, FILE *file);
void load_nn(neural_network *nn, FILE *file);

#endif // NEURAL_NETWORK_H
