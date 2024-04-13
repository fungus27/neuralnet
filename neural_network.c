#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <blis.h>

#include "activation.h"
#include "loss.h"

#include "neural_network.h"

static const float zero = 0.0f;
static const float one = 1.0f;

unsigned int max(unsigned int *arr, unsigned int n) {
    unsigned int m = 0;
    for (unsigned int i = 0; i < n; ++i) {
        if (arr[i] > m)
            m = arr[i];
    }
    return m;
}

void initialize_nn(neural_network *nn, unsigned int layer_count, unsigned int *layer_sizes, unsigned int loss_id, unsigned int *activation_ids) {
    nn->layer_count = layer_count;
    nn->layer_sizes = malloc(layer_count * sizeof(unsigned int));
    memcpy(nn->layer_sizes, layer_sizes, layer_count * sizeof(unsigned int));

    nn->max_layer_size = max(layer_sizes, layer_count);

    nn->loss = get_loss(loss_id);
    nn->activations = malloc((layer_count - 1) * sizeof(activation_function));
    for (unsigned int i = 0; i < layer_count - 1; ++i)
        nn->activations[i] = get_activation(activation_ids[i]);

    // calculate counts
    nn->node_count = 0;
    nn->weight_count = 0;
    nn->bias_count = 0;

    for (unsigned int i = 1; i < layer_count; ++i) {
        nn->bias_count += layer_sizes[i];
        nn->weight_count += layer_sizes[i - 1] * layer_sizes[i];
    }
    nn->node_count = nn->bias_count + layer_sizes[0]; 
    
    nn->layers = malloc(layer_count * sizeof(float*));
    float *node_ptr = calloc(nn->node_count, sizeof(float));
    for (unsigned int i = 0; i < layer_count; ++i) {
        nn->layers[i] = node_ptr;
        node_ptr += layer_sizes[i];
    }

    nn->weights = malloc((layer_count - 1) * sizeof(float*));
    float *weight_ptr = calloc(nn->weight_count, sizeof(float));
    for (unsigned int i = 0; i < layer_count - 1; ++i) {
        nn->weights[i] = weight_ptr;
        weight_ptr += layer_sizes[i] * layer_sizes[i + 1];
    }
    for (unsigned int i = 0; i < nn->weight_count; ++i) {
        nn->weights[0][i] = (rand()/(float)RAND_MAX - 0.5f) * 2.0f;
    }

    nn->biases = malloc((layer_count - 1) * sizeof(float*));
    float *bias_ptr = calloc(nn->bias_count, sizeof(float));
    for (unsigned int i = 0; i < layer_count - 1; ++i) {
        nn->biases[i] = bias_ptr;
        bias_ptr += layer_sizes[i + 1];
    }
}

void free_nn(neural_network *nn) {
    free(nn->layers[0]);
    free(nn->weights[0]);
    free(nn->biases[0]);

    free(nn->layers);
    free(nn->weights);
    free(nn->biases);
    free(nn->layer_sizes);
    free(nn->activations);
}

void feed_forward(neural_network *nn) {

    //bli_sprintv("beginning layer", nn->layer_sizes[0], nn->layers[0], 1, "%4.1f", "");
    for (unsigned int i = 1; i < nn->layer_count; ++i) {
        // todo: inspect efficiency
        bli_scopyv(BLIS_NO_CONJUGATE, nn->layer_sizes[i], nn->biases[i - 1], 1, nn->layers[i], 1);
        // a_i = 0.0 * a_i + 1.0 * w_{i - 1} * a_{i - 1} = w_{i - 1} * a_{i + 1}
        //                                              m                   n                     1.0   w_{i - 1}                                  a_{i - 1}             1.0    a_{i}
        bli_sgemv(BLIS_NO_TRANSPOSE, BLIS_NO_CONJUGATE, nn->layer_sizes[i], nn->layer_sizes[i-1], &one, nn->weights[i - 1], 1, nn->layer_sizes[i], nn->layers[i - 1], 1, &one, nn->layers[i], 1);


        for (unsigned int j = 0; j < nn->layer_sizes[i]; ++j)
            nn->layers[i][j] = nn->activations[i - 1].activation(nn->layers[i][j]);

        //bli_sprintv("feed layer", nn->layer_sizes[i], nn->layers[i], 1, "%4.1f", "");
    }
}

// calculates gradient for each weight matrix and bias vector using backpropagation
// adds to given matrix pointers
// returns loss
float calculate_gradients(neural_network *nn, float *expected_values, float **w_gradients, float **b_gradients) {
    float *der_vec = malloc(nn->max_layer_size * sizeof(float));
    float *next_der_vec = malloc(nn->max_layer_size * sizeof(float));

    // calculate the last layer derivative vector and the loss value
    float loss = 0.0f;
    for (unsigned int i = 0; i < nn->layer_sizes[nn->layer_count - 1]; ++i) {
        der_vec[i] = nn->loss.derivative(nn->layers[nn->layer_count - 1][i], expected_values[i]) * nn->activations[nn->layer_count - 2].derivative(nn->layers[nn->layer_count - 1][i]);
        loss += nn->loss.loss(nn->layers[nn->layer_count - 1][i], expected_values[i]);
    }

    // add this vector to the bias gradient
    bli_saddv(BLIS_NO_CONJUGATE, nn->layer_sizes[nn->layer_count - 1], der_vec, 1, b_gradients[nn->layer_count - 2], 1);

    for (int i = nn->layer_count - 2; i >= 0; --i) {

        // calculate gradient for current weight matrix (outer product of current layer and der_vec)
        bli_sger(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, nn->layer_sizes[i + 1], nn->layer_sizes[i], &one, der_vec, 1, nn->layers[i], 1, w_gradients[i], 1, nn->layer_sizes[i + 1]);

        // todo: optimize this
        if (i == 0)
            continue;

        // calculate the next node derivative vector from the previous one 
        bli_sgemv(BLIS_TRANSPOSE, BLIS_NO_CONJUGATE, nn->layer_sizes[i + 1], nn->layer_sizes[i], &one, nn->weights[i], 1, nn->layer_sizes[i + 1], der_vec, 1, &zero, next_der_vec, 1); 

        // multiply each node derivative element with the derivative of the loss function w.r.t its input to 'convert' it into a bias derivative (used for the chain rule and to update the bias gradient)
        for (unsigned int j = 0; j < nn->layer_sizes[i]; ++j)
            next_der_vec[j] *= nn->activations[i].derivative(nn->layers[i][j]); // TODO: replace this with the jacobian matrix

        // add calculated bias derivative into the gradient 
        bli_saddv(BLIS_NO_CONJUGATE, nn->layer_sizes[i], next_der_vec, 1, b_gradients[i - 1], 1);

        float *t = der_vec;
        der_vec = next_der_vec;
        next_der_vec = t;
    }

    free(der_vec);
    free(next_der_vec);

    return loss;
}

void apply_gradients(neural_network *nn, float **w_gradients, float **b_gradients, float eta) {
    float minus_eta = -eta;

    bli_saxpyv(BLIS_NO_CONJUGATE, nn->weight_count, &minus_eta, w_gradients[0], 1, nn->weights[0], 1);
    bli_saxpyv(BLIS_NO_CONJUGATE, nn->bias_count, &minus_eta, b_gradients[0], 1, nn->biases[0], 1);
}

void evaluate_nn(neural_network *nn, float *input, float *output) {
    bli_scopyv(BLIS_NO_CONJUGATE, nn->layer_sizes[0], input, 1, nn->layers[0], 1);
    feed_forward(nn);
    bli_scopyv(BLIS_NO_CONJUGATE, nn->layer_sizes[nn->layer_count - 1], nn->layers[nn->layer_count - 1], 1, output, 1);
}

// returns average loss and writes the classification accuracy into accuracy
float test_class_nn(neural_network *nn, float *test_inputs, unsigned char *labels, unsigned int set_count, float *accuracy) {
    unsigned int input_size = nn->layer_sizes[0];
    unsigned int output_size = nn->layer_sizes[nn->layer_count - 1];

    float *output = malloc(output_size * sizeof(float));

    unsigned int correct = 0;
    float loss = 0.0f;

    for (unsigned int i = 0; i < set_count; ++i) {
        evaluate_nn(nn, test_inputs + i * input_size, output);
        unsigned int max_index = 0;
        float max_activation = output[0];

        for (unsigned int j = 0; j < output_size; ++j) {
            loss += nn->loss.loss(output[j], (j == labels[i]) ? 1.0f : 0.0f);

            if (output[j] > max_activation) {
                max_activation = output[j];
                max_index = j;
            }
        }

        if (max_index == labels[i])
            ++correct;

    }

    *accuracy = (float)correct/set_count;

    return loss/set_count;
}

void print_nn_params(neural_network *nn) {
    for (unsigned int i = 0; i < nn->layer_count - 1; ++i) {
        printf("weights from layer %u to %u\n", i, i + 1);
        bli_sprintm("", nn->layer_sizes[i + 1], nn->layer_sizes[i], nn->weights[i], 1, nn->layer_sizes[i + 1], "%4.1f", "");

        printf("biases of layer %u\n", i + 1);
        bli_sprintv("", nn->layer_sizes[i+1], nn->biases[i], 1, "%4.1f", "");
    }
}

void save_nn(neural_network *nn, FILE *file) {
    fwrite(&nn->layer_count, sizeof(char), 1, file);
    fwrite(nn->layer_sizes, sizeof(unsigned int), nn->layer_count, file);

    fwrite(&nn->loss.id, sizeof(unsigned int), 1, file);

    for (unsigned int i = 0; i < nn->layer_count - 1; ++i)
        fwrite(&nn->activations[i].id, sizeof(unsigned int), 1, file);

    fwrite(nn->weights[0], sizeof(float), nn->weight_count, file);
    fwrite(nn->biases[0], sizeof(float), nn->bias_count, file);
}

void load_nn(neural_network *nn, FILE *file) {
    unsigned char layer_count;
    fread(&layer_count, sizeof(char), 1, file);

    unsigned int *layer_sizes = malloc(layer_count * sizeof(unsigned int));
    fread(layer_sizes, sizeof(unsigned int), layer_count, file);

    unsigned int loss_id;
    fread(&loss_id, sizeof(unsigned int), 1, file);
    
    unsigned int *activation_ids = malloc((layer_count - 1) * sizeof(unsigned int));
    fread(activation_ids, sizeof(unsigned int), layer_count - 1, file);

    initialize_nn(nn, layer_count, layer_sizes, loss_id, activation_ids);

    fread(nn->weights[0], sizeof(float), nn->weight_count, file);
    fread(nn->biases[0], sizeof(float), nn->bias_count, file);

    free(layer_sizes);
    free(activation_ids);
}
