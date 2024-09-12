#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <blis.h>

#include "activation.h"

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

void initialize_nn(neural_network *nn, unsigned int layer_count, unsigned int *layer_sizes, unsigned int *activation_ids) {
    nn->layer_count = layer_count;
    nn->layer_sizes = malloc(layer_count * sizeof(unsigned int));
    memcpy(nn->layer_sizes, layer_sizes, layer_count * sizeof(unsigned int));

    nn->max_layer_size = max(layer_sizes + 1, layer_count - 1);

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

    for (unsigned int i = 0; i < layer_count - 1; ++i) {
        for (unsigned int j = 0; j < layer_sizes[i] * layer_sizes[i + 1]; ++j) {
            if (activation_ids[i] == NN_ACTIVATION_ALG_SIGMOID || activation_ids[i] == NN_ACTIVATION_SOFTMAX)
                nn->weights[i][j] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f / sqrtf((float)layer_sizes[i]);
            else {
                float uniform = 0.0f;
                for (unsigned int k = 0; k < 12; ++k)
                    uniform += (float)rand()/RAND_MAX;
                uniform = (uniform - 6.0f) / 6.0f;
                nn->weights[i][j] = uniform * sqrtf(2.0f / layer_sizes[i]);
            }
        }
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

        nn->activations[i - 1].activation(nn->layers[i], nn->layer_sizes[i], nn->layers[i]);

        // TODO: to remove
        //for (unsigned int j = 0; j < nn->layer_sizes[i]; ++j)
        //    nn->layers[i][j] = nn->activations[i - 1].activation(nn->layers[i][j]);

        //bli_sprintv("feed layer", nn->layer_sizes[i], nn->layers[i], 1, "%4.1f", "");
    }
}


void evaluate_nn(neural_network *nn, float *input, float *output) {
    bli_scopyv(BLIS_NO_CONJUGATE, nn->layer_sizes[0], input, 1, nn->layers[0], 1);
    feed_forward(nn);
    bli_scopyv(BLIS_NO_CONJUGATE, nn->layer_sizes[nn->layer_count - 1], nn->layers[nn->layer_count - 1], 1, output, 1);
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

    unsigned int *activation_ids = malloc((layer_count - 1) * sizeof(unsigned int));
    fread(activation_ids, sizeof(unsigned int), layer_count - 1, file);

    initialize_nn(nn, layer_count, layer_sizes, activation_ids);

    fread(nn->weights[0], sizeof(float), nn->weight_count, file);
    fread(nn->biases[0], sizeof(float), nn->bias_count, file);

    free(layer_sizes);
    free(activation_ids);
}
