#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <blis.h>

static const float zero = 0.0f;
static const float one = 1.0f;

typedef struct {
    unsigned int layer_count; // must be at least 2
    unsigned int *layer_sizes;
    unsigned int max_layer_size;
    float **layers;
    float **weights; // array of matrices
    float **biases;
    unsigned int node_count;
    unsigned int weight_count;
    unsigned int bias_count;
    // TODO: parametarize the following:
    // loss function
    // activation function per layer (or per node even)
} neural_network;

unsigned int max(unsigned int *arr, unsigned int n) {
    unsigned int m = 0;
    for (unsigned int i = 0; i < n; ++i) {
        if (arr[i] > m)
            m = arr[i];
    }
    return m;
}

float relu(float x) {
    return x >= 0.0f ? x : 0.0f;
}

float relu_der(float fx) {
    return fx >= 0.0f ? 1.0f : 0.0f;
}

float mean_square(float x, float y) {
    float t = x - y;
    return t * t;
}

float mean_square_der(float x, float y) {
    return 2 * (x - y);
}


void init_nn(unsigned int layer_count, unsigned int *layer_sizes, neural_network *nn) {
    nn->layer_count = layer_count;
    nn->layer_sizes = malloc(layer_count * sizeof(unsigned int));
    memcpy(nn->layer_sizes, layer_sizes, layer_count * sizeof(unsigned int));

    nn->max_layer_size = max(layer_sizes, layer_count);

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
    for (unsigned int i = 0; i < nn->weight_count; ++i)
        nn->weights[0][i] = rand() / (float)RAND_MAX;

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
}

void feed_forward(neural_network *nn) {

    //bli_sprintv("beginning layer", nn->layer_sizes[0], nn->layers[0], 1, "%4.1f", "");
    for (unsigned int i = 1; i < nn->layer_count; ++i) {
        // TODO: inspect efficiency
        bli_scopyv(BLIS_NO_CONJUGATE, nn->layer_sizes[i], nn->biases[i - 1], 1, nn->layers[i], 1);
        // a_i = 0.0 * a_i + 1.0 * w_{i - 1} * a_{i - 1} = w_{i - 1} * a_{i + 1}
        //                                              m                   n                     1.0   w_{i - 1}                                  a_{i - 1}             1.0    a_{i}
        bli_sgemv(BLIS_NO_TRANSPOSE, BLIS_NO_CONJUGATE, nn->layer_sizes[i], nn->layer_sizes[i-1], &one, nn->weights[i - 1], 1, nn->layer_sizes[i], nn->layers[i - 1], 1, &one, nn->layers[i], 1);


        for (unsigned int j = 0; j < nn->layer_sizes[i]; ++j)
            nn->layers[i][j] = relu(nn->layers[i][j]);

        //bli_sprintv("feed layer", nn->layer_sizes[i], nn->layers[i], 1, "%4.1f", "");
    }
}

// calculates gradient for each weight matrix and bias vector using backpropagation
// adds to given matrix pointers
void calculate_gradients(neural_network *nn, float *expected_values, float **w_gradients, float **b_gradients) {
    float *der_vec = malloc(nn->max_layer_size * sizeof(float));
    float *next_der_vec = malloc(nn->max_layer_size * sizeof(float));

    // calculate the last layer derivative vector
    for (unsigned int i = 0; i < nn->layer_sizes[nn->layer_count - 1]; ++i)
        der_vec[i] = mean_square_der(nn->layers[nn->layer_count - 1][i], expected_values[i]) * relu_der(nn->layers[nn->layer_count - 1][i]);

    // add this vector to the bias gradient
    bli_saddv(BLIS_NO_CONJUGATE, nn->layer_sizes[nn->layer_count - 1], der_vec, 1, b_gradients[nn->layer_count - 2], 1);

    for (int i = nn->layer_count - 2; i >= 0; --i) {

        // calculate gradient for current weight matrix (outer product of current layer and der_vec)
        //bli_sprintv("der vec", nn->layer_sizes[i + 1], der_vec, 1, "%4.1f", "");
        //bli_sprintv("current layer", nn->layer_sizes[i], nn->layers[i], 1, "%4.1f", "");
        bli_sger(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, nn->layer_sizes[i + 1], nn->layer_sizes[i], &one, der_vec, 1, nn->layers[i], 1, w_gradients[i], 1, nn->layer_sizes[i + 1]);
        //bli_sprintm("updated weight gradient", nn->layer_sizes[i+1], nn->layer_sizes[i], w_gradients[i], 1, nn->layer_sizes[i + 1], "%4.1f", "");

        // TODO: optimize this
        if (i == 0)
            continue;

        // calculate the next node derivative vector from the previous one 
        bli_sgemv(BLIS_TRANSPOSE, BLIS_NO_CONJUGATE, nn->layer_sizes[i + 1], nn->layer_sizes[i], &one, nn->weights[i], 1, nn->layer_sizes[i + 1], der_vec, 1, &zero, next_der_vec, 1); 

        // multiply each node derivative element with the derivative of the loss function w.r.t its input to 'convert' it into a bias derivative (used for the chain rule and to update the bias gradient)
        for (unsigned int j = 0; j < nn->layer_sizes[i]; ++j)
            next_der_vec[j] *= relu_der(nn->layers[i][j]);

        // add calculated bias derivative into the gradient 
        bli_saddv(BLIS_NO_CONJUGATE, nn->layer_sizes[i], next_der_vec, 1, b_gradients[i - 1], 1);
        //bli_sprintv("updated bias gradient", nn->layer_sizes[i], b_gradients[i - 1], 1, "%4.1f", "");

        //bli_sprintv("next der vec", nn->layer_sizes[i], next_der_vec, 1, "%4.1f", "");

        float *t = der_vec;
        der_vec = next_der_vec;
        next_der_vec = t;

        // for debugging
        /*  bli_sprintv("current layer vec", nn->layer_sizes[i], nn->layers[i], 1, "%4.1f", "");
            bli_sprintv("der vec", nn->layer_sizes[i + 1], der_vec, 1, "%4.1f", "");

            bli_sprintm("gradient", nn->layer_sizes[i+1], nn->layer_sizes[i], gradients[i], 1, nn->layer_sizes[i + 1], "%4.1f", "");


            bli_sprintv("v before matrix mult", nn->layer_sizes[i + 1], der_vec, 1, "%4.1f", "");
            bli_sprintm("weights (not transposed)", nn->layer_sizes[i + 1], nn->layer_sizes[i], nn->weights[i], 1, nn->layer_sizes[i + 1], "%4.1f", "");

            bli_sprintv("v after matrix mult", nn->layer_sizes[i], next_der_vec, 1, "%4.1f", ""); */
    }

    free(der_vec);
    free(next_der_vec);
}

void apply_gradients(neural_network *nn, float **w_gradients, float **b_gradients, float eta) {
    float minus_eta = -eta;

    bli_saxpyv(BLIS_NO_CONJUGATE, nn->weight_count, &minus_eta, w_gradients[0], 1, nn->weights[0], 1);
    bli_saxpyv(BLIS_NO_CONJUGATE, nn->bias_count, &minus_eta, b_gradients[0], 1, nn->biases[0], 1);

    //for (unsigned int i = 0; i < nn->layer_count - 1; ++i) {
    //    //printf("i:%u\n", i);
    //    //bli_sprintm("w before", nn->layer_sizes[i + 1], nn->layer_sizes[i], nn->weights[i], 1, nn->layer_sizes[i + 1], "%4.1f", "");
    //    bli_saxpym(0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_NO_TRANSPOSE, nn->layer_sizes[i + 1], nn->layer_sizes[i], &minus_eta, w_gradients[i], 1, nn->layer_sizes[i + 1], nn->weights[i], 1, nn->layer_sizes[i + 1]);
    //    //bli_sprintm("w after", nn->layer_sizes[i + 1], nn->layer_sizes[i], nn->weights[i], 1, nn->layer_sizes[i + 1], "%4.1f", "");

    //    //bli_sprintv("b before", nn->layer_sizes[i + 1], nn->biases[i], 1, "%4.1f", "");
    //    bli_saxpyv(BLIS_NO_CONJUGATE, nn->layer_sizes[i + 1], &minus_eta, b_gradients[i], 1, nn->biases[i], 1);
    //    //bli_sprintv("b after", nn->layer_sizes[i + 1], nn->biases[i], 1, "%4.1f", "");
    //}
}

void evaluate_nn(neural_network *nn, float *input, float *output) {
    bli_scopyv(BLIS_NO_CONJUGATE, nn->layer_sizes[0], input, 1, nn->layers[0], 1);
    feed_forward(nn);
    bli_scopyv(BLIS_NO_CONJUGATE, nn->layer_sizes[nn->layer_count - 1], nn->layers[nn->layer_count - 1], 1, output, 1);
}

void online_sgd_train(neural_network *nn, float *input, float *expected_output, float learning_rate) {

    float **w_gradients = malloc((nn->layer_count - 1) * sizeof(float*));    
    float **b_gradients = malloc((nn->layer_count - 1) * sizeof(float*));    
    
    float *w_ptr = calloc(nn->weight_count, sizeof(float));
    float *b_ptr = calloc(nn->bias_count, sizeof(float));

    for (unsigned int i = 0; i < nn->layer_count - 1; ++i) {
        w_gradients[i] = w_ptr;
        b_gradients[i] = b_ptr;
        w_ptr += nn->layer_sizes[i] * nn->layer_sizes[i + 1];
        b_ptr += nn->layer_sizes[i + 1];
    }

    bli_scopyv(BLIS_NO_CONJUGATE, nn->layer_sizes[0], input, 1, nn->layers[0], 1);
    feed_forward(nn);

    calculate_gradients(nn, expected_output, w_gradients, b_gradients);


    apply_gradients(nn, w_gradients, b_gradients, learning_rate);

    free(w_gradients[0]);
    free(b_gradients[0]);
    free(w_gradients);
    free(b_gradients);
}

void print_nn_params(neural_network *nn) {
    for (unsigned int i = 0; i < nn->layer_count - 1; ++i) {
        printf("weights from layer %u to %u\n", i, i + 1);
        bli_sprintm("", nn->layer_sizes[i + 1], nn->layer_sizes[i], nn->weights[i], 1, nn->layer_sizes[i + 1], "%4.1f", "");

        printf("biases of layer %u\n", i + 1);
        bli_sprintv("", nn->layer_sizes[i+1], nn->biases[i], 1, "%4.1f", "");
    }
}

void print_in_out(float *input, float *expected_output, unsigned int input_count, unsigned int output_count) {
    printf("input:");
    for (unsigned int i = 0; i < input_count; ++i)
        printf(" %4.1f", input[i]);
    printf("\n");

    printf("exp output:");
    for (unsigned int i = 0; i < output_count; ++i)
        printf(" %4.1f", expected_output[i]);
    printf("\n");
}

void minibatch_sgd_train(neural_network *nn, float *inputs, float *expected_outputs, unsigned int batch_size, float learning_rate) {

    unsigned int input_size = nn->layer_sizes[0];
    unsigned int output_size = nn->layer_sizes[nn->layer_count - 1];

    float **w_gradients = malloc((nn->layer_count - 1) * sizeof(float*));    
    float **b_gradients = malloc((nn->layer_count - 1) * sizeof(float*));    
    
    float *w_ptr = calloc(nn->weight_count, sizeof(float));
    float *b_ptr = calloc(nn->bias_count, sizeof(float));

    for (unsigned int i = 0; i < nn->layer_count - 1; ++i) {
        w_gradients[i] = w_ptr;
        b_gradients[i] = b_ptr;
        w_ptr += nn->layer_sizes[i] * nn->layer_sizes[i + 1];
        b_ptr += nn->layer_sizes[i + 1];
    }
    
    for (unsigned int i = 0; i < batch_size; ++i) {
        //printf("i:%u\n", i);
        bli_scopyv(BLIS_NO_CONJUGATE, nn->layer_sizes[0], inputs + i * input_size, 1, nn->layers[0], 1);
        feed_forward(nn);

        calculate_gradients(nn, expected_outputs + i * output_size, w_gradients, b_gradients);
    }

    //printf("before\n");
    //print_nn_params(nn);
    apply_gradients(nn, w_gradients, b_gradients, learning_rate/batch_size);
    //printf("after\n");
    //print_nn_params(nn);

    free(w_gradients[0]);
    free(b_gradients[0]);
    free(w_gradients);
    free(b_gradients);
}

typedef struct {
    float alpha;
    float beta_one;
    float beta_two;
    float epsilon;

    // state
    float *w_m, *b_m;
    float *w_v, *b_v;
    float b1t;
    float b2t;
} adam_state;

void init_adam_state(float learning_rate, float beta_one, float beta_two, float epsilon, neural_network *nn, adam_state *state) {
    state->alpha = learning_rate;
    state->beta_one = beta_one;
    state->beta_two = beta_two;
    state->epsilon = epsilon;

    state->w_m = calloc(nn->weight_count, sizeof(float));
    state->w_v = calloc(nn->weight_count, sizeof(float));
    state->b_m = calloc(nn->bias_count, sizeof(float));
    state->b_v = calloc(nn->bias_count, sizeof(float));
    state->b1t = 1.0f;
    state->b2t = 1.0f;
}

void free_adam_state(adam_state *state) {
    free(state->w_m);
    free(state->w_v);
    free(state->b_m);
    free(state->b_v);
}

void minibatch_adam_train(neural_network *nn, float *inputs, float *expected_outputs, unsigned int batch_size, adam_state *state) {

    unsigned int input_size = nn->layer_sizes[0];
    unsigned int output_size = nn->layer_sizes[nn->layer_count - 1];

    float **w_gradients = malloc((nn->layer_count - 1) * sizeof(float*));    
    float **b_gradients = malloc((nn->layer_count - 1) * sizeof(float*));    
    
    float *w_ptr = calloc(nn->weight_count, sizeof(float));
    float *b_ptr = calloc(nn->bias_count, sizeof(float));

    for (unsigned int i = 0; i < nn->layer_count - 1; ++i) {
        w_gradients[i] = w_ptr;
        b_gradients[i] = b_ptr;
        w_ptr += nn->layer_sizes[i] * nn->layer_sizes[i + 1];
        b_ptr += nn->layer_sizes[i + 1];
    }
    
    for (unsigned int i = 0; i < batch_size; ++i) {
        //printf("i:%u\n", i);
        bli_scopyv(BLIS_NO_CONJUGATE, nn->layer_sizes[0], inputs + i * input_size, 1, nn->layers[0], 1);
        feed_forward(nn);

        calculate_gradients(nn, expected_outputs + i * output_size, w_gradients, b_gradients);
    }

    for (unsigned int i = 0; i < nn->layer_count - 1; ++i)
            bli_sprintm("calculated w gradient", nn->layer_sizes[i + 1], nn->layer_sizes[i], w_gradients[i], 1, nn->layer_sizes[i + 1], "%4.2f", "");

    for (unsigned int i = 0; i < nn->layer_count - 1; ++i)
            bli_sprintv("calculated b gradient", nn->layer_sizes[i + 1], b_gradients[i], 1, "%4.2f", "");

    state->b1t *= state->beta_one;
    state->b2t *= state->beta_two;
    float alpha_t = state->alpha * sqrtf(1.0f - state->b2t) / (1.0f - state->b1t);

    // TODO: possibly replace sqrtf with a faster (less precise) sqrt algorithm

    // update weights
    for (unsigned int i = 0; i < nn->weight_count; ++i) {
        float g = w_gradients[0][i] / batch_size;
        state->w_m[i] = state->beta_one * state->w_m[i] + (1.0f - state->beta_one) * g; 
        state->w_v[i] = state->beta_two * state->w_v[i] + (1.0f - state->beta_two) * g*g;

        nn->weights[0][i] -= alpha_t * state->w_m[i] / (sqrtf(state->w_v[i]) + state->epsilon); 
    }

    // update biases
    for (unsigned int i = 0; i < nn->bias_count; ++i) {
        float g = b_gradients[0][i] / batch_size;
        state->b_m[i] = state->beta_one * state->b_m[i] + (1.0f - state->beta_one) * g; 
        state->b_v[i] = state->beta_two * state->b_v[i] + (1.0f - state->beta_two) * g*g;

        nn->biases[0][i] -= alpha_t * state->b_m[i] / (sqrtf(state->b_v[i]) + state->epsilon);
    }
    
    free(w_gradients[0]);
    free(b_gradients[0]);
    free(w_gradients);
    free(b_gradients);
}

int main() {
    // TODO: implement quick way to visualize data
    // TODO: use one dimensional arrays for efficiency maybe?
    // TODO: implement way to quickly wipe out matrices or switch between adding to them and overwriting them
    // TODO: zero optimze
    // TODO: clean up comments
    srand(1337);
    neural_network nn;
    unsigned int layer_sizes[] = {2, 2, 1};
    init_nn(3, layer_sizes, &nn);
    
    float inputs[] = {0.0f, 0.0f,
                      1.0f, 0.0f,
                      0.0f, 1.0f,
                      1.0f, 1.0f};

    float outputs[] = {0.0f, 1.0f, 1.0f, 0.0f};

    float output;

    adam_state state;
    init_adam_state(0.01f, 0.9f, 0.999f, 0.000000001f, &nn, &state);

    for (unsigned int i = 0; i < 5; ++i) {

        minibatch_sgd_train(&nn, inputs + 6, outputs + 3, 1, 0.1f);
        printf("i: %u\n", i);
        printf("\n");
        print_in_out(inputs + 6, outputs + 3, 2, 1);
        evaluate_nn(&nn, inputs + 6, &output);
        printf("out: %4.2f\n\n", output);
    }

    print_nn_params(&nn);

    free_adam_state(&state);
    free_nn(&nn);
}
