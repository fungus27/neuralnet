#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <blis.h>
#include <stdarg.h>

#include "loss.h"
#include "neural_network.h"

#include "optimizer.h"

#define SWAP(a, b, t) {t = a; a = b; b = t;}

#define OPTIMIZER_COUNT 2
optimizer optimizers[OPTIMIZER_COUNT];

static const float zero = 0.0f;
static const float one = 1.0f;

void alloc_gradients(neural_network *nn, float ***w_gradients, float ***b_gradients) {
    *w_gradients = malloc((nn->layer_count - 1) * sizeof(float*));
    *b_gradients = malloc((nn->layer_count - 1) * sizeof(float*));    

    float *w_ptr = calloc(nn->weight_count, sizeof(float));
    float *b_ptr = calloc(nn->bias_count, sizeof(float));

    for (unsigned int i = 0; i < nn->layer_count - 1; ++i) {
        (*w_gradients)[i] = w_ptr;
        (*b_gradients)[i] = b_ptr;
        w_ptr += nn->layer_sizes[i] * nn->layer_sizes[i + 1];
        b_ptr += nn->layer_sizes[i + 1];
    }
}

// calculates gradient for each weight matrix and bias vector using backpropagation
// adds to given matrix pointers
// returns loss
float calculate_gradients(neural_network *nn, loss_function loss, float *expected_values, float **w_gradients, float **b_gradients) {
    // TODO: optimize activation derivatives with if statements (vector/jacobian)
    float *der_vec = malloc(nn->max_layer_size * sizeof(float));
    float *next_der_vec = malloc(nn->max_layer_size * sizeof(float));
    float *jacobian = malloc(nn->max_layer_size * nn->max_layer_size * sizeof(float));
    float *t;

    // calculate the last layer derivative vector and the loss value
    float loss_val = 0.0f;
    for (unsigned int i = 0; i < nn->layer_sizes[nn->layer_count - 1]; ++i) {
        der_vec[i] = loss.derivative(nn->layers[nn->layer_count - 1][i], expected_values[i]);
        loss_val += loss.loss(nn->layers[nn->layer_count - 1][i], expected_values[i]);
    }
    
    nn->activations[nn->layer_count - 2].derivative(nn->layers[nn->layer_count - 1], nn->layer_sizes[nn->layer_count - 1], jacobian);
    bli_sgemv(BLIS_NO_TRANSPOSE, BLIS_NO_CONJUGATE, nn->layer_sizes[nn->layer_count - 1], nn->layer_sizes[nn->layer_count - 1], &one, jacobian, 1, nn->layer_sizes[nn->layer_count - 1], der_vec, 1, &zero, next_der_vec, 1);
    SWAP(der_vec, next_der_vec, t);

    // add this vector to the bias gradient
    bli_saddv(BLIS_NO_CONJUGATE, nn->layer_sizes[nn->layer_count - 1], der_vec, 1, b_gradients[nn->layer_count - 2], 1);

    for (int i = nn->layer_count - 2; i >= 1; --i) {

        // calculate gradient for current weight matrix (outer product of current layer and der_vec)
        bli_sger(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, nn->layer_sizes[i + 1], nn->layer_sizes[i], &one, der_vec, 1, nn->layers[i], 1, w_gradients[i], 1, nn->layer_sizes[i + 1]);

        // calculate the next node derivative vector from the previous one 
        bli_sgemv(BLIS_TRANSPOSE, BLIS_NO_CONJUGATE, nn->layer_sizes[i + 1], nn->layer_sizes[i], &one, nn->weights[i], 1, nn->layer_sizes[i + 1], der_vec, 1, &zero, next_der_vec, 1); 

        // multiply by jacobian for chain rule
        nn->activations[i].derivative(nn->layers[i], nn->layer_sizes[i], jacobian);
        bli_sgemv(BLIS_TRANSPOSE, BLIS_NO_CONJUGATE, nn->layer_sizes[i], nn->layer_sizes[i], &one, jacobian, 1, nn->layer_sizes[i], next_der_vec, 1, &zero, der_vec, 1);
        SWAP(next_der_vec, der_vec, t);

        // add calculated bias derivative into the gradient 
        bli_saddv(BLIS_NO_CONJUGATE, nn->layer_sizes[i], next_der_vec, 1, b_gradients[i - 1], 1);

        SWAP(der_vec, next_der_vec, t);
    }

    bli_sger(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, nn->layer_sizes[1], nn->layer_sizes[0], &one, der_vec, 1, nn->layers[0], 1, w_gradients[0], 1, nn->layer_sizes[1]);

    free(der_vec);
    free(next_der_vec);
    free(jacobian);

    return loss_val/*/nn->layer_sizes[nn->layer_count - 1]*/;
}

void apply_gradients(neural_network *nn, float **w_gradients, float **b_gradients, float eta) {
    float minus_eta = -eta;

    bli_saxpyv(BLIS_NO_CONJUGATE, nn->weight_count, &minus_eta, w_gradients[0], 1, nn->weights[0], 1);
    bli_saxpyv(BLIS_NO_CONJUGATE, nn->bias_count, &minus_eta, b_gradients[0], 1, nn->biases[0], 1);
}

// returns average loss and writes the classification accuracy into accuracy
float test_class_nn(neural_network *nn, unsigned int loss_id, float *test_inputs, unsigned char *labels, unsigned int set_count, float *accuracy) {
    unsigned int input_size = nn->layer_sizes[0];
    unsigned int output_size = nn->layer_sizes[nn->layer_count - 1];
    loss_function loss_func = get_loss(loss_id);

    float *output = malloc(output_size * sizeof(float));

    unsigned int correct = 0;
    float loss = 0.0f;

    for (unsigned int i = 0; i < set_count; ++i) {
        evaluate_nn(nn, test_inputs + i * input_size, output);
        unsigned int max_index = 0;
        float max_activation = output[0];

        for (unsigned int j = 0; j < output_size; ++j) {
            loss += loss_func.loss(output[j], (j == labels[i]) ? 1.0f : 0.0f);

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


// SGD


typedef struct {
    float learning_rate;
    // store the allocated memory in this struct to improve performance
    loss_function loss;
    float **w_gradients;
    float **b_gradients;
} sgd_state;

// args: float learning_rate
void sgd_init(void **state, neural_network *nn, unsigned int loss_id, ...) {
    *state = malloc(sizeof(sgd_state)); 

    va_list argptr;
    va_start(argptr, loss_id);

    sgd_state *sgd = (sgd_state*)(*state);
    sgd->learning_rate = (float)va_arg(argptr, double);

    va_end(argptr);

    sgd->loss = get_loss(loss_id);
    alloc_gradients(nn, &sgd->w_gradients, &sgd->b_gradients);
}

float sgd_train(neural_network *nn, float *inputs, float *expected_outputs, unsigned int batch_size, void *state) {
    unsigned int input_size = nn->layer_sizes[0];
    unsigned int output_size = nn->layer_sizes[nn->layer_count - 1];

    float **w_gradients = ((sgd_state*)state)->w_gradients;
    float **b_gradients = ((sgd_state*)state)->b_gradients;
    float learning_rate = ((sgd_state*)state)->learning_rate;
    loss_function loss = ((sgd_state*)state)->loss;
    
    float avg_loss = 0.0f;
    for (unsigned int i = 0; i < batch_size; ++i) {
        bli_scopyv(BLIS_NO_CONJUGATE, nn->layer_sizes[0], inputs + i * input_size, 1, nn->layers[0], 1);
        feed_forward(nn);

        avg_loss += calculate_gradients(nn, loss, expected_outputs + i * output_size, w_gradients, b_gradients);
    }
    avg_loss /= batch_size;

    apply_gradients(nn, w_gradients, b_gradients, learning_rate/batch_size);

    bli_ssetv(BLIS_NO_CONJUGATE, nn->weight_count, &zero, w_gradients[0], 1);
    bli_ssetv(BLIS_NO_CONJUGATE, nn->bias_count, &zero, b_gradients[0], 1);

    return avg_loss;
}

void sgd_free(void *state) {
    free(((sgd_state*)state)->w_gradients[0]);
    free(((sgd_state*)state)->b_gradients[0]);
    free(((sgd_state*)state)->w_gradients);
    free(((sgd_state*)state)->b_gradients);
    free(state);
}


// ADAM


typedef struct {
    float alpha;
    float beta_one;
    float beta_two;
    float epsilon;

    float *w_m, *b_m;
    float *w_v, *b_v;
    float b1t;
    float b2t;
    loss_function loss;
    float **w_gradients;
    float **b_gradients;
} adam_state;

// args: float learning_rate, float beta_one, float beta_two, float epsilon
void adam_init(void **state, neural_network *nn, unsigned int loss_id, ...) {
    *state = malloc(sizeof(adam_state));

    va_list argptr;
    va_start(argptr, loss_id);

    adam_state *adam = (adam_state*)(*state);
    adam->alpha = (float)va_arg(argptr, double);
    adam->beta_one = (float)va_arg(argptr, double);
    adam->beta_two = (float)va_arg(argptr, double);
    adam->epsilon = (float)va_arg(argptr, double);

    adam->w_m = calloc(nn->weight_count, sizeof(float));
    adam->w_v = calloc(nn->weight_count, sizeof(float));
    adam->b_m = calloc(nn->bias_count, sizeof(float));
    adam->b_v = calloc(nn->bias_count, sizeof(float));
    adam->b1t = 1.0f;
    adam->b2t = 1.0f;

    adam->loss = get_loss(loss_id);
    alloc_gradients(nn, &adam->w_gradients, &adam->b_gradients);
}

float adam_train(neural_network *nn, float *inputs, float *expected_outputs, unsigned int batch_size, void *v_state) {
    unsigned int input_size = nn->layer_sizes[0];
    unsigned int output_size = nn->layer_sizes[nn->layer_count - 1];

    adam_state *state = (adam_state*)v_state;
    float **w_gradients = state->w_gradients;
    float **b_gradients = state->b_gradients;

    float avg_loss = 0.0f;
    for (unsigned int i = 0; i < batch_size; ++i) {
        bli_scopyv(BLIS_NO_CONJUGATE, nn->layer_sizes[0], inputs + i * input_size, 1, nn->layers[0], 1);
        feed_forward(nn);

        avg_loss += calculate_gradients(nn, state->loss, expected_outputs + i * output_size, w_gradients, b_gradients);
    }
    avg_loss /= batch_size;

    state->b1t *= state->beta_one;
    state->b2t *= state->beta_two;
    float alpha_t = state->alpha * sqrtf(1.0f - state->b2t) / (1.0f - state->b1t);

    // todo: possibly replace sqrtf with a faster (less precise) sqrt algorithm

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

    bli_ssetv(BLIS_NO_CONJUGATE, nn->weight_count, &zero, w_gradients[0], 1);
    bli_ssetv(BLIS_NO_CONJUGATE, nn->bias_count, &zero, b_gradients[0], 1);
    
    return avg_loss;
}

void adam_free(void *state) {
    free(((adam_state*)state)->w_m);
    free(((adam_state*)state)->b_m);
    free(((adam_state*)state)->w_v);
    free(((adam_state*)state)->b_v);

    free(((adam_state*)state)->w_gradients[0]);
    free(((adam_state*)state)->b_gradients[0]);
    free(((adam_state*)state)->w_gradients);
    free(((adam_state*)state)->b_gradients);
    free(state);
}

optimizer get_optimizer(unsigned int optimizer_id) {
    optimizer sgd;
    sgd.train = &sgd_train;
    sgd.initialize_state = &sgd_init;
    sgd.free_state = &sgd_free;

    optimizer adam;
    adam.train = &adam_train;
    adam.initialize_state = &adam_init;
    adam.free_state = &adam_free;

    switch (optimizer_id) {
        case NN_OPTIMIZER_SGD:
            return sgd;
        case NN_OPTIMIZER_ADAM:
            return adam;
    }
}
