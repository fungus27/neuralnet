#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <blis.h>
#include <assert.h>

#include "neural_network.h"
#include "loss.h"
#include "activation.h"
#include "optimizer.h"

#define SWAP32(num) (((num)>>24)&0xff) | (((num)<<8)&0xff0000) | (((num)>>8)&0xff00) | (((num)<<24)&0xff000000)

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

void read_data(char *images_name, char *labels_name, float *inputs, float *expected_outputs, unsigned char *labels) {
    
    {
        FILE *f_images = fopen(images_name, "r");

        int magic_number; 
        fread(&magic_number, sizeof(int), 1, f_images);
        magic_number = SWAP32(magic_number);
        assert(magic_number == 2051);

        int number_of_images;
        fread(&number_of_images, sizeof(int), 1, f_images);
        number_of_images = SWAP32(number_of_images);
        //assert(number_of_images == 60000);

        int rows, cols;
        fread(&rows, sizeof(int), 1, f_images);
        fread(&cols, sizeof(int), 1, f_images);
        rows = SWAP32(rows);
        cols = SWAP32(cols);
        assert(rows == 28 && cols == 28);

        for (unsigned int i = 0; i < number_of_images; ++i) {
            for (unsigned int j = 0; j < rows * cols; ++j) {
                unsigned char pixel;
                fread(&pixel, sizeof(unsigned char), 1, f_images);
                inputs[i * rows * cols + j] = pixel/255.0f;
            }
        }

        fclose(f_images);
    }

    {
        FILE *f_labels = fopen(labels_name, "r");

        int magic_number; 
        fread(&magic_number, sizeof(int), 1, f_labels);
        magic_number = SWAP32(magic_number);
        assert(magic_number == 2049);

        int number_of_labels;
        fread(&number_of_labels, sizeof(int), 1, f_labels);
        number_of_labels = SWAP32(number_of_labels);
        //assert(number_of_labels == 60000);

        for (unsigned int i = 0; i < number_of_labels; ++i) {
            fread(&labels[i], sizeof(unsigned char), 1, f_labels);

            if (expected_outputs) {
                for (unsigned int j = 0; j < 10; ++j)
                    expected_outputs[i * 10 + j] = (j == labels[i]) ? 1.0f : 0.0f;
            }

        }

        fclose(f_labels);
    }
}

void softmax(float *input, unsigned int count, float *exped) {
#ifdef NN_SOFTMAX_UNSTABLE
    float sum = 0.0f;

    for (unsigned int i = 0; i < count; ++i) {
        exped[i] = expf(input[i]);
        sum += exped[i];
    }

    bli_sinvscalv(BLIS_NO_CONJUGATE, count, &sum, exped, 1);
#else
    unsigned int max_index;
    bli_samaxv(count, input, 1, &max_index);
    float max = input[max_index];

    float sum = 0.0f;

    for (unsigned int i = 0; i < count; ++i) {
        exped[i] = expf(input[i] - max);
        sum += exped[i];
    }

    bli_sinvscalv(BLIS_NO_CONJUGATE, count, &sum, exped, 1);
#endif
}

//int main() {
//    // TODO: move loss and gradient calculation into optimizer
//    // TODO: implement various layer types (softmax)
//    // TODO: merge state into optimizer struct
//    // TODO: plurarize initialization function names
//    srand(10933);
//
//    float *inputs = malloc(60000 * 28 * 28 * sizeof(float));
//    float *expected_outputs = malloc(60000 * 10 * sizeof(float));
//    unsigned char *labels = malloc(60000 * sizeof(unsigned char));
//    read_data("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte", inputs, expected_outputs, labels);
//
//    float *test_inputs = malloc(10000 * 28 * 28 * sizeof(float));
//    unsigned char *test_labels = malloc(10000 * sizeof(unsigned char));
//    read_data("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte", test_inputs, NULL, test_labels);
//
//    neural_network nn;
//    unsigned int layer_sizes[] = {784, 300, 2, 300, 784};
//    unsigned int activation_ids[] = {NN_ACTIVATION_RELU, NN_ACTIVATION_ALG_SIGMOID, NN_ACTIVATION_RELU, NN_ACTIVATION_ALG_SIGMOID};
//
//    initialize_nn(&nn, 4, layer_sizes, NN_LOSS_MEAN_SQUARE, activation_ids);
//
//    optimizer adam = get_optimizer(NN_OPTIMIZER_ADAM);
//    void *state;
//    adam.initialize_state(&state, &nn, 0.01f, 0.99f, 0.999f, 0.00000001f);
//
//    unsigned int batch_size = 100;
//
//    for (unsigned int j = 0; j < 20; ++j) {
//        float loss = 0.0f;
//        for (unsigned int i = 0; i < 60000; i += batch_size) {
//            loss += adam.train(&nn, inputs + i * 28 * 28, inputs + i * 28 * 28, batch_size, state);
//        }
//        printf("epoch: %u, avg loss: %4.3f\n", j, loss/60000 * batch_size);
//    }
//
//    adam.free_state(state);
//
//    FILE *saved = fopen("reconstructor.nn", "w");
//    save_nn(&nn, saved);
//    fclose(saved);
//
//    free_nn(&nn);
//
//    free(inputs);
//    free(expected_outputs);
//    free(labels);
//}

int main() {
    float input[6] = {-4.5f, -1.f, 3.0f, -2.0f, 8.0f, 4.0f};
    float output[6];
    softmax(input, 6, output);
    bli_sprintv("", 6, output, 1, "%4.2f", "");
}
