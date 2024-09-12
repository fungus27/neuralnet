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
            for (unsigned int x = 0; x < rows; ++x) {
                for (int y = cols - 1; y >= 0; --y) {
                    unsigned char pixel;
                    fread(&pixel, sizeof(unsigned char), 1, f_images);
                    inputs[i * rows * cols + y * cols + x] = pixel/255.0f;
                }
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

int main() {
    // TODO: move loss and gradient calculation into optimizer
    // TODO: implement various layer types (softmax)
    // TODO: merge state into optimizer struct
    // TODO: plurarize initialization function names

    srand(10953);

    float *inputs = malloc(60000 * 28 * 28 * sizeof(float));
    float *expected_outputs = malloc(60000 * 10 * sizeof(float));
    unsigned char *labels = malloc(60000 * sizeof(unsigned char));
    read_data("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte", inputs, expected_outputs, labels);

    float *test_inputs = malloc(10000 * 28 * 28 * sizeof(float));
    unsigned char *test_labels = malloc(10000 * sizeof(unsigned char));
    read_data("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte", test_inputs, NULL, test_labels);

    neural_network nn;
    unsigned int layer_sizes[] = {784, 300, 10};
    //unsigned int activation_ids[] = {NN_ACTIVATION_RELU, NN_ACTIVATION_SOFTMAX};
    unsigned int activation_ids[] = {NN_ACTIVATION_ALG_SIGMOID, NN_ACTIVATION_ALG_SIGMOID};
    //unsigned int activation_ids[] = {NN_ACTIVATION_RELU, NN_ACTIVATION_ALG_SIGMOID};

    initialize_nn(&nn, 3, layer_sizes, activation_ids);

    optimizer optimizer = get_optimizer(NN_OPTIMIZER_ADAM);
    void *state;
    optimizer.initialize_state(&state, &nn, NN_LOSS_MEAN_SQUARE, 0.0005f, 0.99f, 0.999f, 0.00001f);

    unsigned int batch_size = 100;

    //float last_loss = 0.0f;
    //for (unsigned int j = 0; j < 1000; ++j) {
    //    float loss = optimizer.train(&nn, inputs, expected_outputs, batch_size, state);

    //    printf("epoch: %u, avg loss: %4.3f, loss delta: %4.3f\n", j, loss, loss - last_loss);
    //    last_loss = loss;
    //}

    float loss;
    for (unsigned int i = 0; i < 5; ++i) {
        loss = 0.0f;
        for (unsigned int j = 0; j < 60000; j += batch_size)
            loss += optimizer.train(&nn, inputs + j * 28 * 28, expected_outputs + j * 10, batch_size, state);
        float acc;
        float test_loss = test_class_nn(&nn, NN_LOSS_MEAN_SQUARE, test_inputs, test_labels, 10000, &acc);
        printf("epoch: %u, avg loss: %4.3f, avg test loss: %4.3f, acc: %4.2f\n", i, loss/60000 * batch_size, test_loss, acc);
    }

    optimizer.free_state(state);

    //FILE *saved = fopen("test.nn", "w");
    //save_nn(&nn, saved);
    //fclose(saved);

    free_nn(&nn);

    free(inputs);
    free(expected_outputs);
    free(labels);
    free(test_inputs);
    free(test_labels);
}

//int main() {
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
//    unsigned int layer_sizes[] = {784, 300, 50, 10};
//    unsigned int activation_ids[] = {NN_ACTIVATION_RELU, NN_ACTIVATION_RELU, NN_ACTIVATION_SOFTMAX};
//
//    initialize_nn(&nn, 4, layer_sizes, activation_ids);
//    float output[10];
//    char temp[100];
//
//    for (unsigned int i = 0; i < 60000; ++i) {
//        evaluate_nn(&nn, inputs + i * 28 * 28, output);
//        //for (unsigned int j = 0; j < 10; ++j)
//        //    printf("%4.2f ", output[j]);
//        //printf("\n");
//        //scanf("%s", temp);
//    }
//
//    free_nn(&nn);
//    free(inputs);
//    free(expected_outputs);
//    free(labels);
//    free(test_inputs);
//    free(test_labels);
//}

