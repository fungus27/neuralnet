#include <math.h>

#include "loss.h"

static const float epsilon = 0.000001f;

float mean_square(float x, float y) {
    float t = x - y;
    return t * t;
}

float mean_square_der(float x, float y) {
    return 2.0f * (x - y);
}

float categorical_cross_entropy(float x, float y) {
    return -(y * logf(x + epsilon));
}

float categorical_cross_entropy_der(float x, float y) {
    return -(y / (x + epsilon));
}

loss_function get_loss(unsigned int loss_id) {
    loss_function l_mean_square = {NN_LOSS_MEAN_SQUARE, &mean_square, &mean_square_der};
    loss_function l_categorical_cross_entropy = {NN_LOSS_CATEGORICAL_CROSS_ENTROPY, &categorical_cross_entropy, &categorical_cross_entropy_der};
    switch (loss_id) {
        case NN_LOSS_MEAN_SQUARE:
            return l_mean_square;
        case NN_LOSS_CATEGORICAL_CROSS_ENTROPY:
            return l_categorical_cross_entropy;
    }
}
