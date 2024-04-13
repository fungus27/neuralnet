#include "loss.h"

float mean_square(float x, float y) {
    float t = x - y;
    return t * t;
}

float mean_square_der(float x, float y) {
    return 2.0f * (x - y);
}

loss_function get_loss(unsigned int loss_id) {
    loss_function l_mean_square = {NN_LOSS_MEAN_SQUARE, &mean_square, &mean_square_der};
    switch (loss_id) {
        case NN_LOSS_MEAN_SQUARE:
            return l_mean_square;
    }
}
