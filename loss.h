#ifndef LOSS_H
#define LOSS_H

enum loss_ids {
    NN_LOSS_MEAN_SQUARE = 0,
    NN_LOSS_CATEGORICAL_CROSS_ENTROPY
};

typedef struct {
    unsigned int id;
    float (*loss)(float x, float expected);
    float (*derivative)(float x, float expected);
} loss_function;

loss_function get_loss(unsigned int loss_id);

#endif // LOSS_H
