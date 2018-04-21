#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

#define INPUT 5// 4096
#define LAYER_1 50 // 8192
// #define LAYER_2 6144
// #define LAYER_3 3072
// #define LAYER_4 1024
#define OUTPUT 5

#define N 20

double evaluate(Matrix *pred_val, Matrix *true_val) {

    int correct_preds = 0;
    // printf("--PREDICTIONS--\n");
    // printf("PRED | CORRECT\n");
    for (int i = 0; i < pred_val->height; ++i) {
        
        float max_pred = pred_val->elements[pred_val->width * i];
        float max_true = true_val->elements[true_val->width * i];
        int arg_max_pred = 0, arg_max_true = 0;
        
        for(int j = 0; j < pred_val->width; ++j) {
            if (pred_val->elements[pred_val->width * i + j] > max_pred) {
                max_pred = pred_val->elements[pred_val->width * i + j];
                arg_max_pred = j;
            }

            if (true_val->elements[true_val->width * i + j] > max_true) {
                max_true = true_val->elements[true_val->width * i + j];
                arg_max_true = j;
            }        
        }

        // printf("%d  %d\n", arg_max_pred, arg_max_true);

        if (arg_max_pred == arg_max_true) {
            correct_preds += 1;
        }
    }
    // printf("\n");

    return ((double) correct_preds) / pred_val->height;
}

void fit(float *data_X, float *data_Y, int len, float eps, float learning_rate, int epochs, int random) {

    printf("Running Neural Network\n");

    // Matrix g0;
    // g0.height = len;
    // g0.width = INPUT;
    // g0.elements = data_X;


    // Matrix y;
    // y.height = len;
    // y.width = OUTPUT;
    // y.elements = data_Y;

    //----------------------------------------
    // Test input
    Matrix *g0 = genRandomMatrix(INPUT, N);
    Matrix *y = genMatrix(OUTPUT, N, 0);

    for (int i = 0; i < N; ++i) {
        y->elements[y->width*i + (i % y->width)] = 1;
    }
    printf("DATA_X:\n");
    matPrint(g0);
    printf("DATA_Y\n");
    matPrint(y);
    //----------------------------------------

    // printf("Allocated input matrices\n");

    // Allocate memory
    printf("Weight matrices:\n");
    Matrix *w1 = genRandomMatrix(LAYER_1, INPUT);
    printf("WEIGHTS INPUT -> LAYER 1\n");
    matPrint(w1);
    Matrix *w2 = genRandomMatrix(OUTPUT, LAYER_1);
    printf("WEIGHTS LAYER 1 -> OUTPUT\n");
    matPrint(w2);
    // Matrix *w3 = genRandomMatrix(LAYER_3, LAYER_2);
    // Matrix *w4 = genRandomMatrix(LAYER_4, LAYER_3);
    // Matrix *w5 = genRandomMatrix(OUTPUT, LAYER_4);


    Matrix *dw1 = genRandomMatrix(LAYER_1, INPUT);
    Matrix *dw2 = genRandomMatrix(OUTPUT, LAYER_1);
    
    printf("Allocated weight matrices\n");

    Matrix *g1 = genMatrix(LAYER_1, N, 0);
    Matrix *g2 = genMatrix(OUTPUT, N, 0);
    // Matrix *g3 = genMatrix(LAYER_3, len, 0);
    // Matrix *g4 = genMatrix(LAYER_4, len, 0);
    // Matrix *g5 = genMatrix(OUTPUT, len, 0);

    Matrix *d1 = genMatrix(LAYER_1, N, 0);
    Matrix *d2 = genMatrix(OUTPUT, N, 0);
    // Matrix *d3 = genMatrix(LAYER_3, len, 0);
    // Matrix *d4 = genMatrix(LAYER_4, len, 0);
    // Matrix *d5 = genMatrix(OUTPUT, len, 0);

    printf("Allocated result matrices\n");

    Matrix *minus_y = genMatrix(OUTPUT, N, 0);
    matSum(minus_y, y, minus_y);
    matScalarMul(minus_y, -1.0f);

    double score = 0.0;

    printf("Allocated variables...\n");

    // Gradient Descent
    for (int i = 0; i < 2000; ++i) {
        
        printf("-------------\n");
        printf("Epoch %d\n", i);
        printf("-------------\n");

        // Forward Pass
        // printf("INPUT X W1 -> LAYER_1\n");
        matMul(g0, w1, g1);
        // matPrint(g1);
        // printf("RESULT 1 AFTER RELU\n");
        matRelu(g1);
        // matPrint(g1);
        
        // printf("LAYER_1 X W2 -> OUTPUT\n");
        matMul(g1, w2, g2);
        // matPrint(g2);
        // printf("RESULT AFTER SOFTMAX\n");
        matSoftmax(g2);
        // matPrint(g2);

        //---------------------------------
        // Evaluate result
        //---------------------------------

        printf("Evaluate:\n");
        score = evaluate(g2, y);
        printf("ACCURACY:  %f\n", score);

        // Backward pass

        // delta
        // printf("DELTA OUTPUT\n");
        matSum(g2, minus_y, d2);
        // matPrint(d2);

        // Push one layer back
        Matrix *w2t = genTransposed(w2);
        // printf("TRANSPOSED WEIGHTS LAYER1->OUT\n");
        // matPrint(w2t);
        matMul(d2, w2t, d1);
        // printf("DELTA_A LAYER 1\n");
        // matPrint(d1);
        matReluBack(g1);
        // printf("RELU BACKWARD MASK\n");
        // matPrint(g1);
        // printf("DELTA LAYER 1");
        matElMul(d1, g1, d1);
        // matPrint(d1);
        freeMatrix(w2t);


        Matrix *g1t = genTransposed(g1);
        Matrix *g0t = genTransposed(g0);

        // Sum weights
        // printf("OLD W INPUT-> LAYER_1\n");
        // matPrint(w1);
        // printf("OLD W LAYER_1 -> OUTPUT\n");
        // matPrint(w2);

        // printf("dW INPUT -> LAYER_1\n");
        matMul(g0t, d1, dw1);      
        // matPrint(dw1);
        // printf("dW LAYER_1 -> OUT\n");
        matMul(g1t, d2, dw2);
        // matPrint(dw2);

        matScalarMul(dw2, -0.1/(float) len);
        matScalarMul(dw1, -0.1/(float) len);
        
        // printf("UPDATE_WEIGHTS\n");
        // printf("W1 INPUT->LAYER_1\n");
        matSum(w1, dw1, w1);
        // matPrint(w1);  
        // printf("W2 LAYER_1->OUTPUT\n");
        matSum(w2, dw2, w2);
        // matPrint(w2);

        freeMatrix(g1t);
        freeMatrix(g0t);
    }
}