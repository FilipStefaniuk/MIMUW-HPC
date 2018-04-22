#include "model.hh"
#include <vector>
#include <stdio.h>
#include <stdlib.h>

Model::Model(Matrix *input, Matrix *output)
 : input(input), output(output), layers(),
   gs(), gsT(), ds() {}


void Model::add(Layer *layer) {
    layers.push_back(layer);
}

void Model::build() {
    
    unsigned input_size = input->height;
    gs.push_back(this->input);
    ds.push_back(genMatrix(input_size, input->width, 0));
    gsT.push_back(genMatrix(input->width, input_size, 0));
    
    for (Layer *layer : layers) {
        
        layer->build(input_size);

        input_size = layer->getSize();

        gs.push_back(genMatrix(input_size, input->width, 0));
        ds.push_back(genMatrix(input_size, input->width, 0));
        gsT.push_back(genMatrix(input->width, input_size, 0));
    }
}

double Model::evaluate(Matrix *pred_val, Matrix *true_val) {

    int correct_preds = 0;

    for (int i = 0; i < pred_val->width; ++i) {
        
        float max_pred = pred_val->elements[i];
        float max_true = true_val->elements[i];
        int arg_max_pred = 0, arg_max_true = 0;
        
        for(int j = 0; j < pred_val->height; ++j) {
            if (pred_val->elements[pred_val->width * j + i] > max_pred) {
                max_pred = pred_val->elements[pred_val->width * j + i];
                arg_max_pred = j;
            }

            if (true_val->elements[true_val->width * j + i] > max_true) {
                max_true = true_val->elements[true_val->width * j + i];
                arg_max_true = j;
            }        
        }

        if (arg_max_pred == arg_max_true) {
            correct_preds += 1;
        }
    }

    return ((double) correct_preds) / pred_val->width;
}

void Model::fit(unsigned epochs, float learning_rate, float eps) {
    
    for (int i = 0; i < epochs; ++i) {
        
        printf("Epoch %d\n", i);
        
        // Forward Pass
        // printf("--- FORWARD PASS ---\n");
        for (int j = 0; j < layers.size(); ++j) {
            // printf("LAYER: %d\n", j);
            layers[j]->forward_pass(gs[j], gs[j + 1]);
            // matPrint(gs[j + 1]);
            // matPrintSize(gs[j + 1]);
        }

        for (int j = 0; j < gs.size(); ++j) {
            matTranspose(gs[j], gsT[j]);
        }

        // Evaluate
        printf("--- EVALUATE ---\n");
        // matPrint(gs[gs.size() - 1]);
        // matPrint(output);
        double accuracy = evaluate(gs[gs.size() - 1], output);
        printf("ACCURACY: %f\n", accuracy);

        // Derrivative of cost function
        // printf("--- DELTA ---\n");
        matSub(gs[gs.size() - 1], output, ds[ds.size() - 1]);
        // matPrint(ds[ds.size() - 1]);

        // Backward Pass
        // printf("--- BACKWARD PASS ---\n");
        for (int j = layers.size() - 1; j >= 0; --j) {
            // printf("LAYER %d\n", j);
            layers[j]->backward_pass(ds[j + 1], gs[j + 1], ds[j]);
            // matPrint(ds[j]);
        }

        // printf("----- UPDATE WEIGHTS ------\n");

        for (int j = 0; j < layers.size(); ++j) {
            layers[j]->update_weights(gsT[j], ds[j + 1], 
                    learning_rate/(float)this->input->height);
        }
    }
}