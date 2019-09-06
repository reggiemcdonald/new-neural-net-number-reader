package com.reggiemcdonald.neural.net;

import java.util.List;

public class ConvolutionalNetwork {

    private int inputDimension, hiddenLayers, stride;

    private Layer input;
    private Layer sigmoidalOutput; // Feeds into the Softmax Output layer
    private Layer softmaxOutput; // Softmax output layer (optimal for CNNs)

    private List<Layer> hidden;

    /**
     * Here we define a constructor for the convolutional network - probably will change
     * @param inputDimension - the dimension of the input layer
     * @param hiddenLayers - The number of hidden layers
     * @param stride - The stride length
     */
    public ConvolutionalNetwork (int inputDimension, int hiddenLayers, int stride) {
        // TODO
        this.inputDimension = inputDimension;
        this.hiddenLayers   = hiddenLayers;
        this.stride         = stride;
        create3DHiddenLayer ();
        create2DInputLayer  ();
        createSigmoidalOutputLayer ();
        createSoftmaxOutputLayer   ();
    }

    /**
     * Creates a 3-dimensional hidden layer
     */
    private void create3DHiddenLayer () {
        // TODO: CNN Init
    }

    /**
     * Creates a 2-dimensional input layer
     */
    private void create2DInputLayer () {
        // TODO: CNN Init
    }

    /**
     * Creates the sigmoidal output layer that will feed
     * into the Softmax input layer
     */
    private void createSigmoidalOutputLayer () {
        // TODO: CNN Init
    }

    /**
     * Creates the Softmax output layer that approximates a probability
     * distribution for the likelihood of the result
     */
    private void createSoftmaxOutputLayer () {
        // TODO: CNN Init
    }

    /**
     * Performs the conversion from coordinate to linear index (x,y) -> z
     * using dim for modulo arithmetic
     * @param x
     * @param y
     * @param dim
     * @return
     */
    private int toLinearIndex (int x, int y, int dim) {
        // TODO: CNN Misc
        return 0; // Stub
    }
}
