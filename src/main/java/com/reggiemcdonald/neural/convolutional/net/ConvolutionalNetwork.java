package com.reggiemcdonald.neural.convolutional.net;

import com.reggiemcdonald.neural.convolutional.net.layer.CNNLayer;
import com.reggiemcdonald.neural.convolutional.net.layer.*;

import java.util.ArrayList;
import java.util.List;

public class ConvolutionalNetwork {

    private int inputDimension, hiddenLayers, stride;

    private List<CNNLayer> inputs;
    private List<CNNLayer> convolutionals;
    private List<CNNLayer> poolings;

    private CNNLayer sigmoidalOutput;
    private CNNLayer softmaxOutput;

    /**
     * A really rough constructor :( TODO make this better
     * @param inputLayerSizes      an int[] of perfect squares describing the sizes of the input layers
     * @param convolutionalSizes   an int[] of perfect squares describing the sizes of the convolutions (keep small)
     * @param poolingSizes         an int[] of the pooling sizes
     * @param numberOfOutputs      the number of neurons in the output layer
     * @param hasSigmoidal         set to true if a sigmoidal output layer should be included
     */
    public ConvolutionalNetwork (int[] inputLayerSizes, int[] convolutionalSizes,
                                 int[] poolingSizes, int numberOfOutputs, boolean hasSigmoidal) {

        if (inputLayerSizes == null || inputLayerSizes.length == 0)
            throw new RuntimeException("Initialization Error: Must have at least one input layer");
        if (convolutionalSizes == null || convolutionalSizes.length == 0)
            throw new RuntimeException ("Initialization Error: Must have at least one convolution");
        if (poolingSizes == null || poolingSizes.length == 0)
            throw new RuntimeException ("Initialization Error: Must have at least one pooling layer");
        createInputs       (inputLayerSizes);
        createConvolutions (convolutionalSizes);

        createPoolings (poolingSizes);
        createOutputs  (hasSigmoidal, poolingSizes[poolingSizes.length-1], numberOfOutputs);
    }

    private void createInputs (int[] sizes) {
        inputs = new ArrayList<>(sizes.length);
        for (int i : sizes)
            inputs.add (new InputLayer(i));
    }

    private void createConvolutions (int[] sizes) {
        // TODO: CNN Init
        convolutionals = new ArrayList<>(sizes.length);
        for (int i : sizes)
            convolutionals.add (new ConvolutionLayer(i));
    }

    private void createPoolings (int[] sizes) {
        // TODO: CNN Init
        poolings = new ArrayList<>(sizes.length);
        for (int i : sizes)
            poolings.add (new PoolingLayer(i));
    }

    private void createOutputs (boolean hasSigmoidal, int sigmoidalSize, int softmaxSize) {
        // TODO: CNN init
        if (hasSigmoidal)
            sigmoidalOutput = new SigmoidalLayer(sigmoidalSize);
        softmaxOutput   = new SoftmaxLayer(softmaxSize);
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
