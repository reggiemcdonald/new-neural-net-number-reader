package com.reggiemcdonald.neural.convolutional.net;

import com.reggiemcdonald.neural.convolutional.net.layer.CNNLayer;
import com.reggiemcdonald.neural.convolutional.net.layer.*;

import java.util.ArrayList;
import java.util.List;

public class ConvolutionalNetwork {

    private int inputDimension, hiddenLayers, stride;

    private List<CNNLayer> inputs;
    private List<List<CNNLayer>> convolutionals;
    private List<CNNLayer> poolings;

    private CNNLayer sigmoidalOutput;
    private CNNLayer softmaxOutput;

    /**
     * A really rough constructor :( TODO make this better
     * @param inputLayerSizes      an int[] of dimensions for the square 2D input layers
     * @param convolutionalSizes   an int[] describing the sizes of the convolutions (keep small)
     * @param poolingSizes         an int[] of the pooling sizes
     * @param numberOfOutputs      the number of neurons in the output layer
     * @param stride               the stride length - TODO: Should this be an int[]
     * @param hasSigmoidal         set to true if a sigmoidal output layer should be included
     */
    public ConvolutionalNetwork (int[] inputLayerSizes, int[] convolutionalSizes,
                                 int[] poolingSizes, int numberOfOutputs, int stride, boolean hasSigmoidal) {

        if (inputLayerSizes == null || inputLayerSizes.length == 0)
            throw new RuntimeException("Initialization Error: Must have at least one input layer");
        if (convolutionalSizes == null || convolutionalSizes.length == 0)
            throw new RuntimeException ("Initialization Error: Must have at least one convolution");
        if (poolingSizes == null || poolingSizes.length == 0)
            throw new RuntimeException ("Initialization Error: Must have at least one pooling layer");
        createInputs       (inputLayerSizes, stride);
        createConvolutions (convolutionalSizes);

        createPoolings (poolingSizes);
        createOutputs  (hasSigmoidal, poolingSizes[poolingSizes.length-1], numberOfOutputs);
    }

    private void createInputs (int[] sizes, int stride) {
        inputs = new ArrayList<>(sizes.length);
        for (int i : sizes)
            inputs.add (new InputLayer(i, stride));
    }

    private void createConvolutions (int[] sizes) {
        // TODO: CNN Init
        convolutionals = new ArrayList<>(sizes.length);
        // The first convolutional Layer
        List<CNNLayer> l = new ArrayList<>();
        for (CNNLayer layer : inputs) {
            int dim_x, dim_y;
            dim_x = dim_y = getNextDim(layer.dim_x(), sizes[0]);
            l.add (new ConvolutionLayer(dim_x, dim_y, sizes[0]));
        }
        convolutionals.add (l);
        int pastDim = convolutionals.get(0).get(0).dim_x();
        for (int i = 1; i < sizes.length; i++) {
            int dim_x, dim_y;
            dim_x = dim_y = getNextDim(pastDim, sizes[i]);
            List<CNNLayer> l1 = new ArrayList<>();
            for (int j = 0; j < inputs.size(); j++) {
                l1.add (new ConvolutionLayer(dim_x, dim_y, sizes[i]));
            }
            convolutionals.add (l1);
            pastDim = dim_x;
        }
    }

    private void createPoolings (int[] sizes) {
        // TODO: CNN Init
    }

    private void createOutputs (boolean hasSigmoidal, int sigmoidalSize, int softmaxSize) {
        // TODO: CNN init
    }

    private void connect () {
        // Connect the input layers to the first convolutional layers
        connect (inputs, convolutionals.get(0));
        // connect the later convolutional layers to each other
        for (int i = 1; i < convolutionals.size()-1; i++) {
            connect(convolutionals.get(i), convolutionals.get(i+1));
        }
        // Connect the last convolutional to the pooling layer
        connect (convolutionals.get(convolutionals.size()-1), poolings);

        // TODO: Connect poolings to sigmoidal output

        // Connect the sigmoidal output to the softmax layer
        softmaxOutput.connect(sigmoidalOutput);

    }

    private void connect (List<CNNLayer> layersFrom, List<CNNLayer> layersTo) {
        assert(layersFrom.size() == layersTo.size());
        for (int i = 0; i < layersFrom.size(); i++)
            layersTo.get(i).connect(layersFrom.get(i));
    }

    private static int getNextDim (int dim, int window) {
        int x = 0, count = 0;
        while (x + window <= dim) {
            count++;
            x++;
        }
        return count;
    }
}
