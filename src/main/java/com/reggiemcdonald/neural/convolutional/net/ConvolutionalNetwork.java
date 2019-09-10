package com.reggiemcdonald.neural.convolutional.net;

import com.reggiemcdonald.neural.convolutional.net.layer.CNNLayer;
import com.reggiemcdonald.neural.convolutional.net.layer.*;
import com.reggiemcdonald.neural.convolutional.net.util.LayerUtilities;

import java.util.ArrayList;
import java.util.List;

public class ConvolutionalNetwork {

    private InputAggregateLayer inputAggregateLayer;
    private List<CAggregateLayer> convolutionalLayers;

    private CNNLayer sigmoidalOutput;
    private CNNLayer softmaxOutput;

    /**
     * Another bad constructor
     * @param inputLayerDimension an int of dimensions of the square 2D input layers
     * @param convolutionWindowSizes and int[] of the window sizes for each convolution layer
     * @param poolingWindowSizes an int[] of the window sizes for each of the poolings
     * @param depths an int[] of the depths for the input and convolution layers
     * @param numberOfOutputs the number of neurons to be in the output layer
     * @param stride an int[] of stride lengths
     * @param hasSigmoidalLayer true if the network should have a sigmoidal layer
     */
    public ConvolutionalNetwork (int inputLayerDimension, int[] convolutionWindowSizes, int[] poolingWindowSizes,
                             int[] depths, int numberOfOutputs, int[] stride, boolean hasSigmoidalLayer) {
        if (inputLayerDimension == 0)
            throw new RuntimeException("Init Error: Must have at least one input layer");
        if (convolutionWindowSizes == null || convolutionWindowSizes.length == 0)
            throw new RuntimeException ("Init Error: Must have at least one convolution");
        if (poolingWindowSizes == null || poolingWindowSizes.length == 0)
            throw new RuntimeException ("Init Error: Must have at least one pooling layer");
        if (convolutionWindowSizes.length != poolingWindowSizes.length)
            throw new RuntimeException("Init Error: Length of convolution sizes and pooling sizes should be equal");
        if (depths == null || depths.length != convolutionWindowSizes.length + 1)
            throw new RuntimeException("Init Error: Insufficient number of entries in the depths array");
        if (stride == null || stride.length != convolutionWindowSizes.length + poolingWindowSizes.length)
            throw new RuntimeException("Init Error: Insufficient stride measurements");

        inputAggregateLayer = new InputAggregateLayer (inputLayerDimension, depths[0], stride[0]);
        createConvolutions (convolutionWindowSizes, poolingWindowSizes, depths, stride);
    }

    private void createConvolutions (int[] cSizes, int[] pSizes, int[] depths, int[] stride) {
        convolutionalLayers = new ArrayList<>(cSizes.length);
        // Make the first CAggregateLayer immediately following the Input Layer
        int cDim, pDim;
        cDim = LayerUtilities.nextDimension(inputAggregateLayer.dim(), cSizes[0], stride[0]);
        pDim = LayerUtilities.nextDimension(cDim, pSizes[0], stride[1]);
        convolutionalLayers.add (new CAggregateLayer(cDim, pDim, depths[0], cSizes[0], pSizes[0], stride[0], stride[1]));

        for (int i = 1; i < cSizes.length; i++) {
            cDim = LayerUtilities.nextDimension(convolutionalLayers.get(i-1).cDim(), cSizes[i], stride[i+1]);
            pDim = LayerUtilities.nextDimension(convolutionalLayers.get(i-1).pDim(), pSizes[i], stride[i+2]);
            convolutionalLayers.add (new CAggregateLayer(cDim, pDim, depths[i], cSizes[i], pSizes[i], stride[i+1], stride[i+2]));
        }

    }

    private void createOutputs (boolean hasSigmoidal, int sigmoidalSize, int softmaxSize) {
        // TODO: CNN init
    }

    private void connect () {
        // Connect the input layers to the first convolutional layers
        connect (inputAggregateLayer.inputLayers(), convolutionalLayers.get(0).convolutionalLayers());

        // connect the later convolutional layers to each other
        for (int i = 0; i < convolutionalLayers.size()-1; i++) {
            connect(convolutionalLayers.get(i).convolutionalLayers(),
                    convolutionalLayers.get(i+1).convolutionalLayers());
        }

        // TODO: Create the next layers first, then run connection

        // Connect the sigmoidal output to the softmax layer
//        softmaxOutput.connect(sigmoidalOutput);

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
