package com.reggiemcdonald.neural.convolutional.net;

import com.reggiemcdonald.neural.convolutional.net.decipher.Decipher;
import com.reggiemcdonald.neural.convolutional.net.decipher.IndexOfMaxDecipher;
import com.reggiemcdonald.neural.convolutional.net.layer.*;
import com.reggiemcdonald.neural.convolutional.net.util.LayerUtilities;

import java.util.ArrayList;
import java.util.List;

public class ConvolutionalNetwork {

    private InputAggregateLayer inputAggregateLayer;

    private List<CAggregateLayer> convolutionalLayers;
    private List<CNNLayer> sigmoidalOutputs;

    private CNNLayer softmaxOutput;

    private Decipher<?> decipher = new IndexOfMaxDecipher(); // Default behaviour

    /**
     * Another bad constructor
     * @param inputLayerDimension an int of dimensions of the square 2D input layers
     * @param convolutionWindowSizes and int[] of the window sizes for each convolution layer
     * @param poolingWindowSizes an int[] of the window sizes for each of the poolings
     * @param depths an int[] of the depths for the input and convolution layers
     * @param sigmoidalOutputSizes an int[] of sizes for additional sigmoidal outputs (beyond flatten layer)
     * @param numberOfOutputs the number of neurons to be in the output layer
     * @param stride an int[] of stride lengths
     * @param hasSigmoidalLayer true if the network should have a sigmoidal layer
     */
    public ConvolutionalNetwork (int inputLayerDimension, int[] convolutionWindowSizes, int[] poolingWindowSizes,
                             int[] depths, int[] sigmoidalOutputSizes, int numberOfOutputs, int[] stride, boolean hasSigmoidalLayer) {
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
        if (hasSigmoidalLayer && (sigmoidalOutputSizes == null))
            throw new RuntimeException("Init Error: Must have at least one sigmoidal output layer; preferably more");

        inputAggregateLayer = new InputAggregateLayer (inputLayerDimension, depths[0], stride[0]);
        createConvolutions (convolutionWindowSizes, poolingWindowSizes, depths, stride);
        createOutputs (sigmoidalOutputSizes, numberOfOutputs, hasSigmoidalLayer);
        connect ();
        System.out.println("Here");
    }

    /**
     * Create the convolutional layers of this complete layer
     * @param cSizes a list of window sizes for the convolutions
     * @param pSizes a list of window sizes for the associated poolings
     * @param depths a list of depths at each level of the network
     * @param stride a list of stride measurements
     * Note: cSizes.length == pSizes.length
     */
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

    /**
     * Produces the output layers of this network and assumes the existance of a single flatten layer
     * for feed-forward
     * @param sigmoidalOutputSizes
     * @param softmaxSize
     * @param hasSigmoidalLayer
     */
    private void createOutputs (int[] sigmoidalOutputSizes, int softmaxSize, boolean hasSigmoidalLayer) {
        // TODO: CNN init
        sigmoidalOutputs = new ArrayList<>();
        CAggregateLayer lastConvLayer = convolutionalLayers.get(convolutionalLayers.size()-1);
        sigmoidalOutputs.add (lastConvLayer.flatten());
        if (hasSigmoidalLayer) {
            for (int sigmoidalLayerSize : sigmoidalOutputSizes)
                sigmoidalOutputs.add (new SigmoidalLayer(sigmoidalLayerSize));
        }
        softmaxOutput = new SoftmaxLayer (softmaxSize);
    }

    /**
     * Connects the layers
     */
    private void connect () {
        // Connect the input layers to the first convolutional layers
        convolutionalLayers.get(0).connectToThis (inputAggregateLayer.inputLayers());
        // connect the later convolutional layers to each other
        for (int i = 1; i < convolutionalLayers.size(); i++) {
            convolutionalLayers
                    .get(i)
                    .connectToThis(convolutionalLayers.get(i-1).poolingLayers());
        }

        // Connect the sigmoidal layers
        for (int i = 1; i < sigmoidalOutputs.size(); i++) {
            sigmoidalOutputs
                    .get(i)
                    .connect(sigmoidalOutputs.get(i-1), 0);
        }
        // Connect the final sigmoidal layer to the softmax layer
        softmaxOutput.connect(sigmoidalOutputs.get(sigmoidalOutputs.size()-1), 0);
    }

    /**
     * Connects adjacent layers
     * @param layersFrom
     * @param layersTo
     */
    private void connect (List<CNNLayer> layersFrom, List<CNNLayer> layersTo) {
        assert(layersFrom.size() == layersTo.size());
        for (int i = 0; i < layersTo.size(); i++)
            for (int j = 0; j < layersFrom.size(); j++)
                layersTo.get(i).connect(layersFrom.get(j), j);
    }

    /**
     * Initialize the signal propagation
     */
    public void propagate () {
        inputAggregateLayer.propagate();

        for (Propagatable p : convolutionalLayers)
            p.propagate();

        for (Propagatable p : sigmoidalOutputs)
            p.propagate();
    }

    public void setDecipher (Decipher decipher) {
        this.decipher = decipher;
    }
}
