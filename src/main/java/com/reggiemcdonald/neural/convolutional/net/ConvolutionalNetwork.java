package com.reggiemcdonald.neural.convolutional.net;

import com.reggiemcdonald.neural.convolutional.net.decipher.Decipher;
import com.reggiemcdonald.neural.convolutional.net.decipher.IndexOfMaxDecipher;
import com.reggiemcdonald.neural.convolutional.net.layer.cnn.ConvolutionalPoolings;
import com.reggiemcdonald.neural.convolutional.net.layer.cnn.InputLayer;
import com.reggiemcdonald.neural.convolutional.net.layer.fc.FCInputLayer;
import com.reggiemcdonald.neural.convolutional.net.layer.fc.FullyConnectedLayer;
import com.reggiemcdonald.neural.convolutional.net.layer.fc.SigmoidalLayer;
import com.reggiemcdonald.neural.convolutional.net.layer.fc.SoftmaxLayer;
import com.reggiemcdonald.neural.convolutional.net.learning.layer.fc.FullyConnectedLayerLearner;
import com.reggiemcdonald.neural.convolutional.net.util.DefaultInputWrapper;
import com.reggiemcdonald.neural.convolutional.net.util.InputWrapper;
import com.reggiemcdonald.neural.convolutional.net.util.LayerUtilities;
import com.reggiemcdonald.neural.convolutional.net.util.Matrix;
import com.reggiemcdonald.neural.feedforward.res.NumberImage;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ConvolutionalNetwork {

    /**
     * The set of layers comprising the ConvNet
     */
    private InputLayer inputLayer;
    private List<ConvolutionalPoolings> convolutionalLayers;
    private FCInputLayer fcInputLayer;
    private List<FullyConnectedLayer> fullyConnectedLayers;
    private FullyConnectedLayer fcOutputLayer;

    /**
     * Adapters here. Instantiate defaults in the constructor
     */
    private Decipher<?> decipher;
    private InputWrapper inputWrapper;

    /**
     * Another bad constructor
     * @param inputLayerDimension an int of dimensions of the square 2D wrapInput layers
     * @param convolutionWindowSizes and int[] of the window sizes for each convolution layer
     * @param poolingWindowSizes an int[] of the window sizes for each of the poolings
     * @param depths an int[] of the depths for the wrapInput and convolution layers
     * @param sigmoidalOutputSizes an int[] of sizes for additional sigmoidal outputs (beyond flatten layer)
     * @param numberOfOutputs the number of neurons to be in the output layer
     * @param stride an int[] of stride lengths
     * @param hasSigmoidalLayer true if the network should have a sigmoidal layer
     */
    public ConvolutionalNetwork (int inputLayerDimension, int[] convolutionWindowSizes, int[] poolingWindowSizes,
                             int[] depths, int[] sigmoidalOutputSizes, int numberOfOutputs, int[] stride, boolean hasSigmoidalLayer) {
        // TODO: Create a ConvNet factory for this
        if (inputLayerDimension == 0)
            throw new RuntimeException("Init Error: Must have at least one wrapInput layer");
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

        decipher             = new IndexOfMaxDecipher();
        inputWrapper         = new DefaultInputWrapper();
        convolutionalLayers  = new ArrayList<>();
        fullyConnectedLayers = new ArrayList<>();
        inputLayer           = new InputLayer(inputLayerDimension, inputLayerDimension, depths[0]);

        createConvolutions(convolutionWindowSizes, poolingWindowSizes, depths, stride);
        createFullyConnectedLayers(sigmoidalOutputSizes, numberOfOutputs, hasSigmoidalLayer);
        connect();
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
        // Make the first ConvolutionalPoolings immediately following the Input Layer
        int cDim, pDim;
        cDim = LayerUtilities.nextDimension(inputLayer.dimX(), cSizes[0], stride[0]);
        pDim = LayerUtilities.nextDimension(cDim, pSizes[0], stride[1]);
        convolutionalLayers.add (new ConvolutionalPoolings(cDim, pDim, depths[1], depths[0], cSizes[0], pSizes[0], stride[0], stride[1]));

        for (int i = 1; i < cSizes.length; i++) {
            cDim = LayerUtilities.nextDimension(convolutionalLayers.get(i-1).cDim(), cSizes[i], stride[i+1]);
            pDim = LayerUtilities.nextDimension(convolutionalLayers.get(i-1).pDim(), pSizes[i], stride[i+2]);
            convolutionalLayers.add (new ConvolutionalPoolings(cDim, pDim, depths[i+1], depths[i], cSizes[i], pSizes[i], stride[i+1], stride[i+2]));
        }

    }

    /**
     * Produces the output layers of this network and assumes the existance of a single flatten layer
     * for feed-forward
     * @param sigmoidalOutputSizes
     * @param softmaxSize
     * @param hasSigmoidalLayer
     */
    private void createFullyConnectedLayers(int[] sigmoidalOutputSizes, int softmaxSize, boolean hasSigmoidalLayer) {

        int sizeFromConvolution = convolutionalLayers
                .get(convolutionalLayers.size() - 1)
                .poolingDensity();

        fcInputLayer = new FCInputLayer(sizeFromConvolution);

        if (hasSigmoidalLayer) {
            for (int sigmoidalLayerSize : sigmoidalOutputSizes)
                fullyConnectedLayers.add (new SigmoidalLayer(sigmoidalLayerSize));
        }

        fcOutputLayer = new SoftmaxLayer(softmaxSize);
    }

    /**
     * Connects the layers
     */
    private void connect () {

        // Connect the first input layer to the first sigmoidal layer
        fullyConnectedLayers.get(0).connect(fcInputLayer);

        // Connect the following layers
        for (int i = 1; i < fullyConnectedLayers.size(); i++) {
            fullyConnectedLayers
                    .get(i)
                    .connect(fullyConnectedLayers.get(i-1));
        }

        // Connect the final sigmoidal layer to the softmax layer
        fcOutputLayer.connect(fullyConnectedLayers.get(fullyConnectedLayers.size()-1));
    }

    /**
     * Initiate signal propagation
     */
    public void propagate () {
        // TODO
        double[][][] output = inputLayer.outputs();

        for (ConvolutionalPoolings cp : convolutionalLayers) {
            cp.propagate(output);
            output = cp.outputs();
        }

        double[] flattenedOutput = convolutionalLayers.get(convolutionalLayers.size() - 1).flatten();
        fcInputLayer.setAll(flattenedOutput);
        fcInputLayer.propagate();

        for (FullyConnectedLayer layer : fullyConnectedLayers)
            layer.propagate();

        fcOutputLayer.propagate();
    }

    /**
     * Change /255 when I get a chance to make this generic
     * @param input
     */
    public void input (double[][][] input) {
        for (double[][] d : input) {
            Matrix.elementWiseDivide(d, 255.);
        }
        inputLayer.set (input);
    }

    /**
     * Sets the output decipher for this network
     * @param decipher
     */
    public void setDecipher (Decipher decipher) {
        this.decipher = decipher;
    }

    /**
     * @return the decipher of this network
     */
    public Decipher decipher() {return decipher;}

    /**
     * @return the wrapInput wrapper of this network
     */
    public InputWrapper inputWrapper() {return inputWrapper;}

    /**
     * Set the wrapInput wrapper of this network
     * @param wrapper
     */
    public void setInputWrapper (InputWrapper wrapper) {
        this.inputWrapper = wrapper;
    }

    /**
     * Returns the output of this layer
     * @return
     */
    public double[] output() {
        return ((SoftmaxLayer) fcOutputLayer).output();
    }

    /**
     * Train the network
     * TODO: Make this generic - remove references to NumberImage
     * @param trainingData
     * @param epochs
     * @param batchSize
     * @param eta
     * @param verbose
     */
    public void learn (List<NumberImage> trainingData, int epochs, int batchSize, double eta, boolean verbose) {
        // Randomize the order of the images
        Collections.shuffle(trainingData);
        // Partition into batches of size batchSize
        List<List<NumberImage>> batches = new ArrayList<>();
        int idx = 0;
        for (int i = 0; i < trainingData.size() / batchSize; i++) {
            List<NumberImage> batch = new ArrayList<>();
            for (int j = 0; j < batchSize; j++) {
                batch.add(trainingData.get(idx));
                idx++;
            }
            batches.add(batch);
        }
        // For 0 to epoch times ...
        for (int i = 0; i < epochs; i++) {
            // For each batch
            System.out.println("Beginning Epoch " + (i + 1));
            for (List<NumberImage> batch : batches)
                learnBatch (batch, eta);
            if (verbose)
                test (trainingData);
        }
    }

    @SuppressWarnings("unchecked")
    private void learnBatch (List<NumberImage> batch, double eta) {
        // TODO: Remove reference to NumberImage class (make generic)
        // TODO: Fix unchecked type warning
        // Input each image and the backwards propagate
        for (NumberImage x : batch) {
            input(inputWrapper.wrapInput(x.image));
            propagate();
            backwardsPropagate(x.getResult());
        }
        finalizeLearning(batch.size(), eta);
    }

    /**
     * Run the backwards propagation algorithm using SGD
     * @param expected
     */
    private ConvolutionalNetwork backwardsPropagate (double[] expected) {
        // TODO

        // Get the error array of the output layer
        // Update Bias and Weights

        double[] delta = fcOutputLayer.learner().delta(expected);

        fcOutputLayer
                .learner()
                .incrementBiasUpdate(delta)
                .incrementWeightUpdate(delta);

        delta = workBackwards(delta);
        // Get the dimensions of the last pooling layer
        int dimX = lastConvolutionalPooling().lastPooling().dim_x();
        int dimY = lastConvolutionalPooling().lastPooling().dim_y();
        double[][] reshapedDelta = Matrix.arrayToMatrix(delta, dimX, dimY);
        // Convolutional Layers
        reshapedDelta = workBackwards(reshapedDelta);

        return this;
    }

    /**
     * Backpropagate through the fully connected layers
     * @param delta
     * @return
     */
    private double[] workBackwards(double[] delta) {
        for (int i = fullyConnectedLayers.size() - 1; i > -1; i--) {
            FullyConnectedLayerLearner learner =
                    fullyConnectedLayers.get(i).learner();
            delta = learner.delta(delta);
            learner
                    .incrementBiasUpdate(delta)
                    .incrementWeightUpdate(delta);
            return delta;

        }
        return delta;
    }

    /**
     * Backpropagate throughout the convolutional layers
     * @param delta
     * @return
     */
    private double[][] workBackwards (double[][] delta) {
        for (int i = convolutionalLayers.size() - 1; i > -1; i--) {
            delta = convolutionalLayers
                    .get(i)
                    .backPropagate(delta);
        }
        return delta;
    }

    /**
     * R
     * @param batchSize
     * @param eta
     */
    private ConvolutionalNetwork finalizeLearning (int batchSize, double eta) {
        // TODO
        // Apply updates to the network layer by layer, and zero after
        fcOutputLayer.learner().finalizeLearning(batchSize, eta);

        for (FullyConnectedLayer layer : fullyConnectedLayers)
            layer.learner().finalizeLearning(batchSize, eta);



        return this;
    }

    @SuppressWarnings("unchecked")
    public void test (List<NumberImage> testData) {
        int correct = 0;
        for (NumberImage i : testData) {
            input (inputWrapper.wrapInput(i));
            propagate ();
            if (decipher.decode(output()) == i.getResult())
                correct++;
        }
        System.out.println("Correct: " + correct + " out of " + testData.size());
    }

    /**
     * @return the last convolutional pooling layer
     */
    private ConvolutionalPoolings lastConvolutionalPooling() {
        return convolutionalLayers.get(
                convolutionalLayers.size() - 1
        );
    }


}
