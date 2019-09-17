package com.reggiemcdonald.neural.convolutional.net;

import com.reggiemcdonald.neural.convolutional.net.util.InputWrapper;
import org.junit.Test;

public class ConvolutionalNetworkTest {

    InputWrapper<double[][]> myWrapper = (double[][] originalInput) -> {

        double[][][] newInput = new double[1][originalInput.length][originalInput[0].length];
        for (int i = 0; i < originalInput.length; i++)
            for (int j = 0; j < originalInput[i].length; j++)
                newInput[0][i][j] = originalInput[i][j];
        return newInput;

    };

    @Test
    public void testNetwork () {
        /**
         * @param inputLayerDimension an int of dimensions of the square 2D wrapInput layers
         * @param convolutionWindowSizes and int[] of the window sizes for each convolution layer
         * @param poolingWindowSizes an int[] of the window sizes for each of the poolings
         * @param depths an int[] of the depths for the wrapInput and convolution layers
         * @param sigmoidalOutputSizes an int[] of sizes for additional sigmoidal outputs (beyond flatten layer)
         * @param numberOfOutputs the number of neurons to be in the output layer
         * @param stride an int[] of stride lengths
         * @param hasSigmoidalLayer true if the network should have a sigmoidal layer
         */

        ConvolutionalNetwork cnn = new ConvolutionalNetwork(
                28,
                new int[] {5},
                new int[] {2},
                new int[] {1,3},
                new int[] {},
                10,
                new int[] {1,2},
                true);
//        List<NumberImage> numberImages = ImageLoader.load(ImageLoader.TEST_IMAGES, ImageLoader.TEST_LABELS);
//        cnn.learn(numberImages, 30, 10, 3.0, true);
        System.out.println("Done");
    }


}
