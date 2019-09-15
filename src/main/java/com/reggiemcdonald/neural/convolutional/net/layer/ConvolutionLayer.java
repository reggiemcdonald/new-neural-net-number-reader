package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.net.learning.layer.CLayerLearner;
import com.reggiemcdonald.neural.convolutional.net.util.Matrix;

import java.util.Random;

/**
 * A convolutional layer in a CNN
 * The kernel is an array of weights sized according to the receiving window
 * that are common among each dimX * dimY neurons in the layer
 */
public class ConvolutionLayer  {
    private int dimX, dimY;
    private int window_width;
    private int stride;
    private double[][][] kernel;
    private double[][] maps;
    private double bias;
    private CLayerLearner learner;

    public ConvolutionLayer (int dimX, int dimY, int window_width, double[][][] kernel, int stride) {
        assert (dimX == dimY);
        Random r          = new Random ();
        this.dimX = dimX;
        this.dimY = dimY;
        this.window_width = window_width;
        this.kernel       = kernel;
        this.bias         = r.nextGaussian();
        this.stride       = stride;
        this.maps         = Matrix.zeros(dimX, dimY);
    }

    public void propagate (double[][][] input ) {
        maps = Matrix.elementWiseAdd(
                Matrix.validConvolve(input, kernel, stride),
                bias
        );
    }

    public int dimX() {
        return dimX;
    }

    public int dimY() {
        return dimY;
    }

    public int windowWidth() {
        return window_width;
    }

    public int stride () {
        return stride;
    }

    public double bias() {
        return bias;
    }

    public double[][][] kernel() {
        return kernel;
    }

    public double[][] outputs() {
        return maps;
    }


}
