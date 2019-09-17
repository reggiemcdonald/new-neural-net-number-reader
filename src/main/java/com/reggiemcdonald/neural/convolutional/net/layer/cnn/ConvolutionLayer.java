package com.reggiemcdonald.neural.convolutional.net.layer.cnn;

import com.reggiemcdonald.neural.convolutional.net.learning.layer.cnn.CNNLayerLearner;
import com.reggiemcdonald.neural.convolutional.net.learning.layer.cnn.ConvolutionLayerLearner;
import com.reggiemcdonald.neural.convolutional.net.util.Matrix;

import java.util.Random;

/**
 * A convolutional layer in a CNN
 * The kernel is an array of weights sized according to the receiving window
 * that are common among each dimX * dimY neurons in the layer
 */
public class ConvolutionLayer implements CNNLayer  {
    private int dimX, dimY;
    private int window_width;
    private int stride;
    private double[][][] kernel;
    private double[][] map;
    private double bias;
    private CNNLayerLearner learner;

    public ConvolutionLayer (int dimX, int dimY, int window_width, double[][][] kernel, int stride) {
        assert (dimX == dimY);
        Random r          = new Random ();
        this.dimX         = dimX;
        this.dimY         = dimY;
        this.window_width = window_width;
        this.kernel       = kernel;
        this.bias         = r.nextGaussian();
        this.stride       = stride;
        this.learner      = new ConvolutionLayerLearner(this);
        this.map          = Matrix.zeros(dimX, dimY);
    }

    public void propagate (double[][][] input ) {
        map = Matrix.elementWiseAdd(
                Matrix.validConvolve(input, kernel, stride),
                bias
        );
    }

    @Override
    // The layer is flat
    public double get(int x, int y, int z) {
        return map[x][y];
    }

    @Override
    public int depth() {
        return dimX();
    }

    @Override
    public int dimX() {
        return dimX;
    }

    @Override
    public int dimY() {
        return dimY;
    }

    @Override
    public int dimZ() {
        return 1;
    }

    @Override
    public CNNLayerLearner learner() {
        return learner;
    }

    @Override
    public int stride () {
        return stride;
    }

    @Override
    public double[][][] kernel() {
        return kernel;
    }

    public double[][] outputs() {
        return map;
    }

    public double bias() {
        return bias;
    }

    public int windowWidth() {
        return window_width;
    }




}
