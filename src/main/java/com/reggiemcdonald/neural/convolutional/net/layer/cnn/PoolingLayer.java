package com.reggiemcdonald.neural.convolutional.net.layer.cnn;

import com.reggiemcdonald.neural.convolutional.net.learning.layer.cnn.CNNLayerLearner;
import com.reggiemcdonald.neural.convolutional.net.learning.layer.cnn.PoolingLayerLearner;
import com.reggiemcdonald.neural.convolutional.net.util.Matrix;

public class PoolingLayer implements CNNLayer {
    private int dim_x, dim_y, window_width, stride;
    private boolean isMaxPooling = true; // Default to this because its better
    private double[][] map;
    private CNNLayerLearner learner;


    public PoolingLayer (int dim_x, int dim_y, int window_width, int stride) {
        this.dim_x        = dim_x;
        this.dim_y        = dim_y;
        this.window_width = window_width;
        this.stride       = stride;
        this.learner      = new PoolingLayerLearner(this);
        this.map          = Matrix.zeros(dim_x, dim_y);
    }

    public PoolingLayer (int dim_x, int dim_y, int window_width, int stride, boolean isMaxPooling) {
        this.dim_x        = dim_x;
        this.dim_y        = dim_y;
        this.window_width = window_width;
        this.stride       = stride;
        this.isMaxPooling = isMaxPooling;
        this.learner      = new PoolingLayerLearner(this);
        this.map          = Matrix.zeros(dim_x, dim_y);
    }


    public int size() {
        return dim_x * dim_y;
    }

    public int dim_x() {
        return dim_x;
    }

    public int dim_y() {
        return dim_y;
    }

    public int window_width() {
        return window_width;
    }

    @Override
    public double get(int x, int y, int z) {
        return 0;
    }

    @Override
    public int depth() {
        return 0; // TODO
    }

    @Override
    public int dimX() {
        return dim_x;
    }

    @Override
    public int dimY() {
        return dim_y;
    }

    @Override
    public int dimZ() {
        return 1;
    }

    public int stride() {
        return stride;
    }

    @Override
    public int windowWidth() {
        return window_width;
    }

    @Override
    public double[][][] kernel() {
        // TODO
        return new double[0][][];
    }

    public double[][] outputs () {
        return map;
    }

    @Override
    public CNNLayerLearner learner() {
        return learner;
    }

    public void propagate(double[][] map) {
        // TODO
    }
}
