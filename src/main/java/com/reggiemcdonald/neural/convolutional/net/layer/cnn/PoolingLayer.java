package com.reggiemcdonald.neural.convolutional.net.layer.cnn;

import com.reggiemcdonald.neural.convolutional.net.util.Matrix;

public class PoolingLayer {
    private int dim_x, dim_y, window_width, stride;
    private boolean isMaxPooling = true; // Default to this because its better
    private double[][] map;

    public PoolingLayer (int dim_x, int dim_y, int window_width, int stride) {
        this.dim_x        = dim_x;
        this.dim_y        = dim_y;
        this.window_width = window_width;
        this.stride       = stride;
        this.map          = Matrix.zeros(dim_x, dim_y);
    }

    public PoolingLayer (int dim_x, int dim_y, int window_width, int stride, boolean isMaxPooling) {
        this.dim_x        = dim_x;
        this.dim_y        = dim_y;
        this.window_width = window_width;
        this.stride       = stride;
        this.isMaxPooling = isMaxPooling;
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

    public int stride() {
        return stride;
    }

    public double[][] outputs () {
        return map;
    }

    public void propagate(double[][] map) {
        // TODO
    }
}
