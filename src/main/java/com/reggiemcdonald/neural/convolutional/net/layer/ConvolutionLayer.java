package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;

import java.util.List;

/**
 * A convolutional layer in a CNN
 */
public class ConvolutionLayer implements CNNLayer {
    List<CNeuron> neurons;
    private double dim_x, dim_y;

    public ConvolutionLayer (int dim) {
        // TODO: Layer init
    }

    @Override
    public CNeuron get(int x, int y) {
        return null;
    }

    @Override
    public CNNLayer connect(CNNLayer to) {
        return this;
    }

    @Override
    public int size() {
        // TODO: Stub
        return 0;
    }
}
