package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;

import java.util.List;

public class SoftmaxLayer implements CNNLayer {
    private List<CNeuron> neurons;

    public SoftmaxLayer (int size) {
        // TODO: Layer init
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