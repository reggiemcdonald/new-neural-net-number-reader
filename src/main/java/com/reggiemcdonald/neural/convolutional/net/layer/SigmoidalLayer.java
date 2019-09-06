package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;

import java.util.List;

public class SigmoidalLayer implements CNNLayer {
    private List<CNeuron> neurons;

    public SigmoidalLayer (int size) {
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