package com.reggiemcdonald.neural.convolutional.net.learning.neuron;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;

public class ConvCLearner implements CLearner {
    private CNeuron neuron;

    public ConvCLearner (CNeuron neuron) {
        this.neuron = neuron;
    }

    @Override
    public double derivative() {
        return 0;
    }


    // TODO
}
