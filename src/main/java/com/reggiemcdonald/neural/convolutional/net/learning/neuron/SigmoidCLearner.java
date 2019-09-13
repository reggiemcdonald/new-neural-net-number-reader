package com.reggiemcdonald.neural.convolutional.net.learning.neuron;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;

public class SigmoidCLearner implements CLearner {
    private CNeuron neuron;

    public SigmoidCLearner (CNeuron neuron) {
        this.neuron = neuron;
    }

    @Override
    public double derivative() {
        return 0;
    }
}
