package com.reggiemcdonald.neural.convolutional.net.learning.neuron;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;

public class InputCLearner implements CLearner {
    private CNeuron neuron;

    public InputCLearner (CNeuron neuron) {
        this.neuron = neuron;
    }

    @Override
    public double derivative() {
        return 0;
    }

    // TODO
}
