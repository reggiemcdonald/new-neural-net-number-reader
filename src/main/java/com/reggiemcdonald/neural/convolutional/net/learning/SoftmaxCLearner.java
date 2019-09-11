package com.reggiemcdonald.neural.convolutional.net.learning;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;

public class SoftmaxCLearner implements CLearner {
    private CNeuron neuron;

    public SoftmaxCLearner (CNeuron neuron) {
        this.neuron = neuron;
    }
}
