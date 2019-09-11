package com.reggiemcdonald.neural.convolutional.math;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;

public class SoftmaxCOutput implements COutput {
    private CNeuron neuron;

    public SoftmaxCOutput (CNeuron neuron) {
        this.neuron = neuron;
    }
}
