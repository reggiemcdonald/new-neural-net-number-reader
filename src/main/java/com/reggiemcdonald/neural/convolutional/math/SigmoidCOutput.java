package com.reggiemcdonald.neural.convolutional.math;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;

public class SigmoidCOutput implements COutput {
    private CNeuron neuron;

    public SigmoidCOutput (CNeuron neuron) {
        this.neuron = neuron;
    }
}
