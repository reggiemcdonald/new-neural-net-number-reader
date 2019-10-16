package com.reggiemcdonald.neural.convolutional.math;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;

public class InputCOutput implements COutput {
    private CNeuron neuron;

    public InputCOutput (CNeuron neuron) {
        this.neuron = neuron;
    }

    @Override
    public double compute() {
        return neuron.output();
    }

    // TODO
}
