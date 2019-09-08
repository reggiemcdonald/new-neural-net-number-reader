package com.reggiemcdonald.neural.convolutional.math;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;

public class PoolCOutput implements COutput {
    private CNeuron neuron;

    public PoolCOutput (CNeuron neuron) {
        this.neuron = neuron;
    }
}
