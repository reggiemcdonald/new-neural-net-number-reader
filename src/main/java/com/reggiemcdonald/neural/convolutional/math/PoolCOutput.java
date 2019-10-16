package com.reggiemcdonald.neural.convolutional.math;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.CSynapse;

/**
 * TODO Right now this is only a max pooling
 * Must add two different output functions
 */
public class PoolCOutput implements COutput {
    private CNeuron neuron;

    public PoolCOutput (CNeuron neuron) {
        this.neuron = neuron;
    }

    @Override
    public double compute() {
        double output = neuron.synapsesToThis().get(0).from().output();
        for (CSynapse c : neuron.synapsesToThis())
            output = Math.max(output, c.from().output());
        return output;
    }
}
