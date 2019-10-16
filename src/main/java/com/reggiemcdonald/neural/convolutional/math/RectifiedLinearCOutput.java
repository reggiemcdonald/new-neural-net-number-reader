package com.reggiemcdonald.neural.convolutional.math;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.CSynapse;

public class RectifiedLinearCOutput implements COutput {
    private CNeuron neuron;

    public RectifiedLinearCOutput(CNeuron neuron) {
        this.neuron = neuron;
    }

    @Override
    public double compute() {
        return Math.max (0, z ());
    }

    public double z () {
        double z = 0.;
        for (CSynapse c : neuron.synapsesToThis())
            z += (c.weight() * c.from().output());
        return z + neuron.bias();
    }
}
