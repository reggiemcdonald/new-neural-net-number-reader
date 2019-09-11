package com.reggiemcdonald.neural.convolutional.math;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.CSynapse;

public class SigmoidCOutput implements COutput {
    private CNeuron neuron;

    public SigmoidCOutput (CNeuron neuron) {
        this.neuron = neuron;
    }

    @Override
    public double compute() {
        return sigmoid (z ());
    }

    public double sigmoid (double z) {
        return 1 / (1 + Math.exp (-z));
    }

    public double z () {
        double z = 0.;
        for (CSynapse c : neuron.synapsesToThis())
            z += (c.weight() * c.from().output());
        return z + neuron.bias();
    }

}
