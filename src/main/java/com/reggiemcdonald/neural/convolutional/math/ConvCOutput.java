package com.reggiemcdonald.neural.convolutional.math;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.CSynapse;

public class ConvCOutput implements COutput {
    private CNeuron neuron;

    public ConvCOutput (CNeuron neuron) {
        this.neuron = neuron;
    }

    @Override
    public double compute() {
        double output = 0.;
        for (CSynapse c : neuron.synapsesToThis())
            output += (c.weight() * c.from().output());
        return output + neuron.bias();
    }

    // TODO
}
