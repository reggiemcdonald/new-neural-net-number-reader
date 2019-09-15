package com.reggiemcdonald.neural.convolutional.math;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.CSynapse;
import com.reggiemcdonald.neural.convolutional.net.layer.SoftmaxLayer;

public class SoftmaxCOutput implements COutput {
    private CNeuron neuron;

    public SoftmaxCOutput (CNeuron neuron) {
        this.neuron = neuron;
    }

    @Override
    public double compute() {
        SoftmaxLayer softmaxLayer = (SoftmaxLayer) neuron.layer();
        // TODO Optimize here: dont need to compute z twice
        double sum = sum( expAll( softmaxLayer.allZ() ) );
        return Math.exp( z() ) / sum;
    }

    private double[] expAll (double[] d) {
        double[] expAll = new double[d.length];
        for (int i = 0; i < d.length; i++)
            expAll[i] = Math.exp(d[i]);
        return expAll;
    }

    private double sum (double[] a) {
        double s = 0f;
        for (double d : a)
            s += d;
        return s;
    }

    public double z () {
        double output = 0.;
        for (CSynapse c : neuron.synapsesToThis())
            output += (c.weight() * c.from().output());
        return output;
    }
}
