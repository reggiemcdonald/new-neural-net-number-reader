package com.reggiemcdonald.neural.convolutional.net.learning.neuron;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;

public class ConvCLearner implements CLearner {
    private CNeuron neuron;

    public ConvCLearner (CNeuron neuron) {
        this.neuron = neuron;
    }

    @Override
    public double derivative() {
        return 0;
    }

    @Override
    public void incrementBiasUpdate(double delta) {

    }

    @Override
    public void incrementWeightUpdate(double[] delta) {

    }

    @Override
    public void applyBiasUpdate(int batchSize, double eta) {

    }

    @Override
    public void applyWeightUpdate(int batchSize, double eta) {

    }

    @Override
    public double[] deltaWeight(double deltaBias) {
        return new double[0];
    }


    // TODO
}
