package com.reggiemcdonald.neural.net.math;

import com.reggiemcdonald.neural.net.Neuron;

public class HiddenNeuronLearner implements Learner {
    private Neuron neuron;

    public HiddenNeuronLearner(Neuron neuron) {
        this.neuron = neuron;
    }

    @Override
    public double computeWeightUpdate(double error) {
        return 0;
    }

    @Override
    public double computeBiasUpdate(double error) {
        return 0;
    }

    @Override
    public void applyWeightUpdate(int n, double eta) {

    }

    @Override
    public void applyBiasUpdate(int n, double eta) {

    }

    @Override
    public double biasUpdateSum() {
        return 0;
    }

    @Override
    public double weightUpdateSum() {
        return 0;
    }
}
