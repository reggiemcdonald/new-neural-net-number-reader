package com.reggiemcdonald.neural.convolutional.net.learning.neuron;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.CSynapse;

import java.util.Arrays;
import java.util.List;

public class SoftmaxCLearner implements CLearner {
    private CNeuron neuron;
    private transient double biasUpdate;
    private transient double[] weightUpdates = null; // Initialize to null

    public SoftmaxCLearner (CNeuron neuron) {
        this.neuron = neuron;
    }

    @Override
    public double derivative() {
        return 0;
    }

    @Override
    public void incrementBiasUpdate(double delta) {
        biasUpdate += delta;
    }

    @Override
    public void incrementWeightUpdate(double[] delta) {
        if (weightUpdates == null || weightUpdates.length != neuron.synapsesToThis().size())
            weightUpdates = new double[neuron.synapsesToThis().size()];
        for (int i = 0; i < delta.length; i++)
            weightUpdates[i] += delta[i];
    }

    @Override
    public void applyBiasUpdate(int batchSize, double eta) {
        neuron.setBias(neuron.bias() - ((eta / biasUpdate) * biasUpdate));
        biasUpdate = 0;
    }

    @Override
    public void applyWeightUpdate(int batchSize, double eta) {
        List<CSynapse> synapses = neuron.synapsesToThis();
        for (int i = 0; i < weightUpdates.length; i++) {
            CSynapse synapse = synapses.get(i);
            synapse.weight(synapse.weight() - ((eta / batchSize) * weightUpdates[i]));
        }
        Arrays.fill(weightUpdates, 0);
    }

    @Override
    public double[] deltaWeight(double deltaBias) {
        double[] activations = activations();
        for (int i = 0; i < activations.length; i++)
            activations[i] *= deltaBias;
        return activations;
    }

    private double[] activations () {
        List<CSynapse> synapses = neuron.synapsesToThis();
        double[] d = new double[synapses.size()];
        for (int i = 0; i < synapses.size(); i++) {
            d[i] = synapses.get(i).from().output();
        }
        return d;
    }
}
