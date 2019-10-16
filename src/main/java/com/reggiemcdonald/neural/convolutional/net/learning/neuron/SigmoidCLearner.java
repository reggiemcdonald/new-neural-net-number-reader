package com.reggiemcdonald.neural.convolutional.net.learning.neuron;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.CSynapse;

import java.util.Arrays;
import java.util.List;

public class SigmoidCLearner implements CLearner {
    private CNeuron neuron;
    private transient double     biasUpdate;
    private transient double[] weightUpdates = null; // set to null initially

    public SigmoidCLearner (CNeuron neuron) {
        this.neuron = neuron;
    }

    @Override
    public double derivative() {
        return 0.; // TODO STUB
    }

    @Override
    public void incrementBiasUpdate(double delta) {
        biasUpdate += delta;
    }

    @Override
    public void incrementWeightUpdate(double[] delta) {
        if (weightUpdates == null || weightUpdates.length != neuron.synapsesToThis().size())
            weightUpdates = new double[neuron.synapsesToThis().size()];
        if (delta.length != weightUpdates.length)
            throw new RuntimeException("Fatal: length of delta should equal inputs");
        for (int i = 0; i < weightUpdates.length; i++)
            weightUpdates[i] += delta[i];
    }

    @Override
    public void applyBiasUpdate(int batchSize, double eta) {
        neuron.setBias(neuron.bias() - ((eta / batchSize) * biasUpdate));
        biasUpdate = 0;
    }

    @Override
    public void applyWeightUpdate(int batchSize, double eta) {
        List<CSynapse> synapses = neuron.synapsesToThis();
        for (int i = 0; i < weightUpdates.length; i++) {
            CSynapse synapse = synapses.get(i);
            synapse.weight(synapse.weight() - ((eta / batchSize) * weightUpdates[i]));
        }
        Arrays.fill(weightUpdates, 0.);
    }

    /**
     * Compute the weight update
     * @param deltaBias
     * @return
     */
    public double[] deltaWeight(double deltaBias) {
        double[] inputActivations = activations ();
        for (int i = 0; i < inputActivations.length; i++)
            inputActivations[i] *= deltaBias;
        return inputActivations;
    }

    @Override
    public CLearner setWeightUpdates(double[] weightUpdates) {
        // TODO
        return null;
    }

    private double[] activations () {
        double[] d = new double[neuron.synapsesToThis().size()];
        List<CSynapse> synapses = neuron.synapsesToThis();
        for (int i = 0; i < d.length; i++) {
            d[i] = synapses.get(i).from().output();
        }
        return d;
    }
}
