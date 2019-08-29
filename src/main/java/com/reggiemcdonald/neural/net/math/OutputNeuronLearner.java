package com.reggiemcdonald.neural.net.math;

import com.reggiemcdonald.neural.net.Neuron;
import com.reggiemcdonald.neural.net.Synapse;

import java.util.List;

public class OutputNeuronLearner implements Learner {
    private Neuron neuron;
    private transient double biasUpdate;
    private transient double[] weightUpdate;

    public OutputNeuronLearner(Neuron neuron) {
        this.neuron = neuron;
        this.biasUpdate = 0;
        this.weightUpdate = new double[neuron.getSynapsesToThis().size()];
    }

    @Override
    public double[] computeWeightUpdates(double error) {
        // TODO: Optimize this and extract to abstract class
        double[] input_activations = activations ();
        double[] weightUpdates = new double[input_activations.length];
        for (int i = 0; i < weightUpdates.length; i++)
            weightUpdates[i] = input_activations[i] * error;
        return weightUpdates;
    }

    @Override
    public double computeBiasUpdate(double error) {
        return error;
    }

    @Override
    public void applyWeightUpdate(int n, double eta) {
        // TODO: Learning
    }

    @Override
    public void applyBiasUpdate(int n, double eta) {
        // TODO: Learning
    }

    @Override
    public void addToBiasUpdate(double biasUpdate) {
        this.biasUpdate += biasUpdate;
    }

    @Override
    public void addToWeightUpdate(double[] weightUpdate) {
        if (this.weightUpdate == null || this.weightUpdate.length != weightUpdate.length)
            this.weightUpdate = new double[weightUpdate.length];
        for (int i = 0; i < this.weightUpdate.length; i++)
            this.weightUpdate[i] += weightUpdate[i];
    }

    @Override
    public double biasUpdateSum() {
        return biasUpdate;
    }

    @Override
    public double[] weightUpdateSum() {
        if (weightUpdate == null || neuron.getSynapsesToThis().size() != weightUpdate.length)
            return new double[neuron.getSynapsesToThis().size()];
        else
            return weightUpdate;
    }

    @Override
    public double delta(double[] deltaNext) {
        OutputFunction outputFunction = neuron.outputFunction();
        return deltaNext[neuron.neuralIndex()] * outputFunction.derivative(outputFunction.compute());
    }

    private double[] activations () {
        List<Synapse> synapses = neuron.getSynapsesToThis();
        double[] activations = new double[synapses.size()];
        for (Synapse s : synapses)
            activations[s.from().neuralIndex()] = s.from().getOutput();
        return activations;
    }
}
