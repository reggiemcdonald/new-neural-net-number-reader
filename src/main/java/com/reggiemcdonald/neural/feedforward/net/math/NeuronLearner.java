package com.reggiemcdonald.neural.feedforward.net.math;

import com.reggiemcdonald.neural.feedforward.net.Neuron;
import com.reggiemcdonald.neural.feedforward.net.Synapse;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

public abstract class NeuronLearner implements Serializable {
    protected Neuron neuron;
    protected transient double biasUpdate;
    protected transient double[] weightUpdate;

    public NeuronLearner(Neuron neuron, double biasUpdate) {
        this.neuron = neuron;
        this.biasUpdate = biasUpdate;
    }

    /**
     * @param deltaNext
     * @return the delta value for this neuron
     */
    public abstract double delta(double[] deltaNext);

    /**
     * @return the neuron of this NeuronLearner
     */
    public Neuron neuron() {
        return neuron;
    }

    /**
     * Computes the amount of weight update to apply
     * @return the amount of weight update to apply
     * @param error
     */
    public double[] computeWeightUpdates(double error) {
        // TODO: Optimize this and extract to abstract class
        double[] input_activations = activations ();
        double[] weightUpdates = new double[input_activations.length];
        for (int i = 0; i < weightUpdates.length; i++)
            weightUpdates[i] = input_activations[i] * error;
        return weightUpdates;
    }

    /**
     * Computes the amount of bias update to apply
     * @return the amount of bias update to apply
     */
    public double computeBiasUpdate(double error) {
        return error;
    }

    /**
     * Applies the weight update to the parent neuron
     * @param n
     * @param eta
     */
    public void applyWeightUpdate(int n, double eta) {
        List<Synapse> synapses = neuron.getSynapsesToThis();
        for (int i = 0; i < synapses.size(); i++) {
            Synapse s = synapses.get(i);
            s.setWeight( s.getWeight() - ((eta/n) * weightUpdate[i]) );
        }
        Arrays.fill(weightUpdate,0);
    }

    /**
     * Applies the bias update to the parent neuron
     * @param n
     * @param eta
     */
    public void applyBiasUpdate(int n, double eta) {
        neuron.setBias( neuron.getBias() - ((eta/n) * biasUpdate) );
        biasUpdate = 0;
    }

    /**
     * Adds biasUpdate to the current amount of bias update
     * @param biasUpdate
     */
    public void addToBiasUpdate(double biasUpdate) {
        this.biasUpdate += biasUpdate;
    }

    /**
     * Increments the weight updates with the given weightUpdate array
     * @param weightUpdate
     */
    public void addToWeightUpdate(double[] weightUpdate) {
        if (this.weightUpdate == null || this.weightUpdate.length != weightUpdate.length)
            this.weightUpdate = new double[weightUpdate.length];
        for (int i = 0; i < this.weightUpdate.length; i++)
            this.weightUpdate[i] += weightUpdate[i];
    }

    /**
     * Returns the current sum of bias updates (not averaged)
     * @return
     */
    public double biasUpdateSum() {
        return biasUpdate;
    }

    /**
     * Returns the current sum of weight updates (not averaged)
     * @return
     */
    public double[] weightUpdateSum() {
        if (weightUpdate == null || neuron.getSynapsesToThis().size() != weightUpdate.length)
            return new double[neuron.getSynapsesToThis().size()];
        else
            return weightUpdate;
    }

    /**
     * @return the output activations of the lesser layer
     */
    private double[] activations () {
        List<Synapse> synapses = neuron.getSynapsesToThis();
        double[] activations = new double[synapses.size()];
        for (Synapse s : synapses)
            activations[s.from().neuralIndex()] = s.from().getOutput();
        return activations;
    }



}
