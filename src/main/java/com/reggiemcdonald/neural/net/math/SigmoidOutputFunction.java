package com.reggiemcdonald.neural.net.math;

import com.reggiemcdonald.neural.net.Neuron;
import com.reggiemcdonald.neural.net.Synapse;

import java.util.List;

/**
 * An outputting function object for Sigmoid Neurons
 * compute() will generate the sigmoid of the weighted average of the connections into the neuron
 */
public class SigmoidOutputFunction implements OutputFunction {

    private Neuron neuron;

    public SigmoidOutputFunction (Neuron neuron) {
        this.neuron = neuron;
    }

    @Override
    public double compute () {
        List<Synapse> synapses = neuron.getSynapseToThis ();
        return sigmoid (z (synapses));
    }

    @Override
    public double derivative (double sigma) {
        return sigmoidDerivative (sigma);
    }

    /**
     * Calculates sigmoidal
     * @param z
     * @return
     */
    public double sigmoid (double z) {
        return 1 / (1 + Math.exp (-z));
    }

    /**
     * Calculates the z param (weighted sum of inputs to neuron)
     * @param synapses
     * @return
     */
    public double z (List<Synapse> synapses) {
        double output = 0;
        for (Synapse s : synapses)
            output += ( s.getWeight() * s.from().getOutput () );
        output += neuron.getBias ();
        return output;
    }

    /**
     * Calculate the sigmoid derivative for backwards propagation
     * @param sigma
     * @return
     */
    public double sigmoidDerivative (double sigma) {
        return sigma * (1 - sigma);
    }

}
