package com.reggiemcdonald.neural.feedforward.net.math;

import com.reggiemcdonald.neural.feedforward.net.SigmoidNeuron;
import com.reggiemcdonald.neural.feedforward.net.Synapse;

import java.util.List;

public class HiddenNeuronLearner extends NeuronLearner {

    public HiddenNeuronLearner(SigmoidNeuron neuron) {
        super(neuron, 0);
        this.weightUpdate = null;
    }

    /**
     * This isn't entirely right but works.
     * I need to sum the product of deltaNext[i] with the ith weight of each neuron in the current layer
     * To accomplish this, I need an additional abstraction so that layers will compute deltas differently
     * @param deltaNext
     * @return
     */
    @Override
    public double delta(double[] deltaNext) {
        return 0;
    }

    private double[] weights() {
        List<Synapse> synapses = neuron.getSynapsesToThis();
        double[] weights = new double[synapses.size()];
        for (Synapse s : synapses)
            weights[s.from().neuralIndex()] = s.getWeight();
        return weights;
    }
}
