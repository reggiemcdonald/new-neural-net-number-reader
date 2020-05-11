package com.reggiemcdonald.neural.feedforward.net.math;

import com.reggiemcdonald.neural.feedforward.net.OutputNeuron;

public class OutputNeuronLearner extends NeuronLearner {

    public OutputNeuronLearner(OutputNeuron neuron) {
        super(neuron, 0);
        this.weightUpdate = new double[neuron.getSynapsesToThis().size()];
    }

    @Override
    public double delta(double[] deltaNext) {
        OutputFunction outputFunction = neuron.outputFunction();
        return deltaNext[neuron.neuralIndex()] * outputFunction.derivative(outputFunction.compute());
    }
}
