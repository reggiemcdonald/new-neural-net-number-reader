package com.reggiemcdonald.neural.feedforward.net;

import org.junit.jupiter.api.Disabled;

public class OutputNeuronTest extends SigmoidalNeuronTest {

    @Override
    protected Neuron newNeuron() {
        return new OutputNeuron();
    }

    @Override
    protected Layer newLayer() {
        return new Layer(2, LayerType.OUTPUT);
    }

    @Override
    @Disabled
    public void testAddSynapseFromThis() {}
}
