package com.reggiemcdonald.neural.feedforward.net;

import org.junit.jupiter.api.Disabled;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class InputNeuronTest extends AbstractNeuronTest {
    @Override
    protected Neuron newNeuron() {
        return new InputNeuron();
    }

    @Override
    protected Layer newLayer() {
        return new Layer(0, LayerType.INPUT);
    }

    @Override
    public void testCompute() {
        neuron.setOutput(TEST_OUTPUT);
        assertEquals(TEST_OUTPUT, neuron.compute(), 0.01);
    }

    @Override
    public void testAddSynapseToThis() {
        // An InputNeuron does not have any synapses incident on itself
    }
}
