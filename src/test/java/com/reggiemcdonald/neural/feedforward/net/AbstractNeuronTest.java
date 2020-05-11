package com.reggiemcdonald.neural.feedforward.net;


import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;


public abstract class AbstractNeuronTest {
    protected final double TEST_OUTPUT = 0.45;
    protected final double TEST_BIAS = 0.01;

    protected Neuron neuron;
    protected Layer layer;

    @BeforeEach
    public void runBefore() {
        neuron = newNeuron();
        layer = newLayer();
    }


    /**
     * @return a new neuron to test
     */
    protected abstract Neuron newNeuron();

    /**
     * @return a new Layer for this function
     */
    protected abstract Layer newLayer();

    @Test
    public void testGetAndSetOutput() {
        neuron.setOutput(TEST_OUTPUT);
        assertEquals(TEST_OUTPUT, neuron.getOutput(), 0.05);
    }

    public abstract void testCompute();

    @Test
    public void runTestCompute() {
        testCompute();
    }

    @Test
    public void testGetAndSetBias() {
        neuron.setBias(TEST_BIAS);
        assertEquals(TEST_BIAS, neuron.getBias(), 0.01);
    }

    @Test
    public void testAddSynapseFromThis() {
        Synapse synapse = new Synapse(newNeuron(), newNeuron(), 1);
        neuron.addSynapseFromThis(synapse);
        List<Synapse> synapses = neuron.getSynapsesFromThis();
        assertSame(synapse, synapses.get(0));
    }

    @Test
    public void testAddSynapseToThis() {
        Synapse synapse = new Synapse(newNeuron(), newNeuron(), 1);
        neuron.addSynapseToThis(synapse);
        List<Synapse> synapses = neuron.getSynapsesToThis();
        assertEquals(1, synapses.size());
        assertSame(synapse, synapses.get(0));
    }

    @Test
    public void testLayer() {
        neuron.setLayerAndIndex(layer, 1);
        assertSame(layer, neuron.layer());
        assertEquals(1, neuron.neuralIndex());
        neuron.setLayerAndIndex(layer, 2);
        assertSame(layer, neuron.layer());
        assertEquals(2, neuron.neuralIndex());
    }

}
