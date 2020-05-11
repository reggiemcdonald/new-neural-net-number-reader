package com.reggiemcdonald.neural.feedforward.net;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class SynapseTest {

    private final double WEIGHT = 0.3;

    private Neuron from;
    private Neuron to;
    private Synapse synapse;

    @BeforeEach
    public void runBefore() {
        from = new InputNeuron();
        to = new SigmoidNeuron();
        synapse = new Synapse(from, to, WEIGHT);
    }

    @Test
    public void testNewSynapse() {
        assertSame(from, synapse.from());
        assertSame(to, synapse.to());
        assertEquals(WEIGHT, synapse.getWeight());
    }

    @Test
    public void testSetFrom() {
        Neuron newNeuron = new InputNeuron();
        assertSame(from, synapse.from());
        synapse.setFrom(newNeuron);
        assertSame(newNeuron, synapse.from());
    }

    @Test
    public void testSetTo() {
        Neuron newNeuron = new InputNeuron();
        assertSame(to, synapse.to());
        synapse.setTo(newNeuron);
        assertSame(newNeuron, synapse.to());
    }

    @Test
    public void testSetWeight() {
        double newWeight = 0.5;
        assertEquals(WEIGHT, synapse.getWeight());
        assertNotEquals(newWeight, WEIGHT);
        synapse.setWeight(newWeight);
        assertEquals(newWeight, synapse.getWeight());
    }
}
