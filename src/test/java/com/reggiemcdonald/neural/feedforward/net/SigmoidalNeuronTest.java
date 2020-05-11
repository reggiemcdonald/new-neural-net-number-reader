package com.reggiemcdonald.neural.feedforward.net;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public abstract class SigmoidalNeuronTest extends AbstractNeuronTest {

    protected Neuron newMockNeuron(double output) {
        Neuron n = mock(Neuron.class);
        when(n.compute()).thenReturn(output);
        when(n.getOutput()).thenReturn(output);
        return n;
    }

    @Override
    public void testCompute() {
        double bias = 0.1;
        neuron.setBias(bias);
        double[] outputs = { 0., 0.9, 0.8, 0., };
        double[] weights = { 0.1, 0.2, 0.1, 0.4, };
        double z = 0;
        for (int i = 0; i < outputs.length; i++)
            z  += (weights[i] * outputs[i]);
        z += bias;
        double sigmoid = (1 / (1 + Math.exp(-z)));
        List<Synapse> synapses = new ArrayList<>();
        for (int i = 0; i < weights.length; i++)
            synapses.add(new Synapse(newMockNeuron(outputs[i]), this.neuron, weights[i]));
        for (Synapse synapse : synapses)
            neuron.addSynapseToThis(synapse);
        assertEquals(sigmoid, neuron.compute(), 0.01);
    }
}
