package com.reggiemcdonald.neural.feedforward.net.math;

import com.reggiemcdonald.neural.feedforward.net.Layer;
import com.reggiemcdonald.neural.feedforward.net.Neuron;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.when;

public abstract class AbstractLayerLearnerTest {

    private static final double[] ERRORS = {.1, .2, .3, .4, .5};
    private static final double[] OUTPUTS = {0., .9, .8, 0, 0};
    private static final double[] WEIGHTS = {.1, .6, .8, .4, .5};
    private static final int N = 5;
    private static final double ETA = 3.;

    protected Layer layer;

    @BeforeEach
    public void runBefore() {
        layer = newLayer();
    }

    /**
     * Creates a layer with a layer type that reflects the layer that is
     * being tested by a given concrete implementation of this abstract test class
     * @return a layer of the correct type for the concrete implementation
     */
    protected abstract Layer newLayer();

    /**
     * Creates a layer with a layer type that reflects the layer before the layer
     * that a given concrete implementation of this test class is trying to test
     * @return the layer
     */
    protected abstract Layer newPreviousLayer();

    /**
     * Creates a neuron that is correct for the layer that the concrete
     * implementation of this class is intending to test.
     * @return spied neuron
     */
    protected abstract Neuron newNeuron();

    /**
     * Creates a neuron that is reflective of the layer that comes before
     * the layer that concrete implementations of this class intend to test.
     * This should be a spied instance.
     * @return spied neuron
     */
    protected abstract Neuron newPreviousNeuron();


    /**
     * Takes a spy and sets return output and neural index
     * @param output desired output value
     * @param idx
     * @return spy with stubbed methods
     */
    private Neuron newMockNeuron(double output, int idx) {
        Neuron n = newNeuron();
        when(n.getOutput()).thenReturn(output);
        when(n.neuralIndex()).thenReturn(idx);
        return n;
    }

    /**
     * Takes a spy and sets return output and neural index
     * @param output
     * @param idx
     * @return spy with stubbed methods
     */
    private Neuron newMockPreviousNeuron(double output, int idx) {
        Neuron n = newPreviousNeuron();
        when(n.getOutput()).thenReturn(output);
        when(n.neuralIndex()).thenReturn(idx);
        return n;
    }

    private void fillLayer(Layer layer) {
        for (int i = 0; i < OUTPUTS.length; i++) {
            layer.addNeuronToLayer(newMockNeuron(OUTPUTS[i], i));
        }
    }


    @Test
    public void testAddBiasUpdate() {
        fillLayer(layer);
        layer.addBiasUpdate(ERRORS);
        for (Neuron n : layer) {
            NeuronLearner learner = n.learner();
            assertEquals(ERRORS[n.neuralIndex()], learner.biasUpdateSum());
        }
    }
}
