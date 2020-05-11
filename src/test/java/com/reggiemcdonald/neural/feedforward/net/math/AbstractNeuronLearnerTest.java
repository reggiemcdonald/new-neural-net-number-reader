package com.reggiemcdonald.neural.feedforward.net.math;

import com.reggiemcdonald.neural.feedforward.net.Neuron;
import com.reggiemcdonald.neural.feedforward.net.Synapse;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.when;

public abstract class AbstractNeuronLearnerTest {

    public static final double[] TEST_NEURON_OUTPUTS
            = {.0, .0, .8, .9, .0};
    private static final double[] TEST_SYNAPSE_WEIGHTS
            = {.1, .2, .3, .4, .5};
    private static final double[] TEST_WEIGHT_UPDATES
            = {.03, .04, .05, .06, .07};
    private static final int N = 5;
    private static final double ETA = 3.;
    private static final double TEST_BIAS = .09;

    protected NeuronLearner learner;

    @BeforeEach
    public void runBefore() {
        learner = newLearner();
    }

    /**
     * Builds a new learner for the concrete test with
     * a neuron that has no synapses
     * @return Learner
     */
    protected abstract NeuronLearner newLearner();

    /**
     * @return a new neuron specific to this learner
     */
    protected abstract Neuron newMockNeuron();

    /**
     * @param output
     * @return a neuron that outputs the given output
     */
    protected Neuron newMockNeuron(double output, int idx) {
        Neuron neuron = newMockNeuron();
        when(neuron.getOutput()).thenReturn(output);
        when(neuron.neuralIndex()).thenReturn(idx);
        return neuron;
    }

    /**
     * Creates the previous layer of neurons that return an expected output
     * @param n the to neuron in the synapse
     */
    protected void createPreviousLayer(Neuron n) {
        for (int i = 0; i < TEST_NEURON_OUTPUTS.length; i++) {
            n.addSynapseToThis(new Synapse(newMockNeuron(
                    TEST_NEURON_OUTPUTS[i], i),
                    n, TEST_SYNAPSE_WEIGHTS[i]));
        }
    }

    @Test
    public void testComputeWeightUpdates() {
        double error = .2;
        Neuron neuron = learner.neuron();
        createPreviousLayer(neuron);
        double[] expected = new double[neuron.getSynapsesToThis().size()];
        List<Synapse> synapses = neuron.getSynapsesToThis();
        for (int i = 0; i < synapses.size(); i++) {
            Synapse s = synapses.get(i);
            expected[i] = s.from().getOutput() * error;
        }
        double[] actual = learner.computeWeightUpdates(error);
        assertArrayEquals(expected, actual);
    }

    @Test
    public void testComputeBiasUpdate() {
        double error = .4;
        double biasUpdate = learner.computeBiasUpdate(.4);
        assertEquals(error, biasUpdate);
    }

    @Test
    public void testApplyBiasUpdate() {
        Neuron neuron = learner.neuron();
        neuron.setBias(0.05);
        double expected = (neuron.getBias() - (ETA/N * TEST_BIAS));
        learner.addToBiasUpdate(TEST_BIAS);
        learner.applyBiasUpdate(N, ETA);
        assertEquals(expected, neuron.getBias());
    }
}
