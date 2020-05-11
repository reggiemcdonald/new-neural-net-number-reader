package com.reggiemcdonald.neural.feedforward.net.math;

import com.reggiemcdonald.neural.feedforward.net.Neuron;
import com.reggiemcdonald.neural.feedforward.net.OutputNeuron;
import com.reggiemcdonald.neural.feedforward.net.SigmoidNeuron;

import static org.mockito.Mockito.mock;

public class OutputNeuronLearnerTest extends AbstractNeuronLearnerTest {
    @Override
    protected NeuronLearner newLearner() {
        return new OutputNeuronLearner(new OutputNeuron(0, 0));
    }

    @Override
    protected Neuron newMockNeuron() {
        return mock(SigmoidNeuron.class);
    }
}
