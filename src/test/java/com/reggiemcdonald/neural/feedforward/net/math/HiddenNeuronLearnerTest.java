package com.reggiemcdonald.neural.feedforward.net.math;

import com.reggiemcdonald.neural.feedforward.net.Neuron;
import com.reggiemcdonald.neural.feedforward.net.SigmoidNeuron;

import static org.mockito.Mockito.mock;

public class HiddenNeuronLearnerTest extends AbstractNeuronLearnerTest {
    @Override
    protected NeuronLearner newLearner() {
        return new HiddenNeuronLearner(new SigmoidNeuron(0, 0));
    }

    @Override
    protected Neuron newMockNeuron() {
        return mock(Neuron.class);
    }
}
