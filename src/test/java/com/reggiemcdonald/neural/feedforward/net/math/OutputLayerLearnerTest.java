package com.reggiemcdonald.neural.feedforward.net.math;

import com.reggiemcdonald.neural.feedforward.net.*;

import static org.mockito.Mockito.spy;

public class OutputLayerLearnerTest extends AbstractLayerLearnerTest {
    @Override
    protected Layer newLayer() {
        return new Layer(2, LayerType.OUTPUT);
    }

    @Override
    protected Layer newPreviousLayer() {
        return new Layer(1, LayerType.HIDDEN);
    }

    @Override
    protected Neuron newNeuron() {
        return spy(new OutputNeuron(0, 0));
    }

    @Override
    protected Neuron newPreviousNeuron() {
        return spy(new SigmoidNeuron(0, 0));
    }
}
