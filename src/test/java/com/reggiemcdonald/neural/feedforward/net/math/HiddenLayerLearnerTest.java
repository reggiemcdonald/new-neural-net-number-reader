package com.reggiemcdonald.neural.feedforward.net.math;

import com.reggiemcdonald.neural.feedforward.net.*;

import static org.mockito.Mockito.spy;


public class HiddenLayerLearnerTest extends AbstractLayerLearnerTest {
    @Override
    protected Layer newLayer() {
        return new Layer(1, LayerType.HIDDEN);
    }

    @Override
    protected Layer newPreviousLayer() {
        return new Layer(1, LayerType.INPUT);
    }

    @Override
    protected Neuron newNeuron() {
        return spy(new SigmoidNeuron(0, 0));
    }

    @Override
    protected Neuron newPreviousNeuron() {
        return spy(new InputNeuron(0));
    }
}
