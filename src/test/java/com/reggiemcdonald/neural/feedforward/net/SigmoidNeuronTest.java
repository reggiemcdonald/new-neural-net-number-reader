package com.reggiemcdonald.neural.feedforward.net;

public class SigmoidNeuronTest extends SigmoidalNeuronTest {

    @Override
    protected Neuron newNeuron() {
        return new SigmoidNeuron();
    }

    @Override
    protected Layer newLayer() {
        return new Layer(2, LayerType.HIDDEN);
    }

    @Override
    protected Neuron newMockNeuron(double output) {
        return null;
    }

    @Override
    public void testCompute() {

    }
}
