package com.reggiemcdonald.neural.feedforward.net;

import com.reggiemcdonald.neural.feedforward.net.math.OutputFunction;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Matchers.anyDouble;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class LayerTest {
    private Layer inputLayer;
    private Layer hiddenLayer;
    private Layer outputLayer;

    private final int INPUT_LAYER_INDEX = 0;
    private final int HIDDEN_LAYER_INDEX = 1;
    private final int OUTPUT_LAYER_INDEX = 2;

    private Neuron newNeuron() {
        return mock(Neuron.class);
    }

    private Neuron newNeuron(double output, int neuralIndex) {
        Neuron n = newNeuron();
        OutputFunction f = mock(OutputFunction.class);
        when(n.getOutput()).thenReturn(output);
        when(n.neuralIndex()).thenReturn(neuralIndex);
        when(n.outputFunction()).thenReturn(f);
        when(f.derivative(anyDouble())).thenReturn(output);
        return n;
    }

    @BeforeEach
    public void runBefore() {
        inputLayer = new Layer(INPUT_LAYER_INDEX, LayerType.INPUT);
        hiddenLayer = new Layer(HIDDEN_LAYER_INDEX, LayerType.HIDDEN);
        outputLayer = new Layer(OUTPUT_LAYER_INDEX, LayerType.OUTPUT);
    }

    @Test
    public void testNewLayer() {
        assertEquals(INPUT_LAYER_INDEX, inputLayer.layerIndex());
        assertEquals(HIDDEN_LAYER_INDEX, hiddenLayer.layerIndex());
        assertEquals(OUTPUT_LAYER_INDEX, outputLayer.layerIndex());

        assertEquals(LayerType.INPUT, inputLayer.type());
        assertEquals(LayerType.HIDDEN, hiddenLayer.type());
        assertEquals(LayerType.OUTPUT, outputLayer.type());
    }

    @Test
    public void testGetNeuronAt() {
        Neuron n1 = newNeuron();
        Neuron n2 = newNeuron();
        inputLayer
                .addNeuronToLayer(n1)
                .addNeuronToLayer(n2);
        assertSame(n1, inputLayer.getNeuronAt(0));
        assertSame(n2, inputLayer.getNeuronAt(1));
    }

    @Test
    public void addNeuronToLayer() {
        Neuron neuron = newNeuron();
        inputLayer
                .addNeuronToLayer(neuron)
                .addNeuronToLayer(neuron)
                .addNeuronToLayer(neuron);
        assertEquals(1, inputLayer.size());
        assertSame(neuron, inputLayer.getNeuronAt(0));
    }

    @Test
    public void testContains() {
        Neuron neuron = newNeuron();
        assertFalse(inputLayer.contains(neuron));
        inputLayer.addNeuronToLayer(neuron);
        assertTrue(inputLayer.contains(neuron));
    }

    @Test
    public void testRemoveNeuronFromLayer() {
        Neuron[] neurons = { newNeuron(), newNeuron(), newNeuron() };
        for (Neuron n : neurons) {
            inputLayer.addNeuronToLayer(n);
            assertTrue(inputLayer.contains(n));
        }
        inputLayer.removeNeuronFromLayer(neurons[0]);
        assertFalse(inputLayer.contains(neurons[0]));
    }

    @Test
    public void testActivations() {
        int neuralIndex = 0;
        double[] outputs = { 0.1, 0.2, 0.3, 0.4 };
        Neuron[] neurons = {
                newNeuron(outputs[0], neuralIndex++),
                newNeuron(outputs[1], neuralIndex++),
                newNeuron(outputs[2], neuralIndex++),
                newNeuron(outputs[3], neuralIndex)
        };
        Layer layer = new Layer(Arrays.asList(neurons), 0, LayerType.INPUT);
        assertArrayEquals(outputs, layer.activations());
    }

    @Test
    public void testDerive() {
        int neuralIndex = 0;
        double[] outputs = { .1, .2, .3, .4, };
        Neuron[] neurons = {
                newNeuron(outputs[0], neuralIndex++),
                newNeuron(outputs[1], neuralIndex++),
                newNeuron(outputs[2], neuralIndex++),
                newNeuron(outputs[3], neuralIndex)
        };
        Layer layer = new Layer(Arrays.asList(neurons), 0, LayerType.INPUT);
        assertArrayEquals(outputs, layer.derive(layer.activations()));
    }
}
