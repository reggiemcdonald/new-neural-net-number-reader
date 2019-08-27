package com.reggiemcdonald.neural.net;

import com.reggiemcdonald.neural.net.math.OutputFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class InputNeuron implements Neuron {
    private List<Synapse> fromThis;
    private Layer         layer;
    private double        output;
    private int           neuralIndex;

    public InputNeuron () {
        Random r    = new Random ();
        fromThis    = new ArrayList<>();
        layer       = null;
        output      = r.nextGaussian ();
        neuralIndex = -1;
    }

    public InputNeuron (double output) {
        this.fromThis = new ArrayList<>();
        layer         = null;
        this.output   = output;
        neuralIndex   = -1;
    }


    @Override
    public double getOutput() {
        return output;
    }

    @Override
    public double compute() {
        // Input Neurons directly output their set signal
        return output;
    }

    @Override
    public Neuron setOutput(double output) {
        this.output = output;
        return this;
    }

    @Override
    public void propagate() {
        // TODO: Remove ??
    }

    @Override
    public double getBias() {
        return 0;
    }

    @Override
    public void setBias(float bias) {
        // Do nothing as Input Neurons do not need bias
    }

    @Override
    public void addBiasUpdate(float biasUpdate) {
        // Do nothing as InputNeurons do not need bias updates
    }

    @Override
    public double getBiasUpdate() {
        return 0;
    }

    @Override
    public void zeroBiasUpdate() {
        // Do nothing
    }

    @Override
    public Neuron addSynapseFromThis(Synapse s) {
        if (fromThis.add (s)) {
            s.from().addSynapseToThis (s);
        }
        return this;
    }

    @Override
    public Neuron addSynapseToThis(Synapse s) {
        return this;
    }

    @Override
    public List<Synapse> getSynapsesFromThis() {
        return this.fromThis;
    }

    @Override
    public List<Synapse> getSynapsesToThis() {
        return null;
    }

    @Override
    public Layer layer() {
        return layer;
    }

    @Override
    public void setLayerAndIndex(Layer layer, int index) {
        this.layer      = layer;
        this.neuralIndex = index;
    }

    @Override
    public int getNeuralIndex() {
        return neuralIndex;
    }

    @Override
    public OutputFunction outputFunction() {
        return null;
    }
}
