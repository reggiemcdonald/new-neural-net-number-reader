package com.reggiemcdonald.neural.net;

import com.reggiemcdonald.neural.net.math.OutputFunction;
import com.reggiemcdonald.neural.net.math.SigmoidOutputFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class OutputNeuron implements Neuron {
    private List<Synapse>    toThis;
    private Layer            layer;
    private double           bias;
    private transient double output;
    private int              neuralIndex;
    private OutputFunction outputFunction;

    public OutputNeuron () {
        Random r    = new Random();
        toThis      = new ArrayList<>();
        layer       = null;
        bias        = r.nextGaussian();
        output      = r.nextGaussian();
        neuralIndex = -1;
        outputFunction = new SigmoidOutputFunction(this);
    }

    public OutputNeuron (double bias, double output) {
        toThis      = new ArrayList<>();
        layer       = null;
        this.bias   = bias;
        this.output = output;
        neuralIndex = -1;
        outputFunction = new SigmoidOutputFunction(this);
    }


    @Override
    public double getOutput() {
        return output;
    }

    @Override
    public double compute() {
        output = outputFunction.compute();
        return output;
    }

    @Override
    public Neuron setOutput(double output) {
        this.output = output;
        return this;
    }

    @Override
    public double getBias() {
        return bias;
    }

    @Override
    public void setBias(float bias) {
        this.bias = bias;
    }

    @Override
    public void addBiasUpdate(float biasUpdate) {
        // TODO: Learning
    }

    @Override
    public double getBiasUpdate() {
        // TODO: Learning
        return 0;
    }

    @Override
    public void zeroBiasUpdate() {
        // TODO: Learning
    }

    @Override
    public Neuron addSynapseFromThis(Synapse s) {
        // Do nothing
        return this;
    }

    @Override
    public Neuron addSynapseToThis(Synapse s) {
        if (toThis.add (s))
            s.from().addSynapseFromThis(s);
        return this;
    }

    @Override
    public List<Synapse> getSynapsesFromThis() {
        return null;
    }

    @Override
    public List<Synapse> getSynapsesToThis() {
        return toThis;
    }

    @Override
    public Layer layer() {
        return layer;
    }

    @Override
    public void setLayerAndIndex(Layer layer, int index) {
        this.layer       = layer;
        this.neuralIndex = index;
    }

    @Override
    public int neuralIndex() {
        return neuralIndex;
    }

    @Override
    public OutputFunction outputFunction() {
        return outputFunction;
    }
}
