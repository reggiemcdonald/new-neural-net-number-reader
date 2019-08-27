package com.reggiemcdonald.neural.net;

import com.reggiemcdonald.neural.net.math.OutputFunction;
import com.reggiemcdonald.neural.net.math.SigmoidOutputFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A class to represent a neuron of the hidden layer with a sigmoid activation curve
 */
public class SigmoidNeuron implements Neuron {
    private List<Synapse>  to, from;
    private Layer          layer;
    private double         bias, output;
    private int            neuralIndex;
    private OutputFunction outputFunction;

    public SigmoidNeuron () {
        Random r            = new Random ();
        this.to             = new ArrayList<>();
        this.from           = new ArrayList<>();
        this.layer          = null;
        this.bias           = r.nextGaussian();
        this.output         = r.nextGaussian();
        this.neuralIndex    = -1;
        this.outputFunction = new SigmoidOutputFunction(this);
    }

    public SigmoidNeuron (double bias, double output) {
        this.to             = new ArrayList<>();
        this.from           = new ArrayList<>();
        this.layer          = null;
        this.bias           = bias;
        this.output         = output;
        this.neuralIndex    = -1;
        this.outputFunction = new SigmoidOutputFunction(this);
    }

    @Override
    public double getOutput() {
        return output;
    }

    @Override
    public double compute() {
        output = outputFunction.compute ();
        return output;
    }

    @Override
    public Neuron setOutput(float signal) {
        this.output = signal;
        return this;
    }

    @Override
    public void propagate() {
        // TODO: Remove?
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
        if (from.add (s))
            s.to().addSynapseToThis (s);
        return this;
    }

    @Override
    public Neuron addSynapseToThis(Synapse s) {
        if (to.add (s))
            s.from().addSynapseFromThis (s);
        return this;
    }

    @Override
    public List<Synapse> getSynapsesFromThis() {
        return from;
    }

    @Override
    public List<Synapse> getSynapseToThis() {
        return to;
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
    public int getNeuralIndex() {
        return neuralIndex;
    }

    @Override
    public OutputFunction outputFunction() {
        return outputFunction;
    }
}
