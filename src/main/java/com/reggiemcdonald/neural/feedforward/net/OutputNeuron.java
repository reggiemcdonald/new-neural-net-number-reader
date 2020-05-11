package com.reggiemcdonald.neural.feedforward.net;

import com.reggiemcdonald.neural.feedforward.net.math.*;

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
    private NeuronLearner learner;

    public OutputNeuron () {
        Random r         = new Random();
        toThis           = new ArrayList<>();
        layer            = null;
        bias             = r.nextGaussian();
        output           = r.nextGaussian();
        neuralIndex      = -1;
        outputFunction   = new SigmoidOutputFunction(this);
        learner = new OutputNeuronLearner(this);
    }

    public OutputNeuron (double bias, double output) {
        this.toThis           = new ArrayList<>();
        this.layer            = null;
        this.bias             = bias;
        this.output           = output;
        this.neuralIndex      = -1;
        this.outputFunction   = new SigmoidOutputFunction (this);
        this.learner = new OutputNeuronLearner(this);
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
    public void setBias(double bias) {
        this.bias = bias;
    }

    @Override
    public Neuron addSynapseFromThis(Synapse s) {
        // Do nothing
        return this;
    }

    @Override
    public Neuron addSynapseToThis(Synapse s) {
        if (!toThis.contains (s)) {
            toThis.add (s);
            s.from().addSynapseFromThis(s);
        }
        return this;
    }

    @Override
    public List<Synapse> getSynapsesFromThis() {
        return new ArrayList<>();
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

    @Override
    public NeuronLearner learner() {
        return learner;
    }
}
