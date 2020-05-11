package com.reggiemcdonald.neural.feedforward.net;

import com.reggiemcdonald.neural.feedforward.net.math.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A class toThis represent a neuron of the hidden layer with a sigmoid activation curve
 */
public class SigmoidNeuron implements Neuron {
    private List<Synapse>    toThis, fromThis;
    private Layer            layer;
    private transient double output;
    private double           bias;
    private int              neuralIndex;
    private OutputFunction outputFunction;
    private NeuronLearner learner;

    public SigmoidNeuron () {
        Random r            = new Random ();
        this.toThis         = new ArrayList<>();
        this.fromThis       = new ArrayList<>();
        this.layer          = null;
        this.bias           = r.nextGaussian ();
        this.output         = r.nextGaussian ();
        this.neuralIndex    = -1;
        this.outputFunction = new SigmoidOutputFunction(this);
        this.learner        = new HiddenNeuronLearner(this);
    }

    public SigmoidNeuron (double bias, double output) {
        this.toThis         = new ArrayList<>();
        this.fromThis       = new ArrayList<>();
        this.layer          = null;
        this.bias           = bias;
        this.output         = output;
        this.neuralIndex    = -1;
        this.outputFunction = new SigmoidOutputFunction (this);
        this.learner        = new HiddenNeuronLearner (this);
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
        if (!fromThis.contains (s)) {
            fromThis.add (s);
            s.to().addSynapseToThis(s);
        }
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
        return fromThis;
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
