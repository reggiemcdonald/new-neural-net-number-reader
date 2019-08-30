package com.reggiemcdonald.neural.net;


import com.reggiemcdonald.neural.net.math.Learner;
import com.reggiemcdonald.neural.net.math.OutputFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class InputNeuron implements Neuron {
    private List<Synapse>    fromThis;
    private Layer            layer;
    private transient double output;
    private int              neuralIndex;

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
        // Input Neurons directly output their set signal (the input)
        return output;
    }

    @Override
    public Neuron setOutput(double output) {
        this.output = output;
        return this;
    }

    @Override
    public double getBias() {
        return 0;
    }

    @Override
    public void setBias(double bias) {
        // Do nothing as Input Neurons do not need bias
    }

    @Override
    public Neuron addSynapseFromThis(Synapse s) {
        if (!fromThis.contains (s)) {
            fromThis.add (s);
            s.to().addSynapseToThis (s);
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
    public int neuralIndex() {
        return neuralIndex;
    }

    @Override
    public OutputFunction outputFunction() {
        return null;
    }

    @Override
    public Learner learner() {
        return null;
    }
}
