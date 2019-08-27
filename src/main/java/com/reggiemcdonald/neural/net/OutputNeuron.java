package com.reggiemcdonald.neural.net;

import com.reggiemcdonald.neural.net.math.OutputFunction;

import java.util.List;

public class OutputNeuron implements Neuron {
    @Override
    public double getOutput() {
        return 0;
    }

    @Override
    public double compute() {
        return 0;
    }

    @Override
    public Neuron setOutput(float signal) {
        return null;
    }

    @Override
    public void propagate() {

    }

    @Override
    public double getBias() {
        return 0;
    }

    @Override
    public void setBias(float bias) {

    }

    @Override
    public void addBiasUpdate(float biasUpdate) {

    }

    @Override
    public double getBiasUpdate() {
        return 0;
    }

    @Override
    public void zeroBiasUpdate() {

    }

    @Override
    public Neuron addSynapseFromThis(Synapse s) {
        return null;
    }

    @Override
    public Neuron addSynapseToThis(Synapse s) {
        return null;
    }

    @Override
    public List<Synapse> getSynapsesFromThis() {
        return null;
    }

    @Override
    public List<Synapse> getSynapseToThis() {
        return null;
    }

    @Override
    public Layer layer() {
        return null;
    }

    @Override
    public void setLayerAndIndex(Layer layer, int index) {

    }

    @Override
    public int getNeuralIndex() {
        return 0;
    }

    @Override
    public OutputFunction outputFunction() {
        return null;
    }
}
