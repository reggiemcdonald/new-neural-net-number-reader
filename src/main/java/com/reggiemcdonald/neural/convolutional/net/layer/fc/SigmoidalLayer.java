package com.reggiemcdonald.neural.convolutional.net.layer.fc;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.CNeuronFactory;
import com.reggiemcdonald.neural.convolutional.net.CSynapse;
import com.reggiemcdonald.neural.convolutional.net.Propagatable;
import com.reggiemcdonald.neural.convolutional.net.learning.layer.FullyConnectedLayerLearner;
import com.reggiemcdonald.neural.convolutional.net.learning.layer.SigmoidalLayerLearner;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

public class SigmoidalLayer implements FullyConnectedLayer {
    private List<CNeuron> neurons;
    private FullyConnectedLayerLearner learner;

    public SigmoidalLayer (int size) {
        learner = new SigmoidalLayerLearner (this);
        makeNeurons (size);
    }

    /**
     * Produce size neurons to fill this layer with gaussian intializations
     * @param size
     */
    private void makeNeurons (int size) {
        Random r = new Random();
        this.neurons = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            CNeuron neuron = CNeuronFactory.makeNeuron(
                    CNeuronFactory.CN_TYPE_SIGM, this, r.nextGaussian(), r.nextGaussian()
            );
            neurons.add(neuron);
        }
    }

    @Override
    public CNeuron get(int idx) {
        return neurons.get (idx);
    }

    @Override
    public FullyConnectedLayer connect(FullyConnectedLayer fromLayer, int k) {
        Random r = new Random();
        FullyConnectedLayer toLayer = this;
        for (CNeuron to : toLayer)
            for (CNeuron from : fromLayer)
                to.addConnectionToThis(new CSynapse(from, to, r.nextGaussian(), fromLayer.size()));
        return this;
    }

    @Override
    public int size() {
        return neurons.size();
    }

    @Override
    public FullyConnectedLayerLearner learner() {
        return learner;
    }

    @Override
    public Iterator<CNeuron> iterator() {
        return neurons.iterator();
    }

    @Override
    public void propagate() {
        for (Propagatable p : this)
            p.propagate();
    }

    public void input (double[] inputs) {
        for (int i = 0; i < inputs.length; i++)
            neurons.get(i).setOutput(inputs[i]);
    }
}
