package com.reggiemcdonald.neural.convolutional.net.layer.fc;

import com.reggiemcdonald.neural.convolutional.math.SoftmaxCOutput;
import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.CNeuronFactory;
import com.reggiemcdonald.neural.convolutional.net.CSynapse;
import com.reggiemcdonald.neural.convolutional.net.Propagatable;
import com.reggiemcdonald.neural.convolutional.net.learning.layer.FullyConnectedLayerLearner;
import com.reggiemcdonald.neural.convolutional.net.learning.layer.SoftmaxLayerLearner;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

public class SoftmaxLayer implements FullyConnectedLayer {
    private List<CNeuron> neurons;
    private double[] allZ;
    private boolean allZNeedsUpdate = true;
    private FullyConnectedLayerLearner learner;

    public SoftmaxLayer (int size) {
        this.learner = new SoftmaxLayerLearner(this);
        makeNeurons (size);
    }

    /**
     * Make the neurons for this layer. Initialize values to 0
     * @param size
     */
    private void makeNeurons (int size) {
        neurons  = new ArrayList<>();
        Random r = new Random();
        for (int i = 0; i < size; i++) {
            CNeuron neuron = CNeuronFactory.makeNeuron(
                    CNeuronFactory.CN_TYPE_SFTM, this, r.nextGaussian(), 0
            );
            neurons.add (neuron);
        }
    }

    @Override
    public CNeuron get(int idx) {
        return neurons.get(idx);
    }

    @Override
    public FullyConnectedLayer connect(FullyConnectedLayer fromLayer, int k) {
        // We initialize the weights to be 0 for softmax layers
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
        allZ();
        allZNeedsUpdate = false;
        for (Propagatable p : this)
            p.propagate();
        allZNeedsUpdate = true;
    }

    public double[] allZ () {
        if (allZNeedsUpdate) {
            double[] d = new double[size()];
            int size = size();
            for (int i = 0; i < size; i++) {
                SoftmaxCOutput outputF = (SoftmaxCOutput)get(i).outputFunction();
                d[i] = outputF.z();
            }
            allZ = d;
        }
        return allZ;
    }

    public double[] output() {
        double[] d = new double[size()];
        for (int i = 0; i < d.length; i++)
            d[i] = neurons.get(i).output();
        return d;
    }
}