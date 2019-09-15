package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.math.SigmoidCOutput;
import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.CNeuronFactory;
import com.reggiemcdonald.neural.convolutional.net.CSynapse;
import com.reggiemcdonald.neural.convolutional.net.Propagatable;
import com.reggiemcdonald.neural.convolutional.net.learning.layer.CLayerLearner;
import com.reggiemcdonald.neural.convolutional.net.learning.layer.SoftmaxLayerLearner;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class SoftmaxLayer implements CNNLayer {
    private List<CNeuron> neurons;
    private double[] allZ;
    private boolean allZNeedsUpdate = true;
    private CLayerLearner learner;

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
        for (int i = 0; i < size; i++) {
            CNeuron neuron = CNeuronFactory.makeNeuron(
                    CNeuronFactory.CN_TYPE_SFTM, this, 0, 0
            );
            neurons.add (neuron);
        }
    }

    @Override
    public CNeuron get(int x, int y) {
        return neurons.get(y);
    }

    @Override
    public CNeuron get(int idx) {
        return neurons.get(idx);
    }

    @Override
    public CNNLayer connect(CNNLayer fromLayer, int k) {
        // We initialize the weights to be 0 for softmax layers
        CNNLayer toLayer = this;
        for (CNeuron to : toLayer)
            for (CNeuron from : fromLayer)
                to.addConnectionToThis(new CSynapse(from, to, 0., fromLayer.size()));
        return this;
    }

    @Override
    public int size() {
        return neurons.size();
    }

    @Override
    public int dim_x() {
        return 0;
    }

    @Override
    public int dim_y() {
        return 0;
    }

    @Override
    public int window_width() {
        // Does not have a window width
        return 0;
    }

    @Override
    public CLayerLearner learner() {
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
                SigmoidCOutput outputF = (SigmoidCOutput)get(i).outputFunction();
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