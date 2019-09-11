package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.CNeuronFactory;
import com.reggiemcdonald.neural.convolutional.net.CSynapse;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

public class SoftmaxLayer implements CNNLayer {
    private List<CNeuron> neurons;

    public SoftmaxLayer (int size) {
        makeNeurons (size);
    }

    /**
     * Make the neurons for this layer
     * @param size
     */
    private void makeNeurons (int size) {
        Random r = new Random();
        neurons  = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            CNeuron neuron = CNeuronFactory.makeNeuron(
                    CNeuronFactory.CN_TYPE_SFTM, r.nextGaussian(), r.nextGaussian()
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
        Random r = new Random();
        CNNLayer toLayer = this;
        for (CNeuron to : toLayer)
            for (CNeuron from : fromLayer)
                to.addConnectionToThis(new CSynapse(from, to, r.nextGaussian(), fromLayer.size()));
        return this;
    }

    @Override
    public int size() {
        // TODO: Stub
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
    public Iterator<CNeuron> iterator() {
        return neurons.iterator();
    }

}