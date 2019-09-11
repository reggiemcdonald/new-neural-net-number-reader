package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.CNeuronFactory;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * An layer of inputs to a convolutional neural network
 * Neurons will be stored in a linear list, but accessible as if they were
 * stored in a 2-Dimensional data structure (size is a perfect square)
 */
public class InputLayer implements CNNLayer {
    private List<CNeuron> neurons;
    private int dim_x, dim_y;
    private int stride;

    public InputLayer (int dim, int stride) {
        dim_x = dim_y = dim;
        this.stride = stride;
        makeNeurons (dim);
    }

    @Override
    public CNeuron get(int x, int y) {
        int idx = (int) (dim_x * x) + y;
        return neurons.get (idx);
    }

    @Override
    public CNeuron get(int idx) {
        return neurons.get(idx);
    }

    @Override
    public CNNLayer connect(CNNLayer from, int k) {
        // Does nothing ??
        return this;
    }

    @Override
    public int size() {
        // TODO: Stub
        return 0;
    }

    @Override
    public int dim_x() {
        return dim_x;
    }

    @Override
    public int dim_y() {
        return dim_y;
    }

    /**
     * Produce size neurons and add them to this layer
     * Set bias to 1, as per typical output
     * @param dim
     */
    private void makeNeurons (int dim) {
        neurons = new ArrayList<>(dim*dim);
        Random r = new Random();
        for (int i = 0; i < dim*dim; i++)
            neurons.add (CNeuronFactory.makeNeuron (CNeuronFactory.CN_TYPE_INPT, 1., r.nextGaussian()));
    }

    @Override
    public Iterator<CNeuron> iterator() {
        return neurons.iterator();
    }

    @Override
    public void propagate() {
        // TODO: Propagate
    }
}
