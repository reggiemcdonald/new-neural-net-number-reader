package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.CNeuronFactory;
import com.reggiemcdonald.neural.convolutional.net.CSynapse;

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

    public InputLayer (int size, int stride) {
        if (Math.pow (Math.sqrt(size), 2) != size)
            throw new RuntimeException("Initialization Error: Layer size must be a perfect square");
        dim_x = dim_y = (int) Math.sqrt (size); // Right now, assume this has to be square
        this.stride = stride;
        makeNeurons (size);
    }

    @Override
    public CNeuron get(int x, int y) {
        int idx = (int) (dim_x * x) + y - 1;
        return neurons.get (idx);
    }

    @Override
    public CNeuron get(int idx) {
        return neurons.get(idx);
    }

    @Override
    public CNNLayer connect(CNNLayer to) {
        int x = 0, y = 0;
        int window = to.dim_x();
        for (CNeuron neuron : to) {
            makeConnections (neuron, x, y, window);
            if (x + window < dim_x) {
                x += window;
            } else {
                x = 0;
                y += window;
            }
        }
        return this;
    }

    private void makeConnections (CNeuron to, int x, int y, int window) {
        int currX = 0;
        Random r = new Random();
        while (currX < window) {
            int currY = 0;
            while (currY < window) {
                to.addConnectionToThis(
                        new CSynapse(get(currX+x, currY+y),to, r.nextGaussian(), window*window)
                );
                currY++;
            }
            currX++;
        }
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
     * @param size
     */
    private void makeNeurons (int size) {
        neurons = new ArrayList<>(size);
        Random r = new Random();
        for (int i = 0; i < size; i++)
            neurons.add (CNeuronFactory.makeNeuron (CNeuronFactory.CN_TYPE_INPT, 1., r.nextGaussian()));
    }

    @Override
    public Iterator<CNeuron> iterator() {
        return neurons.iterator();
    }
}
