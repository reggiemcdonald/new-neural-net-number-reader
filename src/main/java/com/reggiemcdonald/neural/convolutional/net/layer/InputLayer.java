package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;

import java.util.ArrayList;
import java.util.List;

/**
 * An layer of inputs to a convolutional neural network
 * Neurons will be stored in a linear list, but accessible as if they were
 * stored in a 2-Dimensional data structure (size is a perfect square)
 */
public class InputLayer implements CNNLayer {
    private List<CNeuron> neurons;
    private double dim_x, dim_y;

    public InputLayer (int size) {
        if (Math.pow (Math.sqrt(size), 2) != size)
            throw new RuntimeException("Initialization Error: Layer size must be a perfect square");
        dim_x = dim_y = Math.sqrt (size); // Right now, assume this has to be square
        makeNeurons (size);
    }

    @Override
    public CNeuron getNeuron(int x, int y) {
        int idx = (int) (dim_x * x) + y - 1;
        return neurons.get (idx);
    }

    @Override
    public CNNLayer connect(CNNLayer to) {
        return this;
    }

    @Override
    public int size() {
        // TODO: Stub
        return 0;
    }

    /**
     * Produce size neurons and add them to this layer
     * @param size
     */
    private void makeNeurons (int size) {
        neurons = new ArrayList<>(size);
        for (int i = 0; i < size; i++)
            neurons.add (new CNeuron());
    }
}
