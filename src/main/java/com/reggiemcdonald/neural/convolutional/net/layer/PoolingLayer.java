package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.CNeuronFactory;
import com.reggiemcdonald.neural.convolutional.net.CSynapse;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class PoolingLayer implements CNNLayer {
    private int dim_x, dim_y, window_width;
    private boolean isMaxPooling = true; // Default to this because its better
    private List<CNeuron> neurons;

    public PoolingLayer (int dim_x, int dim_y, int window_width) {
        this.dim_x        = dim_x;
        this.dim_y        = dim_y;
        this.window_width = window_width;
        makeNeurons ();
    }

    public PoolingLayer (int dim_x, int dim_y, int window_width, boolean isMaxPooling) {
        this.dim_x        = dim_x;
        this.dim_y        = dim_y;
        this.window_width = window_width;
        this.isMaxPooling = isMaxPooling;
        makeNeurons ();
    }

    /**
     * Produce the neurons that are present in this layer
     */
    private void makeNeurons () {
        this.neurons = new ArrayList<>();
        int size = dim_x * dim_y;
        for (int i = 0; i < size; i++) {
            CNeuron neuron = CNeuronFactory
                    .makeNeuron(CNeuronFactory.CN_TYPE_POOL, 1.0, 0.);
            neurons.add (neuron);
        }
    }

    @Override
    public CNeuron get(int x, int y) {
        int idx = (x * dim_x) + y;
        return neurons.get(idx);
    }

    @Override
    public CNeuron get(int idx) {
        return neurons.get(idx);
    }

    @Override
    public CNNLayer connect(CNNLayer from, int k) {
        int x = 0, y = 0;
        for (CNeuron neuron : neurons) {
            makeConnections(neuron, from, x, y);
            if (x + window_width < from.dim_x())
                x++;
            else {
                x = 0;
                y++;
            }
        }
        return this;
    }

    /**
     * Produce connections from layer from to the given neuron to
     * For pooling layers, we keep the weights of the connections to be 1
     * @param to the neuron receiving the connections
     * @param from the layer that is sending
     * @param x the x coordinate of the upper-left corner of the window
     * @param y the y coordinate in the upper-left corner of the window
     */
    private void makeConnections(CNeuron to, CNNLayer from, int x, int y) {
        int currX = 0;
        while (currX < window_width) {
            int currY = 0;
            while (currY < window_width) {
                to.addConnectionToThis(
                        new CSynapse(from.get(currX+x, currY+y),to, 1.0, window_width * window_width)
                );
                currY++;
            }
            currX++;
        }
    }

    @Override
    public int size() {
        // TODO: Stub
        return neurons.size();
    }

    @Override
    public int dim_x() {
        return dim_x;
    }

    @Override
    public int dim_y() {
        return dim_y;
    }

    @Override
    public Iterator<CNeuron> iterator() {
        return neurons.iterator();
    }
}
