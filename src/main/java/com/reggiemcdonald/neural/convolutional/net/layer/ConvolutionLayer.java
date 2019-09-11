package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.CNeuronFactory;
import com.reggiemcdonald.neural.convolutional.net.CSynapse;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * A convolutional layer in a CNN
 * The kernel is an array of weights sized according to the receiving window
 * that are common among each dim_x * dim_y neurons in the layer
 */
public class ConvolutionLayer implements CNNLayer {
    List<CNeuron> neurons;
    private int dim_x, dim_y;
    private int window_width;
    private double[][][] kernel;
    private double bias;

    public ConvolutionLayer (int dim_x, int dim_y, int window_width, double[][][] kernel) {
        assert (dim_x == dim_y);
        Random r          = new Random ();
        this.dim_x        = dim_x;
        this.dim_y        = dim_y;
        this.window_width = window_width;
        this.kernel       = kernel;
        this.bias         = r.nextGaussian();
        makeNeurons ();
    }

    @Override
    public CNeuron get (int x, int y) {
        int idx = x + (y * dim_y);
        return neurons.get(idx);
    }

    @Override
    public CNeuron get (int idx) {
        return neurons.get(idx);
    }

    @Override
    public CNNLayer connect (CNNLayer from, int k) {
        int x = 0, y = 0;
        for (CNeuron neuron : neurons) {
            makeConnections(neuron, from, x, y, k);
            if (x + window_width < from.dim_x()) {
                x ++;
            } else {
                x = 0;
                y ++;
            }
        }
        return this;
    }

    private void makeConnections (CNeuron to, CNNLayer from, int x, int y, int k) {
        int currX = 0;
        Random r = new Random();
        while (currX < window_width) {
            int currY = 0;
            while (currY < window_width) {
                to.addConnectionToThis(
                        new CSynapse(from.get(currX+x, currY+y),to, kernel[k][currX][currY], window_width * window_width)
                );
                currY++;
            }
            currX++;
        }
    }

    @Override
    public int size () {
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

    private void makeNeurons () {
        neurons = new ArrayList<>();
        Random r = new Random();
        int size = dim_x() * dim_y();
        for (int i = 0; i < size; i++) {
            CNeuron neuron = CNeuronFactory
                    .makeNeuron(CNeuronFactory.CN_TYPE_CONV, bias, r.nextGaussian());
            neurons.add (neuron);
        }
    }
}
