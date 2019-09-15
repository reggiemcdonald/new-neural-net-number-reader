package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.CNeuronFactory;
import com.reggiemcdonald.neural.convolutional.net.CSynapse;
import com.reggiemcdonald.neural.convolutional.net.Propagatable;
import com.reggiemcdonald.neural.convolutional.net.learning.layer.CLayerLearner;
import com.reggiemcdonald.neural.convolutional.net.learning.layer.ConvolutionLayerLearner;

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
    private int stride;
    private double[][][] kernel;
    private double bias;
    private CLayerLearner learner;

    public ConvolutionLayer (int dim_x, int dim_y, int window_width, double[][][] kernel, int stride) {
        assert (dim_x == dim_y);
        Random r          = new Random ();
        this.dim_x        = dim_x;
        this.dim_y        = dim_y;
        this.window_width = window_width;
        this.kernel       = kernel;
        this.bias         = r.nextGaussian();
        this.stride       = stride;
        // New CLayerLearner
        learner = new ConvolutionLayerLearner(this);
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
            if (x + window_width < from.dim_x() - 1) {
                x++;
            } else {
                x = 0;
                y++;
            }
        }
        return this;
    }

    private void makeConnections (CNeuron to, CNNLayer from, int x, int y, int k) {
        int currX = 0;

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
    public int window_width() {
        return window_width;
    }

    @Override
    public CLayerLearner learner() {
        return learner;
    }

    @Override
    public Iterator<CNeuron> iterator() {
        return neurons.iterator();
    }

    private void makeNeurons () {
        neurons = new ArrayList<>();
        int size = dim_x() * dim_y();
        for (int i = 0; i < size; i++) {
            CNeuron neuron = CNeuronFactory
                    .makeNeuron(CNeuronFactory.CN_TYPE_CONV, this, bias, 0.);
            neurons.add (neuron);
        }
    }

    @Override
    public void propagate() {
        for (Propagatable p : this)
            p.propagate();
    }

    public int stride () {
        return stride;
    }
}
