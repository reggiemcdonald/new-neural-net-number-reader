package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.Propagatable;
import com.reggiemcdonald.neural.convolutional.net.learning.layer.CLayerLearner;

public interface CNNLayer extends Iterable<CNeuron>, Propagatable {
    // TODO: Specify the CNN Layer

    /**
     * Returns a neuron at (x,y) in the CNNLayer
     * @param x the x-coordinate
     * @param y the y-coordinate
     * @return the neuron
     */
    CNeuron get(int x, int y);

    /**
     * Returns a neuron at idx (linear get)
     * @param idx
     * @return
     */
    CNeuron get (int idx);

    /**
     * Creates connections from from, to this
     * @param from
     * @param k the kernel level to use for weights where inputs to the convolution layer have d channels,
     *          0 <= k < d
     * @return
     */
    CNNLayer connect (CNNLayer from, int k);



    /**
     * Returns the number of neurons in this layer
     * @return
     */
    int size();

    /**
     * Returns the x-dimension of this neuron;
     * size() if linear
     * @return
     */
    int dim_x();

    /**
     * returns the y-dimension of this neuron
     * 1 if linear
     * @return
     */
    int dim_y();

    /**
     * @return the window width
     */
    int window_width();

    /**
     * @return the layer learner for this
     */
    CLayerLearner learner();


}
