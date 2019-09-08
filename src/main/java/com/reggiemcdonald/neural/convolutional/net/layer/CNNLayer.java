package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;

public interface CNNLayer extends Iterable<CNeuron> {
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
     * @return
     */
    CNNLayer connect (CNNLayer from);



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


}
