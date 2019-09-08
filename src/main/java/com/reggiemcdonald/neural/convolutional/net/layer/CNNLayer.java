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
     * Connects this layer to an adjacent layer
     * @param to
     * @return
     */
    CNNLayer connect (CNNLayer to);



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
