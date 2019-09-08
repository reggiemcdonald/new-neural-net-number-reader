package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;

public interface CNNLayer {
    // TODO: Specify the CNN Layer

    /**
     * Returns a neuron at (x,y) in the CNNLayer
     * @param x the x-coordinate
     * @param y the y-coordinate
     * @return the neuron
     */
    CNeuron getNeuron (int x, int y);

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


}
