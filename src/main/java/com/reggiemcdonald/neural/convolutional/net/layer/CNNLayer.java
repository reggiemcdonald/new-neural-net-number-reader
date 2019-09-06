package com.reggiemcdonald.neural.convolutional.net.layer;

public interface CNNLayer {
    // TODO: Specify the CNN Layer
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
