package com.reggiemcdonald.neural.convolutional.net.layer.fc;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.Propagatable;
import com.reggiemcdonald.neural.convolutional.net.learning.layer.fc.FullyConnectedLayerLearner;

public interface FullyConnectedLayer extends Iterable<CNeuron>, Propagatable {
    // TODO: Specify the CNN Layer

    /**
     * Returns a neuron at idx (linear get)
     * @param idx
     * @return
     */
    CNeuron get (int idx);

    /**
     * Creates connections from from, to this
     * @param fromLayer layer
     * @return
     */
    FullyConnectedLayer connect (FullyConnectedLayer fromLayer);

    /**
     * Returns the number of neurons in this layer
     * @return
     */
    int size();

    /**
     * @return the layer learner for this
     */
    FullyConnectedLayerLearner learner();

}
