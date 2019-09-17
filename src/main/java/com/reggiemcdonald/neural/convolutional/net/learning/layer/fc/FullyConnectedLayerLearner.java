package com.reggiemcdonald.neural.convolutional.net.learning.layer.fc;

public interface FullyConnectedLayerLearner {

    /**
     * Compute the cost of the layer
     * @param deltaNextLayer
     * @return
     */
    double[][] delta(double[] deltaNextLayer);

    /**
     * Applies a bias update to the current layer
     * @param delta
     * @return
     */
    FullyConnectedLayerLearner incrementBiasUpdate(double[] delta);

    /**
     * Applies weight updates to the current layer
     * @param delta
     * @return
     */
    FullyConnectedLayerLearner incrementWeightUpdate(double[] delta);

    /**
     * Applies the learning to the given layer
     * @return
     */
    FullyConnectedLayerLearner finalizeLearning(int batchSize, double eta);
}
