package com.reggiemcdonald.neural.convolutional.net.learning.layer;

public interface CLayerLearner {

    /**
     * Compute the cost of the layer
     * @param deltaNextLayer
     * @return
     */
    double[] delta(double[] deltaNextLayer);

    /**
     * Applies a bias update to the current layer
     * @param delta
     * @return
     */
    CLayerLearner applyBiasUpdates(double[] delta);

    /**
     * Applies weight updates to the current layer
     * @param delta
     * @return
     */
    CLayerLearner applyWeightUpdates(double[] delta);

    /**
     * Applies the learning to the given layer
     * @return
     */
    CLayerLearner finalizeLearning(int batchSize, double eta);
}
