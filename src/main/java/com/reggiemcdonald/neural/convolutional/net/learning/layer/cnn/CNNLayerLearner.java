package com.reggiemcdonald.neural.convolutional.net.learning.layer.cnn;

import com.reggiemcdonald.neural.convolutional.net.layer.cnn.CNNLayer;

public interface CNNLayerLearner {
    /**
     * Compute the cost gradient
     * @param layerBefore
     * @param deltaNextLayer
     * @return
     */
    double[][] delta (CNNLayer layerBefore, double[][] deltaNextLayer);

    /**
     * Updates the aggregate bias updates
     * @param delta
     */
    void incrementBiasUpdates(double[][] delta);

    /**
     * Increments the aggregate weight updates
     * @param delta
     */
    void incrementWeightUpdates(double[][] delta);

    /**
     * Applies the aggregate updates averaged over batch size and scaled according to learning rate
     * @param batchSize
     * @param eta
     */
    void finalizeLearning (int batchSize, double eta);
}