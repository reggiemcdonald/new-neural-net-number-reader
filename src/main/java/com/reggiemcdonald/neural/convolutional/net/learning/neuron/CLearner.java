package com.reggiemcdonald.neural.convolutional.net.learning.neuron;

/**
 * A learning interface for convolutional neural network
 */
public interface CLearner {
    /**
     * Returns the derivative of the activation function
     * @return
     */
    double derivative();

    /**
     * Increment the bias update to be applied
     * @param delta
     */
    void incrementBiasUpdate(double delta);

    /**
     * Increment the weight update to be applied
     * @param delta
     */
    void incrementWeightUpdate(double[][] delta);

    /**
     * Applies the bias update, then resets them
     * @param batchSize
     * @param eta
     */
    void applyBiasUpdate(int batchSize, double eta);

    /**
     * Applies the weight updates, then resets them
     * @param batchSize
     * @param eta
     */
    void applyWeightUpdate(int batchSize, double eta);


    /**
     * Sets the weight updates
     * @param weightUpdates
     * @return
     */
    CLearner setWeightUpdates(double[][] weightUpdates);


}
