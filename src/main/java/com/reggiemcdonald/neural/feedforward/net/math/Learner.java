package com.reggiemcdonald.neural.feedforward.net.math;

import java.io.Serializable;

public interface Learner extends Serializable {

    /**
     * Computes the amount of weight update to apply
     * @return
     * @param error
     */
    double[] computeWeightUpdates(double error);

    /**
     * Computes the amount of bias update to apply
     * @return
     */
    double computeBiasUpdate (double error);

    /**
     * Applies the weight update to the parent neuron
     * @param n
     * @param eta
     */
    void applyWeightUpdate (int n, double eta);

    /**
     * Applies the bias update to the parent neuron
     * @param n
     * @param eta
     */
    void applyBiasUpdate (int n, double eta);

    /**
     * Adds biasUpdate to the current amount of bias update
     * @param biasUpdate
     */
    void addToBiasUpdate (double biasUpdate);

    /**
     * Increments the weight updates with the given weightUpdate array
     * @param weightUpdate
     */
    void addToWeightUpdate (double[] weightUpdate);

    /**
     * Returns the current sum of bias updates (not averaged)
     * @return
     */
    double biasUpdateSum ();

    /**
     * Returns the current sum of weight updates (not averaged)
     * @return
     */
    double[] weightUpdateSum ();

    /**
     * Return the delta value for this neuron
     * @param deltaNext
     * @return
     */
    double delta (double[] deltaNext);
}
