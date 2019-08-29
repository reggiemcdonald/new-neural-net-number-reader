package com.reggiemcdonald.neural.net.math;

import java.io.Serializable;

public interface Learner extends Serializable {

    /**
     * Computes the amount of weight update to apply
     * @return
     */
    double computeWeightUpdate (double error);

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
     * Returns the current sum of bias updates (not averaged)
     * @return
     */
    double biasUpdateSum ();

    /**
     * Returns the current sum of weight updates (not averaged)
     * @return
     */
    double weightUpdateSum ();
}
