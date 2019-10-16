package com.reggiemcdonald.neural.feedforward.net;

import java.io.Serializable;

/**
 * A class to represent the connection between two neurons in adjacent layers,
 * where this.from.layer().layerIndex() < this.to.layer().layerIndex() is strictly true
 */
public class Synapse implements Serializable {
    private Neuron from, to;
    private double weight;

    public Synapse (Neuron from, Neuron to, double weight) {
        this.from   = from;
        this.to     = to;
        this.weight = weight;
    }

    /**
     * Specify n_in to construct synapse with an optimal weight intialization
     * @param from
     * @param to
     * @param gaussian
     * @param n_in
     */
    public Synapse (Neuron from, Neuron to, double gaussian, int n_in) {
        this.from   = from;
        this.to     = to;
        this.weight = optimizeWeightInit (gaussian, n_in);
    }

    /**
     * Return an optimization of the initial weight for this synapse
     * which is a gaussian with a tightened distribution
     * std. dev. is 1/sqrt(n_in)
     * @param gaussian
     * @param n_in
     * @return
     */
    private double optimizeWeightInit (double gaussian, int n_in) {
        return gaussian * (1 / Math.sqrt(n_in));
    }

    /**
     * Sets the from neuron of this synapse
     * @param from
     */
    public void setFrom (Neuron from) {
        this.from = from;
    }

    /**
     * Sets the to neuron from this synapse
     * @param to
     */
    public void setTo (Neuron to) {
        this.to = to;
    }

    /**
     * Sets the weight of this synapse
     * @param weight
     */
    public void setWeight (double weight) {
        this.weight = weight;
    }

    /**
     * Returns the from neuron of this synapse
     * @return
     */
    public Neuron from () {
        return from;
    }

    /**
     * Returns the to neuron of this synapse
     * @return
     */
    public Neuron to () {
        return to;
    }

    /**
     * Returns the weight of this connection
     * @return
     */
    public double getWeight () {
        return weight;
    }

}
