package com.reggiemcdonald.neural.convolutional.net;

public class CSynapse {
    private CNeuron from, to;
    private double weight;

    public CSynapse (CNeuron from, CNeuron to, double weight) {
        this.from   = from;
        this.to     = to;
        this.weight = weight;
    }

    public CSynapse (CNeuron from, CNeuron to, double gaussian, int n_in) {
        this.from = from;
        this.to   = to;
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
     * Returns the sending neuron
     * @return
     */
    public CNeuron to () {
        return to;
    }

    /**
     * Returns the receiving neuron
     * @return
     */
    public CNeuron from () {
        return from;
    }

    /**
     * Sets the sending neuron
     * @param to
     */
    public void to (CNeuron to) {
        this.to = to;
    }

    /**
     * Sets the receiving neuron
     * @param from
     */
    public void from (CNeuron from) {
        this.from = from;
    }

    /**
     * Gets the weight of this connection
     * @return
     */
    public double weight () {
        return weight;
    }

    /**
     * Sets the weight of this connection
     * @param weight
     */
    public void weight (double weight) {
        this.weight = weight;
    }
}
