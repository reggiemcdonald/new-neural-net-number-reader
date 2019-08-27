package com.reggiemcdonald.neural.net;

/**
 * A class to represent the connection between two neurons in adjacent layers,
 * where this.from.layer().layerIndex() < this.to.layer().layerIndex() is strictly true
 */
public class Synapse {
    private Neuron from, to;
    private double weight;

    public Synapse (Neuron from, Neuron to, double weight) {
        this.from   = from;
        this.to     = to;
        this.weight = weight;
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
