package com.reggiemcdonald.neural.convolutional.net;

import com.reggiemcdonald.neural.feedforward.net.Synapse;

import java.util.List;

/**
 * A neuron in the convolutional neural network
 * Synapses from the feedforward network are sufficient
 * Initialize hyperparameters with gaussians that have a restrained
 * standard deviation to avoid middle-layer saturation
 */
public class CNeuron {
    // TODO
    private List<Synapse> toThis, fromThis;
    private double bias, output;
    /**
     * Default no-arg constructor
     */
    public CNeuron () {
        // TODO
    }

    /**
     * Constructor to allow user to avoid creating many Random instances by
     * supplying the initialized values
     * @param bias
     * @param output
     */
    public CNeuron (double bias, double output) {

    }
 }
