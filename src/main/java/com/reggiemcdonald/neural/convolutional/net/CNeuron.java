package com.reggiemcdonald.neural.convolutional.net;

import com.reggiemcdonald.neural.convolutional.math.COutput;
import com.reggiemcdonald.neural.convolutional.net.learning.CLearner;
import com.reggiemcdonald.neural.feedforward.net.Synapse;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A neuron in the convolutional neural network
 * Synapses from the feedforward network are sufficient
 * Initialize hyperparameters with gaussians
 */
public class CNeuron {
    // TODO
    private List<Synapse> toThis, fromThis;
    private double bias, output;
    private CLearner learner;
    private COutput outputFunction;

    /**
     * Default no-arg constructor
     */
    public CNeuron () {
        Random r       = new Random     ();
        toThis         = new ArrayList<>();
        fromThis       = new ArrayList<>();
        bias           = r.nextGaussian ();
        output         = r.nextGaussian ();
        learner        = null;
        outputFunction = null;
    }

    /**
     * Constructor to allow user to avoid creating many Random instances by
     * supplying the initialized values
     * @param bias
     * @param output
     */
    public CNeuron (double bias, double output) {
        this.toThis    = new ArrayList<>();
        this.fromThis  = new ArrayList<>();
        this.bias      = bias;
        this.output    = output;
        learner        = null;
        outputFunction = null;
    }

    /**
     * sets the learner to the specified learner, returning this
     * @param learner
     * @return
     */
    public CNeuron learner (CLearner learner) {
        this.learner = learner;
        return this;
    }

    /**
     * Sets the output function to the specified output function, returning this
     * @param outputFunction
     * @return
     */
    public CNeuron outputFunction (COutput outputFunction) {
        this.outputFunction = outputFunction;
        return this;
    }

    /**
     * Returns the learner of this
     * @return
     */
    public CLearner learner () {
        return learner;
    }

    /**
     * Returns the COutput of this
     * @return
     */
    public COutput outputFunction () {
        return outputFunction;
    }
 }
