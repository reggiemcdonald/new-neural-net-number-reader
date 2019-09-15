package com.reggiemcdonald.neural.convolutional.net;

import com.reggiemcdonald.neural.convolutional.math.COutput;
import com.reggiemcdonald.neural.convolutional.net.layer.CNNLayer;
import com.reggiemcdonald.neural.convolutional.net.learning.neuron.CLearner;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A neuron in the convolutional neural network
 * Synapses from the feedforward network are sufficient
 * Initialize hyperparameters with gaussians
 */
public class CNeuron implements Propagatable{
    // TODO
    private List<CSynapse> toThis, fromThis;
    private double bias, output, biasUpdate;
    private CLearner learner;
    private COutput outputFunction;
    private CNNLayer layer;

    /**
     * Default no-arg constructor
     */
    public CNeuron () {
        Random r       = new Random     ();
        toThis         = new ArrayList<>();
        fromThis       = new ArrayList<>();
        bias           = r.nextGaussian ();
        output         = r.nextGaussian ();
        biasUpdate     = 0.;
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
        this.toThis     = new ArrayList<>();
        this.fromThis   = new ArrayList<>();
        this.bias       = bias;
        this.output     = output;
        this.biasUpdate = 0.;
        learner         = null;
        outputFunction  = null;
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

    /**
     * Adds a connection to this
     * @param s
     * @return
     */
    public CNeuron addConnectionToThis (CSynapse s) {
        if (!toThis.contains(s)) {
            toThis.add (s);
            s.from().addConnectionFromThis(s);
        }
        return this;
    }

    /**
     * Adds a synapse from this to some other neuron
     * @param s
     * @return
     */
    public CNeuron addConnectionFromThis (CSynapse s) {
        if (!fromThis.contains(s)) {
            fromThis.add (s);
            s.to().addConnectionToThis(s);
        }
        return this;
    }

    /**
     * Removes a synapse to this
     * @param s
     * @return
     */
    public CNeuron removeConnectionToThis (CSynapse s) {
        if (toThis.contains(s)) {
            toThis.remove(s);
            s.from().removeConnectionFromThis(s);
        }
        return this;
    }

    /**
     * Removes a synapse coming from this neuron
     * @param s
     * @return
     */
    public CNeuron removeConnectionFromThis (CSynapse s) {
        if (fromThis.contains(s)) {
            fromThis.remove(s);
            s.to().removeConnectionToThis(s);
        }
        return this;
    }

    /**
     * Returns the output of this
     * @return
     */
    public double output() {
        return output;
    }

    /**
     * sets the output of this
     * @param output
     */
    public void setOutput(double output) {
        this.output = output;
    }

    /**
     * Returns the bias of this neuron
     * @return
     */
    public double bias () {
        return bias;
    }

    /**
     * Sets the bias
     * @param bias
     */
    public void setBias (double bias) {
        this.bias = bias;
    }

    @Override
    public void propagate() {
        setOutput(outputFunction.compute());
    }

    /**
     * Returns a list of synapses to this
     * @return
     */
    public List<CSynapse> synapsesToThis () {
        return toThis;
    }

    /**
     * returns synapses from this
     * @return
     */
    public List<CSynapse> synapsesFromThis () {
        return fromThis;
    }

    /**
     * Returns the parent layer
     * @return
     */
    public CNNLayer layer () {
        return layer;
    }

    /**
     * Sets the layer of this
     * @param layer
     * @return
     */
    public CNeuron layer (CNNLayer layer) {
        this.layer = layer;
        return this;
    }

    /**
     * Returns the current amount of bias update
     * @return
     */
    public double biasUpdate () {return biasUpdate;}

    /**
     * sets the bias update, returning this
     * @param d
     * @return
     */
    public CNeuron biasUpdate (double d) {biasUpdate = d; return this;}

}
