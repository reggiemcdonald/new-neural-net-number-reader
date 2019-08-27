package com.reggiemcdonald.neural.net;

import com.reggiemcdonald.neural.net.math.OutputFunction;

import java.util.List;

/**
 * An abstract neuron, representing a node in an artificial neural network
 */

public interface Neuron {

    /**
     * Return the last set output
     * @return
     */
    double getOutput ();

    /**
     * Compute the output of this
     * @return
     */
    double compute ();

    /**
     * Sets the output of this neuron
     * @param signal
     * @return
     */
    Neuron setOutput (float signal);

    /**
     * Propagates signal forward through the network
     */
    void propagate ();

    /**
     * Gets the bias of this neuron
     * @return
     */
    double getBias ();

    /**
     * Sets the bias of this neuron
     * @param bias
     */
    void setBias (float bias);

    /**
     * Adds to the bias update to this neuron
     * @param biasUpdate
     */
    void addBiasUpdate (float biasUpdate);

    /**
     * Returns the bias update of this neuron
     * @return
     */
    double getBiasUpdate ();

    /**
     * Sets the bias update of this neuron to zero
     */
    void zeroBiasUpdate ();

    /**
     * Adds a synapse from this neuron, to another neuron in the next layer
     * @param s
     * @return
     */
    Neuron addSynapseFromThis (Synapse s);

    /**
     * Adds a synapse to this neuron, from a differnt neuron
     * @param s
     * @return
     */
    Neuron addSynapseToThis(Synapse s);

    /**
     * Returns the list of synapses from this neuron, to neurons in the next layer
     * @return
     */
    List<Synapse> getSynapsesFromThis ();

    /**
     * Returns the list of synapses from neurons of the previous layer, to this neuron
     * @return
     */
    List<Synapse> getSynapseToThis ();

    /**
     * Returns the layer that this neuron is contained within
     * @return
     */
    Layer layer ();

    /**
     * Sets the parent layer of this, and its index in the layer
     */
    void setLayerAndIndex (Layer layer, int index);


    /**
     * Returns the position of the neuron in the layer
     * @return
     */
    int getNeuralIndex ();

    /**
     * Returns the outputting function for this neuron
     * @return
     */
    OutputFunction outputFunction ();


}
