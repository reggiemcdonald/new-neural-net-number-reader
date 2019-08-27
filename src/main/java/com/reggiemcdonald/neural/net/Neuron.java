package com.reggiemcdonald.neural.net;

import java.util.List;

/**
 * An abstract neuron, representing a node in an artifical neural network
 */

public interface Neuron {

    /**
     * Return the outputting value from this
     * @return
     */
    double getOutput ();

    Neuron setOutput (float signal);

    void propagate ();

    double getBias ();

    void setBias (float bias);

    void addBiasUpdate (float biasUpdate);

    double getBiasUpdate ();

    void zeroBiasUpdate ();

    Neuron addSynapseFromThis (Synapse s);

    Neuron addSyanpseToThis (Synapse s);

    List<Synapse> getSynapsesFromThis ();

    List<Synapse> getSynapseToThis ();





}
