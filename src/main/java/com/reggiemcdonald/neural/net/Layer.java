package com.reggiemcdonald.neural.net;

import java.util.ArrayList;
import java.util.List;

/**
 * An abstraction of a layer in a neural network
 */
public class Layer {
    private List<Neuron>  neurons;
    private int           layerIndex;

    public Layer (int layerIndex) {
        this.neurons     = new ArrayList<>();
        this.layerIndex  = layerIndex;
    }

    public Layer (List<Neuron> neurons, int layerIndex) {
        this.neurons     = neurons;
        this.layerIndex  = layerIndex;
        setThisAsParentLayer ();
    }

    // TODO: layer-wise computation

    public void update () {
        for (Neuron neuron : neurons)
            neuron.compute ();
    }

    /**
     * Return the neuron at the specified neural index
     * @param neuralIndex
     * @return
     */
    public Neuron getNeuronAt (int neuralIndex) {
        return neurons.get (neuralIndex);
    }

    /**
     * Returns true if neuron is in this layer
     * @param neuron
     * @return
     */
    public boolean contains (Neuron neuron) {
        return neurons.contains (neuron);
    }

    /**
     * Adds a neuron to this layer and sets this as its parent layer
     * @param neuron
     * @return
     */
    public Layer addNeuronToLayer (Neuron neuron) {
        if (neurons.add (neuron))
            neuron.setLayerAndIndex (this, neurons.size () - 1);
        return this;
    }

    /**
     * Removes the given neuron from this layer, if it is currently in this layer
     * @param neuron
     * @return
     */
    public Layer removeNeuronFromLayer (Neuron neuron) {
        if (neurons.remove (neuron)) {
            neuron.setLayerAndIndex (null, -1);
        }
        return this;
    }

    /**
     * Set this as the parent layer to all neurons in this layer
     */
    private void setThisAsParentLayer () {
        for (int i = 0 ; i < neurons.size() ; i++)
            neurons.get(i).setLayerAndIndex(this, i);
    }




}
