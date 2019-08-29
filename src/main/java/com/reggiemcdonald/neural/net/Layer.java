package com.reggiemcdonald.neural.net;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * An abstraction of a layer in a neural network
 */
public class Layer implements Iterable<Neuron>, Serializable {
    private List<Neuron>  neurons;
    private int           layerIndex;
    private LayerType     type;

    public Layer (int layerIndex, LayerType type) {
        this.neurons     = new ArrayList<>();
        this.layerIndex  = layerIndex;
        this.type        = type;
    }

    public Layer (List<Neuron> neurons, int layerIndex, LayerType type) {
        this.neurons     = neurons;
        this.layerIndex  = layerIndex;
        this.type        = type;
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
     * Return the type of this layer
     * @return
     */
    public LayerType type () {
        return type;
    }

    /**
     * Returns the index of this layer in the network
     * @return
     */
    public int layerIndex () {
        return layerIndex;
    }

    /**
     * Returns the number of neurons in this layer
     * @return
     */
    public int size () {
        return neurons.size();
    }

    /**
     * Set this as the parent layer to all neurons in this layer
     */
    private void setThisAsParentLayer () {
        for (int i = 0 ; i < neurons.size() ; i++)
            neurons.get(i).setLayerAndIndex(this, i);
    }

    /**
     * Returns the activation array of this layer
     * @return
     */
    public double[] activations () {
        double[] a = new double[size()];
        for (Neuron n : neurons)
            a[n.neuralIndex()] = n.getOutput();
        return a;
    }

    public double [] deriveActivations (double[] activations) {
        for (Neuron n : neurons) {
            int idx = n.neuralIndex();
            activations[idx] = n.outputFunction().derivative(activations[idx]);
        }
        return activations;
    }


    @Override
    public Iterator<Neuron> iterator() {
        return neurons.iterator();
    }
}
