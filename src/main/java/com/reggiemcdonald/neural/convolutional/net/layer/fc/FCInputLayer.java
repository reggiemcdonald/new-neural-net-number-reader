package com.reggiemcdonald.neural.convolutional.net.layer.fc;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.CNeuronFactory;
import com.reggiemcdonald.neural.convolutional.net.CSynapse;
import com.reggiemcdonald.neural.convolutional.net.learning.layer.fc.FCInputLayerLearner;
import com.reggiemcdonald.neural.convolutional.net.learning.layer.fc.FullyConnectedLayerLearner;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * A traditional MLP Input Layer, but with specialized learning
 * for propagation back to the final CNN Pooling layer
 */
public class FCInputLayer implements FullyConnectedLayer {
    private List<CNeuron> neurons;
    private FullyConnectedLayerLearner learner;

    public FCInputLayer (int size) {
        this.neurons  = new ArrayList<> (size);
        this.learner  = new FCInputLayerLearner(this);
        makeNeurons (size);
    }

    /**
     * Generate the neurons for this layer, setting all their biases to 1
     * @param size
     */
    private void makeNeurons (int size) {
        for (int i = 0 ; i < size ; i++) {
            CNeuron neuron = CNeuronFactory.makeNeuron(
                    CNeuronFactory.CN_TYPE_INPT,
                    this,
                    1.,
                    0.
            );
            neurons.add (neuron);
        }
    }

    @Override
    public CNeuron get(int idx) {
        return null;
    }

    @Override
    public FullyConnectedLayer connect(FullyConnectedLayer fromLayer) {
        Random r = new Random();
        FullyConnectedLayer toLayer = this;
        for (CNeuron to : toLayer)
            for (CNeuron from : fromLayer)
                to.addConnectionToThis(new CSynapse(from, to, r.nextGaussian(), fromLayer.size()));
        return this;
    }

    @Override
    public int size() {
        return neurons.size();
    }

    @Override
    public FullyConnectedLayerLearner learner() {
        return learner;
    }

    @Override
    public void propagate() {
        // This doesn't need to do anything in the input layer
    }

    @Override
    public Iterator<CNeuron> iterator() {
        return neurons.iterator();
    }

    /**
     * Set the input layer
     * @param inputs
     */
    public void setAll (double[] inputs) {
        for (int i = 0 ; i < inputs.length; i++) {
            neurons.get(i).setOutput(inputs[i]);
        }
    }
}
