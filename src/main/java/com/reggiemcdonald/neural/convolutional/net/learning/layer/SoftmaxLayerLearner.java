package com.reggiemcdonald.neural.convolutional.net.learning.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.layer.CNNLayer;
import com.reggiemcdonald.neural.convolutional.net.layer.SoftmaxLayer;
import com.reggiemcdonald.neural.convolutional.net.learning.neuron.CLearner;

public class SoftmaxLayerLearner implements CLayerLearner {
    private CNNLayer layer;

    public SoftmaxLayerLearner(CNNLayer layer) {
        this.layer = layer;
    }

    /**
     * Compute the cost of the softmax layer
     * @param deltaNextLayer
     * @return
     */
    public double[] delta (double[] deltaNextLayer) {
        double[] outputs = ((SoftmaxLayer)layer).output();
        return crossEntropyCost(outputs, deltaNextLayer);
    }

    @Override
    public CLayerLearner incrementBiasUpdate(double[] delta) {
        for (int i = 0; i < delta.length; i++) {
            layer.get(i).learner().incrementBiasUpdate(delta[i]);
        }
        return this;
    }

    @Override
    public CLayerLearner incrementWeightUpdate(double[] delta) {
        for (int i = 0; i < layer.size(); i++) {
            CNeuron neuron = layer.get(i);
            neuron.learner().incrementWeightUpdate(deltaWeight(neuron, delta[i]));
        }
        return this;
    }

    @Override
    public CLayerLearner finalizeLearning(int batchSize, double eta) {
        for (CNeuron neuron : layer) {
            CLearner learner = neuron.learner();
            learner.applyBiasUpdate(batchSize, eta);
            learner.applyWeightUpdate(batchSize, eta);
        }
        return this;
    }

    /**
     * Compute the partial derivative with respect to weight
     * using the partial derivative with respect to the neural bias
     * @param delta
     * @return
     */
    private double[] deltaWeight (CNeuron neuron, double delta) {
        return neuron.learner().deltaWeight(delta);
    }

    private double[] crossEntropyCost (double[] outputs, double[] expected) {
        for (int i = 0; i < outputs.length; i++)
            outputs[i] -= expected[i];
        return outputs;
    }

}
