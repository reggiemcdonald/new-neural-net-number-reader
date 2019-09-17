package com.reggiemcdonald.neural.convolutional.net.learning.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.layer.fc.FullyConnectedLayer;
import com.reggiemcdonald.neural.convolutional.net.layer.fc.SoftmaxLayer;
import com.reggiemcdonald.neural.convolutional.net.learning.neuron.CLearner;
import com.reggiemcdonald.neural.convolutional.net.learning.neuron.SoftmaxCLearner;

public class SoftmaxLayerLearner implements FullyConnectedLayerLearner {
    private FullyConnectedLayer layer;

    public SoftmaxLayerLearner(FullyConnectedLayer layer) {
        this.layer = layer;
    }

    /**
     * Compute the cost of the softmax layer
     * @param deltaNextLayer
     * @return
     */
    public double[][] delta (double[][] deltaNextLayer) {
        double[][] outputs = new double[1][];
        outputs[0] = ((SoftmaxLayer)layer).output();
        return crossEntropyCost(outputs, deltaNextLayer);
    }

    @Override
    public FullyConnectedLayerLearner incrementBiasUpdate(double[][] delta) {
        for (int i = 0; i < delta.length; i++) {
            layer.get(i).learner().incrementBiasUpdate(delta[0][i]);
        }
        return this;
    }

    @Override
    public FullyConnectedLayerLearner incrementWeightUpdate(double[][] delta) {
        for (int i = 0; i < layer.size(); i++) {
            CNeuron neuron = layer.get(i);
            neuron.learner().incrementWeightUpdate(deltaWeight(neuron, delta[0][i]));
        }
        return this;
    }

    @Override
    public FullyConnectedLayerLearner finalizeLearning(int batchSize, double eta) {
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
    private double[][] deltaWeight (CNeuron neuron, double delta) {
        return ((SoftmaxCLearner)neuron.learner()).deltaWeight(delta);
    }

    private double[][] crossEntropyCost (double[][] outputs, double[][] expected) {
        for (int i = 0; i < outputs[0].length; i++)
            outputs[0][i] -= expected[0][i];
        return outputs;
    }

}
