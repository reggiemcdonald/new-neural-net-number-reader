package com.reggiemcdonald.neural.convolutional.net.learning.layer;

import com.reggiemcdonald.neural.convolutional.net.layer.CNNLayer;
import com.reggiemcdonald.neural.convolutional.net.layer.SoftmaxLayer;

public class SoftmaxLayerLearner implements CLayerLearner {
    private CNNLayer layer;

    /**
     * Compute the cost of the softmax layer
     * @param deltaNextLayer
     * @return
     */
    public double[] delta (double[] deltaNextLayer) {
        double[] outputs = ((SoftmaxLayer)layer).output();
        return crossEntropyCost(outputs, deltaNextLayer);
    }

    public double[] deltaWeight (double[] delta) {
        double[] deltaWeight = new double[delta.length];
        for (int i = 0; i < delta.length; i++)
            deltaWeight[i] = delta[i] * layer.get(i).output();
        return deltaWeight;
    }

    private double[] crossEntropyCost (double[] outputs, double[] expected) {
        for (int i = 0; i < outputs.length; i++)
            outputs[i] -= expected[i];
        return outputs;
    }

}
