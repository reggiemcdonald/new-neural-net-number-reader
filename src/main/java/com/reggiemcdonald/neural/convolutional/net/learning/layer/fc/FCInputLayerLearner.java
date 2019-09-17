package com.reggiemcdonald.neural.convolutional.net.learning.layer.fc;

import com.reggiemcdonald.neural.convolutional.net.layer.fc.FullyConnectedLayer;

public class FCInputLayerLearner implements FullyConnectedLayerLearner {
    private FullyConnectedLayer layer;

    public FCInputLayerLearner (FullyConnectedLayer layer) {
        this.layer = layer;
    }

    @Override
    public double[][] delta(double[][] deltaNextLayer) {
        return new double[0][];
    }

    @Override
    public FullyConnectedLayerLearner incrementBiasUpdate(double[][] delta) {
        return null;
    }

    @Override
    public FullyConnectedLayerLearner incrementWeightUpdate(double[][] delta) {
        return null;
    }

    @Override
    public FullyConnectedLayerLearner finalizeLearning(int batchSize, double eta) {
        return null;
    }
}
