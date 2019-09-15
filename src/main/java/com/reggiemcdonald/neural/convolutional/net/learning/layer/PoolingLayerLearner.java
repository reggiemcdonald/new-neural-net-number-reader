package com.reggiemcdonald.neural.convolutional.net.learning.layer;

import com.reggiemcdonald.neural.convolutional.net.layer.CNNLayer;

public class PoolingLayerLearner implements CLayerLearner {
    private CNNLayer layer;
    @Override
    public double[] delta(double[] delta_next) {
        return delta_next;
    }

    @Override
    public CLayerLearner incrementBiasUpdate(double[] delta) {
        // Do nothing
        return this;
    }

    @Override
    public CLayerLearner incrementWeightUpdate(double[] delta) {
        // TODO
        return this;
    }

    @Override
    public CLayerLearner finalizeLearning(int batchSize, double eta) {
        return null;
    }
}
