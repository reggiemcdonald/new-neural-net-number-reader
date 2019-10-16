package com.reggiemcdonald.neural.convolutional.net.learning.layer.cnn;

import com.reggiemcdonald.neural.convolutional.net.layer.cnn.CNNLayer;
import com.reggiemcdonald.neural.convolutional.net.layer.cnn.ConvolutionLayer;
import com.reggiemcdonald.neural.convolutional.net.util.Matrix;

public class ConvolutionLayerLearner implements CNNLayerLearner {
    private ConvolutionLayer layer;

    public ConvolutionLayerLearner (ConvolutionLayer layer) {
        this.layer = layer;
    }

    @Override
    public double[][] delta(CNNLayer layerBefore, double[][] deltaNextLayer) {
        return Matrix.zeros(0,0);
        // TODO Stub
    }

    @Override
    public CNNLayerLearner incrementWeightUpdates (double[][] delta) {
        // TODO
        return this;
    }

    @Override
    public CNNLayerLearner finalizeLearning(int batchSize, double eta) {
        // TODO
        return this;
    }

    @Override
    public CNNLayerLearner incrementBiasUpdates (double[][] delta) {
        // TODO
        return this;
    }


    // TODO
}
