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
    public void incrementWeightUpdates (double[][] delta) {
        // TODO
    }

    @Override
    public void finalizeLearning(int batchSize, double eta) {
        // TODO
    }

    @Override
    public void incrementBiasUpdates (double[][] delta) {
        // TODO
    }


    // TODO
}
