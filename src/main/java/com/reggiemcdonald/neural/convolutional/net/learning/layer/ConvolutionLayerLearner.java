package com.reggiemcdonald.neural.convolutional.net.learning.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.layer.fc.FullyConnectedLayer;
import com.reggiemcdonald.neural.convolutional.net.layer.cnn.ConvolutionLayer;
import com.reggiemcdonald.neural.convolutional.net.util.LayerUtilities;

public class ConvolutionLayerLearner implements FullyConnectedLayerLearner {
    private FullyConnectedLayer layer;
    private double[][] weightUpdates;

    public ConvolutionLayerLearner (FullyConnectedLayer layer) {
        this.layer = layer;
    }

    @Override
    public double[][] delta(double[][] deltaNextLayer) {
        return LayerUtilities.convolve(inputActivations(), deltaNextLayer, ((ConvolutionLayer)layer).stride());
    }

    @Override
    public FullyConnectedLayerLearner incrementBiasUpdate(double[][] delta) {
        double biasUpdate = LayerUtilities.sum(delta);
        for (CNeuron neuron : layer)
            neuron.learner().incrementBiasUpdate(biasUpdate);
        return this;
    }

    @Override
    public FullyConnectedLayerLearner incrementWeightUpdate(double[][] delta) {
        // When a weight is shared across connections,
        // the gradient update for that weight is the sum of the gradients
        // across all the connections that share the same weight

        // Build the array to hold the weight updates
        int weightUpdateLength = layer.window_width() * layer.window_width();
        if (weightUpdates == null || weightUpdates.length != weightUpdateLength)
            weightUpdates = new double[weightUpdateLength][weightUpdateLength];
        // TODO STUB

        for (int i = 0; i < weightUpdateLength; i++)
            for (int j = 0; j < weightUpdateLength; j++)
                weightUpdates[i][j] += delta[i][j];




        return this;
    }

    @Override
    public FullyConnectedLayerLearner finalizeLearning(int batchSize, double eta) {
        for (CNeuron neuron : layer) {
            neuron.learner().applyBiasUpdate(batchSize, eta);
            neuron.learner().setWeightUpdates(weightUpdates).applyWeightUpdate(batchSize, eta);
        }

        return this;
    }

    private double[][] inputActivations() {
        FullyConnectedLayer earlylayer = layer
                .get(0)
                .synapsesToThis()
                .get(0)
                .from()
                .layer();
        int x = earlylayer.dim_x();
        int y = earlylayer.dim_y();
        double[][] d = new double[x][y];
        for (int i = 0; i < d.length; i++)
            for (int j = 0; j < y; j++)
                d[i][j] = earlylayer.get(i,j).output();
        return d;
    }

}
