package com.reggiemcdonald.neural.convolutional.net.learning.layer.cnn;

import com.reggiemcdonald.neural.convolutional.net.layer.cnn.CNNLayer;
import com.reggiemcdonald.neural.convolutional.net.util.Matrix;

public class PoolingLayerLearner implements CNNLayerLearner {
    private CNNLayer layer;

    public PoolingLayerLearner (CNNLayer layer) {
        this.layer = layer;
    }

    /**
     * For a max pooling layer, we compute a delta which is a padded matrix
     * To pad, find the indices of the max nodes in the earlier layer, and set the
     * elements of the detla maxtrix to be the gradient concerned
     * @param delta_next
     * @return a matrix (put flat for interface consistency)
     */
    @Override
    public double[][] delta(CNNLayer previousLayer, double[][] delta_next) {

        int dimX             = layer.dimX();
        int dimY             = layer.dimY();
        int prevLayerDimX    = previousLayer.dimX();
        int prevLayerDimY    = previousLayer.dimY();

        int windowWidth      = layer.windowWidth();
        int stride           = layer.stride();

        double[][] delta     = Matrix.zeros(prevLayerDimX, prevLayerDimY);
        double[][] thisMap   = layer.outputs();
        double[][] backMap   = previousLayer.outputs();

        for (int i = 0; i < dimX; i+=stride) {

            for (int j = 0; j < dimY; j+=stride) {
                double output = thisMap[i][j];
                boolean found = false;

                for (int x = i; x < i + windowWidth; x++) {

                    for (int y = j; y < j + windowWidth; y++) {

                        if (backMap[x][y] == output) {
                            found = true;
                            delta[x][y] = delta_next[i][j];
                            break;
                        }
                    }
                    if (found) break;
                }

            }
        }


        return delta;
    }

    @Override
    public void incrementBiasUpdates(double[][] delta) {
        // Nothing to update for this layer
    }

    @Override
    public void incrementWeightUpdates(double[][] delta) {
        // Nothing to update for this layer
    }

    @Override
    public void finalizeLearning(int batchSize, double eta) {
        // TODO
    }
}
