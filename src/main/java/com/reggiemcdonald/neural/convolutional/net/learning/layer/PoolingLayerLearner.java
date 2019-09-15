package com.reggiemcdonald.neural.convolutional.net.learning.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.layer.CNNLayer;
import com.reggiemcdonald.neural.convolutional.net.layer.PoolingLayer;
import com.reggiemcdonald.neural.convolutional.net.util.LayerUtilities;

public class PoolingLayerLearner implements CLayerLearner {
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
    public double[][] delta(double[][] delta_next) {

        CNNLayer previousLayer =
                layer.get(0)
                .synapsesToThis()
                .get(0)
                .from()
                .layer();

        int dimX = previousLayer.dim_x();
        int dimY = previousLayer.dim_y();

        int windowWidth = layer.window_width();
        int stride      = ((PoolingLayer)layer).stride();
        double[][] delta = new double[dimX][dimY];

        // Pad the matrix and set values at the indices of the maximums
        int count = 0, i = 0, j = 0;
        for (int x = 0; x < layer.dim_x(); x++) {

            for (int y = 0; y < layer.dim_y(); y++) {

                double gradient = delta_next[x][y];
                CNeuron neuron  = layer.get(x, y);
                for (int x_in = 0; x_in < windowWidth; x_in++) {
                    boolean found = false;
                    for (int y_in = 0; y_in < windowWidth; y_in++) {

                        int idx = LayerUtilities.coordinatesToIndex(x_in, y_in, windowWidth);
                        double output = neuron.synapsesToThis().get(idx).from().output();
                        if (output == neuron.output()) {
                            delta[i + x_in][j + y_in] = gradient;
                            System.out.println(i + " " + j);
                            found = true;
                            count++;
                            break;
                        }
                    }
                    if (found) {
                        if (j + stride < dimX) {
                            j += stride;
                        } else {
                            i += stride;
                            j = 0;
                        }
                        break;
                    }
                }
            }
        }
        return delta;
    }

    @Override
    public CLayerLearner incrementBiasUpdate(double[][] delta) {
        // Do nothing
        return this;
    }

    @Override
    public CLayerLearner incrementWeightUpdate(double[][] delta) {
        // TODO
        return this;
    }

    @Override
    public CLayerLearner finalizeLearning(int batchSize, double eta) {
        return null;
    }
}
