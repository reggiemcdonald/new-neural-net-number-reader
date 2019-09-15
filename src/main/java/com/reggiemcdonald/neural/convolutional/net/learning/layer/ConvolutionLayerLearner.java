package com.reggiemcdonald.neural.convolutional.net.learning.layer;

import com.reggiemcdonald.neural.convolutional.net.layer.CNNLayer;

public class ConvolutionLayerLearner implements CLayerLearner {
    private CNNLayer layer;

    public ConvolutionLayerLearner (CNNLayer layer) {
        this.layer = layer;
    }

    @Override
    public double[] delta(double[] deltaNextLayer) {
        // Produce a matrix the size of the convolutional dimensions
        int dimX = layer.dim_x(), dimY = layer.dim_y();
        double[] delta = new double[dimX * dimY];

        // Get the window width of the pooling layer connected to this
        CNNLayer forwardLayer = layer
                .get(0)
                .synapsesFromThis()
                .get(0)
                .to()
                .layer();
        int windowWidth = forwardLayer.window_width();
        // Set the indices of the neurons that are max to the gradient
        // And then every other index in the delta array stays 0
        int i = 0, x = 0, y = 0;
        while (x < dimX) {
            while (y < dimY) {
                for (int x_in = 0; x_in < x + windowWidth; x_in++) {
                    for (int y_in = 0; y_in < y + windowWidth; y_in++) {
                        int coord = coordinatesToIndex(x, y);
                        delta[coord] =
                                (layer.get(x_in, y_in).output() == forwardLayer.get(i).output() ? deltaNextLayer[i] : 0.);
                    }
                }
                // Update the indices
                i++;
                if (x + windowWidth < dimX) {
                    x += windowWidth;
                } else {
                    x = 0;
                    y += windowWidth;
                }
            }
        }
        return delta;

    }

    @Override
    public CLayerLearner incrementBiasUpdate(double[] delta) {
        return null;
    }

    @Override
    public CLayerLearner incrementWeightUpdate(double[] delta) {
        return null;
    }

    @Override
    public CLayerLearner finalizeLearning(int batchSize, double eta) {
        return null;
    }

    private int coordinatesToIndex (int x, int y) {
        return (x * layer.dim_x()) + y;
    }
}
