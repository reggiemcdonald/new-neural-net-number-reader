package com.reggiemcdonald.neural.convolutional.net.layer.cnn;

import com.reggiemcdonald.neural.convolutional.net.util.Matrix;

/**
 * An layer of inputs to a convolutional neural network
 * Neurons will be stored in a linear list, but accessible as if they were
 * stored in a 2-Dimensional data structure (size is a perfect square)
 */
public class InputLayer {
    private double[][][] maps;
    private int dimX, dimY, dimZ;

    public InputLayer (int dimX, int dimY, int dimZ) {
        this.maps = Matrix.zeros(dimX, dimY, dimZ);
        this.dimX = dimX;
        this.dimY = dimY;
        this.dimZ = dimZ;
    }

    /**
     * set the inputs of this input layer
     * @param input
     */
    public void set (double[][][] input) {
        this.maps = Matrix.deepCopy(input);
    }

    /**
     * Get the output
     * @return
     */
    public double[][][] outputs () {
        return this.maps;
    }

    public int dimX () { return dimX; }

    public int dimY () { return dimY; }

    public int dimZ () { return dimZ; }


}
