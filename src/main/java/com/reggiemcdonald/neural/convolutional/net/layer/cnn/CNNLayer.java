package com.reggiemcdonald.neural.convolutional.net.layer.cnn;

import com.reggiemcdonald.neural.convolutional.net.learning.layer.cnn.CNNLayerLearner;

import java.io.Serializable;

public interface CNNLayer extends Serializable {

    /**
     * Return the map value at (x,y) at the zth depth
     * z <= depth()
     * @param x
     * @param y
     * @return
     */
    double get(int x, int y, int z);

    /**
     * @return the depth of the layer
     */
    int depth();

    /**
     * @return the dimension of the layer along the x-axis
     */
    int dimX();

    /**
     * @return the dimension of the layer along the y-axis
     */
    int dimY();

    /**
     * @return the dimension of the layer along the z-axis
     * alias to depth()
     */
    int dimZ();

    /**
     * @return the stride used for propagation in this layer
     */
    int stride();

    /**
     * @return the kernel for this layer
     */
    double[][][] kernel();

    /**
     * @return the outputs of this layer
     */
    double[][] outputs();

    /**
     * @return the learner of this
     */
    CNNLayerLearner learner();

}
