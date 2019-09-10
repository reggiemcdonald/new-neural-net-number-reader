package com.reggiemcdonald.neural.convolutional.net.util;

/**
 * A static utility class for simple layer functions
 */
public class LayerUtilities {
    /**
     * @param oldDim the dimension of the previous layer
     * @param windowSize the size of the window for convolving/pooling
     * @return the dimension of the next layer
     */
    public static int nextDimension(int oldDim, int windowSize, int stride) {
        int x = 0, count = 0;
        while (x + windowSize <= oldDim) {
            count++;
            x += stride;
        }
        return count;
    }
}
