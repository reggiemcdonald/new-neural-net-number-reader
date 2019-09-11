package com.reggiemcdonald.neural.convolutional.net.decipher;

public interface Decipher<T> {

    /**
     * Decodes a an output array
     * @param output
     * @return
     */
    T decode (double[] output);
}
