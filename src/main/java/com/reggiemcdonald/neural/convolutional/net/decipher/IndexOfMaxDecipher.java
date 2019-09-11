package com.reggiemcdonald.neural.convolutional.net.decipher;

/**
 * The default decipher behaviour -
 * find the most strongly outputting neuron in the output layer
 */
public class IndexOfMaxDecipher implements Decipher<Integer> {

    @Override
    public Integer decode(double[] output) {
        if (output == null || output.length == 0)
            throw new RuntimeException("Decode given ambiguous input");
        int idx = 0;
        double max = output[0];
        for (int i = 1; i < output.length; i++) {
            if (output[i] > max) {
                idx = i;
                max = output[i];
            }
        }
        return idx;
    }
}
