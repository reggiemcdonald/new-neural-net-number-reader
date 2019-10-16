package com.reggiemcdonald.neural.convolutional.net.util;

public class DefaultInputWrapper implements InputWrapper<double[][]> {
    @Override
    public double[][][] wrapInput(double[][] rawInput) {
        double[][][] formattedInput = new double[1][rawInput.length][];
        System.arraycopy(
                rawInput,
                0,
                formattedInput[0],
                0,
                rawInput.length
        );
        return formattedInput;
    }
}
