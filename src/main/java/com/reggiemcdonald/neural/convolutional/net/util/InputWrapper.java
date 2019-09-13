package com.reggiemcdonald.neural.convolutional.net.util;

public interface InputWrapper<T> {
    double[][][] wrapInput(T rawInput);
}
