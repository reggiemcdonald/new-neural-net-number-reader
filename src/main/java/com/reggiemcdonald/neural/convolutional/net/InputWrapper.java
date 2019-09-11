package com.reggiemcdonald.neural.convolutional.net;

public interface InputWrapper<T> {
    double[][][] input(T rawInput);
}
