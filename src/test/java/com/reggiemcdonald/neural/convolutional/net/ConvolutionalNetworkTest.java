package com.reggiemcdonald.neural.convolutional.net;

import org.junit.Test;

public class ConvolutionalNetworkTest {

    @Test
    public void testNetwork () {
        int[] inputs = {28, 28, 28}, conv = {5, 5, 5}, pooling = {5};
        ConvolutionalNetwork cnn = new ConvolutionalNetwork(inputs, conv, pooling, 10, 1, true);
    }


}
