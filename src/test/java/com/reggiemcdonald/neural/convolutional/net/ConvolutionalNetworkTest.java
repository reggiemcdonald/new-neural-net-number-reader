package com.reggiemcdonald.neural.convolutional.net;

import org.junit.Test;

public class ConvolutionalNetworkTest {

    @Test
    public void testNetwork () {
        ConvolutionalNetwork cnn = new ConvolutionalNetwork(28, new int[] {5}, new int[] {2},
                new int[] {3,3}, 10, new int[] {1,2}, true);
        System.out.println("Done");
    }


}
