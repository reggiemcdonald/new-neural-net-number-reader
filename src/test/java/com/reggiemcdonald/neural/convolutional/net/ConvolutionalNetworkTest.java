package com.reggiemcdonald.neural.convolutional.net;

import com.reggiemcdonald.neural.convolutional.net.util.InputWrapper;
import com.reggiemcdonald.neural.feedforward.res.ImageLoader;
import com.reggiemcdonald.neural.feedforward.res.NumberImage;
import org.junit.Test;

import java.util.List;

public class ConvolutionalNetworkTest {

    InputWrapper<double[][]> myWrapper = (double[][] originalInput) -> {

        double[][][] newInput = new double[1][originalInput.length][originalInput[0].length];
        for (int i = 0; i < originalInput.length; i++)
            for (int j = 0; j < originalInput[i].length; j++)
                newInput[0][i][j] = originalInput[i][j];
        return newInput;

    };

    @Test
    public void testNetwork () {
        ConvolutionalNetwork cnn = new ConvolutionalNetwork(28, new int[] {5}, new int[] {2},
                new int[] {3,3}, new int[] {},10, new int[] {1,2}, true);
        List<NumberImage> numberImages = ImageLoader.load(ImageLoader.TEST_IMAGES, ImageLoader.TEST_LABELS);
        cnn.input(myWrapper.wrapInput(numberImages.get(0).image));
        System.out.println("Done");
    }


}
