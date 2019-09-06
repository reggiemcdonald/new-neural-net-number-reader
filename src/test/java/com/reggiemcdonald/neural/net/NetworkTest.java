package com.reggiemcdonald.neural.net;

import com.reggiemcdonald.neural.res.ImageLoader;
import com.reggiemcdonald.neural.res.NumberImage;
import org.junit.Test;

import java.util.List;

public class NetworkTest {

    @Test
    public void testNetwork () {
        Network network = new Network (new int [] {784,100,10});
        List<NumberImage> trainingImages = ImageLoader.load (ImageLoader.TRAIN_IMAGES, ImageLoader.TRAIN_LABELS);
        network.test (trainingImages);
        network.learn (trainingImages,30,10,3.0,false);
        network.test (ImageLoader.load (ImageLoader.TEST_IMAGES, ImageLoader.TEST_LABELS));
    }

    @Test
    public void testNetworkExtended () {
        Network network = new Network(new int [] {784,100,100,46});
        List<NumberImage> trainingImages = ImageLoader.load (ImageLoader.EXTENDED_TRAIN_IMAGES, ImageLoader.EXTENDED_TRAIN_LABELS);
        // The Extended-MNIST dataset is too large for my computer I think :(
        network. learn (trainingImages, 30, 10, 3, false);
        trainingImages = ImageLoader.load (ImageLoader.EXTENDED_TEST_IMAGES, ImageLoader.EXTENDED_TRAIN_LABELS);
        network.test (trainingImages);
    }
}
