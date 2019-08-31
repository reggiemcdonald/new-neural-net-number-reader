package com.reggiemcdonald.neural.net;

import com.reggiemcdonald.neural.res.ImageLoader;
import com.reggiemcdonald.neural.res.NumberImage;
import org.junit.Test;

import java.util.List;

public class NetworkTest {

    @Test
    public void testNetwork () {
        Network network = new Network (new int [] {784,30,10});
        List<NumberImage> trainingImages = ImageLoader.load (ImageLoader.TEST_IMAGES, ImageLoader.TEST_LABELS);
        network.test (trainingImages);
        network.learn (trainingImages,30,10,3.0,true);
        network.test (ImageLoader.load (ImageLoader.TRAIN_IMAGES, ImageLoader.TRAIN_LABELS));
    }
}
