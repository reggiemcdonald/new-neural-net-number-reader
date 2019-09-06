package com.reggiemcdonald.neural.net;

import com.reggiemcdonald.neural.res.ImageLoader;
import com.reggiemcdonald.neural.res.NumberImage;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class NetworkTest {

    @Test
    public void testNetwork () {
        Network network = new Network (new int [] {784,30,10});
        List<NumberImage> trainingImages = ImageLoader.load (ImageLoader.TRAIN_IMAGES, ImageLoader.TRAIN_LABELS);
        network.test (trainingImages);
        network.learn (trainingImages,30,10,3.0,true);
        network.test (ImageLoader.load (ImageLoader.TEST_IMAGES, ImageLoader.TEST_LABELS));
    }

    @Test
    public void testNetworkExtended () {
        Network network = new Network(new int [] {784,100,46});
        List<NumberImage> trainingImages = ImageLoader.load (ImageLoader.EXTENDED_TRAIN_IMAGES, ImageLoader.EXTENDED_TRAIN_LABELS);
        Collections.shuffle(trainingImages);
        List<NumberImage> trainingImagesSmall = new ArrayList<>(50000);
        for (int i = 0; i < 50000; i++)
            trainingImagesSmall.add (trainingImages.get(i));
        network. learn (trainingImagesSmall, 30, 10, 3, false);
        trainingImages = ImageLoader.load (ImageLoader.EXTENDED_TEST_IMAGES, ImageLoader.EXTENDED_TRAIN_LABELS);
        network.test (trainingImages);
    }
}
