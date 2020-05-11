package com.reggiemcdonald.neural.feedforward.net;

import com.reggiemcdonald.neural.feedforward.res.ImageLoader;
import com.reggiemcdonald.neural.feedforward.res.NumberImage;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class NetworkTest {

    @Test
    public void testNetwork() {
        Network network = new Network (new int[] {784,30,10});
        List<NumberImage> trainingImages = ImageLoader.load(ImageLoader.TRAIN_IMAGES, ImageLoader.TRAIN_LABELS);
        List<NumberImage> testingImages = ImageLoader.load(ImageLoader.TEST_IMAGES, ImageLoader.TEST_LABELS);
        network.test(trainingImages);
        network.learn(trainingImages,5,10,3.0,false);
        int numberCorrect = network.test(ImageLoader.load(ImageLoader.TEST_IMAGES, ImageLoader.TEST_LABELS));
        double proportion = (double) numberCorrect / testingImages.size();
        assertTrue(proportion > 0.93);
    }
}
