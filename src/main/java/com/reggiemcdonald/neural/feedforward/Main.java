package com.reggiemcdonald.neural.feedforward;

import com.reggiemcdonald.neural.feedforward.net.Network;
import com.reggiemcdonald.neural.feedforward.res.ImageLoader;
import com.reggiemcdonald.neural.feedforward.res.NumberImage;

import java.awt.*;
import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class Main {
    Network network;
    Scanner s;

    public Main () {
        load();
        s = new Scanner(System.in);
    }

    public void load () {
        network = Network.load("network_state.nerl");
        if (network == null) {
            network = new Network(new int[]{784, 30, 10});
            network.learn (ImageLoader.load(ImageLoader.TEST_IMAGES, ImageLoader.TEST_LABELS),30,10,3.0,false);
            network.test (ImageLoader.load(ImageLoader.TRAIN_IMAGES, ImageLoader.TRAIN_LABELS));
            Network.save (network, Network.SAVE_PATH);
        } else {
            System.out.println("Loaded network from previous state");
        }
    }

    public void run (List<NumberImage> images) {
        try {
            System.out.println("Enter any number to randomly pick an image to input into the network.");
            s.nextInt();
            NumberImage img = randomPick(images);
            System.out.println(Arrays.deepToString(img.image));
            Desktop.getDesktop().open(new File("number.png"));
            System.out.println("View the image that has been randomly selected. Enter any number to continue");
            s.nextInt();
            network.input(img.image).propagate();
            System.out.println("Network answer: " + network.result(network.output()));
            System.out.println("Actual answer: " + img.label);
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    public NumberImage randomPick (List<NumberImage> images) {
        try {
            Random r = new Random();
            NumberImage img = images.get(r.nextInt(images.size()));
            ImageLoader.saveImage(img.image, "number.png");
            return img;
        } catch (Exception e) {
            System.out.println(e.getMessage());
            throw new RuntimeException();
        }
    }

    public static void main(String[] args) {
        Main main = new Main();
        while (true) {
            main.run(ImageLoader.load(ImageLoader.TRAIN_IMAGES, ImageLoader.TRAIN_LABELS));
        }
    }
}
