package com.reggiemcdonald.neural;

import com.reggiemcdonald.neural.net.Network;
import com.reggiemcdonald.neural.res.ImageLoader;
import com.reggiemcdonald.neural.res.NumberImage;

import java.awt.*;
import java.io.File;
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
            Desktop.getDesktop().open(new File("saved.bmp"));
            System.out.println("View the image that has been randomly selected. Enter any number to continue");
            s.nextInt();
            network.input(img.image).propagate();
            System.out.println("Network answer: " + network.result(network.output()));
            System.out.println("Actual answer " + network.result(img.label));
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    public NumberImage randomPick (List<NumberImage> images) {
        try {
            Random r = new Random();
            NumberImage img = images.get(r.nextInt(images.size()));
            ImageLoader.saveImage(img.image);
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
