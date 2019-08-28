package com.reggiemcdonald.neural.res;

public class NumberImage {
    public double[][] image;
    public int[]      label;

    public NumberImage (double[][] image, int label) {
        this.image = image;
        this.label = new int[10];
        this.label [label] = 1;
    }
}
