package com.reggiemcdonald.neural.res;

public class NumberImage {
    public double[][] image;
    public double[] label;

    public NumberImage (double[][] image, int label) {
        this.image = image;
        this.label = new double[10];
        this.label [label] = 1;
    }
}
