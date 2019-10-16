package com.reggiemcdonald.neural.feedforward.res;

public class NumberImage {
    public double[][] image;
    public int label;

    public NumberImage (double[][] image, int label) {
        this.image = image;
        this.label = label;
    }

    public double[] getResult () {
        double[] res = new double[10];
        res[label] = 1;
        return res;
    }
}
