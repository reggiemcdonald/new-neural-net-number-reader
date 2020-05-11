package com.reggiemcdonald.neural.feedforward.res;

public class NumberImage {
    public double[][] image;
    public int label;

    public NumberImage (double[][] image, int label) {
        if (label < 0 || label > 9) {
            throw new RuntimeException(
                    String.format("label must be in range [0, 9], received %d", label));
        }
        this.image = image;
        this.label = label;
    }

    public double[] getResult () {
        double[] res = new double[10];
        res[label] = 1;
        return res;
    }
}
