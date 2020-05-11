package com.reggiemcdonald.neural.feedforward.res;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class NumberImageTest {

    private NumberImage numberImage;

    @BeforeEach
    public void runBefore() {
        numberImage = new NumberImage(new double[728][728], 0);
    }

    private double[] expected(int idx) {
        double[] arr = new double[10];
        arr[idx] = 1;
        return arr;
    }

    @Test
    public void testInvalidLabel() {
        assertThrows(RuntimeException.class, () -> {
            new NumberImage(new double[728][728], 10);
        });
        assertThrows(RuntimeException.class, () -> {
            new NumberImage(new double[728][728], -1);
        });
    }

    @Test
    public void testGetResult() {
        numberImage.label = 0;
        double[] expected = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        assertArrayEquals(expected, numberImage.getResult());

        numberImage.label = 5;
        expected = new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
        assertArrayEquals(expected, numberImage.getResult());

        numberImage.label = 9;
        expected = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
        assertArrayEquals(expected, numberImage.getResult());
    }
}
