package com.reggiemcdonald.neural.convolutional.net.util;

import com.reggiemcdonald.neural.convolutional.exception.MatrixException;
import org.junit.Test;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;

public class MatrixTest {
    private double[][] map = {
            {1, 0, 0.5, 0.5},
            {0, 0.5, 1, 0  },
            {0, 1, 0.5, 1  },
            {1, 0.5, 0.5, 1}
    };
    private double[][] kernel = {
            {1, 0  },
            {0, 0.5},
    };
    private double[][] expectedConvolution = {
            {1.25, 0.5, 0.5},
            {0.5, 0.75, 1.5},
            {0.25, 1.25, 1 }
    };

    @Test
    public void testConvolution2D () {

        double[][] answer = Matrix.validConvolution(map, kernel, 1);
        System.out.println(Matrix.toString(answer));
        assertArrayEquals (answer, expectedConvolution);

    }

    @Test
    public void testConvolution3D () {
        double[][][] map    = { this.map, this.map, this.map };
        double[][][] kernel = { this.kernel, this.kernel, this.kernel};
        double[][]   answer = Matrix.validConvolution(map, kernel, 1);
        double[][]   expected = Matrix.elementWiseMultiply(answer, 3);
        assertArrayEquals(expected, answer);
    }

    @Test
    public void testElementWiseAdd () {
        double[][] test = {
                { 1,  0,  1 },
                { 0,  1,  0,},
                {-1, -1, -1,}
        };
        double[][] expected = {
                {2, 1, 2},
                {1, 2, 1},
                {0, 0, 0}
        };
        double[][] answer = Matrix.elementWiseAdd(test, 1);
        assertArrayEquals(answer, expected);
    }

    @Test
    public void testElementWiseSubtract () {
        double[][] test = {
                { 1,  0,  1 },
                { 0,  1,  0,},
                {-1, -1, -1,}
        };
        double[][] expected = {
                { 0, -1,  0},
                {-1,  0, -1},
                {-2, -2, -2}
        };
        double[][] answer = Matrix.elementWiseSubtract(test, 1);
        assertArrayEquals(answer, expected);
    }

    @Test
    public void testElementWiseMultiply () {
        double[][] test = {
                { 1,  0,  1 },
                { 0,  1,  0,},
                {-1, -1, -1,}
        };
        double[][] expected = {
                { 2,  0,  2},
                { 0,  2,  0},
                {-2, -2, -2}
        };
        double[][] answer = Matrix.elementWiseMultiply(test,2);
        assertArrayEquals(answer, expected);
    }

    @Test
    public void testElementWiseDivide () {
        double[][] test = {
                { 2,  2,  2 },
                { 4,  4,  4,},
                { 6,  6,  6,}
        };
        double[][] expected = {
                { 1,  1,  1 },
                { 2,  2,  2,},
                { 3,  3,  3,}
        };

        double[][] answer = Matrix.elementWiseDivide(test, 2);
        assertArrayEquals(answer, expected);
    }

    @Test
    public void testMatrixRotation () {
        double[][] test = {
                { 2,  2,  2 },
                { 4,  4,  4,},
                { 6,  6,  6,}
        };
        double[][] expected = {
                {6, 6, 6},
                {4, 4, 4},
                {2, 2, 2},
        };

        double[][] answer = Matrix.rotate180(test);
        System.out.println(Matrix.toString(Matrix.rotate180(this.map)));
        assertArrayEquals(answer, expected);
    }

    @Test
    public void testSameDimensions () {
        double[][] test = {
                { 2,  2,  2 },
                { 4,  4,  4,},
                { 6,  6,  6,}
        };
        double[][] expected = {
                {6, 6, 6},
                {4, 4, 4},
                {2, 2, 2},
        };
        assertTrue(Matrix.sameDimensions(test, expected));
        assertFalse(Matrix.sameDimensions(test, this.map));
    }

    @Test
    public void testSum () {
        double[][] test = {
                { 2,  2,  2 },
                { 4,  4,  4,},
                { 6,  6,  6,}
        };
        double[][] test2 = {
                {6, 6, 6},
                {4, 4, 4},
                {2, 2, 2},
        };
        double[][] expected = {
                {8, 8, 8},
                {8, 8, 8},
                {8, 8, 8},
        };
        double[][] answer = new double[3][3];
        Matrix.sum (test, test2, answer);
        assertArrayEquals(answer, expected);
    }

    @Test (expected = MatrixException.class)
    public void testSumIllegal () {
        double[][] test = {
                { 2,  2,  2 },
                { 4,  4,  4,},
                { 6,  6,  6,}
        };
        Matrix.sum (test, this.map, test);
    }


}
