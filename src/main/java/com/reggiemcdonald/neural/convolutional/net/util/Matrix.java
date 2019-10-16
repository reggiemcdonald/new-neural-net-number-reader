package com.reggiemcdonald.neural.convolutional.net.util;

import com.reggiemcdonald.neural.convolutional.exception.MatrixException;

import java.util.Arrays;
import java.util.Random;

/**
 * A simple matrix class
 */
public class Matrix {

    /**
     * An interface for lambda expression syntax
     */
    private interface Operator {
        double operate (double a, double b);
    }

    /**
     * Produce a 2D x * y matrix
     * @param x
     * @param y
     * @return
     */
    public static double[][] zeros (int x, int y) {
        return new double[x][y];
    }

    /**
     * Produce a 3D x * y * z matrix
     * Note that for this notation, we will have z x*y 2D square matrices
     * @param x
     * @param y
     * @param z
     * @return
     */
    public static double[][][] zeros (int x, int y, int z) {
        return new double[z][x][y];
    }

    /**
     * Generate a gaussian x * y random matrix
     * @param x
     * @param y
     * @return
     */
    public static double[][] gaussianMatrix (int x, int y) {
        Random r = new Random();
        double[][] d = new double[x][y];
        for (int i = 0; i < d.length; i++)
            for (int j = 0; j < d[i].length; j++)
                d[i][j] = r.nextGaussian();

        return d;
    }

    /**
     * Generate a z * x * y random matrix with all elements sampled from a normal gaussian distribution
     * @param x
     * @param y
     * @param z
     * @return
     */
    public static double[][][] gaussianMatrix (int x, int y, int z) {
        Random r = new Random();
        double[][][] d = new double[z][x][y];
        for (int i = 0; i < d.length; i++)
            for (int j = 0; j < d[i].length; j++)
                for (int k = 0; k < d[i][j].length; k++)
                    d[i][j][k] = r.nextGaussian();

        return d;
    }

    /**
     * Perform a valid convolution
     * @param map
     * @param kernel
     * @param stride
     * @return
     */
    public static double[][] validConvolution(double[][] map, double[][] kernel, int stride) {
        int mapX = map.length;
        int mapY = map[0].length;

        int kX   = LayerUtilities.nextDimension(mapX, kernel.length, stride);
        int kY   = LayerUtilities.nextDimension(mapY, kernel[0].length, stride);

        double[][] output = new double[kX][kY];

        for (int i = 0; i < kX; i++) {

            for (int j = 0; j < kY; j++) {

                double d = 0.;

                for (int x = 0; x < kernel.length; x++) {

                    for (int y = 0; y < kernel[x].length; y++) {

                        d += (map[(x * stride) + i][(y * stride) + j] * kernel[x][y]);

                    }
                }

                output[i][j] = d;
            }
        }
        return output;
    }

    /**
     * Perform a valid convolution with 3-dimensional input
     * @param maps
     * @param kernels
     * @param stride
     * @return
     */
    public static double[][] validConvolution(double[][][] maps, double[][][] kernels, int stride) {

        int kX = LayerUtilities.nextDimension(maps[0].length, kernels[0].length, stride);
        int kY = LayerUtilities.nextDimension(maps[0][0].length, kernels[0][0].length, stride);

        double[][] out = new double[kX][kY];

        for (double[][] map : maps) {

            for (double[][] kernel : kernels) {

                sum (out, validConvolution(map, kernel, stride), out);
            }
        }
        return out;
    }

    /**
     * Sum two matrices
     * For in-place operation: out == one || out == two
     * @param one
     * @param two
     * @param out
     * @return
     */
    public static double[][] sum (double[][] one, double[][] two, double[][] out) {
        if (!sameDimensions (one, two))
            throw new MatrixException("Fatal: Dimension mismatch");

        for (int i = 0; i < one.length; i++) {

            for (int j = 0; j < one[0].length; j++) {

                out[i][j] = one[i][j] + two[i][j];
            }
        }
        return out;
    }


    /**
     * True if the given matrices have the same dimensions
     * @param one
     * @param two
     * @return
     */
    public static boolean sameDimensions (double[][] one, double[][] two) {
        return (one.length == two.length) && (one[0].length == two[0].length);
    }

    /**
     * Perform an in-place matrix rotation
     * @param mat
     * @return
     */
    public static double[][] rotate180 (double[][] mat) {
        int dimX = mat.length, dimY = mat[0].length;
        for (int i = 0; i < dimX / 2; i++) {
            for (int j = 0; j < dimY; j++) {

                double swap = mat[i][j];
                mat[i][j] = mat[dimX - i - 1][dimY - j - 1];
                mat[dimX - i - 1][dimY - j - 1] = swap;

            }

        }

        if (dimX % 2 == 0) {
            for (int i = 0; i < dimY / 2; i++) {
                double swap = mat[dimX / 2][i];
                mat[dimX / 2][i] = mat[dimX / 2][dimY - i - 1];
                mat[dimX / 2][dimY - i - 1] = swap;
            }
        }
        return mat;
    }

    /**
     * Perform a deep array copy
     * @param toCopy
     * @return
     */
    public static double[][][] deepCopy (double[][][] toCopy ) {
        double[][][] out = new double[toCopy.length][toCopy[0].length][toCopy[0][0].length];

        for (int i = 0; i < out.length; i++) {

            for (int j = 0; j < out[i].length; j++) {

                for (int k = 0; k < out[i][j].length; k++) {

                    out[i][j][k] = toCopy[i][j][k];
                }
            }
        }
        return out;
    }


    private static double[][] elementWiseScalarOperation(double[][] input, double d, Operator opp) {

        for (int i = 0; i < input.length; i++ ) {

            for (int j = 0; j < input[0].length; j++) {

                input[i][j] = opp.operate(input[i][j], d);
            }
        }
        return input;
    }

    /**
     * Add to each element in the matrix, the value d
     * @param input
     * @param d
     * @return
     */
    public static double[][] elementWiseAdd (double[][] input, double d) {
        return elementWiseScalarOperation(input, d, (double a, double b) -> (a + b));
    }

    /**
     * Divide each element in input by d
     * @param input
     * @param d
     * @return
     */
    public static double[][] elementWiseDivide (double[][] input, double d) {
        return elementWiseScalarOperation(input, d, (double a, double b) -> (a / b));
    }

    /**
     * Multiply each element of input by d
     * @param input
     * @param d
     * @return
     */
    public static double[][] elementWiseMultiply (double[][] input, double d) {
        return elementWiseScalarOperation(input, d, (double a, double b) -> (a * b));
    }

    /**
     * Subtract d from each element of input
     * @param input
     * @param d
     * @return
     */
    public static double[][] elementWiseSubtract (double[][] input, double d) {
        return elementWiseScalarOperation(input, d, (double a, double b) -> (a - b));
    }

    /**
     * Resize a matrix to a linear array
     * @param matrix
     * @return
     */
    public static double[] toArray (double[][] matrix) {
        double[] d = new double[matrix.length * matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {

            for (int j = 0; j < matrix[i].length; j++) {

                int idx = ((i * matrix.length) + j);
                d[idx] = matrix[i][j];
            }
        }
        return d;
    }

    public static double[] toArray (double[][][] matrix) {
        int squareArea = matrix[0].length * matrix[0][0].length;
        double[] d = new double[matrix.length * matrix[0].length * matrix[0][0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                for (int k = 0; k < matrix[i][j].length; k++) {
                    int idx = ((i * squareArea) + (j * matrix[0].length) + k);
                    d[idx] = matrix[i][j][k];
                }
            }
        }
        return d;
    }

    /**
     * Convert an array to a dimX * dimY matrix
     * @param arr
     * @param dimX
     * @param dimY
     * @return
     */
    public static double[][] arrayToMatrix (double[] arr, int dimX, int dimY) {
        // TODO: Error handling
        double[][] matrix = new double[dimX][dimY];
        for (int i = 0; i < dimX; i++) {

            for (int j = 0; j < dimY; j++) {
                int idx = (i * dimX) + j;
                matrix[i][j] = arr[idx];
            }
        }
        return matrix;
    }

    /**
     * An alias to Arrays.deepToString
     * @param matrix
     * @return
     */
    public static String toString (double[][] matrix) {
        StringBuilder sb = new StringBuilder();
        for (double[] m : matrix) {
            sb.append(Arrays.toString(m));
            sb.append("\n");
        }
        return sb.toString();
    }

    /**
     * An alias to Arrays.deepToString
     * @param matrix
     * @return
     */
    public static String toString (double[][][] matrix) {
        StringBuilder sb = new StringBuilder();
        for (double[][] d : matrix)
            sb.append(toString(d));
        return sb.toString();
    }
}
