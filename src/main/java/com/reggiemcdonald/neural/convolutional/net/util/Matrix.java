package com.reggiemcdonald.neural.convolutional.net.util;

import java.util.Random;

/**
 * A simple matrix class
 */
public class Matrix {

    public static double[][] zeros (int x, int y) {
        return new double[x][y];
    }

    public static double[][][] zeros (int x, int y, int z) {
        return new double[x][y][z];
    }

    public static double[][] gaussianMatrix (int x, int y) {
        Random r = new Random();
        double[][] d = new double[x][y];
        for (int i = 0; i < d.length; i++)
            for (int j = 0; j < d[i].length; j++)
                d[i][j] = r.nextGaussian();

        return d;
    }

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
    public static double[][] validConvolve (double[][] map, double[][] kernel, int stride) {
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
    public static double[][] validConvolve (double[][][] maps, double[][][] kernels, int stride) {

        int kX = LayerUtilities.nextDimension(maps[0].length, kernels[0].length, stride);
        int kY = LayerUtilities.nextDimension(maps[0][0].length, kernels[0][0].length, stride);

        double[][] out = new double[kX][kY];

        for (double[][] map : maps) {

            for (double[][] kernel : kernels) {

                sum (out, validConvolve(map, kernel, stride), out);
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
            throw new RuntimeException("Fatal: Dimension mismatch");

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
        if (dimX % 2 == 1) {
            for (int i = 0; i < dimY / 2; i++) {
                double swap = mat[dimX / 2][i];
                mat[dimX / 2][i] = mat[dimX / 2][dimY - i - 1];
                mat[dimX][dimY - i - 1] = swap;
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

    /**
     * Add to each element in the matrix, the value d
     * @param input
     * @param d
     * @return
     */
    public static double[][] elementWiseAdd (double[][] input, double d) {
        for (int i = 0; i < input.length; i++ ) {

            for (int j = 0; j < input[0].length; j++) {

                input[i][j] += d;
            }
        }
        return input;
    }

    public static double[][] elementWiseDivide (double[][] input, double d) {
        for (int i = 0; i < input.length; i++ ) {

            for (int j = 0; j < input[0].length; j++) {

                input[i][j] /= d;
            }
        }
        return input;
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
}
