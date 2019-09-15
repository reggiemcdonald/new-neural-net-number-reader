package com.reggiemcdonald.neural.convolutional.net.util;

/**
 * A static utility class for simple layer functions
 */
public class LayerUtilities {
    /**
     * @param oldDim the dimension of the previous layer
     * @param windowSize the size of the window for convolving/pooling
     * @return the dimension of the next layer
     */
    public static int nextDimension(int oldDim, int windowSize, int stride) {
        int x = 0, count = 0;
        while (x + windowSize <= oldDim) {
            count++;
            x += stride;
        }
        return count;
    }

    /**
     * Returns the index of the position (x,y) of a dim*dim square
     * @param x
     * @param y
     * @param dim
     * @return
     */
    public static int coordinatesToIndex (int x, int y, int dim) {
        return (x * dim) + y;
    }

    /**
     * Convert a linear array to a dim * dim square matrix
     * @param arr
     * @param dim
     * @return
     */
    public static double[][] reshapeToSquareMatrix(double[] arr, int dim) {
        double[][] mat = new double[dim][dim];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                int idx = coordinatesToIndex(i, j, dim);
                mat[i][j] = arr[idx];
            }
        }
        return mat;
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
     * Perform a convolution
     * @param big
     * @param small
     * @return
     */
    public static double[][] convolve (double[][] big, double[][] small, int stride) {
        int dim = nextDimension (big.length, small.length, stride);
        int convolveWidth = small.length;
        double[][] convolution = new double[dim][dim];

        int x = 0, y = 0, i = 0, j = 0;
        while (x < dim) {
            while (y < dim) {

                double d = 0.;
                for (int x_win = 0; x_win < convolveWidth; x_win++) {
                    for (int y_win = 0; y_win < convolveWidth; y_win++) {
                        d += (big[i + x_win][j + y_win] * small[x_win][y_win]);
                        j++;
                    }
                    i++;
                }

                convolution[i][j] = d;
                if (y + 1 < dim) {
                    y++;
                    j += stride;
                } else if (x + 1 < dim) {
                    x++;
                    y = 0;
                    i += stride;
                } else {
                    x++;
                    y++;
                }
            }
        }

        return convolution;
        // TODO
    }

    /**
     * Sum a 2D Matrix
     * @param matrix
     * @return
     */
    public static double sum (double[][] matrix) {
        double sum = 0.;
        for (int i = 0; i < matrix.length; i++)
            for (int j = 0; j < matrix[i].length; j++)
                sum += matrix[i][j];
        return sum;
    }

    /**
     * Sum a 3D matrix
     * @param matrix
     * @return
     */
    public static double sum (double[][][] matrix) {
        double sum = 0.;
        for (int i = 0; i < matrix.length; i++)
            for (int j = 0; j < matrix[i].length; j++)
                for (int k = 0; k < matrix[i][j].length; k++)
                    sum += matrix[i][j][k];
        return sum;
    }
}
