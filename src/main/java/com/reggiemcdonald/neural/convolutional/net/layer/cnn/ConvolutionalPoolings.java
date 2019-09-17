package com.reggiemcdonald.neural.convolutional.net.layer.cnn;

import com.reggiemcdonald.neural.convolutional.net.util.Matrix;

import java.util.ArrayList;
import java.util.List;

/**
 * A full layer of a neural network containing
 * 1. An d deep convolutional layer
 * 2. A d-deep pooling layer
 * Maintains convolutional and pooling layers in their respective positions
 */
public class ConvolutionalPoolings {
    private int cDim, pDim, depth, cWindowWidth, pWindowWidth, cStride, pStride, kernelDepth;
    private boolean isMaxPooling = true; // Default. False is average pooling
    private List<ConvolutionLayer> convolutionLayers;
    private List<PoolingLayer> poolingLayers;

    public ConvolutionalPoolings(int cDim, int pDim, int layerDepth, int kernelDepth, int cWindowWidth, int pWindowWidth, int cStride, int pStride) {
        this.cDim              = cDim;
        this.pDim              = pDim;
        this.depth             = layerDepth;
        this.cWindowWidth      = cWindowWidth;
        this.pWindowWidth      = pWindowWidth;
        this.cStride           = cStride;
        this.pStride           = pStride;
        this.convolutionLayers = new ArrayList<>(layerDepth);
        this.poolingLayers     = new ArrayList<>(layerDepth);
        this.kernelDepth       = kernelDepth;
        makeLayers ();
    }

    public ConvolutionalPoolings(int cDim, int pDim, int layerDepth, int kernelDepth, int cWindowWidth, int pWindowWidth, int cStride, int pStride, boolean isMaxPooling) {
        this.cDim              = cDim;
        this.pDim              = pDim;
        this.depth             = layerDepth;
        this.cWindowWidth      = cWindowWidth;
        this.pWindowWidth      = pWindowWidth;
        this.cStride           = cStride;
        this.pStride           = pStride;
        this.isMaxPooling      = isMaxPooling;
        this.convolutionLayers = new ArrayList<>(layerDepth);
        this.poolingLayers     = new ArrayList<>(layerDepth);
        this.kernelDepth       = kernelDepth;
        makeLayers ();
    }

    /**
     * Build all the layers of this ConvolutionalPoolings
     */
    private void makeLayers() {
        for (int i = 0; i < depth; i++) {
            convolutionLayers.add (new ConvolutionLayer(cDim, cDim, cWindowWidth, Matrix.gaussianMatrix(cDim, cDim, kernelDepth), cStride));
            poolingLayers.add (new PoolingLayer (pDim, pDim, pWindowWidth, pStride,true));
        }
    }

    /**
     * @return the dimension of the convolutional layers of this
     */
    public int cDim() {
        return cDim;
    }

    /**
     * @return the dimension of the pooling layers of this
     */
    public int pDim() {
        return pDim;
    }

    /**
     * @return the list of convolutional layers in the network
     */
    public List<ConvolutionLayer> convolutionalLayers() {
        return new ArrayList<>(convolutionLayers);
    }

    /**
     * @return the list of pooling layers in the network
     */
    public List<PoolingLayer> poolingLayers () {
        return new ArrayList<>(poolingLayers);
    }

    /**
     * Returns the total number of outputting pooling neurons in this layer
     * @return
     */
    public int poolingDensity() {
        return pDim * pDim * depth;
    }

    /**
     * Flattens the pooling layers, returning a linear sigmoidal layer
     * @return SigmoidalLayer
     */
    public double[] flatten () {
        double[][][] maps = new double[poolingLayers.size()][][];
        int i = 0;
        for (PoolingLayer p : poolingLayers) {
            maps[i] = p.outputs();
            i++;
        }
        return Matrix.toArray (maps);
    }

    /**
     * Get the pooled feature maps from this Convolutional Pooling layer
     * @return
     */
    public double[][][] outputs() {
        double[][][] out = new double[poolingLayers.size()][][];
        for (int i = 0 ; i < poolingLayers.size(); i++) {
            out[i] = poolingLayers.get(i).outputs();
        }
        return out;
    }

    /**
     * Propagate through the convolutional layers
     * @param input
     */
    public void propagate (double[][][] input) {
        for (int i = 0; i < convolutionLayers.size(); i++) {
            ConvolutionLayer c = convolutionLayers.get(i);
            PoolingLayer p = poolingLayers.get(i);
            c.propagate(input);
            p.propagate (c.outputs());
        }
    }

    /**
     * Run the backpropagation algorithm through the convolutional layers
     * @param delta
     * @return
     */
    public double[][] backPropagate(double[][] delta) {
        // TODO: Learning
        return new double[0][0];
    }

    /**
     * @return the last pooling layer in the CP layer
     */
    public PoolingLayer lastPooling() {
        return poolingLayers.get(
                poolingLayers.size() - 1
        );
    }

    /**
     * @return the last convolution layer in the CP layer
     */
    public ConvolutionLayer lastConvolution() {
        return convolutionLayers.get(
                convolutionLayers.size() - 1
        );
    }

}
