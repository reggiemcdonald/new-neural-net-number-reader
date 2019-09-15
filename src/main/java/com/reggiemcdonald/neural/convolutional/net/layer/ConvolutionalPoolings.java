package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.CSynapse;
import com.reggiemcdonald.neural.convolutional.net.Propagatable;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A full layer of a neural network containing
 * 1. An d deep convolutional layer
 * 2. A d-deep pooling layer
 * Maintains convolutional and pooling layers in their respective positions
 */
public class ConvolutionalPoolings implements Propagatable {
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
        connectConvolutionToPooling ();
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
        makeKernel (kernelDepth);
        makeLayers ();
        connectConvolutionToPooling ();
    }

    /**
     * Build all the layers of this ConvolutionalPoolings
     */
    private void makeLayers() {
        for (int i = 0; i < depth; i++) {
            convolutionLayers.add (new ConvolutionLayer(cDim, cDim, cWindowWidth, makeKernel(kernelDepth), cStride));
            poolingLayers.add (new PoolingLayer (pDim, pDim, pWindowWidth, pStride,true));
        }
    }

    /**
     * Produce the cDim * cDim * depth kernel to be passed to the convolutional layers
     */
    private double[][][] makeKernel (int kernelDepth) {
        Random r = new Random();
        double[][][] kernel = new double[kernelDepth][cDim][cDim];
        for (int i = 0; i < kernelDepth; i++) {
            for (int j = 0; j < cDim; j++)
                for (int k = 0; k < cDim; k++)
                    kernel[i][j][k] = r.nextGaussian();
        }
        return kernel;
    }

    /**
     * Establish the connection between the Convolution and Pooling layers
     * Pooling layer only receives wrapInput from one channel
     */
    private void connectConvolutionToPooling () {
        for (int i = 0; i < poolingLayers.size(); i++)
            poolingLayers.get(i).connect(convolutionLayers.get(i), i);
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
    public List<CNNLayer> convolutionalLayers() {
        return new ArrayList<>(convolutionLayers);
    }

    /**
     * @return the list of pooling layers in the network
     */
    public List<CNNLayer> poolingLayers () {
        return new ArrayList<>(poolingLayers);
    }

    /**
     * Produce a connection from the list of from layers
     * to the convolution layers of this complete layer
     * @param from
     */
    public void connectToThis (List<CNNLayer> from) {
        for (ConvolutionLayer c : convolutionLayers)
            for (int i = 0; i < from.size(); i++)
                c.connect(from.get(i), i);
    }

    /**
     * Returns the total number of outputting pooling neurons in this layer
     * @return
     */
    public int density() {
        return pDim * pDim * depth;
    }

    /**
     * Flattens the pooling layers, returning a linear sigmoidal layer
     * @return SigmoidalLayer
     */
    public CNNLayer flatten () {
        Random r = new Random();
        int density = density();
        CNNLayer sigmoidalFlat = new SigmoidalLayer(density);
        for (int i = 0; i < density; i++) {
            CNeuron from = getPooling(i),
                    to   = sigmoidalFlat.get(i);
            sigmoidalFlat.get(i).addConnectionToThis(
                    new CSynapse(from, to, r.nextGaussian())
            );
        }
        return sigmoidalFlat;
    }

    public CNeuron getPooling (int i) {
        int perLayer = pDim * pDim;
        int x = i / perLayer;
        i -= (x * perLayer);
        return poolingLayers.get(x).get(i);
    }

    public CNeuron getPooling (int x, int y, int k) {
        int i = (x * pDim) + y;
        return poolingLayers.get(k).get(i);
    }

    @Override
    public void propagate() {
        for (Propagatable p : convolutionLayers)
            p.propagate();
        for (Propagatable p : poolingLayers)
            p.propagate();
    }

    public ConvolutionalPoolings workBackwards (double[][] delta) {
        for (int i = poolingLayers.size() - 1; i > -1; i--) {
            PoolingLayer p = poolingLayers.get(i);
            ConvolutionLayer c = convolutionLayers.get(i);
            double[][] poolDelta = p.learner().delta(delta);
            double[][] convDelta = c.learner().delta(poolDelta);
            c.learner().incrementBiasUpdate(convDelta);
            c.learner().incrementWeightUpdate(convDelta);
        }
        return this;
    }
}
