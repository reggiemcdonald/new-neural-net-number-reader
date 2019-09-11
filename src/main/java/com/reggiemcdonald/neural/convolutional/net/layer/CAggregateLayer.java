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
public class CAggregateLayer implements Propagatable {
    private int cDim, pDim, depth, cWindowWidth, pWindowWidth, cStride, pStride;
    private boolean isMaxPooling = true; // Default. False is average pooling
    private List<ConvolutionLayer> convolutionLayers;
    private List<PoolingLayer> poolingLayers;
    private double[][][] kernel;

    public CAggregateLayer(int cDim, int pDim,  int depth, int cWindowWidth, int pWindowWidth, int cStride, int pStride) {
        this.cDim  = cDim;
        this.pDim  = pDim;
        this.depth = depth;
        this.cWindowWidth = cWindowWidth;
        this.pWindowWidth = pWindowWidth;
        this.cStride = cStride;
        this.pStride = pStride;
        this.convolutionLayers = new ArrayList<>(depth);
        this.poolingLayers = new ArrayList<>(depth);
        makeKernel ();
        makeLayers ();
        connectConvolutionToPooling ();
    }

    public CAggregateLayer(int cDim, int pDim, int depth, int cWindowWidth, int pWindowWidth, int cStride, int pStride, boolean isMaxPooling) {
        this.cDim              = cDim;
        this.pDim              = pDim;
        this.depth             = depth;
        this.cWindowWidth      = cWindowWidth;
        this.pWindowWidth      = pWindowWidth;
        this.cStride           = cStride;
        this.pStride           = pStride;
        this.isMaxPooling      = isMaxPooling;
        this.convolutionLayers = new ArrayList<>(depth);
        this.poolingLayers     = new ArrayList<>(depth);
        makeKernel ();
        makeLayers ();
        connectConvolutionToPooling ();
    }

    /**
     * Build all the layers of this CAggregateLayer
     */
    private void makeLayers() {
        for (int i = 0; i < depth; i++) {
            convolutionLayers.add (new ConvolutionLayer(cDim, cDim, cWindowWidth, kernel));
            poolingLayers.add (new PoolingLayer (pDim, pDim, pWindowWidth, isMaxPooling));
        }
    }

    /**
     * Produce the cDim * cDim * depth kernel to be passed to the convolutional layers
     */
    private void makeKernel () {
        Random r = new Random();
        kernel = new double[depth][cDim][cDim];
        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < cDim; j++)
                for (int k = 0; k < cDim; k++)
                    kernel[i][j][k] = r.nextGaussian();
        }
    }

    /**
     * Establish the connection between the Convolution and Pooling layers
     * Pooling layer only receives input from one channel
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
            for (int i = 0; i < convolutionLayers.size(); i++)
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
        // TODO: Propagate
    }
}
