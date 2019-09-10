package com.reggiemcdonald.neural.convolutional.net.layer;

import java.util.ArrayList;
import java.util.List;

/**
 * A full layer of a neural network containing
 * 1. An d deep convolutional layer
 * 2. A d-deep pooling layer
 * Maintains convolutional and pooling layers in their respective positions
 */
public class CAggregateLayer {
    private int cDim, pDim, depth, cWindowWidth, pWindowWidth, cStride, pStride;
    private boolean isMaxPooling = true; // Default. False is average pooling
    private List<ConvolutionLayer> convolutionLayers;
    private List<PoolingLayer> poolingLayers;

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
        makeLayers();
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
        makeLayers();
    }

    /**
     * Build all the layers of this CAggregateLayer
     */
    private void makeLayers() {
        for (int i = 0; i < depth; i++) {
            convolutionLayers.add (new ConvolutionLayer(cDim, cDim, cWindowWidth));
            poolingLayers.add (new PoolingLayer (pDim, pDim, cWindowWidth, isMaxPooling));
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
    public List<CNNLayer> convolutionalLayers() {
        return new ArrayList<>(convolutionLayers);
    }

    /**
     * @return the list of pooling layers in the network
     */
    public List<CNNLayer> poolingLayers () {
        return new ArrayList<>(poolingLayers);
    }


}
