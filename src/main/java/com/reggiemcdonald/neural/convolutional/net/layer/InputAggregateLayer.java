package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.Propagatable;

import java.util.ArrayList;
import java.util.List;

public class InputAggregateLayer implements Propagatable {
    private int dim, depth, stride;
    private List<InputLayer> inputLayers;

    public InputAggregateLayer (int dim, int depth, int stride) {
        this.dim         = dim;
        this.depth       = depth;
        this.stride      = stride;
        this.inputLayers = new ArrayList<>(depth);
        makeLayers ();
    }

    /**
     * Produce the depth wrapInput layers of the aggregate layer
     */
    private void makeLayers () {
        for (int i = 0; i < depth; i++)
            inputLayers.add (new InputLayer(dim, stride));
    }

    public List<CNNLayer> inputLayers () {
        return new ArrayList<>(inputLayers);
    }

    /**
     * @return the dimension of the wrapInput layers
     */
    public int dim () {
        return dim;
    }

    @Override
    public void propagate () {
        for (Propagatable p : inputLayers())
            p.propagate();
    }

    public CNeuron get (int i, int j, int k) {
        return inputLayers.get(i).get(j, k);
    }

}
