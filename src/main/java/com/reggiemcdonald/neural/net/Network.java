package com.reggiemcdonald.neural.net;

import java.util.List;

public class Network {
    private Layer input, output;
    private List<Layer> hidden;

    public Network (int[] layer_dimensions) {
        if (layer_dimensions.length < 3)
            throw new RuntimeException ("Network must have at least three layers");

        input = createLayer (layer_dimensions[0], LayerType.INPUT);

        for (int i = 1; i < layer_dimensions.length - 1; i++)
            hidden.add (createLayer (layer_dimensions[i], LayerType.HIDDEN));

        output = createLayer (layer_dimensions[layer_dimensions.length-1], LayerType.OUTPUT);
    }

    private Layer createLayer (int size, LayerType type) {
        Layer layer;
        switch (type) {
            case INPUT:
                layer = null;
                break;
            case HIDDEN:
                layer = null;
                break;
            case OUTPUT:
                layer = null;
                break;
            default:
                throw new RuntimeException ("Invalid layer type");
        }
        return layer;
    }

}
