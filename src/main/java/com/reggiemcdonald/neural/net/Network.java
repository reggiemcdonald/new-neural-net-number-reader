package com.reggiemcdonald.neural.net;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A class representing a neural network
 * Should refactor to create a NetworkFactory class
 */
public class Network {
    private Layer input, output;
    private List<Layer> hidden;

    public Network (int[] layer_dimensions) {
        if (layer_dimensions.length < 3)
            throw new RuntimeException ("Network must have at least three layers");
        List<List<Neuron>> layerSets = makeLayerSets (layer_dimensions);
        connectAndCreateLayers(layerSets);
    }

    /**
     * Creates a set of neuron sets that will be converted to layers of the network
     * @param layer_dimensions
     * @return
     */
    private List<List<Neuron>> makeLayerSets (int[] layer_dimensions) {
        List< List<Neuron> > layerSets = new ArrayList<>();
        Random r                     = new Random();
        // Input layer
        List<Neuron> inputSet = new ArrayList<>();
        for (int i = 0; i < layer_dimensions[0]; i++) {
            inputSet.add (new InputNeuron( r.nextGaussian() ));
        }
        layerSets.add (inputSet);

        // Hidden Layer
        for (int i = 1; i < layer_dimensions.length; i++) {
            List<Neuron> layer = new ArrayList<>(layer_dimensions[i]);
            for (int j = 0; j < layer_dimensions[i]; j++)
                layer.add (new SigmoidNeuron( r.nextGaussian (), r.nextGaussian () ));
            layerSets.add (layer);
        }

        // Output Layer
        List<Neuron> outputSet = new ArrayList<>();
        for (int i = 0; i < layer_dimensions[layer_dimensions.length - 1]; i++) {
            outputSet.add (new OutputNeuron( r.nextGaussian (), r.nextGaussian () ));
        }

        return layerSets;
    }

    /**
     * Sets the layers of this
     * @param layerSets
     */
    private void connectAndCreateLayers(List< List<Neuron> > layerSets) {
        Random r = new Random();
        int layerIndex = 0;
        // Create the input layer
        input = connectAndCreateLayer(layerSets.get (0), layerSets.get (1), layerIndex, LayerType.INPUT, r);
        layerIndex++;
        // Create the hidden layers
        for (int i = 1; i < layerSets.size() - 2; i++) {
            hidden.add (connectAndCreateLayer(layerSets.get(i), layerSets.get(i+1), layerIndex, LayerType.HIDDEN, r));
            layerIndex++;
        }
        // Create output layer
        output = connectAndCreateLayer(layerSets.get(layerSets.size()-1), null, layerIndex, LayerType.OUTPUT, r);
    }

    /**
     * Returns a layer with the neurons of earlyLayer. Connects earlylayer to lateLayer if lateLayer is given
     * @param earlyLayer
     * @param lateLayer
     * @param layerIndex
     * @param layerType
     * @param r
     * @return
     */
    private Layer connectAndCreateLayer(List<Neuron> earlyLayer, List<Neuron> lateLayer, int layerIndex, LayerType layerType, Random r) {
        if (lateLayer != null) {
            for (Neuron from : earlyLayer) {
                for (Neuron to : lateLayer)
                    from.addSynapseFromThis(new Synapse(from, to, r.nextGaussian()));
            }
        }
        return new Layer (earlyLayer, layerIndex, layerType);
    }



}
