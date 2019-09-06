package com.reggiemcdonald.neural.feedforward.net.math;

import com.reggiemcdonald.neural.feedforward.net.Layer;

public class HiddenLayerLearner extends LayerLearner {

    public HiddenLayerLearner (Layer layer) {
        super (layer);
    }

    /**
     * I need to sum the product of deltaNext [i] with the ith weight
     * of each neuron in the current layer.
     * @param delta_next
     * @return
     */
    @Override
    public double[] delta(double[] delta_next) {
        double[] delta = new double[layer.size()];
        double[] activations = layer.derive(layer.activations());
        Layer forward_layer = layer
                .getNeuronAt(0)
                .getSynapsesFromThis()
                .get(0)
                .to()
                .layer();

        // Get weights of neurons in the next layer
        for (int i = 0; i < delta.length; i++) {
            double d = 0;
            for (int j = 0 ; j < delta_next.length; j++) {
                d +=
                        forward_layer
                        .getNeuronAt(j)
                        .getSynapsesToThis()
                        .get(i)
                        .getWeight() * delta_next[j];
            }
            delta[i] = d;
        }
        for (int i = 0; i < delta.length; i++) {
            delta[i] *= activations[i];
        }
        return delta;
    }
}
