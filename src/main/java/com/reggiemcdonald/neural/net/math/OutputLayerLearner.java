package com.reggiemcdonald.neural.net.math;

import com.reggiemcdonald.neural.net.Layer;
import com.reggiemcdonald.neural.net.Neuron;

public class OutputLayerLearner extends LayerLearner {

    public OutputLayerLearner (Layer layer) {
        super (layer);
    }

    @Override
    public double[] delta (double[] delta_next) {
        double[] newDelta = new double[layer.size()];
        for (Neuron n : layer)
            newDelta[n.neuralIndex()] = n.learner().delta(delta_next);
        return newDelta;
    }
}
