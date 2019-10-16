package com.reggiemcdonald.neural.feedforward.net.math;

import com.reggiemcdonald.neural.feedforward.net.Layer;
import com.reggiemcdonald.neural.feedforward.net.Neuron;

import java.io.Serializable;

public abstract class LayerLearner implements Serializable {
    protected Layer layer;

    public LayerLearner (Layer layer) {
        this.layer = layer;
    }

    public abstract double[] delta (double[] delta_next);

    public void addBiasUpdate (double[] error) {
        assert (error.length == layer.size());
        for (Neuron n : layer)
            n.learner().addToBiasUpdate ( error[n.neuralIndex()] );

    }

    public void addWeightUpdate (double[] error) {
        assert (error.length == layer.size());
        // TODO
        for (Neuron n : layer) {
            Learner learner = n.learner();
            learner.addToWeightUpdate(learner.computeWeightUpdates( error[n.neuralIndex()] ));
        }
    }

    public void applyUpdates (int n, double eta) {
        for (Neuron neuron : layer) {
            neuron.learner().applyBiasUpdate(n, eta);
            neuron.learner().applyWeightUpdate(n, eta);
        }
    }


}
