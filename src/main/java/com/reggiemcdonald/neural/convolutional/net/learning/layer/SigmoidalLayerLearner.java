package com.reggiemcdonald.neural.convolutional.net.learning.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;
import com.reggiemcdonald.neural.convolutional.net.layer.CNNLayer;

public class SigmoidalLayerLearner implements CLayerLearner {
    private CNNLayer layer;

    public SigmoidalLayerLearner (CNNLayer layer) {
        this.layer = layer;
    }
    @Override
    public double[] delta(double[] deltaNextLayer) {
        double[] delta = new double[layer.size()];
        double[] activations = derive(activations());
        CNNLayer forwardLayer = layer
                .get(0)
                .synapsesFromThis()
                .get(0)
                .to()
                .layer();

        for (int i = 0; i < delta.length; i++) {
            double d = 0.;
            for (int j = 0; j < deltaNextLayer.length; j++) {
                d += forwardLayer
                        .get(j)
                        .synapsesToThis()
                        .get(i)
                        .weight() * deltaNextLayer[j];
            }
            delta[i] = d * activations[i];
        }
        return delta;
    }

    @Override
    public CLayerLearner incrementBiasUpdate(double[] delta) {
        for (int i = 0; i < delta.length; i++) {
            layer.get(i).learner().incrementBiasUpdate(delta[i]);
        }
        return this;
    }

    @Override
    public CLayerLearner incrementWeightUpdate(double[] delta) {
        for (int i = 0; i < delta.length; i++) {
            CNeuron neuron = layer.get(i);
            neuron.learner().incrementWeightUpdate(
                    neuron.learner().deltaWeight(delta[i])
            );
        }
        return this;
    }

    @Override
    public CLayerLearner finalizeLearning(int batchSize, double eta) {
        for (CNeuron neuron : layer) {
            neuron.learner().applyBiasUpdate  (batchSize, eta);
            neuron.learner().applyWeightUpdate(batchSize, eta);
        }
        return this;
    }

    private double[] derive (double[] activations) {
        for (int i = 0; i < activations.length; i++) {
            activations[i] = activations[i] * (1 - activations[i]);
        }
        return activations;
    }

    private double[] activations () {
        double[] activations = new double[layer.size()];
        for (int i = 0; i < activations.length; i++)
            activations[i] = layer.get(i).output();
        return activations;
    }
}
