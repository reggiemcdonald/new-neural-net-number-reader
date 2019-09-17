package com.reggiemcdonald.neural.convolutional.net;

import com.reggiemcdonald.neural.convolutional.math.*;
import com.reggiemcdonald.neural.convolutional.net.layer.fc.FullyConnectedLayer;
import com.reggiemcdonald.neural.convolutional.net.learning.neuron.*;

public class CNeuronFactory {

    public static final int CN_TYPE_INPT  = 1;
    public static final int CN_TYPE_CONV  = 2;
    public static final int CN_TYPE_POOL  = 3;
    public static final int CN_TYPE_SIGM  = 4;
    public static final int CN_TYPE_SFTM  = 5;

    /**
     * Creates a neuron of the specified type
     * @param type
     * @return
     */
    public static CNeuron makeNeuron (int type, FullyConnectedLayer layer) {
        CNeuron neuron = new CNeuron();
        switch (type) {
            case CN_TYPE_INPT:
                return neuron
                        .learner(new InputCLearner(neuron))
                        .outputFunction(new InputCOutput(neuron))
                        .layer(layer);
            case CN_TYPE_CONV:
                return neuron
                        .learner(new ConvCLearner(neuron))
                        .outputFunction(new ConvCOutput(neuron))
                        .layer(layer);
            case CN_TYPE_POOL:
                return neuron
                        .learner(new PoolCLearner(neuron))
                        .outputFunction(new PoolCOutput(neuron))
                        .layer(layer);
            case CN_TYPE_SIGM:
                return neuron
                        .learner(new SigmoidCLearner(neuron))
                        .outputFunction(new SigmoidCOutput(neuron))
                        .layer(layer);
            case CN_TYPE_SFTM:
                return neuron
                        .learner(new SoftmaxCLearner(neuron))
                        .outputFunction(new SoftmaxCOutput(neuron))
                        .layer(layer);
            default:
                throw new RuntimeException("Unknown factory request");
        }
    }

    /**
     * Creates a neuron of the specified type with bias and output initialized to the specified
     * values
     * @param type
     * @param bias
     * @param output
     * @return
     */
    public static CNeuron makeNeuron (int type, FullyConnectedLayer layer, double bias, double output) {
        CNeuron neuron = new CNeuron (bias, output);
        switch (type) {
            case CN_TYPE_INPT:
                return neuron
                        .learner(new InputCLearner(neuron))
                        .outputFunction(new InputCOutput(neuron))
                        .layer(layer);
            case CN_TYPE_CONV:
                return neuron
                        .learner(new ConvCLearner(neuron))
                        .outputFunction(new ConvCOutput(neuron))
                        .layer(layer);
            case CN_TYPE_POOL:
                return neuron
                        .learner(new PoolCLearner(neuron))
                        .outputFunction(new PoolCOutput(neuron))
                        .layer(layer);
            case CN_TYPE_SIGM:
                return neuron
                        .learner(new SigmoidCLearner(neuron))
                        .outputFunction(new SigmoidCOutput(neuron))
                        .layer(layer);
            case CN_TYPE_SFTM:
                return neuron
                        .learner(new SoftmaxCLearner(neuron))
                        .outputFunction(new SoftmaxCOutput(neuron))
                        .layer(layer);
            default:
                throw new RuntimeException("Unknown factory request");
        }
    }


}
