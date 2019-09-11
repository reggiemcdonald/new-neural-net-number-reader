package com.reggiemcdonald.neural.convolutional.net;

import com.reggiemcdonald.neural.convolutional.math.*;
import com.reggiemcdonald.neural.convolutional.net.learning.*;

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
    public static CNeuron makeNeuron (int type) {
        CNeuron neuron = new CNeuron();
        switch (type) {
            case CN_TYPE_INPT:
                return neuron
                        .learner(new InputCLearner(neuron))
                        .outputFunction(new InputCOutput(neuron));
            // TODO: Complete neural factory
            case CN_TYPE_CONV:
                return neuron
                        .learner(new ConvCLearner(neuron))
                        .outputFunction(new ConvCOutput(neuron));
            case CN_TYPE_POOL:
                return neuron
                        .learner(new PoolCLearner(neuron))
                        .outputFunction(new PoolCOutput(neuron));
            case CN_TYPE_SIGM:
                return neuron
                        .learner(new SigmoidCLearner(neuron))
                        .outputFunction(new SigmoidCOutput(neuron));
            case CN_TYPE_SFTM:
                return neuron
                        .learner(new SoftmaxCLearner(neuron))
                        .outputFunction(new SoftmaxCOutput(neuron));
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
    public static CNeuron makeNeuron (int type, double bias, double output) {
        CNeuron neuron = new CNeuron (bias, output);
        switch (type) {
            case CN_TYPE_INPT:
                return neuron
                        .learner(new InputCLearner(neuron))
                        .outputFunction(new InputCOutput(neuron));
            // TODO: Complete neural factory
            case CN_TYPE_CONV:
                return neuron
                        .learner(new ConvCLearner(neuron))
                        .outputFunction(new ConvCOutput(neuron));
            case CN_TYPE_POOL:
                return neuron
                        .learner(new PoolCLearner(neuron))
                        .outputFunction(new PoolCOutput(neuron));
            case CN_TYPE_SIGM:
                return neuron
                        .learner(new SigmoidCLearner(neuron))
                        .outputFunction(new SigmoidCOutput(neuron));
            case CN_TYPE_SFTM:
                return neuron
                        .learner(new SoftmaxCLearner(neuron))
                        .outputFunction(new SoftmaxCOutput(neuron));
            default:
                throw new RuntimeException("Unknown factory request");
        }
    }


}
