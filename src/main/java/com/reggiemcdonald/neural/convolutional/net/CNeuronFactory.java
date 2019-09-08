package com.reggiemcdonald.neural.convolutional.net;

import com.reggiemcdonald.neural.convolutional.math.ConvCOutput;
import com.reggiemcdonald.neural.convolutional.math.InputCOutput;
import com.reggiemcdonald.neural.convolutional.net.learning.ConvCLearner;
import com.reggiemcdonald.neural.convolutional.net.learning.InputCLearner;

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
            case CN_TYPE_CONV:
                return neuron
                        .learner(new ConvCLearner(neuron))
                        .outputFunction(new ConvCOutput(neuron));
            // TODO: Complete neural factory
            default:
                throw new RuntimeException("Not yet implemented");
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
                        .learner(new InputCLearner(neuron))
                        .outputFunction(new ConvCOutput(neuron));
            default:
                throw new RuntimeException("Not yet implemented");
        }
    }


}
