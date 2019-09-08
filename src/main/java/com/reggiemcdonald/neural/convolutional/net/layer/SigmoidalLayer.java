package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;

import java.util.Iterator;
import java.util.List;

public class SigmoidalLayer implements CNNLayer {
    private List<CNeuron> neurons;

    public SigmoidalLayer (int size) {
        // TODO: Layer init
    }

    @Override
    public CNeuron get(int x, int y) {
        return null;
    }

    @Override
    public CNNLayer connect(CNNLayer to) {
        return this;
    }

    @Override
    public int size() {
        // TODO: Stub
        return 0;
    }

    @Override
    public Iterator<CNeuron> iterator() {
        return neurons.iterator();
    }
}
