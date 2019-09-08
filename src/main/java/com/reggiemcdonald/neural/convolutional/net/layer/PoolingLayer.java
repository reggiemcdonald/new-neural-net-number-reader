package com.reggiemcdonald.neural.convolutional.net.layer;

import com.reggiemcdonald.neural.convolutional.net.CNeuron;

import java.util.Iterator;
import java.util.List;

public class PoolingLayer implements CNNLayer {
    private List<CNeuron> neurons;

    public PoolingLayer (int size) {
        // TODO: Layer init
    }

    @Override
    public CNeuron get(int x, int y) {
        return null;
    }

    @Override
    public CNeuron get(int idx) {
        return null;
    }

    @Override
    public CNNLayer connect(CNNLayer from) {
        return this;
    }

    @Override
    public int size() {
        // TODO: Stub
        return 0;
    }

    @Override
    public int dim_x() {
        return 0;
    }

    @Override
    public int dim_y() {
        return 0;
    }

    @Override
    public Iterator<CNeuron> iterator() {
        return neurons.iterator();
    }
}
