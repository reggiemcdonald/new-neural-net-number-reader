package com.reggiemcdonald.neural.net;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A class representing a neural network
 * Should refactor to create a NetworkFactory class
 */
public class Network implements Serializable {
    public static final String SAVE_PATH = "network_state.nerl";

    private Layer input, output;
    private List<Layer> hidden;

    public Network (int[] layer_dimensions) {
        if (layer_dimensions.length < 3)
            throw new RuntimeException ("Network must have at least three layers");
        hidden = new ArrayList<>();
        List<List<Neuron>> layerSets = makeLayerSets (layer_dimensions);
        connectAndCreateLayers (layerSets);
    }

    /**
     * Creates a set of neuron sets that will be converted to layers of the network
     * @param layer_dimensions
     * @return
     */
    private List<List<Neuron>> makeLayerSets (int[] layer_dimensions) {
        List< List<Neuron> > layerSets = new ArrayList<>(layer_dimensions.length);
        Random r                     = new Random();
        // Input layer
        List<Neuron> inputSet = new ArrayList<>();
        for (int i = 0; i < layer_dimensions[0]; i++) {
            inputSet.add (new InputNeuron( r.nextGaussian() ));
        }
        layerSets.add (inputSet);

        // Hidden Layer
        for (int i = 1; i < layer_dimensions.length - 1; i++) {
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
        layerSets.add (outputSet);
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
        for (int i = 1; i < layerSets.size() - 1; i++) {
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

    /**
     * Set the input layer for propagation, returning this
     * @param img
     * @return
     */
    public Network input (double[][] img) {
        for (int i = 0 ; i < img.length ; i++) {
            for (int j = 0; j < img[i].length; j++) {
                input.getNeuronAt((i * 28) + j).setOutput(img[i][j]);
            }
        }
        return this;
    }

    /**
     * Forward propagate throughout the neural network, returning this
     * Assumes that the input layer has been set
     * @return
     */
    public Network propagate () {
        // Propagate the signal throughout the layers of the neural network
        for (Layer layer : hidden)
            layer.update ();
        return this;
    }

    /**
     * Return an array of outputs from the output layer
     * @return
     */
    public double[] output () {
        double[] o = new double[output.size()];
        for (Neuron n : output)
            o[n.neuralIndex()] = n.getOutput();
        return o;
    }

    /**
     * Return the index of the neuron in the output layer with the highest output
     * @return
     */
    public int result (double[] o) {
        int res = 0;
        double max = o[0];
        for (int i = 1; i < o.length ; i++) {
            if (o[i] > max) {
                res = i;
                max = o[i];
            }
        }
        return res;
    }

    /**
     * Save the network, layers, neurons, and synapses, their weights and biases
     * Does not save the double output value in each neuron
     * ~~~~~ All fields labelled transient are not serialized ~~~~~~
     * @param network
     */
    public static void save (Network network, String path) {
        File f = new File (path);

        try (FileOutputStream   fos = new FileOutputStream (f);
             ObjectOutputStream oos = new ObjectOutputStream (fos)) {
            oos.writeObject(network);
        } catch (Exception e) {
            System.out.println("Network save issue: " + e.getMessage());
        }
    }

    public static Network load (String path) {
        File f = new File (path);
        Network network = null;
        try (FileInputStream fis = new FileInputStream(f);
            ObjectInputStream ois = new ObjectInputStream(fis)) {
            network = (Network)ois.readObject();
            return network;
        } catch (Exception e) {
            System.out.println("There was an error restoring the network: " + e.getMessage());
        }
        return network;
    }


}
