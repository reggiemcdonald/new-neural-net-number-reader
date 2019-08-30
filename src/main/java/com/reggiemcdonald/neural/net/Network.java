package com.reggiemcdonald.neural.net;

import com.reggiemcdonald.neural.res.NumberImage;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
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
                input.getNeuronAt((i * 28) + j).setOutput(img[i][j] / 255);
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
        output.update();
        return this;
    }

    /**
     * Return an array of outputs from the output layer
     * @return
     */
    public double[] output () {
        return output.activations();
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

    public void learn (List<NumberImage> trainingData, int epochs, int batchSize, double eta, boolean verbose) {
        // Randomize the order of the images
        Collections.shuffle (trainingData);
        // Partition into batches of size batchSize
        List< List<NumberImage> > batches = new ArrayList<>();
        int idx = 0;
        for (int i = 0; i < trainingData.size() / batchSize; i++) {
            List<NumberImage> batch = new ArrayList<>();
            for (int j = 0; j < batchSize; j++) {
                batch.add (trainingData.get (idx));
                idx++;
            }
            batches.add (batch);
        }
        // For 0 to epoch times ...
        for (int i = 0; i < epochs; i++) {
            // For each batch
            System.out.println("Beginning Epoch " + (i+1));
            for (List<NumberImage> batch : batches)
                learn_batch (batch, eta);
            if (verbose)
                test (trainingData);
        }

    }

    private void learn_batch (List<NumberImage> batch, double eta) {
        // TODO
        // Input each image and then backwards propagate
        for (NumberImage x : batch) {
            input (x.image);
            propagate ();
            backwardsPropagate (x.label);
        }
        finalizeLearning (batch.size(), eta);
    }

    private Network backwardsPropagate(double[] expected) {
        // Compute error in output layer
        double[] delta = output.delta(output_cost_derivative(expected));
        output.addBiasUpdate (delta);
        output.addWeightUpdate (delta);
        // Propagate to the earlier layers
        for (int i = hidden.size() - 1; i > -1; i--) {
            Layer layer = hidden.get(i);
            delta = layer.delta(delta);
            layer.addBiasUpdate(delta);
            layer.addWeightUpdate(delta);
        }
        return this;
    }

    private double[] output_cost_derivative (double[] expected) {
        double[] activations = output();
        double[] delta = new double[activations.length];
        for (int i = 0; i < activations.length; i++)
            delta[i] = activations[i] - expected[i];
        return delta;
    }

    /**
     * Count the number of images correctly identified
     * @param data
     */
    public void test (List<NumberImage> data) {
        int num_correct = 0;
        for (NumberImage i : data) {
            input (i.image);
            propagate();
            if (result(output()) == result(i.label))
                num_correct++;
        }
        System.out.println("Correct: " + num_correct + " out of " + data.size());

    }

    private void finalizeLearning (int n, double eta) {
        output.applyUpdates(n, eta);
        for (Layer layer : hidden)
            layer.applyUpdates(n, eta);
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

    /**
     * Restores a Network object from file located at the specified path
     * @param path
     * @return
     */
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
