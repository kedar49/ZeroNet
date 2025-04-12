#ifndef _NEURALNETWORK_HPP_
#define _NEURALNETWORK_HPP_

#include <iostream>
#include <vector>
#include <string>
#include "Matrix.hpp"
#include "Layer.hpp"

using namespace std;

/**
 * @class NeuralNetwork
 * @brief Represents a fully connected neural network
 * 
 * This class implements a multi-layer neural network with configurable topology.
 * It supports forward propagation, backpropagation, and model saving/loading.
 */
class NeuralNetwork {
public:	
    /**
     * @brief Constructor for creating a new neural network
     * @param topology Vector of integers representing the number of neurons in each layer
     * @param learningRate Learning rate for training
     */
    NeuralNetwork(vector<int> topology, double learningRate);
    
    /**
     * @brief Constructor for loading a neural network from a file
     * @param path Path to the saved model file
     */
    NeuralNetwork(const string& path);
    
    /**
     * @brief Destructor to clean up memory
     */
    ~NeuralNetwork();
    
    /**
     * @brief Prints the entire network state to the console
     */
    void printToConsole();
    
    /**
     * @brief Prints the input layer values to the console
     */
    void printInputToConsole();
    
    /**
     * @brief Prints the output layer values to the console
     */
    void printOutputToConsole();
    
    /**
     * @brief Prints the target values to the console
     */
    void printTargetToConsole();
    
    /**
     * @brief Performs forward propagation through the network
     */
    void feedForward();
    
    /**
     * @brief Performs backpropagation to update weights and biases
     */
    void backPropogate();
    
    /**
     * @brief Calculates error between output and target
     */
    void setErrors();
    
    /**
     * @brief Saves the model to a file
     * @param path Path to save the model
     */
    void saveModel(const string& path);

    /**
     * @brief Gets the matrix of raw neuron values for a layer
     * @param index Layer index
     * @return Matrix of neuron values
     */
    Matrix* getNeuronMatrix(int index) { return this->layers.at(index)->matrixifyVals(); }
    
    /**
     * @brief Gets the matrix of activated neuron values for a layer
     * @param index Layer index
     * @return Matrix of activated neuron values
     */
    Matrix* getActivatedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyActivatedVals(); }
    
    /**
     * @brief Gets the matrix of derived neuron values for a layer
     * @param index Layer index
     * @return Matrix of derived neuron values
     */
    Matrix* getDerivedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyDerivedVals(); }
    
    /**
     * @brief Gets the weight matrix between two layers
     * @param index Index of the weight matrix
     * @return Weight matrix
     */
    Matrix* getWeightMatrix(int index) { return this->weightMatrices.at(index); }
    
    /**
     * @brief Gets the bias matrix for a layer
     * @param index Layer index
     * @return Bias matrix
     */
    Matrix* getBiasMatrix(int index) { return this->biasMatrices.at(index); }

    /**
     * @brief Makes a prediction using the neural network
     * @param input Input vector
     * @return Matrix containing the output prediction
     */
    Matrix* predict(vector<double> input);

    /**
     * @brief Sets the value of a specific neuron
     * @param indexLayer Layer index
     * @param indexNeuron Neuron index within the layer
     * @param value New value for the neuron
     */
    void setNeuronValue(int indexLayer, int indexNeuron, double value) { 
        this->layers.at(indexLayer)->setNeuronVal(indexNeuron, value); 
    }
    
    /**
     * @brief Sets the current input vector
     * @param input Input vector
     */
    void setCurrentInput(vector<double> input);
    
    /**
     * @brief Sets the current target vector
     * @param input Target vector
     */
    void setCurrentTarget(vector<double> input) { this->target = input; }
    
    /**
     * @brief Sets a weight matrix
     * @param index Index of the weight matrix
     * @param weightMatrix New weight matrix
     */
    void setWeightMatrix(int index, Matrix* weightMatrix);
    
    /**
     * @brief Sets a bias matrix
     * @param index Index of the bias matrix
     * @param biasMatrix New bias matrix
     */
    void setBiasMatrix(int index, Matrix* biasMatrix);

    /**
     * @brief Gets the error vector
     * @return Vector of errors
     */
    vector<double> getErrors() const { return this->errors; }
    
    /**
     * @brief Gets the total error
     * @return Total error value
     */
    double getError() const { return this->error; }
    
    /**
     * @brief Gets the network topology
     * @return Vector representing the network topology
     */
    vector<int> getTopology() const { return this->topology; }
    
    /**
     * @brief Gets the size of the topology
     * @return Number of layers in the network
     */
    int getTopologySize() const { return this->topologySize; }
    
    /**
     * @brief Gets the historical errors
     * @return Vector of historical error values
     */
    vector<double> getHistoricalErrors() const { return this->historicalErrors; }
private:
    int topologySize;                   ///< Number of layers in the network
    vector<int> topology;          ///< Vector defining neurons per layer
    vector<Layer*> layers;         ///< Vector of layer pointers
    vector<Matrix*> weightMatrices; ///< Weight matrices between layers
    vector<Matrix*> biasMatrices;  ///< Bias matrices for each layer
    vector<double> input;          ///< Current input vector
    vector<double> target;         ///< Current target vector
    vector<double> errors;         ///< Current errors vector
    vector<double> historicalErrors; ///< History of errors during training
    double error;                       ///< Current total error
    double learningRate;                ///< Learning rate for training


};
#endif
