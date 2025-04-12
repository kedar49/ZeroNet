#include <iostream>
#include <vector>
#include <string>
#include "../include/Neuron.hpp" 
#include "../include/Matrix.hpp" 
#include "../include/NeuralNetwork.hpp" 

/**
 * @brief Main function demonstrating neural network training
 * @param argc Argument count
 * @param argv Argument values
 * @return Exit code
 */
int main(int argc, char** argv) {
    // Define network topology: 5 input neurons, 128 and 256 hidden neurons, 10 output neurons
    std::vector<int> topology;
    topology.push_back(5);
    topology.push_back(128);
    topology.push_back(256);
    topology.push_back(10);
    
    // Create input data
    std::vector<double> input;
    input.push_back(1);
    input.push_back(2);
    input.push_back(3);
    input.push_back(4);
    input.push_back(5);
    
    // Create target output data
    std::vector<double> output;
    output.push_back(10);
    output.push_back(4);
    output.push_back(3);
    output.push_back(2);
    output.push_back(1);
    output.push_back(0);
    output.push_back(9);
    output.push_back(8);
    output.push_back(7);
    output.push_back(6);
    
    // Create neural network with learning rate 0.01
    NeuralNetwork* nn = new NeuralNetwork(topology, 0.01);
    
    // Set input and target for training
    nn->setCurrentInput(input);
    nn->setCurrentTarget(output);
    
    std::cout << "Training neural network..." << std::endl;
    
    // Train the network
    const int epochs = 600;
    for (int i = 0; i < epochs; i++) {
        nn->feedForward();
        nn->backPropogate();
        
        // Print progress
        if (i % 100 == 0) {
            std::cout << "Epoch " << i << "/" << epochs << ", Error: " << nn->getError() << std::endl;
        }
    }
    
    // Print final output
    std::cout << "\nFinal output:" << std::endl;
    nn->printOutputToConsole();
    
    // Save the trained model
    nn->saveModel("trained_model.nn");
    std::cout << "Model saved to trained_model.nn" << std::endl;
    
    // Clean up
    delete nn;
    
    return 0;
}
