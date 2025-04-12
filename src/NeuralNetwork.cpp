#include <vector>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include "../include/NeuralNetwork.hpp"
#include "../include/Layer.hpp"
#include "../include/Matrix.hpp"

using namespace std;

/**
 * @brief Constructor for creating a new neural network
 * @param topology Vector of integers representing the number of neurons in each layer
 * @param learningRate Learning rate for training
 */

NeuralNetwork::NeuralNetwork(vector<int> topology, double learningRate) {
	this->topologySize = topology.size();
	this->topology = topology;
	this->learningRate = learningRate;

	for (int i = 0; i < topology.size(); i++) {
		Layer *l = new Layer(topology.at(i));
		Matrix *b = new Matrix(topology.at(i), 1, false);
		this->biasMatrices.push_back(b);
		this->layers.push_back(l);
	}

	for (int i = 0; i < this->topologySize - 1; i++) {
		Matrix *m = new Matrix(topology.at(i + 1), topology.at(i), true);
		this->weightMatrices.push_back(m);
	}

}

NeuralNetwork::NeuralNetwork(const string& path) {
	ifstream model(path);
	string chunk;
	string temp;

	if (model.is_open()) {
		// Setting up topology 
		vector<int> topology;
	
		getline(model, chunk, ';');
		stringstream topologyChunk(chunk);

		while(getline(topologyChunk, temp, ',')) {
			topology.push_back(stoi(temp));
		}
	
		this->topology = topology;
		this->topologySize = this->topology.size();
		
		// setting up weights
		for (int i = 0; i < this->topologySize - 1; i++) {
			getline(model, chunk, ';');
			stringstream weightsChunk(chunk);
			Matrix *m = new Matrix(topology.at(i + 1), topology.at(i), false);
			for (int k = 0; k < m->getNumRows(); k++) {
				for (int l = 0; l < m->getNumCols(); l++) {
					getline(weightsChunk, temp, ',');
					m->setVal(k, l, stod(temp));
				}
			}
			this->weightMatrices.push_back(m);
		}

		// Setting up biases
		for (int i = 0; i < this->topologySize; i++) {
			getline(model, chunk, ';');
			stringstream biasesChunk(chunk);
			Matrix *m = new Matrix(topology.at(i), 1, false);
			for (int k = 0; k < m->getNumRows(); k++) {
				for (int l = 0; l < m->getNumCols(); l++) {
					getline(biasesChunk, temp, ',');
					m->setVal(k, l, stod(temp));
				}
			}
			this->biasMatrices.push_back(m);

		
		}

		// setting up learning rate (eventho not needed)
		getline(model, chunk, ';');
		this->learningRate = stod(chunk);

		// Creating Layers
		for (int i = 0; i < this->topologySize; i++) {
			Layer *l = new Layer(this->topology.at(i));
			this->layers.push_back(l);
		}
	}
	model.close();
}

NeuralNetwork::~NeuralNetwork() {
	for (int i = 0; i < this->layers.size(); i++) {
		this->layers.at(i)->cleanup();
		delete this->biasMatrices.at(i);
		delete layers.at(i);
	}
	for (int i = 0; i < this->weightMatrices.size(); i++) {
		delete this->weightMatrices.at(i);
	}
}

void NeuralNetwork::saveModel(const string& path) {
	ofstream file(path);	

	if (file.is_open()) {
		for ( int i = 0; i < this->topologySize; i++) {
			file << this->topology.at(i);
			if (i != this->topologySize - 1) {
				file << ",";
			}
			else {
				file << ";";
			}
			
		}
		for (int i = 0; i < this->topologySize - 1; i++) {
			Matrix *wM = this->getWeightMatrix(i);

			for (int k = 0; k < wM->getNumRows(); k++) {
				for (int l = 0; l < wM->getNumCols(); l++) {
					file << wM->getVal(k, l);
					if (k == wM->getNumRows() - 1 && l == wM->getNumCols() - 1) {
						file << ";";
					}
					else {
						file << ",";
					}
				}
			}
		}
		for (int i = 0; i < this->topologySize; i++) {
			Matrix *wB = this->getBiasMatrix(i);

			for (int k = 0; k < wB->getNumRows(); k++) {
				for (int l = 0; l < wB->getNumCols(); l++) {
					file << wB->getVal(k, l);
					if (k == wB->getNumRows() - 1 && l == wB->getNumCols() - 1) {
						file << ";";
					}
					else {
						file << ",";
					}
				}
			}
		}
		file << this->learningRate << ";";
	}
	file.close();
}

Matrix *NeuralNetwork::predict(vector<double> input) {
	this->setCurrentInput(input);
	this->feedForward();

	return this->layers.at(this->layers.size() - 1)->matrixifyVals();
}

void NeuralNetwork::feedForward() {
	for (int i = 0; i < (this->layers.size() - 1); i++) {
		Matrix *a;

		if (i != 0) {
			a = this->getActivatedNeuronMatrix(i);
		}	
		else {
			a = this->getNeuronMatrix(i);
		}

		Matrix *b = this->getWeightMatrix(i);
		Matrix *d = this->getBiasMatrix(i + 1);


		Matrix *e = *b * *a;
		Matrix *c = *e + *d;

		for (int k = 0; k < c->getNumRows(); k++) {
			this->setNeuronValue(i + 1, k, c->getVal(k, 0));
		}

		delete a;
		delete e;
		delete c;
	} 
}

void NeuralNetwork::backPropogate() {
	this->setErrors();

	// Hidden -> Output
	int outputLayerIndex = this->layers.size() - 1;
	int lastHiddenLayerIndex = outputLayerIndex - 1;
	Matrix *output = this->layers.at(outputLayerIndex)->matrixifyVals();

	Matrix *target = new Matrix(output->getNumRows(), 1, false);
	for (int i = 0; i < output->getNumRows(); i++) {
		target->setVal(i, 0, this->target.at(i));
	}

	Matrix *derivedVals = this->layers.at(outputLayerIndex)->matrixifyDerivedVals();
	Matrix *error = *output - *target;

	Matrix *delta = error->elementwiseMultiply(derivedVals);


	// cleanup from HIDDEN->OUTPUT
	delete error;
	delete derivedVals;
	delete target;
	delete output;

	// Input to hidden and hidden to hidden
	for (int i = lastHiddenLayerIndex; i >= 0; i--) {
		Matrix *vals = i != 0 ? this->layers.at(i)->matrixifyActivatedVals() : 
					this->layers.at(i)->matrixifyVals();
		// Getting Weights and Biases
		Matrix *weights = this->getWeightMatrix(i);
		Matrix *biases = this->getBiasMatrix(i + 1);

		// Gradient Calculation
		Matrix *valsT = vals->transpose();	
		Matrix *gradient = *delta * *valsT;
	
		// Bias updated first since the delta is updated in each
		// iteration and we want the old delta to update the biases

		// Calculating new biases
		// used deltaSC for not altering the original delta
		Matrix deltaSC = *delta;
		deltaSC.scalarMultiply(this->learningRate);
		Matrix *updatedBiases = *biases - deltaSC;

		// Calculating delta
		Matrix *weightsT = weights->transpose();
		derivedVals = this->layers.at(i)->matrixifyDerivedVals(); 
		Matrix *dA = *weightsT * *delta;
		delete delta;
		delta = dA->elementwiseMultiply(derivedVals); // This is the real DELTA

		
		// Calculating new weights
		gradient->scalarMultiply(this->learningRate);
		Matrix *updatedWeights = *weights - *gradient;


		// Setting up weights and biases
		this->setWeightMatrix(i, updatedWeights);
		this->setBiasMatrix(i + 1, updatedBiases);

		// Input/Hidden -> Hidden Memory cleanup
		// Dont't delete updatedWeights
		delete dA;
		delete gradient;
		delete derivedVals;
		delete weightsT;
		delete vals;
		delete valsT;
		
	}
	
	delete delta;
}

void NeuralNetwork::setErrors() {
	int outputLayerIndex = this->layers.size() - 1;
	if (this->target.size() == 0) {
		cerr << "Target is not set for Neural Network!." << endl;
		assert(false);
	}

	if (this->target.size() != this->layers.at(outputLayerIndex)->getNeurons().size()) {
		cerr << "Target is not same size that of the output layer size: " << endl;
		assert(false);
	}

	this->error = 0.0;
	vector<Neuron *> outputNeurons= this->layers.at(outputLayerIndex)->getNeurons();
	for (int i = 0; i < target.size(); i++) {
		double tempErr = 0.5 * pow(outputNeurons.at(i)->getActivatedVal() - this->target.at(i), 2);
		this->errors.push_back(tempErr);
		this->error += tempErr;
	}

	this->historicalErrors.push_back(this->error);

}

void NeuralNetwork::setCurrentInput(vector<double> input) {
	this->input = input;
	for (int i = 0; i < input.size(); i++) {
		this->layers.at(0)->setNeuronVal(i, input.at(i));
	}
}

void NeuralNetwork::setWeightMatrix(int index, Matrix *weightMatrix) {
	delete this->weightMatrices.at(index);
	this->weightMatrices.at(index) = weightMatrix; 
}

void NeuralNetwork::setBiasMatrix(int index, Matrix *biasMatrix) {
	delete this->biasMatrices.at(index);
	this->biasMatrices.at(index) = biasMatrix; 
}

void NeuralNetwork::printInputToConsole() {
	cout << "==========" << endl;
	cout << "INPUT: " << endl;
	Matrix *m = this->layers.at(0)->matrixifyVals();
	m->printToConsole();
	delete m;
}

void NeuralNetwork::printOutputToConsole() {
	cout << "==========" << endl;
	cout << "OUTPUT: " << endl;
	Matrix *m = this->layers.at(this->layers.size() - 1)->matrixifyVals();
	m->printToConsole();
	delete m;
}

void NeuralNetwork::printTargetToConsole() {
	cout << "==========" << endl;
	cout << "TARGET: " << endl;
	for (int i = 0; i < this->target.size(); i++) {
		cout << this->target.at(i) << "\t"; 
	}
	cout << endl;
}
void NeuralNetwork::printToConsole() {
	for (int i = 0; i < this->layers.size(); i++) {
		cout << "=====================" << endl;
		cout << "LAYER: " << i << endl;
		Matrix *m;
		if (i == 0) {
			m = this->layers.at(i)->matrixifyVals();
			m->printToConsole();
		}
		else {
			m = this->layers.at(i)->matrixifyActivatedVals();
			m->printToConsole();
		}
		if (i != this->layers.size() - 1) {
			cout << "Weight: " << endl;
			Matrix *wM = this->getWeightMatrix(i);
			wM->printToConsole();
			cout << "________________" << endl;
		}
		if (i != 0) {
			cout << "Bias: " << endl;
			Matrix *bM = this->getBiasMatrix(i);
			bM->printToConsole();
		}
		cout << "=====================" << endl;

		delete m;
	}
}
