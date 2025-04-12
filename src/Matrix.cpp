#include <iostream>
#include <random>
#include <vector>
#include <cassert>

#include "../include/Matrix.hpp"

/**
 * @brief Constructor for Matrix
 * @param numRows Number of rows in the matrix
 * @param numCols Number of columns in the matrix
 * @param isRandom Whether to initialize with random values
 */
Matrix::Matrix(int numRows, int numCols, bool isRandom) {
    this->numRows = numRows;
    this->numCols = numCols;
    
    for (int i = 0; i < numRows; i++) {
        double r = 0.0;
        std::vector<double> colVals;
        for (int k = 0; k < numCols; k++) {
            if (isRandom) {
                r = this->getRandNo();
            }
            colVals.push_back(r);
        }
        this->values.push_back(colVals);
    }
}

/**
 * @brief Generates a random number for matrix initialization
 * @return Random double value
 */
double Matrix::getRandNo() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    return dis(gen);
}

/**
 * @brief Prints the matrix to the console
 */
void Matrix::printToConsole() {
    for (int i = 0; i < numRows; i++) {
        for (int k = 0; k < numCols; k++) {
            std::cout << this->values.at(i).at(k) << "\t";
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Creates a transposed version of this matrix
 * @return Pointer to the transposed matrix
 */
Matrix* Matrix::transpose() {
    Matrix* m = new Matrix(this->numCols, this->numRows, false);
    for (int i = 0; i < this->numCols; i++) {
        for (int k = 0; k < this->numRows; k++) {
            m->setVal(i, k, this->getVal(k, i));
        }
    }

    return m;
}

/**
 * @brief Multiplies all elements by a scalar value
 * @param scalar The scalar value to multiply by
 */
void Matrix::scalarMultiply(double scalar) {
    for (int i = 0; i < this->getNumRows(); i++) {
        for (int k = 0; k < this->getNumCols(); k++) {
            this->setVal(i, k, this->getVal(i, k) * scalar);
        }
    }
}

/**
 * @brief Matrix addition operator
 * @param b Matrix to add
 * @return Pointer to the resulting matrix
 */
Matrix* Matrix::operator+(Matrix& b) {
    if (this->getNumRows() != b.getNumRows() || this->getNumCols() != b.getNumCols()) {
        std::cerr << "Rows and Column sizes mismatch: " << std::endl;
        assert(false);
    }

    Matrix* m = new Matrix(this->getNumRows(), this->getNumCols(), false);

    for (int i = 0; i < this->getNumRows(); i++) {
        for (int k = 0; k < this->getNumCols(); k++) {
            m->setVal(i, k, this->getVal(i, k) + b.getVal(i, k));
        }
    }

    return m;
}

/**
 * @brief Matrix subtraction operator
 * @param b Matrix to subtract
 * @return Pointer to the resulting matrix
 */
Matrix* Matrix::operator-(Matrix& b) {
    if (this->getNumRows() != b.getNumRows() || this->getNumCols() != b.getNumCols()) {
        std::cerr << "Rows and Column sizes mismatch: " << std::endl;
        assert(false);
    }

    Matrix* m = new Matrix(this->getNumRows(), this->getNumCols(), false);

    for (int i = 0; i < this->getNumRows(); i++) {
        for (int k = 0; k < this->getNumCols(); k++) {
            m->setVal(i, k, this->getVal(i, k) - b.getVal(i, k));
        }
    }

    return m;
}

/**
 * @brief Matrix multiplication operator
 * @param b Matrix to multiply with
 * @return Pointer to the resulting matrix
 */
Matrix* Matrix::operator*(Matrix& b) {
    if (this->getNumCols() != b.getNumRows()) {
        std::cerr << "Matrix dimensions incompatible for multiplication: " << std::endl;
        std::cerr << "A: " << this->getNumRows() << "x" << this->getNumCols() << std::endl;
        std::cerr << "B: " << b.getNumRows() << "x" << b.getNumCols() << std::endl;
        assert(false);
    }

    Matrix* c = new Matrix(this->getNumRows(), b.getNumCols(), false);

    for (int i = 0; i < this->getNumRows(); i++) {
        for (int k = 0; k < b.getNumCols(); k++) {
            for (int l = 0; l < this->getNumCols(); l++) { 
                double v = this->getVal(i, l) * b.getVal(l, k);
                double nv = c->getVal(i, k) + v;
                c->setVal(i, k, nv);
            }
        }
    }

    return c;
}

/**
 * @brief Performs element-wise multiplication with another matrix
 * @param m Pointer to the matrix to multiply with
 * @return Pointer to the resulting matrix
 */
Matrix* Matrix::elementwiseMultiply(Matrix* m) {
    if (m->getNumRows() != this->getNumRows() || m->getNumCols() != this->getNumCols()) {
        std::cerr << "Dimensions mismatch for element-wise multiplication: " << std::endl;
        assert(false);
    }

    Matrix* temp = new Matrix(m->getNumRows(), m->getNumCols(), false);

    for (int i = 0; i < m->getNumRows(); i++) {
        for (int k = 0; k < m->getNumCols(); k++) {
            temp->setVal(i, k, this->getVal(i, k) * m->getVal(i, k));
        }
    }

    return temp;
}

/**
 * @brief Converts the matrix to a vector
 * @return Vector containing all matrix elements
 */
std::vector<double> Matrix::toVector() {
    std::vector<double> v;
    for (int i = 0; i < this->values.size(); i++) {
        for (int k = 0; k < this->values.at(i).size(); k++) {
            v.push_back(values.at(i).at(k));
        }
    }

    return v;
}
