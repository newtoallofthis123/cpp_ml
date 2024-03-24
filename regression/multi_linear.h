#include <iostream>
#include <vector>
using namespace std;

template <typename dt> class MultiLinearRegression {
public:
  vector<vector<dt>> x;
  vector<dt> y;
  vector<dt> weights;
  dt bias = 0;

  // Construct the MultiLinearRegression
  MultiLinearRegression(vector<vector<dt>> x, vector<dt> y) {
    this->x = x;
    this->y = y;
    weights = vector<dt>(x[0].size(), 0);
  }

private:
  // Calculate the inner product of two vectors
  // This is useful for calculating the dot product of two vectors
  // which would be useful for calculating the error
  dt dot(typename vector<dt>::iterator first1,
         typename vector<dt>::iterator last1,
         typename vector<dt>::iterator first2, dt init) {
    dt result = init;
    while (first1 != last1) {
      result += (*first1) * (*first2);
      first1++;
      first2++;
    }
    return result;
  }

  void cal_gradient_descent(dt learning_rate, int epochs) {
    int n = x.size();
    int m = x[0].size();

    vector<dt> weights = this->weights;
    dt bias = this->bias;

    for (int i = 0; i < n; i++) {
      dt y_pred = dot(weights.begin(), weights.end(), x[i].begin(), 0) + bias;
      dt error = y[i] - y_pred;
      for (int j = 0; j < m; j++) {
        weights[j] += (2.0 / n) * error * x[i][j] * learning_rate;
      }
      bias += (2.0 / n) * error * learning_rate;
    }

    this->weights = weights;
    this->bias = bias;
  }

public:
  // Perform gradient descent to find the best fit line
  // for the given data
  inline void gradient_descent() {
    int epochs = 1000;
    dt learning_rate = 0.01;
    for (int i = 0; i < epochs; i++) {
      cal_gradient_descent(learning_rate, epochs);
    }
  }

  // Perform gradient descent to find the best fit line
  // for the given data
  void gradient_descent(dt learning_rate, int epochs) {
    for (int i = 0; i < epochs; i++) {
      cal_gradient_descent(learning_rate, epochs);
    }
  }

  dt predict(vector<dt> x) {
    return dot(weights.begin(), weights.end(), x.begin(), 0) + bias;
  }

  void print_weights() {
    cout << "Weights: ";
    for (auto w : weights) {
      cout << w << " ";
    }
    cout << endl;
  }

  void print_bias() { cout << "Bias: " << bias << endl; }
};
