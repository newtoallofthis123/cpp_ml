#include <iostream>
#include <vector>
using namespace std;

// Linear Regression class
// This is used to perform linear regression on only 2 variables
// hence, you would get an equation of the form y = mx + b
template <typename dt> class LinearRegression {
public:
  vector<dt> x;
  vector<dt> y;
  dt intercept = 0;
  dt slope = 1;

  LinearRegression(vector<dt> x, vector<dt> y) {
    static_assert(is_arithmetic<dt>::value, "Data type must be arithmetic");
    static_assert(is_floating_point<dt>::value,
                  "Integer data types are usually not good for linear "
                  "regression, especially such a simple implementation as this "
                  "one. Use floating point types instead.");

    this->x = x;
    this->y = y;
  }

  // Performs linear regression
  // using the least squares method
  // to fit a line to the data
  // this is quite a simple implementation, but slow for large datasets
  // uses the formula:
  // slope = (n * Σ(xy) - Σx * Σy) / (n * Σ(x^2) - (Σx)^2)
  // intercept = (Σy - slope * Σx) / n
  void linear_mse() {
    dt x_sum = 0;
    dt y_sum = 0;
    dt xy_sum = 0;
    dt x_squared_sum = 0;
    int n = x.size();

    for (int i = 0; i < n; i++) {
      x_sum += x[i];
      y_sum += y[i];
      xy_sum += x[i] * y[i];
      x_squared_sum += x[i] * x[i];
    }

    slope = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum * x_sum);
    intercept = (y_sum - slope * x_sum) / n;
  }

private:
  // Calculate the gradient descent
  void cal_gradient_descent(dt learning_rate, int epochs) {
    int n = x.size();

    dt slope = this->slope;
    dt intercept = this->intercept;

    for (int i = 0; i < n; i++) {
      dt y_pred = slope * x[i] + intercept;
      dt error = y[i] - y_pred;
      // This does something cool where instead of calculating the
      // one value to subtract from the slope and intercept, it calculates
      // them directly by simply expanding the formula for the derivative
      // So, instead of slope -= (2.0 / n) * error * x[i] and intercept -= (2.0
      // / n) * error then calculating the new slope and intercept, it
      // calculates them directly
      slope += (2.0 / n) * error * x[i] * learning_rate;
      intercept += (2.0 / n) * error * learning_rate;
    }

    this->slope = slope;
    this->intercept = intercept;
  }

  // Calculate the r squared value
  // Formula is 1 - (ss_res / ss_tot)
  // where ss_res is the sum of the squared residuals
  // and ss_tot is the total sum of squares
  dt r_squared() {
    dt y_mean = 0;
    for (int i = 0; i < y.size(); i++) {
      y_mean += y[i];
    }
    y_mean /= y.size();

    dt ss_res = 0;
    dt ss_tot = 0;
    for (int i = 0; i < y.size(); i++) {
      dt y_pred = slope * x[i] + intercept;
      ss_res += (y[i] - y_pred) * (y[i] - y_pred);
      ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }

    return 1 - (ss_res / ss_tot);
  }

public:
  // Perform gradient descent to find the best fit line
  // for the given data
  // This uses Stochastic Gradient Descent, which is a bit faster
  // than the normal gradient descent
  inline void gradient_descent() {
    int epochs = 1000;
    dt learning_rate = 0.01;
    for (int i = 0; i < epochs; i++) {
      cal_gradient_descent(learning_rate, epochs);
    }
  }

  // Perform gradient descent to find the best fit line
  // for the given data
  // This uses Stochastic Gradient Descent, which is a bit faster
  // than the normal gradient descent
  inline void gradient_descent(dt learning_rate, int epochs) {
    for (int i = 0; i < epochs; i++) {
      cal_gradient_descent(learning_rate, epochs);
    }
  }

  // Predict the value of y for a given x
  // Simply plug the x value into the equation of the line
  dt predict(dt x) { return this->slope * x + this->intercept; }

  // Loss is calculated as the mean squared error
  // MSE = Σ(y - (mx + b))^2 / n
  inline dt mse_loss() {
    int n = x.size();
    dt loss = 0;
    for (int i = 0; i < n; i++) {
      dt y_pred = slope * x[i] + intercept;
      loss += (y[i] - y_pred) * (y[i] - y_pred);
    }
    return loss / n;
  }

  // Accuracy is determined using the r squared value
  inline dt accuracy() { return r_squared(); }

  // Print the equation of the line
  inline void print_line() {
    cout << "y = " << slope << "x + " << intercept << endl;
  }
};
