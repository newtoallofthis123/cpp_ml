#include "regression/linear.h"
#include <iostream>

using namespace std;

int main() {
  vector<double> x = {1, 2, 3, 4, 5};
  vector<double> y = {2, 3, 4, 5, 6};

  LinearRegression<double> lr(x, y);

  lr.gradient_descent();

  cout << "Slope: " << lr.slope << endl;
  cout << "Intercept: " << lr.intercept << endl;
  cout << "Accuracy: " << lr.accuracy() << endl;

  return 0;
}
