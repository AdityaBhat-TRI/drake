#include "drake/common/trajectories/exponential_plus_piecewise_polynomial.h"

#include <cmath>
#include <random>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "drake/common/trajectories/test/random_piecewise_polynomial.h"

using std::default_random_engine;
using std::uniform_real_distribution;

namespace drake {
namespace trajectories {

namespace {

template <typename T>
void testSimpleCase() {
  int num_coefficients = 5;

  MatrixX<T> K = MatrixX<T>::Random(1, 1);
  MatrixX<T> A = MatrixX<T>::Random(1, 1);
  MatrixX<T> alpha = MatrixX<T>::Random(1, 1);

  std::vector<double> breaks = {1.2, 3.4};
  default_random_engine generator;
  auto polynomial_part =
      test::MakeRandomPiecewisePolynomial<T>(1, 1, num_coefficients, breaks);

  ExponentialPlusPiecewisePolynomial<T> expPlusPp(K, A, alpha, polynomial_part);
  EXPECT_EQ(expPlusPp.rows(), 1);
  EXPECT_EQ(expPlusPp.cols(), 1);
  EXPECT_EQ(expPlusPp.get_number_of_segments(), 1);
  EXPECT_EQ(expPlusPp.start_time(), breaks[0]);
  EXPECT_EQ(expPlusPp.end_time(), breaks[1]);

  ExponentialPlusPiecewisePolynomial<T> derivative = expPlusPp.derivative();

  uniform_real_distribution<T> uniform(expPlusPp.start_time(),
                                       expPlusPp.end_time());
  double t = uniform(generator);
  auto check = K(0) * std::exp(A(0) * (t - expPlusPp.start_time())) * alpha(0) +
               polynomial_part.scalarValue(t);
  auto derivative_check =
      K(0) * A(0) * std::exp(A(0) * (t - expPlusPp.start_time())) * alpha(0) +
      polynomial_part.derivative().scalarValue(t);

  EXPECT_NEAR(check, expPlusPp.value(t)(0), 1e-8);
  EXPECT_NEAR(derivative_check, derivative.value(t)(0), 1e-8);

  const double kShift = 5.6;
  expPlusPp.shiftRight(kShift);
  EXPECT_EQ(expPlusPp.start_time(), breaks[0] + kShift);
  EXPECT_EQ(expPlusPp.end_time(), breaks[1] + kShift);
}

GTEST_TEST(testExponentialPlusPiecewisePolynomial, BasicTest) {
  testSimpleCase<double>();
}

}  // namespace
}  // namespace trajectories
}  // namespace drake
