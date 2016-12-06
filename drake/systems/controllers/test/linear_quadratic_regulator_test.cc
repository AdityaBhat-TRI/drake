#include "drake/systems/controllers/linear_quadratic_regulator.h"

#include "gtest/gtest.h"

#include "drake/common/eigen_matrix_compare.h"
#include "drake/systems/framework/primitives/linear_system.h"

namespace drake {

GTEST_TEST(TestLQR, TestException) {
  Eigen::Matrix2d A;
  Eigen::Vector2d B;

  Eigen::Matrix2d Q = Eigen::Matrix2d::Identity();
  Eigen::Matrix<double, 1, 1> R;
  Eigen::Vector2d N = Eigen::Vector2d::Zero();
  R << 1;

  EXPECT_NO_THROW(systems::LinearQuadraticRegulator(A, B, Q, R, N));
  EXPECT_NO_THROW(systems::LinearQuadraticRegulator(A, B, Q, R));

  // R is not positive definite, should throw exception.
  EXPECT_THROW(systems::LinearQuadraticRegulator(
        A, B, Q, Eigen::Matrix<double, 1, 1>::Zero()), std::runtime_error);
  EXPECT_THROW(systems::LinearQuadraticRegulator(
        A, B, Q, Eigen::Matrix<double, 1, 1>::Zero(), N), std::runtime_error);
}

GTEST_TEST(TestLQR, DoubleIntegrator) {
  // Double integrator dynamics: qddot = u, where q is the position coordinate.
  Eigen::Matrix2d A;
  Eigen::Vector2d B;
  A << 0, 1, 0, 0;
  B << 0, 1;
  systems::LinearSystem<double> sys(A, B, Eigen::Matrix<double, 0, 2>::Zero(),
                                    Eigen::Matrix<double, 0, 1>::Zero());

  // Trivial cost:
  Eigen::Matrix2d Q;
  Eigen::Matrix<double, 1, 1> R;
  Q << 1, 0, 0, 1;
  R << 1;

  // Analytical solution
  Eigen::Matrix<double, 1, 2> K;
  K << 1, std::sqrt(3);

  double tol = 1e-10;

  // Test LinearSystem version of the LQR
  std::unique_ptr<systems::LinearSystem<double>> linear_lqr =
      LinearQuadraticRegulator(sys, Q, R);

  EXPECT_TRUE(CompareMatrices(linear_lqr->A(),
                              Eigen::Matrix<double, 0, 0>::Zero(), tol,
                              MatrixCompareType::absolute));
  EXPECT_TRUE(CompareMatrices(linear_lqr->B(),
                              Eigen::Matrix<double, 0, 2>::Zero(), tol,
                              MatrixCompareType::absolute));
  EXPECT_TRUE(CompareMatrices(linear_lqr->C(),
                              Eigen::Matrix<double, 1, 0>::Zero(), tol,
                              MatrixCompareType::absolute));
  EXPECT_TRUE(
      CompareMatrices(linear_lqr->D(), -K, tol, MatrixCompareType::absolute));

  // Call it as a generic System (by passing in a Context).
  // Should get the same result, but as an affine system.
  auto context = sys.CreateDefaultContext();
  context->FixInputPort(0, Eigen::Matrix<double, 1, 1>::Zero());
  context->get_mutable_continuous_state()->SetFromVector(
      Eigen::Vector2d::Zero());
  std::unique_ptr<systems::AffineSystem<double>> lqr =
      LinearQuadraticRegulator(sys, *context, Q, R);

  EXPECT_TRUE(CompareMatrices(lqr->A(), Eigen::Matrix<double, 0, 0>::Zero(),
                              tol, MatrixCompareType::absolute));
  EXPECT_TRUE(CompareMatrices(lqr->B(), Eigen::Matrix<double, 0, 2>::Zero(),
                              tol, MatrixCompareType::absolute));
  EXPECT_TRUE(CompareMatrices(lqr->xDot0(), Eigen::Matrix<double, 0, 1>::Zero(),
                              tol, MatrixCompareType::absolute));
  EXPECT_TRUE(CompareMatrices(lqr->C(), Eigen::Matrix<double, 1, 0>::Zero(),
                              tol, MatrixCompareType::absolute));
  EXPECT_TRUE(CompareMatrices(lqr->D(), -K, tol, MatrixCompareType::absolute));
  EXPECT_TRUE(CompareMatrices(lqr->y0(), Eigen::Matrix<double, 1, 1>::Zero(),
                              tol, MatrixCompareType::absolute));

  // A different cost function with the same Q and R, and an extra N = [1; 0]
  systems::LinearQuadraticRegulatorResult result =
      systems::LinearQuadraticRegulator(A, B, Q, R, Eigen::Vector2d(1, 0));

  K = Eigen::Vector2d(1, 1);
  Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
  EXPECT_TRUE(CompareMatrices(K, result.K, tol, MatrixCompareType::absolute));
  EXPECT_TRUE(CompareMatrices(S, result.S, tol, MatrixCompareType::absolute));
}

}  // namespace drake
