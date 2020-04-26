#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
   // Calculate the RMSE.
   VectorXd rmse(4);
   rmse << 0, 0, 0, 0;

   if (estimations.size() != ground_truth.size() || estimations.size() == 0)
   {
      std::cout << "Invalid estimation or ground_truth data" << std::endl;
      return rmse;
   }

   // accumulate squared residuals
   for (size_t i = 0; i < estimations.size(); ++i)
   {
      VectorXd residual(4);
      residual = (estimations[i] - ground_truth[i]).array().square();
      rmse += residual;
   }

   // calculate the mean
   rmse /= estimations.size();

   // calculate the squared root
   rmse = rmse.array().sqrt();

   return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state)
{
   // Calculate the Jacobian matrix.
   MatrixXd jacobian(3, 4);

   // recover state parameters
   double px = x_state(0);
   double py = x_state(1);
   double vx = x_state(2);
   double vy = x_state(3);

   // pre-compute a set of terms to avoid repeated calculation
   double denominator = pow(px, 2) + pow(py, 2);

   // check division by zero
   if (denominator < 1e-4)
   {
      std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
      return jacobian;
   }

   // compute the Jacobian matrix
   jacobian << px/sqrt(denominator), py/sqrt(denominator), 0, 0,
              -py/denominator, px/denominator, 0, 0,
               py*(vx*py-vy*px)/pow(denominator,1.5), px*(vy*px-vx*py)/pow(denominator,1.5), px/sqrt(denominator), py/sqrt(denominator);

   return jacobian;
}
