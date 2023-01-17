#include "OptimizerNM.h"
#include "Objective.h"
#include <iostream>

using namespace std;
using namespace Eigen;

OptimizerNM::OptimizerNM() :
	tol(1e-6),
	iterMax(100),
	iter(0),
	check(false)
{
	
}

OptimizerNM::~OptimizerNM()
{
	
}

VectorXd OptimizerNM::optimize(const shared_ptr<Objective> objective, const VectorXd &xInit)
{
	int n = xInit.rows();
	VectorXd x = xInit;
	VectorXd g(n);
	MatrixXd H(n,n);
	MatrixXd H_filler(n, n);
	iter = 0;

	while (iter < iterMax) {
		iter++;

		objective->evalObjective(x, g, H);

		// Finite Differencing Check
		if(check) {
			double e = 1e-7;
			MatrixXd H_(n, n);
			for(int i = 0; i < n; ++i) {
				VectorXd theta_ = x;
				theta_(i) += e;
				VectorXd g_(n);
				objective->evalObjective(theta_, g_, H_filler);
				H_.col(i) = (g_ - g)/e;
			}
			cout << "H: " << (H_ - H).norm() << endl;
		}

		VectorXd dx = -1.0 * (H.ldlt().solve(g));
		x += dx;
		if(dx.norm() < tol) { break; }
	}

	lastF = objective->evalObjective(x, g, H);

	return x;
}
