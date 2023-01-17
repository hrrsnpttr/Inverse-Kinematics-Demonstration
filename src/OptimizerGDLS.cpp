#include "OptimizerGDLS.h"
#include "Objective.h"
#include <iostream>

using namespace std;
using namespace Eigen;

OptimizerGDLS::OptimizerGDLS() :
	alphaInit(1.0),
	gamma(0.5),
	tol(1e-6),
	iterMax(100),
	iter(0),
	check(false)
{
	
}

OptimizerGDLS::~OptimizerGDLS()
{
	
}

VectorXd OptimizerGDLS::optimize(const shared_ptr<Objective> objective, const VectorXd &xInit)
{
	int n = xInit.rows();
	VectorXd x = xInit;
	VectorXd g(n);
	MatrixXd H(n, n);
	VectorXd g_filler(n);
	MatrixXd H_filler(n, n);
	VectorXd dx(n);
	iter = 0;

	while (iter < iterMax) {
		iter++;
		
		double f = objective->evalObjective(x, g, H);

		// Finite Differencing Check
		if(check) {
			double e = 1e-7;
			VectorXd g_(n);
			for(int i = 0; i < n; ++i) {
				VectorXd theta_ = x;
				theta_(i) += e;
				double f_ = objective->evalObjective(theta_, g_filler, H_filler);
				g_(i) = (f_ - f)/e;
			}
			cout << "g: " << (g_ - g).norm() << endl;
		}

		double alpha = alphaInit;
		int iterLS = 0;
		

		while (iterLS < iterMax) {
			dx = (-1 * alpha) * g;
			VectorXd sum = x + dx;
			double fnew = objective->evalObjective(sum, g_filler, H_filler);	

			if(fnew < f) { break; }
			alpha *= gamma;

			iterLS++;
		}
		x += dx;
		

		if(dx.norm() < tol) { break; }
	}

	lastF = objective->evalObjective(x, g_filler, H_filler);
	
	return x;
}
