#include "ObjectiveAssign.h"
#include <cmath>
#include <iostream>

using namespace std;
using namespace Eigen;

ObjectiveAssign::ObjectiveAssign()
{
	r << 1.0, 0, 1.0;
	w_tar = 0;
	w_reg = 0;
	std::vector<Matrix3d> T;
}

ObjectiveAssign::~ObjectiveAssign()
{
	
}

double ObjectiveAssign::evalObjective(const VectorXd &x)
{
	return -1;

}

double ObjectiveAssign::evalObjective(const VectorXd &x, VectorXd &g)
{
	return evalObjective(x);
}

double ObjectiveAssign::evalObjective(const VectorXd &x, VectorXd &g, MatrixXd &H)
{
	// Create pDelta
	Vector3d pTheta = pCalc(x, -1, -1); // GOOD

	Vector2d pThetaTrunc;
	pThetaTrunc << pTheta(0), pTheta(1);

	Vector2d pDelta = pThetaTrunc - pTarget; // GOOD


	// Create ppTheta
	MatrixXd ppTheta(2, x.rows()); // GOOD

	for(int i = 0; i < ppTheta.cols(); i++) {
		Vector3d tmp = pCalc(x, i, -1);
		ppTheta(0, i) = tmp(0);
		ppTheta(1, i) = tmp(1);
	}
	

	// Create pppTheta
	MatrixXd pppTheta(2 * x.rows(), x.rows()); // GOOD

	for(int i = 0; i < pppTheta.cols(); i++) {
		for(int j = 0; j < pppTheta.rows(); j+=2) {
			Vector3d tmp = pCalc(x, i, (j / 2));
			pppTheta(0 + j, i) = tmp(0);
			pppTheta(1 + j, i) = tmp(1);
		}
	}

	

	// Create pDelta.transpose() * ppTheta matrix
	VectorXd gradOne(x.rows());
	for(int i = 0; i < gradOne.rows(); i++) {
		Vector2d tmp;
		tmp << ppTheta(0, i), ppTheta(1, i);
		gradOne(i) = pDelta.transpose() * tmp;
	}

	
	// Calculate gradient
	g = (w_tar * gradOne) + (w_reg * x); // GOOD


	// Calculate Hessian
	MatrixXd hessOne(x.rows(), x.rows());
	for(int i = 0; i < hessOne.cols(); i++) {
		for(int j = 0; j < hessOne.rows(); j++) {
			Vector2d tmpOne;
			tmpOne << ppTheta(0, j), ppTheta(1, j);
			Vector2d tmpTwo;
			tmpTwo << ppTheta(0, i), ppTheta(1, i);

			hessOne(j, i) = tmpOne.transpose() * tmpTwo;
		}
	}

	MatrixXd hessTwo(x.rows(), x.rows());
	for(int i = 0; i < hessTwo.cols(); i++) {
		for(int j = 0; j < hessTwo.rows(); j++) {
			Vector2d tmp;
			tmp << pppTheta(j * 2, i), pppTheta((j * 2) + 1, i);

			hessTwo(j, i) = pDelta.transpose() * tmp;
		}
	}

	MatrixXd I(x.rows(), x.rows()); // GOOD
	for(int i = 0; i < I.cols(); i++) {
		for(int j = 0; j < I.rows(); j++) {
			if(j == i) {
				I(j, i) = 1.0;
			} else {
				I(j, i) = 0.0;
			}
		}
	}
	
	H = ((w_tar * (hessOne + hessTwo)) + (w_reg * I)); // GOOD


	// Calculate f(theta)
	return (0.5 * w_tar * pDelta.dot(pDelta)) + (0.5 * w_reg * x.dot(x)); // GOOD
}

Matrix3d ObjectiveAssign::R(double theta)
{
	Matrix3d tmp;
	tmp << cos(theta), (-1.0 * sin(theta)), 0,
		   sin(theta), cos(theta), 0,
		   0, 0, 1;
	return tmp;
}

Matrix3d ObjectiveAssign::Rp(double theta)
{
	Matrix3d tmp;
	tmp << (-1.0 * sin(theta)), (-1.0 * cos(theta)), 0,
		   cos(theta), (-1.0 * sin(theta)), 0,
		   0, 0, 0;
	return tmp;
}

Matrix3d ObjectiveAssign::Rpp(double theta)
{
	Matrix3d tmp;
	tmp << (-1.0 * cos(theta)), (sin(theta)), 0,
		   (-1.0 * sin(theta)), (-1.0 * cos(theta)), 0,
		   0, 0, 0;
	return tmp;
}

Vector3d ObjectiveAssign::pCalc(const Eigen::VectorXd &x, int locOne, int locTwo) 
{
	MatrixXd pTheta(3,3);

	for(int i = 0; i < x.rows(); i++) {
		MatrixXd tmp = (T[i] * R(x(i)));

		if(i == locOne && i == locTwo) {
			tmp = (T[i] * Rpp(x(i)));
		} else if (i == locOne || i == locTwo) {
			tmp = (T[i] * Rp(x(i)));
		}

		if(i == 0) {
			pTheta = tmp;
		} else {
			pTheta *= tmp;
		}
	}

	return pTheta * r;
}