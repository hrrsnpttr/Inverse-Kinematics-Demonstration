#pragma once
#ifndef OPTIMIZER_GDLS_H
#define OPTIMIZER_GDLS_H

#include "Optimizer.h"

class Objective;

class OptimizerGDLS : public Optimizer
{
public:
	OptimizerGDLS();
	virtual ~OptimizerGDLS();
	virtual Eigen::VectorXd optimize(const std::shared_ptr<Objective> objective, const Eigen::VectorXd &xInit);
	
	void setAlphaInit(double alphaInit) { this->alphaInit = alphaInit; }
	void setGamma(double gamma) { this->gamma = gamma; }
	void setTol(double tol) { this->tol = tol; }
	void setIterMax(int iterMax) { this->iterMax = iterMax; }
	int getIter() const { return iter; }
	void checkDiff(bool chk) { this->check = chk; }
	double getLastF() { return lastF; }
	
private:
	double alphaInit;
	double gamma;
	double tol;
	int iterMax;
	int iter;
	bool check;
	double lastF;
};

#endif
