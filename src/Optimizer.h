#pragma once
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <memory>
#include <Eigen/Dense>

class Objective;

class Optimizer
{
public:
	Optimizer() {};
	virtual ~Optimizer() {};
	virtual Eigen::VectorXd optimize(const std::shared_ptr<Objective> objective, const Eigen::VectorXd &xInit) = 0;
};

#endif
