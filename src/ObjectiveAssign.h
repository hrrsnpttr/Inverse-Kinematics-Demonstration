#pragma once
#ifndef OBJECTIVE_Assign_H
#define OBJECTIVE_Assign_H

#include "Objective.h"

#include <vector>
#include "Link.h"

class ObjectiveAssign : public Objective
{
public:
	ObjectiveAssign();
	virtual ~ObjectiveAssign();
	virtual double evalObjective(const Eigen::VectorXd &x);
	virtual double evalObjective(const Eigen::VectorXd &x, Eigen::VectorXd &g);
	virtual double evalObjective(const Eigen::VectorXd &x, Eigen::VectorXd &g, Eigen::MatrixXd &H);
	Eigen::Vector3d pCalc(const Eigen::VectorXd &x, int locOne, int locTwo);
	Eigen::Matrix3d R(double theta);
	Eigen::Matrix3d Rp(double theta);
	Eigen::Matrix3d Rpp(double theta);
	
	void setpTarget(Eigen::Vector2d target) { pTarget = target; }
	void setWtar(double wTar) { w_tar = wTar; }
	void setWreg(double wReg) { w_reg = wReg; }
	void setT(std::vector<Eigen::Matrix3d> inputT) { T = inputT; }
private:
	Eigen::Vector3d r;
	Eigen::Vector2d pTarget;
	std::vector<Eigen::Matrix3d> T; 
	double w_tar;
	double w_reg;
};

#endif
