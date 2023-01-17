#include <assert.h>
#include <iostream>

#define GLEW_STATIC
#include <GL/glew.h>

#include <glm/gtc/type_ptr.hpp>

#include "Link.h"
#include "Shape.h"
#include "MatrixStack.h"
#include "Program.h"

using namespace std;
using namespace Eigen;

Link::Link() :
	parent(NULL),
	child(NULL),
	depth(0),
	position(0.0, 0.0),
	angle(0.0),
	meshMat(Matrix4d::Identity())
{
	
}

Link::~Link()
{
	
}

void Link::addChild(shared_ptr<Link> child)
{
	child->parent = shared_from_this();
	child->depth = depth + 1;
	this->child = child;
}

void Link::computeMeshMatrix()
{
	Eigen::Matrix4d tmp;
    tmp << cos(angle), (-1.0 * sin(angle)), 0, (getPosition()(0)), 
		   sin(angle), cos(angle), 0, (getPosition()(1)), 
		   0, 0, 1, 0, 
		   0, 0, 0, 1;
	setMeshMatrix(tmp);

	// Print new meshMat
	//cout << meshMat << endl << endl;
}

void Link::updateLinks()
{
	// Create Own meshMat
	this->computeMeshMatrix();

	if(child != NULL) {child->updateLinks(); }	
}

void Link::draw(const shared_ptr<Program> prog, shared_ptr<MatrixStack> MV, const shared_ptr<Shape> shape) const
{
	assert(prog);
	assert(MV);
	assert(shape);
	
	MV->pushMatrix();
	
	MV->multMatrix(meshMat);

	MV->pushMatrix();
	MV->translate(0.5, 0.0, 0.0);

	glUniformMatrix4fv(prog->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
	shape->draw();

	MV->popMatrix();
	if(child != NULL) { child->draw(prog, MV, shape); }
	
	MV->popMatrix();
}
