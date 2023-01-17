#include <iostream>
#include <vector>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <Eigen/Dense>

#include "GLSL.h"
#include "Program.h"
#include "MatrixStack.h"
#include "Shape.h"
#include "Link.h"
#include "Texture.h"

#include "ObjectiveAssign.h"
#include "OptimizerGDLS.h"
#include "OptimizerNM.h"

using namespace std;
using namespace glm;
using namespace Eigen;

bool keyToggles[256] = {false}; // only for English keyboards!

GLFWwindow *window; // Main application window
string RESOURCE_DIR = ""; // Where the resources are loaded from

shared_ptr<Program> progSimple;
shared_ptr<Program> progTex;
shared_ptr<Shape> shape;
shared_ptr<Texture> texture;

const int NLINKS = 8; // <----- Change Number of Links
vector<shared_ptr<Link> > links;

shared_ptr<ObjectiveAssign> objective;
shared_ptr<OptimizerGDLS> optGDLS;
shared_ptr<OptimizerNM> optNM;

VectorXd thetas(NLINKS);

double w_tar = 1e3;
double w_reg = 1e0;
int iterMax = 10 * NLINKS;
double alphaInit = 1e-4;
double tol = 1e-3;

bool checkDiff = false;

static void error_callback(int error, const char *description)
{
	cerr << description << endl;
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
	if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}

	if(key == GLFW_KEY_D && action == GLFW_PRESS) {
		checkDiff = !checkDiff;
		optGDLS->checkDiff(checkDiff);
		optNM->checkDiff(checkDiff);
	}
}

static void char_callback(GLFWwindow *window, unsigned int key)
{
	keyToggles[key] = !keyToggles[key];
	switch(key) {
		case 'r':
			// Reset all angles to 0.0
			for(int i = 0; i < NLINKS; ++i) {
				links[i]->setAngle(0.0);
			}
			links[0]->updateLinks();
			break;
		case '.':
			// Increment all angles
			if(!keyToggles[(unsigned)' ']) {
				for(int i = 0; i < NLINKS; ++i) {
					links[i]->setAngle(links[i]->getAngle() + 0.1);
				}
				links[0]->updateLinks();
			}
			break;
		case ',':
			// Decrement all angles
			if(!keyToggles[(unsigned)' ']) {
				for(int i = 0; i < NLINKS; ++i) {
					links[i]->setAngle(links[i]->getAngle() - 0.1);
				}
				links[0]->updateLinks();
			}
			break;
	}
}

static void cursor_position_callback(GLFWwindow* window, double xmouse, double ymouse)
{
	// Get current window size.
	int width, height;
	glfwGetWindowSize(window, &width, &height);
	// Convert from window coord to world coord assuming that we're
	// using an orthgraphic projection
	double aspect = (double)width/height;
	double ymax = NLINKS;
	double xmax = aspect*ymax;
	Vector2d x;
	x(0) = 2.0 * xmax * ((xmouse / width) - 0.5);
	x(1) = 2.0 * ymax * (((height - ymouse) / height) - 0.5);
	if(keyToggles[(unsigned)' ']) {
		//cout << x << endl;
		objective->setpTarget(x);
		VectorXd gdlsThetas = optGDLS->optimize(objective, thetas);
		VectorXd nmThetas = optNM->optimize(objective, gdlsThetas);
		if(optNM->getLastF() < optGDLS->getIter()) {
			thetas = nmThetas;
		} else {
			thetas = gdlsThetas;
		}
		for(int i = 0; i < NLINKS; i++) {
			while(thetas(i) > M_PI){ thetas(i) -= 2.0*M_PI; }
			while(thetas(i) < -1.0*M_PI){ thetas(i) += 2.0*M_PI; }
			links[i]->setAngle(thetas(i));
		}
		links[0]->updateLinks();
	}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	// Get the current mouse position.
	double xmouse, ymouse;
	glfwGetCursorPos(window, &xmouse, &ymouse);
}

static void init()
{
	GLSL::checkVersion();
	
	// Set background color
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	// Enable z-buffer test
	glEnable(GL_DEPTH_TEST);
	
	keyToggles[(unsigned)'c'] = true;
	
	progSimple = make_shared<Program>();
	progSimple->setShaderNames(RESOURCE_DIR + "simple_vert.glsl", RESOURCE_DIR + "simple_frag.glsl");
	progSimple->setVerbose(true); // Set this to true when debugging.
	progSimple->init();
	progSimple->addUniform("P");
	progSimple->addUniform("MV");
	progSimple->setVerbose(false);
	
	progTex = make_shared<Program>();
	progTex->setVerbose(true); // Set this to true when debugging.
	progTex->setShaderNames(RESOURCE_DIR + "tex_vert.glsl", RESOURCE_DIR + "tex_frag.glsl");
	progTex->init();
	progTex->addUniform("P");
	progTex->addUniform("MV");
	progTex->addAttribute("aPos");
	progTex->addAttribute("aTex");
	progTex->addUniform("texture0");
	progTex->setVerbose(false);
	
	texture = make_shared<Texture>();
	texture->setFilename(RESOURCE_DIR + "metal_texture_15_by_wojtar_stock.jpg");
	texture->init();
	texture->setUnit(0);
	
	shape = make_shared<Shape>();
	shape->loadMesh(RESOURCE_DIR + "link.obj");
	shape->setProgram(progTex);
	shape->init();
	
	// Initialize time.
	glfwSetTime(0.0);
	
	// If there were any OpenGL errors, this will print something.
	// You can intersperse this line in your code to find the exact location
	// of your OpenGL error.
	GLSL::checkError(GET_FILE_LINE);
	
	// Create the links
	for(int i = 0; i < NLINKS; i++) {
		links.push_back(make_shared<Link>());
		if(i > 0) {
			links[i]->setPosition(1, 0);
		} else {
			links[i]->setPosition(0, 0);
		}
		if(i > 0) { links[i - 1]->addChild(links[i]); }
		thetas(i) = 0.0;
	}
	links[0]->updateLinks();

	// Initialize objective and optimizers
	objective = make_shared<ObjectiveAssign>();
	objective->setWtar(w_tar);
	objective->setWreg(w_reg);
	vector<Matrix3d> T;
	for(int i = 0; i < NLINKS; i++) {
		Matrix3d tmp;
		tmp << 1, 0, (links[i]->getPosition()(0)), 
		       0, 1, (links[i]->getPosition()(1)), 
		       0, 0, 1;
		T.push_back(tmp);
	}
	objective->setT(T);

	optGDLS = make_shared<OptimizerGDLS>();
	optGDLS->setAlphaInit(alphaInit);
	optGDLS->setIterMax(iterMax);
	optGDLS->setTol(tol);

	optNM = make_shared<OptimizerNM>();
	optNM->setIterMax(iterMax);
	optNM->setTol(tol);
}

void render()
{
	// Get current frame buffer size.
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	glViewport(0, 0, width, height);
	
	// Clear buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if(keyToggles[(unsigned)'c']) {
		glEnable(GL_CULL_FACE);
	} else {
		glDisable(GL_CULL_FACE);
	}
	if(keyToggles[(unsigned)'l']) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	} else {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
	
	auto P = make_shared<MatrixStack>();
	auto MV = make_shared<MatrixStack>();
	P->pushMatrix();
	MV->pushMatrix();
	
	// Apply camera transforms
	double aspect = (double)width/height;
	double ymax = NLINKS;
	double xmax = aspect*ymax;
	P->multMatrix(glm::ortho(-xmax, xmax, -ymax, ymax, -1.0, 1.0));
	
	// Draw grid
	progSimple->bind();
	glUniformMatrix4fv(progSimple->getUniform("P"), 1, GL_FALSE, glm::value_ptr(P->topMatrix()));
	glUniformMatrix4fv(progSimple->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
	// Draw axes
	glLineWidth(2.0f);
	glColor3d(0.2, 0.2, 0.2);
	glBegin(GL_LINES);
	glVertex2d(-xmax, 0.0);
	glVertex2d( xmax, 0.0);
	glVertex2d(0.0, -ymax);
	glVertex2d(0.0,  ymax);
	glEnd();
	// Draw grid lines
	glLineWidth(1.0f);
	glColor3d(0.8, 0.8, 0.8);
	glBegin(GL_LINES);
	for(int x = 1; x < xmax; ++x) {
		glVertex2d( x, -ymax);
		glVertex2d( x,  ymax);
		glVertex2d(-x, -ymax);
		glVertex2d(-x,  ymax);
	}
	for(int y = 1; y < ymax; ++y) {
		glVertex2d(-xmax,  y);
		glVertex2d( xmax,  y);
		glVertex2d(-xmax, -y);
		glVertex2d( xmax, -y);
	}
	glEnd();
	progSimple->unbind();

	// Draw shape
	progTex->bind();
	texture->bind(progTex->getUniform("texture0"));
	glUniformMatrix4fv(progTex->getUniform("P"), 1, GL_FALSE, glm::value_ptr(P->topMatrix()));
	MV->pushMatrix();
	// TODO: draw the links recursively
	links[0]->draw(progTex, MV, shape);
	MV->popMatrix();
	texture->unbind();
	progTex->unbind();
	
	//////////////////////////////////////////////////////
	// Cleanup
	//////////////////////////////////////////////////////
	
	// Pop stacks
	MV->popMatrix();
	P->popMatrix();
	
	GLSL::checkError(GET_FILE_LINE);
}

int main(int argc, char **argv)
{
	if(argc < 2) {
		cout << "Please specify the resource directory." << endl;
		return 0;
	}
	RESOURCE_DIR = argv[1] + string("/");
	
	// Set error callback.
	glfwSetErrorCallback(error_callback);
	// Initialize the library.
	if(!glfwInit()) {
		return -1;
	}
	// Create a windowed mode window and its OpenGL context.
	window = glfwCreateWindow(640, 480, "Harrison Potter - A4", NULL, NULL);
	if(!window) {
		glfwTerminate();
		return -1;
	}
	// Make the window's context current.
	glfwMakeContextCurrent(window);
	// Initialize GLEW.
	glewExperimental = true;
	if(glewInit() != GLEW_OK) {
		cerr << "Failed to initialize GLEW" << endl;
		return -1;
	}
	glGetError(); // A bug in glewInit() causes an error that we can safely ignore.
	cout << "OpenGL version: " << glGetString(GL_VERSION) << endl;
	cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;
	// Set vsync.
	glfwSwapInterval(1);
	// Set keyboard callback.
	glfwSetKeyCallback(window, key_callback);
	// Set char callback.
	glfwSetCharCallback(window, char_callback);
	// Set cursor position callback.
	glfwSetCursorPosCallback(window, cursor_position_callback);
	// Set mouse button callback.
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	// Initialize scene.
	init();
	// Loop until the user closes the window.
	while(!glfwWindowShouldClose(window)) {
		if(!glfwGetWindowAttrib(window, GLFW_ICONIFIED)) {
			// Render scene.
			render();
			// Swap front and back buffers.
			glfwSwapBuffers(window);
		}
		// Poll for and process events.
		glfwPollEvents();
	}
	// Quit program.
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
