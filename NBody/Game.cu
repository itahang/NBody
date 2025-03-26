#ifndef GAME_PART
#define GAME_PART

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include<iostream>
#include<random>
#include<thread>
#include<functional>



#include <cuda_gl_interop.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include"GPUPart.cuh"
#include"Shaders.h"
#include"Body.cu"

#define SOME_ERROR -1


class Game
{
private:


	int init() {
		if (!glfwInit()) return SOME_ERROR;
		window = glfwCreateWindow(width, height, "N Body", NULL, NULL);
		if (!window) {
			glfwTerminate();
			return SOME_ERROR;
		}




		glfwMakeContextCurrent(window);
		glewInit();
		glfwSwapInterval(1);
		size = Nrow * Ncol;

		glfwSwapInterval(1);
		cudaSetDevice(0);
		cudaFree(0);
		return 0;

	}



	int height;
	int width;
	Body* points;
	GLuint shader;


	size_t Nrow = 100;
	size_t Ncol = 100;

	//GLFWmonitor* monitor = glfwGetPrimaryMonitor();
	//const GLFWvidmode* mode = glfwGetVideoMode(monitor);


	GLFWwindow* window = nullptr;
	GLuint VAO, VBO;
	cudaGraphicsResource* cudaVBO;

	int size;
public:
	Game(int height, int width) :height(height), width(width) {
		if (init() == SOME_ERROR) {
			std::cerr << "Error while opening the windows";
			exit(-1);
		}
	}

	Game(int height, int width, size_t Nrows, size_t Ncols) :height(height), width(width), Nrow(Nrows), Ncol(Ncols) {
		if (init() == SOME_ERROR) {
			std::cerr << "Error while opening the windows";
			exit(-1);
		}
	}

	size_t getNrows() {
		return Nrow;
	}
	size_t getNcols() {
		return Ncol;
	}

	void loadShader() {
		shader = Shaders::createShaderProgram();
		glUseProgram(shader);
	}

	void callGlfwTerminate() {
		glfwTerminate();
	}

	void setPoints() {
		points = new Body[size];
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<float> dist(0.0, 0.10f);
		std::normal_distribution<float> dist2(0.0, 1.0f);
		std::uniform_real_distribution<float> uni(0.0, 1.0f);


		points[0].position = { dist(gen),dist(gen) };
		points[0].velocity = { dist(gen),dist(gen) };
		points[0].acceleration = { 0,0 };
		points[0].prev_position = { 0,0 };
		points[0].mass = 10000000 ;


		for (int i = 1; i < size / 2; i += 1) {
			points[i].position = { dist(gen),dist(gen) };
			points[i].velocity = { dist(gen),dist(gen) };
			points[i].acceleration = { 0,0 };
			points[i].prev_position = { 0,0 };
			points[i].mass = 100000 * uni(gen);
		}


		for (int i = static_cast<int>(size / 2); i < size; i += 1) {
			points[i].position = { dist2(gen),dist2(gen) };
			points[i].velocity = { dist(gen),dist(gen) };
			points[i].acceleration = { 0,0 };
			points[i].mass = 100000 * uni(gen);
		}



		std::cout << size << " " << sizeof(Body) << " " << sizeof(float2);


	}

	void openGLInits() {
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);

		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, (sizeof(Body)) * size, points, GL_DYNAMIC_DRAW);

		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Body), (void*)offsetof(Body, position));
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Body), (void*)offsetof(Body, velocity));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Body), (void*)offsetof(Body, acceleration));
		glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(Body), (void*)offsetof(Body, prev_position));
		glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(Body), (void*)offsetof(Body, mass));

		glEnableVertexAttribArray(0);
		glPointSize(2.0f);

		cudaGraphicsGLRegisterBuffer(&cudaVBO, VBO, cudaGraphicsRegisterFlagsWriteDiscard);
		cudaError_t launchErr = cudaGetLastError();

		if (launchErr != cudaSuccess) {
			std::cerr << "Kernel launch error: " << cudaGetErrorString(launchErr) << std::endl;

		}
	}

	void render() {
		Body* d_pixels = nullptr;
		size_t num_bytes = 0;

		glFinish();

		cudaError_t err = cudaGraphicsMapResources(1, &cudaVBO, 0);
		if (err != cudaSuccess) {
			std::cerr << "MapResources failed: " << cudaGetErrorString(err) << std::endl;
			return;
		}

		err = cudaGraphicsResourceGetMappedPointer((void**)&d_pixels, &num_bytes, cudaVBO);
		if (err != cudaSuccess) {
			std::cerr << "GetMappedPointer failed: " << cudaGetErrorString(err);
			cudaGraphicsUnmapResources(1, &cudaVBO, 0);
			return;
		}

		dim3 threads(32, 32);
		dim3 blocks((Ncol + threads.x - 1) / threads.x, (Nrow + threads.y - 1) / threads.y);

		kernel << <blocks, threads >> > (d_pixels, Ncol, Nrow);
		changeMean << <blocks, threads >> > (d_pixels, Ncol, Nrow);

		cudaError_t launchErr = cudaGetLastError();
		cudaError_t syncErr = cudaDeviceSynchronize();

		if (launchErr != cudaSuccess) {
			std::cerr << "Kernel launch error: " << cudaGetErrorString(launchErr) << std::endl;
		}
		if (syncErr != cudaSuccess) {
			std::cerr << "Kernel execution error: " << cudaGetErrorString(syncErr) << std::endl;
		}

		err = cudaGraphicsUnmapResources(1, &cudaVBO, 0);

	}


	void loop() {
		while (!glfwWindowShouldClose(window)) {
			glClear(GL_COLOR_BUFFER_BIT);
			render();
			glUseProgram(shader);
			glBindVertexArray(VAO);
			glDrawArrays(GL_POINTS, 0, size);
			glfwSwapBuffers(window);
			glfwPollEvents();
		}

	}

	~Game() {
		glDeleteVertexArrays(1, &VAO);
		glDeleteBuffers(1, &VBO);
		glDeleteProgram(shader);
		delete[] points;
		glfwTerminate();
	}
};




#endif // !GAME_PART


