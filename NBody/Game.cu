#ifndef GAME_PART
#define GAME_PART

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include<iostream>
#include<random>

#include <cuda_gl_interop.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include"GPUPart.cuh"
#include"Shaders.h"
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
		size = width * height;

		glfwSwapInterval(1);
		cudaSetDevice(0);
		cudaFree(0);
		return 0;

	}



	int height;
	int width;
	float* points;
	GLuint shader;

	size_t Nrow = 100;
	size_t Ncol = 100;

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
		points = new float[size * 2];
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<float> dist(0.0, 0.10f);
		std::normal_distribution<float> dist2(0.0, 0.50f);


		for (int i = 0; i < size; i += 1) {
			points[i] = dist(gen);
		}

		for (int i = size; i < size * 2; i += 1) {
			points[i] = dist2(gen);
		}

	}

	void openGLInits() {
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);

		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * size * 2, points, GL_DYNAMIC_DRAW);

		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);
		glPointSize(1.0f);

	}

	void render() {
		float2* d_pixels = nullptr;
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
		dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
		std::cout << "Launching " << blocks.x << " blocks of " << threads.x << " threads" << std::endl;

		kernel << <blocks, threads >> > (d_pixels, width, height);

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
		cudaGraphicsGLRegisterBuffer(&cudaVBO, VBO, cudaGraphicsRegisterFlagsWriteDiscard);
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


