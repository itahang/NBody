#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <random>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#define NVERTEX 10000


GLuint VAO, VBO;
cudaGraphicsResource* cudaVBO;

const int width = 1000;
const int height = 1000;

// Vertex Shader source code
const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
)";

// Fragment Shader source code
const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    void main() {
        FragColor = vec4(1.0, 1.0, 1.0, 1.0); // White color
    }
)";

__global__ void kernel(float2* d_pixels) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;

	// Ensure you stay within bounds
	if (idx < width && idy < height) {
		int pixelIndex = idx + width * idy;
		int x = d_pixels[pixelIndex].x;
		int y = d_pixels[pixelIndex].y;

		if (x > 1.0 || y > 1.0) {
			d_pixels[pixelIndex].x = -1.0;
			d_pixels[pixelIndex].y = -1.0;
			return;

		}
		d_pixels[pixelIndex].x += 0.0001f;
		d_pixels[pixelIndex].y += 0.0001f;
	}
}

void render() {
	float2* d_pixels = nullptr;
	size_t num_bytes = 0;

	// 1. Explicit synchronization with OpenGL
	glFinish();

	// 2. Error-checked resource mapping
	cudaError_t err = cudaGraphicsMapResources(1, &cudaVBO, 0);
	if (err != cudaSuccess) {
		std::cerr << "MapResources failed: " << cudaGetErrorString(err) << std::endl;
		return;
	}

	err = cudaGraphicsResourceGetMappedPointer((void**)&d_pixels, &num_bytes, cudaVBO);
	if (err != cudaSuccess || !d_pixels || num_bytes < width * height * sizeof(float)) {
		std::cerr << "GetMappedPointer failed: " << cudaGetErrorString(err)
			<< " | Buffer size: " << num_bytes
			<< " | Required: " << width * height * sizeof(float) << std::endl;
		cudaGraphicsUnmapResources(1, &cudaVBO, 0);
		return;
	}

	// 3. Validate kernel launch parameters
	dim3 threads(32, 32);  // 32x32 threads per block
	dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
	std::cout << "Launching " << blocks.x << " blocks of " << threads.x << " threads" << std::endl;

	kernel << <blocks, threads >> > (d_pixels);

	// 4. Proper error checking sequence
	cudaError_t launchErr = cudaGetLastError();
	cudaError_t syncErr = cudaDeviceSynchronize();

	if (launchErr != cudaSuccess) {
		std::cerr << "Kernel launch error: " << cudaGetErrorString(launchErr) << std::endl;
	}
	if (syncErr != cudaSuccess) {
		std::cerr << "Kernel execution error: " << cudaGetErrorString(syncErr) << std::endl;
	}

	// 5. Clean unmapping
	err = cudaGraphicsUnmapResources(1, &cudaVBO, 0);
}

GLuint compileShader(GLenum type, const char* source) {
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &source, NULL);
	glCompileShader(shader);

	int success;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		char infoLog[512];
		glGetShaderInfoLog(shader, 512, NULL, infoLog);
		std::cout << "Shader Compilation Error:\n" << infoLog << std::endl;
	}
	return shader;
}

GLuint createShaderProgram() {
	GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
	GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	return shaderProgram;
}

int main() {
	if (!glfwInit()) return -1;
	GLFWwindow* window = glfwCreateWindow(1080, 1080, "3 Points in OpenGL", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glewInit();

	glfwSwapInterval(1);
	cudaSetDevice(0);
	cudaFree(0);

	int size = width * height;

	float* points = new float[size * 2];
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> dist(0.0, 0.10f); // Range: [-1,1]
	std::normal_distribution<float> dist2(0.0, 0.50f); // Range: [-1,1]


	for (int i = 0; i < size; i += 1) {
		points[i] = dist(gen); // Random value in [-1,1]
	}

	for (int i = size; i < size * 2; i += 1) {
		points[i] = dist2(gen); // Random value in [-1,1]
	}


	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * size * 2, points, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);


	GLuint shaderProgram = createShaderProgram();
	glUseProgram(shaderProgram);

	// Set point size
	glPointSize(1.0f);

	cudaGraphicsGLRegisterBuffer(&cudaVBO, VBO, cudaGraphicsRegisterFlagsWriteDiscard);

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT);
		render();
		glUseProgram(shaderProgram);
		glBindVertexArray(VAO);
		glDrawArrays(GL_POINTS, 0, size);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	delete[] points;

	// Cleanup
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteProgram(shaderProgram);
	glfwTerminate();
	return 0;
}
