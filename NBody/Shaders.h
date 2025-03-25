#ifndef SHADER_PART
#define SHADER_PART
#endif // !SHADER_PART

#include <GL/glew.h>
#include <iostream>

namespace Shaders

{
	const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;
	out vec4 color;
    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
		color = vec4((gl_Position.x+1),(gl_Position.y+1),0.5,1);
    }
)";

	// Fragment Shader source code
	const char* fragmentShaderSource = R"(
    #version 330 core
    in vec4 color;
    out vec4 FragColor;
    void main() {
        FragColor = color;// White color
    }
)";


	extern "C" GLuint compileShader(GLenum type, const char* source) {
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

	extern "C" GLuint createShaderProgram() {
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

}
