#include"Game.cu"

int main() {
	Game g(800, 800, 50, 50);
	g.loadShader();
	g.setPoints();
	g.openGLInits();
	g.loop();
}