#include"Game.cu"

int main() {
	Game g(800, 800, 1000, 1000);
	g.loadShader();
	g.setPoints();
	g.openGLInits();
	g.loop();
}