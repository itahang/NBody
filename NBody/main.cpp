#include"Game.cu"

int main() {
	Game g(800, 800, 100, 100);
	g.loadShader();
	g.setPoints();
	g.openGLInits();
	g.loop();
}