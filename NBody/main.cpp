#include"Game.cu"

int main() {
	Game g(800, 800, 80, 80);
	g.loadShader();
	g.setPoints();
	g.openGLInits();
	g.loop();
}