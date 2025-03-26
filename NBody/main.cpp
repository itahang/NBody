#include"Game.cu"

int main() {
	Game g(800, 1080, 400, 400);
	g.loadShader();
	g.setPoints();
	g.openGLInits();
	g.loop();
}