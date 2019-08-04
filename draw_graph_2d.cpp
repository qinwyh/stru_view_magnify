#pragma once

// 头文件
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif


#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <draw_graph_2d.h>
#include <fstream> 


using namespace std;

//const float testCons = 0.842;
const float pointSize = 4.0;
//const float edgeWidth = 1.0;
const int   windowSize = 800;

// 放大倍数
//float m_factor = 2.0;
//
//int vertexNum = 17;
//int edgeNum = 19;
float m_factor = 2.0;

int vertexNum = 0;
int edgeNum = 0;

// 初始坐标数据
//float vertexX[] = { -120.0, -120.0, -55.0, -55.0, -114.0, -49.0, -70.0, -45.0, -20.0, -46.0, -46.0, 35.0,  -12.5, -12.5, 35.0,   82.5,  82.5 };
//float vertexY[] = { 0.0,    -60.0,  -60.0, 0.0,   10.0,   10.0,  50.0,  90.0,  50.0,  11.0,  -49.0, 10.0, -17.5, -72.5, -100.0, -72.5, -17.5 };
float *vertexX, *vertexY;

// 修正坐标数据
float *posX, *posY;
float *resultPosX, *resultPosY;
float *targetVertexX, *targetVertexY;

// 焦点坐标
float focalPointX, focalPointY;

// 是否处于放大状态
BOOLEAN isMagnified = FALSE;

BOOLEAN *edge;

// 共轭梯度法所使用的向量与矩阵
float *matrixA;
float *vectorB_X, *vectorB_Y;
float *cons_vectorB_X, *cons_vectorB_Y;

//int tmpEdge[19][2] = { {0, 1}, {0, 3}, {0, 5}, {1, 2}, {2, 3}, {3, 4}, {4, 6}, {5, 6}, {6, 7}, {7, 8}, {8, 9},
//					   {8, 11}, {9, 10}, {11, 12}, {11, 16} , {12, 13}, {13, 14}, {14, 15}, {15, 16}};
int *tmpEdge;


// 初始化背景颜色的函数
void windowInit() {
	readData();

	// 设置窗口大小
	glutInitWindowPosition(100, 0);
	glutInitWindowSize(windowSize, windowSize);
	glutCreateWindow("Fisheye Views");

	// 设置背景颜色
	glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
}

// 绘制所有顶点的函数
void drawVertex() {

	glPointSize(pointSize);
	// 进行平滑处理　
	glEnable(GL_POINT_SMOOTH);
	glHint(GL_POINT_SMOOTH, GL_NICEST);
	// 设置顶点颜色为蓝色
	glColor3f(0.0f, 0.0f, 1.0f);


	glBegin(GL_POINTS);
	if (!isMagnified) {
		for (int i = 0; i < vertexNum; i++) {
			glVertex2f(posX[i], posY[i]);
			resultPosX[i] = posX[i];
			resultPosY[i] = posY[i];
			
		}
	}
	else {
		//printf("is magnified!\n");
		//printf("resultPosX 0 is %6.2f\n", resultPosX[0]);
		for (int i = 0; i < vertexNum; i++) {
			glVertex2f(resultPosX[i], resultPosY[i]);

		}

		// 修改焦点颜色
		glColor3f(1.0f, 0.0f, 0.0f);
		glVertex2f(focalPointX, focalPointY);
	}
	glEnd();
}

// 绘制所有边的函数
void drawEdge() {
	//glLineWidth(edgeWidth);
	// 进行平滑处理　
	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH, GL_NICEST);
	glColor3f(0.0f, 0.0f, 0.0f);
	glBegin(GL_LINES);

	if (!isMagnified) {
		for (int i = 0; i < vertexNum * vertexNum; i++) {
			if (edge[i] == TRUE) {
				glVertex2f(posX[i / vertexNum], posY[i / vertexNum]);
				glVertex2f(posX[i % vertexNum], posY[i % vertexNum]);
			}
		}
	}
	else {
		for (int i = 0; i < vertexNum * vertexNum; i++) {
			if (edge[i] == TRUE) {
				glVertex2f(resultPosX[i / vertexNum], resultPosY[i / vertexNum]);
				glVertex2f(resultPosX[i % vertexNum], resultPosY[i % vertexNum]);
			}
		}
	}

	glEnd();
}

// 读入所有数据的函数
void readData() {
	// 点集数据与边集数据
	// 读取边集数据
	ifstream infileEdge, infileNode; 
	int src, des;
	int maxNodeIndex = 0;
	int cur = 0;

	infileEdge.open("D:\\MagnifyDataset\\presentation.txt", ios::in);
	//infileEdge.open("D:\\MagnifyDataset\\facebook.txt", ios::in);
	if (!infileEdge.is_open()) {
		cout << "Open edge file failure" << endl;
		exit(0);
	}

	// 读取第一行边的总数
	infileEdge >> edgeNum;

	//printf("edgeNum is %d\n", edgeNum);
	tmpEdge = new int[edgeNum * 2];

	// 读取每一条边的数据
	while (!infileEdge.eof())
	{
		infileEdge >> src >> des;
		//printf("src is %d, des is %d\n", src, des);

		tmpEdge[cur++] = src;
		tmpEdge[cur++] = des;

		maxNodeIndex = (maxNodeIndex > src) ? maxNodeIndex : src;
		maxNodeIndex = (maxNodeIndex > des) ? maxNodeIndex : des;
	}

	// 关闭线集文件
	infileEdge.close();


	// 尝试打开点坐标数据集
	infileNode.open("D:\\MagnifyDataset\\presentation_node.txt", ios::_Nocreate);
	//infileNode.open("D:\\MagnifyDataset\\hhhhh.txt", ios::_Nocreate);
	//if (!infileNode.fail()) {
	if (!infileNode) {
		vertexNum = maxNodeIndex + 1;
		vertexX = new float[vertexNum];
		vertexY = new float[vertexNum];

		// 随机生成坐标数据
		srand((int)time(0));

		for (int i = 0; i < vertexNum; i++) {
			vertexX[i] = (rand() / double(RAND_MAX) - 0.5) * 360;
			vertexY[i] = (rand() / double(RAND_MAX) - 0.5) * 360;
		}
	}
	else {
		cur = 0;

		printf("测试14...........\n");
		// 读取第一行边的总数
		infileNode >> vertexNum;
		vertexX = new float[vertexNum];
		vertexY = new float[vertexNum];

		while (!infileNode.eof())
		{
			infileNode >> vertexX[cur];
			infileNode >> vertexY[cur++];
			printf("vertexX is %5.2f, vertexY is %5.2f\n", vertexX[cur - 1], vertexY[cur-1]);
		}
	}

	infileNode.close();

	printf("vertexNum is %d\n", vertexNum);

	dataTrans();

}

// 数据转换函数
void dataTrans() {

	// 显示用
	posX = new float[vertexNum];
	posY = new float[vertexNum];
	resultPosX = new float[vertexNum];
	resultPosY = new float[vertexNum];
	targetVertexX = new float[vertexNum];
	targetVertexY = new float[vertexNum];

	// 申请共轭梯度法的运算矩阵与向量
	matrixA = new float[vertexNum * vertexNum];
	vectorB_X = new float[vertexNum];
	vectorB_Y = new float[vertexNum];
	cons_vectorB_X = new float[vertexNum];
	cons_vectorB_Y = new float[vertexNum];


	// 点集坐标变换
	for (int i = 0; i < vertexNum; i++) {
		if (fabs(vertexX[i]) < 181) {
			posX[i] = vertexX[i] / 200.0;
		}
		else if (vertexX[i] > 0) {
			posX[i] = 0.9;
		}
		else {
			posX[i] = -0.9;
		}

		if (fabs(vertexY[i]) < 200) {
			posY[i] = vertexY[i] / 200.0;
		}
		else if (vertexY[i] > 0) {
			posY[i] = 0.9;
		}
		else {
			posY[i] = -0.9;
		}

		resultPosX[i] = posX[i];
		resultPosY[i] = posY[i];
	}

	// 边集存储
	edge = new BOOLEAN[vertexNum * vertexNum];

	for (int j = 0; j < edgeNum; j++) {
		if (tmpEdge[j * 2 + 0] > tmpEdge[j * 2 + 1]) {
			edge[tmpEdge[j * 2 + 1] * vertexNum + tmpEdge[j * 2 + 0]] = TRUE;
		}
		else {
			edge[tmpEdge[j * 2 + 0] * vertexNum + tmpEdge[j * 2 + 1]] = TRUE;
		}
	}
}


// Graph显示函数
void myDisplay() {

	int times = 1;

	if (isMagnified) {
		calculateNewPos();
		times = 30;
	}

	for (int i = 0; i < times; i++) {
		if (isMagnified) {
			conjGradientSolver();
		}

		Sleep(10);

		// 清屏
		glClear(GL_COLOR_BUFFER_BIT);

		glFlush();

		drawEdge();

		drawVertex();

		glFlush();

		printf("display function times %d\n", i);
	}
}

// 鼠标点击事件
void mouse(int button, int state, int x, int y) {

	if (state == GLUT_DOWN)
	{
		if (button == GLUT_LEFT_BUTTON)
		{
			printf("x is %d, y is %d\n", x, y);

			// 记录焦点位置
			focalPointX = ((float)x / (39 * windowSize / 80)) - 1.0;
			focalPointY = 1.0 - ((float)y / (39 * windowSize / 80));

			// 改变图像状态
			isMagnified = TRUE;

			char title[256];
			sprintf(title, "Fisheye View (%3.1f times)", m_factor);
			glutSetWindowTitle(title);

			glFlush();

		}
		else if (button == GLUT_RIGHT_BUTTON)
		{
			// 回到初始状态
			isMagnified = FALSE;
			glutSetWindowTitle("Fisheye View (original)");

			focalPointX = -46.0 / 200.0;
			focalPointY = 11.0 / 200.0;

			/*glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT);*/
			glFlush();

		}

	}
	return;

}