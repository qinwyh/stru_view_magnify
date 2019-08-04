#pragma once

// ͷ�ļ�
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

// �Ŵ���
//float m_factor = 2.0;
//
//int vertexNum = 17;
//int edgeNum = 19;
float m_factor = 2.0;

int vertexNum = 0;
int edgeNum = 0;

// ��ʼ��������
//float vertexX[] = { -120.0, -120.0, -55.0, -55.0, -114.0, -49.0, -70.0, -45.0, -20.0, -46.0, -46.0, 35.0,  -12.5, -12.5, 35.0,   82.5,  82.5 };
//float vertexY[] = { 0.0,    -60.0,  -60.0, 0.0,   10.0,   10.0,  50.0,  90.0,  50.0,  11.0,  -49.0, 10.0, -17.5, -72.5, -100.0, -72.5, -17.5 };
float *vertexX, *vertexY;

// ������������
float *posX, *posY;
float *resultPosX, *resultPosY;
float *targetVertexX, *targetVertexY;

// ��������
float focalPointX, focalPointY;

// �Ƿ��ڷŴ�״̬
BOOLEAN isMagnified = FALSE;

BOOLEAN *edge;

// �����ݶȷ���ʹ�õ����������
float *matrixA;
float *vectorB_X, *vectorB_Y;
float *cons_vectorB_X, *cons_vectorB_Y;

//int tmpEdge[19][2] = { {0, 1}, {0, 3}, {0, 5}, {1, 2}, {2, 3}, {3, 4}, {4, 6}, {5, 6}, {6, 7}, {7, 8}, {8, 9},
//					   {8, 11}, {9, 10}, {11, 12}, {11, 16} , {12, 13}, {13, 14}, {14, 15}, {15, 16}};
int *tmpEdge;


// ��ʼ��������ɫ�ĺ���
void windowInit() {
	readData();

	// ���ô��ڴ�С
	glutInitWindowPosition(100, 0);
	glutInitWindowSize(windowSize, windowSize);
	glutCreateWindow("Fisheye Views");

	// ���ñ�����ɫ
	glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
}

// �������ж���ĺ���
void drawVertex() {

	glPointSize(pointSize);
	// ����ƽ������
	glEnable(GL_POINT_SMOOTH);
	glHint(GL_POINT_SMOOTH, GL_NICEST);
	// ���ö�����ɫΪ��ɫ
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

		// �޸Ľ�����ɫ
		glColor3f(1.0f, 0.0f, 0.0f);
		glVertex2f(focalPointX, focalPointY);
	}
	glEnd();
}

// �������бߵĺ���
void drawEdge() {
	//glLineWidth(edgeWidth);
	// ����ƽ������
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

// �����������ݵĺ���
void readData() {
	// �㼯������߼�����
	// ��ȡ�߼�����
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

	// ��ȡ��һ�бߵ�����
	infileEdge >> edgeNum;

	//printf("edgeNum is %d\n", edgeNum);
	tmpEdge = new int[edgeNum * 2];

	// ��ȡÿһ���ߵ�����
	while (!infileEdge.eof())
	{
		infileEdge >> src >> des;
		//printf("src is %d, des is %d\n", src, des);

		tmpEdge[cur++] = src;
		tmpEdge[cur++] = des;

		maxNodeIndex = (maxNodeIndex > src) ? maxNodeIndex : src;
		maxNodeIndex = (maxNodeIndex > des) ? maxNodeIndex : des;
	}

	// �ر��߼��ļ�
	infileEdge.close();


	// ���Դ򿪵��������ݼ�
	infileNode.open("D:\\MagnifyDataset\\presentation_node.txt", ios::_Nocreate);
	//infileNode.open("D:\\MagnifyDataset\\hhhhh.txt", ios::_Nocreate);
	//if (!infileNode.fail()) {
	if (!infileNode) {
		vertexNum = maxNodeIndex + 1;
		vertexX = new float[vertexNum];
		vertexY = new float[vertexNum];

		// ���������������
		srand((int)time(0));

		for (int i = 0; i < vertexNum; i++) {
			vertexX[i] = (rand() / double(RAND_MAX) - 0.5) * 360;
			vertexY[i] = (rand() / double(RAND_MAX) - 0.5) * 360;
		}
	}
	else {
		cur = 0;

		printf("����14...........\n");
		// ��ȡ��һ�бߵ�����
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

// ����ת������
void dataTrans() {

	// ��ʾ��
	posX = new float[vertexNum];
	posY = new float[vertexNum];
	resultPosX = new float[vertexNum];
	resultPosY = new float[vertexNum];
	targetVertexX = new float[vertexNum];
	targetVertexY = new float[vertexNum];

	// ���빲���ݶȷ����������������
	matrixA = new float[vertexNum * vertexNum];
	vectorB_X = new float[vertexNum];
	vectorB_Y = new float[vertexNum];
	cons_vectorB_X = new float[vertexNum];
	cons_vectorB_Y = new float[vertexNum];


	// �㼯����任
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

	// �߼��洢
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


// Graph��ʾ����
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

		// ����
		glClear(GL_COLOR_BUFFER_BIT);

		glFlush();

		drawEdge();

		drawVertex();

		glFlush();

		printf("display function times %d\n", i);
	}
}

// ������¼�
void mouse(int button, int state, int x, int y) {

	if (state == GLUT_DOWN)
	{
		if (button == GLUT_LEFT_BUTTON)
		{
			printf("x is %d, y is %d\n", x, y);

			// ��¼����λ��
			focalPointX = ((float)x / (39 * windowSize / 80)) - 1.0;
			focalPointY = 1.0 - ((float)y / (39 * windowSize / 80));

			// �ı�ͼ��״̬
			isMagnified = TRUE;

			char title[256];
			sprintf(title, "Fisheye View (%3.1f times)", m_factor);
			glutSetWindowTitle(title);

			glFlush();

		}
		else if (button == GLUT_RIGHT_BUTTON)
		{
			// �ص���ʼ״̬
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