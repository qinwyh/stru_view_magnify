#pragma once
// ȫ�ֱ���

#ifndef DRAW_GRAPH_2D
#define EPSINON 0.00001
#define REAL_AXIS 200.0
#define PI 3.14159
#define MYINFINITY 1000000

#include <stdio.h> 
#include <stdlib.h>



// ��������
// ����¼�
void mouse(int button, int state, int x, int y);

// ͼ�δ��ڳ�ʼ��
void windowInit();

// �������ж���
void drawVertex();

// �������б�
void drawEdge();

// ��ȡͼ������
void readData();

// ����ת������
void dataTrans();

// ��ʾ����
void myDisplay();

// �����µ���ʾλ��
void calculateNewPos();

// ����TargetGraph�ĺ���
void calTargetGraph();

// �����ݶȷ�����µĵ㼯
void conjGradientSolver();

#endif // !1

