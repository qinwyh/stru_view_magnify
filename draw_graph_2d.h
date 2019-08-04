#pragma once
// 全局变量

#ifndef DRAW_GRAPH_2D
#define EPSINON 0.00001
#define REAL_AXIS 200.0
#define PI 3.14159
#define MYINFINITY 1000000

#include <stdio.h> 
#include <stdlib.h>



// 函数声明
// 鼠标事件
void mouse(int button, int state, int x, int y);

// 图形窗口初始化
void windowInit();

// 绘制所有顶点
void drawVertex();

// 绘制所有边
void drawEdge();

// 读取图像数据
void readData();

// 数据转换函数
void dataTrans();

// 显示函数
void myDisplay();

// 计算新的显示位置
void calculateNewPos();

// 计算TargetGraph的函数
void calTargetGraph();

// 共轭梯度法求解新的点集
void conjGradientSolver();

#endif // !1

