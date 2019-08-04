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
#include <malloc.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>


// 自定义头文件
#include <draw_graph_2d.h>

using namespace std;

// focal area
vector<int> focalArea;


//const float testCons = 0.842;
extern const float pointSize = 4.0;
extern const float edgeWidth = 1.0;
extern const int   windowSize = 800;

// 放大倍数
extern float m_factor;

extern int vertexNum;
extern int edgeNum;
// 初始坐标数据
extern float *vertexX, *vertexY;
// 修正坐标数据
extern float *posX, *posY;
extern float *resultPosX, *resultPosY;
extern float *targetVertexX, *targetVertexY;

// 焦点坐标
extern float focalPointX, focalPointY;

// 是否处于放大状态
extern BOOLEAN isMagnified;

extern BOOLEAN *edge;

// 共轭梯度法所使用的向量与矩阵
extern float *matrixA;
extern float *vectorB_X, *vectorB_Y;
extern float *cons_vectorB_X, *cons_vectorB_Y;

extern int *tmpEdge;


// CUDA函数
// 向量并行求和
cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int size);

// 向量并行求差
cudaError_t minusWithCuda(float *c, const float *a, const float *b, unsigned int size);

// 矩阵并行乘积
cudaError_t mulWithCuda(const float *a, const float *b, float *result, const int M, const int N, const int S);

// 释放所有空间
void freeAll();

// 计算所有constraints
void computeAllCons();


// 共轭梯度法求解结果坐标（测试用）
void conjGradientSolver() {
	//printf("conj!!!!! %6.2f\n", resultPosX[0]);

	// 获取此次time constraints
	for (int i = 0; i < vertexNum; i++) {
		vectorB_X[i] = cons_vectorB_X[i] + resultPosX[i] * 2 * REAL_AXIS;
		vectorB_Y[i] = cons_vectorB_Y[i] + resultPosY[i] * 2 * REAL_AXIS;
	}


	// 初始化残差向量等变量
	float *tmpVector = new float[vertexNum];
	float *tmpX = new float[vertexNum];
	float *tmpY = new float[vertexNum];
	float *vectorD = new float[vertexNum];
	float *vectorR = new float[vertexNum];
	
	::memset(tmpX, 0, vertexNum * sizeof(float));
	::memset(tmpY, 0, vertexNum * sizeof(float));
	::memset(vectorD, 0, vertexNum * sizeof(float));

	// 计算X方向上的向量
	for (int i = 0; i < vertexNum; i++) {
		vectorR[i] = -vectorB_X[i];
	}

	for (int nor = 0; nor < vertexNum * 3; nor++) {
		float denom = 0, num = 0;
		float denom2 = 0, num2 = 0;

		// 计算r的转置乘以r
		/*for (int i = 0; i < vertexNum; i++) {
			denom += vectorR[i] * vectorR[i];
		}*/
		mulWithCuda(vectorR, vectorR, &denom, 1, vertexNum, 1);


		// 计算残差r = Ax-b
		/*for (int i = 0; i < vertexNum; i++) {
			float cur = 0;
			for (int j = 0; j < vertexNum; j++) {
				cur += matrixA[i * vertexNum + j] * tmpX[j];
			}
			vectorR[i] = cur - vectorB_X[i];
		}*/
		mulWithCuda(matrixA, tmpX, tmpVector, vertexNum, vertexNum, 1);
		minusWithCuda(vectorR, tmpVector, vectorB_X, vertexNum);


		//计算r的转置乘以r
		/*for (int i = 0; i < vertexNum; i++) {
			num += vectorR[i] * vectorR[i];
		}*/
		mulWithCuda(vectorR, vectorR, &num, 1, vertexNum, 1);

		if (num < 0.000001) {
			//printf("x is break\n");
			break;
		}

		//计算方向向量d
		for (int i = 0; i < vertexNum; i++) {
			vectorD[i] = -vectorR[i] + num / denom * vectorD[i];
		}

		//计算d的转置乘以r
		/*for (int i = 0; i < vertexNum; i++) {
			num2 += vectorD[i] * vectorR[i];
		}*/
		mulWithCuda(vectorD, vectorR, &num2, 1, vertexNum, 1);

		//计算d的转置乘以A乘以d最后
		/*for (int i = 0; i < vertexNum; i++) {
			float cur = 0;
			for (int j = 0; j < vertexNum; j++) {
				cur += matrixA[i * vertexNum + j] * vectorD[j];
			}
			denom2 += cur * vectorD[i];
		}*/
		mulWithCuda(matrixA, vectorD, tmpVector, vertexNum, vertexNum, 1);
		mulWithCuda(vectorD, tmpVector, &denom2, 1, vertexNum, 1);

		//计算步长
		double a = -num2 / denom2;
		for (int i = 0; i < vertexNum; i++) {
			tmpX[i] += a * vectorD[i];
		}

		//printf("all X with CUDA!\n");
	}


	// 计算y方向上的向量
	::memset(vectorD, 0, vertexNum * sizeof(float));
	for (int i = 0; i < vertexNum; i++) {
		vectorR[i] = -vectorB_Y[i];
	}

	for (int nor = 0; nor < vertexNum * 3; nor++) {
		float denom = 0, num = 0;
		float denom2 = 0, num2 = 0;

		// 计算r的转置乘以r
		mulWithCuda(vectorR, vectorR, &denom, 1, vertexNum, 1);

		// 计算残差r = Ax-b

		mulWithCuda(matrixA, tmpY, tmpVector, vertexNum, vertexNum, 1);
		minusWithCuda(vectorR, tmpVector, vectorB_Y, vertexNum);

		//计算r的转置乘以r
		mulWithCuda(vectorR, vectorR, &num, 1, vertexNum, 1);

		if (num < 0.000001) {
			//printf("y is break\n");
			break;
		}

		//计算方向向量d
		for (int i = 0; i < vertexNum; i++) {
			vectorD[i] = -vectorR[i] + num / denom * vectorD[i];
		}

		//计算d的转置乘以r
		mulWithCuda(vectorD, vectorR, &num2, 1, vertexNum, 1);

		//计算d的转置乘以A乘以d最后
		mulWithCuda(matrixA, vectorD, tmpVector, vertexNum, vertexNum, 1);
		mulWithCuda(vectorD, tmpVector, &denom2, 1, vertexNum, 1);

		//计算步长
		double a = -num2 / denom2;
		for (int i = 0; i < vertexNum; i++) {
			tmpY[i] += a * vectorD[i];
		}

		//printf("all Y with CUDA!\n");
	}

	for (int i = 0; i < vertexNum; i++) {
		resultPosX[i] = tmpX[i] / REAL_AXIS;
		resultPosY[i] = tmpY[i] / REAL_AXIS;
		//printf("resultPosX %d is %6.2f\n", i, resultPosX[i]);
	}

	// free

	//delete struConsVecX, struConsVecY;
	delete tmpX, tmpY, tmpVector;
	delete vectorD;
	delete vectorR;
}


// 计算targetGraph的所有坐标
void calTargetGraph() {

	float disX, disY;
	float focalX = focalPointX * REAL_AXIS;
	float focalY = focalPointY * REAL_AXIS;

	focalArea.clear();

	for (int i = 0; i < vertexNum; i++) {
		disX = posX[i] * REAL_AXIS - focalX;
		disY = posY[i] * REAL_AXIS - focalY;

		// 焦点与某点重合
		if ((fabs(disX) <= EPSINON) && (fabs(disY) <= EPSINON)) {
			targetVertexX[i] = focalX;
			targetVertexY[i] = focalY;
		}
		else if (fabs(disX) <= EPSINON) {
			targetVertexX[i] = focalX;
			float b_iY = (posY[i] > focalPointY) ? 0.9 * REAL_AXIS : -0.9 * REAL_AXIS;
			float b_i = (posY[i] * REAL_AXIS - focalY) / (b_iY - focalY);
			float B_I = (m_factor + 1) * b_i / (m_factor * b_i + 1);

			targetVertexY[i] = focalY + (b_iY - focalY) * B_I;
		}
		else if (fabs(disY) <= EPSINON) {
			targetVertexY[i] = focalY;
			float b_iX = (posX[i] > focalPointX) ? 0.9 * REAL_AXIS : -0.9 * REAL_AXIS;
			float b_i = (posX[i] * REAL_AXIS - focalX) / (b_iX - focalX);
			float B_I = (m_factor + 1) * b_i / (m_factor * b_i + 1);

			targetVertexX[i] = focalX + (b_iX - focalX) * B_I;
		}
		else {
			float b_iX = (disX > 0) ? 0.9 * REAL_AXIS : -0.9 * REAL_AXIS;
			float b_iY = (b_iX - focalX) * disY / disX;

			if (fabs(b_iY) > 0.9 * REAL_AXIS) {
				b_iY = (disY > 0) ? 0.9 * REAL_AXIS : -0.9 * REAL_AXIS;
				b_iX = focalX + ((b_iY - focalY) / (disY / disX));
			}

			float b_i = (posX[i] * REAL_AXIS - focalX) / (b_iX - focalX);
			float B_I = (m_factor + 1) * b_i / (m_factor * b_i + 1);

			targetVertexX[i] = focalX + (b_iX - focalX) * B_I;
			targetVertexY[i] = focalY + (b_iY - focalY) * B_I;
			/*
						printf("focalX is %f, focalY is %f\n", focalX, focalY);
						printf("disX is %f, disY is %f\n", disX, disY);
						targetVertexX[i] = b_iX;
						targetVertexY[i] = b_iY;
			*/
		}

		// 记录focal area里的点集
		float length = sqrt((targetVertexX[i] - focalX) * (targetVertexX[i] - focalX) + 
						    (targetVertexY[i] - focalY) * (targetVertexY[i] - focalY));

		if (length < 0.3 * REAL_AXIS)
			focalArea.push_back(i);

	}

	/*for (int i = 0; i < focalArea.size(); i++) {
		cout <<"focal area: " << focalArea[i] << endl;
	}*/
}

// 计算新位置数据的函数
void calculateNewPos() {

	calTargetGraph();

	computeAllCons();

	//conjGradientSolver();


	/*for (int i = 0; i < vertexNum; i++) {
		if (fabs(targetVertexX[i]) < REAL_AXIS) {
			resultPosX[i] = targetVertexX[i] / REAL_AXIS;
		}
		else if (targetVertexX[i] > 0) {
			resultPosX[i] = 0.9;
		}
		else {
			resultPosX[i] = -0.9;
		}
		if (fabs(targetVertexY[i]) < REAL_AXIS) {
			resultPosY[i] = targetVertexY[i] / REAL_AXIS;
		}
		else if (targetVertexY[i] > 0) {
			resultPosY[i] = 0.9;
		}
		else {
			resultPosY[i] = -0.9;
		}
	}*/

}

// 计算所有constraints
void computeAllCons() {
	//printf("computeAllCons\n");

	::memset(matrixA, 0, vertexNum * vertexNum * sizeof(float));
	::memset(vectorB_X, 0, vertexNum * sizeof(float));
	::memset(vectorB_Y, 0, vertexNum * sizeof(float));

	//float constraints;

	// 遍历边集, 双精度浮点数提高精度
	double struConsX, struConsY, length, e_x, e_y;

	// 测试用
	float w_Stru = 4.0, w_Focal = 2.0, w_Time = 1.0;

	/************************************************/
	// Structure constraints
	/************************************************/
	for (int i = 0; i < vertexNum; i++) {
		for (int j = i; j < vertexNum; j++) {
			if (edge[i * vertexNum + j] == TRUE) {
				matrixA[i * vertexNum + i] += 2.0 * w_Stru;
				matrixA[j * vertexNum + j] += 2.0 * w_Stru;
				matrixA[i * vertexNum + j] -= 2.0 * w_Stru;
				matrixA[j * vertexNum + i] -= 2.0 * w_Stru;
				if (fabs(posX[i] - posX[j]) < EPSINON) {
					struConsX = 0.0;

					//int factor = (posY[i] - posY[j]) ? 1 : -1;
					struConsY = (targetVertexY[i] - targetVertexY[j]) * w_Stru;
					//printf("struConsY%d,%d is %6.2f, %6.2f\n", i, j, targetVertexY[i] - targetVertexY[j], struConsY);
				}
				else {
					double tmpLength = sqrt((posY[i] - posY[j]) * (posY[i] - posY[j]) + (posX[i] - posX[j]) * (posX[i] - posX[j]));
					e_x = (posX[i] - posX[j]) / tmpLength;
					e_y = (posY[i] - posY[j]) / tmpLength;

					length = sqrt((targetVertexX[i] - targetVertexX[j]) * (targetVertexX[i] - targetVertexX[j])
						+ (targetVertexY[i] - targetVertexY[j]) * (targetVertexY[i] - targetVertexY[j]));

					struConsX = e_x * length * w_Stru;
					struConsY = e_y * length * w_Stru;

				}

				vectorB_X[i] += (float)struConsX * 2;
				vectorB_X[j] -= (float)struConsX * 2;
				vectorB_Y[i] += (float)struConsY * 2;
				vectorB_Y[j] -= (float)struConsY * 2;
			}
		}
	}


	/************************************************/
	// Readability constraints
	/************************************************/
	for (int m = 0; m < focalArea.size(); m++) {
		for (int n = m + 1; n < focalArea.size(); n++) {
			int i = focalArea[m];
			int j = focalArea[n];

			matrixA[i * vertexNum + i] += 2.0 * w_Focal;
			matrixA[j * vertexNum + j] += 2.0 * w_Focal;
			matrixA[i * vertexNum + j] -= 2.0 * w_Focal;
			matrixA[j * vertexNum + i] -= 2.0 * w_Focal;

			float length = sqrt((targetVertexX[i] - targetVertexX[j]) * (targetVertexX[i] - targetVertexX[j]) +
				(targetVertexY[i] - targetVertexY[j]) * (targetVertexY[i] - targetVertexY[j]));

			float length_overlapped = (pointSize * 2 / windowSize + 0.04) * REAL_AXIS;

			// 顶点发生重合
			if (length < length_overlapped) {
				printf("i %d and j %d is overlapped!\n", i, j);

				if (fabs(posX[i] - posX[j]) < EPSINON) {
					struConsX = 0.0;

					int factor = (posY[i] - posY[j]) ? 1 : -1;
					struConsY = length_overlapped * factor * w_Focal;
				}
				else {
					double tmpLength = sqrt((posY[i] - posY[j]) * (posY[i] - posY[j]) + (posX[i] - posX[j]) * (posX[i] - posX[j]));
					e_x = (posX[i] - posX[j]) / tmpLength;
					e_y = (posY[i] - posY[j]) / tmpLength;

					struConsX = e_x * length_overlapped * w_Focal;
					struConsY = e_y * length_overlapped * w_Focal;

				}
			}
			// 非重叠情况
			else {
				if (fabs(posX[i] - posX[j]) < EPSINON) {
					struConsX = 0.0;

					int factor = (posY[i] - posY[j]) ? 1 : -1;
					struConsY = length * factor * w_Focal;
				}
				else {
					double tmpLength = sqrt((posY[i] - posY[j]) * (posY[i] - posY[j]) + (posX[i] - posX[j]) * (posX[i] - posX[j]));
					e_x = (posX[i] - posX[j]) / tmpLength;
					e_y = (posY[i] - posY[j]) / tmpLength;

					struConsX = e_x * length * w_Focal;
					struConsY = e_y * length * w_Focal;

				}
			}


			vectorB_X[i] += (float)struConsX * 2;
			vectorB_X[j] -= (float)struConsX * 2;
			vectorB_Y[i] += (float)struConsY * 2;
			vectorB_Y[j] -= (float)struConsY * 2;
		}
	}

	/************************************************/
	// Time Corhenrecy constraints
	/************************************************/
	for (int i = 0; i < vertexNum; i++) {
		matrixA[i * vertexNum + i] += 2.0 * w_Time;
	}

	for (int i = 0; i < vertexNum; i++) {
		cons_vectorB_X[i] = vectorB_X[i];
		cons_vectorB_Y[i] = vectorB_Y[i];
	}
}

// 释放所有全局变量空间
void freeAll() {
	printf("all has been free!\n");

	delete vertexX, vertexY;

	delete posX, posY;
	delete resultPosX, resultPosY;
	delete targetVertexX, targetVertexY;

	// 释放共轭梯度法的运算矩阵与向量
	delete matrixA;
	delete vectorB_X, vectorB_Y;

	delete edge;
	delete tmpEdge;
}


// 主函数
int main(int argc, char *argv[]) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);

	windowInit();

	// 显示图像
	glutDisplayFunc(&myDisplay);
	glutMouseFunc(mouse);
	glutMainLoop();

	// 释放内存
	freeAll();

	return 0;
}



/***************************************************************************************
****************************************************************************************
								CUDA部分
****************************************************************************************
****************************************************************************************/


// add kernel
__global__ void addKernel(float *c, const float *a, const float *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int size)
{
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> > (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}


// minus kernel
__global__ void minusKernel(float *c, const float *a, const float *b)
{
	int i = threadIdx.x;
	c[i] = a[i] - b[i];
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t minusWithCuda(float *c, const float *a, const float *b, unsigned int size)
{
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	minusKernel << <1, size >> > (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}


/* gpuMatMultKernel：GPU下矩阵乘法核函数
*  a:第一个矩阵指针，表示a[M][N]
*  b:第二个矩阵指针，表示b[N][S]
*  result:结果矩阵，表示result[M][S]
*/
__global__ void gpuMatMultKernel(const float *a, const float *b, float *result, const int M, const int N, const int S)
{
	//int threadId = threadIdx.x + blockIdx.x * blockDim.x;

	int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < M * S)
	{
		int row = threadId / S;
		int column = threadId % S;

		result[threadId] = 0;
		for (int i = 0; i < N; i++)
		{
			result[threadId] += a[row * N + i] * b[i * S + column];
		}
	}
}


// 调用CUDA运行GPU矩阵乘法核函数
cudaError_t mulWithCuda(const float *a, const float *b, float *result, const int M, const int N, const int S)
{
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_result = 0;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_a, M * N * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dev_a failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_b, N * S * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dev_b failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_result, M * S * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dev_result failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_a, a, M * N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudamemcpy dev_a failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, N * S * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy dev_b failed!\n");
		goto Error;
	}

	/*cudaEvent_t gpuStart, gpuFinish;
	float elapsedTime;
	cudaEventCreate(&gpuStart);
	cudaEventCreate(&gpuFinish);
	cudaEventRecord(gpuStart, 0);*/

	/*const int THREADNUM = 256;
	const int BLOCKNUM = (M * S + 255) / 256;*/

	const int BLOCK_SIZE = 16;
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((S + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
	gpuMatMultKernel << <grid, block >> > (dev_a, dev_b, dev_result, M, N, S);
	//gpuMatMultWithSharedKernel<16> << <grid, block >> >(dev_a, dev_b, dev_result, M, N, S);
	//printf("This is NOT shared kernel!\n");

	/*cudaEventRecord(gpuFinish, 0);
	cudaEventSynchronize(gpuFinish);
	cudaEventElapsedTime(&elapsedTime, gpuStart, gpuFinish);
	printf("\nThe runing time of GPU on Mat Multiply is %f seconds.\n", elapsedTime / 1000.0);*/

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "MulKernel launch failed: %s!\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize return Error code %d after Kernel launched!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(result, dev_result, M * S * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy result failed!\n");
		goto Error;
	}

Error:
	//printf("is free!\n");
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_result);

	return cudaStatus;
}


