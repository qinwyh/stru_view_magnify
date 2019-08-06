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
#include <malloc.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>


// �Զ���ͷ�ļ�
#include <draw_graph_2d.h>

#define TILE_WIDTH 16 

using namespace std;

// focal area
vector<int> focalArea;


//const float testCons = 0.842;
extern const float pointSize = 4.0;
extern const float edgeWidth = 1.0;
extern const int   windowSize = 800;

// �Ŵ���
extern float m_factor;

extern int vertexNum;
extern int edgeNum;
// ��ʼ��������
extern float *vertexX, *vertexY;
// ������������
extern float *posX, *posY;
extern float *resultPosX, *resultPosY;
extern float *targetVertexX, *targetVertexY;

// ��������
extern float focalPointX, focalPointY;

// �Ƿ��ڷŴ�״̬
extern BOOLEAN isMagnified;

extern BOOLEAN *edge;

// �����ݶȷ���ʹ�õ����������
extern float *matrixA;
extern float *vectorB_X, *vectorB_Y;
extern float *cons_vectorB_X, *cons_vectorB_Y;

extern float *tmpVector, *tmpX, *tmpY;
extern float *vectorD, *vectorR;

extern int *tmpEdge;


// CUDA����
// �����������
cudaError_t addMulWithCuda(float *c, const float *a, const float *b, unsigned int size, const float m, const float n);

// �����������
cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int size);

// �����������
cudaError_t minusWithCuda(float *c, const float *a, const float *b, unsigned int size);

// �����г˻�
cudaError_t mulWithCuda(const float *a, const float *b, float *result, const int M, const int N, const int S);

// �ͷ����пռ�
void freeAll();

// ��������constraints
void computeAllCons();


// �����ݶȷ���������꣨�����ã�
void conjGradientSolver() {
	printf("add with new new CUDA\n");

	float* tmpUse = new float[vertexNum];

	// ��ȡ�˴�time constraints
	/*for (int i = 0; i < vertexNum; i++) {
		vectorB_X[i] = cons_vectorB_X[i] + resultPosX[i] * 2 * REAL_AXIS;
		vectorB_Y[i] = cons_vectorB_Y[i] + resultPosY[i] * 2 * REAL_AXIS;
	}*/

	addMulWithCuda(vectorB_X, cons_vectorB_X, resultPosX, vertexNum, 1, 2 * REAL_AXIS);
	addMulWithCuda(vectorB_Y, cons_vectorB_Y, resultPosY, vertexNum, 1, 2 * REAL_AXIS);

	
	::memset(tmpX, 0, vertexNum * sizeof(float));
	::memset(tmpY, 0, vertexNum * sizeof(float));
	::memset(vectorD, 0, vertexNum * sizeof(float));

	// ����X�����ϵ�����
	/*for (int i = 0; i < vertexNum; i++) {
		vectorR[i] = -vectorB_X[i];
	}*/
	minusWithCuda(vectorR, tmpX, vectorB_X, vertexNum);

	for (int nor = 0; nor < vertexNum * 3; nor++) {
		float denom = 0, num = 0;
		float denom2 = 0, num2 = 0;

		// ����r��ת�ó���r
		/*for (int i = 0; i < vertexNum; i++) {
			denom += vectorR[i] * vectorR[i];
		}*/
		mulWithCuda(vectorR, vectorR, &denom, 1, vertexNum, 1);


		// ����в�r = Ax-b
		/*for (int i = 0; i < vertexNum; i++) {
			float cur = 0;
			for (int j = 0; j < vertexNum; j++) {
				cur += matrixA[i * vertexNum + j] * tmpX[j];
			}
			vectorR[i] = cur - vectorB_X[i];
		}*/
		mulWithCuda(matrixA, tmpX, tmpVector, vertexNum, vertexNum, 1);
		minusWithCuda(vectorR, tmpVector, vectorB_X, vertexNum);


		//����r��ת�ó���r
		/*for (int i = 0; i < vertexNum; i++) {
			num += vectorR[i] * vectorR[i];
		}*/
		mulWithCuda(vectorR, vectorR, &num, 1, vertexNum, 1);

		if (num < 0.00001) {
			//printf("x is break\n");
			break;
		}

		//���㷽������d
		/*for (int i = 0; i < vertexNum; i++) {
			vectorD[i] = -vectorR[i] + num / denom * vectorD[i];
		}*/
		::memcpy(tmpUse, vectorD, vertexNum * sizeof(float));
		addMulWithCuda(vectorD, tmpUse, vectorR, vertexNum, num / denom, -1.0);
		

		//����d��ת�ó���r
		/*for (int i = 0; i < vertexNum; i++) {
			num2 += vectorD[i] * vectorR[i];
		}*/
		mulWithCuda(vectorD, vectorR, &num2, 1, vertexNum, 1);

		//����d��ת�ó���A����d���
		/*for (int i = 0; i < vertexNum; i++) {
			float cur = 0;
			for (int j = 0; j < vertexNum; j++) {
				cur += matrixA[i * vertexNum + j] * vectorD[j];
			}
			denom2 += cur * vectorD[i];
		}*/
		mulWithCuda(matrixA, vectorD, tmpVector, vertexNum, vertexNum, 1);
		mulWithCuda(vectorD, tmpVector, &denom2, 1, vertexNum, 1);

		//���㲽��
		double a = -num2 / denom2;
		/*for (int i = 0; i < vertexNum; i++) {
			tmpX[i] += a * vectorD[i];
		}*/
		::memcpy(tmpUse, tmpX, vertexNum * sizeof(float));
		addMulWithCuda(tmpX, tmpUse, vectorD, vertexNum, 1.0, (float)a);

		//printf("all X with CUDA!\n");
	}


	// ����y�����ϵ�����
	::memset(vectorD, 0, vertexNum * sizeof(float));
	/*for (int i = 0; i < vertexNum; i++) {
		vectorR[i] = -vectorB_Y[i];
	}*/
	minusWithCuda(vectorR, tmpY, vectorB_Y, vertexNum);

	for (int nor = 0; nor < vertexNum * 3; nor++) {
		float denom = 0, num = 0;
		float denom2 = 0, num2 = 0;

		// ����r��ת�ó���r
		mulWithCuda(vectorR, vectorR, &denom, 1, vertexNum, 1);

		// ����в�r = Ax-b

		mulWithCuda(matrixA, tmpY, tmpVector, vertexNum, vertexNum, 1);
		minusWithCuda(vectorR, tmpVector, vectorB_Y, vertexNum);

		//����r��ת�ó���r
		mulWithCuda(vectorR, vectorR, &num, 1, vertexNum, 1);

		if (num < 0.00001) {
			//printf("y is break\n");
			break;
		}

		//���㷽������d
		/*for (int i = 0; i < vertexNum; i++) {
			vectorD[i] = -vectorR[i] + num / denom * vectorD[i];
		}*/
		::memcpy(tmpUse, vectorD, vertexNum * sizeof(float));
		addMulWithCuda(vectorD, tmpUse, vectorR, vertexNum, num / denom, -1.0);

		//����d��ת�ó���r
		mulWithCuda(vectorD, vectorR, &num2, 1, vertexNum, 1);

		//����d��ת�ó���A����d���
		mulWithCuda(matrixA, vectorD, tmpVector, vertexNum, vertexNum, 1);
		mulWithCuda(vectorD, tmpVector, &denom2, 1, vertexNum, 1);

		//���㲽��
		double a = -num2 / denom2;
		/*for (int i = 0; i < vertexNum; i++) {
			tmpY[i] += a * vectorD[i];
		}*/
		::memcpy(tmpUse, tmpY, vertexNum * sizeof(float));
		addMulWithCuda(tmpY, tmpUse, vectorD, vertexNum, 1.0, (float)a);

		//printf("all Y with CUDA!\n");
	}

	for (int i = 0; i < vertexNum; i++) {
		resultPosX[i] = tmpX[i] / REAL_AXIS;
		resultPosY[i] = tmpY[i] / REAL_AXIS;
		//printf("resultPosX %d is %6.2f\n", i, resultPosX[i]);
	}

	// free
	delete tmpUse;
	
}


// ����targetGraph����������
void calTargetGraph() {

	float disX, disY;
	float focalX = focalPointX * REAL_AXIS;
	float focalY = focalPointY * REAL_AXIS;

	focalArea.clear();

	for (int i = 0; i < vertexNum; i++) {
		disX = posX[i] * REAL_AXIS - focalX;
		disY = posY[i] * REAL_AXIS - focalY;

		// ������ĳ���غ�
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

		// ��¼focal area��ĵ㼯
		float length = sqrt((targetVertexX[i] - focalX) * (targetVertexX[i] - focalX) + 
						    (targetVertexY[i] - focalY) * (targetVertexY[i] - focalY));

		if (length < 0.3 * REAL_AXIS)
			focalArea.push_back(i);

	}

	/*for (int i = 0; i < focalArea.size(); i++) {
		cout <<"focal area: " << focalArea[i] << endl;
	}*/
}

// ������λ�����ݵĺ���
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

// ��������constraints
void computeAllCons() {
	//printf("computeAllCons\n");

	::memset(matrixA, 0, vertexNum * vertexNum * sizeof(float));
	::memset(vectorB_X, 0, vertexNum * sizeof(float));
	::memset(vectorB_Y, 0, vertexNum * sizeof(float));

	//float constraints;

	// �����߼�, ˫���ȸ�������߾���
	double struConsX, struConsY, length, e_x, e_y;

	// ������
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

			// ���㷢���غ�
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
			// ���ص����
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

// �ͷ�����ȫ�ֱ����ռ�
void freeAll() {
	printf("all has been free!\n");

	delete vertexX, vertexY;

	delete posX, posY;
	delete resultPosX, resultPosY;
	delete targetVertexX, targetVertexY;

	// �ͷŹ����ݶȷ����������������
	delete matrixA;
	delete vectorB_X, vectorB_Y;

	delete edge;
	delete tmpEdge;

	//delete struConsVecX, struConsVecY;
	delete tmpX, tmpY, tmpVector;
	delete vectorD;
	delete vectorR;
}


// ������
int main(int argc, char *argv[]) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);

	windowInit();

	// ��ʾͼ��
	glutDisplayFunc(&myDisplay);
	glutMouseFunc(mouse);
	glutMainLoop();

	// �ͷ��ڴ�
	freeAll();

	return 0;
}



/***************************************************************************************
****************************************************************************************
								CUDA����
****************************************************************************************
****************************************************************************************/
// mul kernel
__global__ void addMulKernel(float *c, const float *a, const float *b, const float *m, const float *n)
{
	int i = threadIdx.x;
	c[i] = a[i] * m[0] + b[i] * n[0];
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addMulWithCuda(float *c, const float *a, const float *b, unsigned int size, const float m, const float n)
{
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_c = 0;
	float *dev_m = 0;
	float *dev_n = 0;
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

	cudaStatus = cudaMalloc((void**)&dev_m, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_n, sizeof(float));
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

	cudaStatus = cudaMemcpy(dev_m, &m, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_n, &n, sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addMulKernel << <1, size >> > (dev_c, dev_a, dev_b, dev_m, dev_n);

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


/* gpuMatMultKernel��GPU�¾���˷��˺���
*  a:��һ������ָ�룬��ʾa[M][N]
*  b:�ڶ�������ָ�룬��ʾb[N][S]
*  result:������󣬱�ʾresult[M][S]
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


__global__ void gpuMatMultWithSharedKernel(float *A, float  *B, float *C, int m, int n, int k)
{
	//���빲���ڴ棬������ÿ��block��
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

	//������Ƿ�,��������6����ʾ�ĵط����ǲ��еĵط���
	int bx = blockIdx.x;		int by = blockIdx.y;
	int tx = threadIdx.x;		int ty = threadIdx.y;

	//ȷ����������е��к���
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	//��ʱ����
	float Cvalue = 0;

	//ѭ������A,B��Ƭ�����������󣬷ֽ׶ν��м���
	for (int t = 0; t < (n - 1) / TILE_WIDTH + 1; ++t)
	{
		//��A,B������Ƭ���Ľ������shared memory�У�ÿ���̼߳�����Ӧ��CԪ�ص�A/B����Ԫ��
		if (Row < m && t * TILE_WIDTH + tx < n)		//Խ�紦�����������С�ľ�����ˣ���ѡ��
			//ds_A[tx][ty] = A[t*TILE_WIDTH + tx][Row];
			ds_A[tx][ty] = A[Row*n + t * TILE_WIDTH + tx];//�Ժϲ��ķ�ʽ������Ƭ
		else
			ds_A[tx][ty] = 0.0;

		if (t * TILE_WIDTH + ty < n && Col < k)
			//ds_B[tx][ty] = B[Col][t*TILE_WIDTH + ty];
			ds_B[tx][ty] = B[(t*TILE_WIDTH + ty)*k + Col];
		else
			ds_B[tx][ty] = 0.0;

		//��֤tile�����е�Ԫ�ر�����
		__syncthreads();

		for (int i = 0; i < TILE_WIDTH; ++i)
			Cvalue += ds_A[i][ty] * ds_B[tx][i];//��shared memory��ȡֵ

		//ȷ�������߳���ɼ���󣬽�����һ���׶εļ���
		__syncthreads();

		if (Row < m && Col < k)
			C[Row*k + Col] = Cvalue;
	}
}



// ����CUDA����GPU����˷��˺���
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

	/*const int BLOCK_SIZE = 16;
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((S + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
	gpuMatMultKernel << <grid, block >> > (dev_a, dev_b, dev_result, M, N, S);*/
	//gpuMatMultWithSharedKernel<16> << <grid, block >> >(dev_a, dev_b, dev_result, M, N, S);
	//printf("This is NOT shared kernel!\n");

	/*cudaEventRecord(gpuFinish, 0);
	cudaEventSynchronize(gpuFinish);
	cudaEventElapsedTime(&elapsedTime, gpuStart, gpuFinish);
	printf("\nThe runing time of GPU on Mat Multiply is %f seconds.\n", elapsedTime / 1000.0);*/

	dim3 dimGrid((S - 1) / TILE_WIDTH + 1, (M - 1) / TILE_WIDTH + 1, 1);	//����ȡ��
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

	//�����ں˺���
	gpuMatMultWithSharedKernel << <dimGrid, dimBlock >> > (dev_a, dev_b, dev_result, M, N, S);

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


