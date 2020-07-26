#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pmmintrin.h>
#include <process.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <Windows.h>
#include <string>

#define N 1024 // rows
#define M 1024 // columns
#define TIMES 1 // number of iterations of blur function (recommended: 100)

#define P 1024

#define DEBUG false;

void Gaussian_Blur_AVX();
void Gaussian_Blur_SSE();
void Gaussian_Blur_Seperable_AVX();
void Gaussian_Blur_Seperable();
void Gaussian_Blur_default();
void Gaussian_Blur_default_unrolled();
bool compare_Gaussian_images();

int getVectorSum(__m256i vec);

void writeMatrixToFile(std::string name, unsigned short int matrix[N][M]);

void Sobel_default();
bool compare_Sobel_images();

void scale_image();

__declspec(align(64))  unsigned short int  imag[N][M], in_image[N][M], filt_image[N][M], out_image[N][M], first_pass[N * M], conv_hx[N][M];
__declspec(align(64))  int edgeDir[N][M];
__declspec(align(64))  int gradient[N][M];

const unsigned short int gaussianMask[5][5] = {
	{1,3,4,3,1} ,
	{3,9,12,9,3},
	{4,12,16,12,4},
	{3,9,12,9,3},
	{1,3,4,3,1}
};
const short int GxMask[3][3] = {
	{-1,0,1} ,
	{-2,0,2},
	{-1,0,1}
};

const short int GyMask[3][3] = {
	{-1,-2,-1} ,
	{0,0,0},
	{1,2,1}
};

char message[20];
void print_message(char* s, bool outcome);

//PLEASE AMEND THE DIRECTORY BELOW
char in[100] = "D:\\UNI WORK\\Coursework\\Coursework\\AVXGaussianBlur\\rec.pgm";
char out[100] = "D:\\UNI WORK\\Coursework\\Coursework\\output\\filtered.pgm";
char out2[100] = "D:\\UNI WORK\\Coursework\\Coursework\\output\\gradient.pgm";

int getNewInt(__m256i vec);

FILE* fin;
errno_t err;

void read_image(char* filename, unsigned short int  image[N][M]);
void write_image(char* filename, unsigned short int  imag[N][M]);
void write_image2(char* filename, unsigned short int  imag[N][M]);

void openfile(char* filename, FILE** finput);
int getint(FILE* fp);

void debugPrint(const char const* debugText);