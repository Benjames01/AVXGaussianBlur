#include "Header.h"
#include <fstream>

using namespace std;

int main() {
	//the following command pins the current process to the 1st core
	//otherwise, the OS tongles this process between different cores
	BOOL success = SetProcessAffinityMask(GetCurrentProcess(), 1);
	if (success == 0) {
		cout << "SetProcessAffinityMask failed" << endl;
		system("pause");
		return -1;
	}

	//--------------read the input image
	read_image(in, in_image);

	//------Gaussian Blur
	auto start = std::chrono::high_resolution_clock::now();

	for (int it = 0; it != TIMES; it++) {
		//Gaussian_Blur_default_unrolled();
		Gaussian_Blur_AVX(); // Average: 6.72602 s for TIMES==100
		//Gaussian_Blur_Separable();// Average 2.48279 s for TIMES=100
		//Gaussian_Blur_Separable_AVX(); // Average 6.3694 s for TIMES==100
		//Gaussian_Blur_default(); // Average: 11.9685 s for TIMES==100
	}

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Gaussian blur Elapsed time: " << elapsed.count() << " s\n";

	//write output image
	write_image(out, filt_image);

	snprintf(message, sizeof(message) - 1, "Gaussian Blur");
	print_message(message, compare_Gaussian_images());

	//------Sobel
	start = std::chrono::high_resolution_clock::now();

	for (int it = 0; it != TIMES; it++) {
		Sobel_default();
	}

	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	std::cout << "Sobel Elapsed time: " << elapsed.count() << " s\n";

	scale_image();

	//write output image
	write_image(out2, imag);

	snprintf(message, sizeof(message) - 1, "Sobel");
	print_message(message, compare_Sobel_images());

	system("pause");
	return 0;
}

void Gaussian_Blur_Separable_AVX() {
	// kernel separated as two 1d kernels
	// Nx1 -> 1xN convolution
	__m256i const0 = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 4, 3, 1); // Horizontal pass
	int hy[] = { 1,3,4,3,1 }; // Vertical pass

	__m256i r0;

	int i, j, i2, j2;
	int conv;

	int count = 0;

	// Horizontal pass
	for (i = 0; i < M; i++) {
		for (j = 0; j < N - 14; j++) {
			conv = 0;

			r0 = _mm256_loadu_si256((__m256i*) & in_image[i][j - 2]);
			r0 = _mm256_mullo_epi16(r0, const0);

			int total = _mm256_extract_epi16(r0, 0);
			total += _mm256_extract_epi16(r0, 1);
			total += _mm256_extract_epi16(r0, 2);
			total += _mm256_extract_epi16(r0, 3);
			total += _mm256_extract_epi16(r0, 4);

			int total2 = getVectorSum(r0);

			conv_hx[i][j] = total;
		}
	}

	// Vertical pass
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			conv = 0;

			for (i2 = -2; i2 <= 2; i2++) {
				conv += hy[i2 + 2] * conv_hx[i][j + i2];
			}

			filt_image[i][j] = conv / 144; // Normalise output by dividing by sum of kernel
		}
	}
	//writeMatrixToFile("conv_separable_avx.txt", filt_image);
	//writeMatrixToFile("filt_separable_avx.txt", filt_image);
}

void Gaussian_Blur_Separable() {
	int hx[] = { 1,3,4,3,1 };
	int hy[] = { 1,3,4,3,1 };

	int i, j, i2, j2;
	int index, index2;
	unsigned short conv;
	int count = 0;

	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			conv = 0;
			for (j2 = -2; j2 <= 2; j2++) {
				// bounds check for column index
				/*if (j + j2 < 0 || j + j2 > N)
					continue;*/

				conv += hx[j2 + 2] * in_image[i][j + j2];
			} // for(j2)
			conv_hx[i][j] = conv;
		}
	}

	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			conv = 0;

			for (i2 = -2; i2 <= 2; i2++) {
				// bounds check for row index
				/*if (i + i2 < 0 || i + i2 > M)
					continue;*/

				conv += hy[i2 + 2] * conv_hx[i][j + i2];
			}

			filt_image[i][j] = conv / 144;
		}
	}
	writeMatrixToFile("conv_seperable.txt", conv_hx);
	writeMatrixToFile("filt_separable.txt", filt_image);
}

void Gaussian_Blur_AVX() {
	__m256i r0, r1, r2, r3, r4, t0;
	__m256i const0, const1, const2;

	int row, col;

	//NxN convolution
	const0 = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 4, 3, 1);
	const1 = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 9, 12, 9, 3);
	const2 = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 12, 16, 12, 4);

	for (row = 2; row < N - 2; row++) {
		for (col = 2; col < M - 14; col++) {
			t0 = _mm256_setzero_si256();

			// Load 256 bits of packed integers from 5 adjacent rows
			r0 = _mm256_loadu_si256((__m256i*) & in_image[row - 2][col - 2]);
			r1 = _mm256_loadu_si256((__m256i*) & in_image[row - 1][col - 2]);
			r2 = _mm256_loadu_si256((__m256i*) & in_image[row][col - 2]);
			r3 = _mm256_loadu_si256((__m256i*) & in_image[row + 1][col - 2]);
			r4 = _mm256_loadu_si256((__m256i*) & in_image[row + 2][col - 2]);

			// Multiply each row by corresponding kernel row
			r0 = _mm256_madd_epi16(r0, const0);
			r1 = _mm256_madd_epi16(r1, const1);
			r2 = _mm256_madd_epi16(r2, const2);
			r3 = _mm256_madd_epi16(r3, const1);
			r4 = _mm256_madd_epi16(r4, const0);

			// Calculate the sum of the adjacent rows
			t0 = _mm256_add_epi16(t0, r0);
			t0 = _mm256_add_epi16(t0, r1);
			t0 = _mm256_add_epi16(t0, r2);
			t0 = _mm256_add_epi16(t0, r3);
			t0 = _mm256_add_epi16(t0, r4);

			// Calculate output pixel and normalise it by dividing by sum of kernel
			filt_image[row][col] = getVectorSum(t0) / 144;
		}

		// padding required to avoid going out of bounds
		for (col = 1008; col < M - 2; col++) {
			int temp = 0;
			for (int rowoffset = -2; rowoffset <= 2; rowoffset++) {
				for (int coloffset = -2; coloffset <= 2; coloffset++) {
					temp += in_image[row + rowoffset][col + coloffset] * gaussianMask[2 + rowoffset][2 + coloffset];
				}
			}
			// Calculate output pixel and normalise it by dividing by sum of kernel
			filt_image[row][col] = temp / 144;
		}
	}
	writeMatrixToFile("conv_avx.txt", filt_image);
}

void Gaussian_Blur_default_unrolled() {
	short int row, col;
	short int newPixel;
	int count = 0;

	for (row = 2; row < N - 2; row++) {
		for (col = 2; col < M - 2; col++) {
			newPixel = 0;

			newPixel += in_image[row - 2][col - 2] * gaussianMask[0][0];
			newPixel += in_image[row - 2][col - 1] * gaussianMask[0][1];
			newPixel += in_image[row - 2][col] * gaussianMask[0][2];
			newPixel += in_image[row - 2][col + 1] * gaussianMask[0][3];
			newPixel += in_image[row - 2][col + 2] * gaussianMask[0][4];

			newPixel += in_image[row - 1][col - 2] * gaussianMask[1][0];
			newPixel += in_image[row - 1][col - 1] * gaussianMask[1][1];
			newPixel += in_image[row - 1][col] * gaussianMask[1][2];
			newPixel += in_image[row - 1][col + 1] * gaussianMask[1][3];
			newPixel += in_image[row - 1][col + 2] * gaussianMask[1][4];

			newPixel += in_image[row][col - 2] * gaussianMask[2][0];
			newPixel += in_image[row][col - 1] * gaussianMask[2][1];
			newPixel += in_image[row][col] * gaussianMask[2][2];
			newPixel += in_image[row][col + 1] * gaussianMask[2][3];
			newPixel += in_image[row][col + 2] * gaussianMask[2][4];

			newPixel += in_image[row + 1][col - 2] * gaussianMask[3][0];
			newPixel += in_image[row + 1][col - 1] * gaussianMask[3][1];
			newPixel += in_image[row + 1][col] * gaussianMask[3][2];
			newPixel += in_image[row + 1][col + 1] * gaussianMask[3][3];
			newPixel += in_image[row + 1][col + 2] * gaussianMask[3][4];

			newPixel += in_image[row + 2][col - 2] * gaussianMask[4][0];
			newPixel += in_image[row + 2][col - 1] * gaussianMask[4][1];
			newPixel += in_image[row + 2][col] * gaussianMask[4][2];
			newPixel += in_image[row + 2][col + 1] * gaussianMask[4][3];
			newPixel += in_image[row + 2][col + 2] * gaussianMask[4][4];

			filt_image[row][col] = newPixel / 144;
		}
	}
}

void Gaussian_Blur_default() {
	short int row, col, rowOffset, colOffset;
	short int newPixel;

	for (row = 2; row < N - 2; row++) {
		for (col = 2; col < M - 2; col++) {
			newPixel = 0;
			for (rowOffset = -2; rowOffset <= 2; rowOffset++) {
				for (colOffset = -2; colOffset <= 2; colOffset++) {
					newPixel +=
						in_image[row + rowOffset][col + colOffset]
						* gaussianMask[2 + rowOffset][2 + colOffset];
				}
			}
			filt_image[row][col] = newPixel / 144;
		}
	}
}

//returns false/true, when the output image is incorrect/correct, respectively
bool compare_Gaussian_images() {
	int row, col, rowOffset, colOffset;
	int newPixel;

	bool passed = true;
	int count = 0;

	for (row = 2; row < N - 2; row++) {
		for (col = 2; col < M - 2; col++) {
			newPixel = 0;
			for (rowOffset = -2; rowOffset <= 2; rowOffset++) {
				for (colOffset = -2; colOffset <= 2; colOffset++) {
					newPixel += in_image[row + rowOffset][col + colOffset] * gaussianMask[2 + rowOffset][2 + colOffset];
				}
			}
			newPixel = newPixel / 144;
			if (newPixel != filt_image[row][col]) {
				printf("\n %d %d - %d %d\n", row, col, newPixel, filt_image[row][col]);
				if (count == 9 & !passed)
					return false;
				passed = false;
				count++;
			}
		}
	}

	return passed;
}

void scale_image() {
	/* the output of Sobel (gradient has values larger than 255, thus those are capped to 255
	alternatively, we can scale it, or use canny algorithm*/

	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++) {
			if (gradient[i][j] <= 255) imag[i][j] = (unsigned char)gradient[i][j];
			else imag[i][j] = 255;
		}
}

void Sobel_default() {
	int row, col, rowOffset, colOffset;
	int Gx, Gy;
	float thisAngle;
	int newAngle;

	/*---------------------------- Determine edge directions and gradient strengths -------------------------------------------*/
	for (row = 1; row < N - 1; row++) {
		for (col = 1; col < M - 1; col++) {
			Gx = 0;
			Gy = 0;

			/* Calculate the sum of the Sobel mask times the nine surrounding pixels in the x and y direction */
			for (rowOffset = -1; rowOffset <= 1; rowOffset++) {
				for (colOffset = -1; colOffset <= 1; colOffset++) {
					Gx += filt_image[row + rowOffset][col + colOffset] * GxMask[rowOffset + 1][colOffset + 1];
					Gy += filt_image[row + rowOffset][col + colOffset] * GyMask[rowOffset + 1][colOffset + 1];
				}
			}

			//gradient[row][col] = sqrt(pow(Gx, 2.0) + pow(Gy, 2.0));	/* Calculate gradient strength		*/
			gradient[row][col] = abs(Gx) + abs(Gy); // this is an optimized version of the above

			thisAngle = (atan2(Gx, Gy) / 3.14159) * 180.0;		/* Calculate actual direction of edge [-180, +180]*/

			/* Convert actual edge direction to approximate value */
			if (((thisAngle >= -22.5) && (thisAngle <= 22.5)) || (thisAngle >= 157.5) || (thisAngle <= -157.5))
				newAngle = 0;
			if (((thisAngle > 22.5) && (thisAngle < 67.5)) || ((thisAngle > -157.5) && (thisAngle < -112.5)))
				newAngle = 45;
			if (((thisAngle >= 67.5) && (thisAngle <= 112.5)) || ((thisAngle >= -112.5) && (thisAngle <= -67.5)))
				newAngle = 90;
			if (((thisAngle > 112.5) && (thisAngle < 157.5)) || ((thisAngle > -67.5) && (thisAngle < -22.5)))
				newAngle = 135;

			edgeDir[row][col] = newAngle;
		}
	}
}

bool compare_Sobel_images() {
	int row, col, rowOffset, colOffset;
	int Gx, Gy, test1, test2;
	float thisAngle;
	int newAngle;

	/*---------------------------- Determine edge directions and gradient strengths -------------------------------------------*/
	for (row = 1; row < N - 1; row++) {
		for (col = 1; col < M - 1; col++) {
			Gx = 0;
			Gy = 0;

			/* Calculate the sum of the Sobel mask times the nine surrounding pixels in the x and y direction */
			for (rowOffset = -1; rowOffset <= 1; rowOffset++) {
				for (colOffset = -1; colOffset <= 1; colOffset++) {
					Gx += filt_image[row + rowOffset][col + colOffset] * GxMask[rowOffset + 1][colOffset + 1];
					Gy += filt_image[row + rowOffset][col + colOffset] * GyMask[rowOffset + 1][colOffset + 1];
				}
			}

			test1 = abs(Gx) + abs(Gy);
			thisAngle = (atan2(Gx, Gy) / 3.14159) * 180.0;		/* Calculate actual direction of edge [-180, +180]*/

			/* Convert actual edge direction to approximate value */
			if (((thisAngle >= -22.5) && (thisAngle <= 22.5)) || (thisAngle >= 157.5) || (thisAngle <= -157.5))
				newAngle = 0;
			if (((thisAngle > 22.5) && (thisAngle < 67.5)) || ((thisAngle > -157.5) && (thisAngle < -112.5)))
				newAngle = 45;
			if (((thisAngle >= 67.5) && (thisAngle <= 112.5)) || ((thisAngle >= -112.5) && (thisAngle <= -67.5)))
				newAngle = 90;
			if (((thisAngle > 112.5) && (thisAngle < 157.5)) || ((thisAngle > -67.5) && (thisAngle < -22.5)))
				newAngle = 135;

			if (test1 != gradient[row][col]) {
				return false;
			}

			if (edgeDir[row][col] != newAngle)
				return false;
		}
	}

	return true;
}

void read_image(char* filename, unsigned short int image[N][M])
{
	int inint = -1;
	int c;
	FILE* finput;
	int i, j;

	printf("  Reading image from disk (%s)...\n", filename);
	//finput = NULL;
	openfile(filename, &finput);

	for (j = 0; j < N; j++)
		for (i = 0; i < M; i++) {
			c = getc(finput);

			image[j][i] = (unsigned short int)c;
		}

	/* for (j=0; j<N; ++j)
	   for (i=0; i<M; ++i) {
		 if (fscanf(finput, "%i", &inint)==EOF) {
		   fprintf(stderr,"Premature EOF\n");
		   exit(-1);
		 } else {
		   image[j][i]= (unsigned char) inint; //printf("\n%d",inint);
		 }
	   }*/

	fclose(finput);
}

void write_image(char* filename, unsigned short int image[N][M])
{
	FILE* foutput;
	int i, j;

	printf("  Writing result to disk (%s)...\n", filename);
	if ((err = fopen_s(&foutput, filename, "wb")) != NULL) {
		printf("Unable to open file %s for writing\n", filename);
		exit(-1);
	}

	fprintf(foutput, "P2\n");
	fprintf(foutput, "%d %d\n", M, N);
	fprintf(foutput, "%d\n", 255);

	for (j = 0; j < N; ++j) {
		for (i = 0; i < M; ++i) {
			fprintf(foutput, "%3d ", image[j][i]);
			if (i % 32 == 31) fprintf(foutput, "\n");
		}
		if (M % 32 != 0) fprintf(foutput, "\n");
	}
	fclose(foutput);
}

void openfile(char* filename, FILE** finput)
{
	int x0, y0;
	char header[255];
	int aa;

	if ((err = fopen_s(finput, filename, "rb")) != NULL) {
		printf("Unable to open file %s for reading\n");
		exit(-1);
	}

	aa = fscanf_s(*finput, "%s", header, 20);

	/*if (strcmp(header,"P2")!=0) {
	   fprintf(stderr,"\nFile %s is not a valid ascii .pgm file (type P2)\n",
			   filename);
	   exit(-1);
	 }*/

	x0 = getint(*finput);
	y0 = getint(*finput);

	if ((x0 != M) || (y0 != N)) {
		printf("Image dimensions do not match: %ix%i expected\n", N, M);
		exit(-1);
	}

	x0 = getint(*finput); /* read and throw away the range info */
}

int getint(FILE* fp) /* adapted from "xv" source code */
{
	int c, i, firstchar, garbage;

	/* note:  if it sees a '#' character, all characters from there to end of
	   line are appended to the comment string */

	   /* skip forward to start of next number */
	c = getc(fp);
	while (1) {
		/* eat comments */
		if (c == '#') {
			/* if we're at a comment, read to end of line */
			char cmt[256], * sp;

			sp = cmt;  firstchar = 1;
			while (1) {
				c = getc(fp);
				if (firstchar && c == ' ') firstchar = 0;  /* lop off 1 sp after # */
				else {
					if (c == '\n' || c == EOF) break;
					if ((sp - cmt) < 250) *sp++ = c;
				}
			}
			*sp++ = '\n';
			*sp = '\0';
		}

		if (c == EOF) return 0;
		if (c >= '0' && c <= '9') break;   /* we've found what we were looking for */

		/* see if we are getting garbage (non-whitespace) */
		if (c != ' ' && c != '\t' && c != '\r' && c != '\n' && c != ',') garbage = 1;

		c = getc(fp);
	}

	/* we're at the start of a number, continue until we hit a non-number */
	i = 0;
	while (1) {
		i = (i * 10) + (c - '0');
		c = getc(fp);
		if (c == EOF) return i;
		if (c < '0' || c>'9') break;
	}
	return i;
}

void print_message(char* s, bool outcome) {
	if (outcome == true)
		printf("\n\n\r ----- %s output is correct -----\n\r", s);
	else
		printf("\n\n\r -----%s output is INcorrect -----\n\r", s);
}

void addIntsToArray(__m256i vec, int index, unsigned short int out_array[]) {
	uint16_t* i = (uint16_t*)&vec;
	int total = 0;

	for (int x = 0; x < 5; x++) {
		out_array[index + x] = i[x];
	}
}

int getVectorSum(__m256i vec) {
	uint16_t* i = (uint16_t*)&vec;
	int total = 0;

	for (int x = 0; x < 16; x++) {
		total += i[x];
	}
	return total;
}

void printVector(__m256i vec, string name) {
	uint16_t* i = (uint16_t*)&vec;
	cout << name;
	for (int x = 0; x < 16; x++) {
		cout << i[x] << " ";
	}
	cout << endl;
}

void debugPrint(const char const* debugText)
{
#ifdef DEBUG
	printf(debugText);
#endif // DEBUG
}

void writeMatrixToFile(string name, unsigned short int matrix[N][M]) {
#ifdef DEBUG
	fstream myfile;
	myfile.open(name, fstream::out);

	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 100; j++)
		{
			myfile << matrix[i][j] << "\t";
		}
		myfile << std::endl;
	}
	myfile.close();
#endif // DEBUG
}