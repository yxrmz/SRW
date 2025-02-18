// CppUndulatorCLtest.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include "cpp_common/cl.hpp"

#include "cpp_common/util.hpp" // utility library
#include "cpp_common/err_code.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>
#include <complex>
#include <cmath>

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

//#include "err_code.h"
#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c
#define E2G (1956.951269331419)  // electron energy in GeV to gamma
#define K2B (10.71020159392642)

std::vector<double> readGaussNodes(const char* filename, int vLen)
{
	//from https://stackoverflow.com/questions/15138353/how-to-read-a-binary-file-into-a-vector-of-unsigned-chars
	// open the file:
	std::ifstream file(filename, std::ios::binary);

	// Stop eating new lines in binary mode!!!
	file.unsetf(std::ios::skipws);

	// get its size:
	std::streampos fileSize;

	file.seekg(0, std::ios::end);
	fileSize = file.tellg();
	//printf("file size: %i\n", fileSize);
	file.seekg(0, std::ios::beg);

	// reserve capacity
	std::vector<double> vec(vLen);
	//vec.reserve(fileSize);

	// read the data:
	//vec.insert(vec.begin(),
	//	std::istream_iterator<double>(file),
	//	std::istream_iterator<double>());
	file.read(reinterpret_cast<char*>(vec.data()), vec.size() * sizeof(double));

	return vec;
}

void calcSRField_CL(double hR0, double hKx, double hKy, double hPhase, double hGamma, double phEnergy, double hL0, int hNp,
	double thetaMax, int nX, double psiMax, int nY, int gLength, int gIntervals,
	std::vector<std::complex<double>> &Is, std::vector<std::complex<double>> &Ip)
{
	try {
	std::vector<double> tg(gLength);                // gauss nodes vector 
	std::vector<double> ag(gLength);                // gauss weights vector 	

	tg = readGaussNodes("gaussNodes.dat", gLength);
	ag = readGaussNodes("gaussWeights.dat", gLength);

	//printf("tg size: %g", tg.size());

	//for (int jj = 0; jj < gLength; jj++) {
//
	//	printf("%i; node: %g; weight: %g\n", jj, tg[jj], ag[jj]);
	//}

	cl::Buffer dev_tg;
	cl::Buffer dev_ag;
	cl::Buffer dev_ampSigma;
	cl::Buffer dev_ampPi;

	//device init C++ code borrowed from github/Dakkers/OpenCL-examples

	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);

	if (all_platforms.size() == 0) {
		std::cout << " No platforms found.\n";
		exit(1);
	}

	cl::Platform default_platform = all_platforms[0];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	// get default device (CPUs, GPUs) of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0) {
		std::cout << " No devices found.\n";
		exit(1);
	}

	// use device[1]: Intel CPU
	cl::Device default_device = all_devices[1];
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

	cl::Context context({ default_device });

	// create the program that we want to execute on the device
	cl::Program uFieldFunctions(context, util::loadProgram("undulator.cl"), true);

	// create a queue (a queue of commands that the GPU will execute)
	cl::CommandQueue queue(context, default_device);

	dev_tg = cl::Buffer(context, tg.begin(), tg.end(), true);
	dev_ag = cl::Buffer(context, ag.begin(), ag.end(), true);


	dev_ampSigma = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(std::complex<double>) * nX * nY);
	dev_ampPi = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(std::complex<double>) * nX * nY);

	cl::make_kernel<double, double, double, double, double, double, double, int, double, double,
		int, int, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
		uFieldOnMesh(uFieldFunctions, "undulatorFieldOnMesh");



	uFieldOnMesh(cl::EnqueueArgs(queue, cl::NDRange(nX, nY)),
		hR0, hKx, hKy, hPhase, hGamma, phEnergy, hL0, hNp,
		thetaMax, psiMax, gLength, gIntervals, dev_tg, dev_ag, dev_ampSigma, dev_ampPi);

	queue.finish();
	cl::copy(queue, dev_ampSigma, Is.begin(), Is.end());
	cl::copy(queue, dev_ampSigma, Ip.begin(), Ip.end());
}
catch (cl::Error err)
{
	std::cout << "Exception\n";
	std::cerr << "ERROR: "
		<< err.what()
		<< "("
		<< err_code(err.err())
		<< ")"
		<< std::endl;
}
}

int main()
{
	double eE0 = 3; // GeV
	double eI = 0.5; // ring current in Ampere

	double lambda0 = 25; //period in mm
	int nPeriods = 50; // number of undulator periods

	double gamma = eE0 * E2G;
	//double Bx = 0.2; // peak magnetic field in Tesla
	//double Kx = K2B / Bx / lambda0;
	double Kx = 1.;
	double By = 0;
	double Ky = K2B / By / lambda0;
	double phase = 0;
	double energy = 8000;
	double thetaMax = 1e-6;
	int nTheta = 11;
	double psiMax = 1e-6;
	int nPsi = 11;
	int gaussDim = 25;
	int gaussIntervals = 4;

	std::vector<std::complex<double>> ampSigma(nTheta*nPsi);
	std::vector<std::complex<double>> ampPi(nTheta*nPsi);

	calcSRField_CL(0, Kx, 0, 0, gamma, energy, lambda0, nPeriods, thetaMax, nTheta, psiMax, nPsi, 
		gaussDim, gaussIntervals, ampSigma, ampPi);
/*
	for (int nx=0; nx < nTheta; nx++) {
		for (int ny = 0; ny < nPsi; ny++) {
			printf("Point[%i, %i], ampSigma: %g + i%g\n",
				nx, ny, ampSigma[nx*nPsi + ny].real(), ampSigma[nx*nPsi + ny].imag());
		}
	}
	*/
	
	return 0;
}





