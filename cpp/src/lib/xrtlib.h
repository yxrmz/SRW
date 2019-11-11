//#include "stdafx.h"
//#include "cpp_common/cl.hpp"

#include "srradstr.h"

//#include "cpp_common/util.hpp" // utility library
//#include "cpp_common/err_code.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>
#include <complex>
#include <cmath>


class xrtRadIntCL {

	double eE0; // GeV
	double eI; // ring current in Ampere
	double lambda0; //period in mm
	int nPeriods; // number of undulator periods
	double gamma;
	double Bx; // peak magnetic field in Tesla
	double Kx;
	double By;
	double Ky;
	double phase;
	double energy;
	double xStart, xEnd, xStep;
	double zStart, zEnd, zStep;
	double eStart, eEnd, eStep;
	long nx, nz, ne;
	long tg, gaussDim;
	long ag, gaussIntervals;
	srTSRWRadStructAccessData wfr;

public:

	xrtRadIntCL(srTSRWRadStructAccessData* wFront)
	{
		Initialize(wFront);
	}
	~xrtRadIntCL()
	{
		DeallocateMemForxrtRadInt();	
	}

	void Initialize(srTSRWRadStructAccessData* wFront);
	void DeallocateMemForxrtRadInt();
	int calcGaussGrid();
	std::vector<double> readGaussNodes(const char* filename, int vLen); 
	int calcSRField();

};