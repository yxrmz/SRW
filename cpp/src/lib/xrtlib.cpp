#include "xrtlib.h"
//#include "srmagfld.h"
//#include "srradstr.h"


void xrtRadIntCL::Initialize(srTSRWRadStructAccessData* wFront) {

	eE0 = 0; // GeV
	eI = 0; // ring current in Ampere
	lambda0 = 0; //period in mm
	nPeriods = 0; // number of undulator periods
	gamma = 0;
	Bx = 0; // peak magnetic field in Tesla
	Kx = 0;
	By = 0;
	Ky = 0;
	phase = 0;
	energy = 0; // Must be modified for the case of multiple energies
	wfr = wFront;
	//xStart = 0;
	//xEnd = 0;
	//xStep = 0;
	//nx = 0;
	//zStart = 0;
	//zEnd = 0;
	//zStep = 0;
	//nz = 0;
	//eStart = 0;
	//eEnd = 0;
	//eStep = 0;
	//ne = 0;
	tg = readGaussNodes("gaussNodes.dat", gLength);
	ag = readGaussNodes("gaussWeights.dat", gLength);

	int result = calcGaussGrid();
	//gaussDim = 0;
	//gaussIntervals = 0;
}
int xrtRadIntCL::InitCLDevice() {
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
}

int xrtRadIntCL::calcGaussGrid() {

	double tmpXStep = xEnd - xStart;
	std::vector<double> tg(gLength);                // gauss nodes vector 
	std::vector<double> ag(gLength);                // gauss weights vector 	

	cl::Buffer dev_tg = cl::Buffer(context, tg.begin(), tg.end(), true);
	cl::Buffer dev_ag = cl::Buffer(context, ag.begin(), ag.end(), true);

}

int xrtRadIntCL::createKernel() {
	// create the program that we want to execute on the device
	cl::Program uFieldFunctions(context, util::loadProgram("undulator.cl"), true);

	// create a queue (a queue of commands that the GPU will execute)
	cl::CommandQueue queue(context, default_device);

	cl::make_kernel<double, double, double, double, double, double, double, int, double, double,
		int, int, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> uFieldOnMesh(uFieldFunctions, "undulatorFieldOnMesh");

}


std::vector<double> xrtRadIntCL::readGaussNodes(const char* filename, int vLen)
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

int xrtRadIntCL::calcSRField(srTSRWRadStructAccessData* wFront)
{
	try {
		std::vector<std::complex<double>> Is;
		std::vector<std::complex<double>> Ip;

	cl::Buffer dev_ampSigma;
	cl::Buffer dev_ampPi;

	dev_ampSigma = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(std::complex<double>) * nx * nz);
	dev_ampPi = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(std::complex<double>) * nz * nz);

	uFieldOnMesh(cl::EnqueueArgs(queue, cl::NDRange(nx, nz)),
		hR0, hKx, hKy, hPhase, hGamma, phEnergy, hL0, hNp,
		thetaMax, psiMax, gLength, gIntervals, dev_tg, dev_ag, dev_ampSigma, dev_ampPi);
///
	float *pEx0 = wFront.pBaseRadX;
	float *pEz0 = wFront.pBaseRadZ;


	long long Offset = izPerZ + ixPerX + (iLamb << 1);
	float *pEx = pEx0 + Offset, *pEz = pEz0 + Offset;

	*pEx = float(RadIntegValues->real());
	*(pEx + 1) = float(RadIntegValues->imag());
	*pEz = float(RadIntegValues[1].real());
	*(pEz + 1) = float(RadIntegValues[1].imag());
///

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