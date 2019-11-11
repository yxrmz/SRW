//__author__ = "Konstantin Klementiev, Roman Chernikov"
//__date__ = "03 Jul 2016"

#ifdef cl_khr_fp64
#if __OPENCL_VERSION__<120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
#endif
__constant double QUAR = 0.25;
__constant double HALF = 0.5;
__constant double TWO = 2.;
__constant double SIX = 6.;
//__constant bool isHiPrecision = sizeof(TWO) == 8;
//__constant bool isHiPrecision = false;
__constant double2 zero2 = (double2)(0, 0);

__constant double PI2 = (double)6.283185307179586476925286766559;
__constant double PI = (double)3.1415926535897932384626433832795;

__constant double E2W = 1.51926751475e15;
__constant double C = 2.99792458e11;
__constant double SIE0 = 1.602176565e-19;
__constant double SIM0 = 9.10938291e-31;
__constant double A2F = 4.554649401756719e+16;
__constant double sqrtA2F = 213416245.90824193;

__kernel void undulatorFieldOnMesh(const double R0,
	const double Kx,
	const double Ky,
	const double phase,
	const double gamma,
const double w,
const double L0,
const int Np,
const double thetaMax,
//const int nX;
const double psiMax,
//const int nY;
//__global double* gamma,
//__global double* wu,
//__global double* w,
//__global double* ww1,
//__global double* ddphi,
//__global double* ddpsi,
const int jend,
const int jintervals,
__global double* tg,
__global double* ag,
__global double2* Is_gl,
__global double2* Ip_gl)
{
	//unsigned int ii = get_global_id(0);
	unsigned int iX = get_global_id(0);
	unsigned int iY = get_global_id(1);
	unsigned int nX = get_global_size(0);
	unsigned int nY = get_global_size(1);
	int j, gInt;
	double ddtheta = -thetaMax + 2.*thetaMax*iX / (nX - 1);
	double ddpsi = -psiMax + 2.*psiMax*iY / (nY - 1);
	double nz2 = ddtheta * ddtheta + ddpsi * ddpsi;
	double3 beta, betaP, n, nnb;
	double2 eucos;
	double ucos, sinucos, cosucos, sintg, costg, sintgph, costgph, krel;
	double2 Is = zero2;
	double2 Ip = zero2;
	double gamma2 = gamma * gamma;
	double LGE = L0 * gamma2 * E2W;
	double Kx2 = Kx * Kx;
	double Ky2 = Ky * Ky;
	double okk = 1. + 0.5 * (Kx2 + Ky2);
	double wu = PI * C * (2.*gamma2 - okk) / LGE;
	double ww1 = w * (okk + gamma2 * nz2) / (2.*gamma2*wu);
	double wwu2 = w / (wu * wu);
	double revg = 1. / gamma;
	double revg2 = revg * revg;
	double wug = wu * revg;
	double dstep = PI2 / (double)(jintervals);
	n.x = ddtheta;
	n.y = ddpsi;
	//    n.z = sqrt(1. - n.x*n.x - n.y*n.y);
	n.z = 1. - HALF * nz2;

	double gCenter = -PI;
	double ab = sin(PI*Np*ww1) / (PI2 * wu * sin(PI*ww1));
	double normConstant = sqrtA2F * ab * 0.5 * dstep;
	for (gInt = 0; gInt < jintervals; gInt++) {
		gCenter += dstep;
		for (j = 0; j < jend; j++) {
			double gNode = tg[j] + gCenter;
			sintg = sincos(gNode, &costg);
			sintgph = sincos(gNode + phase, &costgph);

			beta.x = Ky * revg * costg;
			beta.y = -Kx * revg * costgph;
			//        beta.z = sqrt(1. - revg2 - beta.x*beta.x - beta.y*beta.y);
			beta.z = 1. - HALF * (revg2 + beta.x*beta.x + beta.y*beta.y);

			betaP.x = -Ky * wug * sintg;
			betaP.y = Kx * wug * sintgph;
			betaP.z = 0.;

			ucos = ww1 * gNode + wwu2 * dot((n - QUAR * beta), betaP);

			betaP.z = -(betaP.x*beta.x + betaP.y*beta.y) / beta.z;
			sinucos = sincos(ucos, &cosucos);
			eucos.x = cosucos;
			eucos.y = sinucos;

			krel = 1. - dot(n, beta);
			nnb = cross(n, cross((n - beta), betaP)) / (krel*krel);
		}

		//        nnb = (n-beta)*dot(n, betaP)/(krel*krel) - betaP/krel;

		Is += (ag[j] * nnb.x) * eucos; 
		Ip += (ag[j] * nnb.y) * eucos;
	}
    mem_fence(CLK_LOCAL_MEM_FENCE);
	Is_gl[iX*nY + iY] = Is * normConstant;			//to be corrected for the ring current and bandwidth
    Ip_gl[iX*nY+iY] = Ip * normConstant;
}
	
/*	
__kernel void undulator_taper(const double alpha,
                              const double Kx,
                              const double Ky,
                              const double phase,
                              const int jend,
                              __global double* gamma,
                              __global double* wu,
                              __global double* w,
                              __global double* ww1,
                              __global double* ddphi,
                              __global double* ddpsi,
                              __global double* tg,
                              __global double* ag,
                              __global double2* Is_gl,
                              __global double2* Ip_gl)
{
    unsigned int ii = get_global_id(0);
    int j;

    double2 eucos;
    double3 n, nnb, beta, betaP;
    double ucos, sinucos, cosucos, sintg, sin2tg, sin2tgph, costg, sintgph, costgph, krel;
    double2 Is = zero2;
    double2 Ip = zero2;
    double Kx2 = Kx * Kx;
    double Ky2 = Ky * Ky;
    double alphaS = alpha * C / wu[ii] / E2W;
    double revg = 1. / gamma[ii];
    double revg2 = revg * revg;
    double wug = wu[ii] * revg;
    double wgwu = w[ii] * revg / wu[ii];
    n.x = ddphi[ii];
    n.y = ddpsi[ii];
    //n.z = sqrt(1. - n.x*n.x - n.y*n.y);
    n.z = 1 - HALF*(n.x*n.x + n.y*n.y);

    for (j=0; j<jend; j++) {
        sintg = sincos(tg[j], &costg);
        sintgph = sincos(tg[j] + phase, &costgph);
        sin2tg = TWO * sintg * costg; //sin(2*tg[j]);
        sin2tgph = TWO * sintgph * costgph; //sin(2*(tg[j] + phase));
        ucos = ww1[ii] * tg[j] + wgwu *
            (-Ky * n.x * (sintg + alphaS * (1. - costg - tg[j] * sintg)) +
             Kx * n.y * sintgph + 0.125 * revg *
                   (Ky2 * (sin2tg - TWO * alphaS *
                    (tg[j] * tg[j] + costg * costg + tg[j] * sin2tg)) +
                    Kx2 * sin2tgph));

        sinucos = sincos(ucos, &cosucos);
        eucos.x = cosucos;
        eucos.y = sinucos;

        beta.x = Ky * revg * costg * (1 - alphaS * tg[j]);
        beta.y = -Kx * revg * costgph;
        beta.z = 1 - HALF*(revg2 + beta.x*beta.x + beta.y*beta.y);

        betaP.x = -Ky * wug * (alphaS*costg + (1 - alphaS*tg[j])*sintg);
        betaP.y = Kx * wug * sintgph;
        betaP.z = wu[ii] * revg2 *
            (Kx2*sintgph*costgph + Ky2*(1 - alphaS*tg[j])*
             (alphaS * costg * costg + (1 - alphaS*tg[j])*sintg*costg));

        if (isHiPrecision) {
            krel = 1. - dot(n, beta);
            nnb = cross(n, cross((n - beta), betaP))/(krel*krel); }
        else
            nnb = (n - beta) * w[ii];

        Is += (ag[j] * nnb.x) * eucos;
        Ip += (ag[j] * nnb.y) * eucos; }

    mem_fence(CLK_LOCAL_MEM_FENCE);
    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
}

__kernel void undulator_nf(const double R0,
                            const double L0,
                            const double Kx,
                            const double Ky,
                            const double phase,
                            const int jend,
                            __global double* gamma,
                            __global double* wu,
                            __global double* w,
                            __global double* ww1,
                            __global double* ddphi,
                            __global double* ddpsi,
                            __global double* tg,
                            __global double* ag,
                            __global double2* Is_gl,
                            __global double2* Ip_gl)
{
    unsigned int ii = get_global_id(0);
    int j;

    double2 eucos;
    double ucos, sinucos, cosucos, sintg, costg, sintgph, costgph, krel;
    double2 Is = zero2;
    double2 Ip = zero2;
    double3 r, r0, n, nnb, beta, betaP;
    double Kx2 = Kx * Kx;
    double Ky2 = Ky * Ky;
    double revg = 1. / gamma[ii];
    double revg2 = revg * revg;
    double wug = wu[ii] * revg;
    double wwu = w[ii] / wu[ii];
    double betam = 1 - (1 + HALF * Kx2 + HALF * Ky2) * HALF * revg2;
    double wR0 = R0 * PI2 / L0;

    r0.x = wR0 * tan(ddphi[ii]);
    r0.y = wR0 * tan(ddpsi[ii]);
    r0.z = wR0 * cos(sqrt(ddphi[ii]*ddphi[ii] + ddpsi[ii]*ddpsi[ii]));

    n.x = ddphi[ii];
    n.y = ddpsi[ii];
//    n.z = sqrt(1. - n.x*n.x - n.y*n.y);
    n.z = 1 - HALF*(n.x*n.x + n.y*n.y);

    for (j=0; j<jend; j++) {
        sintg = sincos(tg[j], &costg);
        sintgph = sincos(tg[j] + phase, &costgph);

        r.x = Ky * revg * sintg;
        r.y = -Kx * revg * sintgph;
        r.z = betam * tg[j] - QUAR * revg2 *
        (Ky2 * sintg * costg + Kx2 * sintgph * costgph);

        ucos = wwu * (tg[j] + length(r0 - r));

        sinucos = sincos(ucos, &cosucos);
        eucos.x = cosucos;
        eucos.y = sinucos;

        beta.x = Ky * revg * costg;
        beta.y = -Kx * revg * costgph;
//        beta.z = sqrt(1. - revg2 - beta.x*beta.x - beta.y*beta.y);
        beta.z = 1 - HALF*(revg2 + beta.x*beta.x + beta.y*beta.y);

        betaP.x = -Ky * wug * sintg;
        betaP.y = Kx * wug * sintgph;
        betaP.z = wu[ii] * revg2 * (Ky2*sintg*costg + Kx2*sintgph*costgph);
        //betaP.z = wu[ii]/beta.z*revg2*(Ky2*sintg*costg + Kx2*sintgph*costgph);

        if (isHiPrecision) {
            krel = 1. - dot(n, beta);
            nnb = cross(n, cross((n - beta), betaP))/(krel*krel); }
        else
            nnb = (n - beta) * w[ii];

        Is += (ag[j] * nnb.x) * eucos;
        Ip += (ag[j] * nnb.y) * eucos; }

    mem_fence(CLK_LOCAL_MEM_FENCE);

    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
}

__kernel void undulator_full(const double alpha,
                             const double Kx,
                             const double Ky,
                             const double phase,
                             const int jend,
                             __global double* gamma,
                             __global double* wu,
                             __global double* w,
                             __global double* ww1,
                             __global double* ddphi,
                             __global double* ddpsi,
                             __global double* beta0x,
                             __global double* beta0y,
                             __global double* tg,
                             __global double* ag,
                             __global double2* Is_gl,
                             __global double2* Ip_gl)
{
    unsigned int ii = get_global_id(0);
    int j;

    double2 eucos;
    double2 beta0 = (double2)(beta0x[ii], beta0y[ii]);
    double ucos, sinucos, cosucos, sintg, costg, sintgph, costgph, krel;
    double2 Is = zero2;
    double2 Ip = zero2;
    double3 r, n, nnb, beta, betaP;
    double Kx2 = Kx * Kx;
    double Ky2 = Ky * Ky;
    double revg = 1. / gamma[ii];
    double revg2 = revg * revg;
    double wug = wu[ii] * revg;
    double wwu = w[ii] / wu[ii];
    double betam = 1 - (1 + HALF * Kx2 + HALF * Ky2) * HALF * revg2;

    n.x = ddphi[ii];
    n.y = ddpsi[ii];
    //double betax0 = 0; //1e-6;
    //double betay0 = 0;
    //double rx0 = 2e-2*PI2/L0/wu[ii];
    //double ry0 = 0;
    //n.z = sqrt(1. - n.x*n.x - n.y*n.y);
    n.z = 1 - HALF*(n.x*n.x + n.y*n.y);

    for (j=0; j<jend; j++) {
        sintg = sincos(tg[j], &costg);
        sintgph = sincos(tg[j] + phase, &costgph);

        beta.x = Ky*costg*revg + beta0.x;
        beta.y= -Kx*costgph*revg + beta0.y;
        beta.z = 1. - HALF*(revg2 + beta.x*beta.x + beta.y*beta.y);
        
        r.x = Ky*sintg*revg + beta0.x*tg[j];
        r.y = -Kx*sintgph*revg + beta0.y*tg[j];
        r.z = -QUAR*revg2*(Kx2*sintgph*costgph + Ky2*sintg*costg) + tg[j]*betam +
            Kx*beta0.y*sintgph*revg - Ky*beta0.x*sintg*revg -
            tg[j]*HALF*(beta0.x*beta0.x + beta0.y*beta0.y);


        //r.x = Ky / gamma[ii] * sintg;
        //r.y = -Kx / gamma[ii] * sintgph;
        //r.z = betam * tg[j] - 0.25 / gamma2 *
        //(Ky2 * sintg * costg + Kx2 * sintgph * costgph);

        ucos = wwu * (tg[j] - dot(n, r));

        sinucos = sincos(ucos, &cosucos);
        eucos.x = cosucos;
        eucos.y = sinucos;

        //beta.x = Ky / gamma[ii] * costg;
        //beta.y = -Kx / gamma[ii] * costgph;
        //beta.z = sqrt(1. - 1./gamma2 - beta.x*beta.x - beta.y*beta.y);
        //beta.z = 1 - 0.5*(1./gamma2 + beta.x*beta.x + beta.y*beta.y);

        betaP.x = -Ky * wug * sintg;
        betaP.y = Kx * wug * sintgph;
        betaP.z = wu[ii] * revg2 * (Ky2*sintg*costg + Kx2*sintgph*costgph);
        //betaP.z = wu[ii]/beta.z/gamma2*(Ky2*sintg*costg + Kx2*sintgph*costgph);

        if (isHiPrecision) {
            krel = 1. - dot(n, beta);
            nnb = cross(n, cross((n - beta), betaP))/(krel*krel); }
        else
            nnb = (n - beta) * w[ii];

        Is += (ag[j] * nnb.x) * eucos;
        Ip += (ag[j] * nnb.y) * eucos; }

    mem_fence(CLK_LOCAL_MEM_FENCE);

    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
}

__kernel void undulator_nf_full(const double R0,
                            const double L0,
                            const double Kx,
                            const double Ky,
                            const double phase,
                            const int jend,
                            __global double* gamma,
                            __global double* wu,
                            __global double* w,
                            __global double* ww1,
                            __global double* ddphi,
                            __global double* ddpsi,
                            __global double* beta0x,
                            __global double* beta0y,
                            __global double* tg,
                            __global double* ag,
                            __global double2* Is_gl,
                            __global double2* Ip_gl)
{
    unsigned int ii = get_global_id(0);
    int j;

    double2 eucos;
    double ucos, sinucos, cosucos, sintg, costg, sintgph, costgph, krel;
    double2 beta0 = (double2)(beta0x[ii], beta0y[ii]);
    double2 Is = zero2;
    double2 Ip = zero2;
    double3 r, r0, n, nnb, beta, betaP;
    double Kx2 = Kx * Kx;
    double Ky2 = Ky * Ky;
    double revg = 1. / gamma[ii];
    double revg2 = revg * revg;
    double wug = wu[ii] * revg;
    double wwu = w[ii] / wu[ii];
    double betam = 1 - (1 + HALF * Kx2 + HALF * Ky2) * HALF * revg2;
    double wR0 = R0 * PI2 / L0;

    r0.x = wR0 * tan(ddphi[ii]);
    r0.y = wR0 * tan(ddpsi[ii]);
    r0.z = wR0 * cos(sqrt(ddphi[ii]*ddphi[ii] + ddpsi[ii]*ddpsi[ii]));

    n.x = ddphi[ii];
    n.y = ddpsi[ii];
    //n.z = sqrt(1. - n.x*n.x - n.y*n.y);
    n.z = 1 - HALF*(n.x*n.x + n.y*n.y);

    for (j=0; j<jend; j++) {
        sintg = sincos(tg[j], &costg);
        sintgph = sincos(tg[j] + phase, &costgph);

        //r.x = Ky / gamma[ii] * sintg;
        //r.y = -Kx / gamma[ii] * sintgph;
        //r.z = betam * tg[j] - 0.25 / gamma2 *
        //(Ky2 * sintg * costg + Kx2 * sintgph * costgph);
        beta.x = Ky*costg*revg + beta0.x;
        beta.y= -Kx*costgph*revg + beta0.y;
        beta.z = 1. - HALF*(revg2 + beta.x*beta.x + beta.y*beta.y);
        
        r.x = Ky*sintg*revg + beta0.x*tg[j];
        r.y = -Kx*sintgph*revg + beta0.y*tg[j];
        r.z = -QUAR*revg2*(Kx2*sintgph*costgph + Ky2*sintg*costg) + tg[j]*betam +
            Kx*beta0.y*sintgph*revg - Ky*beta0.x*sintg*revg -
            tg[j]*HALF*(beta0.x*beta0.x + beta0.y*beta0.y);

        ucos = wwu * (tg[j] + length(r0 - r));

        sinucos = sincos(ucos, &cosucos);
        eucos.x = cosucos;
        eucos.y = sinucos;

        //beta.x = Ky / gamma[ii] * costg;
        //beta.y = -Kx / gamma[ii] * costgph;
        //beta.z = sqrt(1. - 1./gamma2 - beta.x*beta.x - beta.y*beta.y);
        //beta.z = 1 - 0.5*(1./gamma2 + beta.x*beta.x + beta.y*beta.y);

        betaP.x = -Ky * wug * sintg;
        betaP.y = Kx * wug * sintgph;
        betaP.z = wu[ii] * revg2 * (Ky2*sintg*costg + Kx2*sintgph*costgph);
        //betaP.z = wu[ii]/beta.z/gamma2*(Ky2*sintg*costg + Kx2*sintgph*costgph);

        if (isHiPrecision) {
            krel = 1. - dot(n, beta);
            nnb = cross(n, cross((n - beta), betaP))/(krel*krel); }
        else
            nnb = (n - beta) * w[ii];

        Is += (ag[j] * nnb.x) * eucos;
        Ip += (ag[j] * nnb.y) * eucos; }

    mem_fence(CLK_LOCAL_MEM_FENCE);

    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
}

*/
/*
__kernel void undulator_nf_byparts(const double alpha,
                        const double Kx,
                        const double Ky,
                        const double phase,
                        const int jend,
                        const double alim,
                        const double blim,
                        __global double* gamma,
                        __global double* wu,
                        __global double* w,
                        __global double* ww1,
                        __global double* ddphi,
                        __global double* ddpsi,
                        __global double* tg,
                        __global double* ag,
                        __global double2* Is_gl,
                        __global double2* Ip_gl)
{
    unsigned int ii = get_global_id(0);
    int j;

    double2 eg;
    double f, fP, varx;
    double g, gP, gPP, sing, cosg, sintgph, costgph;
    double sintg, costg;
    double sinph, cosph;
    double2 Is = zero2;
    double2 Ip = zero2;
    double gam = gamma[ii];
    double wwu = w[ii] / wu[ii];
    double Kx2 = Kx * Kx;
    double Ky2 = Ky * Ky;
    double Ky2Kx2;
    double gamma2 = gam * gam;
    double phi = ddphi[ii];
    double psi = ddpsi[ii];
    double3 r, rP, rPP, n;
    double betam = 1 - (0.5 + 0.25*Kx2 + 0.25*Ky2) / gamma2;


    //n.x = phi;
    //n.y = psi;
    //n.z = sqrt(1 - phi*phi - psi*psi);

    r0.x = wR0 * tan(ddphi[ii]);
    r0.y = wR0 * tan(ddpsi[ii]);
    r0.z = wR0 * cos(sqrt(ddphi[ii]*ddphi[ii] + ddpsi[ii]*ddpsi[ii]));

    sinph = sincos(phase, &cosph);
    for (j=0; j<jend; j++) {
        varx = tg[j];
        sintg = sincos(varx, &costg);
        sintgph = sintg*cosph + costg*sinph;
        costgph = costg*cosph - sintg*sinph;
        Ky2Kx2 = (Ky2 * sintg*costg + Kx2 * sintgph*costgph) / gamma2;

        r.x = Ky / gam * sintg;
        r.y = -Kx / gam * sintgph;
        r.z = betam * varx - 0.25 * Ky2Kx2;

        g = wwu * (varx + length(r0 - r));

        rP.x = Ky / gam * costg;
        rP.y = -Kx / gam * costgph;
        rP.z = betam - 0.25 / gamma2 *
            (Ky2 * (costg*costg - sintg*sintg) +
             Kx2 * (costgph*costgph - sintgph*sintgph));
        gP = wwu * (1 + dot(n, rP));

        rPP.x = -r.x;
        rPP.y = -r.y;
        rPP.z = Ky2Kx2;
        gPP = -wwu * dot(n, rPP);

        sing = sincos(g, &cosg);
        eg.x = -sing;
        eg.y = cosg;

        f = phi - Ky / gam * costg;
        fP = Ky / gam * sintg;
        Is += ag[j] * (fP - f*gPP/gP)/gP * eg;

        f = psi + Kx / gam * costgph;
        fP = -Kx / gam * sintgph;
        Ip += ag[j] * (fP - f*gPP/gP)/gP * eg;}


    mem_fence(CLK_LOCAL_MEM_FENCE);
    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
    }
*/
/*
double2 f_beta(double Bx, double By, double Bz,
               double emcg,
               double2 beta)
{
    return emcg*(double2)(beta.y*Bz - By,
                          Bx - beta.x*Bz);
}

double3 f_traj(double revgamma, double2 beta)
{
    return (double3)(beta.x,
                     beta.y,
                     sqrt(revgamma-beta.x*beta.x-beta.y*beta.y));
}

double2 next_beta_rk(double2 beta, int iZeroStep, int iHalfStep,
                     int iFullStep, double rkStep, double emcg,
                     __global double* Bx,
                     __global double* By,
                     __global double* Bz)
{
    double2 k1Beta, k2Beta, k3Beta, k4Beta;

    k1Beta = rkStep * f_beta(Bx[iZeroStep],
                             By[iZeroStep],
                             Bz[iZeroStep],
                             emcg, beta);
    k2Beta = rkStep * f_beta(Bx[iHalfStep],
                             By[iHalfStep],
                             Bz[iHalfStep],
                             emcg, beta + HALF*k1Beta);
    k3Beta = rkStep * f_beta(Bx[iHalfStep],
                             By[iHalfStep],
                             Bz[iHalfStep],
                             emcg, beta + HALF*k2Beta);
    k4Beta = rkStep * f_beta(Bx[iFullStep],
                             By[iFullStep],
                             Bz[iFullStep],
                             emcg, beta + k3Beta);
    return beta + (k1Beta + TWO*k2Beta + TWO*k3Beta + k4Beta) / SIX;
}

double8 next_traj_rk(double2 beta, double3 traj, int iZeroStep, int iHalfStep,
                     int iFullStep, double rkStep, double emcg,
                     double revgamma,
                     __global double* Bx,
                     __global double* By,
                     __global double* Bz)
{
    double2 k1Beta, k2Beta, k3Beta, k4Beta;
    double3 k1Traj, k2Traj, k3Traj, k4Traj;

    k1Beta = rkStep * f_beta(Bx[iZeroStep],
                             By[iZeroStep],
                             Bz[iZeroStep],
                             emcg, beta);
    k1Traj = rkStep * f_traj(revgamma, beta);

    k2Beta = rkStep * f_beta(Bx[iHalfStep],
                             By[iHalfStep],
                             Bz[iHalfStep],
                             emcg, beta + HALF*k1Beta);
    k2Traj = rkStep * f_traj(revgamma, beta + HALF*k1Beta);

    k3Beta = rkStep * f_beta(Bx[iHalfStep],
                             By[iHalfStep],
                             Bz[iHalfStep],
                             emcg, beta + HALF*k2Beta);
    k3Traj = rkStep * f_traj(revgamma, beta + HALF*k2Beta);

    k4Beta = rkStep * f_beta(Bx[iFullStep],
                             By[iFullStep],
                             Bz[iFullStep],
                             emcg, beta + k3Beta);
    k4Traj = rkStep * f_traj(revgamma, beta + k3Beta);

    return (double8)(beta + (k1Beta + TWO*k2Beta + TWO*k3Beta + k4Beta)/SIX,
                     traj + (k1Traj + TWO*k2Traj + TWO*k3Traj + k4Traj)/SIX,
                     0., 0., 0.);
}


__kernel void undulator_custom(const int jend,
                                const int nwt,
                                const double lUnd,
                                __global double* gamma,
                                __global double* w,
                                __global double* ddphi,
                                __global double* ddpsi,
                                __global double* tg,
                                __global double* ag,
                                __global double* Bx,
                                __global double* By,
                                __global double* Bz,
                                __global double2* Is_gl,
                                __global double2* Ip_gl)
{
    unsigned int ii = get_global_id(0);
    int j, k, jb;
    int iBase, iZeroStep, iHalfStep, iFullStep;

    double ucos, sinucos, cosucos, rkStep, wu_int, betam_int, krel;
    double revg = 1. / gamma[ii];
    double revg2 = revg * revg;
    double emcg = lUnd * SIE0 / SIM0 / C * revg / PI2;
    double revgamma = 1. - revg2;
    double8 betaTraj;
    double2 eucos;
    double2 Is = zero2;
    double2 Ip = zero2;
    double2 beta, beta0;
    double3 traj, n, traj0, betaC, betaP, nnb;

    n.x = ddphi[ii];
    n.y = ddpsi[ii];
    n.z = 1. - HALF*(n.x*n.x + n.y*n.y);

    beta = zero2;
    beta0 = zero2;
    betam_int = 0;

    for (j=1; j<jend; j++) {
        iBase = 2*(j-1)*nwt;
        rkStep = (tg[j] - tg[j-1]) / nwt;
        for (k=0; k<nwt; k++) {
            iZeroStep  = iBase + 2*k;
            iHalfStep = iBase + 2*k + 1;
            iFullStep = iBase + 2*(k + 1);
            beta = next_beta_rk(beta, iZeroStep, iHalfStep, iFullStep,
                                rkStep, emcg, Bx, By, Bz);
            beta0 += beta * rkStep; } }

    mem_fence(CLK_LOCAL_MEM_FENCE);
    beta0 /= -(tg[jend-1] - tg[0]);
    beta = beta0;
    traj = (double3)(0., 0., 0.);
    traj0 = (double3)(0., 0., 0.);

    for (j=1; j<jend; j++) {
        iBase = (j-1)*2*nwt;
        rkStep = (tg[j] - tg[j-1]) / nwt;
        for (k=0; k<nwt; k++) {
            iZeroStep  = iBase + 2*k;
            iHalfStep = iBase + 2*k + 1;
            iFullStep = iBase + 2*(k + 1);
            betaTraj = next_traj_rk(beta, traj,
                                    iZeroStep, iHalfStep, iFullStep,
                                    rkStep, emcg, revgamma, Bx, By, Bz);
            beta = betaTraj.s01;
            traj = betaTraj.s234;
            traj0 += traj * rkStep;
            betam_int += rkStep * sqrt(revgamma - beta.x*beta.x -
                                       beta.y*beta.y); } }

    mem_fence(CLK_LOCAL_MEM_FENCE);
    traj0 /= -(tg[jend-1] - tg[0]);
    beta = beta0;
    traj = traj0;
    betam_int /= (tg[jend-1] - tg[0]);
    wu_int = PI2 * C * betam_int / lUnd / E2W;

    for (j=1; j<jend; j++) {
        iBase = 2*(j-1)*nwt;
        rkStep = (tg[j] - tg[j-1]) / nwt;
        for (k=0; k<nwt; k++) {
            iZeroStep  = iBase + 2*k;
            iHalfStep = iBase + 2*k + 1;
            iFullStep = iBase + 2*(k + 1);
            betaTraj = next_traj_rk(beta, traj,
            iZeroStep, iHalfStep, iFullStep,
            rkStep, emcg, revgamma, Bx, By, Bz);
            beta = betaTraj.s01;
            traj = betaTraj.s234; }

        mem_fence(CLK_LOCAL_MEM_FENCE);
        ucos = w[ii] / wu_int * (tg[j]  - dot(n, traj));
        sinucos = sincos(ucos, &cosucos);
        eucos.x = cosucos;
        eucos.y = sinucos;

        jb = 2*j*nwt;

        betaC.x = beta.x;
        betaC.y = beta.y;
        betaC.z = 1 - HALF*(revg2 + betaC.x*betaC.x + betaC.y*betaC.y);

        betaP.x = wu_int * emcg * (betaC.y*Bz[jb] - By[jb]);
        betaP.y = wu_int * emcg * (-betaC.x*Bz[jb] + Bx[jb]);
        betaP.z = wu_int * emcg * (betaC.x*By[jb] - betaC.y*Bx[jb]);

        if (isHiPrecision) {
            krel = 1. - dot(n, betaC);
            nnb = cross(n, cross((n - betaC), betaP))/(krel*krel); }
        else
            nnb = (n - betaC) * w[ii];

        Is += (ag[j] * nnb.x) * eucos;
        Ip += (ag[j] * nnb.y) * eucos; }

        //Is += ag[j] * (ddphi[ii] - beta.x) * eucos;
        //Ip += ag[j] * (ddpsi[ii] - beta.y) * eucos; }

    mem_fence(CLK_LOCAL_MEM_FENCE);

    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
}

__kernel void get_trajectory(const int jend,
                             const int nwt,
                             const double emcg,
                             const double gamma,
                             __global double* tg,
                             __global double* Bx,
                             __global double* By,
                             __global double* Bz,
                             __global double* betax,
                             __global double* betay,
                             __global double* betazav,
                             __global double* trajx,
                             __global double* trajy,
                             __global double* trajz)
{
    int j, k;
    int iBase, iZeroStep, iHalfStep, iFullStep;

    double rkStep;
    double revgamma = 1. - 1./gamma/gamma;
    double betam_int = 0;
    double2 beta, beta0;
    double3 traj, traj0;
    double8 betaTraj;

    beta = zero2;
    beta0 = zero2;

    for (j=1; j<jend; j++) {
        iBase = (j-1)*2*nwt;
        rkStep = (tg[j] - tg[j-1]) / nwt;
        for (k=0; k<nwt; k++) {
            iZeroStep  = iBase + 2*k;
            iHalfStep = iBase + 2*k + 1;
            iFullStep = iBase + 2*(k + 1);
            beta = next_beta_rk(beta, iZeroStep, iHalfStep, iFullStep,
                                rkStep, emcg, Bx, By, Bz);
            beta0 += beta * rkStep; } }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    beta0 /= -(tg[jend-1] - tg[0]);
    beta = beta0;
    traj = (double3)(0., 0., 0.);
    traj0 = (double3)(0., 0., 0.);

    for (j=1; j<jend; j++) {
        iBase = 2*(j-1)*nwt;
        rkStep = (tg[j] - tg[j-1]) / nwt;
        for (k=0; k<nwt; k++) {
            iZeroStep  = iBase + 2*k;
            iHalfStep = iBase + 2*k + 1;
            iFullStep = iBase + 2*(k + 1);
            betaTraj = next_traj_rk(beta, traj,
                                    iZeroStep, iHalfStep, iFullStep,
                                    rkStep, emcg, revgamma, Bx, By, Bz);
            beta = betaTraj.s01;
            traj = betaTraj.s234;
            traj0 += traj * rkStep;
            betam_int += rkStep*sqrt(revgamma - beta.x*beta.x -
                                     beta.y*beta.y); } }
    mem_fence(CLK_LOCAL_MEM_FENCE);
    traj0 /= -(tg[jend-1] - tg[0]);
    beta = beta0;
    traj = traj0;
    betam_int /= tg[jend-1] - tg[0];

    for (j=1; j<jend; j++) {
        iBase = 2*(j-1)*nwt;
        rkStep = (tg[j] - tg[j-1]) / nwt;
        for (k=0; k<nwt; k++) {
            iZeroStep  = iBase + 2*k;
            iHalfStep = iBase + 2*k + 1;
            iFullStep = iBase + 2*(k + 1);
            betaTraj = next_traj_rk(beta, traj,
                                    iZeroStep, iHalfStep, iFullStep,
                                    rkStep, emcg, revgamma, Bx, By, Bz);
            beta = betaTraj.s01;
            traj = betaTraj.s234; }

        betax[j] = beta.x;
        betay[j] = beta.y;
        betazav[j] = betam_int;
        trajx[j] = traj.x;
        trajy[j] = traj.y;
        trajz[j] = traj.z; }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void undulator_custom_filament(const int jend,
                                        const int nwt,
                                        const double emcg,
                                        const double gamma2,
                                        const double wu,
                                        const double L0,
                                        const double R0,
                                        __global double* w,
                                        __global double* ddphi,
                                        __global double* ddpsi,
                                        __global double* tg,
                                        __global double* ag,
                                        __global double* Bx,
                                        __global double* By,
                                        __global double* Bz,
                                        __global double* betax,
                                        __global double* betay,
                                        __global double* trajx,
                                        __global double* trajy,
                                        __global double* trajz,
                                        __global double2* Is_gl,
                                        __global double2* Ip_gl)
{
    unsigned int ii = get_global_id(0);
    int j, jb;

    double ucos, sinucos, cosucos, wR0, krel;
    double revg2 = 1./gamma2;

    double2 eucos;
    double2 Is = (double2)(0., 0.);
    double2 Ip = (double2)(0., 0.);

    double3 traj, n, r0, betaC, betaP, nnb;

    n.x = ddphi[ii];
    n.y = ddpsi[ii];
    n.z = 1. - HALF*(n.x*n.x + n.y*n.y);

    if (R0>0) {
        wR0 = R0 * PI2 / L0;
        r0.x = wR0 * tan(ddphi[ii]);
        r0.y = wR0 * tan(ddpsi[ii]);
        r0.z = wR0 * cos(sqrt(ddphi[ii]*ddphi[ii] + ddpsi[ii]*ddpsi[ii])); }

    for (j=1; j<jend; j++) {
        traj.x = trajx[j];
        traj.y = trajy[j];
        traj.z = trajz[j];
        if (R0 > 0) {
            ucos = w[ii] / wu * (tg[j] + length(r0 - traj)); }
        else {
            ucos = w[ii] / wu * (tg[j] - dot(n, traj)); }
        sinucos = sincos(ucos, &cosucos);
        eucos.x = cosucos;
        eucos.y = sinucos;

        jb = 2*j*nwt;

        betaC.x = betax[j];
        betaC.y = betay[j];
        betaC.z = 1 - HALF*(revg2 + betaC.x*betaC.x + betaC.y*betaC.y);

        betaP.x = wu * emcg * (betaC.y*Bz[jb] - By[jb]);
        betaP.y = wu * emcg * (-betaC.x*Bz[jb] + Bx[jb]);
        betaP.z = wu * emcg * (betaC.x*By[jb] - betaC.y*Bx[jb]);

        if (isHiPrecision) {
            krel = 1. - dot(n, betaC);
            nnb = cross(n, cross((n - betaC), betaP))/(krel*krel); }
        else
            nnb = (n - betaC) * w[ii];

        Is += (ag[j] * nnb.x) * eucos;
        Ip += (ag[j] * nnb.y) * eucos; }

        //Is += ag[j] * (n.x - betax[j]) * eucos;
        //Ip += ag[j] * (n.y - betay[j]) * eucos; }

    mem_fence(CLK_LOCAL_MEM_FENCE);
    Is_gl[ii] = Is;
    Ip_gl[ii] = Ip;
}
*/
