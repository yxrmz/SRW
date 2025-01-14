#############################################################################
# SRWLib for Python v 0.066
#############################################################################

from __future__ import print_function #Python 2.7 compatibility
import srwlpy as srwl
from array import *
from math import *
from copy import *
#from random import *
import random
import sys
import traceback

#****************************************************************************
#****************************************************************************
# SRWLib Python Classes
#****************************************************************************
#****************************************************************************
class SRWLParticle(object):
    """Charged Particle"""

    def __init__(self, _x=0, _y=0, _z=0, _xp=0, _yp=0, _gamma=1, _relE0=1, _nq=-1):
        """
        :param _x: instant coordinates [m]
        :param _y: instant coordinates [m]
        :param _z: instant coordinates [m]
        :param _xp: instant transverse velocity component btx = vx/c (angles for relativistic particle)
        :param _yp: instant transverse velocity component bty = vy/c (angles for relativistic particle)
        :param _gamma: relative energy
        :param _relE0: rest mass (energy) in units of electron rest mass, e.g. 1 for electron, 1836.1526988 (=938.272013/0.510998902) for proton
        :param _nq: charge of the particle related to absolute value of electron charge, -1 for electron, 1 for positron and for proton
        """
        self.x = _x
        self.y = _y
        self.z = _z
        self.xp = _xp
        self.yp = _yp
        self.gamma = _gamma
        self.relE0 = _relE0
        self.nq = _nq

#****************************************************************************
class SRWLPartBeam(object):
    """Particle Beam"""

    def __init__(self, _Iavg=0, _nPart=0, _partStatMom1=None, _arStatMom2=None):
        """
        :param _Iavg: average current [A]
        :param _nPart: number of electrons (in a bunch)
        :param _partStatMom1: particle type and 1st order statistical moments
        :param _arStatMom2: 2nd order statistical moments
            [0]: <(x-x0)^2>
            [1]: <(x-x0)*(xp-xp0)>
            [2]: <(xp-xp0)^2>
            [3]: <(y-y0)^2>
            [4]: <(y-y0)*(yp-yp0)>
            [5]: <(yp-yp0)^2>
            [6]: <(x-x0)*(y-y0)>
            [7]: <(xp-xp0)*(y-y0)>
            [8]: <(x-x0)*(yp-yp0)>
            [9]: <(xp-xp0)*(yp-yp0)>
            [10]: <(E-E0)^2>/E0^2
            [11]: <(s-s0)^2>
            [12]: <(s-s0)*(E-E0)>/E0
            [13]: <(x-x0)*(E-E0)>/E0
            [14]: <(xp-xp0)*(E-E0)>/E0
            [15]: <(y-y0)*(E-E0)>/E0
            [16]: <(yp-yp0)*(E-E0)>/E0
            [17]: <(x-x0)*(s-s0)>
            [18]: <(xp-xp0)*(s-s0)>
            [19]: <(y-y0)*(s-s0)>
            [20]: <(yp-yp0)*(s-s0)>
        """
        self.Iavg = _Iavg
        self.nPart = _nPart
        self.partStatMom1 = SRWLParticle() if _partStatMom1 is None else _partStatMom1
        self.arStatMom2 = array('d', [0] * 21) if _arStatMom2 is None else _arStatMom2

#****************************************************************************
class SRWLMagFld(object):
    """Magnetic Field (base class)"""
    
class SRWLMagFld3D(SRWLMagFld):
    """Magnetic Field: Arbitrary 3D"""
    
    def __init__(self, _arBx=None, _arBy=None, _arBz=None, _nx=0, _ny=0, _nz=0, _rx=0, _ry=0, _rz=0, _nRep=1, _interp=1, _arX=None, _arY=None, _arZ=None):
        """
        :param _arBx: horizontal magnetic field component array [T]
        :param _arBy: vertical magnetic field component array [T]
        :param _arBz: longitudinal magnetic field component array [T]
        :param _nx: number of magnetic field data points in the horizontal direction
        :param _ny: number of magnetic field data points in the vertical direction
        :param _nz: number of magnetic field data points in the longitudinal direction
        :param _rx: range of horizontal coordinate for which the field is defined [m]
        :param _ry: range of vertical coordinate for which the field is defined [m]
        :param _rz: range of longitudinal coordinate for which the field is defined [m]
        :param _nRep: "number of periods", i.e. number of times the field is "repeated" in the longitudinal direction
        :param _interp: interpolation method to use (e.g. for trajectory calculation), 1- bi-linear (3D), 2- (bi-)quadratic (3D), 3- (bi-)cubic (3D)
        :param _arX: optional array of horizontal transverse coordinate of an irregular 3D mesh (if this array is defined, rx will be ignored)
        :param _arY: optional array of vertical transverse coordinate of an irregular 3D mesh (if this array is defined, ry will be ignored)
        :param _arZ: optional array of longitudinal coordinate of an irregular 3D mesh (if this array is defined, rz will be ignored)
        """
        self.arBx = array('d') if _arBx is None else _arBx
        self.arBy = array('d') if _arBy is None else _arBy
        self.arBz = array('d') if _arBz is None else _arBz
        self.nx = _nx
        self.ny = _ny
        self.nz = _nz
        self.rx = _rx
        self.ry = _ry
        self.rz = _rz
        self.arX = array('d') if _arX is None else _arX
        self.arY = array('d') if _arY is None else _arY
        self.arZ = array('d') if _arZ is None else _arZ
        self.nRep = _nRep
        self.interp = _interp

class SRWLMagFldM(SRWLMagFld):
    """Magnetic Field: Multipole Magnet"""
    
    def __init__(self, _G=0, _m=2, _n_or_s='n', _Leff=0, _Ledge=0):
        """
        :param _G: field parameter [T] for dipole, [T/m] for quadrupole (negative means defocusing for x), [T/m^2] for sextupole, [T/m^3] for octupole
        :param _m: multipole order 1 for dipole, 2 for quadrupoole, 3 for sextupole, 4 for octupole
        :param _n_or_s: normal ('n') or skew ('s')
        :param _Leff: effective length [m]
        :param _Ledge: "soft" edge length for field variation from 10% to 90% [m]; G/(1 + ((z-zc)/d)^2)^2 fringe field dependence is assumed
        """
        self.G = _G
        self.m = _m
        self.n_or_s = _n_or_s
        self.Leff = _Leff
        self.Ledge = _Ledge

class SRWLMagFldS(SRWLMagFld):
    """Magnetic Field: Solenoid"""
    
    def __init__(self, _B=0, _Leff=0):
        """
        :param _B: magnetic field [T]
        :param _Leff: effective length [m]
        """
        self.B = _B
        self.Leff = _Leff

class SRWLMagFldH(SRWLMagFld):
    """Magnetic Field: Undulator Harmonic"""
    
    def __init__(self, _n=1, _h_or_v='v', _B=0, _ph=0, _s=1, _a=1):
        """
        :param _n: harmonic number
        :param _h_or_v: magnetic field plane horzontal ('h') or vertical ('v')
        :param _B: magnetic field amplitude [T]
        :param _ph: initial phase [rad]
        :param _s: symmetry vs longitudinal position 1 - symmetric (B ~ cos(2*Pi*n*z/per + ph)) , -1 - anti-symmetric (B ~ sin(2*Pi*n*z/per + ph))
        :param _a: coefficient for transverse depenednce B*cosh(2*Pi*n*a*y/per)*cos(2*Pi*n*z/per + ph)
        """
        self.n = _n
        self.h_or_v = _h_or_v
        self.B = _B
        self.ph = _ph
        self.s = _s
        self.a = _a

class SRWLMagFldU(SRWLMagFld):
    """Magnetic Field: Undulator"""
    
    def __init__(self, _arHarm=None, _per=0, _nPer=0):
        """
        :param _arHarm: array of field harmonics
        :param _per: period length [m]
        :param _nPer: number of periods (will be rounded to integer)
        """
        self.arHarm = [] if _arHarm is None else _arHarm
        self.per = _per
        self.nPer = _nPer

    def allocate(self, _nHarm):
        self.arHarm = [SRWLMagFldH()]*_nHarm

class SRWLMagFldC(SRWLMagFld):
    """Magnetic Field: Container"""
    
    def __init__(self, _arMagFld=None, _arXc=None, _arYc=None, _arZc=None):
        """
        :param _arMagFld: magnetic field structures array
        :param _arXc: horizontal center positions of magnetic field elements in arMagFld array [m]
        :param _arYc: vertical center positions of magnetic field elements in arMagFld array [m]
        :param _arZc: longitudinal center positions of magnetic field elements in arMagFld array [m]
        """
        self.arMagFld = [] if _arMagFld is None else _arMagFld
        self.arXc = array('d') if _arXc is None else _arXc
        self.arYc = array('d') if _arYc is None else _arYc
        self.arZc = array('d') if _arZc is None else _arZc

    def allocate(self, _nElem):
        self.arMagFld = [SRWLMagFld()]*_nElem
        self.arXc = array('d', [0]*_nElem)
        self.arYc = array('d', [0]*_nElem)
        self.arZc = array('d', [0]*_nElem)

#****************************************************************************
class SRWLPrtTrj(object):
    """Charged Particle Trajectory"""

    def __init__(self, _arX=None, _arXp=None, _arY=None, _arYp=None, _arZ=None, _arZp=None, _np=0, _ctStart=0, _ctEnd=0, _partInitCond=None):
        """
        :param _arX: array of horizontal position [m]
        :param _arXp: array of horizontal relative velocity (trajectory angle) [rad]
        :param _arY: array of vertical position [m]
        :param _arYp: array of vertical relative velocity (trajectory angle) [rad]
        :param _arZ: array of longitudinal positions [m]
        :param _arZp: array of longitudinal relative velocity [rad]
        :param _np: number of trajectory points
        :param _ctStart: start value of independent variable (c*t) for which the trajectory should be (/is) calculated (is constant step enough?)
        :param _ctEnd: end value of independent variable (c*t) for which the trajectory should be (/is) calculated (is constant step enough?)
        :param _partInitCond: particle type and initial conditions for which the trajectory should be (/is) calculated
        """
        self.arX = array('d') if _arX is None else _arX
        self.arY = array('d') if _arY is None else _arY
        self.arY = array('d') if _arZ is None else _arZ
        self.arXp = array('d') if _arXp is None else _arXp
        self.arYp = array('d') if _arYp is None else _arYp
        self.arZp = array('d') if _arZp is None else _arZp
        self.np = _np
        self.ctStart = _ctStart
        self.ctEnd = _ctEnd
        self.partInitCond = SRWLParticle() if _partInitCond is None else _partInitCond

    def allocate(self, _np):
        _np = int(_np)
        self.arX = array('d', [0] * _np)
        self.arXp = array('d', [0] * _np)
        self.arY = array('d', [0] * _np)
        self.arYp = array('d', [0] * _np)
        self.arZ = array('d', [0] * _np)
        self.arZp = array('d', [0] * _np)
        self.np = _np

#****************************************************************************
class SRWLKickM(object):
    """Kick Matrix (for fast trajectory calculation)"""
    
    def __init__(self, _arKickMx=None, _arKickMy=None, _order=2, _nx=0, _ny=0, _nz=0, _rx=0, _ry=0, _rz=0, _x=0, _y=0, _z=0):
        """
        :param _arKickMx: horizontal kick-matrix (tabulated on the same transverse grid vs x and y as vertical kick-matrix)
        :param _arKickMy: vertical kick-matrix (tabulated on the same transverse grid vs x and y as horizontal kick-matrix)
        :param _order: kick order: 1- first order (in this case kick matrix data is assumed to be in [T*m]), 2- second order (kick matrix data is assumed to be in [T^2*m^2])
        :param _nx: numbers of points in kick matrices in horizontal direction
        :param _ny: numbers of points in kick matrices in vertical direction
        :param _nz: number of steps in longitudinal direction
        :param _rx: range covered by kick matrices in horizontal direction [m]
        :param _ry: range covered by kick matrices in vertical direction [m]
        :param _rz: extension in longitudinal direction [m]
        :param _x: horizontal coordinate of center point [m]
        :param _y: vertical coordinate of center point [m]
        :param _z: longitudinal coordinate of center point [m]
        """
        self.arKickMx = array('d') if _arKickMx is None else _arKickMx
        self.arKickMy = array('d') if _arKickMy is None else _arKickMy
        self.order = _order
        self.nx = _nx
        self.ny = _ny
        self.nz = _nz
        self.rx = _rx
        self.ry = _ry
        self.rz = _rz
        self.x = _x
        self.y = _y
        self.z = _z

#****************************************************************************
class SRWLGsnBm(object):
    """Gaussian Beam"""
    
    def __init__(self, _x=0, _y=0, _z=0, _xp=0, _yp=0, _avgPhotEn=1, _pulseEn=1, _repRate=1, _polar=1, _sigX=10e-06,
                 _sigY=10e-06, _sigT=1e-15, _mx=0, _my=0):
        """
        :param _x: average horizontal coordinates of waist [m]
        :param _y: average vertical coordinates of waist [m]
        :param _z: average longitudinal coordinate of waist [m]
        :param _xp: average horizontal angle at waist [rad]
        :param _yp: average verical angle at waist [rad]
        :param _avgPhotEn: average photon energy [eV]
        :param _pulseEn: energy per pulse [J]
        :param _repRate: rep. rate [Hz]
        :param _polar: polarization 1- lin. hor., 2- lin. vert., 3- lin. 45 deg., 4- lin.135 deg., 5- circ. right, 6- circ. left
        :param _sigX: rms beam size vs horizontal position [m] at waist (for intensity)
        :param _sigY: rms beam size vs vertical position [m] at waist (for intensity)
        :param _sigT: rms pulse duration [s] (for intensity)
        :param _mx: transverse Gauss-Hermite mode order in horizontal direction
        :param _my: transverse Gauss-Hermite mode order in vertical direction
        """
        self.x = _x
        self.y = _y
        self.z = _z
        self.xp = _xp
        self.yp = _yp
        self.avgPhotEn = _avgPhotEn
        self.pulseEn = _pulseEn
        self.repRate = _repRate
        self.polar = _polar
        self.sigX = _sigX
        self.sigY = _sigY
        self.sigT = _sigT
        self.mx = _mx
        self.my = _my

#****************************************************************************
class SRWLRadMesh(object):
    """Radiation Mesh (Sampling)"""
    
    def __init__(self, _eStart=0, _eFin=0, _ne=1, _xStart=0, _xFin=0, _nx=1, _yStart=0, _yFin=0, _ny=1, _zStart=0):
        """
        :param _eStart: initial value of photon energy (/time)
        :param _eFin: final value of photon energy (/time)
        :param _ne: number of points vs photon energy (/time)
        :param _xStart: initial value of horizontal position (/angle)
        :param _xFin: final value of horizontal position (/angle)
        :param _nx: number of points vs horizontal position (/angle)
        :param _yStart: initial value of vertical position (/angle)
        :param _yFin: final value of vertical position (/angle)
        :param _ny: number of points vs vertical position (/angle)
        :param _zStart: longitudinal position
        """
        self.eStart = _eStart
        self.eFin = _eFin
        self.ne = _ne
        self.xStart = _xStart
        self.xFin = _xFin
        self.nx = _nx
        self.yStart = _yStart
        self.yFin = _yFin
        self.ny = _ny
        self.zStart = _zStart

    def set_from_other(self, _mesh):
        self.eStart = _mesh.eStart; self.eFin = _mesh.eFin; self.ne = _mesh.ne; 
        self.xStart = _mesh.xStart; self.xFin = _mesh.xFin; self.nx = _mesh.nx; 
        self.yStart = _mesh.yStart; self.yFin = _mesh.yFin; self.ny = _mesh.ny;
        self.zStart = _mesh.zStart

#****************************************************************************
class SRWLStokes(object):
    """Radiation Stokes Parameters"""
    
    #def __init__(self, _arS0=None, _arS1=None, _arS2=None, _arS3=None, _typeStokes='f', _eStart=0, _eFin=0, _ne=0, _xStart=0, _xFin=0, _nx=0, _yStart=0, _yFin=0, _ny=0):
    def __init__(self, _arS=None, _typeStokes='f', _eStart=0, _eFin=0, _ne=0, _xStart=0, _xFin=0, _nx=0, _yStart=0, _yFin=0, _ny=0):
        """
        :param _arS: flat C-aligned array of all Stokes components (outmost loop over Stokes parameter number); NOTE: only 'f' (float) is supported for the moment (Jan. 2012)
        :param _typeStokes: electric field numerical type: 'f' (float) or 'd' (double, not supported yet)
        :param _eStart: initial value of photon energy (/time)
        :param _eFin: final value of photon energy (/time)
        :param _ne: numbers of points vs photon energy
        :param _xStart: initial value of horizontal position
        :param _xFin: final value of photon horizontal position
        :param _nx: numbers of points vs horizontal position
        :param _yStart: initial value of vertical position
        :param _yFin: final value of vertical position
        :param _ny: numbers of points vs vertical position
        """
        self.arS = _arS #flat C-aligned array of all Stokes components (outmost loop over Stokes parameter number); NOTE: only 'f' (float) is supported for the moment (Jan. 2012)
        self.numTypeStokes = _typeStokes #electric field numerical type: 'f' (float) or 'd' (double)
        self.mesh = SRWLRadMesh(_eStart, _eFin, _ne, _xStart, _xFin, _nx, _yStart, _yFin, _ny) #to make mesh an instance variable
        self.avgPhotEn = 0 #average photon energy for time-domain simulations    
        self.presCA = 0 #presentation/domain: 0- coordinates, 1- angles
        self.presFT = 0 #presentation/domain: 0- frequency (photon energy), 1- time
        self.unitStokes = 1 #electric field units: 0- arbitrary, 1- Phot/s/0.1%bw/mm^2 ?

        nProd = _ne*_nx*_ny #array length to store one component of complex electric field
        if((_arS == 1) and (nProd > 0)):
            self.allocate(_ne, _nx, _ny, _typeStokes)          
        #s0needed = 0
        #s1needed = 0
        #s2needed = 0
        #s3needed = 0
        #if((_arS0 == 1) and (nProd > 0)):
        #    s0needed = 1
        #if((_arS1 == 1) and (nProd > 0)):
        #    s1needed = 1
        #if((_arS2 == 1) and (nProd > 0)):
        #    s2needed = 1
        #if((_arS3 == 1) and (nProd > 0)):
        #    s3needed = 1
        #if((s0needed > 0) or (s1needed > 0) or (s2needed > 0) or (s3needed > 0)):
        #    self.allocate(_ne, _nx, _ny, s0needed, s1needed, s2needed, s3needed)


    #def allocate(self, _ne, _nx, _ny, s0needed=1, s1needed=1, s2needed=1, s3needed=1, _typeStokes='f'):
    def allocate(self, _ne, _nx, _ny, _typeStokes='f'):
        #print('') #debugging
        #print('          (re-)allocating: old point numbers: ne=',self.mesh.ne,' nx=',self.mesh.nx,' ny=',self.mesh.ny,' type:',self.numTypeStokes)
        #print('                           new point numbers: ne=',_ne,' nx=',_nx,' ny=',_ny,' type:',_typeStokes)
        #nTot = _ne*_nx*_ny #array length to one Stokes component
        #if s0needed:
        #    del self.arS0
        #    self.arS0 = array(_typeStokes, [0]*nTot)
        #if s1needed:
        #    del self.arS1
        #    self.arS1 = array(_typeStokes, [0]*nTot)
        #if s2needed:
        #    del self.arS2
        #    self.arS2 = array(_typeStokes, [0]*nTot)
        #if s3needed:
        #    del self.arS3
        #    self.arS3 = array(_typeStokes, [0]*nTot)
        nTot = _ne*_nx*_ny*4 #array length of all Stokes components
        self.arS = array(_typeStokes, [0]*nTot)
        self.numTypeStokes = _typeStokes
        self.mesh.ne = _ne
        self.mesh.nx = _nx
        self.mesh.ny = _ny

    def avg_update_same_mesh(self, _more_stokes, _iter, _n_stokes_comp=4):
        """ Update this Stokes data structure with new data, contained in the _more_stokes structure, calculated on the same mesh, so that this structure would represent estimation of average of (_iter + 1) structures
        :param _more_stokes: Stokes data structure to "add" to the estimation of average
        :param _iter: number of Stokes structures already "added" previously
        :param _n_stokes_comp: number of Stokes components to treat (1 to 4)
        """
        nStPt = self.mesh.ne*self.mesh.nx*self.mesh.ny*_n_stokes_comp
        for ir in range(nStPt):
            self.arS[ir] = (self.arS[ir]*_iter + _more_stokes.arS[ir])/(_iter + 1)

    def avg_update_interp(self, _more_stokes, _iter, _ord, _n_stokes_comp=4):
        """ Update this Stokes data structure with new data, contained in the _more_stokes structure, calculated on a different 2D mesh, so that it would represent estimation of average of (_iter + 1) structures
        :param _more_stokes: Stokes data structure to "add" to the estimation of average
        :param _iter: number of Stokes structures already "added" previously
        :param _ord: order of 2D interpolation to use (1- bilinear, ..., 3- bi-cubic)
        :param _n_stokes_comp: number of Stokes components to treat (1 to 4)
        """
        eNpMeshRes = self.mesh.ne
        xNpMeshRes = self.mesh.nx
        xStartMeshRes = self.mesh.xStart
        xStepMeshRes = 0
        if(xNpMeshRes > 1):
            xStepMeshRes = (self.mesh.xFin - xStartMeshRes)/(xNpMeshRes - 1)
        yNpMeshRes = self.mesh.ny
        yStartMeshRes = self.mesh.yStart
        yStepMeshRes = 0
        if(yNpMeshRes > 1):
            yStepMeshRes = (self.mesh.yFin - yStartMeshRes)/(yNpMeshRes - 1)

        eNpWfr = _more_stokes.mesh.ne
        xStartWfr = _more_stokes.mesh.xStart
        xNpWfr = _more_stokes.mesh.nx
        xStepWfr = 0
        if(xNpWfr  > 1):
            xStepWfr = (_more_stokes.mesh.xFin - xStartWfr)/(xNpWfr - 1)
        yStartWfr = _more_stokes.mesh.yStart
        yNpWfr = _more_stokes.mesh.ny
        yStepWfr = 0
        if(yNpWfr  > 1):
            yStepWfr = (_more_stokes.mesh.yFin - yStartWfr)/(yNpWfr - 1)
        #DEBUG
        #print('avg_update_interp: iter=', _iter)
        #END DEBUG

        nRadWfr = eNpWfr*xNpWfr*yNpWfr
        iOfstSt = 0
        ir = 0
        for iSt in range(_n_stokes_comp):
            for iy in range(yNpMeshRes):
                yMeshRes = yStartMeshRes + iy*yStepMeshRes
                for ix in range(xNpMeshRes):
                    xMeshRes = xStartMeshRes + ix*xStepMeshRes
                    for ie in range(eNpMeshRes):
                        #calculate Stokes parameters of propagated wavefront on the resulting mesh
                        #fInterp = srwl_uti_interp_2d(xMeshRes, yMeshRes, xStartWfr, xStepWfr, xNpWfr, yStartWfr, yStepWfr, yNpWfr, workArStokes, 1, eNpWfr, iOfstStokes)
                        fInterp = 0
                        loc_ix_ofst = iOfstSt + ie
                        nx_ix_per = xNpWfr*eNpWfr
                        if(_ord == 1): #bi-linear interpolation based on 4 points
                            ix0 = int(trunc((xMeshRes - xStartWfr)/xStepWfr + 1.e-09))
                            if((ix0 < 0) or (ix0 >= xNpWfr - 1)):
                                self.arS[ir] = self.arS[ir]*_iter/(_iter + 1); ir += 1
                                continue
                            ix1 = ix0 + 1
                            tx = (xMeshRes - (xStartWfr + xStepWfr*ix0))/xStepWfr
                            iy0 = int(trunc((yMeshRes - yStartWfr)/yStepWfr + 1.e-09))
                            if((iy0 < 0) or (iy0 >= yNpWfr - 1)):
                                self.arS[ir] = self.arS[ir]*_iter/(_iter + 1); ir += 1
                                continue
                            iy1 = iy0 + 1
                            ty = (yMeshRes - (yStartWfr + yStepWfr*iy0))/yStepWfr
                            iy0_nx_ix_per = iy0*nx_ix_per
                            iy1_nx_ix_per = iy1*nx_ix_per
                            ix0_ix_per_p_ix_ofst = ix0*eNpWfr + loc_ix_ofst
                            ix1_ix_per_p_ix_ofst = ix1*eNpWfr + loc_ix_ofst
                            a00 = _more_stokes.arS[iy0_nx_ix_per + ix0_ix_per_p_ix_ofst]
                            f10 = _more_stokes.arS[iy0_nx_ix_per + ix1_ix_per_p_ix_ofst]
                            f01 = _more_stokes.arS[iy1_nx_ix_per + ix0_ix_per_p_ix_ofst]
                            f11 = _more_stokes.arS[iy1_nx_ix_per + ix1_ix_per_p_ix_ofst]
                            a10 = f10 - a00
                            a01 = f01 - a00
                            a11 = a00 - f01 - f10 + f11
                            fInterp = a00 + tx*(a10 + ty*a11) + ty*a01

                        elif(_ord == 2): #bi-quadratic interpolation based on 6 points
                            ix0 = int(round((xMeshRes - xStartWfr)/xStepWfr))
                            if((ix0 < 0) or (ix0 >= xNpWfr - 1)):
                                self.arS[ir] = self.arS[ir]*_iter/(_iter + 1); ir += 1
                                continue
                            ixm1 = ix0 - 1
                            ix1 = ix0 + 1
                            tx = (xMeshRes - (xStartWfr + xStepWfr*ix0))/xStepWfr
                            iy0 = int(round((yMeshRes - yStartWfr)/yStepWfr))
                            if((iy0 < 0) or (iy0 >= yNpWfr - 1)):
                                self.arS[ir] = self.arS[ir]*_iter/(_iter + 1); ir += 1
                                continue
                            iym1 = iy0 - 1
                            iy1 = iy0 + 1
                            ty = (yMeshRes - (yStartWfr + yStepWfr*iy0))/yStepWfr
                            iym1_nx_ix_per = iym1*nx_ix_per
                            iy0_nx_ix_per = iy0*nx_ix_per
                            iy1_nx_ix_per = iy1*nx_ix_per
                            ixm1_ix_per_p_ix_ofst = ixm1*eNpWfr + loc_ix_ofst
                            ix0_ix_per_p_ix_ofst = ix0*eNpWfr + loc_ix_ofst
                            ix1_ix_per_p_ix_ofst = ix1*eNpWfr + loc_ix_ofst
                            fm10 = _more_stokes.arS[iy0_nx_ix_per + ixm1_ix_per_p_ix_ofst]
                            a00 = _more_stokes.arS[iy0_nx_ix_per + ix0_ix_per_p_ix_ofst]
                            f10 = _more_stokes.arS[iy0_nx_ix_per + ix1_ix_per_p_ix_ofst]
                            f0m1 = _more_stokes.arS[iym1_nx_ix_per + ix0_ix_per_p_ix_ofst]
                            f01 = _more_stokes.arS[iy1_nx_ix_per + ix0_ix_per_p_ix_ofst]
                            f11 = _more_stokes.arS[iy1_nx_ix_per + ix1_ix_per_p_ix_ofst]
                            a10 = 0.5*(f10 - fm10)
                            a01 = 0.5*(f01 - f0m1)
                            a11 = a00 - f01 - f10 + f11
                            a20 = 0.5*(f10 + fm10) - a00
                            a02 = 0.5*(f01 + f0m1) - a00
                            fInterp = a00 + tx*(a10 + tx*a20 + ty*a11) + ty*(a01 + ty*a02)
    
                        elif(_ord == 3): #bi-cubic interpolation based on 12 points
                            ix0 = int(trunc((xMeshRes - xStartWfr)/xStepWfr + 1.e-09))
                            if((ix0 < 0) or (ix0 >= xNpWfr - 1)):
                                self.arS[ir] = self.arS[ir]*_iter/(_iter + 1); ir += 1
                                continue
                            elif(ix0 < 1):
                                ix0 = 1
                            elif(ix0 >= xNpWfr - 2):
                                ix0 = xNpWfr - 3
                            ixm1 = ix0 - 1
                            ix1 = ix0 + 1
                            ix2 = ix0 + 2
                            tx = (xMeshRes - (xStartWfr + xStepWfr*ix0))/xStepWfr
                            iy0 = int(trunc((yMeshRes - yStartWfr)/yStepWfr + 1.e-09))
                            if((iy0 < 0) or (iy0 >= yNpWfr - 1)):
                                self.arS[ir] = self.arS[ir]*_iter/(_iter + 1); ir += 1
                                continue
                            elif(iy0 < 1):
                                iy0 = 1
                            elif(iy0 >= yNpWfr - 2):
                                iy0 = yNpWfr - 3
                            iym1 = iy0 - 1
                            iy1 = iy0 + 1
                            iy2 = iy0 + 2
                            ty = (yMeshRes - (yStartWfr + yStepWfr*iy0))/yStepWfr
                            iym1_nx_ix_per = iym1*nx_ix_per
                            iy0_nx_ix_per = iy0*nx_ix_per
                            iy1_nx_ix_per = iy1*nx_ix_per
                            iy2_nx_ix_per = iy2*nx_ix_per
                            ixm1_ix_per_p_ix_ofst = ixm1*eNpWfr + loc_ix_ofst
                            ix0_ix_per_p_ix_ofst = ix0*eNpWfr + loc_ix_ofst
                            ix1_ix_per_p_ix_ofst = ix1*eNpWfr + loc_ix_ofst
                            ix2_ix_per_p_ix_ofst = ix2*eNpWfr + loc_ix_ofst
                            f0m1 = _more_stokes.arS[iym1_nx_ix_per + ix0_ix_per_p_ix_ofst]
                            f1m1 = _more_stokes.arS[iym1_nx_ix_per + ix1_ix_per_p_ix_ofst]
                            fm10 = _more_stokes.arS[iy0_nx_ix_per + ixm1_ix_per_p_ix_ofst]
                            a00 = _more_stokes.arS[iy0_nx_ix_per + ix0_ix_per_p_ix_ofst]
                            f10 = _more_stokes.arS[iy0_nx_ix_per + ix1_ix_per_p_ix_ofst]
                            f20 = _more_stokes.arS[iy0_nx_ix_per + ix2_ix_per_p_ix_ofst]
                            fm11 = _more_stokes.arS[iy1_nx_ix_per + ixm1_ix_per_p_ix_ofst]
                            f01 = _more_stokes.arS[iy1_nx_ix_per + ix0_ix_per_p_ix_ofst]
                            f11 = _more_stokes.arS[iy1_nx_ix_per + ix1_ix_per_p_ix_ofst]
                            f21 = _more_stokes.arS[iy1_nx_ix_per + ix2_ix_per_p_ix_ofst]
                            f02 = _more_stokes.arS[iy2_nx_ix_per + ix0_ix_per_p_ix_ofst]
                            f12 = _more_stokes.arS[iy2_nx_ix_per + ix1_ix_per_p_ix_ofst]
                            a10 = -0.5*a00 + f10 - f20/6 - fm10/3
                            a01 = -0.5*a00 + f01 - f02/6 - f0m1/3
                            a11 = -0.5*(f01 + f10) + (f02 - f12 + f20 - f21)/6 + (f0m1 - f1m1 + fm10 - fm11)/3 + f11
                            a20 = -a00 + 0.5*(f10 + fm10)
                            a02 = -a00 + 0.5*(f01 + f0m1)
                            a21 = a00 - f01 + 0.5*(f11 - f10 - fm10 + fm11)
                            a12 = a00 - f10 + 0.5*(f11 - f01 - f0m1 + f1m1)
                            a30 = 0.5*(a00 - f10) + (f20 - fm10)/6
                            a03 = 0.5*(a00 - f01) + (f02 - f0m1)/6
                            a31 = 0.5*(f01 + f10 - f11 - a00) + (f21 + fm10 - f20 - fm11)/6
                            a13 = 0.5*(f10 - f11 - a00 + f01) + (f0m1 + f12 - f02 - f1m1)/6
                            fInterp = a00 + tx*(a10 + tx*(a20 + tx*(a30 + ty*a31) + ty*a21) + ty*a11) + ty*(a01 + ty*(a02 + ty*(a03 + tx*a13) + tx*a12))

                        self.arS[ir] = (self.arS[ir]*_iter + fInterp)/(_iter + 1)
                        ir += 1
            iOfstSt += nRadWfr        

#****************************************************************************
class SRWLWfr(object):
    """Radiation Wavefront (Electric Field)"""
    #arEx = 0 #array('f', [0]*2) #horizontal complex electric field component array; NOTE: only 'f' (float) is supported for the moment (Jan. 2011)
    #arEy = 0 #array('f', [0]*2) #vertical complex electric field component array
    #mesh = SRWLRadMesh()
    ##eStart = 0 #initial and final values of photon energy (/time), horizontal and vertical positions
    ##eFin = 0
    ##xStart = 0
    ##xFin = 0
    ##yStart = 0
    ##yFin = 0
    ##zStart = 0
    ##ne = 0 #numbers of points vs photon energy, horizontal and vertical positions
    ##nx = 0
    ##ny = 0
    #Rx = 0 #instant wavefront radii
    #Ry = 0 
    #dRx = 0 #error of wavefront radii
    #dRy = 0
    #xc = 0 #instant transverse coordinates of wavefront instant "source center"
    #yc = 0
    #avgPhotEn = 0 #average photon energy for time-domain simulations    
    #presCA = 0 #presentation/domain: 0- coordinates, 1- angles
    #presFT = 0 #presentation/domain: 0- frequency (photon energy), 1- time
    #numTypeElFld = 'f' #electric field numerical type: 'f' (float) or 'd' (double)
    #unitElFld = 1 #electric field units: 0- arbitrary, 1- sqrt(Phot/s/0.1%bw/mm^2) ?
    #partBeam = SRWLPartBeam() #particle beam source; strictly speaking, it should be just SRWLParticle; however, "multi-electron" information can appear useful for those cases when "multi-electron intensity" can be deduced from the "single-electron" one by convolution
    #arElecPropMatr = array('d', [0]*20) #effective 1st order "propagation matrix" for electron beam parameters
    #arMomX = array('d', [0]*11) #statistical moments (of Wigner distribution); to check the exact number of moments required
    #arMomY = array('d', [0]*11)
    #arWfrAuxData = array('d', [0]*30) #array of auxiliary wavefront data

    def __init__(self, _arEx=None, _arEy=None, _typeE='f', _eStart=0, _eFin=0, _ne=0, _xStart=0, _xFin=0, _nx=0, _yStart=0, _yFin=0, _ny=0, _zStart=0, _partBeam=None):
        """
        :param _arEx: horizontal complex electric field component array; NOTE: only 'f' (float) is supported for the moment (Jan. 2011)
        :param _arEy: vertical complex electric field component array
        :param _typeE: electric field numerical type: 'f' (float) or 'd' (double)
        :param _eStart: initial value of photon energy (/time)
        :param _eFin: final value of photon energy (/time)
        :param _ne: numbers of points vs photon energy
        :param _xStart: initial value of horizontal positions
        :param _xFin: final value of horizontal positions
        :param _nx: numbers of points vs horizontal positions
        :param _yStart: initial vertical positions
        :param _yFin: final value of vertical positions
        :param _ny: numbers of points vs vertical positions
        :param _zStart: longitudinal position
        :param _partBeam: particle beam source; strictly speaking, it should be just SRWLParticle; however, "multi-electron" information can appear useful for those cases when "multi-electron intensity" can be deduced from the "single-electron" one by convolution

        Some additional parameters, that are not included in constructor arguments:
        Rx, Ry: instant wavefront radii
        dRx, dRy: error of wavefront radii
        xc, yc: transverse coordinates of wavefront instant "source center"
        avgPhotEn: average photon energy for time-domain simulations
        presCA: presentation/domain: 0- coordinates, 1- angles
        presFT: presentation/domain: 0- frequency (photon energy), 1- time
        unitElFld: electric field units: 0- arbitrary, 1- sqrt(Phot/s/0.1%bw/mm^2)
        arElecPropMatr: effective 1st order "propagation matrix" for electron beam parameters
        arMomX, arMomY: statistical moments (of Wigner distribution); to check the exact number of moments required
        arWfrAuxData: array of auxiliary wavefront data
        """
        self.arEx = _arEx
        self.arEy = _arEy
        #self.mesh = SRWLRadMesh(_eStart, _eFin, _ne, _xStart, _xFin, _nx, _yStart, _yFin, _ny)
        self.mesh = SRWLRadMesh(_eStart, _eFin, _ne, _xStart, _xFin, _nx, _yStart, _yFin, _ny, _zStart)
        self.numTypeElFld = _typeE
        self.partBeam = SRWLPartBeam() if _partBeam is None else _partBeam

        self.Rx = 0 #instant wavefront radii
        self.Ry = 0
        self.dRx = 0 #error of wavefront radii
        self.dRy = 0
        self.xc = 0 #instant transverse coordinates of wavefront instant "source center"
        self.yc = 0
        self.avgPhotEn = 0 #average photon energy for time-domain simulations
        self.presCA = 0 #presentation/domain: 0- coordinates, 1- angles
        self.presFT = 0 #presentation/domain: 0- frequency (photon energy), 1- time
        self.unitElFld = 1 #electric field units: 0- arbitrary, 1- sqrt(Phot/s/0.1%bw/mm^2) ?
        self.arElecPropMatr = array('d', [0] * 20) #effective 1st order "propagation matrix" for electron beam parameters
        self.arMomX = array('d', [0] * 11 * _ne) #statistical moments (of Wigner distribution); to check the exact number of moments required
        self.arMomY = array('d', [0] * 11 * _ne)
        self.arWfrAuxData = array('d', [0] * 30) #array of auxiliary wavefront data

        nProd = _ne * _nx * _ny #array length to store one component of complex electric field
        EXNeeded = 0
        EYNeeded = 0
        if(_arEx == 1) and (nProd > 0):
            EXNeeded = 1
        if(_arEy == 1) and (nProd > 0):
            EYNeeded = 1
        if(EXNeeded > 0) or (EYNeeded > 0):
            self.allocate(_ne, _nx, _ny, EXNeeded, EYNeeded)

    def allocate(self, _ne, _nx, _ny, EXNeeded=1, EYNeeded=1, typeE='f'):
        #print('') #debugging
        #print('          (re-)allocating: old point numbers: ne=',self.mesh.ne,' nx=',self.mesh.nx,' ny=',self.mesh.ny) #,' type:',self.numTypeElFld)
        #print('                           new point numbers: ne=',_ne,' nx=',_nx,' ny=',_ny) #,' type:',typeE)
        nTot = 2*_ne*_nx*_ny #array length to store one component of complex electric field
        nMom = 11*_ne
        if EXNeeded:
            #print('          trying to (re-)allocate Ex ... ', end='')  
            del self.arEx
            self.arEx = array(typeE, [0]*nTot)
            #print('done')           
            if len(self.arMomX) != nMom:
                del self.arMomX
                self.arMomX = array('d', [0]*nMom)
        if EYNeeded:
            #print('          trying to (re-)allocate Ey ... ', end='')  
            #del self.arEy
            self.arEy = array(typeE, [0]*nTot)
            #print('done')
            if len(self.arMomY) != nMom:
                del self.arMomY
                self.arMomY = array('d', [0]*nMom)
        self.numTypeElFld = typeE
        self.mesh.ne = _ne
        self.mesh.nx = _nx
        self.mesh.ny = _ny

    def calc_stokes(self, _stokes):
        """Calculate Stokes parameters from Electric Field"""
        nTot = self.mesh.ne*self.mesh.nx*self.mesh.ny
        #if(type(_stokes).__name__ != 'SRWLStokes')):
        if(isinstance(_stokes, SRWLStokes) == False):
            raise Exception("Incorrect Stokes parameters object submitted") 
        nTotSt = nTot*4
        nTot2 = nTot*2
        nTot3 = nTot*3
        if(_stokes.arS != None):
            if(len(_stokes.arS) < nTotSt):
                _stokes.arS = array('f', [0]*nTotSt)
        else:
            _stokes.arS = array('f', [0]*nTotSt)           
        for i in range(nTot):
            i2 = i*2
            i2p1 = i2 + 1
            reEx = self.arEx[i2]
            imEx = self.arEx[i2p1]
            reEy = self.arEy[i2]
            imEy = self.arEy[i2p1]
            intLinX = reEx*reEx + imEx*imEx
            intLinY = reEy*reEy + imEy*imEy
            _stokes.arS[i] = intLinX + intLinY
            _stokes.arS[i + nTot] = intLinX - intLinY #check sign
            _stokes.arS[i + nTot2] = -2*(reEx*reEy + imEx*imEy) #check sign
            _stokes.arS[i + nTot3] = 2*(-reEx*reEy + imEx*imEy) #check sign
        _stokes.mesh.set_from_other(self.mesh)
        
#****************************************************************************
class SRWLOpt(object):
    """Optical Element (base class)"""

class SRWLOptD(SRWLOpt):
    """Optical Element: Drift Space"""
    
    def __init__(self, _L=0):
        """
        :param _L: Length [m]
        """
        self.L = _L

class SRWLOptA(SRWLOpt):
    """Optical Element: Aperture / Obstacle"""
    
    def __init__(self, _shape='r', _ap_or_ob='a', _Dx=0, _Dy=0, _x=0, _y=0):
        """
        :param _shape: 'r' for rectangular, 'c' for circular
        :param _ap_or_ob: 'a' for aperture, 'o' for obstacle
        :param _Dx: horizontal transverse dimension [m]; in case of circular aperture, only Dx is used for diameter
        :param _Dy: vertical transverse dimension [m]; in case of circular aperture, Dy is ignored
        :param _x: horizontal transverse coordinate of center [m]
        :param _y: vertical transverse coordinate of center [m]
        """
        self.shape = _shape #'r' for rectangular, 'c' for circular
        self.ap_or_ob = _ap_or_ob #'a' for aperture, 'o' for obstacle
        self.Dx = _Dx #transverse dimensions [m]; in case of circular aperture, only Dx is used for diameter
        self.Dy = _Dy
        self.x = _x #transverse coordinates of center [m]
        self.y = _y

class SRWLOptL(SRWLOpt):
    """Optical Element: Thin Lens"""
    
    def __init__(self, _Fx=1e+23, _Fy=1e+23, _x=0, _y=0):
        """
        :param _Fx: focal length in horizontal plane [m]
        :param _Fy: focal length in vertical plane [m]
        :param _x: horizontal coordinate of center [m]
        :param _y: vertical coordinate of center [m]
        """
        self.Fx = _Fx #focal lengths [m]
        self.Fy = _Fy
        self.x = _x #transverse coordinates of center [m]
        self.y = _y

class SRWLOptZP(SRWLOpt):
    """Optical Element: Thin Lens"""
    
    def __init__(self, _nZones=100, _rn=0.1e-03, _thick=10e-06, _delta1=1e-06, _atLen1=0.1, _delta2=0, _atLen2=1e-06, _x=0, _y=0):
        """
        :param _nZones: total number of zones
        :param _rn: auter zone radius [m]
        :param _thick: thickness [m]
        :param _delta1: refractuve index decrement of the "main" material
        :param _atLen1: attenuation length [m] of the "main" material
        :param _delta2: refractuve index decrement of the "complementary" material
        :param _atLen2: attenuation length [m] of the "complementary" material
        :param _x: horizontal transverse coordinate of center [m]
        :param _y: vertical transverse coordinates of center [m]
        """
        self.nZones = _nZones #total number of zones
        self.rn = _rn #auter zone radius [m]
        self.thick = _thick #thickness [m]
        self.delta1 = _delta1 #refractuve index decrement of the "main" material
        self.delta2 = _delta2 #refractuve index decrement of the "complementary" material
        self.atLen1 = _atLen1 #attenuation length [m] of the "main" material
        self.atLen2 = _atLen2 #attenuation length [m] of the "complementary" material
        self.x = _x #transverse coordinates of center [m]
        self.y = _y

class SRWLOptWG(SRWLOpt):
    """Optical Element: Waveguide"""
    
    def __init__(self, _L=1, _Dx=10e-03, _Dy=10e-03, _x=0, _y=0):
        """
        :param _L: length [m]
        :param _Dx: horizontal transverse dimension [m]
        :param _Dy: vertical transverse dimension [m]
        :param _x: horizontal transverse coordinate of center [m]
        :param _y: vertical transverse coordinate of center [m]
        """
        self.L = _L #length [m]
        self.Dx = _Dx #transverse dimensions [m]
        self.Dy = _Dy
        self.x = _x #transverse coordinates of center [m]
        self.y = _y

class SRWLOptG(SRWLOpt):
    """Optical Element: Grating (planar)"""
    
    def __init__(self, _grDen=100, _disPl='v', _ang=0.7854, _m=1, _refl=1):
        """
        :param _grDen: groove density [lines/mm]
        :param _disPl: dispersion plane: 'x' ('h') or 'y' ('v')
        :param _ang: angle between optical axis and grating plane [rad]
        :param _m: output order
        :param _refl: average reflectivity (with respect to intensity)
        """
        self.grDen = _grDen #groove density [lines/mm]
        self.disPl = _disPl #dispersion plane: 'x' ('h') or 'y' ('v')
        self.ang = _ang #angle between optical axis and grating plane [rad]
        self.m = _m #output order
        self.refl = _refl #average reflectivity (with resp. to intensity)

class SRWLOptT(SRWLOpt):
    """Optical Element: Transmission (generic)"""
    
    def __init__(self, _nx=1, _ny=1, _rx=1e-03, _ry=1e-03, _arTr=None, _extTr=0, _Fx=1e+23, _Fy=1e+23, _x=0, _y=0, _ne=1, _eStart=0, _eFin=0):
        """
        :param _nx: number of transmission data points in the horizontaldirection
        :param _ny: number of transmission data points in the vertical direction
        :param _rx: range of the horizontal coordinate [m] for which the transmission is defined
        :param _ry: range of the vertical coordinate [m] for which the transmission is defined
        :param _arTr: complex C-aligned data array (of 2*ne*nx*ny length) storing amplitude transmission and optical path difference as function of transverse coordinates
        :param _extTr: transmission outside the grid/mesh is zero (0), or it is same as on boundary (1)
        :param _Fx: estimated focal length in the horizontal plane [m]
        :param _Fy: estimated focal length in the vertical plane [m]
        :param _x: horizontal transverse coordinate of center [m]
        :param _y: vertical transverse coordinate of center [m]
        :param _ne: number of transmission data points vs photon energy
        :param _eStart: initial value of photon energy
        :param _eFin: final value of photon energy
        """
        self.arTr = _arTr #complex C-aligned data array (of 2*ne*nx*ny length) storing amplitude transmission and optical path difference as function of transverse position
        if((_arTr == None) or ((len(_arTr) != _ne*_nx*_ny*2) and (_ne*_nx*_ny > 0))):
            self.allocate(_ne, _nx, _ny)
        self.ne = _ne #number of transmission data points vs photon energy
        self.nx = _nx #numbers of transmission data points in the horizontal and vertical directions
        self.ny = _ny
        self.eStart = _eStart #initial and final values of photon energy
        self.eFin = _eFin
        self.rx = _rx #ranges of horizontal and vertical coordinates [m] for which the transmission is defined
        self.ry = _ry
        self.extTr = _extTr #0- transmission outside the grid/mesh is zero; 1- it is same as on boundary
        self.Fx = _Fx #estimated focal lengths [m]
        self.Fy = _Fy
        self.x = _x #transverse coordinates of center [m]
        self.y = _y
        #if _ne > 1: _Fx, _Fy should be arrays vs photon energy?

    def allocate(self, _ne, _nx, _ny):
        self.ne = _ne
        self.nx = _nx
        self.ny = _ny
        nTot = 2*_ne*_nx*_ny #total array length to store amplitude transmission and optical path difference
        self.arTr = array('d', [0]*nTot)        

class SRWLOptMir(SRWLOpt):
    """Optical Element: Mirror (focusing)"""

    def set_dim_sim_meth(self, _size_tang=1, _size_sag=1, _ap_shape='r', _sim_meth=2, _npt=500, _nps=500, _treat_in_out=1, _ext_in=0, _ext_out=0):
        """Sets Mirror Dimensions, Aperture Shape and its simulation method
        :param _size_tang: size in tangential direction [m]
        :param _size_sag: size in sagital direction [m]
        :param _ap_shape: shape of aperture in local frame ('r' for rectangular, 'e' for elliptical)
        :param _sim_meth: simulation method (1 for "thin" approximation, 2 for "thick" approximation)
        :param _npt: number of mesh points to represent mirror in tangential direction (used for "thin" approximation)
        :param _nps: number of mesh points to represent mirror in sagital direction (used for "thin" approximation)
        :param _treat_in_out: switch specifying how to treat input and output wavefront before and after the main propagation through the optical element:
                0- assume that the input wavefront is defined in the plane before the optical element, and the output wavefront is required in a plane just after the element;
                1- assume that the input wavefront is defined in the plane at the optical element center and the output wavefront is also required at the element center;
                2- assume that the input wavefront is defined in the plane at the optical element center and the output wavefront is also required at the element center; however, before the propagation though the optical element, the wavefront should be propagated through a drift back to a plane just before the optical element, then a special propagator will bring the wavefront to a plane at the optical element exit, and after this the wavefront will be propagated through a drift back to the element center;
        :param _ext_in: optical element extent on the input side, i.e. distance between the input plane and the optical center (positive, in [m]) to be used at wavefront propagation manipulations; if 0, this extent will be calculated internally from optical element parameters
        :param _ext_out: optical element extent on the output side, i.e. distance between the optical center and the output plane (positive, in [m]) to be used at wavefront propagation manipulations; if 0, this extent will be calculated internally from optical element parameters        
        """
        if((_sim_meth < 1) or (_sim_meth > 2)):
            raise Exception("Simulation method is not specified correctly (should be 1 for \"thin\", 2 for \"thick\" element approximation)")
        self.dt = _size_tang
        self.ds = _size_sag
        self.apShape = _ap_shape
        self.meth = _sim_meth
        self.npt = _npt
        self.nps = _nps
        self.treatInOut = _treat_in_out
        self.extIn = _ext_in
        self.extOut = _ext_out
        self.Fx = 0 #i.e. focal lengthes are not set
        self.Fy = 0

    def set_reflect(self, _refl=1, _n_ph_en=1, _n_ang=1, _n_comp=1, _ph_en_start=0, _ph_en_fin=0, _ph_en_scale_type='lin', _ang_start=0, _ang_fin=0, _ang_scale_type='lin'):
        """Sets Mirror Reflectivity
        :param _refl: reflectivity coefficient to set (can be one number or C-aligned flat array complex array vs photon energy vs grazing angle vs component (sigma, pi))
        :param _n_ph_en: number of photon energy values for which the reflectivity coefficient is specified
        :param _n_ang: number of grazing angle values for which the reflectivity coefficient is specified
        :param _n_comp: number of electric field components for which the reflectivity coefficient is specified (can be 1 or 2)
        :param _ph_en_start: initial photon energy value for which the reflectivity coefficient is specified
        :param _ph_en_fin: final photon energy value for which the reflectivity coefficient is specified
        :param _ph_en_scale_type: photon energy sampling type ('lin' for linear, 'log' for logarithmic)
        :param _ang_start: initial grazing angle value for which the reflectivity coefficient is specified
        :param _ang_fin: final grazing angle value for which the reflectivity coefficient is specified
        :param _ang_scale_type: angle sampling type ('lin' for linear, 'log' for logarithmic)
        """
        nTot = int(_n_ph_en*_n_ang*_n_comp*2)
        if(nTot < 2):
            raise Exception("Incorrect Reflectivity array parameters")
        _n_comp = int(_n_comp)
        if((_n_comp < 1) or (_n_comp > 2)):
            raise Exception("Number of reflectivity coefficient components can be 1 or 2")
    
        if(not(isinstance(_refl, list) or isinstance(_refl, array))):
            self.arRefl = array('d', [_refl]*nTot)
            for i in range(int(round(nTot/2))):
                i2 = i*2
                self.arRefl[i2] = _refl
                self.arRefl[i2 + 1] = 0
        else:
            self.arRefl = _refl
        
        self.reflNumPhEn = int(_n_ph_en)
        self.reflNumAng = int(_n_ang)
        self.reflNumComp = _n_comp
        self.reflPhEnStart = _ph_en_start
        self.reflPhEnFin = _ph_en_fin
        self.reflPhEnScaleType = _ph_en_scale_type
        self.reflAngStart = _ang_start
        self.reflAngFin = _ang_fin
        self.reflAngScaleType = _ang_scale_type

    def set_orient(self, _nvx=0, _nvy=0, _nvz=-1, _tvx=1, _tvy=0, _x=0, _y=0):
        """Defines Mirror Orientation in the frame of the incident photon beam
        :param _nvx: horizontal coordinate of central normal vector [m]
        :param _nvy: vertical coordinate of central normal vector [m]
        :param _nvz: longitudinal coordinate of central normal vector [m]
        :param _tvx: horizontal coordinate of central tangential vector [m]
        :param _tvy: vertical coordinate of central tangential vector [m]
        :param _x: horizontal position of mirror center
        :param _y: vertical position of mirror center
        """
        self.nvx = _nvx
        self.nvy = _nvy
        self.nvz = _nvz
        self.tvx = _tvx
        self.tvy = _tvy
        self.x = _x
        self.y = _y
        self.Fx = 0 #i.e. focal lengthes are not set
        self.Fy = 0

class SRWLOptMirEl(SRWLOptMir):
    """Optical Element: Mirror: Ellipsoid"""
    
    def __init__(self, _p=1, _q=1, _ang_graz=1e-03, _r_sag=1.e+23,
                 _size_tang=1, _size_sag=1, _ap_shape='r', _sim_meth=2, _npt=500, _nps=500, _treat_in_out=1, _ext_in=0, _ext_out=0,
                 _nvx=0, _nvy=0, _nvz=-1, _tvx=1, _tvy=0, _x=0, _y=0,
                 _refl=1, _n_ph_en=1, _n_ang=1, _n_comp=1, _ph_en_start=1000., _ph_en_fin=1000., _ph_en_scale_type='lin', _ang_start=0, _ang_fin=0, _ang_scale_type='lin'):
        """
        :param _p: distance from first focus (\"source\") to mirror center [m]
        :param _q: distance from mirror center to second focus (\"image\") [m]
        :param _ang_graz: grazing angle at mirror center at perfect orientation [rad]
        :param _r_sag: sagital radius of curvature at mirror center [m]
        :param _size_tang: size in tangential direction [m]
        :param _size_sag: size in sagital direction [m]
        :param _ap_shape: shape of aperture in local frame ('r' for rectangular, 'e' for elliptical)
        :param _sim_meth: simulation method (1 for "thin" approximation, 2 for "thick" approximation)
        :param _npt: number of mesh points to represent mirror in tangential direction (used for "thin" approximation)
        :param _nps: number of mesh points to represent mirror in sagital direction (used for "thin" approximation)
        :param _treat_in_out: switch specifying how to treat input and output wavefront before and after the main propagation through the optical element:
                0- assume that the input wavefront is defined in the plane before the optical element, and the output wavefront is required in a plane just after the element;
                1- assume that the input wavefront is defined in the plane at the optical element center and the output wavefront is also required at the element center;
                2- assume that the input wavefront is defined in the plane at the optical element center and the output wavefront is also required at the element center; however, before the propagation though the optical element, the wavefront should be propagated through a drift back to a plane just before the optical element, then a special propagator will bring the wavefront to a plane at the optical element exit, and after this the wavefront will be propagated through a drift back to the element center;
        :param _ext_in: optical element extent on the input side, i.e. distance between the input plane and the optical center (positive, in [m]) to be used at wavefront propagation manipulations; if 0, this extent will be calculated internally from optical element parameters
        :param _ext_out: optical element extent on the output side, i.e. distance between the optical center and the output plane (positive, in [m]) to be used at wavefront propagation manipulations; if 0, this extent will be calculated internally from optical element parameters        
        :param _nvx: horizontal coordinate of central normal vector [m]
        :param _nvy: vertical coordinate of central normal vector [m]
        :param _nvz: longitudinal coordinate of central normal vector [m]
        :param _tvx: horizontal coordinate of central tangential vector [m]
        :param _tvy: vertical coordinate of central tangential vector [m]
        :param _x: horizontal position of mirror center
        :param _y: vertical position of mirror center
        :param _refl: reflectivity coefficient to set (can be one number or C-aligned flat complex array vs photon energy vs grazing angle vs component (sigma, pi))
        :param _n_ph_en: number of photon energy values for which the reflectivity coefficient is specified
        :param _n_ang: number of grazing angle values for which the reflectivity coefficient is specified
        :param _n_comp: number of electric field components for which the reflectivity coefficient is specified (can be 1 or 2)
        :param _ph_en_start: initial photon energy value for which the reflectivity coefficient is specified
        :param _ph_en_fin: final photon energy value for which the reflectivity coefficient is specified
        :param _ph_en_scale_type: photon energy sampling type ('lin' for linear, 'log' for logarithmic)
        :param _ang_start: initial grazing angle value for which the reflectivity coefficient is specified
        :param _ang_fin: final grazing angle value for which the reflectivity coefficient is specified
        :param _ang_scale_type: angle sampling type ('lin' for linear, 'log' for logarithmic)      
        """
        self.p = _p
        self.q = _q
        self.angGraz = _ang_graz
        self.radSag = _r_sag
        
        #finishing of the mirror setup requires calling these 3 functions (with their required arguments):
        self.set_dim_sim_meth(_size_tang, _size_sag, _ap_shape, _sim_meth, _npt, _nps, _treat_in_out, _ext_in, _ext_out)
        self.set_orient(_nvx, _nvy, _nvz, _tvx, _tvy, _x, _y)
        self.set_reflect(_refl, _n_ph_en, _n_ang, _n_comp, _ph_en_start, _ph_en_fin, _ph_en_scale_type, _ang_start, _ang_fin, _ang_scale_type)

class SRWLOptMirTor(SRWLOptMir):
    """Optical Element: Mirror: Ellipsoid"""
    
    def __init__(self, _rt=1, _rs=1):
        """
        :param _rt: tangential (major) radius [m]
        :param _rs: sagittal (minor) radius [m]
       """
        self.radTan = _rt
        self.radSag = _rs
        
        #finishing of the mirror setup requires calling these 3 functions (with their required arguments):
        self.set_reflect()
        self.set_dim_sim_meth()
        self.set_orient()

class SRWLOptC(SRWLOpt):
    """Optical Element: Container"""
    
    def __init__(self, _arOpt=None, _arProp=None):
        """
        :param _arOpt: optical element structures list (or array)
        :param _arProp: list of lists of propagation parameters to be used for each individual optical element
            Each element _arProp[i] is a list in which elements mean:
            [0]: Auto-Resize (1) or not (0) Before propagation
            [1]: Auto-Resize (1) or not (0) After propagation
            [2]: Relative Precision for propagation with Auto-Resizing (1. is nominal)
            [3]: Allow (1) or not (0) for semi-analytical treatment of the quadratic (leading) phase terms at the propagation
            [4]: Do any Resizing on Fourier side, using FFT, (1) or not (0)
            [5]: Horizontal Range modification factor at Resizing (1. means no modification)
            [6]: Horizontal Resolution modification factor at Resizing
            [7]: Vertical Range modification factor at Resizing
            [8]: Vertical Resolution modification factor at Resizing
            [9]: Type of wavefront Shift before Resizing (vs which coordinates; not yet implemented)
            [10]: New Horizontal wavefront Center position after Shift (not yet implemented)
            [11]: New Vertical wavefront Center position after Shift (not yet implemented)
        """
        self.arOpt = _arOpt #optical element structures array
        if(_arOpt == None):
            self.arOpt = []
        self.arProp = _arProp #list of lists of propagation parameters to be used for individual optical elements
        if(_arProp == None):
            self.arProp = []
            
    def allocate(self, _nElem):
        self.arOpt = [SRWLOpt()]*_nElem
        self.arProp = [[0]*12]*_nElem

#****************************************************************************
#****************************************************************************
#Setup some transmission-type optical elements
#****************************************************************************
#****************************************************************************
def srwl_opt_setup_CRL(_foc_plane, _delta, _atten_len, _shape, _apert_h, _apert_v, _r_min, _n, _wall_thick, _xc, _yc, _void_cen_rad=None, _e_start=0, _e_fin=0):
    """
    Setup Transmission type Optical Element which simulates Compound Refractive Lens (CRL)
    :param _foc_plane: plane of focusing: 1- horizontal, 2- vertical, 3- both
    :param _delta: refractive index decrement (can be one number of array vs photon energy)
    :param _atten_len: attenuation length [m] (can be one number of array vs photon energy)
    :param _shape: 1- parabolic, 2- circular (spherical)
    :param _apert_h: horizontal aperture size [m]
    :param _apert_v: vertical aperture size [m]
    :param _r_min: radius (on tip of parabola for parabolic shape) [m]
    :param _n: number of lenses ("holes")
    :param _wall_thick: min. wall thickness between "holes" [m]
    :param _xc: horizontal coordinate of center [m]
    :param _yc: vertical coordinate of center [m]
    :param _void_cen_rad: flat array/list of void center coordinates and radii: [x1, y1, r1, x2, y2, r2,...] 
    :param _e_start: initial photon energy
    :param _e_fin: final photon energy
    :return: transmission (SRWLOptT) type optical element which simulates CRL
    """
    def ray_path_in_one_CRL(_x, _y, _foc_plane, _shape, _half_apert, _r_min, _wall_thick): #CRL is always centered
        rE2 = 0
        if((_foc_plane == 1) or (_foc_plane == 3)): #focusing in horizontal plane
            rE2 += _x*_x
        if((_foc_plane == 2) or (_foc_plane == 3)): #focusing in vertical or in both planes
            rE2 += _y*_y
            
        halfApE2 = _half_apert*_half_apert
        
        sectLen = 0
        if(_shape == 1): #parabolic
            a = 1./_r_min
            sectLen = _wall_thick + a*halfApE2
            if(rE2 < halfApE2):
                return _wall_thick + a*rE2
        elif(_shape == 2): #circular (or spherical)
            radE2 = _r_min*_r_min
            sectLen = _wall_thick + 2*_r_min
            if(_half_apert < _r_min):
                sectLen = _wall_thick + 2*(_r_min - sqrt(radE2 - halfApE2))
                if(rE2 < halfApE2):
                    return sectLen - 2*sqrt(radE2 - rE2)
            elif(rE2 < radE2):
                return sectLen - 2*sqrt(radE2 - rE2)
        return sectLen

    def ray_path_in_spheres(_x, _y, _void_cen_rad):
        n = int(round(len(_void_cen_rad)/3))
        sumPath = 0.
        for i in range(n):
            i3 = i*3
            dx = _x - _void_cen_rad[i3]
            dy = _y - _void_cen_rad[i3 + 1]
            rVoid = _void_cen_rad[i3 + 2]
            uE2 = dx*dx + dy*dy
            rVoidE2 = rVoid*rVoid
            if(uE2 < rVoidE2):
                sumPath += 2*sqrt(rVoidE2 - uE2)
                #print('Void crossed:', dx, dy, rVoid, sumPath)
        return sumPath

    foc_len = (0.5*_r_min/(_n*_delta))
    print('Optical Element Setup: CRL Focal Length:', foc_len, 'm')

    fx = 1e+23
    fy = 1e+23
    if(_foc_plane != 1):
        fy = foc_len
    if(_foc_plane != 2):
        fx = foc_len

    rx = _apert_h*1.1
    ry = _apert_v*1.1
    nx = 1001
    ny = 1001
    ne = 1
    arDelta = [0]
    arAttenLen = [0]

    #if(((type(_delta).__name__ == 'list') or (type(_delta).__name__ == 'array')) and ((type(_atten_len).__name__ == 'list') or (type(_atten_len).__name__ == 'array'))):
    if(isinstance(_delta, list) or isinstance(_delta, array)) and (isinstance(_atten_len, list) or isinstance(_atten_len, array)):
        ne = len(_delta)
        ne1 = len(_atten_len)
        if(ne > ne1):
            ne = ne1
        arDelta = _delta
        arAttenLen = _atten_len
    else:
        arDelta[0] = _delta
        arAttenLen[0] = _atten_len
    
    opT = SRWLOptT(nx, ny, rx, ry, None, 1, fx, fy, _xc, _yc, ne, _e_start, _e_fin)

    halfApert = 0.5*_apert_h
    halfApertV = 0.5*_apert_v

    if(_foc_plane == 2): #1D lens, vertical is focusing plane
        halfApert = halfApertV
    elif(_foc_plane == 3): #2D lens
        if(halfApert > halfApertV):
            halfApert = halfApertV
    
    hx = rx/(nx - 1)
    hy = ry/(ny - 1)
    ofst = 0
    y = -0.5*ry #CRL is always centered on the grid, however grid can be shifted
    for iy in range(ny):
        x = -0.5*rx
        for ix in range(nx):
            pathInBody = _n*ray_path_in_one_CRL(x, y, _foc_plane, _shape, halfApert, _r_min, _wall_thick)

            if(_void_cen_rad != None): #eventually subtract path in voids
                pathInBody -= ray_path_in_spheres(x, y, _void_cen_rad)
                
            for ie in range(ne):
                opT.arTr[ofst] = exp(-0.5*pathInBody/arAttenLen[ie]) #amplitude transmission
                opT.arTr[ofst + 1] = -arDelta[ie]*pathInBody #optical path difference
                ofst += 2
            x += hx
        y += hy
    return opT

#****************************************************************************
def srwl_opt_setup_cyl_fiber(_foc_plane, _delta_ext, _delta_core, _atten_len_ext, _atten_len_core, _diam_ext, _diam_core, _xc, _yc):
    """
    Setup Transmission type Optical Element which simulates Cylindrical Fiber
    :param _foc_plane: plane of focusing: 1- horizontal (i.e. fiber is parallel to vertical axis), 2- vertical (i.e. fiber is parallel to horizontal axis)
    :param _delta_ext: refractive index decrement of extenal layer
    :param _delta_core: refractive index decrement of core
    :param _atten_len_ext: attenuation length [m] of external layer
    :param _atten_len_core: attenuation length [m] of core
    :param _diam_ext: diameter [m] of external layer
    :param _diam_core: diameter [m] of core
    :param _xc: horizontal coordinate of center [m]
    :param _yc: vertical coordinate of center [m]
    :return: transmission (SRWLOptT) type optical element which simulates Cylindrical Fiber
    """

    def ray_path_in_cyl(_dx, _diam):
        r = 0.5*_diam
        pathInCyl = 0
        if((_dx > -r) and (_dx < r)):
            pathInCyl = 2*sqrt(r*r - _dx*_dx)
        return pathInCyl

    ne = 1
    nx = 101
    ny = 1001
    rx = 10e-03
    ry = _diam_ext*1.2
    if(_foc_plane == 1): #focusing plane is horizontal
        nx = 1001
        ny = 101
        rx = _diam_ext*1.2
        ry = 10e-03

    opT = SRWLOptT(nx, ny, rx, ry, None, 1, 1e+23, 1e+23, _xc, _yc)

    hx = rx/(nx - 1)
    hy = ry/(ny - 1)
    ofst = 0
    pathInExt = 0
    pathInCore = 0

    if(_foc_plane == 2): #focusing plane is vertical
        y = -0.5*ry #cylinder is always centered on the grid, however grid can be shifted
        for iy in range(ny):
            pathInExt = 0; pathInCore = 0
            if(_diam_core > 0):
                pathInCore = ray_path_in_cyl(y, _diam_core)
            pathInExt = ray_path_in_cyl(y, _diam_ext) - pathInCore
            argAtten = -0.5*pathInExt/_atten_len_ext
            if(_atten_len_core > 0):
                argAtten -= 0.5*pathInCore/_atten_len_core
            ampTr = exp(argAtten) #amplitude transmission
            optPathDif = -_delta_ext*pathInExt - _delta_core*pathInCore #optical path difference
            for ix in range(nx):                    
                opT.arTr[ofst] = ampTr #amplitude transmission
                opT.arTr[ofst + 1] = optPathDif #optical path difference
                ofst += 2
            y += hy
    else: #focusing plane is horizontal
        perY = 2*nx
        x = -0.5*rx #cylinder is always centered on the grid, however grid can be shifted
        for ix in range(nx):
            pathInExt = 0; pathInCore = 0
            if(_diam_core > 0):
                pathInCore = ray_path_in_cyl(x, _diam_core)
            pathInExt = ray_path_in_cyl(x, _diam_ext) - pathInCore
            argAtten = -0.5*pathInExt/_atten_len_ext
            if(_atten_len_core > 0):
                argAtten -= 0.5*pathInCore/_atten_len_core
            ampTr = exp(argAtten) #amplitude transmission
            optPathDif = -_delta_ext*pathInExt - _delta_core*pathInCore #optical path difference
            ix2 = ix*2
            for iy in range(ny):
                ofst = iy*perY + ix2
                opT.arTr[ofst] = ampTr #amplitude transmission
                opT.arTr[ofst + 1] = optPathDif #optical path difference
            x += hx
    return opT

#****************************************************************************
#****************************************************************************
#Auxiliary utility functions
#****************************************************************************
#****************************************************************************
def srwl_uti_interp_1d(_x, _x_min, _x_step, _nx, _ar_f, _ord=3, _ix_per=1, _ix_ofst=0):
    """
    Interpolate 1D function value tabulated on equidistant mesh, using polynomial interpolation
    :param _x: argument at which function value should be calculated
    :param _x_min: minimal argument value of the tabulated function
    :param _x_step: step of mesh at which function is tabulated
    :param _nx: number of points in mesh at which function is tabulated
    :param _ar_f: tabulated function list or array
    :param _ord: order of polynomial interpolation (1- linear, 2- quadratic, 3- cubic)
    :param _ix_per: argument index period of function data alignment (e.g. to interpolate one component of complex data, or in one dimension of multi-dimensional data)
    :param _ix_ofst: argument index offset of function data alignment
    :return: function value found by polynomial interpolation
    """
    if(_ord == 1):
        i0 = int(trunc((_x - _x_min)/_x_step + 1.e-09))
        if(i0 < 0):
            i0 = 0
        elif(i0 >= _nx - 1):
            i0 = _nx - 2
        i1 = i0 + 1
        f0 = _ar_f[i0*_ix_per + _ix_ofst]
        f1 = _ar_f[i1*_ix_per + _ix_ofst]
        t = (_x - (_x_min + _x_step*i0))/_x_step
        return f0 + (f1 - f0)*t
    elif(_ord == 2):
        i0 = int(round((_x - _x_min)/_x_step))
        if(i0 < 1):
            i0 = 1
        elif(i0 >= _nx - 1):
            i0 = _nx - 2
        im1 = i0 - 1
        i1 = i0 + 1
        t = (_x - (_x_min + _x_step*i0))/_x_step
        a0 = _ar_f[i0*_ix_per + _ix_ofst]
        fm1 = _ar_f[im1*_ix_per + _ix_ofst]
        f1 = _ar_f[i1*_ix_per + _ix_ofst]
        a1 = 0.5*(f1 - fm1)
        a2 = 0.5*(fm1 + f1 - 2*a0)
        return a0 + t*(a1 + t*a2)
    elif(_ord == 3):
        i0 = int(trunc((_x - _x_min)/_x_step + 1.e-09))
        if(i0 < 1):
            i0 = 1
        elif(i0 >= _nx - 2):
            i0 = _nx - 3
        im1 = i0 - 1
        i1 = i0 + 1
        i2 = i0 + 2
        t = (_x - (_x_min + _x_step*i0))/_x_step
        a0 = _ar_f[i0*_ix_per + _ix_ofst]
        fm1 = _ar_f[im1*_ix_per + _ix_ofst]
        f1 = _ar_f[i1*_ix_per + _ix_ofst]
        f2 = _ar_f[i2*_ix_per + _ix_ofst]
        a1 = -0.5*a0 + f1 - f2/6. - fm1/3.
        a2 = -a0 + 0.5*(f1 + fm1)
        a3 = 0.5*(a0 - f1) + (f2 - fm1)/6.
        return a0 + t*(a1 + t*(a2 + t*a3))
    return 0

#****************************************************************************
def srwl_uti_interp_2d(_x, _y, _x_min, _x_step, _nx, _y_min, _y_step, _ny, _ar_f, _ord=3, _ix_per=1, _ix_ofst=0):
    """
    Interpolate 2D function value tabulated on equidistant rectangular mesh and represented by C-aligned flat array, using polynomial interpolation
    :param _x: first argument at which function value should be calculated
    :param _y: second argument at which function value should be calculated
    :param _x_min: minimal value of the first argument of the tabulated function
    :param _x_step: step of the first argument at which function is tabulated
    :param _nx: number of points vs first argument at which function is tabulated
    :param _y_min: minimal value of the second argument of the tabulated function
    :param _y_step: step of the second argument at which function is tabulated
    :param _ny: number of points vs second argument at which function is tabulated
    :param _ar_f: function tabulated on 2D mesh, aligned as "flat" C-type list or array (first argument is changing most frequently)
    :param _ord: "order" of polynomial interpolation (1- bi-linear (on 4 points), 2- "bi-quadratic" (on 6 points), 3- "bi-cubic" (on 12 points))
    :param _ix_per: period of first argument index of the function data alignment (e.g. to interpolate one component of complex data, or in one dimension of multi-dimensional data)
    :param _ix_ofst: offset of the first argument index in function data alignment
    :return: function value found by 2D polynomial interpolation
    """
    if(_ord == 1): #bi-linear interpolation based on 4 points
        ix0 = int(trunc((_x - _x_min)/_x_step + 1.e-09))
        if(ix0 < 0):
            ix0 = 0
        elif(ix0 >= _nx - 1):
            ix0 = _nx - 2
        ix1 = ix0 + 1
        tx = (_x - (_x_min + _x_step*ix0))/_x_step
        
        iy0 = int(trunc((_y - _y_min)/_y_step + 1.e-09))
        if(iy0 < 0):
            iy0 = 0
        elif(iy0 >= _ny - 1):
            iy0 = _ny - 2
        iy1 = iy0 + 1
        ty = (_y - (_y_min + _y_step*iy0))/_y_step

        nx_ix_per = _nx*_ix_per
        iy0_nx_ix_per = iy0*nx_ix_per
        iy1_nx_ix_per = iy1*nx_ix_per
        ix0_ix_per_p_ix_ofst = ix0*_ix_per + _ix_ofst
        ix1_ix_per_p_ix_ofst = ix1*_ix_per + _ix_ofst
        a00 = _ar_f[iy0_nx_ix_per + ix0_ix_per_p_ix_ofst]
        f10 = _ar_f[iy0_nx_ix_per + ix1_ix_per_p_ix_ofst]
        f01 = _ar_f[iy1_nx_ix_per + ix0_ix_per_p_ix_ofst]
        f11 = _ar_f[iy1_nx_ix_per + ix1_ix_per_p_ix_ofst]
        a10 = f10 - a00
        a01 = f01 - a00
        a11 = a00 - f01 - f10 + f11
        return a00 + tx*(a10 + ty*a11) + ty*a01

    elif(_ord == 2): #bi-quadratic interpolation based on 6 points
        ix0 = int(round((_x - _x_min)/_x_step))
        if(ix0 < 1):
            ix0 = 1
        elif(ix0 >= _nx - 1):
            ix0 = _nx - 2
        ixm1 = ix0 - 1
        ix1 = ix0 + 1
        tx = (_x - (_x_min + _x_step*ix0))/_x_step

        iy0 = int(round((_y - _y_min)/_y_step))
        if(iy0 < 1):
            iy0 = 1
        elif(iy0 >= _ny - 1):
            iy0 = _ny - 2
        iym1 = iy0 - 1
        iy1 = iy0 + 1
        ty = (_y - (_y_min + _y_step*iy0))/_y_step

        nx_ix_per = _nx*_ix_per
        iym1_nx_ix_per = iym1*nx_ix_per
        iy0_nx_ix_per = iy0*nx_ix_per
        iy1_nx_ix_per = iy1*nx_ix_per
        ixm1_ix_per_p_ix_ofst = ixm1*_ix_per + _ix_ofst
        ix0_ix_per_p_ix_ofst = ix0*_ix_per + _ix_ofst
        ix1_ix_per_p_ix_ofst = ix1*_ix_per + _ix_ofst
        fm10 = _ar_f[iy0_nx_ix_per + ixm1_ix_per_p_ix_ofst]
        a00 = _ar_f[iy0_nx_ix_per + ix0_ix_per_p_ix_ofst]
        f10 = _ar_f[iy0_nx_ix_per + ix1_ix_per_p_ix_ofst]
        f0m1 = _ar_f[iym1_nx_ix_per + ix0_ix_per_p_ix_ofst]
        f01 = _ar_f[iy1_nx_ix_per + ix0_ix_per_p_ix_ofst]
        f11 = _ar_f[iy1_nx_ix_per + ix1_ix_per_p_ix_ofst]
        a10 = 0.5*(f10 - fm10)
        a01 = 0.5*(f01 - f0m1)
        a11 = a00 - f01 - f10 + f11
        a20 = 0.5*(f10 + fm10) - a00
        a02 = 0.5*(f01 + f0m1) - a00
        return a00 + tx*(a10 + tx*a20 + ty*a11) + ty*(a01 + ty*a02)
    
    elif(_ord == 3): #bi-cubic interpolation based on 12 points
        ix0 = int(trunc((_x - _x_min)/_x_step + 1.e-09))
        if(ix0 < 1):
            ix0 = 1
        elif(ix0 >= _nx - 2):
            ix0 = _nx - 3
        ixm1 = ix0 - 1
        ix1 = ix0 + 1
        ix2 = ix0 + 2
        tx = (_x - (_x_min + _x_step*ix0))/_x_step

        iy0 = int(trunc((_y - _y_min)/_y_step + 1.e-09))
        if(iy0 < 1):
            iy0 = 1
        elif(iy0 >= _ny - 2):
            iy0 = _ny - 3
        iym1 = iy0 - 1
        iy1 = iy0 + 1
        iy2 = iy0 + 2
        ty = (_y - (_y_min + _y_step*iy0))/_y_step

        nx_ix_per = _nx*_ix_per
        iym1_nx_ix_per = iym1*nx_ix_per
        iy0_nx_ix_per = iy0*nx_ix_per
        iy1_nx_ix_per = iy1*nx_ix_per
        iy2_nx_ix_per = iy2*nx_ix_per
        ixm1_ix_per_p_ix_ofst = ixm1*_ix_per + _ix_ofst
        ix0_ix_per_p_ix_ofst = ix0*_ix_per + _ix_ofst
        ix1_ix_per_p_ix_ofst = ix1*_ix_per + _ix_ofst
        ix2_ix_per_p_ix_ofst = ix2*_ix_per + _ix_ofst
        f0m1 = _ar_f[iym1_nx_ix_per + ix0_ix_per_p_ix_ofst]
        f1m1 = _ar_f[iym1_nx_ix_per + ix1_ix_per_p_ix_ofst]
        fm10 = _ar_f[iy0_nx_ix_per + ixm1_ix_per_p_ix_ofst]
        a00 = _ar_f[iy0_nx_ix_per + ix0_ix_per_p_ix_ofst]
        f10 = _ar_f[iy0_nx_ix_per + ix1_ix_per_p_ix_ofst]
        f20 = _ar_f[iy0_nx_ix_per + ix2_ix_per_p_ix_ofst]
        fm11 = _ar_f[iy1_nx_ix_per + ixm1_ix_per_p_ix_ofst]
        f01 = _ar_f[iy1_nx_ix_per + ix0_ix_per_p_ix_ofst]
        f11 = _ar_f[iy1_nx_ix_per + ix1_ix_per_p_ix_ofst]
        f21 = _ar_f[iy1_nx_ix_per + ix2_ix_per_p_ix_ofst]
        f02 = _ar_f[iy2_nx_ix_per + ix0_ix_per_p_ix_ofst]
        f12 = _ar_f[iy2_nx_ix_per + ix1_ix_per_p_ix_ofst]
        a10 = -0.5*a00 + f10 - f20/6 - fm10/3
        a01 = -0.5*a00 + f01 - f02/6 - f0m1/3
        a11 = -0.5*(f01 + f10) + (f02 - f12 + f20 - f21)/6 + (f0m1 - f1m1 + fm10 - fm11)/3 + f11
        a20 = -a00 + 0.5*(f10 + fm10)
        a02 = -a00 + 0.5*(f01 + f0m1)
        a21 = a00 - f01 + 0.5*(f11 - f10 - fm10 + fm11)
        a12 = a00 - f10 + 0.5*(f11 - f01 - f0m1 + f1m1)
        a30 = 0.5*(a00 - f10) + (f20 - fm10)/6
        a03 = 0.5*(a00 - f01) + (f02 - f0m1)/6
        a31 = 0.5*(f01 + f10 - f11 - a00) + (f21 + fm10 - f20 - fm11)/6
        a13 = 0.5*(f10 - f11 - a00 + f01) + (f0m1 + f12 - f02 - f1m1)/6
        return a00 + tx*(a10 + tx*(a20 + tx*(a30 + ty*a31) + ty*a21) + ty*a11) + ty*(a01 + ty*(a02 + ty*(a03 + tx*a13) + tx*a12))
    return 0

#****************************************************************************
def srwl_uti_rand_fill_vol(_np, _x_min, _x_max, _nx, _ar_y_vs_x_min, _ar_y_vs_x_max, _y_min, _y_max, _ny, _ar_z_vs_xy_min, _ar_z_vs_xy_max):
    """
    Generate coordinates of ponts randomly filling 3D volume limited by two arbitrary curves (defining base) and two surfaces
    :param _np: number of random points in rectangular parallelepiped to try
    :param _x_min: min. x coordinate
    :param _x_max: max. x coordinate
    :param _nx: number of points vs x coord.
    :param _ar_y_vs_x_min: min. y vs x array
    :param _ar_y_vs_x_max: max. y vs x array
    :param _y_min: min. y coordinate
    :param _y_max: max. y coordinate
    :param _ny: number of points vs y coord.
    :param _ar_z_vs_xy_min: min. z vs x and y flat 2D array
    :param _ar_z_vs_xy_max: max. z vs x and y flat 2D array
    :returns: flat array of point coordinates: array('d', [x1,y1,z1,x2,y2,z2,...])
    """
    yMin = _ar_y_vs_x_min[0]
    yMax = _ar_y_vs_x_max[0]
    for ix in range(_nx):
        yMinCur = _ar_y_vs_x_min[ix]
        if(yMin > yMinCur):
            yMin = yMinCur
        yMaxCur = _ar_y_vs_x_max[ix]
        if(yMax < yMaxCur):
            yMax = yMaxCur

    nxy = _nx*_ny
    zMin = _ar_z_vs_xy_min[0]
    zMax = _ar_z_vs_xy_max[0]
    for ixy in range(nxy):
        zMinCur = _ar_z_vs_xy_min[ixy]
        if(zMin > zMinCur):
            zMin = zMinCur
        zMaxCur = _ar_z_vs_xy_max[ixy]
        if(zMax < zMaxCur):
            zMax = zMaxCur

    xStep = (_x_max - _x_min)/(_nx - 1)
    yStep = (_y_max - _y_min)/(_ny - 1)
    xCen = 0.5*(_x_min + _x_max)
    yCen = 0.5*(yMin + yMax)
    zCen = 0.5*(zMin + zMax)
    xRange = _x_max - _x_min
    yRange = yMax - yMin
    zRange = zMax - zMin

    arPtCoord = array('d', [0]*(_np*3))
    iPtCount = 0
    random.seed()
    for i in range(_np):
        x = xCen + xRange*(random.random() - 0.5)
        y = yCen + yRange*(random.random() - 0.5)
        yTestMin = srwl_uti_interp_1d(x, _x_min, xStep, _nx, _ar_y_vs_x_min)
        yTestMax = srwl_uti_interp_1d(x, _x_min, xStep, _nx, _ar_y_vs_x_max)
        if((y >= yTestMin) and (y <= yTestMax)):
            z = zCen + zRange*(random.random() - 0.5)
            zTestMin = srwl_uti_interp_2d(x, y, _x_min, xStep, _nx, _y_min, yStep, _ny, _ar_z_vs_xy_min)
            zTestMax = srwl_uti_interp_2d(x, y, _x_min, xStep, _nx, _y_min, yStep, _ny, _ar_z_vs_xy_max)
            if((z >= zTestMin) and (z <= zTestMax)):
                ofst = iPtCount*3
                arPtCoord[ofst] = x
                arPtCoord[ofst + 1] = y
                arPtCoord[ofst + 2] = z
                iPtCount += 1

    if(iPtCount == _np):
        return arPtCoord
    else: #is there faster way to truncate array?
        nResCoord = iPtCount*3
        arResPtCoord = array('d', [0]*nResCoord)
        for i in range(nResCoord):
            arResPtCoord[i] = arPtCoord[i]
            
        return arResPtCoord

#****************************************************************************
def srwl_uti_proc_is_master():
    """
    Check if process is Master (in parallel processing sense)
    """
    try:
        resImpMPI4Py = __import__('mpi4py', globals(), locals(), ['MPI'], -1) #MPI module dynamic load
        #multiply re-import won't hurt; but it would be better to avoid this(?)
        MPI = resImpMPI4Py.MPI
        comMPI = MPI.COMM_WORLD
        rankMPI = comMPI.Get_rank()
        if(rankMPI == 0):
            return True
        else:
            return False
    except:
        return True

#**********************Auxiliary function to write tabulated resulting Intensity data to ASCII file:
def srwl_uti_save_intens_ascii(_ar_intens, _mesh, _file_path, _n_stokes=1):
    f = open(_file_path, 'w')
    f.write('#C-aligned Intensity (inner loop is vs photon energy, outer loop vs vertical position)\n')
    f.write('#' + repr(_mesh.eStart) + ' #Initial Photon Energy [eV]\n')
    f.write('#' + repr(_mesh.eFin) + ' #Final Photon Energy [eV]\n')
    f.write('#' + repr(_mesh.ne) + ' #Number of points vs Photon Energy\n')
    f.write('#' + repr(_mesh.xStart) + ' #Initial Horizontal Position [m]\n')
    f.write('#' + repr(_mesh.xFin) + ' #Final Horizontal Position [m]\n')
    f.write('#' + repr(_mesh.nx) + ' #Number of points vs Horizontal Position\n')
    f.write('#' + repr(_mesh.yStart) + ' #Initial Vertical Position [m]\n')
    f.write('#' + repr(_mesh.yFin) + ' #Final Vertical Position [m]\n')
    f.write('#' + repr(_mesh.ny) + ' #Number of points vs Vertical Position\n')
    f.write('#' + repr(_n_stokes) + ' #Number of Stokes Components\n')
    nVal = _mesh.ne*_mesh.nx*_mesh.ny*_n_stokes
    for i in range(nVal): #write all data into one column using "C-alignment" as a "flat" 1D array
        f.write(' ' + repr(_ar_intens[i]) + '\n')
    f.close()

#****************************************************************************
#****************************************************************************
#Wavefront manipulation functions
#****************************************************************************
#****************************************************************************
def srwl_wfr_emit_prop_multi_e(_e_beam, _mag, _mesh, _sr_meth, _sr_rel_prec, _n_part_tot, _n_part_avg_proc=1, _n_save_per=100, _file_path=None, _sr_samp_fact=-1, _opt_bl=None, _pres_ang=0):
    """
    Calculate Stokes Parameters of Emitted (and Propagated, if beamline is defined) Partially-Coherent SR
    :param _e_beam: Finite-Emittance e-beam (SRWLPartBeam type)
    :param _mag: Magnetic Field container (magFldCnt type)
    :param _mesh: mesh vs photon energy, horizontal and vertical positions (SRWLRadMesh type) on which initial SR should be calculated
    :param _sr_meth: SR Electric Field calculation method to be used (0- "manual", 1- "auto-undulator", 2- "auto-wiggler")
    :param _sr_rel_prec: relative precision for SR Electric Field calculation (usually 0.01 is OK, the smaller the more accurate)
    :param _n_part_tot: total number of "macro-electrons" to be used in the calculation
    :param _n_part_avg_proc: number of "macro-electrons" to be used in calculation at each "slave" before sending Stokes data to "master" (effective if the calculation is run via MPI)
    :param _n_save_per: periodicity of saving intermediate average Stokes data to file by master process
    :param _file_path: path to file for saving intermediate average Stokes data by master process
    :param _sr_samp_fact: oversampling factor for calculating of initial wavefront for subsequent propagation (effective if >0)
    :param _opt_bl: optical beamline (container) to propagate the radiation through (SRWLOptC type)
    :param _pres_ang: switch specifying presentation of the resulting Stokes parameters: coordinate (0) or angular (1)
    """

    nProc = 1
    rank = 1
    MPI = None
    comMPI = None
    try:
        resImpMPI4Py = __import__('mpi4py', globals(), locals(), ['MPI'], -1) #MPI module load
        MPI = resImpMPI4Py.MPI
        comMPI = MPI.COMM_WORLD
        rank = comMPI.Get_rank()
        nProc = comMPI.Get_size()

    except:
        print('Calculation will be sequential (non-parallel), because "mpi4py" module can not be loaded')

    if(nProc <= 1):
        _n_part_avg_proc = _n_part_tot

    wfr = SRWLWfr() #Wavefronts to be used in each process
    wfr.allocate(_mesh.ne, _mesh.nx, _mesh.ny) #Numbers of points vs Photon Energy, Horizontal and Vertical Positions
    wfr.mesh.set_from_other(_mesh)
    wfr.partBeam = deepcopy(_e_beam)
    arPrecParSR = [_sr_meth, _sr_rel_prec, 0, 0, 50000, 0, _sr_samp_fact]
    meshRes = SRWLRadMesh()

    resStokes = None
    workStokes = None
    iAvgProc = 0
    iSave = 0
        
    if(((rank == 0) or (nProc == 1)) and (_opt_bl != None)): #calculate once the central wavefront in the master process (this has to be done only if propagation is required)
        srwl.CalcElecFieldSR(wfr, 0, _mag, arPrecParSR)
        #print('DEBUG MESSAGE: Central Wavefront calculated')
        srwl.PropagElecField(wfr, _opt_bl)
        #print('DEBUG MESSAGE: Central Wavefront propagated')
        if(_pres_ang > 0):
            srwl.SetRepresElecField(wfr, 'a')
        
        meshRes.set_from_other(wfr.mesh)

        if(nProc > 1): #send resulting mesh to all workers
            #comMPI.send(wfr.mesh, dest=)
            arMesh = array('f', [meshRes.eStart, meshRes.eFin, meshRes.ne, meshRes.xStart, meshRes.xFin, meshRes.nx, meshRes.yStart, meshRes.yFin, meshRes.ny])
            #comMPI.Bcast([arMesh, MPI.FLOAT], root=MPI.ROOT)
            #comMPI.Bcast([arMesh, MPI.FLOAT])

            for iRank in range(nProc - 1):
                dst = iRank + 1
                comMPI.Send([arMesh, MPI.FLOAT], dest=dst)
            #print('DEBUG MESSAGE: Mesh of Propagated central wavefront broadcasted')

        resStokes = SRWLStokes(1, 'f', meshRes.eStart, meshRes.eFin, meshRes.ne, meshRes.xStart, meshRes.xFin, meshRes.nx, meshRes.yStart, meshRes.yFin, meshRes.ny)
        wfr.calc_stokes(resStokes)
        workStokes = SRWLStokes(1, 'f', meshRes.eStart, meshRes.eFin, meshRes.ne, meshRes.xStart, meshRes.xFin, meshRes.nx, meshRes.yStart, meshRes.yFin, meshRes.ny)

        #print('DEBUG MESSAGE:  parameters of Propagated central wavefront calculated')
        #srwl_uti_save_intens_ascii(resStokes.arS, meshRes, _file_path, 1)
        #sys.exit(0)
        iAvgProc += 1
        iSave += 1
 
    elecX0 = _e_beam.partStatMom1.x
    elecXp0 = _e_beam.partStatMom1.xp
    elecY0 = _e_beam.partStatMom1.y
    elecYp0 = _e_beam.partStatMom1.yp
    elecGamma0 = _e_beam.partStatMom1.gamma
    elecE0 = elecGamma0*(0.51099890221e-03)
    
    elecSigXe2 = _e_beam.arStatMom2[0] #<(x-x0)^2>
    elecMXXp = _e_beam.arStatMom2[1] #<(x-x0)*(xp-xp0)>
    elecSigXpe2 = _e_beam.arStatMom2[2] #<(xp-xp0)^2>
    elecSigYe2 =_e_beam.arStatMom2[3] #<(y-y0)^2>
    elecMYYp = _e_beam.arStatMom2[4] #<(y-y0)*(yp-yp0)>
    elecSigYpe2 = _e_beam.arStatMom2[5] #<(yp-yp0)^2>
    elecRelEnSpr = sqrt(_e_beam.arStatMom2[10]) #<(E-E0)^2>/E0^2
    elecAbsEnSpr = elecE0*elecRelEnSpr
    #print('DEBUG MESSAGE: elecAbsEnSpr=', elecAbsEnSpr)
    
    multX = 0.5/(elecSigXe2*elecSigXpe2 - elecMXXp*elecMXXp)
    BX = elecSigXe2*multX
    GX = elecSigXpe2*multX
    AX = elecMXXp*multX
    SigPX = 1/sqrt(2*GX)
    SigQX = sqrt(GX/(2*(BX*GX - AX*AX)))
    multY = 0.5/(elecSigYe2*elecSigYpe2 - elecMYYp*elecMYYp)
    BY = elecSigYe2*multY
    GY = elecSigYpe2*multY
    AY = elecMYYp*multY
    SigPY = 1/sqrt(2*GY)
    SigQY = sqrt(GY/(2*(BY*GY - AY*AY)))

    nPartPerProc = int(_n_part_tot/nProc)

    #print('DEBUG MESSAGE: rank=', rank)
    if((rank > 0) or (nProc == 1)):

        if((nProc > 1) and (_opt_bl != None)): #receive mesh for the resulting wavefront from the master
            arMesh = array('f', [0]*9)
            #comMPI.Recv([arMesh, MPI.FLOAT], source=0)
            comMPI.Recv([arMesh, MPI.FLOAT], source=MPI.ANY_SOURCE)
            #comMPI.Bcast([arMesh, MPI.FLOAT], root=0)
            meshRes.eStart = arMesh[0]
            meshRes.eFin = arMesh[1]
            meshRes.ne = int(arMesh[2])
            meshRes.xStart = arMesh[3]
            meshRes.xFin = arMesh[4]
            meshRes.nx = int(arMesh[5])
            meshRes.yStart = arMesh[6]
            meshRes.yFin = arMesh[7]
            meshRes.ny = int(arMesh[8])
            #sys.exit(0)

        nRadPt = meshRes.ne*meshRes.nx*meshRes.ny
        nStPt = nRadPt*4
        randAr = array('d', [0]*5) #to expend to 6D eventually

        random.seed(rank)
        
        for i in range(nPartPerProc): #loop over macro-electrons

            for ir in range(5): #to expend to 6D eventually
                randAr[ir] = random.gauss(0, 1)
            #DEBUG
            #if(i == 0):
            #    randAr = array('d', [0,0,0,2,0])
            #if(i == 1):
            #    randAr = array('d', [0,0,0,-2,0])
            #END DEBUG

            auxPXp = SigQX*randAr[0]
            auxPX = SigPX*randAr[1] + AX*auxPXp/GX
            wfr.partBeam.partStatMom1.x = elecX0 + auxPX
            wfr.partBeam.partStatMom1.xp = elecXp0 + auxPXp
            auxPYp = SigQY*randAr[2]
            auxPY = SigPY*randAr[3] + AY*auxPYp/GY
            wfr.partBeam.partStatMom1.y = elecY0 + auxPY
            wfr.partBeam.partStatMom1.yp = elecYp0 + auxPYp
            #wfr.partBeam.partStatMom1.gamma = (elecEn0 + elecAbsEnSpr*randAr[4])/0.51099890221e-03 #Relative Energy
            wfr.partBeam.partStatMom1.gamma = elecGamma0*(1 + elecAbsEnSpr*randAr[4]/elecE0)

            #reset mesh, because it may be modified by CalcElecFieldSR and PropagElecField
            wfr.mesh.set_from_other(_mesh)
            wfr.presCA = 0 #presentation/domain: 0- coordinates, 1- angles
            wfr.presFT = 0 #presentation/domain: 0- frequency (photon energy), 1- time

            if(nProc == 1):
                print('i=', i, 'Electron Coord.: x=', wfr.partBeam.partStatMom1.x, 'x\'=', wfr.partBeam.partStatMom1.xp, 'y=', wfr.partBeam.partStatMom1.y, 'y\'=', wfr.partBeam.partStatMom1.yp, 'E=',  wfr.partBeam.partStatMom1.gamma*0.51099890221e-03)

            try:
                srwl.CalcElecFieldSR(wfr, 0, _mag, arPrecParSR) #calculate Electric Field emitted by current electron

                if(_opt_bl != None):
                    srwl.PropagElecField(wfr, _opt_bl) #propagate Electric Field emitted by the electron

                if(_pres_ang > 0):
                    srwl.SetRepresElecField(wfr, 'a')

            except:
                traceback.print_exc()

            if(workStokes == None):
                workStokes = SRWLStokes(1, 'f', wfr.mesh.eStart, wfr.mesh.eFin, wfr.mesh.ne, wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.nx, wfr.mesh.yStart, wfr.mesh.yFin, wfr.mesh.ny)
            else:
                nPtCur = wfr.mesh.ne*wfr.mesh.nx*wfr.mesh.ny*4
                if(len(workStokes.arS) < nPtCur):
                    del workStokes.arS
                    workStokes.arS = array('f', [0]*nPtCur)
                    #workStokes.mesh.set_from_other(wfr.mesh)

            wfr.calc_stokes(workStokes) #calculate Stokes parameters from Electric Field

            if(resStokes == None):
                resStokes = SRWLStokes(1, 'f', meshRes.eStart, meshRes.eFin, meshRes.ne, meshRes.xStart, meshRes.xFin, meshRes.nx, meshRes.yStart, meshRes.yFin, meshRes.ny)
 
            if(_opt_bl == None):
                resStokes.avg_update_same_mesh(workStokes, iAvgProc, 1)
            else:
                #print('DEBUG MESSAGE: Started interpolation of current wavefront on resulting mesh')
                resStokes.avg_update_interp(workStokes, iAvgProc, 1, 1)
                #print('DEBUG MESSAGE: Finished interpolation of current wavefront on resulting mesh')

            iAvgProc += 1
            if(iAvgProc >= _n_part_avg_proc):
                if(nProc > 1):
                    #sys.exit(0)
                    comMPI.Send([resStokes.arS, MPI.FLOAT], dest=0)
                    for ir in range(nStPt):
                        resStokes.arS[ir] = 0
                iAvgProc = 0

            if(nProc == 1):
                #DEBUG
                #if(i == 1):
                #    srwl_uti_save_intens_ascii(resStokes.arS, meshRes, _file_path, 1)
                #    sys.exit(0)
                #END DEBUG
                iSave += 1
                if((_file_path != None) and (iSave == _n_save_per)):
                    #Saving results from time to time in the process of calculation:
                    srwl_uti_save_intens_ascii(resStokes.arS, meshRes, _file_path, 1)
                    #sys.exit(0)
                    iSave = 0

    elif((rank == 0) and (nProc > 1)):

        #sys.exit(0)        
        nRecv = int(nPartPerProc*nProc/_n_part_avg_proc + 1e-09)
        for i in range(nRecv): #loop over messages from workers

            comMPI.Recv([workStokes.arS, MPI.FLOAT], source=MPI.ANY_SOURCE) #receive

            resStokes.avg_update_same_mesh(workStokes, i + 1)

            iSave += 1
            if(iSave == _n_save_per):
                #Saving results from time to time in the process of calculation
                srwl_uti_save_intens_ascii(resStokes.arS, meshRes, _file_path, 1)  
                iSave = 0

    if((rank == 0) or (nProc == 1)):
        #Saving final results:
        if(_file_path != None):
            srwl_uti_save_intens_ascii(resStokes.arS, meshRes, _file_path, 1)
            
        return resStokes
    else:
        return None
