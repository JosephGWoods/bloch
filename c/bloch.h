#ifndef BLOCH_H
#define BLOCH_H

#ifdef __cplusplus
extern "C" {
#endif

#define TWOPI   6.28318530717959
#define GAMMA   TWOPI

/* ===== Linear-algebra helpers ===== */
void   multmatvec(double *mat, double *vec, double *matvec);
void   addvecs(double *vec1, double *vec2, double *vecsum);
void   adjmat(double *mat, double *adj);
void   zeromat(double *mat);
void   eyemat(double *mat);
double detmat(double *mat);
void   scalemat(double *mat, double scalar);
void   invmat(double *mat, double *imat);
void   addmats(double *mat1, double *mat2, double *matsum);
void   multmats(double *mat1, double *mat2, double *matproduct);
void   calcrotmat(double nx, double ny, double nz, double *rmat);
void   zerovec(double *vec);

/* ===== Utilities ===== */
int    times2intervals(double *endtimes, double *intervals, long n);

/* ===== Simulation functions ===== */
void   blochsim(double *b1real, double *b1imag,
                double *xgrad, double *ygrad, double *zgrad,
                double *tsteps, int ntime, double *e1, double *e2,
                double df, double dx, double dy, double dz,
                double dxv, double dyv, double dzv,
                double *mx, double *my, double *mz, int mode, double *spoil);

void   blochsimfz(double *b1real, double *b1imag,
                  double *xgrad, double *ygrad, double *zgrad,
                  double *tsteps, int ntime, double t1, double t2,
                  double *dfreq, int nfreq,
                  double *dxpos, double *dypos, double *dzpos, int npos,
                  double *dxvel, double *dyvel, double *dzvel, int nvel,
                  double *mx, double *my, double *mz, int mode, double *spoil);

#ifdef __cplusplus
}
#endif

#endif /* BLOCH_H */
