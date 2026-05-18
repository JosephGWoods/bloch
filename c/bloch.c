#include "bloch.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>


/* Multiply 3x3 matrix by 3x1 vector. */
void multmatvec(double *mat, double *vec, double *matvec)
{
    matvec[0] = mat[0]*vec[0] + mat[3]*vec[1] + mat[6]*vec[2];
    matvec[1] = mat[1]*vec[0] + mat[4]*vec[1] + mat[7]*vec[2];
    matvec[2] = mat[2]*vec[0] + mat[5]*vec[1] + mat[8]*vec[2];
}


/* Add two 3x1 Vectors */
void addvecs(double *vec1, double *vec2, double *vecsum)
{
    int count;
    for (count=0; count<3; count++)
        vecsum[count] = vec1[count] + vec2[count];
}


/* ======== Adjoint of a 3x3 matrix ========= */
void adjmat(double *mat, double *adj)
{
    adj[0] = (mat[4]*mat[8]-mat[7]*mat[5]);
    adj[1] =-(mat[1]*mat[8]-mat[7]*mat[2]);
    adj[2] = (mat[1]*mat[5]-mat[4]*mat[2]);
    adj[3] =-(mat[3]*mat[8]-mat[6]*mat[5]);
    adj[4] = (mat[0]*mat[8]-mat[6]*mat[2]);
    adj[5] =-(mat[0]*mat[5]-mat[3]*mat[2]);
    adj[6] = (mat[3]*mat[7]-mat[6]*mat[4]);
    adj[7] =-(mat[0]*mat[7]-mat[6]*mat[1]);
    adj[8] = (mat[0]*mat[4]-mat[3]*mat[1]);
}


/* ====== Set a 3x3 matrix to all zeros	======= */
void zeromat(double *mat)
{
    int count;
    for (count=0; count<9; count++)
        mat[count] = 0;
}


/* ======== Return 3x3 Identity Matrix  ========= */
void eyemat(double *mat)
{
    zeromat(mat);
    mat[0]=1;
    mat[4]=1;
    mat[8]=1;
}


/* ======== Determinant of a 3x3 matrix ======== */
double detmat(double *mat)
{
    double det;
    
    det = mat[0]*mat[4]*mat[8];
    det+= mat[3]*mat[7]*mat[2];
    det+= mat[6]*mat[1]*mat[5];
    det-= mat[0]*mat[7]*mat[5];
    det-= mat[3]*mat[1]*mat[8];
    det-= mat[6]*mat[4]*mat[2];
    
    return det;
}


/* ======== multiply a matrix by a scalar ========= */
void scalemat(double *mat, double scalar)
{
    int count;
    for (count=0; count<9; count++)
        mat[count] *= scalar;
}


/* ======== Inverse of a 3x3 matrix ========= */
/*	DO NOT MAKE THE OUTPUT THE SAME AS ONE OF THE INPUTS!! */
void invmat(double *mat, double *imat)
{
    int count;
    double det;
    
    det = detmat(mat);	/* Determinant */
    adjmat(mat, imat);	/* Adjoint */
    
    for (count=0; count<9; count++)
        imat[count] = imat[count] / det;
}


/* ====== Add two 3x3 matrices.	====== */
void addmats(double *mat1, double *mat2, double *matsum)
{
    int count;
    for (count=0; count<9; count++)
        matsum[count] = mat1[count] + mat2[count];
}


/* ======= Multiply two 3x3 matrices. ====== */
/*	DO NOT MAKE THE OUTPUT THE SAME AS ONE OF THE INPUTS!! */
void multmats(double *mat1, double *mat2, double *matproduct)
{
    matproduct[0] = mat1[0]*mat2[0] + mat1[3]*mat2[1] + mat1[6]*mat2[2];
    matproduct[1] = mat1[1]*mat2[0] + mat1[4]*mat2[1] + mat1[7]*mat2[2];
    matproduct[2] = mat1[2]*mat2[0] + mat1[5]*mat2[1] + mat1[8]*mat2[2];
    matproduct[3] = mat1[0]*mat2[3] + mat1[3]*mat2[4] + mat1[6]*mat2[5];
    matproduct[4] = mat1[1]*mat2[3] + mat1[4]*mat2[4] + mat1[7]*mat2[5];
    matproduct[5] = mat1[2]*mat2[3] + mat1[5]*mat2[4] + mat1[8]*mat2[5];
    matproduct[6] = mat1[0]*mat2[6] + mat1[3]*mat2[7] + mat1[6]*mat2[8];
    matproduct[7] = mat1[1]*mat2[6] + mat1[4]*mat2[7] + mat1[7]*mat2[8];
    matproduct[8] = mat1[2]*mat2[6] + mat1[5]*mat2[7] + mat1[8]*mat2[8];
}


/* Find the rotation matrix that rotates |n| radians about the vector
 * given by nx,ny,nz                                                  */
void calcrotmat(double nx, double ny, double nz, double *rmat)
{
    double ar, ai, br, bi, hp, cp, sp;
    double arar, aiai, arai2, brbr, bibi, brbi2, arbi2, aibr2, arbr2, aibi2;
    double phi;
    
    phi = sqrt(nx*nx+ny*ny+nz*nz);
    
    if (phi == 0.0) {
        rmat[0] = 1;
        rmat[1] = 0;
        rmat[2] = 0;
        rmat[3] = 0;
        rmat[4] = 1;
        rmat[5] = 0;
        rmat[6] = 0;
        rmat[7] = 0;
        rmat[8] = 1;
    } else {
        /* First define Cayley-Klein parameters */
        hp = phi/2;
        cp = cos(hp);
        sp = sin(hp)/phi;	/* /phi because n is unit length in defs. */
        ar = cp;
        ai = -nz*sp;
        br = ny*sp;
        bi = -nx*sp;
        
        /* Make auxiliary variables to speed this up */
        arar = ar*ar;
        aiai = ai*ai;
        arai2 = 2*ar*ai;
        brbr = br*br;
        bibi = bi*bi;
        brbi2 = 2*br*bi;
        arbi2 = 2*ar*bi;
        aibr2 = 2*ai*br;
        arbr2 = 2*ar*br;
        aibi2 = 2*ai*bi;
        
        /* Make rotation matrix. */
        rmat[0] = arar-aiai-brbr+bibi;
        rmat[1] = -arai2-brbi2;
        rmat[2] = -arbr2+aibi2;
        rmat[3] =  arai2-brbi2;
        rmat[4] = arar-aiai+brbr-bibi;
        rmat[5] = -aibr2-arbi2;
        rmat[6] =  arbr2+aibi2;
        rmat[7] =  arbi2-aibr2;
        rmat[8] = arar+aiai-brbr-bibi;
    }
}


/*	Set a 3x1 vector to all zeros	*/
void zerovec(double *vec)
{
    int count;
    for (count=0; count<3; count++)
        vec[count] = 0;
}


int times2intervals( double *endtimes, double *intervals, long n)
/* ------------------------------------------------------------
 * Function takes the given endtimes of intervals, and
 * returns the interval lengths in an array, assuming that
 * the first interval starts at 0.
 *
 * If the intervals are all greater than 0, then this
 * returns 1, otherwise it returns 0.
 * ------------------------------------------------------------ */
{
    int count, allpos;
    double lasttime;
    
    allpos=1;
    lasttime = 0.0;
    
    for (count=0; count<n; count++) {
        intervals[count] = endtimes[count]-lasttime;
        lasttime = endtimes[count];
        if (intervals[count] <= 0)
            allpos =0;
    }
    
    return (allpos);
}


void blochsim(double *b1real, double *b1imag, double *xgrad, double *ygrad, double *zgrad,
        double *tsteps, int ntime, double *e1, double *e2,
        double df, double dx, double dy, double dz, double dxv, double dyv, double dzv,
        double *mx, double *my, double *mz, int mode, double *spoil)
/* Go through time for one df and one dx,dy,dz and one dxv,dyv,dzv. */
{
    int count;
    int tcount;
    
    double rotmat[9];
    double amat[9], bvec[3];	/* A and B propagation matrix and vector.   */
    double arot[9], brot[3];	/* A and B after rotation step.             */
    double decmat[9];           /* Decay matrix for each time step.         */
    double decvec[3];           /* Recovery vector for each time step.      */
    double rotx,roty,rotz;		/* Rotation axis coordinates.               */
    double mstart[3];
    double mfinish[3];
    double imat[9], mvec[3];
    double mcurr0[3];           /* Current magnetization before rotation.   */
    double mcurr1[3];           /* Current magnetization before decay.      */
    
    eyemat(amat);               /* A is the identity matrix.                */
    eyemat(imat);               /* I is the identity matrix.                */
    
    zerovec(bvec);
    zerovec(decvec);
    zeromat(decmat);
    
    /* Linear gradient position terms */
    double gammadx = GAMMA*dx;   /* Convert to units•rad/Hz   */
    double gammady = GAMMA*dy;   /* Convert to units•rad/Hz   */
    double gammadz = GAMMA*dz;   /* Convert to units•rad/Hz   */
    
    mcurr0[0] = *mx; /* Set starting x magnetization */
    mcurr0[1] = *my; /* Set starting y magnetization */
    mcurr0[2] = *mz; /* Set starting z magnetization */
    
    for (tcount = 0; tcount < ntime; tcount++) {
        
        /* Spoiling */
        if (spoil[tcount] == 1) {
            mcurr0[0] = 0.0; /* Set starting x magnetization to 0 */
            mcurr0[1] = 0.0; /* Set starting y magnetization to 0 */
        }
        
        /*	Rotation */
        /* N.B. The SENSE of ROTATION was changed in code on the B Hargreaves'
         * website. Use NEW (2013) convention here, but keep z-rotation
         * following the convention in M. Levitt. "Spin Dynamics" for the
         * (observable) -1 coherence order. */
        rotz = (xgrad[tcount]*gammadx + ygrad[tcount]*gammady + zgrad[tcount]*gammadz + df*TWOPI ) * tsteps[tcount]; /* BH code is -(xgrad[tcount] ...) */
        rotx = (+ b1real[tcount] * GAMMA * tsteps[tcount] ); /* BH code is (- b1real[tcount] ...) */
        roty = (+ b1imag[tcount] * GAMMA * tsteps[tcount] );
        /* End of change. */
        
        calcrotmat(rotx, roty, rotz, rotmat);
        multmatvec(rotmat,mcurr0,mcurr1);
        
        /* Decay */
        decvec[2]= 1 - e1[tcount];
        decmat[0]= e2[tcount];
        decmat[4]= e2[tcount];
        decmat[8]= e1[tcount];
        
        multmatvec(decmat,mcurr1,mcurr0);
        addvecs(mcurr0,decvec,mcurr0);
        
        /* Sample output at times. */
        if (mode == 1) {
            *mx = mcurr0[0];
            *my = mcurr0[1];
            *mz = mcurr0[2];
            mx++;
            my++;
            mz++;
        }
        
        /* Update position based on velocity. */
        if ((dxv!=0.0) || (dyv!=0.0) || (dzv!=0.0)) {
            if (dxv!=0.0) {
                dx += tsteps[tcount] * dxv;
                gammadx = GAMMA*dx;
            }
            if (dyv!=0.0) {
                dy += tsteps[tcount] * dyv;
                gammady = GAMMA*dy;
            }
            if (dzv!=0.0) {
                dz += tsteps[tcount] * dzv;
                gammadz = GAMMA*dz;
            }
        }

    } /* End of time loop */
    
    /* If only recording the endpoint. */
    if (mode == 0)
    {
        *mx = mcurr0[0];
        *my = mcurr0[1];
        *mz = mcurr0[2];
    }
}


void blochsimfz(double *b1real, double *b1imag, double *xgrad, double *ygrad, double *zgrad,
        double *tsteps, int ntime, double t1, double t2, double *dfreq, int nfreq,
        double *dxpos, double *dypos, double *dzpos, int npos,
        double *dxvel, double *dyvel, double *dzvel, int nvel,
        double *mx, double *my, double *mz, int mode, double *spoil)
{
    int count, pcount, fcount, vcount, totpoints, ntout;
    int totcount = 0;
    
    if (mode == 1)
        ntout = ntime;
    else
        ntout = 1;
    
    /* First calculate the E1 and E2 values at each time step. */
    double *e1 = (double *) malloc(ntime * sizeof(double));
    double *e2 = (double *) malloc(ntime * sizeof(double));
    for (count=0; count<ntime; count++) {
        e1[count] = exp( -tsteps[count] / t1);
        e2[count] = exp( -tsteps[count] / t2);
    }
    
    totpoints = npos*nfreq*nvel;
    
    for (vcount=0; vcount<nvel; vcount++) {
        
        for (fcount=0; fcount<nfreq; fcount++) {
            
            for (pcount=0; pcount<npos; pcount++) {
                
                blochsim(b1real, b1imag, xgrad, ygrad, zgrad,
                            tsteps, ntime, e1, e2, dfreq[fcount], dxpos[pcount], dypos[pcount], dzpos[pcount],
                            dxvel[vcount], dyvel[vcount], dzvel[vcount], mx, my, mz, mode, spoil);
                
                mx += ntout;
                my += ntout;
                mz += ntout;
                
                totcount++;
                if ((totpoints > 40000) && ( ((10*totcount)/totpoints) > (10*(totcount-1)/totpoints) ))
                { printf("%d%% Complete.\n",(100*totcount/totpoints)); }
            }

        }

    }
    free(e1);
    free(e2);
}