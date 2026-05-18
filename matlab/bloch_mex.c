#include "mex.h"
#include "../c/bloch.h"
#include <stdio.h>
#include <string.h>

#define DEBUG

/* CTR */
/* JGW updated to do{} while(0) form for robustness */
/* Debugging macro. Use like printf, but only prints if debugflag is true. */
#define DEBUG_printf( ... ) \
do { if (debugflag) mexPrintf(__VA_ARGS__); } while(0)
/* (The (void)0 gives an error if DEBUG_printf is missing a terminating ;. */
static bool debugflag = false;
/* End CTR. */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
/* bloch(b1,grad,dt,t1,t2,df,dxyz,dvxyz,mode,mx,my,mz,spoiled,offset) */
{
    double *b1r;	   /* Real-part of B1 field             */
    double *b1i;	   /* Imag-part of B1 field             */
    double *gx;        /* X              gradient (Hz/cm)   */
    double *gy;        /* Y              gradient (Hz/cm)   */
    double *gz;        /* Z              gradient (Hz/cm)   */
    double *tp;        /* Time steps (s)                    */
    double *ti;        /* Time intervals (s)                */
    double t1;         /* T1 time constant (s)              */
    double t2;         /* T2 time constant (s)              */
    double *df;        /* Off-resonance Frequencies (Hz)	*/
    double *dx;        /* X Positions (cm)                  */
    double *dy;        /* Y Positions (cm)                  */
    double *dz;        /* Z Positions (cm)                  */
    double *dxv;       /* X Velocities (cm/s)               */
    double *dyv;       /* Y Velocities (cm/s)               */
    double *dzv;       /* Z Velocities (cm/s)               */
    int md;            /* Mode - 0=from M0, 1=steady-state	*/
    double *mxin;      /* Input points                      */
    double *myin;
    double *mzin;
    double *spoil;  /* Transverse magnetisation = 0 when spoil == 1. */
    double tstep;	/* Time step, if single parameter */
    double *mxout;	/* Input points  */
    double *myout;  /* Input points  */
    double *mzout;  /* Input points  */
    double *mx;	    /* Output Arrays */
    double *my;     /* Output Arrays */
    double *mz;     /* Output Arrays */
    
    int gyaflag=0;        /* 1 if gy was allocated.        */
    int gzaflag=0;        /* 1 if gz was allocated.        */
    int dyaflag=0;        /* 1 if dy was allocated.        */
    int dzaflag=0;        /* 1 if dz was allocated.        */
    int spoilflag=0;      /* 1 if spoiling allocated.      */
    
    int ntime;        /* Number of time points.              */
    int ntout;        /* Number of time poitns at output.    */
    int outsize[4];	  /* Output matrix sizes                 */
    int ngrad;        /* Number of gradient dimensions       */
    int nf;           /* Number of off-resonance frequencies */
    int npos;         /* Number of positions.  Calculated from nposN and nposM, depends on them. */
    int nposM;        /* Height of passed position matrix.   */
    int nposN;        /* Width of passed position matrix.    */
    int nfnpos;       /* Number of frequencies * number of positions. */
    int count;
    int gcount=1;     /* Gradient counter                   */
    int nvel;         /* Number of velocities.  Calculated from nvelN and nvelM. */
    int nvelM;        /* Height of passed velocity matrix.  */
    int nvelN;        /* Width of passed velocity matrix.   */
    int nfnposnvel;   /* Number of frequencies * positions * velocities.                */
    int ntnfnposnvel; /* Number of output times * frequencies * positions * velocities. */
    int dyvaflag;     /* 1 if dyv was allocated.            */
    int dzvaflag;     /* 1 if dzv was allocated.            */
    int noutdim;      /* Number of output matrix dimensions */
    
    /* CTR: ERROR CHECKING. Test number of inputs. */
    
    /* Special case - allow bloch('debug',true) or bloch('debug',false) syntax. */
    if (nrhs == 2) {
        char str1[1024];
        str1[0] = '\0';
        
        if (mxGetString(prhs[0],str1,sizeof(str1)-1)==0) {
            if (strcmp(str1,"debug") == 0) {
                debugflag = mxIsLogicalScalarTrue(prhs[1]);
                mexPrintf("Setting debug flag to %s.\n", debugflag ? "true" : "false" );
                return;
            }
        }
    }

    /* Special case - allow bloch('gamma') syntax. */
    if (nrhs == 1) {
        char str1[1024];
        str1[0] = '\0';
        
        if (mxGetString(prhs[0],str1,sizeof(str1)-1)==0) {
            if (strcmp(str1,"gamma") == 0) {
                if (nlhs < 1) {
                    mexPrintf("GAMMA = %g.\n", GAMMA );
                } else {
                    double *gamma_return;
                    
                    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
                    gamma_return = mxGetPr(plhs[0]);
                    *gamma_return = GAMMA;
                }
                return;
            }
        }
    }
    
    /* Check number of inputs and outputs: */
    if (nrhs < 8) {
        mexPrintf("Hint: Type 'doc %s' for help on input parameters.\n",mexFunctionName());
        mexErrMsgIdAndTxt("bloch:BadNInput","At least 8 inputs required.");
    }
    
    /* Check all inputs are of type "double" (or char for first input): */
    for (count = 0; count < nrhs; count++) {
        if (!(
                mxIsDouble(prhs[count]) /* Any param can be double. */
                || (count == 0 && mxIsChar(prhs[count])) /* Or 1st can be char. */
                ))
        {
            mexPrintf("Hint: Type 'doc %s' for help on input parameters.\n",mexFunctionName());
            mexErrMsgIdAndTxt("bloch:BadNInput","All inputs must be of type double.");
        }
    }
    /* End CTR. */
    
    #ifdef DEBUG
        DEBUG_printf("----------------------------------------------------------\n");
        DEBUG_printf("3D-position, 1D-frequency, and 3D-velocity Bloch Simulator\n");
        DEBUG_printf("with linear gradients.                                    \n");
        DEBUG_printf("----------------------------------------------------------\n\n");
    #endif
    
    ntime = mxGetM(prhs[0]) * mxGetN(prhs[0]);	/* Number of Time, RF, and Grad points */
    
    /* ====================== RF (B1) =========================
     * :  If complex, split up.  If real, allocate an imaginary part. ==== */
    if (mxIsComplex(prhs[0])) {
        b1r = mxGetPr(prhs[0]);
        b1i = mxGetPi(prhs[0]);
        
    } else {
        b1r = mxGetPr(prhs[0]);
        b1i = (double *)malloc(ntime * sizeof(double));
        for (count=0; count < ntime; count++)
            b1i[count]=0.0;
    }
    #ifdef DEBUG
        DEBUG_printf("%d B1 points.\n",ntime);
    #endif
    if (b1r == NULL)
        mexErrMsgIdAndTxt("bloch:BadPosition","B1 is not allocated.");
        
    /* ======================= Gradients ========================= */
    
    ngrad = mxGetM(prhs[1]) * mxGetN(prhs[1]);	/* Number of Time, RF, and Grad points */
    gx = mxGetPr(prhs[1]); /* X gradient is first N points. */
    #ifdef DEBUG
        DEBUG_printf("X gradient is set.");
        DEBUG_printf(" gcount = %i.\n",gcount);
    #endif
            
    if (ngrad < ++gcount*ntime) {   /* Need to allocate Y gradient. */
        #ifdef DEBUG
            DEBUG_printf("Assuming 1-dimensional gradient.\n");
        #endif
        gy = (double *)malloc(ntime * sizeof(double));
        gyaflag=1;
        for (count=0; count<ntime; count++) { gy[count]=0.0; }
    } else {
        #ifdef DEBUG
            DEBUG_printf("Y gradient is set.");
            DEBUG_printf(" gcount = %i.\n",gcount);
        #endif
        gy = gx + (gcount-1)*ntime;	/* Assign from Nx3 input array. */
    }
    
    if (ngrad < ++gcount*ntime) {  /* Need to allocate Z gradient. */
        gz = (double *)malloc(ntime * sizeof(double));
        gzaflag=1;
        for (count=0; count<ntime; count++) { gz[count]=0.0; }
    } else {
        #ifdef DEBUG
            DEBUG_printf("Z gradient is set.");
            DEBUG_printf(" gcount = %i.\n",gcount);
        #endif
        gz = gx + (gcount-1)*ntime; /* Assign from Nx3 input array. */
    }
    
    /* Warning if Gradient length is not an integer multiple of the RF length. */
    #ifdef DEBUG
        DEBUG_printf("%d Gradient Points (total).\n",ngrad);
    #endif
    /* if ( (ngrad != ntime) && (ngrad != 2*ntime) && (ngrad != 3*ntime) ) */
    if ( (ngrad % ntime) > 0 )
        mexErrMsgIdAndTxt("bloch:BadGradientLength","Gradient length differs from B1 length.");
    if (gx == NULL)
        mexErrMsgIdAndTxt("bloch:BadGradientLength","gx is not allocated.");
    if (gy == NULL)
        mexErrMsgIdAndTxt("bloch:BadGradientLength","gy is not allocated.");
    if (gz == NULL)
        mexErrMsgIdAndTxt("bloch:BadGradientLength","gz is not allocated.");
            
    /* === Time points ===== */
        
    /*	THREE Cases:
        1) Single value given -> this is the interval length for all.
        2) List of intervals given.
        3) Monotonically INCREASING list of end times given.
	For all cases, the goal is for tp to have the intervals. */
        
    ti = NULL;
    tp = mxGetPr(prhs[2]);
    if (tp == NULL)
        mexErrMsgIdAndTxt("bloch:BadPosition","tp is not allocated.");
    
    if (mxGetM(prhs[2]) * mxGetN(prhs[2]) == 1)	{ /* === Case 1 === */
        tp = (double *)malloc(ntime * sizeof(double));
        tstep = *(mxGetPr(prhs[2]));
        for (count =0; count < ntime; count++)
            tp[count]=tstep;
        
    } else if (mxGetM(prhs[2]) * mxGetN(prhs[2]) != ntime)
        mexErrMsgIdAndTxt("bloch:BadB1Length","Time-point length differs from B1 length.");
    
    else {
        tp = mxGetPr(prhs[2]);
        ti = (double *)malloc(ntime * sizeof(double));
        if (( times2intervals( tp, ti, ntime ))) {
            DEBUG_printf("Times are monotonically increasing.\n");
            tp = ti;
        }
    }
    
    /* === Relaxation Times ===== */
    if (mxGetPr(prhs[3]) == NULL)
        mexErrMsgIdAndTxt("bloch:BadPosition","t1 is not allocated.");
    if (mxGetPr(prhs[4]) == NULL)
        mexErrMsgIdAndTxt("bloch:BadPosition","t2 is not allocated.");
    
    t1 = *mxGetPr(prhs[3]);
    t2 = *mxGetPr(prhs[4]);
    
    #ifdef DEBUG
        DEBUG_printf("t1 = %d \n",t1);
        DEBUG_printf("t2 = %d \n",t2);
    #endif
        
    /* === Frequency Points ===== */
    df = mxGetPr(prhs[5]);
    nf = mxGetM(prhs[5]) * mxGetN(prhs[5]);
    
    #ifdef DEBUG
        DEBUG_printf("%d Frequency points.\n",nf);
    #endif
    if (df == NULL)
        mexErrMsgIdAndTxt("bloch:BadPosition","df is not allocated.");

    /* === Position Points ===== */
    nposM = mxGetM(prhs[6]);
    nposN = mxGetN(prhs[6]);
    
    #ifdef DEBUG
        DEBUG_printf("Position vector is %d x %d.\n",nposM,nposN);
    #endif
    
    if (nposN==3) { /* Assume 3 position dimensions given */
        npos = nposM;
        #ifdef DEBUG
            DEBUG_printf("Assuming %d 3-Dimensional Positions.\n",npos);
        #endif
        dx = mxGetPr(prhs[6]);
        dy = dx + npos;
        dz = dy + npos;

    } else if (nposN==2) { /* Assume only 2 position dimensions given */
        npos = nposM;
        #ifdef DEBUG
            DEBUG_printf("Assuming %d 2-Dimensional Positions.\n",npos);
        #endif
        dx = mxGetPr(prhs[6]);
        dy = dx + npos;
        dz = (double *)malloc(npos * sizeof(double));
        dzaflag=1;
        for (count=0; count < npos; count++)
            dz[count]=0.0;
    
    } else { /* Either 1xN, Nx1 or something random.  In all these
     * cases we assume that 1 position is given, because it
     * is too much work to try to figure out anything else! */
        npos = nposM * nposN;
        #ifdef DEBUG
            DEBUG_printf("Assuming %d 1-Dimensional Positions.\n",npos);
        #endif
        dx = mxGetPr(prhs[6]);
        dy = (double *)malloc(npos * sizeof(double));
        dz = (double *)malloc(npos * sizeof(double));
        dyaflag=1;
        dzaflag=1;
        for (count=0; count < npos; count++) {
            dy[count]=0.0;
            dz[count]=0.0;
        }
        #ifdef DEBUG
            if ((nposM !=1) && (nposN!=1)) {
                DEBUG_printf("Position vector should be 1xN, Nx1, Nx2 or Nx3.\n");
                DEBUG_printf(" -> Assuming 1 position dimension is given.\n");
            }
        #endif
    }
    if (dx == NULL)
        mexErrMsgIdAndTxt("bloch:BadPosition","dx is not allocated.");
    if (dy == NULL)
        mexErrMsgIdAndTxt("bloch:BadPosition","dy is not allocated.");
    if (dz == NULL)
        mexErrMsgIdAndTxt("bloch:BadPosition","dz is not allocated.");

    nfnpos = nf*npos;	/* Just used to speed things up below. 	*/ 

    /* === Velocity Points ===== */
    nvelM = mxGetM(prhs[7]);
    nvelN = mxGetN(prhs[7]);
    #ifdef DEBUG
        DEBUG_printf("Velocity vector is %d x %d.\n",nvelM,nvelN);
	#endif

    if (nvelN==3) {         /* Assume 3 velocity dimensions given */
        nvel = nvelM;
        #ifdef DEBUG
            DEBUG_printf("Assuming %d 3-Dimensional Velocities.\n",nvel);
        #endif
        dxv = mxGetPr(prhs[7]);
        dyv = dxv + nvel;
        dzv = dyv + nvel;
    } else if (nvelN==2) {	/* Assume only 2 velocity dimensions given */
        nvel = nvelM;
        #ifdef DEBUG
            DEBUG_printf("Assuming %d 2-Dimensional Velocities.\n",nvel);
        #endif
        dxv = mxGetPr(prhs[7]);
        dyv = dxv + nvel;
        dzv = (double *)malloc(nvel * sizeof(double));
        dzvaflag=1;
        for (count=0; count < nvel; count++)
            dzv[count]=0.0;
    } else {               	/* Assume only 1 velocity dimension given */
        nvel = nvelM * nvelN;
        #ifdef DEBUG
            DEBUG_printf("Assuming %d 1-Dimensional Velocities.\n",npos);
        #endif
        dxv = mxGetPr(prhs[7]);
        dyv = (double *)malloc(nvel * sizeof(double));
        dzv = (double *)malloc(nvel * sizeof(double));
        dyvaflag=1;
        dzvaflag=1;
        for (count=0; count < nvel; count++) {
            dyv[count]=0.0;
            dzv[count]=0.0;
        }
        #ifdef DEBUG
            if ((nvelM !=1) && (nvelN!=1)) {
                DEBUG_printf("Velocity vector should be 1xN, Nx1, Nx2 or Nx3.\n");
                DEBUG_printf(" -> Assuming 1 velocity dimension is given.\n");
            }
        #endif
    }
    if (dxv == NULL)
        mexErrMsgIdAndTxt("bloch:BadVelocity","dxv is not allocated.");
    if (dyv == NULL)
        mexErrMsgIdAndTxt("bloch:BadVelocity","dyv is not allocated.");
    if (dzv == NULL)
        mexErrMsgIdAndTxt("bloch:BadVelocity","dzv is not allocated.");

    nfnposnvel = nfnpos * nvel;
  
    /* ===== Mode, defaults to 0 (simulate single endpoint). ==== */
    if (nrhs > 8)
        md = (int)(*mxGetPr(prhs[8]));
    else
        md = 0;

    if (md == 1)
        ntout = ntime;		/* Include time points.	*/
    else
        ntout = 1;

    ntnfnposnvel = ntout*nfnposnvel;

    #ifdef DEBUG
        if (md == 0)
            DEBUG_printf("Simulation to Endpoint.\n");
        else
            DEBUG_printf("Simulation over Time.\n");
    #endif

    /* ===== Allocate Output Magnetization vectors arrays.	*/
    plhs[0] = mxCreateDoubleMatrix(ntnfnposnvel,1,mxREAL);	/* Mx, output. */
    plhs[1] = mxCreateDoubleMatrix(ntnfnposnvel,1,mxREAL);	/* My, output. */
    plhs[2] = mxCreateDoubleMatrix(ntnfnposnvel,1,mxREAL);	/* Mz, output. */
    
    mx = mxGetPr(plhs[0]);
    my = mxGetPr(plhs[1]);
    mz = mxGetPr(plhs[2]);
    
    mxout = mx;
    myout = my;
    mzout = mz;

    /* ===== If Initial Magnetization is given... */
    if ( (nrhs > 11) &&
            (mxGetM(prhs[9])  * mxGetN(prhs[9])  == nfnposnvel) &&
            (mxGetM(prhs[10]) * mxGetN(prhs[10]) == nfnposnvel) &&
            (mxGetM(prhs[11]) * mxGetN(prhs[11]) == nfnposnvel)  ) {
        /* Set output magnetization to that passed.
         * If multiple time points, then just the first is set. */

        #ifdef DEBUG
            DEBUG_printf("Using Specified Initial Magnetization.\n");
        #endif

        mxin = mxGetPr(prhs[9]);
        myin = mxGetPr(prhs[10]);
        mzin = mxGetPr(prhs[11]);
        for (count=0; count<nfnposnvel; count++) {
            mxout[count*ntout] = mxin[count];
            myout[count*ntout] = myin[count];
            mzout[count*ntout] = mzin[count];
        }
    } else {
        #ifdef DEBUG
            if (nrhs > 11) { /* Magnetization given, but wrong size! */
                mexErrMsgIdAndTxt("bloch:BadMagnetization","Initial magnetization passed, but not Npositions x Nfreq x Nvelocities.");
            }
            DEBUG_printf(" --> Using [0; 0; 1] for initial magnetization.\n");
        #endif
        for (count=0; count<nfnposnvel; count++) {
            mxout[count*ntout] = 0;	/* Set magnetization to Equilibrium */
            myout[count*ntout] = 0;
            mzout[count*ntout] = 1;
        }
    }

    /* WTC: Spoiler vector */
    if ((nrhs > 12) && (mxGetM(prhs[12]) * mxGetN(prhs[12]) == ntime)) {
        spoil = mxGetPr(prhs[12]);
        DEBUG_printf("Spoiler length same as B1 length.\n");
    } else if ((nrhs > 12) && (mxGetM(prhs[12]) * mxGetN(prhs[12]) != ntime)) {
        mexErrMsgIdAndTxt("bloch:BadSpoiler","Spoiler length differs from B1 length.");
    } else {
        #ifdef DEBUG
            DEBUG_printf("Assigning spoiler vector.\n");
        #endif
        spoil = (double *)malloc(ntime * sizeof(double));
        spoilflag = 1;
        for (count=0; count < ntime; count++)
            spoil[count]=0.0;
    }


    /* ======= Do The Simulation! ====== */
    #ifdef DEBUG
        DEBUG_printf("Calling blochsimfz() function.\n");
    #endif

    blochsimfz(b1r,b1i,gx,gy,gz,tp,ntime,t1,t2,df,nf,dx,dy,dz,npos,dxv,dyv,dzv,nvel,mx,my,mz,md,spoil);

    
    /* ======= Reshape Output Matrices ====== */
    noutdim = (int)(ntout>1) + (int)(npos>1) + (int)(nf>1) + (int)(nvel>1);

    if (noutdim == 4) {
        outsize[0]=ntout;
        outsize[1]=npos;
        outsize[2]=nf;
        outsize[3]=nvel;
        mxSetDimensions(plhs[0],outsize,4);  /* Set to 4D array. */
        mxSetDimensions(plhs[1],outsize,4);  /* Set to 4D array. */
        mxSetDimensions(plhs[2],outsize,4);  /* Set to 4D array. */
    } else if (noutdim == 3) { /* Try 3 dimensions */
        if ((ntout > 1) && (npos > 1) && (nf > 1) && (nvel == 1)) {
            outsize[0]=ntout;
            outsize[1]=npos;
            outsize[2]=nf;
        } else if ((ntout > 1) && (npos > 1) && (nf == 1) && (nvel > 1)) {
            outsize[0]=ntout;
            outsize[1]=npos;
            outsize[2]=nvel;
        } else if ((ntout > 1) && (npos == 1) && (nf > 1) && (nvel > 1)) {
            outsize[0]=ntout;
            outsize[1]=nf;
            outsize[2]=nvel;
        } else if ((ntout == 1) && (npos > 1) && (nf > 1) && (nvel > 1)) {
            outsize[0]=npos;
            outsize[1]=nf;
            outsize[2]=nvel;
        }
        mxSetDimensions(plhs[0],outsize,3);  /* Set to 3D array. */
        mxSetDimensions(plhs[1],outsize,3);  /* Set to 3D array. */
        mxSetDimensions(plhs[2],outsize,3);  /* Set to 3D array. */
    } else { /* Only 2 or 1 dimensions */
        if (ntout > 1) {
            outsize[0]=ntout;
            outsize[1]=npos*nf*nvel;
        } else if (npos > 1) {
            outsize[0]=npos;
            outsize[1]=nf*nvel;
        } else if (nf > 1) {
            outsize[0]=nf;
            outsize[1]=nvel;
        } else {
            outsize[0]=nvel;
            outsize[1]=1;
        }
        mxSetDimensions(plhs[0],outsize,2);  /* Set to 2D array. */
        mxSetDimensions(plhs[1],outsize,2);  /* Set to 2D array. */
        mxSetDimensions(plhs[2],outsize,2);  /* Set to 2D array. */
    }

    /* ====== Free up allocated memory, if necessary. ===== */
    if (!mxIsComplex(prhs[0])) free(b1i);
    if (mxGetM(prhs[2]) * mxGetN(prhs[2]) == 1) free(tp);
    if (ti != NULL) free(ti);
    if (dyaflag==1) free(dy);
    if (dzaflag==1) free(dz);
    if (gyaflag==1) free(gy);
    if (gzaflag==1) free(gz);
    if (spoilflag==1) free(spoil);
    if (dyvaflag==1) free(dyv);
    if (dzvaflag==1) free(dzv);
}
