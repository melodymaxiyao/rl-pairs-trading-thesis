#include <math.h>
#include <stdlib.h>
#include <string.h>
static void predict_cov(const double*P,double rho,double qeta,double qzeta,double*out){
 out[0]=P[0]+P[3]+P[1]+P[4]+qeta;out[1]=P[1]+P[4];out[2]=rho*(P[2]+P[5]);
 out[3]=P[3]+P[4];out[4]=P[4]+qzeta;out[5]=rho*P[5];
 out[6]=rho*(P[6]+P[7]);out[7]=rho*P[7];out[8]=rho*rho*P[8]+1;
}
void llt_filter(int n,int T,const double*x,double rho,double qe,double qz,double*out,double*states,int store,int steady){
 double*a=calloc((size_t)n*3,sizeof(double)),*P=calloc((size_t)n*9,sizeof(double)),*Pi=calloc((size_t)n*9,sizeof(double)),*cache=calloc((size_t)n*4,sizeof(double));
 int*seen=calloc(n,sizeof(int));unsigned char*stable=calloc(n,1);
 for(int i=0;i<n;i++){P[9*i+8]=1/(1-rho*rho);Pi[9*i]=1;Pi[9*i+4]=1;out[3*i]=out[3*i+1]=out[3*i+2]=0;}
 for(int t=0;t<T;t++)for(int i=0;i<n;i++){
  double y=x[(size_t)t*n+i],*m=a+3*i,*p=P+9*i,*pi=Pi+9*i,*cc=cache+4*i;
  if(t){m[0]+=m[1];m[2]*=rho;
   if(!(steady&&stable[i]&&isfinite(y))){double pp[9];predict_cov(p,rho,qe,qz,pp);memcpy(p,pp,sizeof(pp));
    if(seen[i]<2){double ii[9];predict_cov(pi,rho,0,0,ii);ii[8]-=1;memcpy(pi,ii,sizeof(ii));}else memset(pi,0,9*sizeof(double));stable[i]=0;}
  }
  if(isfinite(y)){
   double v=y-m[0]-m[2],Fi=pi[0]+2*pi[2]+pi[8],F=p[0]+2*p[2]+p[8];double oldp[9];memcpy(oldp,p,sizeof(oldp));
   if(Fi>1e-9){
    double ki[3],ph[3],pih[3];for(int j=0;j<3;j++){ph[j]=p[j*3]+p[j*3+2];pih[j]=pi[j*3]+pi[j*3+2];ki[j]=pih[j]/Fi;m[j]+=ki[j]*v;}
    for(int j=0;j<3;j++)for(int k=0;k<3;k++){p[j*3+k]-=ph[j]*ki[k]+ki[j]*ph[k]-ki[j]*ki[k]*F;pi[j*3+k]-=pih[j]*pih[k]/Fi;}
   }else{
    double K[3];
    if(steady&&stable[i]){F=cc[0];for(int j=0;j<3;j++)K[j]=cc[j+1];}
    else for(int j=0;j<3;j++)K[j]=(p[3*j]+p[3*j+2])/F;
    if(!(F>0)||!isfinite(F)){out[3*i]=out[3*i+1]=NAN;continue;}
    for(int j=0;j<3;j++)m[j]+=K[j]*v;
    if(seen[i]>=2){out[3*i]+=v*v/F;out[3*i+1]+=log(F);out[3*i+2]++;}
    if(!(steady&&stable[i])){
     for(int j=0;j<3;j++)for(int k=0;k<3;k++)p[j*3+k]-=K[j]*K[k]*F;
     double next[9],error=0;predict_cov(p,rho,qe,qz,next);for(int j=0;j<9;j++)error=fmax(error,fabs(next[j]-oldp[j]));
     /* Detection tolerance is1e-10; refine conservatively to1e-12 so long
        likelihood sums retain the full-recursion numerical result. */
     if(steady&&error<=1e-12){stable[i]=1;cc[0]=F;for(int j=0;j<3;j++)cc[j+1]=K[j];}
    }
   }
   seen[i]++;
  }
  if(store)for(int j=0;j<3;j++){size_t at=((size_t)i*T+t)*6;states[at+j]=seen[i]?m[j]:NAN;states[at+3+j]=pi[j*3+j]>1e-9?NAN:fmax(0,p[j*3+j]);}
 }
 free(a);free(P);free(Pi);free(cache);free(seen);free(stable);
}
