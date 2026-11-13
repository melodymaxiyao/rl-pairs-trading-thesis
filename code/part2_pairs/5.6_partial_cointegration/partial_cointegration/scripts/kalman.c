#include <math.h>
#include <stdlib.h>
#include <stddef.h>
/* Time-major input: contiguous recursion across every series in the group.
   Exact diffuse M, stationary R; first finite diffuse observation excluded. */
void pci_filter(int n,int T,const double*x,double rho,double q,double*out,double*states,int store,int steady,int burn){
 double*m=calloc((size_t)n*12,sizeof(double));
 double*r=m+n,*a=r+n,*b=a+n,*c=b+n,*ss=c+n,*ld=ss+n,*km=ld+n,*kr=km+n,*ff=kr+n,*nn=ff+n,*seen=nn+n;
 unsigned char*stable=calloc(n,1);
 for(int i=0;i<n;i++)c[i]=1/(1-rho*rho);
 for(int t=0;t<T;t++)for(int i=0;i<n;i++){
  double y=x[(size_t)t*n+i];
  if(seen[i]){
   r[i]*=rho;
   if(!(steady&&stable[i]&&isfinite(y))){a[i]+=q;b[i]*=rho;c[i]=rho*rho*c[i]+1;stable[i]=0;}
  }
  if(isfinite(y)){
   if(!seen[i]){m[i]=y;r[i]=0;c[i]=1/(1-rho*rho);a[i]=c[i];b[i]=-c[i];seen[i]=1;}
   else{
    double F,km1,kr1,old_a=a[i],old_b=b[i],old_c=c[i];
    if(steady&&stable[i]){F=ff[i];km1=km[i];kr1=kr[i];}
    else{F=a[i]+2*b[i]+c[i];km1=(a[i]+b[i])/F;kr1=(b[i]+c[i])/F;}
    double v=y-m[i]-r[i];
    if(!(F>0)||!isfinite(F)){ss[i]=NAN;ld[i]=NAN;continue;}
    m[i]+=km1*v;r[i]+=kr1*v;
    if(seen[i]>=burn){ss[i]+=v*v/F;ld[i]+=log(F);nn[i]++;}seen[i]++;
    if(!(steady&&stable[i])){
     a[i]-=km1*km1*F;b[i]-=km1*kr1*F;c[i]-=kr1*kr1*F;
     double error=fmax(fabs(a[i]+q-old_a),fmax(fabs(rho*b[i]-old_b),fabs(rho*rho*c[i]+1-old_c)));
     if(steady&&error<=1e-10){stable[i]=1;ff[i]=F;km[i]=km1;kr[i]=kr1;}
    }
   }
  }
  if(store){states[((size_t)i*T+t)*2]=seen[i]?m[i]:NAN;states[((size_t)i*T+t)*2+1]=seen[i]?r[i]:NAN;}
 }
 for(int i=0;i<n;i++){out[(size_t)i*3]=ss[i];out[(size_t)i*3+1]=ld[i];out[(size_t)i*3+2]=nn[i];}
 free(stable);free(m);
}
