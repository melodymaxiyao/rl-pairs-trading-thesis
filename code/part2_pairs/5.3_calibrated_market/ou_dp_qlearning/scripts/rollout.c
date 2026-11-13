/* Continuous-path evaluator. No training or random-number generation here. */
#include <math.h>
#include <stdint.h>
static int state(double x,int p,double scale){
 double z=x/scale;if(!p){z=fabs(z);return z<.5?0:z<1.5?1:z<2.5?2:3;}
 z=-p*z;return 4+(z>=1.5?0:z>=.5?1:z>=-.5?2:3);
}
void rollout(int n,int T,const double*x,int kind,int bits,double scale,double stop,
 double cost,int N,double xmin,double spacing,const int*grid,double*out){
 for(int i=0;i<n;i++){
  int prev=0,entries=0,flat=0;double gross=0,turnover=0;
  for(int t=0;t<T;t++){
   double v=x[i*(T+1)+t];int p=prev,a=0;
   if(fabs(v)>=stop)p=0;
   else if(kind==3){
    int k=(int)floor((v-xmin)/spacing+.5);if(k<0)k=0;if(k>=N)k=N-1;
    p=grid[(t*3+prev+1)*N+k];
    /* A nearest-grid entry still respects the continuous reverting side. */
    if(!prev&&p)p=v>0?-1:v<0?1:0;
   }else{
    if(kind==0)a=(bits>>state(v,prev,scale))&1;
    else if(kind==1)a=!prev?fabs(v/scale)>1.25:-prev*v/scale<.5;
    else a=!prev?fabs(v/scale)>=2:-prev*v/scale<=0;
    if(a)p=prev?0:(v>0?-1:v<0?1:0);
   }
   if(!prev&&p)entries++;if(!p)flat++;
   turnover+=abs(p-prev);gross+=p*(x[i*(T+1)+t+1]-v);prev=p;
  }
  turnover+=abs(prev);
  out[5*i]=gross-cost*turnover/2;out[5*i+1]=gross;out[5*i+2]=turnover;
  out[5*i+3]=entries;out[5*i+4]=(double)flat/T;
 }
}
