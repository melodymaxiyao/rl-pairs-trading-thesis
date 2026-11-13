#include <math.h>
#include <stdlib.h>
static int state(double x,int p,double scale){
 double z=x/scale;if(!p){z=fabs(z);return z<.5?0:z<1.5?1:z<2.5?2:3;}
 z=-p*z;return 4+(z>=1.5?0:z>=.5?1:z>=-.5?2:3);
}
/* The same8 states and reverting targets; observations now come from rolling OU. */
void observed(int n,int T,const double*s,const double*r,const unsigned char*valid,
 int kind,int bits,int mask,double enter,double exit,double cost,double*out,double*daily,int want_daily){
 for(int i=0;i<n;i++){
  int prev=0,entries=0,flat=0,duration=0,closed=0,holding=0;double gross=0,turn=0;
  for(int t=0;t<T;t++){
   int j=i*(T+1)+t,k=i*T+t,p=prev,a=0;double v=s[j];
   if(!valid[j])p=0;
   else{
    int st=state(v,prev,1.);int baseline=!prev?fabs(v)>1.25:-prev*v<.5;
    if(kind==0)a=(bits>>st)&1;
    else if(kind==1)a=baseline;
    else if(kind==2)a=!prev?fabs(v)>enter:-prev*v<exit;
    else a=(mask&(1<<st))?((bits>>st)&1):baseline;
    if(a)p=prev?0:(v>0?-1:v<0?1:0);
   }
   if(!prev&&p)entries++;
   if(prev&&!p){holding+=duration;closed++;duration=0;}
   if(p)duration++;else flat++;
   double dt=abs(p-prev),g=p*r[k];gross+=g;turn+=dt;
   if(want_daily)daily[k]=g-cost*dt;
   prev=p;
  }
  if(prev){turn+=abs(prev);holding+=duration;closed++;if(want_daily)daily[i*T+T-1]-=cost*abs(prev);}
  out[8*i]=gross-cost*turn;out[8*i+1]=gross;out[8*i+2]=turn;out[8*i+3]=entries;
  out[8*i+4]=(double)flat/T;out[8*i+5]=holding;out[8*i+6]=closed;out[8*i+7]=1.-(double)flat/T;
 }
}
