/* POW trainer: double estimation only; state/reward/RNG imported unchanged. */
#include "frozen_ou.c"

double double_target(const double*A,const double*B,int mask){return B[mask==1?0:mask==2?1:(A[1]>A[0])];}

void double_train(int ne,int T,const double*d,const double*r,uint64_t seed,
 uint64_t coin,double cost,int epochs,int early,double*tables,int64_t*visits,
 double*diag,double*curves,int64_t*result){
 memset(tables,0,32*sizeof(double));memset(visits,0,32*sizeof(int64_t));memset(result,0,6*sizeof(int64_t));
 double mean[16]={0},W[32]={0},V[32]={0},tm[32]={0},t2[32]={0};
 int*order=malloc(ne*sizeof(int));int last[8]={0},have=0,stable=0;
 for(int i=0;i<ne;i++)order[i]=i;
 for(int epoch=1;epoch<=epochs;epoch++){
  for(int i=ne-1;i>0;i--){int j=bounded(&seed,i+1),tmp=order[i];order[i]=order[j];order[j]=tmp;}
  double eps=epoch>=240?.05:1.-.95*(epoch-1)/239.,total=0;
  for(int j=0;j<ne;j++){
   int begin=order[j]*T,end=begin+T;Env e={0,0,0,0};
   for(int t=begin;t<end;t++){
    int terminal=t==end-1,reason,bm,targets[2];
    ta_targets(&e,d[t],terminal,0,INFINITY,0,0,1,&reason,&bm,targets);
    int s=ta_state(d[t],e.prev,0,0,0),a;
    if(terminal)a=e.prev?1:0;
    else{double u=uniform(&seed);int count=0,list[2];for(int k=0;k<2;k++)if(bm&(1<<k))list[count++]=k;int exploratory=list[bounded(&seed,count)];a=u<eps?exploratory:ta_greedy(mean,s,bm);}
    int p=targets[a],which=bounded(&coin,2),offset=16*which,k=2*s+a,idx=offset+k;
    double*A=tables+offset,*B=tables+16*(1-which);double reward=reward_bps(p,e.prev,r[t],2*cost),bootstrap=0;
    Env next=e;advance(&next,p);
    if(!terminal){int nr,nm,nt[2];ta_targets(&next,d[t+1],t==end-2,0,INFINITY,0,0,1,&nr,&nm,nt);int sn=ta_state(d[t+1],p,0,0,0);bootstrap=double_target(A+2*sn,B+2*sn,nm);}
    double step=1./pow((double)visits[idx]+1.,.7),target=reward+bootstrap;
    W[idx]=(1-step)*W[idx]+step;V[idx]=(1-step)*(1-step)*V[idx]+step*step;
    double diff=target-tm[idx];tm[idx]+=diff/((double)visits[idx]+1);t2[idx]+=diff*(target-tm[idx]);
    A[k]+=step*(target-A[k]);visits[idx]++;mean[k]=(tables[k]+tables[16+k])/2.;total+=reward;advance(&e,p);result[4]++;
   }
   result[3]++;
  }
  curves[3*(epoch-1)]=total/ne;curves[3*(epoch-1)+1]=NAN;curves[3*(epoch-1)+2]=eps;result[0]=epoch;
  if(epoch%10==0){int equal=have;for(int s=0;s<8;s++){int a=ta_greedy(mean,s,3);if(have&&a!=last[s])equal=0;last[s]=a;}stable=equal?stable+10:0;have=1;
   if(stable>=50&&epoch>=40){double now=0,before=0;for(int k=epoch-20;k<epoch;k++)now+=curves[3*k]/20.;for(int k=epoch-40;k<epoch-20;k++)before+=curves[3*k]/20.;if(fabs(now-before)/fmax(fabs(before),1e-12)<.01){result[1]=1;result[2]=epoch;if(early)break;}}
  }
 }
 for(int k=0;k<32;k++){double*row=diag+10*k;row[0]=W[k];row[1]=V[k];row[2]=V[k]>0?W[k]*W[k]/V[k]:0;row[6]=tm[k];row[7]=visits[k]>1?sqrt(t2[k]/(visits[k]-1)):NAN;}
 free(order);
}

/* Common-history empirical MC of finite-episode Q^pi; no Markov shortcut. */
void truth_q(int n,int T,const double*s,const double*r,const unsigned char*valid,int bits,double cost,double*out,int64_t*counts){
 double sums[16]={0};double*ys=calloc(n*16,sizeof(double));int*ns=calloc(n*16,sizeof(int));memset(counts,0,16*sizeof(int64_t));
 for(int i=0;i<n;i++){
  double next[3]={cost*-1.,0,cost*-1.};
  for(int t=T-1;t>=0;t--){int j=i*(T+1)+t;double x=s[j],ret=r[i*T+t],current[3];
   for(int prev=-1;prev<=1;prev++){
    int st=ta_state(x,prev,0,0,0),chosen=(bits>>st)&1;double qs[2];
    for(int a=0;a<2;a++){int p=valid[j]?(a?(prev?0:(x>0?-1:x<0?1:0)):prev):0;qs[a]=p*ret-cost*abs(p-prev)+next[p+1];
     if(valid[j]&&(prev!=0||x!=0||a==0)){int k=2*st+a;sums[k]+=qs[a];counts[k]++;ys[i*16+k]+=qs[a];ns[i*16+k]++;}}
    current[prev+1]=qs[chosen];
   }
   memcpy(next,current,sizeof(next));
  }
 }
 for(int k=0;k<16;k++){out[k]=counts[k]?sums[k]/counts[k]:NAN;double ss=0;for(int i=0;i<n;i++){double u=ys[i*16+k]-out[k]*ns[i*16+k];ss+=u*u;}out[16+k]=counts[k]&&n>1?sqrt(ss*n/(n-1.))/counts[k]:NAN;}
 free(ys);free(ns);
}

/* Actual occupancy and entries, with invalid/terminal forced costs in source evaluator. */
void entry_diagnostics(int n,int T,const double*s,const unsigned char*valid,int kind,int bits,double enter,double exit,int64_t*out){
 memset(out,0,16*sizeof(int64_t));
 for(int i=0;i<n;i++){int prev=0;
  for(int t=0;t<T;t++){int j=i*(T+1)+t;double x=s[j];int p=prev;
   if(!valid[j])p=0;
   else{int st=ta_state(x,prev,0,0,0),a=kind==0?((bits>>st)&1):(!prev?fabs(x)>enter:-prev*x<exit);
    if(!prev){out[st]++;if(a&&x!=0)out[4+st]++;}
    if(a)p=prev?0:(x>0?-1:x<0?1:0);
   }
   if(!prev){out[8]++;if(p)out[9]++;if(valid[j])out[10]++;}
   if(p)out[11]++;if(prev&&!p&&!valid[j])out[12]++;
   out[13]++;prev=p;
  }
 }
}

/* Execution environment only. Invalid estimates and terminal close are forced. */
void position_paths(int n,int T,const double*z,int kind,int bits,double enter,double exit,int*out){
 for(int i=0;i<n;i++){int prev=0;
  for(int t=0;t<T;t++){double s=z[i*T+t];int p=prev;
   if(t==T-1||!isfinite(s))p=0;
   else{int st=ta_state(s,prev,0,0,0),a=kind==0?((bits>>st)&1):kind==2?(!prev?fabs(s)>enter:fabs(s)<exit):kind==3?(((bits>>(8+st))&1)?((bits>>st)&1):(!prev?fabs(s)>enter:-prev*s<exit)):(!prev?fabs(s)>enter:-prev*s<exit);
    if(a)p=prev?0:(s>0?-1:s<0?1:0);
   }
   out[i*T+t]=p;prev=p;
  }
 }
}
void enumerate_values(int n,int T,const double*z,const double*r,double c,double*sum,double*sq){
 for(int bits=0;bits<256;bits++){
  double a=0,b=0;
  for(int i=0;i<n;i++){int prev=0;double v=0;
   for(int t=0;t<T;t++){double s=z[i*T+t];int p=prev;
    if(t==T-1||!isfinite(s))p=0;
    else if((bits>>ta_state(s,prev,0,0,0))&1)p=prev?0:(s>0?-1:s<0?1:0);
    v+=p*r[i*T+t]-c*abs(p-prev);prev=p;
   }
   a+=v;b+=v*v;
  }
  sum[bits]=a;sq[bits]=b;
 }
}
void mc_moments(int n,int T,const double*z,const double*r,double*out){
 memset(out,0,5*256*sizeof(double));
 for(int bits=0;bits<256;bits++){
  for(int i=0;i<n;i++){int prev=0;double g=0,u=0;
   for(int t=0;t<T;t++){double s=z[i*T+t];int p=prev;
    if(t==T-1||!isfinite(s))p=0;
    else if((bits>>ta_state(s,prev,0,0,0))&1)p=prev?0:(s>0?-1:s<0?1:0);
    g+=p*r[i*T+t];u+=abs(p-prev);prev=p;
   }
   out[bits]+=g;out[256+bits]+=u;out[512+bits]+=g*g;out[768+bits]+=u*u;out[1024+bits]+=g*u;
  }
 }
}
