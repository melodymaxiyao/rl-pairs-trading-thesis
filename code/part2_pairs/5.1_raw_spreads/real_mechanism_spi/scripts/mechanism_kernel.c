#include "/Users/maxiyao/Desktop/FYP/experiments/15_pair_trading/trend_anchor_trading_v1/scripts/adapter_kernel.c"
void ms_train(int ne,const int*o,const double*d,const double*drift,const int*strong,const double*r,
 int nv,const int*vo,const double*vd,const double*vdrift,const int*vstrong,const double*vr,
 uint64_t seed,double alpha,double gamma,double c,int epochs,int early,int feature,int aligned,
 int H,double stop,int lock,int stepmode,int ordermode,const int*pairids,int npairs,double*q,double*curves,int*snaps,int64_t*visits,int64_t*result,double*diag,double*trajectory){
 int ns=feature?16:8;memset(q,0,ns*2*sizeof(double));memset(visits,0,ns*2*sizeof(int64_t));memset(result,0,6*sizeof(int64_t));
 int *order=malloc(ne*sizeof(int)),*last=malloc(ns*sizeof(int)),stable=0,have=0;
 for(int i=0;i<ne;i++)order[i]=i;
 double W[16]={0},V[16]={0},tm[16]={0},t2[16]={0},we[16]={0},wt[16]={0},wp[16]={0},et[16]={0};int64_t en[16]={0};
 int *tailpair=calloc(npairs,sizeof(int));
 for(int epoch=1;epoch<=epochs;epoch++){
  memset(we,0,sizeof(we));memset(wt,0,sizeof(wt));memset(wp,0,sizeof(wp));memset(et,0,sizeof(et));memset(en,0,sizeof(en));
  for(int i=ne-1;i>0;i--){int j=bounded(&seed,i+1),tmp=order[i];order[i]=order[j];order[j]=tmp;}
  if(ordermode==1)for(int j=0;j<ne;j++)order[j]=j;
  memset(tailpair,0,npairs*sizeof(int));int marked=0,need=(npairs+9)/10;
  for(int j=ne-1;j>=0&&marked<need;j--){int p=pairids[order[j]];if(!tailpair[p]){tailpair[p]=1;marked++;}}
  double eps=epoch>=240?.05:1.-.95*(epoch-1)/239.,total=0;
  for(int j=0;j<ne;j++){int eidx=order[j],end=o[eidx+1];Env e={0,0,0,0};
   for(int t=o[eidx];t<end;t++){
    int terminal=t==end-1,al=(-((d[t]>0)-(d[t]<0))*drift[t]>0),reason,bm,targets[2];
    ta_targets(&e,d[t],terminal,H,stop,lock,al,!aligned,&reason,&bm,targets);
    int s=ta_state(d[t],e.prev,al,strong[t],feature),a;
    if(terminal)a=e.prev?1:0;
    else {double u=uniform(&seed);result[6]=(int64_t)(((uint64_t)result[6])*UINT64_C(1099511628211)^seed);int count=0,list[2];for(int k=0;k<2;k++)if(bm&(1<<k))list[count++]=k;
     int exploratory=list[bounded(&seed,count)];a=u<eps?exploratory:ta_greedy(q,s,bm);}
    int p=targets[a];double reward=reward_bps(p,e.prev,r[t],c),bootstrap=0;
    Env next=e;advance(&next,p);
    if(!terminal){int na=(-((d[t+1]>0)-(d[t+1]<0))*drift[t+1]>0),nr,nm,nt[2];
     ta_targets(&next,d[t+1],t==end-2,H,stop,lock,na,!aligned,&nr,&nm,nt);
     int sn=ta_state(d[t+1],p,na,strong[t+1],feature);bootstrap=q[sn*2+ta_greedy(q,sn,nm)];}
    int k=s*2+a;double step=stepmode?1./pow((double)visits[k]+1.,.7):alpha;if(stepmode==2&&step>.05)step=.05;
    double target=reward+gamma*bootstrap;
    W[k]=(1.-step)*W[k]+step;V[k]=(1.-step)*(1.-step)*V[k]+step*step;
    we[k]=(1.-step)*we[k]+step;wt[k]=(1.-step)*wt[k]+(j>=ne-ne/10?step:0.);wp[k]=(1.-step)*wp[k]+(tailpair[pairids[eidx]]?step:0.);
    double diff=target-tm[k];tm[k]+=diff/((double)visits[k]+1.);t2[k]+=diff*(target-tm[k]);et[k]+=target;en[k]++;
    q[k]+=step*(reward+gamma*bootstrap-q[k]);visits[k]++;total+=reward;advance(&e,p);result[4]++;
   }result[3]++;
  }
  for(int k=0;k<16;k++)trajectory[(epoch-1)*16+k]=q[k];
  curves[3*(epoch-1)]=total/ne;
  curves[3*(epoch-1)+1]=ta_evaluate(nv,vo,vd,vdrift,vstrong,vr,q,c,feature,aligned,H,stop,lock);
  curves[3*(epoch-1)+2]=eps;result[5]+=nv;result[0]=epoch;
  if(epoch%10==0){int equal=have;
   for(int s=0;s<ns;s++){int a=ta_greedy(q,s,3);snaps[(epoch/10-1)*ns+s]=a;if(have&&a!=last[s])equal=0;last[s]=a;}
   stable=equal?stable+10:0;have=1;
   if(stable>=50&&epoch>=40){double now=0,before=0;for(int k=epoch-20;k<epoch;k++)now+=curves[3*k]/20.;for(int k=epoch-40;k<epoch-20;k++)before+=curves[3*k]/20.;
    if(fabs(now-before)/fmax(fabs(before),1e-12)<.01){result[1]=1;result[2]=epoch;if(early)break;}}
  }
 }
 for(int k=0;k<16;k++){
  double *row=diag+10*k;row[0]=W[k];row[1]=V[k];row[2]=V[k]>0?W[k]*W[k]/V[k]:0.;row[3]=we[k];row[4]=wt[k];row[5]=wp[k];row[6]=tm[k];row[7]=visits[k]>1?sqrt(t2[k]/(visits[k]-1)):NAN;row[8]=en[k]?et[k]/en[k]:NAN;row[9]=q[k]-row[8];
 }
 free(tailpair);free(order);free(last);
}

/* Baseline-anchored switch executor. Uses the unchanged masks/environment. */
void ms_policy(int n,const double*d,const double*drift,const int*strong,int bits,int H,double stop,int lock,int*path,int*details){
 Env e={0,0,0,0};for(int t=0;t<n;t++){
  int al=(-((d[t]>0)-(d[t]<0))*drift[t]>0),reason,bm,targets[2];
  ta_targets(&e,d[t],t==n-1,H,stop,lock,al,0,&reason,&bm,targets);
  int b=ta_state(d[t],e.prev,al,strong[t],0)/2,a=0;
  if(e.prev){if(bits&(1<<(6+b)))a=1;}
  else{
   int crossing=t>0&&fabs(d[t-1])<2.&&fabs(d[t])>=2.;
   if(crossing&&!(b>=2&&(bits&(1<<(4+b-2)))))a=1;
   if(bits&(1<<b))a=1;
  }
  if(!(bm&(1<<a)))a=(bm&1)?0:1;int p=targets[a];
  if(details){details[6*t]=a;details[6*t+1]=reason;details[6*t+2]=bm;details[6*t+3]=e.age;details[6*t+4]=e.shortlock;details[6*t+5]=e.longlock;}
  path[t]=p;advance(&e,p);
 }
}
/* Batched net/gross/turnover/entry-count plus cross-sectional daily sums. */
void ms_batch(int ne,int n,const double*z,const double*drift,const double*r,int np,const double*qs,const int*kinds,const double*entries,const double*exits,const int*bits,int H,double stop,int lock,double*out,double*daily){
 int*p=calloc(n,sizeof(int)),*details=calloc(6*n,sizeof(int)),*strong=calloc(n,sizeof(int));
 memset(out,0,(size_t)ne*np*4*sizeof(double));memset(daily,0,(size_t)n*np*sizeof(double));
 for(int a=0;a<np;a++)for(int j=0;j<ne;j++){
  const double*d=z+j*n,*g=drift+j*n,*f=r+j*n;
  if(kinds[a]==2)ms_policy(n,d,g,strong,bits[a],H,stop,lock,p,details);
  else if(kinds[a]==3)memset(p,0,n*sizeof(int));
  else ta_policy(n,d,g,strong,qs+a*16,0,kinds[a],entries[a],exits[a],1,H,stop,lock,p,details);
  int prev=0;for(int t=0;t<n;t++){
   double reward=reward_bps(p[t],prev,f[t],10.);double*row=out+((size_t)a*ne+j)*4;row[0]+=reward;row[1]+=p[t]*f[t];row[2]+=abs(p[t]-prev)/2.;row[3]+=p[t]!=0&&prev==0;daily[(size_t)a*n+t]+=reward;prev=p[t];
  }
 }free(p);free(details);free(strong);
}
void ms_weights_queries(int nq,const int64_t*ns,int mode,double alpha,double*out){
 double W=0.,V=0.;int q=0;int64_t n=0;
 while(q<nq&&ns[q]==0){out[2*q]=W;out[2*q+1]=V;q++;}
 while(q<nq){n++;double a=mode?1./pow((double)n,.7):alpha;if(mode==2&&a>.05)a=.05;W=(1-a)*W+a;V=(1-a)*(1-a)*V+a*a;
  while(q<nq&&ns[q]==n){out[2*q]=W;out[2*q+1]=V;q++;}}
}
