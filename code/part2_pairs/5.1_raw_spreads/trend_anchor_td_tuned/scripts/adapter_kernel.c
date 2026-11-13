/* Original mode exports the original train/policy/ledger bodies exactly. */
#include "original_q_kernel.c"
/* Binary actions: 0 stay/hold, 1 enter/exit. */
static int ta_state(double d,int p,int align,int strong,int feature){
 double x=fabs(d);int b=x<.5?0:x<1.5?1:x<2.5?2:3;
 int s=b*2+(p!=0);return feature?s*2+(feature==1?!align:!strong):s;
}
static int ta_targets(Env *e,double d,int terminal,int H,double stop,int lock,
 int align,int allowcounter,int *reason,int *bm,int targets[2]){
 int oldmask=mask(e,d,terminal,H,stop,lock,0,reason);
 targets[0]=e->prev;targets[1]=e->prev?0:(d>0?-1:d<0?1:0);
 int cross=e->prev && e->prev*d>=0;
 if(terminal||*reason||cross){if(cross&&!*reason)*reason=4;
  *bm=1<<(e->prev?1:0);return oldmask;}
 *bm=0;for(int a=0;a<2;a++){
  int target=targets[a];
  if(!(oldmask&(1<<(target+1))))continue;
  if(!e->prev&&a==1&&(!target||(!allowcounter&&!align)))continue;
  *bm|=1<<a;
 }
 return oldmask;
}
static int ta_greedy(const double*q,int s,int bm){
 if(bm==1)return 0;if(bm==2)return 1;
 /* Same resulting-target preference: previous, zero, short, long. */
 return q[2*s]>=q[2*s+1]?0:1;
}
void ta_policy(int n,const double*d,const double*drift,const int*strong,const double*q,
 int feature,int kind,double entry,double exit,int aligned,int H,double stop,int lock,
 int *path,int *details){
 Env e={0,0,0,0};for(int t=0;t<n;t++){
  int al=(-((d[t]>0)-(d[t]<0))*drift[t]>0),reason,bm,targets[2];
  ta_targets(&e,d[t],t==n-1,H,stop,lock,al,!aligned,&reason,&bm,targets);
  int s=ta_state(d[t],e.prev,al,strong[t],feature),a=0;
  if(kind==0)a=ta_greedy(q,s,bm);
  else {
   int desired=e.prev;
   if(e.prev){if(e.prev*d[t]>=0 || (exit>.0&&fabs(d[t])<=exit))desired=0;}
   else if(t>0&&fabs(d[t-1])<entry&&fabs(d[t])>=entry&&(!aligned||al))desired=targets[1];
   a=(desired!=e.prev);if(!(bm&(1<<a)))a=(bm&1)?0:1;
  }
  int p=targets[a];
  details[6*t]=a;details[6*t+1]=reason;details[6*t+2]=bm;
  details[6*t+3]=e.age;details[6*t+4]=e.shortlock;details[6*t+5]=e.longlock;
  path[t]=p;advance(&e,p);
 }
}
static double ta_evaluate(int ne,const int*o,const double*d,const double*drift,
 const int*strong,const double*r,const double*q,double c,int feature,int aligned,int H,double stop,int lock){
 if(!ne)return NAN;double total=0;
 for(int j=0;j<ne;j++){Env e={0,0,0,0};for(int t=o[j];t<o[j+1];t++){
  int al=(-((d[t]>0)-(d[t]<0))*drift[t]>0),reason,bm,targets[2];
  ta_targets(&e,d[t],t==o[j+1]-1,H,stop,lock,al,!aligned,&reason,&bm,targets);
  int s=ta_state(d[t],e.prev,al,strong[t],feature),a=ta_greedy(q,s,bm),p=targets[a];
  total+=reward_bps(p,e.prev,r[t],c);advance(&e,p);
 }}return total/ne;
}
void ta_train(int ne,const int*o,const double*d,const double*drift,const int*strong,const double*r,
 int nv,const int*vo,const double*vd,const double*vdrift,const int*vstrong,const double*vr,
 uint64_t seed,double alpha,double gamma,double c,int epochs,int early,int feature,int aligned,
 int H,double stop,int lock,double*q,double*curves,int*snaps,int64_t*visits,int64_t*result){
 int ns=feature?16:8;memset(q,0,ns*2*sizeof(double));memset(visits,0,ns*2*sizeof(int64_t));memset(result,0,6*sizeof(int64_t));
 int *order=malloc(ne*sizeof(int)),*last=malloc(ns*sizeof(int)),stable=0,have=0;
 for(int i=0;i<ne;i++)order[i]=i;
 for(int epoch=1;epoch<=epochs;epoch++){
  for(int i=ne-1;i>0;i--){int j=bounded(&seed,i+1),tmp=order[i];order[i]=order[j];order[j]=tmp;}
  double eps=epoch>=240?.05:1.-.95*(epoch-1)/239.,total=0;
  for(int j=0;j<ne;j++){int eidx=order[j],end=o[eidx+1];Env e={0,0,0,0};
   for(int t=o[eidx];t<end;t++){
    int terminal=t==end-1,al=(-((d[t]>0)-(d[t]<0))*drift[t]>0),reason,bm,targets[2];
    ta_targets(&e,d[t],terminal,H,stop,lock,al,!aligned,&reason,&bm,targets);
    int s=ta_state(d[t],e.prev,al,strong[t],feature),a;
    if(terminal)a=e.prev?1:0;
    else {double u=uniform(&seed);int count=0,list[2];for(int k=0;k<2;k++)if(bm&(1<<k))list[count++]=k;
     int exploratory=list[bounded(&seed,count)];a=u<eps?exploratory:ta_greedy(q,s,bm);}
    int p=targets[a];double reward=reward_bps(p,e.prev,r[t],c),bootstrap=0;
    Env next=e;advance(&next,p);
    if(!terminal){int na=(-((d[t+1]>0)-(d[t+1]<0))*drift[t+1]>0),nr,nm,nt[2];
     ta_targets(&next,d[t+1],t==end-2,H,stop,lock,na,!aligned,&nr,&nm,nt);
     int sn=ta_state(d[t+1],p,na,strong[t+1],feature);bootstrap=q[sn*2+ta_greedy(q,sn,nm)];}
    int k=s*2+a;q[k]+=alpha*(reward+gamma*bootstrap-q[k]);visits[k]++;total+=reward;advance(&e,p);result[4]++;
   }result[3]++;
  }
  curves[3*(epoch-1)]=total/ne;
  curves[3*(epoch-1)+1]=ta_evaluate(nv,vo,vd,vdrift,vstrong,vr,q,c,feature,aligned,H,stop,lock);
  curves[3*(epoch-1)+2]=eps;result[5]+=nv;result[0]=epoch;
  if(epoch%10==0){int equal=have;
   for(int s=0;s<ns;s++){int a=ta_greedy(q,s,3);snaps[(epoch/10-1)*ns+s]=a;if(have&&a!=last[s])equal=0;last[s]=a;}
   stable=equal?stable+10:0;have=1;
   if(stable>=50&&epoch>=40){double now=0,before=0;for(int k=epoch-20;k<epoch;k++)now+=curves[3*k]/20.;for(int k=epoch-40;k<epoch-20;k++)before+=curves[3*k]/20.;
    if(fabs(now-before)/fmax(fabs(before),1e-12)<.01){result[1]=1;result[2]=epoch;if(early)break;}}
  }
 }free(order);free(last);
}
