#define ta_train source_ta_train
#include "/Users/maxiyao/Desktop/FYP/experiments/15_pair_trading/trend_anchor_trading_v1/scripts/adapter_kernel.c"
#undef ta_train
void sc_train(int ne,const int*o,const double*d,const double*drift,const int*strong,const double*r,
 int nv,const int*vo,const double*vd,const double*vdrift,const int*vstrong,const double*vr,
 uint64_t seed,double alpha,double gamma,double c,int epochs,int early,int feature,int aligned,
 int H,double stop,int lock,int tuned,double*q,double*curves,int*snaps,int64_t*visits,int64_t*result){
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
    int k=s*2+a;double step=tuned?1./pow((double)visits[k]+1.,.7):alpha;q[k]+=step*(reward+gamma*bootstrap-q[k]);visits[k]++;total+=reward;advance(&e,p);result[4]++;
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

/* Batch execution only; every decision uses the unchanged ta_policy body. */
void sc_batch(int ne,int n,const double*z,const double*drift,const double*r,int np,
 const double*qs,const int*kinds,const double*entries,const double*exits,
 int H,double stop,int lock,double*out,int64_t*vis){
 int *p=calloc(n,sizeof(int)),*details=calloc(6*n,sizeof(int)),*strong=calloc(n,sizeof(int));
 memset(out,0,(size_t)ne*np*3*sizeof(double));memset(vis,0,(size_t)np*8*sizeof(int64_t));
 for(int a=0;a<np;a++)for(int j=0;j<ne;j++){
  const double*d=z+j*n,*g=drift+j*n,*f=r+j*n;
  ta_policy(n,d,g,strong,qs+a*16,0,kinds[a],entries[a],exits[a],1,H,stop,lock,p,details);
  double net=0,gross=0,turn=0;int prev=0;
  for(int t=0;t<n;t++){
   gross+=p[t]*f[t];turn+=abs(p[t]-prev)/2.;net+=reward_bps(p[t],prev,f[t],10.);
   int s=ta_state(d[t],prev,0,0,0);vis[a*8+s]++;prev=p[t];
  }
  out[(a*ne+j)*3]=net;out[(a*ne+j)*3+1]=gross;out[(a*ne+j)*3+2]=turn;
 }free(p);free(details);free(strong);
}
double sc_rate(int64_t visits){return 1./pow((double)visits+1.,.7);}
