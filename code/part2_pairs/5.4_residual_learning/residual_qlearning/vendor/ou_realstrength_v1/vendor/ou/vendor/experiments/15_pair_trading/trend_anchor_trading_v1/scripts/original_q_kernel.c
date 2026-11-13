#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
typedef struct {int prev,age,shortlock,longlock;} Env;
static uint64_t next_u64(uint64_t *s){uint64_t z=(*s+=UINT64_C(0x9e3779b97f4a7c15));z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);return z^(z>>31);}
static double uniform(uint64_t *s){return (next_u64(s)>>11)*0x1.0p-53;}
static int bounded(uint64_t *s,uint64_t n){uint64_t v,threshold=(-n)%n;do{v=next_u64(s);}while(v<threshold);return v%n;}
static int ix(int b,int p,int a){return (b*3+p+1)*3+a+1;}
static int mask(Env *e,double z,int terminal,int H,double stop,int lock,int incremental,int *reason){
    *reason=0;
    if(lock){if(z<=.5)e->shortlock=0;if(z>=-.5)e->longlock=0;}
    if(terminal){if(e->prev)*reason=3;return 1<<((incremental?-e->prev:0)+1);}
    if(stop>0&&((e->prev==-1&&z>=stop)||(e->prev==1&&z<=-stop)))*reason=1;
    else if(H>0&&e->prev&&e->age>=H)*reason=2;
    if(*reason){if(lock){if(e->prev==-1)e->shortlock=1;else e->longlock=1;}return 1<<((incremental?-e->prev:0)+1);}
    int m=0;for(int a=-1;a<=1;a++){
        int p=incremental?e->prev+a:a;
        if(p < -1||p>1)continue;
        if(lock&&p!=e->prev&&((p==-1&&e->shortlock)||(p==1&&e->longlock)))continue;
        m|=1<<(a+1);
    }return m;
}
static int greedy(const double *q,int b,int p,int m,int incremental){
    double best=-INFINITY;for(int a=-1;a<=1;a++)if((m&(1<<(a+1)))&&q[ix(b,p,a)]>best)best=q[ix(b,p,a)];
    int order[4]={incremental?0:p,incremental?-p:0,-1,1};
    for(int i=0;i<4;i++){int a=order[i];if((m&(1<<(a+1)))&&q[ix(b,p,a)]==best)return a;}return 0;
}
static void advance(Env *e,int p){e->age=p?(p==e->prev?e->age+1:1):0;e->prev=p;}
int allowed_mask(int prev,int age,int shortlock,int longlock,double z,int terminal,int H,double stop,int lock,int incremental){Env e={prev,age,shortlock,longlock};int reason;return mask(&e,z,terminal,H,stop,lock,incremental,&reason);}
double max_allowed(const double *q,int b,int prev,int m,int incremental){return q[ix(b,prev,greedy(q,b,prev,m,incremental))];}
double reward_bps(int p,int prev,double forward,double c){return p*forward-c*abs(p-prev)/2.;}
void ledger(int n,const double *r,const int *p,double c,double *out){int prev=0;for(int t=0;t<n;t++){out[3*t]=p[t]*r[t];out[3*t+1]=c*abs(p[t]-prev)/2.;out[3*t+2]=reward_bps(p[t],prev,r[t],c);prev=p[t];}}
void policy(int n,const int *b,const double *z,const double *q,int kind,double entry,double exit,int constant,int H,double stop,int lock,int incremental,int *p,int *details){
    Env e={0,0,0,0};for(int t=0;t<n;t++){
        int reason,m=mask(&e,z[t],t==n-1,H,stop,lock,incremental,&reason),a;
        if(kind==0)a=greedy(q,b[t],e.prev,m,incremental);
        else{
            int desired=constant;
            if(kind==1){desired=e.prev;if(!e.prev)desired=z[t]>=entry?-1:z[t]<=-entry?1:0;else if((e.prev==1&&z[t]>=-exit)||(e.prev==-1&&z[t]<=exit))desired=0;}
            a=incremental?desired-e.prev:desired;
            if(a < -1||a>1||!(m&(1<<(a+1)))){a=incremental?-e.prev:0;}
        }
        if(details){details[6*t]=a;details[6*t+1]=reason;details[6*t+2]=m;details[6*t+3]=e.age;details[6*t+4]=e.shortlock;details[6*t+5]=e.longlock;}
        p[t]=incremental?e.prev+a:a;advance(&e,p[t]);
    }
}
static double evaluate(int ne,const int *o,const int *b,const double *z,const double *r,const double *q,double c,int H,double stop,int lock,int incremental){
    if(!ne)return NAN;double total=0;
    for(int j=0;j<ne;j++){Env e={0,0,0,0};for(int t=o[j];t<o[j+1];t++){int reason,m=mask(&e,z[t],t==o[j+1]-1,H,stop,lock,incremental,&reason);int a=greedy(q,b[t],e.prev,m,incremental),p=incremental?e.prev+a:a;total+=reward_bps(p,e.prev,r[t],c);advance(&e,p);}}
    return total/ne;
}
void train(int ne,const int *o,const int *b,const double *z,const double *r,int nv,const int *vo,const int *vb,const double *vz,const double *vr,uint64_t seed,double alpha,double gamma,double c,int epochs,int early,int nbins,int H,double stop,int lock,int incremental,double *q,double *curves,int *snapshots,int64_t *visits,int64_t *result){
    memset(q,0,nbins*9*sizeof(double));memset(visits,0,nbins*9*sizeof(int64_t));memset(result,0,6*sizeof(int64_t));
    int *order=malloc(ne*sizeof(int)),*last=malloc(nbins*3*sizeof(int));int stable=0,have=0;
    for(int e=0;e<ne;e++)order[e]=e;
    for(int epoch=1;epoch<=epochs;epoch++){
        for(int e=ne-1;e>0;e--){int j=bounded(&seed,e+1),tmp=order[e];order[e]=order[j];order[j]=tmp;}
        double eps=epoch>=240?.05:1.-.95*(epoch-1)/239.,total=0;
        for(int j=0;j<ne;j++){
            int eidx=order[j],end=o[eidx+1];Env e={0,0,0,0};
            for(int t=o[eidx];t<end;t++){
                int terminal=t==end-1,reason,m=mask(&e,z[t],terminal,H,stop,lock,incremental,&reason),a;
                if(terminal)a=incremental?-e.prev:0;
                else{double u=uniform(&seed);int count=0,list[3];for(int k=-1;k<=1;k++)if(m&(1<<(k+1)))list[count++]=k;int exploratory=list[bounded(&seed,count)];a=u<eps?exploratory:greedy(q,b[t],e.prev,m,incremental);}
                int p=incremental?e.prev+a:a;double reward=reward_bps(p,e.prev,r[t],c),bootstrap=0;
                Env next=e;advance(&next,p);
                if(!terminal){int nr,nm=mask(&next,z[t+1],t==end-2,H,stop,lock,incremental,&nr);bootstrap=max_allowed(q,b[t+1],p,nm,incremental);}
                int k=ix(b[t],e.prev,a);q[k]+=alpha*(reward+gamma*bootstrap-q[k]);visits[k]++;total+=reward;advance(&e,p);result[4]++;
            }result[3]++;
        }
        curves[3*(epoch-1)]=total/ne;curves[3*(epoch-1)+1]=evaluate(nv,vo,vb,vz,vr,q,c,H,stop,lock,incremental);curves[3*(epoch-1)+2]=eps;result[5]+=nv;result[0]=epoch;
        if(epoch%10==0){
            int *current=malloc(nbins*3*sizeof(int));int equal=have;
            for(int bidx=0;bidx<nbins;bidx++)for(int p=-1;p<=1;p++){
                int i=bidx*3+p+1;Env probe={p,0,0,0};int reason,m=mask(&probe,0,0,0,0,0,incremental,&reason);current[i]=greedy(q,bidx,p,m,incremental);snapshots[(epoch/10-1)*nbins*3+i]=current[i];if(have&&current[i]!=last[i])equal=0;
            }
            stable=equal?stable+10:0;memcpy(last,current,nbins*3*sizeof(int));have=1;free(current);
            if(stable>=50&&epoch>=40){double now=0,before=0;for(int k=epoch-20;k<epoch;k++)now+=curves[3*k]/20.;for(int k=epoch-40;k<epoch-20;k++)before+=curves[3*k]/20.;if(fabs(now-before)/fmax(fabs(before),1e-12)<.01){result[1]=1;result[2]=epoch;if(early)break;}}
        }
    }free(order);free(last);
}
