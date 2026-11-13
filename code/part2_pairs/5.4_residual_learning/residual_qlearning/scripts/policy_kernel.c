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
