const fs=require('fs');
const G=require('./getup.js');const RS=G.RS;
const NPAR=G.NPAR,P=+process.env.P||16,GENS=+process.env.GENS||60,SIGMA=+process.env.SIGMA||0.2,LR=+process.env.LR||0.08,TAG=process.env.TAG||'su';
function extend(){ // build an 8-keyframe start from the 6-keyframe sitting solution
  const d=JSON.parse(fs.readFileSync(process.env.RESUME,'utf8')),x6=d.x,N6=6,C=G.NCH,x=new Float64Array(G.NPAR);
  for(let q=0;q<6;q++){for(let c=0;c<C;c++)x[q*C+c]=x6[q*C+c];x[G.NKF*C+q]=x6[N6*C+q];}
  const crouch=[0,-0.3,-0.2,1.0,0,0,1.2,-1.6,0.3],stand=[0,0,0,0,0,0,0,0,0];
  for(let c=0;c<C;c++){x[6*C+c]=crouch[c];x[7*C+c]=stand[c];}
  x[G.NKF*C+6]=0.7;x[G.NKF*C+7]=0.7;return x;}
const S={theta:(process.env.EXTEND&&fs.existsSync(process.env.RESUME))?extend():(process.env.RESUME&&fs.existsSync(process.env.RESUME))?Float64Array.from(JSON.parse(fs.readFileSync(process.env.RESUME,'utf8')).x):G.guess(),m:new Float64Array(NPAR),v:new Float64Array(NPAR),t:0};
let best=-1,bestX=S.theta.slice();
const t0=Date.now();
for(let g=0;g<GENS;g++){
  const eps=RS.esNoise(P,NPAR,5000+g),Fp=[],Fm=[],th=new Float64Array(NPAR);
  for(let i=0;i<P;i++){
    for(let k=0;k<NPAR;k++)th[k]=S.theta[k]+SIGMA*eps[i][k];
    const a=G.episode(th,{T:+process.env.T||9});Fp.push(a);
    for(let k=0;k<NPAR;k++)th[k]=S.theta[k]-SIGMA*eps[i][k];
    const b=G.episode(th,{T:+process.env.T||9});Fm.push(b);
    if(a>best){best=a;bestX=Float64Array.from(S.theta.map((v,k)=>v+SIGMA*eps[i][k]));}
    if(b>best){best=b;bestX=Float64Array.from(S.theta.map((v,k)=>v-SIGMA*eps[i][k]));}
  }
  RS.esStep(S,eps,Fp,Fm,SIGMA,LR);
  const cur=G.episode(S.theta,{T:+process.env.T||9});
  console.log('gen',g+1,((Date.now()-t0)/1000).toFixed(0)+'s','current',cur.toFixed(3),'best candidate so far',best.toFixed(3));
  fs.writeFileSync(__dirname+'/out_'+TAG+'.json',JSON.stringify({x:Array.from(S.theta),bestX:Array.from(bestX),best,gen:g+1}));
}
