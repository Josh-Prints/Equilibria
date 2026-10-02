const fs=require('fs');const S2=require('./stage2.js');const RS=S2.RS;
const NPAR=S2.NPAR,P=+process.env.P||12,GENS=+process.env.GENS||80,SIGMA=+process.env.SIGMA||0.25,LR=+process.env.LR||0.08,TAG=process.env.TAG||'s2',T=+process.env.T||12;
const S={theta:(process.env.RESUME&&fs.existsSync(process.env.RESUME))?Float64Array.from(JSON.parse(fs.readFileSync(process.env.RESUME,'utf8')).x):S2.guess2(),m:new Float64Array(NPAR),v:new Float64Array(NPAR),t:0};
let best=-1,bestX=S.theta.slice();const t0=Date.now();
for(let g=0;g<GENS;g++){
  const eps=RS.esNoise(P,NPAR,7000+g),Fp=[],Fm=[],th=new Float64Array(NPAR);
  for(let i=0;i<P;i++){
    for(let k=0;k<NPAR;k++)th[k]=S.theta[k]+SIGMA*eps[i][k];const a=S2.episode(th,{T});Fp.push(a);
    for(let k=0;k<NPAR;k++)th[k]=S.theta[k]-SIGMA*eps[i][k];const b=S2.episode(th,{T});Fm.push(b);
    if(a>best){best=a;bestX=Float64Array.from(S.theta.map((v,k)=>v+SIGMA*eps[i][k]));}
    if(b>best){best=b;bestX=Float64Array.from(S.theta.map((v,k)=>v-SIGMA*eps[i][k]));}
  }
  RS.esStep(S,eps,Fp,Fm,SIGMA,LR);
  if((g+1)%3===0){const cur=S2.episode(S.theta,{T});console.log('gen',g+1,((Date.now()-t0)/1000).toFixed(0)+'s','current',cur.toFixed(3),'best candidate',best.toFixed(3));}
  fs.writeFileSync(__dirname+'/out_'+TAG+'.json',JSON.stringify({x:Array.from(S.theta),bestX:Array.from(bestX),best,gen:g+1}));
}
