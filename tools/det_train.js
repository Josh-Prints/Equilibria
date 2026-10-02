// Trains the neural event detector (calm / stumble / falling / moved / down) by supervised learning.
// 1) simulates lots of episodes (standing, pushes, drops, lying, being grabbed and dragged/lifted like on the page),
// 2) labels every moment with what is really happening (using the future and the hidden drag state, which the
//    network never sees), 3) trains a small MLP on the body's own sensors only, 4) writes det_weights.js.
// Round 2 re-collects data with the trained detector driving the arm reactions, so it learns its own effect.
// usage: node tools/det_train.js [episodesPerRound=700] [epochs=25]      (needs: npm install planck@1.0.0)
'use strict';
const fs=require('fs'),path=require('path'),os=require('os');
const {Worker,isMainThread,parentPort,workerData}=require('worker_threads');
const planck=require('planck');const {Vec2,Box}=planck;
const ROOT=path.join(__dirname,'..');
function loadRS(detw){
  const src=fs.readFileSync(path.join(ROOT,'sim.js'),'utf8');
  const RS=new Function('planck',src+';return RS;')(planck);
  if(detw)RS.setDet(detw);
  return RS;
}
const STRIDE=3; // keep every 3rd policy tick (25 ms)

// ---------------- one simulated episode -> features + labels ----------------
function episode(RS,seed,nn){
  const r=RS.rngMake(seed),u=r();
  const kind=u<0.1?'stand':u<0.4?'push':u<0.55?'drop':u<0.65?'fallen':'drag';
  const sc=RS.sampleScenario(r,{only:kind==='drag'?'stand':kind,pushMax:3.0});
  if(kind==='push'&&r()<0.5){ // page-style chest pushes too
    sc.pushes=[];let t=0.8;const n=1+Math.floor(r()*3);
    for(let k=0;k<n;k++){t+=0.4+r()*2;sc.pushes.push({t:t,body:1,jx:(r()<0.5?-1:1)*(0.2+r()*2.8),jy:0});}
    sc.T=Math.max(8,t+3.5);
  }
  Object.assign(RS.FLAGS,{bal2:1,step:0,air:0,land:0,fall:0,soft:1,smooth:0,nn:nn?1:0});
  const w=new planck.World({gravity:Vec2(0,-10)});
  const gd=w.createBody();gd.createFixture(Box(1000,1,Vec2(0,-1),0),{friction:sc.fric});
  const rig=RS.buildRig(w,0,0,-1,{pose:sc.pose,rootAng:sc.rootAng,h:sc.h,mass:sc.mass,fric:sc.fric,gain:sc.gain,partVar:sc.partVar,clear:sc.clear});
  if(sc.vel)rig.parts.forEach(b=>{b.setLinearVelocity(Vec2(sc.vel.vx,sc.vel.vy));b.setAngularVelocity(sc.vel.w);});
  let M=0;rig.parts.forEach(b=>M+=b.getMass());
  // drag plan (same spring as the page's finger drag)
  let drag=null;
  if(kind==='drag'){
    const bi=[0,1,3,6,12,5,11,9,15,1,3][Math.floor(r()*11)],t0=0.5+r()*2.5,T1=0.15+r()*1.4,T2=r()*2.2;
    drag={bi:bi,t0:t0,T1:T1,T2:T2,dx:(2*r()-1)*1.6,dy:-0.2+r()*1.4,amp:r()*0.35,fq:0.5+r()*2.5,local:null,p0:null};
    sc.T=Math.max(6,t0+T1+T2+3.5);
  }
  const theta=new Float64Array(RS.NP),buf=RS.newBuf(),rr=RS.rngMake(seed^0x9e3779b9);
  const steps=Math.round(sc.T*120),feat=[],head=[],dragOn=[],stp=[],air=[],up=[];
  let pi=0;const x=new Float64Array(RS.DIN);
  for(let i=0;i<steps;i++){
    const t=i/120;
    while(pi<sc.pushes.length&&t>=sc.pushes[pi].t){const p=sc.pushes[pi++],b=rig.parts[p.body];b.applyLinearImpulse(Vec2(p.jx,p.jy),b.getWorldCenter(),true);}
    RS.act(rig,theta,buf,rr,{delay:sc.delay,noise:sc.noise});
    let dOn=false;
    for(let s=0;s<RS.SUB;s++){
      if(drag&&t>=drag.t0&&t<drag.t0+drag.T1+drag.T2){
        const b=rig.parts[drag.bi];
        if(!drag.local){drag.local=b.getLocalCenter().clone();drag.p0=b.getWorldCenter().clone();}
        const tt=t-drag.t0,a=Math.min(1,tt/drag.T1),sm=a*a*(3-2*a),wg=tt>drag.T1?drag.amp*Math.sin(2*Math.PI*drag.fq*(tt-drag.T1)):0;
        const tx=drag.p0.x+drag.dx*sm+wg,ty=Math.max(0.05,drag.p0.y+drag.dy*sm+0.5*wg);
        const wp=b.getWorldPoint(drag.local),v=b.getLinearVelocityFromWorldPoint(wp);
        let ax=120*(tx-wp.x)-22*v.x,ay=120*(ty-wp.y)-22*v.y;const mag=Math.hypot(ax,ay);if(mag>80){ax*=80/mag;ay*=80/mag;}
        b.applyForce(Vec2(ax*M,ay*M),wp,true);dOn=true;
      }
      RS.pdRig(rig);w.step(RS.DT,RS.VEL_IT,RS.POS_IT);
    }
    const hy=rig.parts[0].getPosition().y;if(!(hy===hy))break;
    let low=1e9;for(const b of rig.parts){const y=b.getWorldCenter().y;if(y<low)low=y;}
    head.push(hy);dragOn.push(dOn);stp.push(rig.st2.ph>0||Math.abs(rig.e)>0.15);air.push(low>0.12);
    up.push(Math.cos(RS.wrap(rig.parts[1].getAngle()))>0.6);
    if(i%STRIDE===0&&rig.dh.length>6)feat.push([i,Float32Array.from(RS.detFeat(rig,x))]);
  }
  // labels: 0 calm 1 stumble 2 falling 3 moved 4 down
  const n=head.length,X=[],Y=[];
  for(const [i,f] of feat){
    if(i>=n)break;
    let lastDrag=-1e9;for(let k=Math.max(0,i-24);k<=i;k++)if(dragOn[k])lastDrag=k;
    let fmin=head[i];for(let k=i;k<Math.min(n,i+84);k++)fmin=Math.min(fmin,head[k]);   // lowest head over the next 0.7 s
    let y;
    if(i-lastDrag<=24)y=3;                                                  // being moved (or just let go)
    else if(head[i]<0.75)y=fmin<head[i]-0.25?2:4;                            // low: still going down, or down
    else if(fmin<0.75||(air[i]&&head[i]>0.75&&!up[i]))y=2;                   // will be on the ground soon
    else if(stp[i])y=1;
    else y=0;
    X.push(f);Y.push(y);
  }
  return {X,Y};
}

if(!isMainThread){
  const RS=loadRS(workerData.detw);const X=[],Y=[];
  for(const s of workerData.seeds){const e=episode(RS,s,workerData.nn);for(let k=0;k<e.X.length;k++){X.push(e.X[k]);Y.push(e.Y[k]);}}
  const flat=new Float32Array(X.length*RS.DIN);X.forEach((f,k)=>flat.set(f,k*RS.DIN));
  parentPort.postMessage({flat,Y:Int8Array.from(Y)},[flat.buffer]);
  return;
}

function collect(seeds,nn,detw){
  const NW=os.cpus().length,parts=[];for(let k=0;k<NW;k++)parts.push(seeds.filter((_,i)=>i%NW===k));
  return Promise.all(parts.map(sd=>new Promise((res,rej)=>{const wk=new Worker(__filename,{workerData:{seeds:sd,nn,detw}});wk.on('message',res);wk.on('error',rej);})));
}

// ---------------- MLP training (Adam, weighted cross-entropy) ----------------
function rng(seed){let s=seed>>>0;return ()=>{s=(s+0x6D2B79F5)|0;let t=Math.imul(s^(s>>>15),1|s);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296;};}
function train(X,Y,N,D,epochs,init){
  const H1=32,H2=32,O=5,r=rng(3),g=()=>{const u=1-r(),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};
  // normalisation
  const mu=new Float64Array(D),sd=new Float64Array(D);
  for(let i=0;i<N;i++)for(let d=0;d<D;d++)mu[d]+=X[i*D+d];for(let d=0;d<D;d++)mu[d]/=N;
  for(let i=0;i<N;i++)for(let d=0;d<D;d++){const z=X[i*D+d]-mu[d];sd[d]+=z*z;}for(let d=0;d<D;d++)sd[d]=Math.sqrt(sd[d]/N)+1e-3;
  const P={W1:new Float64Array(H1*D).map(()=>g()*Math.sqrt(1/D)),b1:new Float64Array(H1),W2:new Float64Array(H2*H1).map(()=>g()*Math.sqrt(1/H1)),b2:new Float64Array(H2),W3:new Float64Array(O*H2).map(()=>g()*Math.sqrt(1/H2)),b3:new Float64Array(O)};
  if(init)for(const k in P)P[k].set(init[k]);
  const cnt=new Float64Array(O);for(let i=0;i<N;i++)cnt[Y[i]]++;
  const cw=Array.from(cnt,c=>Math.min(5,Math.pow(N/(O*Math.max(c,1)),0.5)));
  const G={},Mm={},Vv={};for(const k in P){G[k]=new Float64Array(P[k].length);Mm[k]=new Float64Array(P[k].length);Vv[k]=new Float64Array(P[k].length);}
  const xs=new Float64Array(D),h1=new Float64Array(H1),h2=new Float64Array(H2),o=new Float64Array(O),d3=new Float64Array(O),d2=new Float64Array(H2),d1=new Float64Array(H1);
  const idx=Array.from({length:N},(_,i)=>i),BS=256;let t=0;
  for(let ep=0;ep<epochs;ep++){
    for(let i=N-1;i>0;i--){const j=Math.floor(r()*(i+1));const q=idx[i];idx[i]=idx[j];idx[j]=q;}
    const lr=0.002*(ep<epochs*0.7?1:0.3);let loss=0;
    for(let b0=0;b0<N;b0+=BS){
      for(const k in G)G[k].fill(0);const bn=Math.min(BS,N-b0);
      for(let bi=0;bi<bn;bi++){
        const n=idx[b0+bi],y=Y[n],wgt=cw[y];
        for(let d=0;d<D;d++)xs[d]=(X[n*D+d]-mu[d])/sd[d];
        for(let j=0;j<H1;j++){let s=P.b1[j];const ro=j*D;for(let d=0;d<D;d++)s+=P.W1[ro+d]*xs[d];h1[j]=Math.tanh(s);}
        for(let j=0;j<H2;j++){let s=P.b2[j];for(let i=0;i<H1;i++)s+=P.W2[j*H1+i]*h1[i];h2[j]=Math.tanh(s);}
        let m=-1e9;for(let j=0;j<O;j++){let s=P.b3[j];for(let i=0;i<H2;i++)s+=P.W3[j*H2+i]*h2[i];o[j]=s;if(s>m)m=s;}
        let z=0;for(let j=0;j<O;j++){o[j]=Math.exp(o[j]-m);z+=o[j];}for(let j=0;j<O;j++)o[j]/=z;
        loss-=wgt*Math.log(o[y]+1e-9);
        for(let j=0;j<O;j++)d3[j]=wgt*(o[j]-(j===y?1:0));
        d2.fill(0);for(let j=0;j<O;j++){G.b3[j]+=d3[j];for(let i=0;i<H2;i++){G.W3[j*H2+i]+=d3[j]*h2[i];d2[i]+=d3[j]*P.W3[j*H2+i];}}
        for(let i=0;i<H2;i++)d2[i]*=1-h2[i]*h2[i];
        d1.fill(0);for(let j=0;j<H2;j++){G.b2[j]+=d2[j];for(let i=0;i<H1;i++){G.W2[j*H1+i]+=d2[j]*h1[i];d1[i]+=d2[j]*P.W2[j*H1+i];}}
        for(let j=0;j<H1;j++){const dj=d1[j]*(1-h1[j]*h1[j]);G.b1[j]+=dj;const ro=j*D;for(let d=0;d<D;d++)G.W1[ro+d]+=dj*xs[d];}
      }
      t++;const b1=0.9,b2=0.999;
      for(const k in P){const p=P[k],gg=G[k],mm=Mm[k],vv=Vv[k];for(let q=0;q<p.length;q++){const gq=gg[q]/bn+(k[0]==='W'?1e-4*p[q]:0);mm[q]=b1*mm[q]+(1-b1)*gq;vv[q]=b2*vv[q]+(1-b2)*gq*gq;p[q]-=lr*(mm[q]/(1-Math.pow(b1,t)))/(Math.sqrt(vv[q]/(1-Math.pow(b2,t)))+1e-8);}}
    }
    if(ep%5===4||ep===epochs-1)console.log('  epoch',ep+1,'loss',(loss/N).toFixed(4));
  }
  return Object.assign({mu,sd},P);
}
function evaluate(RS,W,X,Y,N,D){
  const conf=[...Array(5)].map(()=>new Array(5).fill(0)),x=new Float64Array(D),o=new Float64Array(5);
  for(let i=0;i<N;i++){for(let d=0;d<D;d++)x[d]=X[i*D+d];RS.detForward(W,x,o);let b=0;for(let j=1;j<5;j++)if(o[j]>o[b])b=j;conf[Y[i]][b]++;}
  let ok=0;for(let j=0;j<5;j++)ok+=conf[j][j];
  console.log('  accuracy',(100*ok/N).toFixed(1)+'%  (rows = truth, cols = guess: '+RS.DCLS.join(' ')+')');
  conf.forEach((row,j)=>{const s=row.reduce((a,b)=>a+b,0);console.log('   '+RS.DCLS[j].padEnd(8),row.map(v=>String(v).padStart(6)).join(''),' recall '+(100*row[j]/Math.max(1,s)).toFixed(0)+'%');});
  return ok/N;
}
function merge(parts,D){let N=0;parts.forEach(p=>N+=p.Y.length);const X=new Float32Array(N*D),Y=new Int8Array(N);let o=0;parts.forEach(p=>{X.set(p.flat,o*D);Y.set(p.Y,o);o+=p.Y.length;});return {X,Y,N};}
function round(a,s){return Array.from(a,v=>+v.toPrecision(s));}
function save(W,file){
  const o={mu:round(W.mu,5),sd:round(W.sd,5),W1:round(W.W1,5),b1:round(W.b1,5),W2:round(W.W2,5),b2:round(W.b2,5),W3:round(W.W3,5),b3:round(W.b3,5)};
  fs.writeFileSync(file,'// neural event detector weights (calm, stumble, falling, moved, down); generated by tools/det_train.js\nvar RS_DETW='+JSON.stringify(o)+';\n');
  return o;
}

(async function main(){
  const EP=+process.argv[2]||700,EPOCHS=+process.argv[3]||25,RS=loadRS(),D=RS.DIN,t0=Date.now();
  const seeds=k=>Array.from({length:EP},(_,i)=>100000*k+i*7+1);
  console.log('round 1: collecting',EP,'episodes (reactions off)');
  let A=merge(await collect(seeds(1),false,null),D);
  let B=merge(await collect(seeds(9).slice(0,Math.round(EP/4)),false,null),D); // held out
  console.log('  samples',A.N,'held-out',B.N,((Date.now()-t0)/1000).toFixed(0)+'s');
  let W=train(A.X,A.Y,A.N,D,EPOCHS);
  evaluate(RS,W,B.X,B.Y,B.N,D);
  let Wj=save(W,path.join(ROOT,'det_weights.js'));
  console.log('round 2: collecting with the detector driving the reactions');
  const C=merge(await collect(seeds(2),true,Wj),D);
  const B2=merge(await collect(seeds(19).slice(0,Math.round(EP/4)),true,Wj),D);
  const AB={X:new Float32Array((A.N+C.N)*D),Y:new Int8Array(A.N+C.N),N:A.N+C.N};AB.X.set(A.X);AB.X.set(C.X,A.N*D);AB.Y.set(A.Y);AB.Y.set(C.Y,A.N);
  W=train(AB.X,AB.Y,AB.N,D,EPOCHS);
  console.log(' held-out, reactions off:');evaluate(RS,W,B.X,B.Y,B.N,D);
  console.log(' held-out, reactions on:');evaluate(RS,W,B2.X,B2.Y,B2.N,D);
  save(W,path.join(ROOT,'det_weights.js'));
  console.log('wrote det_weights.js',((Date.now()-t0)/1000).toFixed(0)+'s');
})();
