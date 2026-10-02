// Optimise a keyframe get-up sequence (low-dimensional ES) on the lying-down scenario.
const fs=require('fs');const planck=require('planck');const {Vec2,Box}=planck;
const RS=new Function('planck',fs.readFileSync(''+__dirname+'/../sim.js','utf8')+';return RS;')(planck);
const NJ=15,JT=RS.JT;
// 9 "channels" tied to both sides: neck, upper spine, lower spine, shoulder, elbow, wrist, hip, knee, ankle
const CH=[0,1,2,3,4,5,6,7,8];
const NKF=+process.env.NKF||6,NCH=9,NPAR=NKF*NCH+NKF;
const SIDE=+process.env.SIDE||1; // +1 = lying on back (head towards -x), -1 = on front
function limits(ch){const j=JT[ch];return [j.lo,j.hi];}
function decode(x){
  const kf=[],dur=[];
  for(let k=0;k<NKF;k++){
    const t=new Float64Array(NJ);
    for(let c=0;c<NCH;c++){
      const [lo,hi]=limits(c),v=Math.max(lo,Math.min(hi,x[k*NCH+c]));
      t[c]=v;if(c>=3){t[c+6]=v;} // second limb set (indices 9..14) mirrors 3..8
    }
    kf.push(t);dur.push(Math.max(0.12,Math.min(1.5,x[NKF*NCH+k])));
  }
  return {kf,dur};
}
function targetAt(seq,t){
  let k=0,acc=0;
  while(k<NKF-1&&t>=acc+seq.dur[k]){acc+=seq.dur[k];k++;}
  if(k>=NKF-1)return seq.kf[NKF-1];
  const u=Math.min(1,(t-acc)/seq.dur[k]),s=u*u*(3-2*u),a=seq.kf[k],b=seq.kf[k+1],o=new Float64Array(NJ);
  for(let j=0;j<NJ;j++)o[j]=a[j]+(b[j]-a[j])*s;
  return o;
}
function episode(x,opt){
  opt=opt||{};
  const seq=decode(x),w=new planck.World({gravity:Vec2(0,-10)}),gd=w.createBody();gd.createFixture(Box(1000,1,Vec2(0,-1),0),{friction:0.9});
  const rig=RS.buildRig(w,0,0,-1,{rootAng:SIDE*(opt.root||1.4),clear:0.01});
  const T=opt.T||9,hz=120,steps=T*hz;let f=0,n=0,trace=[];
  for(let i=0;i<steps;i++){
    const t=i/hz,tg=targetAt(seq,t);
    for(let j=0;j<NJ;j++)rig.ctrls[j].target=tg[j];
    for(let s=0;s<2;s++){RS.pdRig(rig);w.step(1/240,6,2);}
    const hy=rig.parts[0].getPosition().y,ca=RS.wrap(rig.parts[1].getAngle());
    if(!(hy===hy)||hy>20)return -1;
    if(t>=T-2){const hn=Math.max(0,Math.min(1,hy/1.735));f+=0.5*hn*hn+0.3*Math.max(0,Math.cos(ca))+0.2*Math.exp(-8*ca*ca);n++;}
    if(opt.trace&&i%60===0)trace.push([t.toFixed(1),hy.toFixed(2),ca.toFixed(2)].join(' '));
  }
  if(opt.trace)console.log(trace.join(' | '));
  return f/n;
}
// handcrafted starting guess (supine: curl up, feet in, rock forward, stand)
function guess(){
  const k=[
    // neck spU spL  sh   el   wr  hip  knee ank
    [0,    0,   0,   0,   0,   0,  0,   0,   0],      // lying
    [0,    0,   0,   0.5, 0,   0,  1.8, -2.2,0],      // pull knees in
    [-0.3,-0.5,-0.4, 1.4, 0,   0,  1.8, -2.0,0.2],    // sit up, arms forward
    [-0.3,-0.5,-0.4, 1.4, 0,   0,  1.3, -1.8,0.3],    // lean over feet
    [0,   -0.2,-0.2, 1.0, 0,   0,  0.6, -0.9,0.1],    // rise
    [0,    0,   0,   0,   0,   0,  0,   0,   0]];     // stand
  const x=new Float64Array(NPAR);
  for(let q=0;q<NKF;q++){for(let c=0;c<NCH;c++)x[q*NCH+c]=k[Math.min(q,k.length-1)][c];x[NKF*NCH+q]=0.6;}
  return x;
}
module.exports={episode,guess,decode,NPAR,NKF,NCH,RS};
if(require.main===module){
  const x=guess();const t0=Date.now();
  console.log('guess fitness',episode(x,{trace:1}).toFixed(3),'time',Date.now()-t0,'ms; params',NPAR);
}
