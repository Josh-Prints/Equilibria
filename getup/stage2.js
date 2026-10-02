const fs=require('fs');const planck=require('planck');const {Vec2,Box}=planck;
const G=require('./getup.js');const RS=G.RS;const NJ=15,JT=RS.JT,NCH=9,NK=+process.env.NK2||5,NPAR=NK*NCH+NK;
const stage1=G.decode(Float64Array.from(JSON.parse(fs.readFileSync(''+__dirname+'/getup_situp_solution.json','utf8')).x));
const T1=stage1.dur.slice(0,-1).reduce((a,b)=>a+b,0)+1.0; // time stage 1 is played (plus 1 s to settle into sitting)
function expand(ch){const t=new Float64Array(NJ);for(let c=0;c<NCH;c++){const j=JT[c],v=Math.max(j.lo,Math.min(j.hi,ch[c]));t[c]=v;if(c>=3)t[c+6]=v;}return t;}
function decode2(x){const kf=[stage1.kf[stage1.kf.length-1]],dur=[];
  for(let k=0;k<NK;k++){kf.push(expand(x.slice(k*NCH,(k+1)*NCH)));dur.push(Math.max(0.12,Math.min(1.5,x[NK*NCH+k])));}
  return {kf,dur};}
function tgt2(seq,t){let k=0,acc=0;while(k<seq.dur.length-1&&t>=acc+seq.dur[k]){acc+=seq.dur[k];k++;}
  if(k>=seq.dur.length-1&&t>=acc+seq.dur[seq.dur.length-1])return seq.kf[seq.kf.length-1];
  const u=Math.min(1,(t-acc)/seq.dur[k]),s=u*u*(3-2*u),a=seq.kf[k],b=seq.kf[k+1],o=new Float64Array(NJ);
  for(let j=0;j<NJ;j++)o[j]=a[j]+(b[j]-a[j])*s;return o;}
function episode(x,opt){
  opt=opt||{};const s2=decode2(x),w=new planck.World({gravity:Vec2(0,-10)}),gd=w.createBody();gd.createFixture(Box(1000,1,Vec2(0,-1),0),{friction:0.9});
  const rig=RS.buildRig(w,0,0,-1,{rootAng:1.4,clear:0.01}),T=opt.T||14,hz=120;let f=0,n=0,maxhn=0,trace=[];
  for(let i=0;i<T*hz;i++){
    const t=i/hz,tg=t<T1?G.decode2?null:null:null;
    const tt=t<T1?(function(){let k=0,acc=0;const q=stage1;while(k<q.kf.length-1&&t>=acc+q.dur[k]){acc+=q.dur[k];k++;}if(k>=q.kf.length-1)return q.kf[q.kf.length-1];const u=Math.min(1,(t-acc)/q.dur[k]),s=u*u*(3-2*u),a=q.kf[k],b=q.kf[k+1],o=new Float64Array(NJ);for(let j=0;j<NJ;j++)o[j]=a[j]+(b[j]-a[j])*s;return o;})():tgt2(s2,t-T1);
    for(let j=0;j<NJ;j++)rig.ctrls[j].target=tt[j];
    for(let s=0;s<2;s++){RS.pdRig(rig);w.step(1/240,6,2);}
    const hy=rig.parts[0].getPosition().y,ca=RS.wrap(rig.parts[1].getAngle());
    if(!(hy===hy)||hy>20)return -1;
    const hn=Math.max(0,Math.min(1,hy/1.735));if(t>T1)maxhn=Math.max(maxhn,hn);
    if(t>=T-4){f+=0.5*hn*hn+0.3*Math.max(0,Math.cos(ca))+0.2*Math.exp(-8*ca*ca);n++;}
    if(opt.trace&&i%60===0)trace.push([t.toFixed(1),hy.toFixed(2),ca.toFixed(2)].join(' '));
  }
  if(opt.trace)console.log(trace.join(' | '));
  return f/n;
}
function guess2(){ // from sitting with straight legs: tuck feet in, lean forward over them, rise, stand
  const k=[[0,-0.3,-0.2,1.2,0,0,1.8,-2.3,0.3],[0,-0.5,-0.3,1.5,0,0,2.0,-2.2,0.5],[0,-0.4,-0.2,1.2,0,0,1.5,-1.9,0.5],[0,-0.2,-0.1,0.8,0,0,0.9,-1.0,0.2],[0,0,0,0,0,0,0,0,0]];
  const x=new Float64Array(NPAR);for(let q=0;q<NK;q++){for(let c=0;c<NCH;c++)x[q*NCH+c]=k[Math.min(q,4)][c];x[NK*NCH+q]=0.6;}return x;}
module.exports={episode,guess2,NPAR,decode2,T1,G,RS,NK};
if(require.main===module){
  const x=guess2();console.log('T1',T1.toFixed(2),'fitness',episode(x,{trace:1}).toFixed(3));
}
