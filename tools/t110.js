const fs=require('fs');const planck=require('planck');const {Vec2,Box}=planck;
const base=fs.readFileSync(''+__dirname+'/../sim.js','utf8');
function load(reps){let s=base;for(const [a,b] of reps){if(!s.includes(a))throw new Error('missing '+a);s=s.split(a).join(b);}return new Function('planck',s+';return RS;')(planck);}
function maxPush(RS,flags,dir){
  const zero=new Float64Array(RS.NP);
  function trial(J){
    const w=new planck.World({gravity:Vec2(0,-10)});const g=w.createBody();g.createFixture(Box(1000,1,Vec2(0,-1),0),{friction:0.9});
    const r=RS.buildRig(w,0,0,-1,{});const b=RS.newBuf(),q=RS.rngMake(5);Object.assign(RS.FLAGS,flags);
    for(let i=0;i<7*120;i++){if(i===120)r.parts[1].applyLinearImpulse(Vec2(dir*J,0),r.parts[1].getWorldCenter(),true);RS.act(r,zero,b,q,{reflex:true});for(let s=0;s<2;s++){RS.pdRig(r);w.step(1/240,RS.VEL_IT,RS.POS_IT);}}
    return r.parts[0].getPosition().y>1.45&&Math.abs(RS.wrap(r.parts[1].getAngle()))<0.5;
  }
  let lo=0,hi=4;if(trial(hi))return hi;
  for(let i=0;i<7;i++){const m=(lo+hi)/2;if(trial(m))lo=m;else hi=m;}return lo;
}
const off={step:0,air:0,land:0,fall:0},all={step:1,air:1,land:1,fall:1};
function row(label,reps){
  const RS=load(reps);
  console.log(label.padEnd(34),'balance only: fwd',maxPush(RS,off,1).toFixed(2),'back',maxPush(RS,off,-1).toFixed(2),'| all reflexes: fwd',maxPush(RS,all,1).toFixed(2),'back',maxPush(RS,all,-1).toFixed(2));
}
row('current',[]);
row('TMAX 2',[["TMAX=1,","TMAX=2,"]]);
row('TMAX 3',[["TMAX=1,","TMAX=3,"]]);
row('LIMF .45',[["LIMF=0.35","LIMF=0.45"]]);
row('LIMF .45 + TMAX 2',[["LIMF=0.35","LIMF=0.45"],["TMAX=1,","TMAX=2,"]]);
