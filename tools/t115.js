const fs=require('fs');const planck=require('planck');const {Vec2,Box}=planck;
const base=fs.readFileSync(''+__dirname+'/../sim.js','utf8').replace("VEL_IT=6,POS_IT=2","VEL_IT=8,POS_IT=3");
const RS=new Function('planck',base+';return RS;')(planck);
const zero=new Float64Array(RS.NP);
function maxPush(dir,flags){
  function trial(J){
    const w=new planck.World({gravity:Vec2(0,-10)});const g=w.createBody();g.createFixture(Box(1000,1,Vec2(0,-1),0),{friction:0.9});
    const r=RS.buildRig(w,0,0,-1,{});const b=RS.newBuf(),q=RS.rngMake(5);Object.assign(RS.FLAGS,flags);
    for(let i=0;i<7*120;i++){if(i===120)r.parts[1].applyLinearImpulse(Vec2(dir*J,0),r.parts[1].getWorldCenter(),true);RS.act(r,zero,b,q,{reflex:true});for(let s=0;s<2;s++){RS.pdRig(r);w.step(1/240,RS.VEL_IT,RS.POS_IT);}}
    return r.parts[0].getPosition().y>1.45&&Math.abs(RS.wrap(r.parts[1].getAngle()))<0.5;
  }
  let lo=0,hi=4;if(trial(hi))return hi;for(let i=0;i<6;i++){const m=(lo+hi)/2;if(trial(m))lo=m;else hi=m;}return lo;
}
const res=[];
for(const trig of [0.0,0.06,0.12])for(const Ts of [0.16,0.22])for(const land_off of [0.0,0.1]){
  const fl={step:1,air:0,land:0,fall:0,trig,Ts,land_off,clear:0.07};
  const f=maxPush(1,fl),b=maxPush(-1,fl);res.push([f+b,trig,Ts,land_off,f,b]);
  console.log('trig',trig,'Ts',Ts,'off',land_off,'=> fwd',f.toFixed(2),'back',b.toFixed(2));
}
res.sort((a,b)=>b[0]-a[0]);console.log('best:',JSON.stringify(res[0]));
