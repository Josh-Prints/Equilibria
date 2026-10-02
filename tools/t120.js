// twitch/smoothness metrics per scenario: node tools/t120.js [policy.json] [N]
// env FL='{"soft":1}' adds flags (step/air/land/fall are on). Lower jointAcc / targetRate = smoother. Old policy files are converted automatically.
const fs=require('fs');const planck=require('planck'),V=planck.Vec2;
const RS=new Function('planck',fs.readFileSync(__dirname+'/../sim.js','utf8')+';return RS;')(planck);
const pf=process.argv[2],N=+process.argv[3]||8;
const theta=pf?RS.migrate(JSON.parse(fs.readFileSync(pf,'utf8')).theta):new Float64Array(RS.NP);
const FL=Object.assign({step:1,air:1,land:1,fall:1},JSON.parse(process.env.FL||'{}'));
function ep(seed,type){
  Object.assign(RS.FLAGS,FL);
  const r=RS.rngMake(seed),sc=RS.sampleScenario(r,{only:type,pushMax:1.5}),rr=RS.rngMake(seed^0x9e3779b9);
  const w=new planck.World({gravity:V(0,-10)});w.createBody().createFixture(planck.Box(1000,1,V(0,-1),0),{friction:sc.fric});
  const rig=RS.buildRig(w,0,0,-1,{pose:sc.pose,rootAng:sc.rootAng,h:sc.h,mass:sc.mass,fric:sc.fric,gain:sc.gain,partVar:sc.partVar,clear:sc.clear});
  if(sc.vel)rig.parts.forEach(b=>{b.setLinearVelocity(V(sc.vel.vx,sc.vel.vy));b.setAngularVelocity(sc.vel.w);});
  const buf=RS.newBuf(),hz=RS.POLICY_HZ;let pi=0,acc=0,acc1=0,n1=0,dT=0,up=0,pw=null,pt=null,n=0;
  for(let i=0;i<sc.T*hz;i++){const t=i/hz;
    while(pi<sc.pushes.length&&t>=sc.pushes[pi].t){const p=sc.pushes[pi++],b=rig.parts[p.body];b.applyLinearImpulse(V(p.jx,p.jy),b.getWorldCenter(),true);}
    RS.act(rig,theta,buf,rr,{delay:sc.delay,noise:sc.noise});
    for(let s=0;s<RS.SUB;s++){RS.pdRig(rig);w.step(RS.DT,RS.VEL_IT,RS.POS_IT);}
    const ws=rig.ctrls.map(c=>c.j.getJointSpeed()),tg=Array.from(rig.tgt);
    if(pw){let a=0,d=0;for(let k=0;k<RS.NJ;k++){a+=Math.abs(ws[k]-pw[k])*hz;d+=Math.abs(tg[k]-pt[k])*hz;}a/=RS.NJ;d/=RS.NJ;acc+=a;dT+=d;n++;if(t<1.5){acc1+=a;n1++;}}
    pw=ws;pt=tg;
    const hy=rig.parts[0].getPosition().y,ca=RS.wrap(rig.parts[1].getAngle());if(Math.cos(ca)>0.85&&hy>1.45)up++;
  }
  return {acc:acc/n,acc1:acc1/Math.max(1,n1),dT:dT/n,up:up/(sc.T*hz)};
}
console.log('scenario  jointAcc (rad/s²)  first 1.5 s  targetRate (rad/s)  upright%');
for(const c of RS.CASES){const R=[];for(let s=0;s<N;s++)R.push(ep(9000+s*13,c));
  const m=k=>R.reduce((a,x)=>a+x[k],0)/R.length;
  console.log(c.padEnd(9),m('acc').toFixed(2).padStart(10),m('acc1').toFixed(2).padStart(14),m('dT').toFixed(3).padStart(14),(100*m('up')).toFixed(0).padStart(12));}
