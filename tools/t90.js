const fs=require('fs');const planck=require('planck');const wt=require('worker_threads');
const src=fs.readFileSync(''+__dirname+'/../sim.js','utf8');
if(wt.isMainThread){
  const RS=new Function('planck',src+';return RS;')(planck);
  const th=RS.initParams(3);for(let i=0;i<th.length;i++)th[i]+=0.02*Math.sin(i);
  const mk=f=>({only:'push',pushMax:1.5,flags:f});
  const on={step:1,air:1,land:1,fall:1},off={step:0,air:0,land:0,fall:0};
  const a=RS.evalCandidate(th,77,3,mk(on)),b=RS.evalCandidate(th,77,3,mk(off)),a2=RS.evalCandidate(th,77,3,mk(on));
  console.log('reflexes ON',a.toFixed(3),'| ON again',a2.toFixed(3),'| OFF',b.toFixed(3),'-> flags change the episode:',a!==b,'deterministic:',a===a2);
  const w=new wt.Worker(__filename,{workerData:{th:Array.from(th)}});
  w.on('message',m=>{console.log('worker (reflexes ON)',m.toFixed(3),'matches main thread:',m===a);process.exit(0);});
}else{
  const planckW=require('planck');const RS=new Function('planck',src+';return RS;')(planckW);
  const th=Float64Array.from(wt.workerData.th);
  wt.parentPort.postMessage(RS.evalCandidate(th,77,3,{only:'push',pushMax:1.5,flags:{step:1,air:1,land:1,fall:1}}));
}
