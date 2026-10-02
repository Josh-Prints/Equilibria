const fs=require('fs');const planck=require('planck');
const path=process.argv[2]||''+__dirname+'/../sim.js';
const RS=new Function('planck',fs.readFileSync(path,'utf8')+';return RS;')(planck);
const zero=new Float64Array(RS.NP);
const N=+process.argv[3]||14;const o={};const t0=Date.now();
for(const c of ['stand','push','drop','fallen']){let s=0,mn=1e9;for(let q=0;q<N;q++){const r=RS.runEpisode(zero,9000+q*13,{only:c,pushMax:1.5});s+=r;mn=Math.min(mn,r);}o[c]=(s/N).toFixed(0)+' (worst '+mn.toFixed(0)+')';}
console.log(JSON.stringify(o),((Date.now()-t0)/1000).toFixed(0)+'s');
