const fs=require('fs');const planck=require('planck');
const RS=new Function('planck',fs.readFileSync(''+__dirname+'/../sim.js','utf8')+';return RS;')(planck);
const zero=new Float64Array(RS.NP);const N=+process.argv[2]||12;
function ev(label,flags){
  const o={};for(const c of ['stand','push','drop','fallen']){let s=0;for(let q=0;q<N;q++)s+=RS.runEpisode(zero,9000+q*13,{only:c,pushMax:1.5,flags:Object.assign({air:0,land:0,fall:0,step:0},flags)});o[c]=(s/N).toFixed(0);}
  console.log(label.padEnd(26),JSON.stringify(o));
}
ev('balance only',{});
ev('+step',{step:1});
ev('+air',{air:1});
ev('+land',{land:1});
ev('+fall',{fall:1});
ev('all (what you trained)',{step:1,air:1,land:1,fall:1});
