const fs=require('fs');const planck=require('planck');
const src=fs.readFileSync(''+__dirname+'/../sim.js','utf8');
const RS=new Function('planck',src+';return RS;')(planck);
module.exports=RS;
if(require.main===module){
 console.log('NP',RS.NP);
 const th=new Float64Array(RS.NP);
 for(const only of ['stand','push','drop','fallen']){
  const t0=Date.now();const fs_=[];
  for(let s=1;s<=6;s++)fs_.push(RS.runEpisode(th,s*17,{only,pushMax:1.5}));
  console.log(only.padEnd(7),'zero-policy rewards',fs_.map(x=>x.toFixed(0)).join(' '),'| avg ms/episode',((Date.now()-t0)/6).toFixed(0));
 }
}
