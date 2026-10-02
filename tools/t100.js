// headless ES trainer: node t100.js name mode gens P K sigma lr
const fs=require('fs');const planck=require('planck');
const RS=new Function('planck',fs.readFileSync(''+__dirname+'/../sim.js','utf8')+';return RS;')(planck);
const [,, name='run',mode='full',gensS='100',PS='12',KS='3',sigS='0.01',lrS='0.004']=process.argv;
const GENS=+gensS,P=+PS,K=+KS,sigma=+sigS,lr=+lrS,NP=RS.NP;
const FLAGS={step:1,air:1,land:1,fall:1};
// parameter layout (must match sim.js): W1[NH*NI], b1[NH], W2[NH*NH], b2[NH], W3[NO*NH], b3[NO]
const NI=RS.NI,NH=40,NO=RS.NO,B1=NH*NI,W2=B1+NH,B2=W2+NH*NH,W3=B2+NH,B3=W3+NO*NH;
const mask=new Float64Array(NP).fill(mode==='full'?1:0);
if(mode==='out'){for(let i=W3;i<NP;i++)mask[i]=1;}
if(mode==='last2'){for(let i=W2;i<NP;i++)mask[i]=1;}
const S={theta:RS.initParams(7),m:new Float64Array(NP),v:new Float64Array(NP),t:0};
const zero=new Float64Array(NP);
function heldout(th,n){const o={};let tot=0;for(const c of RS.CASES){let s=0;for(let q=0;q<n;q++)s+=RS.runEpisode(th,9000+q*13,{only:c,pushMax:1.5,flags:FLAGS});o[c]=s/n;tot+=s/n;}o.total=tot;return o;}
const fmt=o=>Object.entries(o).map(([k,v])=>k+' '+v.toFixed(0)).join(' · ');
const HN=+process.env.HN||10;const base=heldout(zero,HN);console.log(name,mode,'base:',fmt(base));
const t0=Date.now();
for(let g=0;g<GENS;g++){
  const seed=1000+g*7919,eps=RS.esNoise(P,NP,seed+1),Fp=[],Fm=[],th=new Float64Array(NP);
  for(const e of eps)for(let k=0;k<NP;k++)e[k]*=mask[k];
  const cfg={pushMax:Math.min(1.5,0.7+0.02*g),flags:FLAGS};
  for(let i=0;i<P;i++){
    for(let k=0;k<NP;k++)th[k]=S.theta[k]+sigma*eps[i][k];Fp.push(RS.evalCandidate(th,seed,K,cfg));
    for(let k=0;k<NP;k++)th[k]=S.theta[k]-sigma*eps[i][k];Fm.push(RS.evalCandidate(th,seed,K,cfg));
  }
  RS.esStep(S,eps,Fp,Fm,sigma,lr);
  if((g+1)%10===0){
    const h=heldout(S.theta,HN);
    console.log('gen',g+1,((Date.now()-t0)/1000).toFixed(0)+'s','| heldout',fmt(h),'| vs base total',(h.total-base.total).toFixed(0));
    fs.writeFileSync('/tmp/'+name+'.json',JSON.stringify({format:'ragdoll-policy',version:2,np:NP,gen:g+1,theta:Array.from(S.theta),hist:[]}));
  }
}
