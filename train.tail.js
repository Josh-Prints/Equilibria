

// ---------------------------------------------------------------------------------------------
if(!isMainThread){
  parentPort.on('message',function(d){
    parentPort.postMessage({id:d.id,f:RS.evalCandidate(d.theta,d.seed,d.K,d.cfg)});
  });
}else{
  main().catch(function(e){console.error(e);process.exit(1);});
}

async function main(){
  const env=process.env,num=function(k,d){const v=parseFloat(env[k]);return isNaN(v)?d:v;};
  const MINUTES=Math.min(num('MINUTES',330),340),GENS=num('GENS',1e9),P=Math.round(num('PAIRS',16)),K=Math.round(num('SCENARIOS',3));
  const SIGMA=num('SIGMA',0.01),LR=num('LR',0.004),EVAL_EVERY=Math.round(num('EVAL_EVERY',20)),NP=RS.NP;
  const MODE=env.MODE||'full';                      // full = train every weight, out = output layer only
  const NW=Math.max(1,Math.round(num('WORKERS',os.cpus().length)));
  const FLAGS={step:1,air:1,land:1,fall:1,bal2:env.BAL2==='false'?0:1, // all reflexes on, like the page default; bal2 = new balance + recovery steps
    smooth:env.SMOOTH==='false'?0:1,soft:env.SOFT==='false'?0:1}; // smooth/human-like reward and soft start (on unless set to false)
  const outDir=env.OUT||'out';fs.mkdirSync(outDir,{recursive:true});
  const logFile=outDir+'/log.txt';
  function log(s){const line=new Date().toISOString().slice(11,19)+'  '+s;console.log(line);fs.appendFileSync(logFile,line+'\n');}

  // parameter layout: W1[NH*NI] b1[NH] W2[NH*NH] b2[NH] W3[NO*NH] b3[NO]
  const NH=40,W3=NH*RS.NI+NH+NH*NH+NH;
  const mask=new Float64Array(NP).fill(MODE==='out'?0:1);
  if(MODE==='out')for(let i=W3;i<NP;i++)mask[i]=1;

  const S={theta:RS.initParams(7),m:new Float64Array(NP),v:new Float64Array(NP),t:0};
  let gen0=0;
  if(env.RESUME==='true'&&fs.existsSync('resume.json')){
    const d=JSON.parse(fs.readFileSync('resume.json','utf8'));
    const th=RS.migrate(d.theta);
    if(th){S.theta=th;gen0=d.gen||0;log('resumed from resume.json at generation '+gen0+(d.theta.length!==NP?' (converted from the old network layout)':''));}
    else log('resume.json has the wrong size, starting fresh');
  }
  if(num('MINUTES',330)>340)log('MINUTES capped at 340: GitHub stops a job at 6 hours, so longer runs would lose their results');
  log('cpus='+os.cpus().length+' workers='+NW+' pairs='+P+' scenarios='+K+' sigma='+SIGMA+' lr='+LR+' mode='+MODE+' bal2='+FLAGS.bal2+' smooth='+FLAGS.smooth+' soft='+FLAGS.soft+' minutes='+MINUTES);

  const pool=[];for(let i=0;i<NW;i++)pool.push(new Worker(__filename));
  function runJobs(jobs){
    return new Promise(function(resolve){
      const out=new Array(jobs.length);let next=0,done=0;
      pool.forEach(function(w){
        (function feed(){
          if(next>=jobs.length)return;
          const i=next++;
          w.once('message',function(m){out[m.id]=m.f;done++;if(done===jobs.length)resolve(out);else feed();});
          w.postMessage({id:i,theta:jobs[i].theta,seed:jobs[i].seed,K:jobs[i].K,cfg:jobs[i].cfg});
        })();
      });
    });
  }
  function save(file,gen){
    fs.writeFileSync(outDir+'/'+file,JSON.stringify({format:'ragdoll-policy',version:3,np:NP,gen:gen,theta:Array.prototype.slice.call(S.theta),hist:[]}));
  }
  const zero=new Float64Array(NP),N=8;
  async function heldout(theta){
    const jobs=RS.CASES.map(function(c){return {theta:theta,seed:9000,K:N,cfg:{only:c,pushMax:1.5,flags:FLAGS}};});
    const r=await runJobs(jobs);let tot=0;r.forEach(function(x){tot+=x;});
    return {per:r,total:tot};
  }
  const fmt=function(h){return RS.CASES.map(function(c,i){return c+' '+h.per[i].toFixed(0);}).join(' · ')+'  | total '+h.total.toFixed(0);};

  const base=await heldout(zero);
  log('baseline (reflexes, no network): '+fmt(base));
  let best=-1e18,lastGen=gen0;
  const t0=Date.now();
  for(let g=0;g<GENS;g++){
    const gen=gen0+g,seed=1000+gen*7919,eps=RS.esNoise(P,NP,seed+1),Fp=[],Fm=[];
    eps.forEach(function(e){for(let k=0;k<NP;k++)e[k]*=mask[k];});
    const cfg={pushMax:Math.min(1.5,0.7+0.02*gen),flags:FLAGS},jobs=[];
    for(let i=0;i<P;i++){
      const tp=new Float64Array(NP),tm=new Float64Array(NP);
      for(let k=0;k<NP;k++){tp[k]=S.theta[k]+SIGMA*eps[i][k];tm[k]=S.theta[k]-SIGMA*eps[i][k];}
      jobs.push({theta:tp,seed:seed,K:K,cfg:cfg});jobs.push({theta:tm,seed:seed,K:K,cfg:cfg});
    }
    const F=await runJobs(jobs);
    for(let i=0;i<P;i++){Fp.push(F[2*i]);Fm.push(F[2*i+1]);}
    const st=RS.esStep(S,eps,Fp,Fm,SIGMA,LR);
    const done=g+1,el=(Date.now()-t0)/1000;lastGen=gen+1;
    if(done%10===0)log('gen '+(gen+1)+'  '+(el/done).toFixed(2)+' s/gen  mean '+st.mean.toFixed(0)+' best '+st.best.toFixed(0));
    if(done%EVAL_EVERY===0){
      const h=await heldout(S.theta);
      const tag=h.total>best;if(tag){best=h.total;save('best.json',gen+1);}
      log('TEST gen '+(gen+1)+': '+fmt(h)+'  (baseline total '+base.total.toFixed(0)+')'+(tag?'  <- best so far, saved':''));
      save('policy.json',gen+1);
    }
    if(el>MINUTES*60){log('time limit reached');break;}
  }
  save('policy.json',lastGen);
  pool.forEach(function(w){w.terminate();});
}
