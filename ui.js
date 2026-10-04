(function(){
  var hint=document.getElementById('hint');
  window.onerror=function(m){hint.textContent='Error: '+m;};
  if(typeof planck==='undefined'||typeof RS==='undefined'){hint.textContent='Planck failed to load.';return;}
  var Vec2=planck.Vec2,Box=planck.Box,cv=document.getElementById('c'),ctx=cv.getContext('2d');
  var world,ground,group=0,drag=null,W=0,H=0,dpr=1,rigs=[],stepCount=0,liveRng=RS.rngMake(12345);
  var cam={x:0.2,y:0,z:110};
  var DT=RS.DT,SUB=RS.SUB,NP=RS.NP;
  function opt(id){return document.getElementById(id).checked;}
  function num(id,d){var v=parseFloat(document.getElementById(id).value);return isNaN(v)?d:v;}

  function resize(){
    dpr=window.devicePixelRatio||1;W=cv.clientWidth;H=cv.clientHeight;
    cv.width=Math.round(W*dpr);cv.height=Math.round(H*dpr);
  }
  addEventListener('resize',resize);resize();
  cam.y=0.2*H/cam.z+0.1;
  function toWorld(sx,sy){return Vec2(cam.x+(sx-W/2)/cam.z,cam.y-(sy-H/2)/cam.z);}
  function toScreen(p){return {x:W/2+(p.x-cam.x)*cam.z,y:H/2-(p.y-cam.y)*cam.z};}
  function setZoom(z,sx,sy){
    var w=toWorld(sx,sy);cam.z=Math.max(15,Math.min(600,z));
    cam.x=w.x-(sx-W/2)/cam.z;cam.y=w.y+(sy-H/2)/cam.z;
  }

  // =====================================================================
  // TRAINING: evolution strategies, candidates evaluated in parallel Web Workers (main-thread fallback)
  // =====================================================================
  var ES={theta:RS.initParams(7),m:new Float64Array(NP),v:new Float64Array(NP),t:0,gen:0,hist:[],sigma:0.01,lr:0.004,pushMax:1.5,done:0,total:0};
  var ZERO_TH=new Float64Array(NP); // untrained policy (all outputs 0): standing pose + reflexes
  function esReset(){ES.theta=RS.initParams(7);ES.m=new Float64Array(NP);ES.v=new Float64Array(NP);ES.t=0;ES.gen=0;ES.hist=[];ES.done=0;ES.total=0;}

  var pool={workers:[],ok:0,mode:'main',queue:[],pending:0};
  function workerSource(){
    var src=document.getElementById('simsrc').textContent;
    return "importScripts('https://cdn.jsdelivr.net/npm/planck@1.0.0/dist/planck.min.js');\n"+src+
      "\nonmessage=function(e){var d=e.data;if(d.type==='ping'){postMessage({type:'pong'});return;}"+
      "var f=RS.evalCandidate(d.theta,d.seed,d.K,d.cfg);postMessage({type:'res',id:d.id,f:f});};";
  }
  function stopWorkers(){pool.workers.forEach(function(w){try{w.terminate();}catch(e){}});pool.workers=[];pool.ok=0;pool.mode='main';}
  function startWorkers(n){
    stopWorkers();
    if(n<=0||typeof Worker==='undefined'||typeof Blob==='undefined'||!window.URL||!URL.createObjectURL)return Promise.resolve(0);
    var url;
    try{url=URL.createObjectURL(new Blob([workerSource()],{type:'application/javascript'}));}catch(e){return Promise.resolve(0);}
    var waits=[];
    for(var i=0;i<n;i++){
      waits.push(new Promise(function(res){
        var w;try{w=new Worker(url);}catch(e){res(null);return;}
        var timer=setTimeout(function(){try{w.terminate();}catch(e){}res(null);},6000);
        w.onerror=function(){clearTimeout(timer);try{w.terminate();}catch(e){}res(null);};
        w.onmessage=function(ev){if(ev.data&&ev.data.type==='pong'){clearTimeout(timer);w.onerror=null;res(w);}};
        w.postMessage({type:'ping'});
      }));
    }
    return Promise.all(waits).then(function(ws){
      pool.workers=ws.filter(function(w){return w;});pool.ok=pool.workers.length;pool.mode=pool.ok?'workers':'main';
      return pool.ok;
    });
  }
  // run a list of jobs {theta,seed,K,cfg}; resolves with their fitness values in order
  function runJobs(jobs,onEach){
    return new Promise(function(resolve){
      var out=new Array(jobs.length),next=0,finished=0;
      function done(i,f){out[i]=f;finished++;if(onEach)onEach(finished,jobs.length);if(finished===jobs.length)resolve(out);}
      if(pool.workers.length){
        pool.workers.forEach(function(w){
          function feed(){
            if(next>=jobs.length)return;
            var i=next++,j=jobs[i];
            w.onmessage=function(ev){if(ev.data&&ev.data.type==='res'){done(i,ev.data.f);feed();}};
            w.postMessage({type:'job',id:i,theta:j.theta,seed:j.seed,K:j.K,cfg:j.cfg});
          }
          feed();
        });
      }else{
        (function step(){
          if(next>=jobs.length)return;
          var i=next++,j=jobs[i];
          done(i,RS.evalCandidate(j.theta,j.seed,j.K,j.cfg));
          setTimeout(step,0);
        })();
      }
    });
  }

  // training and Test use whichever reflexes are ticked (a policy learns to work together with them)
  function reflexCfg(){return {noReflex:!opt('oReflex'),flags:{nn:opt('oNN')?1:0,getup:opt('oNN')&&opt('oGetup')?1:0,cower:opt('oCower')?1:0,die:opt('oDie')?1:0,protect:opt('oProtect')?1:0,bal2:opt('oBal2')?1:0,step:opt('oStep')?1:0,air:opt('oFall')?1:0,land:opt('oFall')?1:0,fall:opt('oFall')?1:0,smooth:opt('oSmooth')?1:0,soft:opt('oSoft')?1:0}};}
  function reflexText(){return (opt('oReflex')?('balance'+(opt('oBal2')?' v2 + steps':(opt('oStep')?', step':''))+(opt('oFall')?', fall/landing':'')):'none')+(opt('oSmooth')?' · smooth reward':'')+(opt('oSoft')?' · soft start':'');}
  var training=false;
  function esGeneration(){
    var P=Math.max(2,Math.round(num('nPairs',12))),K=Math.max(1,Math.round(num('nScen',2)));
    var seed=1000+ES.gen*7919,eps=RS.esNoise(P,NP,seed+1),sg=ES.sigma,i,k;
    var cfg=reflexCfg();cfg.pushMax=Math.min(ES.pushMax,0.7+0.02*ES.gen);
    var jobs=[{theta:ES.theta.slice(),seed:seed,K:K,cfg:cfg}];
    for(i=0;i<P;i++){
      var tp=new Float64Array(NP),tm=new Float64Array(NP);
      for(k=0;k<NP;k++){tp[k]=ES.theta[k]+sg*eps[i][k];tm[k]=ES.theta[k]-sg*eps[i][k];}
      jobs.push({theta:tp,seed:seed,K:K,cfg:cfg});jobs.push({theta:tm,seed:seed,K:K,cfg:cfg});
    }
    ES.total=jobs.length;ES.done=0;
    return runJobs(jobs,function(d){ES.done=d;updateInfo();}).then(function(F){
      var Fp=[],Fm=[];for(i=0;i<P;i++){Fp.push(F[1+2*i]);Fm.push(F[2+2*i]);}
      var st=RS.esStep(ES,eps,Fp,Fm,sg,ES.lr);
      ES.gen++;ES.hist.push({cur:F[0],mean:st.mean,best:st.best});
    });
  }
  var info=document.getElementById('info'),chart=document.getElementById('chart'),cx=chart.getContext('2d');
  function updateInfo(){
    var h=ES.hist,l=h.length?h[h.length-1]:null;
    info.textContent='Generation '+ES.gen+(training?'  ('+ES.done+'/'+ES.total+')':'')+
      '\nrunning on: '+(pool.mode==='workers'?pool.ok+' worker threads':'main thread')+'\ntraining with reflexes: '+reflexText()+
      (l?'\nscore now '+l.cur.toFixed(0)+'  best '+l.best.toFixed(0):'')+(testText?'\n'+testText:'');
    cx.clearRect(0,0,chart.width,chart.height);
    if(!h.length)return;
    var lo=1e9,hi=-1e9;h.forEach(function(q){lo=Math.min(lo,q.cur,q.mean);hi=Math.max(hi,q.cur,q.mean);});
    if(hi-lo<1)hi=lo+1;
    var fg=getComputedStyle(document.body).color;
    cx.strokeStyle=fg;cx.globalAlpha=0.25;cx.lineWidth=1;cx.strokeRect(0.5,0.5,chart.width-1,chart.height-1);cx.globalAlpha=1;
    function line(key,color){
      cx.strokeStyle=color;cx.lineWidth=3;cx.beginPath();
      for(var i=0;i<h.length;i++){
        var x=h.length>1?i/(h.length-1)*(chart.width-8)+4:chart.width/2,y=chart.height-4-(h[i][key]-lo)/(hi-lo)*(chart.height-8);
        if(i===0)cx.moveTo(x,y);else cx.lineTo(x,y);
      }
      cx.stroke();
    }
    line('mean','#868e96');line('cur','#2f9e44');
  }
  var testText='';
  var bTrain=document.getElementById('bTrain');
  function setHint(t){hint.textContent=t;}
  bTrain.onclick=function(){
    if(training){training=false;bTrain.textContent='Start training';updateInfo();return;}
    training=true;bTrain.textContent='Stop training';setHint('Starting workers...');
    var want=Math.round(num('nWork',-1));
    if(want<0)want=Math.max(0,Math.min(8,(navigator.hardwareConcurrency||2)-1));
    (pool.ok&&pool.ok===want?Promise.resolve(pool.ok):startWorkers(want)).then(function(n){
      setHint(n?('Training on '+n+' worker threads.'):'Training on the main thread (workers unavailable or set to 0): the live view is paused.');
      updateInfo();
      (function loop(){
        if(!training){setHint('1 finger: drag ragdoll or pan · 2 fingers: zoom');return;}
        esGeneration().then(function(){updateInfo();setTimeout(loop,0);});
      })();
    });
  };
  document.getElementById('bResetP').onclick=function(){esReset();testText='';updateInfo();};
  document.getElementById('bTest').onclick=function(){
    if(training){setHint('Stop training first.');return;}
    setHint('Testing...');
    var go=function(){
      var jobs=[],cases=RS.CASES,zero=new Float64Array(NP);
      var rc=reflexCfg();
      function mk(c){return {only:c,pushMax:1.5,noReflex:rc.noReflex,flags:rc.flags};}
      cases.forEach(function(c){jobs.push({theta:ES.theta.slice(),seed:9000,K:6,cfg:mk(c)});});
      cases.forEach(function(c){jobs.push({theta:zero,seed:9000,K:6,cfg:mk(c)});});
      runJobs(jobs).then(function(F){
        var a=cases.map(function(c,i){return c+' '+F[i].toFixed(0)+' (base '+F[i+4].toFixed(0)+')';});
        testText='test ['+reflexText()+']: '+a.join(' · ');updateInfo();setHint('Done. "base" = the same reflexes with no network.');
      });
    };
    if(pool.ok)go();else startWorkers(Math.max(0,Math.min(8,(navigator.hardwareConcurrency||2)-1))).then(go);
  };

  // ---- save / load policy as a file ----
  function policyToText(){
    return JSON.stringify({format:'ragdoll-policy',version:3,np:NP,gen:ES.gen,theta:Array.prototype.slice.call(ES.theta),hist:ES.hist});
  }
  function loadFromText(text){
    var d=JSON.parse(text);
    var th=d&&RS.migrate(d.theta); // old files (4378 weights) are converted to the current network and behave the same
    if(!th)throw new Error('wrong file (expected '+NP+' weights, got '+(d&&d.theta?d.theta.length:'none')+')');
    ES.theta=th;ES.gen=d.gen||0;ES.hist=d.hist||[];
    ES.m=new Float64Array(NP);ES.v=new Float64Array(NP);ES.t=0;updateInfo();
  }
  document.getElementById('bSave').onclick=function(){
    var cl=window.claude;
    if(!cl||!cl.use){setHint('Saving files is not available here.');return;}
    cl.use('downloads').then(function(dl){
      if(!dl){setHint('Saving files is not available here.');return;}
      return dl.save({filename:'ragdoll-policy-gen'+ES.gen+'.json',data:policyToText()}).then(function(){setHint('Saved.');},
        function(e){setHint((e&&e.code==='declined')?'Save cancelled.':'Save failed: '+((e&&e.code)||e));});
    });
  };
  var fIn=document.getElementById('fIn');
  document.getElementById('bLoad').onclick=function(){fIn.value='';fIn.click();};
  fIn.onchange=function(){
    var f=fIn.files&&fIn.files[0];if(!f)return;
    var rd=new FileReader();
    rd.onload=function(){try{loadFromText(String(rd.result));setHint('Loaded '+f.name);}catch(e){setHint('Load failed: '+e.message);}};
    rd.onerror=function(){setHint('Could not read file.');};
    rd.readAsText(f);
  };

  // =====================================================================
  // LIVE WORLD
  // =====================================================================
  var ARENA=30,ROOF=20; // the arena runs from -30 m to +30 m, walls at each end, roof 20 m up
  function init(){
    world=new planck.World({gravity:Vec2(0,-10)});world.on('begin-contact',meleeContact);
    drag=null;group=0;rigs=[];parts=[];stains=[];humanN=0;camT=null;camFollow=null;items=[];fx=[];trig=null;
    ground=world.createBody();ground.createFixture(Box(ARENA+40,20,Vec2(0,-20),0),{friction:0.9}); // floor (20 m thick so nothing tunnels through it)
    ground.createFixture(Box(20,ROOF+40,Vec2(-ARENA-20,ROOF/2),0),{friction:0.6});ground.createFixture(Box(20,ROOF+40,Vec2(ARENA+20,ROOF/2),0),{friction:0.6}); // walls
    ground.createFixture(Box(ARENA+40,20,Vec2(0,ROOF+20),0),{friction:0.6}); // roof
    spawn('stand');
  }
  var liveBufs=[];
  // ITEMS: things you can spawn besides humans. Handgun: drag it around, tap it to fire along the barrel.
  var items=[],fx=[];
  function spawnGun(){
    var b=world.createBody({type:'dynamic',position:Vec2(cam.x,Math.max(0.6,cam.y+0.3)),angle:0,bullet:true});
    var g={type:'gun',b:b,flash:0,s:1};gunFix(g);b.setUserData({gun:true});items.push(g);
  }
  function gunFix(g){ // s=1 grip down when pointing right, s=-1 mirrored so it stays grip-down pointing left
    var b=g.b,s=g.s,f;while((f=b.getFixtureList()))b.destroyFixture(f);
    b.createFixture(Box(0.09,0.02,Vec2(0.015,0.012*s),0),{density:2.2,friction:0.6,restitution:0.1}); // slide and barrel
    b.createFixture(Box(0.02,0.042,Vec2(-0.045,-0.03*s),-0.25*s),{density:2.2,friction:0.8,restitution:0.1}); // grip
    b.createFixture(Box(0.012,0.006,Vec2(-0.008,-0.014*s),0),{density:1,friction:0.6}); // trigger guard
  }
  function itemOf(b){for(var i=0;i<items.length;i++)if(items[i].b===b)return items[i];return null;}
  function gunOf(b){var g=itemOf(b);return g&&g.type==='gun'?g:null;}
  function aimOf(b){var g=itemOf(b);return g&&(g.type==='gun'||g.type==='knife')?g:null;} // things you point with a second finger
  function aimAt(g,w){ // point the barrel at a world point, flipping the gun over when it swings past vertical
    var p=g.b.getPosition(),a=Math.atan2(w.y-p.y,w.x-p.x),s=Math.cos(a)<0?-1:1;
    if(s!==g.s){g.s=s;itemFix(g);}
    g.aim=a;
  }
  // MELEE: a sledgehammer and a knife you drag and swing. Contacts with people are queued in the physics callback
  // (the world is locked there) and dealt with right after the step.
  function spawnHammer(){
    var b=world.createBody({type:'dynamic',position:Vec2(cam.x,Math.max(0.6,cam.y+0.3)),angle:0,bullet:true});
    b.createFixture(Box(0.32,0.014,Vec2(0,0),0),{density:2,friction:0.7,restitution:0.05,userData:{part:'handle'}}); // ash handle
    b.createFixture(Box(0.045,0.085,Vec2(0.345,0),0),{density:14,friction:0.6,restitution:0.1,userData:{part:'head'}}); // steel head
    var g={type:'hammer',b:b,cd:{}};b.setUserData({item:'hammer'});items.push(g);
  }
  function spawnKnife(){
    var b=world.createBody({type:'dynamic',position:Vec2(cam.x,Math.max(0.6,cam.y+0.3)),angle:0,bullet:true});
    var g={type:'knife',b:b,cd:{},s:1};knifeFix(g);b.setUserData({item:'knife'});items.push(g);
  }
  function knifeFix(g){ // s=-1 mirrors it so the edge stays down when it points left
    var b=g.b,s=g.s,f,gi=g.stuck?g.stuck.gi:0;while((f=b.getFixtureList()))b.destroyFixture(f);
    b.createFixture(Box(0.05,0.014,Vec2(-0.05,0),0),{density:8,friction:0.8,filterGroupIndex:gi,userData:{part:'handle'}});
    b.createFixture(planck.Polygon([Vec2(0,-0.012*s),Vec2(0.13,-0.006*s),Vec2(0.16,0.012*s),Vec2(0,0.012*s)]),{density:20,friction:0.2,filterGroupIndex:gi,userData:{part:'blade'}}); // edge on the bottom, tip forward
  }
  function itemFix(g){if(g.type==='gun')gunFix(g);else if(g.type==='knife')knifeFix(g);}
  function setGroup(b,gi){for(var f=b.getFixtureList();f;f=f.getNext())f.setFilterData({groupIndex:gi,categoryBits:1,maskBits:0xFFFF});}
  var KBL=0.16,KHAND=1,KGRIP=20,KBRAIN=0.08; // blade length; how strong your hand is on a stuck knife (kg-equivalent); flesh grip when you let go; depth into the skull that's fatal
  function brainIn(S){if(!S.brain&&rigs.indexOf(S.r)>=0){S.brain=1;RS.brainHit(S.r,0,true);}}
  function unstick(g,yank){ // pulled out (or the part it was in is gone)
    var S=g.stuck;if(!S)return;g.stuck=null;try{world.destroyJoint(S.j);}catch(e){}
    var P=S.pend,R2=S.r,fq=Math.min(1,(S.dmax||KBL)/KBL);
    if(S.k===0&&!S.brain&&rigs.indexOf(R2)>=0)RS.brainHit(R2,0.08+0.3*fq,false); // shallow into the skull: damage, not death
    if(P&&rigs.indexOf(R2)>=0){ // the wound opens now it's out, worse the deeper it went
      if(P.heart&&fq<0.45){P.heart=null;P.hw=P.hwNH;}if(P.hw)P.hw*=0.3+1.2*fq; // a shallow stab doesn't reach the heart
      if(S.seep)R2.holes=Math.max(0,(R2.holes||0)-S.seep); // swap the seep for the full bleed
      if(P.hw&&RS.FLAGS.bleed)R2.holes=(R2.holes||0)+P.hw;
      if(P.hole){if(!P.holeOn)(R2.hole||(R2.hole=[])).push(P.hole);if(opt('oGore')&&P.hole.k<R2.bodies.length+(R2.chunks?R2.chunks.length:0)){var wp2=partOf(R2,P.hole.k).getWorldPoint(Vec2(P.hole.x,P.hole.y));addDecal(R2,P.hole.k,wp2,0.03);splats(R2,P.hole.k,2,0.02);}}
      if(P.heart&&!R2.heart)R2.heart=P.heart;
    }
    if(drag&&drag.body===g.b)drag.mass=g.b.getMass(); // was grabbed while welded in, so the pull was sized for the whole body: knife alone now or it rockets off
    g.ghost=0.25; // stays passing through that body a moment so it doesn't catch on the way out
    if(yank&&opt('oGore')&&rigs.indexOf(S.r)>=0){var p=g.b.getWorldPoint(Vec2(0.08,0)),d=g.b.getWorldVector(Vec2(-1,0));gL=layerK(S.r,S.k);gush(p.x,p.y,d.x*1.5,d.y*1.5,Math.round(12+40*fq),1+1.4*fq,0.8);sfx('squelch');S.r.painS=Math.min(2,(S.r.painS||0)+0.3);}
  }
  var mq=[],midN=0;
  function meleeContact(c){
    var fa=c.getFixtureA(),fb=c.getFixtureB(),A=fa.getBody(),B=fb.getBody(),ga=itemOf(A),gb=itemOf(B),g=ga||gb;
    if(!g||g.type==='gun'||(ga&&gb))return;
    var wf=ga?fa:fb,ob=ga?B:A,H=partHit(ob);if(!H)return;
    var wm=c.getWorldManifold(null),p=wm&&wm.points&&wm.points[0];if(!p)p=ob.getWorldCenter();
    var va=g.b.getLinearVelocityFromWorldPoint(p),vb=ob.getLinearVelocityFromWorldPoint(p),rx=va.x-vb.x,ry=va.y-vb.y,n=wm.normal||Vec2(0,0);
    var sn=Math.abs(rx*n.x+ry*n.y),st=Math.hypot(rx,ry);
    mq.push({g:g,fp:(wf.getUserData()||{}).part,r:H.r,k:H.k,p:Vec2(p.x,p.y),sn:sn,st:st,d:Vec2(rx/(st||1),ry/(st||1))});
  }
  function meleeStep(){
    for(var ii=0;ii<items.length;ii++){var gi2=items[ii];
      if(gi2.ghost>0){gi2.ghost-=RS.DT||1/240;if(gi2.ghost<=0&&!gi2.stuck)setGroup(gi2.b,0);}
      var S=gi2.stuck;if(!S)continue;
      if(rigs.indexOf(S.r)<0||(S.k<S.r.bodies.length&&S.k>=4&&S.r.inj&&S.r.inj.gone[S.k-1])){unstick(gi2,false);continue;}
      var tr=S.j.getJointTranslation(),dp=S.d0+tr,held=drag&&drag.body===gi2.b; // dp: how deep the tip is now
      if(dp>S.dmax){if(rigs.indexOf(S.r)>=0)S.r.painS=Math.min(2,(S.r.painS||0)+3*(dp-S.dmax));S.dmax=dp;}S.dep=dp; // deeper hurts
      if(S.k===0&&dp>KBRAIN)brainIn(S);
      if(rigs.indexOf(S.r)>=0&&S.pend&&S.pend.hw&&RS.FLAGS.bleed){var fq2=Math.min(1,S.dmax/KBL),hw2=(S.pend.heart&&fq2<0.45?S.pend.hwNH:S.pend.hw)*(0.3+1.2*fq2)*(S.k>=1&&S.k<=3?0.45:0.3),dh=hw2-(S.seep||0);if(Math.abs(dh)>1e-4){S.r.holes=Math.max(0,(S.r.holes||0)+dh);S.seep=hw2;}} // still in: it seeps (30% of what it'll bleed once out), so a deep chest stab still kills, slower
      if(rigs.indexOf(S.r)>=0&&S.pend&&S.pend.hole&&!S.pend.holeOn){S.pend.holeOn=1;(S.r.hole||(S.r.hole=[])).push(S.pend.hole);}
      S.j.setMaxMotorForce(held?3+10*Math.pow(dp/KBL,2):KGRIP); // pushing it in gets harder the deeper it goes; let go and the flesh holds it
      if(dp<0.004&&performance.now()-S.t0>150){unstick(gi2,true);continue;} // pulled back until the tip's out
      if(held&&performance.now()-S.t0>350){var wq=gi2.b.getWorldPoint(drag.local),bq=gi2.b.getWorldVector(Vec2(-1,0)); // a hard yank (finger well back along the handle) rips it out even if it drags them along
        if((drag.target.x-wq.x)*bq.x+(drag.target.y-wq.y)*bq.y>0.35){unstick(gi2,true);continue;}}
    }
    if(!mq.length)return;var q=mq;mq=[];var now=performance.now();
    q.forEach(function(h){
      var g=h.g,r=h.r,k=h.k;if(rigs.indexOf(r)<0)return;var key=(r.mid||(r.mid=++midN))+'_'+k;if(g.cd[key]&&now-g.cd[key]<250)return;
      var f=RS.FLAGS.realism===0?0.6:RS.FLAGS.realism===2?1.25:1,nb=r.bodies.length,gore=opt('oGore');
      if(g.type==='hammer'){
        var sp=h.fp==='head'?h.sn:h.sn*0.5;if(sp<2.5)return;g.cd[key]=now;sfx('thud',sp+3);
        r.painS=Math.min(2,(r.painS||0)+Math.min(0.6,sp*0.05));r.ev=r.ev||[];
        if(gore){gL=layerK(r,k);addDecal(r,k,h.p,0.02+0.004*sp);}
        if(k>=nb)return;
        if(k===0){
          if(sp>7*f&&RS.FLAGS.death!==0){if(!r.dead){r.dead=true;r.ev.push({t:'crush',k:0,d:sp});}if(RS.FLAGS.shatter)RS.burst(r,0,sp>10*f);} // skull caved in (bursts apart on Heavy/Full gore)
          else if(sp>5*f&&!r.dead){r.koT=Math.max(r.koT||0,4);r.koWhy='head';sfx('crunch',true);RS.brainHit(r,0,true);} // fractured skull: fits, then dies
          else if(sp>4*f&&!r.dead){r.koT=Math.max(r.koT||0,3+sp*0.3);r.koWhy='head';sfx('crunch',false);RS.brainHit(r,0.1+0.25*(sp-3*f)/(2*f),false);}
          return;
        }
        if(k>=4){var j=k-1;if(!r.inj||r.inj.gone[j])return;f*=[1,0.9,0.8,1.5,1.15,0.9][(k-4)%6]; // the thigh bone takes the most
          if(sp>5.5*f){r.ev.push({t:'crush',k:j,d:sp});if(RS.FLAGS.shatter&&RS.FLAGS.sever)RS.shatter(r,j,sp>9*f);else RS.breakBone(r,j,true,1);} // crushed (shattered into pieces on Heavy/Full gore)
          else if(sp>2.5*f&&!r.inj.broken[j]){RS.breakBone(r,j,true,(sp-2.5*f)/(3*f));} // a harder blow: more likely the bone comes through the skin
          return;
        }
        // torso: knocked down; hard enough breaks ribs/spine and bleeds inside (stays down)
        if(sp>11*f){r.ev.push({t:'crushT',k:k,d:sp});if(RS.FLAGS.death!==0)r.dead=true;if(RS.FLAGS.shatter)RS.burst(r,k,sp>14*f);if(RS.FLAGS.bleed)r.holes=(r.holes||0)+6;return;} // a monster blow caves the chest / stoves in the belly: dead
        if(sp>3.5*f&&!r.dead){var lp=r.bodies[k].getLocalPoint(h.p),bad=sp>6*f;
          if(bad){if(k<3)RS.breakBone(r,k,true,(sp-6*f)/(3*f));if(RS.FLAGS.bleed)r.holes=(r.holes||0)+1.5;sfx('crunch',true);}
          RS.shot(r,k,lp.x,lp.y,h.d.x,bad);}
      }else{ // knife: only the blade cuts
        if(h.fp!=='blade'||h.st<2||g.stuck||g.ghost>0)return;g.cd[key]=now;
        var ax=g.b.getWorldVector(Vec2(1,0)),along=(h.d.x*ax.x+h.d.y*ax.y),stab=along>0.7&&h.st>2.5,slash=!stab;
        var pend=wound(r,k,h.p,h.d,{cut:1,s:h.st,slash:slash,stab:stab});
        if(stab&&rigs.indexOf(r)>=0){ // point first: it goes in and stays there until you pull it out
          var pb=partOf(r,k),gi=pb.getFixtureList().getFilterGroupIndex();
          var tip=g.b.getWorldPoint(Vec2(KBL,0)),pen=(tip.x-h.p.x)*ax.x+(tip.y-h.p.y)*ax.y,bp=g.b.getPosition();
          var d0=Math.min(KBL,0.02+0.014*h.st),mv=d0-pen; // how far the tip gets in on its own: a hard thrust goes deeper; pushing does the rest
          g.b.setTransform(Vec2(bp.x+ax.x*mv,bp.y+ax.y*mv),g.b.getAngle());
          g.stuck={r:r,k:k,gi:gi,j:null,t0:now,pend:pend,d0:d0,dep:d0,dmax:d0};if(drag&&drag.body===g.b){var dw=g.b.getWorldPoint(drag.local);drag.target=Vec2(dw.x,dw.y);drag.mass=KHAND;}setGroup(g.b,gi);g.b.setLinearVelocity(pb.getLinearVelocity());g.b.setAngularVelocity(pb.getAngularVelocity());
          g.stuck.j=world.createJoint(planck.PrismaticJoint({enableLimit:true,lowerTranslation:-d0-0.01,upperTranslation:KBL-d0,enableMotor:true,motorSpeed:0,maxMotorForce:KGRIP},pb,g.b,h.p,ax));sfx('squelch'); // slides only along the blade; the motor is the flesh gripping it
          if(k===0&&d0>KBRAIN)brainIn(g.stuck);
        }
        if(k>=4&&k<nb&&h.st>7*f&&RS.FLAGS.sever&&gore){var jj=k-1,ty=(k-4)%6;if(ty===2||ty===5||ty===1||ty===4){if(r.inj&&!r.inj.gone[jj])RS.sever(r,jj);}} // a hard chop takes off a hand, foot, forearm or shin
      }
    });
  }
  function partHit(b){ // which human (and which part index, chunks after the body list) a body belongs to
    for(var i=0;i<rigs.length;i++){var r=rigs[i],k=r.bodies.indexOf(b);if(k>=0)return {r:r,k:k};
      if(r.chunks)for(var c=0;c<r.chunks.length;c++)if(r.chunks[c].b===b&&!r.chunks[c].gone)return {r:r,k:r.bodies.length+c,ch:r.chunks[c]};}
    return null;
  }
  function fire(g){
    var b=g.b,mz=b.getWorldPoint(Vec2(0.11,0.012*g.s)),d=b.getWorldVector(Vec2(1,0)),end=Vec2(mz.x+d.x*40,mz.y+d.y*40),best=null;
    world.rayCast(mz,end,function(f,pt,n,fr){if(f.getBody()===b)return -1;best={f:f,p:Vec2(pt.x,pt.y),n:Vec2(n.x,n.y),fr:fr};return fr;});
    var hp=best?best.p:end;
    fx.push({t:'tracer',a:Vec2(mz.x,mz.y),b:hp,life:0.09},{t:'flash',p:Vec2(mz.x,mz.y),d:Vec2(d.x,d.y),life:0.06});
    b.applyLinearImpulse(Vec2(-d.x*0.03,-d.y*0.03),mz,true);b.applyAngularImpulse(0.0009,true); // recoil: kicks back and up
    sfx('bang');
    if(!best)return;
    var hb=best.f.getBody();
    if(hb.isDynamic())hb.applyLinearImpulse(Vec2(d.x*0.16,d.y*0.16),hp,true); // a 9 mm round's momentum (scaled up a bit so you see it)
    var H=partHit(hb);if(!H)return;
    wound(H.r,H.k,hp,d,{gun:1});
  }
  // WOUND: a bullet (o.gun) or a blade (o.cut, o.s = speed) going into part k of rig r at world point hp, direction d
  function wound(r,k,hp,d,o){
    var gore=opt('oGore'),gun=!!o.gun,cut=!!o.cut,plug=!!o.stab,pend={}; // plug: the blade is still in, so it doesn't bleed until it comes out
    var nb=r.bodies.length,ext=k>=4&&k<nb&&((k-4)%6===2||(k-4)%6===5); // hand or foot
    sfx('squelch');r.painS=Math.min(2,(r.painS||0)+(ext?0.25:0.35));
    var sc=cut?0.7:1; // a stab bleeds a bit less than a bullet
    var lp=partOf(r,k).getLocalPoint(hp);
    if(gore){
      gL=layerK(r,k);if(!plug)gush(hp.x,hp.y,-d.x*0.6,-d.y*0.6,14,1.2,0.6);if(plug);else if(gun)gush(hp.x,hp.y,d.x*2.5,d.y*2.5,30,2.4,0.5);else gush(hp.x,hp.y,d.x*0.8,d.y*0.8,22,1.6,0.6); // entry splash, exit spray
      if(!plug){addDecal(r,k,hp,0.03);splats(r,k,2,0.02);}
      var H2=r.holeK||(r.holeK={}),HL=H2[k]||(H2[k]=[]),pb2=partOf(r,k),jag=function(){var a=[];for(var q=0;q<10;q++)a.push(0.7+Math.random()*0.6);return a;};
      if(cut){var dl=pb2.getLocalVector(d);HL.push({x:lp.x,y:lp.y,r:0.012+Math.random()*0.004,j:jag(),a:0,cut:1,ca:Math.atan2(dl.y,dl.x)+Math.PI/2+(Math.random()-0.5)*0.6,len:o.slash?2.6:1.6});}
      else HL.push({x:lp.x,y:lp.y,r:0.017+Math.random()*0.004,j:jag(),a:Math.random()*6.28});
      // exit wound: walk along the shot until it leaves this part; bigger and more torn than the way in
      var fx2=null;for(var f2=pb2.getFixtureList();f2;f2=f2.getNext())if(f2.testPoint(Vec2(hp.x+d.x*0.004,hp.y+d.y*0.004))){fx2=f2;break;}
      if(gun&&fx2){var ex=hp.x,ey=hp.y;for(var st=0;st<80;st++){var nx=ex+d.x*0.004,ny=ey+d.y*0.004;if(!fx2.testPoint(Vec2(nx,ny)))break;ex=nx;ey=ny;}
        var le=pb2.getLocalPoint(Vec2(ex,ey));HL.push({x:le.x,y:le.y,r:0.023+Math.random()*0.006,j:jag(),a:Math.random()*6.28,exit:1});addDecal(r,k,Vec2(ex,ey),0.04);}
      if(plug)pend.hole={k:k,x:lp.x,y:lp.y};else (r.hole||(r.hole=[])).push({k:k,x:lp.x,y:lp.y});
    }
    // where it went in: the heart sits in the middle-front of the chest; chest (lungs), gut and pelvis hits are fatal over time
    var Sc=r.scale||1,fr=lp.x*(r.dir||1),heart=k===1&&lp.y>-0.09*Sc&&lp.y<0.05*Sc&&fr>-0.03*Sc;
    if(heart&&!r.heart&&plug)pend.heart={x:lp.x,y:lp.y};
    else if(heart&&!r.heart){r.heart={x:lp.x,y:lp.y};if(gore){gL=layerK(r,k);gush(hp.x,hp.y,-d.x*1.5,-d.y*1.5,60,2.6,0.8);}}
    var leg=k>=4&&k<nb&&((k-4)%6===3||(k-4)%6===4),thigh=k>=4&&k<nb&&(k-4)%6===3;
    var hw=ext?0.35:heart?10:k===1?5:k===2?3.2:k===3?2.4:thigh?2.2:1; // the thigh has the femoral artery // x0.25 = bleed weight (a torn-off leg is 1)
    if(plug){pend.hw=hw*sc;if(heart)pend.hwNH=5*sc;}else if(RS.FLAGS.bleed)r.holes=(r.holes||0)+hw*sc;
    if(k===0&&gun&&RS.FLAGS.death!==0&&!r.dead){r.dead=true;r.ev=r.ev||[];} // a bullet to the head: dead
    else if(k===0&&cut&&!plug&&!r.dead)RS.brainHit(r,0.15,false); // a blade into the skull: fits, fades out and dies; a slash: some damage
    else if(gun&&k>=4&&k<nb&&!ext&&r.inj){var j=k-1;var pb=RS.FLAGS.realism===0?0.8:RS.FLAGS.realism===2?0.35:0.5;
      if(!r.inj.broken[j]&&!r.inj.gone[j]&&Math.random()<pb)RS.breakBone(r,j,true,0);} // the bullet hole already bleeds
    if(k<nb&&!r.dead)RS.shot(r,k,lp.x,lp.y,d.x,k>=1&&k<=3||leg);
    if(gun&&k<nb){ // knocked back: the whole body gets shoved along the shot, hardest at the part that was hit (hands/feet: just that limb)
      var kv=ext?0:k===0?1.6:k<4?2.2:1.4,kx=d.x,ky=Math.max(-0.3,d.y)+0.15;
      r.bodies.forEach(function(bq,q){var w2=q===k?2.2:kv;if(ext&&Math.abs(q-k)>1)return;if(ext)w2=q===k?2:1;var v=bq.getLinearVelocity();bq.setLinearVelocity(Vec2(v.x+kx*w2,v.y+ky*w2*0.5));bq.setAwake(true);});
    }
    return pend;
    }
  function drawItems(under){ // under: knives stuck in someone go beneath the bodies so only the handle shows
    items.forEach(function(g){
      if(under)return;
      var b=g.b,clipd=false;
      if(g.stuck&&rigs.indexOf(g.stuck.r)>=0){ // stuck in someone: hide the part of the blade that's inside them
        var pb=partOf(g.stuck.r,g.stuck.k);if(pb){ctx.save();clipd=true;ctx.beginPath();ctx.rect(0,0,W,H);
          for(var fq=pb.getFixtureList();fq;fq=fq.getNext()){polyPath(pb,fq.getShape());ctx.closePath();}ctx.clip('evenodd');}}
      if(g.type!=='gun'){ // each fixture its own colour: wood handle, steel head / blade
        for(var f0=b.getFixtureList();f0;f0=f0.getNext()){var vs0=f0.getShape().m_vertices,pt=(f0.getUserData()||{}).part;ctx.beginPath();
          for(var i0=0;i0<vs0.length;i0++){var q0=toScreen(b.getWorldPoint(vs0[i0]));if(i0)ctx.lineTo(q0.x,q0.y);else ctx.moveTo(q0.x,q0.y);}ctx.closePath();
          ctx.fillStyle=pt==='handle'?(g.type==='knife'?'#2a2622':'#6b4a2c'):pt==='blade'?'#a9b0b6':'#5b5e62';ctx.fill();ctx.strokeStyle='#100e0c';ctx.lineWidth=Math.max(1,0.005*cam.z);ctx.stroke();}
        if(g.type==='knife'&&!g.stuck){var e1=toScreen(b.getWorldPoint(Vec2(0.005,-0.01))),e2=toScreen(b.getWorldPoint(Vec2(0.13,-0.005)));ctx.strokeStyle='rgba(230,240,250,0.8)';ctx.lineWidth=Math.max(1,0.003*cam.z);ctx.beginPath();ctx.moveTo(e1.x,e1.y);ctx.lineTo(e2.x,e2.y);ctx.stroke();}
        if(clipd)ctx.restore();
        return;
      }
      ctx.beginPath();
      for(var f=b.getFixtureList();f;f=f.getNext()){var vs=f.getShape().m_vertices;for(var i=0;i<vs.length;i++){var q=toScreen(b.getWorldPoint(vs[i]));if(i)ctx.lineTo(q.x,q.y);else ctx.moveTo(q.x,q.y);}ctx.closePath();}
      ctx.fillStyle='#4a4741';ctx.fill();ctx.strokeStyle='#14120f';ctx.lineWidth=Math.max(1,0.006*cam.z);ctx.stroke();
      var s1=toScreen(b.getWorldPoint(Vec2(-0.07,0.028*g.s))),s2=toScreen(b.getWorldPoint(Vec2(0.1,0.028*g.s))); // slide highlight
      ctx.strokeStyle='rgba(255,176,32,0.35)';ctx.lineWidth=Math.max(1,0.004*cam.z);ctx.beginPath();ctx.moveTo(s1.x,s1.y);ctx.lineTo(s2.x,s2.y);ctx.stroke();
    });
    if(under)return;
    for(var i=fx.length-1;i>=0;i--){var e=fx[i];
      if(e.t==='tracer'){var a=toScreen(e.a),c=toScreen(e.b);ctx.strokeStyle='rgba(255,220,150,'+(e.life/0.09)+')';ctx.lineWidth=Math.max(1,0.006*cam.z);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(c.x,c.y);ctx.stroke();}
      else{var p=toScreen(e.p),r0=0.06*cam.z*(e.life/0.06),ang=Math.atan2(-e.d.y,e.d.x);ctx.save();ctx.translate(p.x,p.y);ctx.rotate(ang);
        ctx.fillStyle='rgba(255,200,80,0.9)';ctx.beginPath();ctx.moveTo(0,-r0*0.35);ctx.lineTo(r0*1.6,0);ctx.lineTo(0,r0*0.35);ctx.closePath();ctx.fill();
        ctx.fillStyle='rgba(255,250,220,0.95)';ctx.beginPath();ctx.arc(0,0,r0*0.35,0,7);ctx.fill();ctx.restore();}}
  }
  function fxStep(dt){for(var i=fx.length-1;i>=0;i--){fx[i].life-=dt;if(fx[i].life<=0)fx.splice(i,1);}}
  function spawn(kind){
    var x=cam.x+(rigs.length?(Math.random()-0.5)*0.6:-0.2);
    for(var tries=0;tries<12;tries++){ // don't spawn inside someone (overlapping bodies would break bones)
      var clash=rigs.some(function(r){return Math.abs(r.bodies[3].getPosition().x-x)<0.55;});if(!clash)break;
      x=cam.x+(tries%2?1:-1)*(0.6+0.3*(tries>>1));
    }
    var sc=RS.sampleScenario(RS.rngMake((Math.random()*1e9)|0),{only:kind==='stand'?'stand':kind,pushMax:1.5});
    var o={pose:kind==='stand'?null:sc.pose,rootAng:kind==='stand'?0:sc.rootAng,h:kind==='drop'?Math.max(0.3,sc.h):0,clear:kind==='fallen'?0.01:0.02};
    RS.FLAGS.soft=opt('oSoft')?1:0;
    o.scale=0.97+Math.random()*0.06;o.wid=0.95+Math.random()*0.1; // everyone slightly different
    var rig=RS.buildRig(world,x,0,-(++group),o);rig.skin=skinTone();
    if(kind==='drop'&&sc.vel)rig.bodies.forEach(function(b){b.setLinearVelocity(Vec2(sc.vel.vx,sc.vel.vy));b.setAngularVelocity(sc.vel.w);});
    rig.num=++humanN;
    rigs.push(rig);liveBufs.push(RS.newBuf());
    if(rigs.length>MAXR){var old=rigs.shift();liveBufs.shift();old.bodies.forEach(function(b){world.destroyBody(b);});} // keep it fast on phones
  }
  function hit(p){
    for(var ii=0;ii<items.length;ii++)for(var f0=items[ii].b.getFixtureList();f0;f0=f0.getNext())if(f0.testPoint(p))return items[ii].b; // weapons are drawn on top, so they're grabbed first (a knife stuck in someone, not the arm behind it)
    for(var b=world.getBodyList();b;b=b.getNext()){
      if(!b.isDynamic())continue;
      for(var f=b.getFixtureList();f;f=f.getNext())if(f.testPoint(p))return b;
    }
    return null;
  }
  function pick(sx,sy){
    var radii=[0,12,24,36];
    for(var i=0;i<radii.length;i++){
      var r=radii[i];
      if(r===0){var b0=hit(toWorld(sx,sy));if(b0)return {b:b0,sx:sx,sy:sy};continue;}
      for(var k=0;k<12;k++){
        var a=k*Math.PI/6,px=sx+Math.cos(a)*r,py=sy+Math.sin(a)*r,b=hit(toWorld(px,py));
        if(b)return {b:b,sx:px,sy:py};
      }
    }
    return null;
  }
  function islandMass(b){
    var seen=[b],st=[b],m=0;
    while(st.length){
      var x=st.pop();m+=x.getMass();
      for(var j=x.getJointList();j;j=j.next){
        var o=j.other;if(o&&o.isDynamic()&&seen.indexOf(o)<0){seen.push(o);st.push(o);}
      }
    }
    return m;
  }
  var mj=null,mjB=null,mjW=null; // the finger: a stiff mouse joint (solved implicitly, so it can be stiff without blowing up)
  function applyDrag(){
    if(mj&&(!drag||drag.body!==mjB||mjW!==world)){if(mjW===world)try{world.destroyJoint(mj);}catch(e){}mj=null;mjB=null;}
    if(!drag)return;
    var b=drag.body;b.setAwake(true);
    var ib=itemOf(b),m=Math.max(drag.mass,b.getMass());
    if(!+$('rGrab').value&&!(ib&&ib.type==='knife')){ // Smooth (default; the knife always uses Sticky): the original soft spring pull, it trails the finger a bit
      if(mj){try{world.destroyJoint(mj);}catch(e){}mj=null;mjB=null;}
      var wp0=b.getWorldPoint(drag.local),v0=b.getLinearVelocityFromWorldPoint(wp0),t0=drag.target,tx=Math.max(-ARENA+0.02,Math.min(ARENA-0.02,t0.x)),ty=Math.max(0.02,Math.min(ROOF-0.02,t0.y)); // still can't be pulled through the floor or a wall
      var ax=120*(tx-wp0.x)-22*v0.x,ay=120*(ty-wp0.y)-22*v0.y,cp=ib?(ib.stuck?40:300):80,mg=Math.sqrt(ax*ax+ay*ay);if(mg>cp){ax*=cp/mg;ay*=cp/mg;}
      b.applyForce(Vec2(ax*m,ay*m),wp0,true);
      var g0=aimOf(b);if(g0&&g0.aim!=null&&!g0.stuck){var e0=g0.aim-b.getAngle();e0=Math.atan2(Math.sin(e0),Math.cos(e0));b.setAngularVelocity(30*e0);}
      return;
    }
    var cap=ib?(ib.stuck?40:600):250; // max accel the hand gives (x mass): a stuck knife gets about 40 N, or the joint gets torn about; a body gets yanked hard enough to rip limbs off
    if(!mj){var wp=b.getWorldPoint(drag.local);mj=world.createJoint(planck.MouseJoint({maxForce:cap*m,frequencyHz:ib?25:15,dampingRatio:0.9},ground,b,wp));mjB=b;mjW=world;}
    var tg=drag.target;tg=Vec2(Math.max(-ARENA+0.02,Math.min(ARENA-0.02,tg.x)),Math.max(0.02,Math.min(ROOF-0.02,tg.y))); // a finger below the floor or past a wall holds it against it, not through
    mj.setMaxForce(cap*m);var soft=false;if(ib&&ib.stuck){var wq=b.getWorldPoint(drag.local),bx=b.getWorldVector(Vec2(1,0));soft=(drag.target.x-wq.x)*bx.x+(drag.target.y-wq.y)*bx.y>0;} // pushing a stuck knife in: held softly, so how far you push sets how hard it goes in; pulling it out: firm
    mj.setFrequency(soft?3:ib?25:15);mj.setTarget(tg);
    var g=aimOf(b);if(g&&g.aim!=null&&!g.stuck){var e=g.aim-b.getAngle();e=Math.atan2(Math.sin(e),Math.cos(e));b.setAngularVelocity(30*e);}
  }


  var ptrs={},mode='none',pan=null,pinch=null;
  function ids(){return Object.keys(ptrs);}
  function pos(e){var r=cv.getBoundingClientRect();return {x:e.clientX-r.left,y:e.clientY-r.top};}
  function startPinch(){
    var k=ids(),a=ptrs[k[0]],b=ptrs[k[1]];
    var mx=(a.x+b.x)/2,my=(a.y+b.y)/2;
    pinch={d:Math.hypot(a.x-b.x,a.y-b.y)||1,z:cam.z,w:toWorld(mx,my)};
    mode='pinch';drag=null;
  }
  var holdT=0,tap=null,trig=null;
  cv.addEventListener('pointerdown',function(e){
    camT=null;
    e.preventDefault();
    try{cv.setPointerCapture(e.pointerId);}catch(_){}
    ptrs[e.pointerId]=pos(e);
    if(ids().length>=2&&mode==='drag'&&drag&&aimOf(drag.body)&&!trig){var tg=aimOf(drag.body);trig={id:e.pointerId,g:tg};clearTimeout(holdT);tap=null;aimAt(tg,toWorld(ptrs[e.pointerId].x,ptrs[e.pointerId].y));return;}
    if(ids().length>=2){startPinch();return;}
    var p=ptrs[e.pointerId],h=pick(p.x,p.y);
    if(h){
      var wp=toWorld(h.sx,h.sy);
      drag={body:h.b,local:h.b.getLocalPoint(wp),target:toWorld(p.x,p.y),mass:itemOf(h.b)&&itemOf(h.b).stuck?KHAND:islandMass(h.b)};
      mode='drag';tap={id:e.pointerId,t:performance.now(),x:p.x,y:p.y,b:h.b};
      var hg=aimOf(h.b);if(hg){hg.aim=h.b.getAngle();var hs=Math.cos(hg.aim)<0?-1:1;if(hs!==hg.s){hg.s=hs;itemFix(hg);}}
      var r0=rigOf(h.b),p0={x:p.x,y:p.y};clearTimeout(holdT);
      holdT=setTimeout(function(){var q=ptrs[e.pointerId];if(r0&&mode==='drag'&&q&&Math.hypot(q.x-p0.x,q.y-p0.y)<10){drag=null;mode='none';openMenu(r0,q);}},600);
    }else{mode='pan';pan={x:p.x,y:p.y};camFollow=null;}
  });
  cv.addEventListener('pointermove',function(e){
    if(!ptrs[e.pointerId])return;
    e.preventDefault();
    var p=pos(e);ptrs[e.pointerId]=p;
    if(mode==='pinch'&&ids().length>=2){
      var k=ids(),a=ptrs[k[0]],b=ptrs[k[1]];
      var mx=(a.x+b.x)/2,my=(a.y+b.y)/2,d=Math.hypot(a.x-b.x,a.y-b.y)||1;
      cam.z=Math.max(15,Math.min(600,pinch.z*d/pinch.d));
      cam.x=pinch.w.x-(mx-W/2)/cam.z;cam.y=pinch.w.y+(my-H/2)/cam.z;
    }else if(trig&&trig.id===e.pointerId){aimAt(trig.g,toWorld(p.x,p.y));}
    else if(mode==='drag'&&drag){drag.target=toWorld(p.x,p.y);}
    else if(mode==='pan'&&pan){cam.x-=(p.x-pan.x)/cam.z;cam.y+=(p.y-pan.y)/cam.z;pan={x:p.x,y:p.y};}
  });
  function up(e){
    clearTimeout(holdT);
    if(trig&&trig.id===e.pointerId){var tq=ptrs[e.pointerId],tg=trig.g;if(tq)aimAt(tg,toWorld(tq.x,tq.y));if(tg.type==='gun'){tg.b.setTransform(tg.b.getPosition(),tg.aim);tg.b.setAngularVelocity(0);fire(tg);}trig=null;delete ptrs[e.pointerId];return;}
    if(tap&&tap.id===e.pointerId){var q=ptrs[e.pointerId];
      if(q&&performance.now()-tap.t<350&&Math.hypot(q.x-tap.x,q.y-tap.y)<12)items.forEach(function(g){if(g.b===tap.b&&g.type==='gun')fire(g);});tap=null;}
    delete ptrs[e.pointerId];
    var k=ids();
    if(k.length===0){mode='none';drag=null;pan=null;pinch=null;}
    else if(mode==='pinch'){mode='pan';pan={x:ptrs[k[0]].x,y:ptrs[k[0]].y};}
  }
  cv.addEventListener('pointerup',up);
  cv.addEventListener('pointercancel',up);
  cv.addEventListener('wheel',function(e){e.preventDefault();var p=pos(e);setZoom(cam.z*Math.exp(-e.deltaY*0.002),p.x,p.y);},{passive:false});
  ['touchstart','touchmove','touchend','gesturestart','gesturechange'].forEach(function(n){
    document.addEventListener(n,function(e){if(e.target===cv)e.preventDefault();},{passive:false});
  });

  document.getElementById('bSpawn').onclick=function(){var m=$('spmenu');m.hidden=!m.hidden;};
  document.getElementById('spHuman').onclick=function(){$('spmenu').hidden=true;spawn('stand');};
  document.getElementById('spGun').onclick=function(){$('spmenu').hidden=true;spawnGun();};
  document.getElementById('spHammer').onclick=function(){$('spmenu').hidden=true;spawnHammer();};
  document.getElementById('spKnife').onclick=function(){$('spmenu').hidden=true;spawnKnife();};
  document.getElementById('bReset').onclick=document.getElementById('bReset2').onclick=function(){init();pauseMenu(false);};
  var userPaused=false,MAXR=Infinity; // no cap on people
  function setOn(id,on){document.getElementById(id).classList.toggle('on',!!on);}
  function $(id){return document.getElementById(id);}
  // GYRO: the arena is fixed to the phone, so tilting (or shaking) the phone tilts the room; gravity keeps pointing the
  // real way down, so people and things fall and slide inside it. Uses the accelerometer (gravity included), smoothed.
  var gyro={on:false,lis:false,gx:0,gy:-10,sgn:0};
  function gyroMotion(e){var a=e.accelerationIncludingGravity;if(!a||a.x==null)return;gyro.got=1;
    var x=a.x,y=a.y,an=((screen.orientation&&screen.orientation.angle)||window.orientation||0)*Math.PI/180,c=Math.cos(an),s=Math.sin(an),sx=x*c+y*s,sy=-x*s+y*c; // into screen axes
    if(!gyro.sgn)gyro.sgn=sy>0?-1:1; // browsers disagree on the sign: it's held roughly upright when switched on, so down is whichever way y points
    var k=10/9.81,tx=gyro.sgn*sx*k,ty=gyro.sgn*sy*k,m=Math.hypot(tx,ty);if(m>40){tx*=40/m;ty*=40/m;}
    gyro.gx+=(tx-gyro.gx)*0.3;gyro.gy+=(ty-gyro.gy)*0.3;}
  function gyroNote(t){var d=document.getElementById('gyDesc');if(d&&gyro.on)d.textContent=t;}
  function gyroOn(on){
    gyro.on=on;if(!on){gyro.sgn=0;gyro.gx=0;gyro.gy=-10;return;}
    if(gyro.lis)return;
    var go=function(){if(!gyro.lis){window.addEventListener('devicemotion',gyroMotion);gyro.lis=true;gyro.t0=performance.now();
      setTimeout(function(){if(gyro.on&&!gyro.got)gyroNote('No motion data: this page is running inside a frame that blocks the motion sensor. Open the GitHub Pages version in Safari instead.');},1500);}};
    if(window.DeviceMotionEvent&&typeof DeviceMotionEvent.requestPermission==='function'){ // iPhone: has to be asked from a tap
      gyroNote('Tap anywhere to allow motion access.');
      if(!gyro.ask){gyro.ask=function(){document.removeEventListener('touchend',gyro.ask,true);document.removeEventListener('click',gyro.ask,true);var a=gyro.ask;gyro.ask=null;
        DeviceMotionEvent.requestPermission().then(function(r){if(r==='granted'){go();gyroNote('Tilt or shake the phone: the room tilts with it, and everything inside falls the real way down.');}else gyroNote('Motion access was refused. iPhone: close and reopen the page to be asked again.');}).catch(function(){gyroNote('Motion access is blocked here (the page is inside a frame). Open the GitHub Pages version in Safari.');});};
        document.addEventListener('touchend',gyro.ask,true);document.addEventListener('click',gyro.ask,true);}
    }else go();
  }
  function tgl(id){var c=$(id);c.checked=!c.checked;syncMenu();}
  // gore setting: 5 levels, each mapped to a 0-10 strength (0 none, 3 light, 5 bleeding + limbs off, 8 shattering, 10 full)
  var GLV=[0,3,5,8,10],GNAME=['None','Light','Medium','Heavy','Full'],
    GDESC=['No blood, no limbs coming off, nobody dies.','A little blood, bones break, can die. Limbs stay on.','Bleeding, limbs can be torn off.','Lots of blood, crushed limbs shatter.','Everything.'];
  var RNAME=['Glass','Default','Realistic'],RDESC=['Everything breaks, crushes and tears off easily.','The original game feel.','Real-world breaking forces and crush speeds.'];
  function goreV(){return GLV[+$('rGore').value]||0;}
  function syncMenu(){
    $('bSlowB').textContent='Slow motion: '+(opt('oSlow')?'on':'off');
    $('mSound').textContent=$('sSound').textContent='Sound: '+(opt('oSound')?'on':'off');
    var cv=+$('rCrush').value,gl=+$('rGore').value,gv=goreV(),rl=+$('rReal').value;$('vCrush').textContent=cv;$('vReal').textContent=RNAME[rl];$('rDesc').textContent=RDESC[rl];$('vGore').textContent=GNAME[gl];
    $('oGore').checked=gv>0;GM=gv/10;
    $('gDesc').textContent=GDESC[gl];
    var gy0=+$('rGyro').value;$('vGyro').textContent=gy0?'On':'Off';$('gyDesc').textContent=gy0?'Tilt or shake the phone: the room tilts with it, and everything inside falls the real way down.':'Gravity always points down the screen.';gyroOn(!!gy0);
    var gr=+$('rGrab').value;$('vGrab').textContent=gr?'Sticky':'Smooth';$('grDesc').textContent=gr?'Things stay stuck to your finger: fast swings, hard yanks rip limbs off.':'Things follow your finger with a soft, smoothed pull (the original feel).';
    try{localStorage.setItem('eqOpts',JSON.stringify({sound:opt('oSound'),goreL:gl,crush:cv,real:rl,grab:gr,gyro:gy0}));}catch(_){}
    if(AC&&master)master.gain.value=opt('oSound')?0.9:0;
  }
  try{var so=JSON.parse(localStorage.getItem('eqOpts')||'null');if(so){$('oSound').checked=so.sound;if(so.goreL!=null)$('rGore').value=so.goreL;else if(so.goreLv!=null)$('rGore').value=Math.round(so.goreLv*0.4);else if(so.gore===false)$('rGore').value=0;if(so.crush!=null)$('rCrush').value=so.crush;if(so.real!=null)$('rReal').value=so.real;if(so.grab!=null)$('rGrab').value=so.grab;if(so.gyro!=null)$('rGyro').value=so.gyro;}}catch(_){}
  function pauseMenu(on){userPaused=on;$('pmenu').hidden=!on;setOn('bPause',on);}
  $('bPause').onclick=function(){pauseMenu(true);};
  $('mResume').onclick=function(){pauseMenu(false);};
  $('bSlowB').onclick=function(){tgl('oSlow');};
  $('mSound').onclick=$('sSound').onclick=function(){tgl('oSound');audioInit();};
  $('mSet').onclick=$('sSet').onclick=function(){$('smenu').hidden=false;};$('sDone').onclick=function(){$('smenu').hidden=true;};
  $('rCrush').oninput=$('rGore').oninput=$('rReal').oninput=$('rGrab').oninput=$('rGyro').oninput=syncMenu;
  $('bClean').onclick=function(){parts=[];stains=[];rigs.forEach(function(r){if(r.bl)r.bl.fill(0);r.dec=[];r.dln=[];r.holeK=null;r.hole=null;r.holes=0;r.heart=null;});pauseMenu(false);};
  $('mMain').onclick=function(){pauseMenu(false);init();var sp=$('splash');sp.hidden=false;sp.style.opacity=1;};
  $('bPlay').onclick=function(){var sp=$('splash');sp.style.transition='opacity .4s';sp.style.opacity=0;setTimeout(function(){sp.hidden=true;},400);audioInit();
    var h=$('hint');h.style.opacity=0.7;setTimeout(function(){h.style.opacity=0;},7000);};
  syncMenu();
  // HUD: blood and state of each ragdoll (top right)
  function esc(t){return String(t).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
  var humanN=0,camT=null,camFollow=null;
  function rigByNum(n){for(var i=0;i<rigs.length;i++)if(rigs[i].num===n)return rigs[i];return null;}
  function tabTap(e){ // tap a tab: pan the camera to that human (the drop-down tab opens the full list instead)
    if(e.target.closest('.hdd')){$('hlist').hidden=false;hud();return;}
    var c=e.target.closest('.hc');if(!c)return;var r=rigByNum(+c.dataset.n);if(!r)return;
    var p=r.bodies[1].getPosition();camT={x:p.x,y:p.y+0.3};$('hlist').hidden=true;
  }
  document.getElementById('hud').addEventListener('click',tabTap);
  document.getElementById('hlc').addEventListener('click',tabTap);
  document.getElementById('hlClose').onclick=function(){$('hlist').hidden=true;};
  // hold on a human to rename them
  var renaming=null;
  function rigOf(b){for(var i=0;i<rigs.length;i++)if(rigs[i].bodies.indexOf(b)>=0)return rigs[i];return null;}
  function openRename(r){renaming=r;var o=document.getElementById('rename'),i=document.getElementById('rnIn');i.value=r.name||'';o.hidden=false;setTimeout(function(){i.focus();},50);}
  function closeRename(save){var o=document.getElementById('rename');if(save&&renaming){var v=document.getElementById('rnIn').value.trim();renaming.name=v||null;}o.hidden=true;renaming=null;}
  document.getElementById('rnOk').onclick=function(){closeRename(true);};
  document.getElementById('rnCancel').onclick=function(){closeRename(false);};
  document.getElementById('rnIn').addEventListener('keydown',function(e){if(e.key==='Enter')closeRename(true);});
  // hold menu: small menu next to the human you held
  var menuRig=null;
  function openMenu(r,at){
    menuRig=r;var m=document.getElementById('hmenu');m.hidden=false;
    document.getElementById('hmFollow').textContent=camFollow===r?'Stop following':'Follow';
    var x=Math.min(W-m.offsetWidth-8,at.x+16),y=Math.max(8,Math.min(H-m.offsetHeight-120,at.y-m.offsetHeight/2));
    m.style.left=x+'px';m.style.top=y+'px';
  }
  function closeMenu(){document.getElementById('hmenu').hidden=true;menuRig=null;}
  function turnAround(r){ // rebuild them facing the other way, in the same pose, keeping everything about them
    var i=rigs.indexOf(r);if(i<0)return;
    var pose=new Float64Array(RS.NJ),k,I=r.inj;
    for(k=0;k<RS.NJ;k++){var a=r.ctrls[k].j.getJointAngle(),J=RS.JT[k];pose[k]=I&&(I.broken[k]||I.gone[k])?a:Math.max(J.lo,Math.min(J.hi,a));}
    var px=r.bodies[3].getPosition().x;
    allParts(r).forEach(function(b){world.destroyBody(b);});
    RS.FLAGS.soft=0;
    var n=RS.buildRig(world,px,0,-(++group),{pose:pose,rootAng:r.parts[3].getAngle(),clear:0.01,dir:-r.dir,scale:r.scale,wid:r.wid});
    ['num','name','skin','blood','bl','koT','dead','headCrushed'].forEach(function(f){if(r[f]!=null)n[f]=r[f];});
    if(I)for(k=0;k<RS.NJ;k++){if(I.gone[k]&&(k%3===0||!I.gone[k-1]))RS.sever(n,k);else if(I.broken[k])RS.breakBone(n,k,true,I.open&&I.open[k]?1:0);}
    n.ev=[];rigs[i]=n;if(camFollow===r)camFollow=n;
  }
  function removeRig(r){var i=rigs.indexOf(r);if(i<0)return;if(camFollow===r)camFollow=null;allParts(r).forEach(function(b){world.destroyBody(b);});rigs.splice(i,1);liveBufs.splice(i,1);}
  document.getElementById('hmRename').onclick=function(){var r=menuRig;closeMenu();if(r)openRename(r);};
  document.getElementById('hmFollow').onclick=function(){var r=menuRig;closeMenu();camFollow=camFollow===r?null:r;camT=null;};
  document.getElementById('hmTurn').onclick=function(){if(menuRig)turnAround(menuRig);closeMenu();};
  document.getElementById('hmHeal').onclick=function(){var r=menuRig;closeMenu();if(!r)return;for(var k=0;k<RS.NJ;k++)RS.breakBone(r,k,false);r.blood=1;r.dead=false;r.koT=0;r.age=0;if(r.bl)r.bl.fill(0);r.dec=[];r.dln=[];r.holeK=null;r.hole=null;r.holes=0;r.shot=null;r.heart=null;r.brain=0;r.bd=null;r.veg=false;r.seizT=0;};
  document.getElementById('hmKill').onclick=function(){if(menuRig)menuRig.dead=true;closeMenu();};
  document.getElementById('hmRemove').onclick=function(){if(menuRig)removeRig(menuRig);closeMenu();};
  cv.addEventListener('pointerdown',function(){if(menuRig)closeMenu();$('spmenu').hidden=true;},true);
  // pixelated number painted on each chest (3x5 font)
  var FONT={0:'111101101101111',1:'010110010010111',2:'111001111100111',3:'111001111001111',4:'101101111001001',5:'111100111001111',6:'111100111101111',7:'111001010010010',8:'111101111101111',9:'111101111001111'};
  function drawNumber(r){
    {
      var b=r.bodies[1],c=toScreen(b.getPosition()),t=String(r.num),px=0.024*cam.z,w=(t.length*4-1)*px,h=5*px;
      ctx.save();ctx.translate(c.x,c.y);ctx.rotate(-b.getAngle());ctx.fillStyle='rgba(40,20,14,0.75)';
      for(var d=0;d<t.length;d++){var g=FONT[t[d]];for(var q=0;q<15;q++)if(g[q]==='1')ctx.fillRect(-w/2+(d*4+q%3)*px,-h/2+(q/3|0)*px,Math.ceil(px),Math.ceil(px));}
      ctx.restore();
    }
  }
  function hud(){
    var h=document.getElementById('hud'),html='',many=rigs.length>5,L=$('hlist');
    if(many)rigs.forEach(function(r){html+=card(r);});
    function card(r){
      var bl=r.blood==null?1:r.blood,I=r.inj,nb=0,ng=0;
      if(I)for(var k=0;k<RS.NJ;k++){if(I.broken[k])nb++;if(I.gone[k]&&(k%3===0||!I.gone[k-1]))ng++;}
      var st=r.dead?'Dead':r.seizing?'Seizure':r.veg?'Vegetable':r.bd!=null?'Brain dying':r.koT>0?(r.koWhy==='pain'?'Passed out':'Knocked out'):r.groggy>0?'Waking up':r.cowering?'In pain':r.gu&&r.gu.ph>=0?'Getting up':nb||ng?'Hurt':'OK';
      return '<div class="hc" data-n="'+r.num+'">'+(r.name?esc(r.name):'#'+r.num)+' '+st+(nb?' · '+nb+' broken':'')+(ng?' · '+ng+' lost':'')+'<div class="hb"><i style="width:'+Math.round(bl*100)+'%"></i></div></div>';
    }
    // up to 5 humans: a tab each; more than that: one drop-down tab that opens the full list
    if(many){var dead=0;rigs.forEach(function(r){if(r.dead)dead++;});
      var lc=$('hlc');if(!L.hidden&&lc.innerHTML!==html)lc.innerHTML=html;
      html='<div class="hc hdd">'+rigs.length+' humans'+(dead?' · '+dead+' dead':'')+' <span class="dda">&#9660;</span></div>';}
    else{if(!L.hidden)L.hidden=true;html='';rigs.forEach(function(r){html+=card(r);});}
    if(h.innerHTML!==html)h.innerHTML=html;
  }
  document.getElementById('bIn').onclick=function(){setZoom(cam.z*1.4,W/2,H/2);};
  document.getElementById('bOut').onclick=function(){setZoom(cam.z/1.4,W/2,H/2);};
  function push(dir){
    var J=parseFloat(document.getElementById('pS').value)*dir;
    rigs.forEach(function(r){var c=r.bodies[1];c.applyLinearImpulse(Vec2(J,0),c.getWorldCenter(),true);});
  }
  document.getElementById('pL').onclick=function(){push(-1);};
  document.getElementById('pR').onclick=function(){push(1);};
  document.getElementById('oUse').onchange=function(){
    if(!opt('oUse'))rigs.forEach(function(r){r.tgt.fill(0);r.stiff.fill(1);r.ctrls.forEach(function(c){c.target=0;});});
  };

  // ---------- draw ----------
  function polyPath(b,sh){
    var vs=sh.m_vertices;
    for(var i=0;i<vs.length;i++){
      var s=toScreen(b.getWorldPoint(vs[i]));
      if(i===0)ctx.moveTo(s.x,s.y);else ctx.lineTo(s.x,s.y);
    }
    ctx.closePath();
  }
  function drawBodies(far){
    for(var b=world.getBodyList();b;b=b.getNext()){
      if(!b.isDynamic())continue;
      var u=b.getUserData()||{};if(!!u.far!==far)continue;
      for(var f=b.getFixtureList();f;f=f.getNext()){ctx.beginPath();polyPath(b,f.getShape());ctx.stroke();}
    }
  }
  // =====================================================================
  // GORE + SOUND: blood particles, ground stains, spurting stumps, crushed parts, skin fill, synthesized sounds
  // (Web Audio, no files) and an ambient room/wind bed. Driven by the sim's events (rig.ev) and blood level.
  // =====================================================================
  var AC=null,master=null,NB=null,lastThud=0,lastSplat=0;
  function audioInit(){
    if(!opt('oSound'))return;
    if(AC){if(AC.state!=='running')try{AC.resume();}catch(_){}return;}
    try{if(navigator.audioSession)navigator.audioSession.type='playback';}catch(_){} // iPhone: play even with the silent switch on
    try{AC=new (window.AudioContext||window.webkitAudioContext)();}catch(_){AC=null;return;}
    master=AC.createGain();master.gain.value=0.9;master.connect(AC.destination);
    var sr=AC.sampleRate,i;
    NB=AC.createBuffer(1,sr*2,sr);var d=NB.getChannelData(0);for(i=0;i<d.length;i++)d[i]=Math.random()*2-1;
    var BB=AC.createBuffer(1,sr*6,sr),bd=BB.getChannelData(0),l=0;for(i=0;i<bd.length;i++){l=(l+0.02*(Math.random()*2-1))/1.02;bd[i]=l*3.5;}
    var src=AC.createBufferSource();src.buffer=BB;src.loop=true;
    var lp=AC.createBiquadFilter();lp.type='lowpass';lp.frequency.value=420;
    var lfo=AC.createOscillator(),lg=AC.createGain();lfo.frequency.value=0.06;lg.gain.value=220;lfo.connect(lg);lg.connect(lp.frequency);lfo.start();
    var ag=AC.createGain();ag.gain.value=0.16;src.connect(lp);lp.connect(ag);ag.connect(master);src.start(); // wind / room tone
    var hum=AC.createOscillator(),hg=AC.createGain();hum.frequency.value=55;hg.gain.value=0.012;hum.connect(hg);hg.connect(master);hum.start(); // distant hum
    if(AC.state!=='running')try{AC.resume();}catch(_){}
  }
  ['pointerdown','touchend','click'].forEach(function(e){window.addEventListener(e,audioInit,true);});
  function nz(t,dur,type,f0,q,vol,f1){
    var s=AC.createBufferSource();s.buffer=NB;s.playbackRate.value=0.8+Math.random()*0.4;
    var f=AC.createBiquadFilter();f.type=type;f.frequency.setValueAtTime(f0,t);if(f1)f.frequency.exponentialRampToValueAtTime(f1,t+dur);f.Q.value=q;
    var g=AC.createGain();g.gain.setValueAtTime(vol,t);g.gain.exponentialRampToValueAtTime(0.001,t+dur);
    s.connect(f);f.connect(g);g.connect(master);s.start(t,Math.random()*1.5,dur+0.05);
  }
  function tn(t,dur,f0,f1,vol){
    var o=AC.createOscillator(),g=AC.createGain();o.frequency.setValueAtTime(f0,t);o.frequency.exponentialRampToValueAtTime(f1,t+dur);
    g.gain.setValueAtTime(vol,t);g.gain.exponentialRampToValueAtTime(0.001,t+dur);o.connect(g);g.connect(master);o.start(t);o.stop(t+dur+0.05);
  }
  var SFX={
    thud:function(d){var t=AC.currentTime;if(t-lastThud<0.05)return;lastThud=t;var v=Math.min(1,(d-2)/9);nz(t,0.14,'lowpass',260+40*d,1,0.7*v);tn(t,0.16,120,42,0.6*v);},
    crunch:function(big){var t=AC.currentTime,n=big?9:5;for(var i=0;i<n;i++)nz(t+i*0.014+Math.random()*0.01,0.045,'bandpass',1100+Math.random()*2800,2.5,big?0.7:0.5);tn(t,0.22,95,38,big?0.9:0.6);nz(t,0.3,'lowpass',700,1,big?0.45:0.25);},
    squelch:function(){var t=AC.currentTime;nz(t,0.4,'bandpass',1500,6,0.7,220);nz(t+0.06,0.3,'bandpass',900,9,0.5,160);nz(t+0.12,0.25,'bandpass',600,7,0.35,140);},
    bang:function(){var t=AC.currentTime;nz(t,0.09,'highpass',1800,0.7,1.0);nz(t,0.35,'lowpass',1400,0.8,0.9,180);tn(t,0.18,160,45,1.0);nz(t+0.02,0.6,'bandpass',500,0.6,0.25,120);},
    spurt:function(v){var t=AC.currentTime;nz(t,0.22,'bandpass',2600,1.4,0.18*v,800);},
    splat:function(v){var t=AC.currentTime;if(t-lastSplat<0.03)return;lastSplat=t;nz(t,0.07,'bandpass',600+Math.random()*500,3,0.12*v);}
  };
  function sfx(n,a){if(AC&&opt('oSound')&&AC.state==='running')SFX[n](a);}

  var parts=[],stains=[],MAXP=Infinity,gL=2,STICK=0.35;
  // draw layers, bottom to top: far arm, far leg, head and torso, near leg, near arm (blood, tears and strands go on the layer they came from)
  // draw slots, bottom to top: 0 far arm, 1 far leg, 2 head and torso, 3 near leg, 4 near arm
  function layerOf(k){if(k<4)return 2;var arm=(k-4)%6<3;return k<10?(arm?0:1):(arm?4:3);}
  // parts are numbered 0-15 for the body, then 16+ for chunks of shattered limbs (rig.chunks)
  function partOf(r,k){var nb=r.bodies.length;return k<nb?r.bodies[k]:r.chunks[k-nb].b;}
  function layerK(r,k){var nb=r.bodies.length;return layerOf(k<nb?k:r.chunks[k-nb].k);}
  function allParts(r){return r.chunks?r.bodies.concat(r.chunks.filter(function(c){return !c.gone;}).map(function(c){return c.b;})):r.bodies;}
  var GM=1; // gore setting / 10: scales how much blood comes out
  function gn(n){return Math.floor(n*GM+Math.random());}
  function gush(x,y,vx,vy,n,spd,spread,gib){
    n=gn(n);
    for(var i=0;i<n&&parts.length<MAXP;i++){
      var a=Math.random()*Math.PI*2,s=spd*(0.3+Math.random()*0.9);
      parts.push({x:x,y:y,vx:vx+Math.cos(a)*s*spread,vy:vy+Math.sin(a)*s*spread+(spread<1?s*0.2:0),r:gib?0.02+Math.random()*0.03:0.006+Math.random()*0.014,g:!!gib,life:6,L:gL,ck:!gib&&Math.random()<STICK});
    }
  }
  function spray(p,dx,dy,n,spd){ // directed jet
    n=gn(n);
    for(var i=0;i<n&&parts.length<MAXP;i++){
      var s=spd*(0.6+Math.random()*0.6),j=(Math.random()-0.5)*0.5,c=Math.cos(j),sn=Math.sin(j);
      parts.push({x:p.x,y:p.y,vx:(dx*c-dy*sn)*s,vy:(dx*sn+dy*c)*s,r:0.008+Math.random()*0.014,g:false,life:6,L:gL,ck:Math.random()<0.2});
    }
  }
  // blood soaked into the skin: splats stored in each body's own frame, so they move with it
  function addDecalL(r,k,lx,ly,s){
    var D=r.dec||(r.dec=[]),a=D[k]||(D[k]=[]);
    if(a.length>=18){var o=a[Math.random()*a.length|0];o.t=Math.min(0.06,o.t+s*0.3);return;}
    // starts small and soaks outwards to its full size; each splat has its own shade and opacity
    a.push({x:lx,y:ly,s:s*0.45,t:s*(1.25+Math.random()*0.6),a:Math.random()*6.28,c:Math.random()*4|0,o:0.45+Math.random()*0.5});
  }
  function addTrickle(r,k,tl,w){ // the path a drip ran down the skin, kept as a thin streak
    if(!tl||tl.length<6)return;var D=r.dln||(r.dln=[]),a=D[k]||(D[k]=[]);
    a.push({p:tl.slice(),w:w,c:Math.random()*4|0,o:0.5+Math.random()*0.4});if(a.length>10)a.shift();
  }
  function spreadDecals(r,dt){var D=r.dec;if(!D)return;for(var k in D){var a=D[k];if(!a)continue;for(var i=0;i<a.length;i++){var d=a[i];if(d.s<d.t)d.s+=(d.t-d.s)*Math.min(1,dt*0.7);}}}
  // lying in a pool on the floor: the side touching it soaks some up (stains the skin there, the pool shrinks a bit)
  function soak(r,dt){
    if(!stains.length)return;var nb=r.bodies.length,n=nb+(r.chunks?r.chunks.length:0);
    for(var k=0;k<n;k++){
      if(k>=nb&&r.chunks[k-nb].gone)continue;var b=partOf(r,k),f=b.getFixtureList();if(!f)continue;
      var sh=f.getShape(),vs=sh.m_vertices;if(!vs)continue;
      var c=b.getWorldCenter();if(c.y>0.4)continue; // nowhere near the floor
      for(var i=0;i<vs.length;i++){
        var w=b.getWorldPoint(vs[i]);if(w.y>0.03)continue;
        for(var j=stains.length-1;j>=0;j--){var st=stains[j];if(Math.abs(w.x-st.x)>st.w*0.5)continue;
          if(Math.random()<dt*Math.min(3,0.6+st.w*4)){
            var lp=b.getLocalPoint(Vec2(w.x+(c.x-w.x)*0.15*Math.random(),w.y+(c.y-w.y)*0.25*Math.random()));
            var A=r.dec&&r.dec[k],sz=0.014+Math.random()*0.016;
            if(A&&A.length>=18){var m=0,md=1e9,q;for(q=0;q<A.length;q++){var dd=Math.hypot(A[q].x-lp.x,A[q].y-lp.y);if(dd<md){md=dd;m=q;}}
              if(md<0.03)A[m].t=Math.min(0.06,A[m].t+sz*0.3);else{for(m=0,q=1;q<A.length;q++)if(A[q].t<A[m].t)m=q;A.splice(m,1);addDecalL(r,k,lp.x,lp.y,sz);}} // full: grow the splat already there, or swap out the smallest
            else addDecalL(r,k,lp.x,lp.y,sz);
            st.w=Math.max(0.02,st.w-0.004);st.h=Math.max(0.006,st.h-0.0004);
          }
          break;}
      }
    }
  }
  function addDecal(r,k,w,s){var lp=partOf(r,k).getLocalPoint(w);addDecalL(r,k,lp.x,lp.y,s);}
  function randPt(b){ // a random point inside a body part
    var vs=b.getFixtureList().getShape().m_vertices,v=vs[Math.random()*vs.length|0],u=vs[Math.random()*vs.length|0],t=Math.random()*0.8,m=Math.random();
    return b.getWorldPoint(Vec2((v.x*m+u.x*(1-m))*t,(v.y*m+u.y*(1-m))*t));
  }
  function splats(r,k,n,s){n=gn(n);for(var i=0;i<n;i++)addDecal(r,k,randPt(partOf(r,k)),s*(0.6+Math.random()*0.8));}
  function ooze(r,k,w){
    if(k<0||parts.length>=MAXP||Math.random()>GM)return;var b=partOf(r,k),f=b&&b.getFixtureList();if(!f)return;
    var q={x:w.x,y:w.y,vx:0,vy:0,r:0.007+Math.random()*0.007,g:false,life:8,ck:true,L:layerK(r,k),vol:4+(Math.random()*6|0)};
    stick(q,{r:r,k:k,b:b,f:f});parts.push(q);
  }
  function hitBody(q){ // which body part (if any) a drop is inside
    for(var i=0;i<rigs.length;i++){var bs=allParts(rigs[i]);
      for(var k=0;k<bs.length;k++){var b=bs[k];if(b===q.skip&&q.skT>0)continue;var f=b.getFixtureList();if(!f)continue;
        var ab=f.getAABB(0);if(q.x<ab.lowerBound.x||q.x>ab.upperBound.x||q.y<ab.lowerBound.y||q.y>ab.upperBound.y)continue;
        if(f.testPoint(Vec2(q.x,q.y)))return {r:rigs[i],k:k,b:b,f:f};}}
    return null;
  }
  function stick(q,h){var lp=h.b.getLocalPoint(Vec2(q.x,q.y));q.on=h.b;q.fx=h.f;q.rig=h.r;q.k=h.k;q.lx=lp.x;q.ly=lp.y;q.tl=null;q.L=layerK(h.r,h.k);if(q.vol==null)q.vol=2+(Math.random()*6|0);q.ph=Math.random()*6;}
  function stain(x,r){
    for(var i=stains.length-1;i>=Math.max(0,stains.length-40);i--){var s=stains[i];if(Math.abs(s.x-x)<s.w*0.6){s.w=Math.min(2,s.w+r*0.9);s.h=Math.min(0.05,s.h+r*0.12);return;}}
    stains.push({x:x,w:r*5,h:0.01+r*0.4});
  }
  function bodyOf(r,j){return r.rc[j].b;}
  function anchorA(c){return c.a.getWorldPoint(c.j.getLocalAnchorA());}
  function stumps(r,cb){ // each place a limb came off: the parent stump and the end of the torn-off piece
    var I=r.inj;if(!I)return;
    for(var k=3;k<RS.NJ;k++)if(I.gone[k]&&(k%3===0||!I.gone[k-1]))cb(r.rc[k],k);
  }
  function cullChunks(max){ // too many loose pieces slow the physics right down: clear the oldest
    var all=[];rigs.forEach(function(r){(r.chunks||[]).forEach(function(ch){if(!ch.gone&&ch.seen)all.push([ch,r]);});});
    if(all.length<=max)return;all.sort(function(a,b){return a[0].t0-b[0].t0;});
    for(var i=0;i<all.length-max;i++){var ch=all[i][0],r=all[i][1];
      if(r.strands)r.strands=r.strands.filter(function(s){return s.A!==ch.b&&s.B!==ch.b;});
      parts=parts.filter(function(q){return q.on!==ch.b;});world.destroyBody(ch.b);ch.gone=1;}
  }
  function goreStep(dt){
    cullChunks(50);
    var gore=opt('oGore');
    rigs.forEach(function(r){
      if(!r.bl)r.bl=new Float32Array(r.bodies.length);
      var evs=r.ev||[];
      evs.forEach(function(e){
        var c,p,b;
        if(e.t==='hit'){
          b=r.bodies[e.k];if(e.d>3.5)sfx('thud',e.d);
          if(gore&&e.d>6){p=randPt(b);gL=layerOf(e.k);gush(p.x,p.y,0,0,Math.round((e.d-5)*5),0.8+0.15*e.d,1);splats(r,e.k,1+(e.d/4|0),0.012+0.003*(e.d-6));}
        }else if(e.t==='break'){
          c=r.rc[e.k];sfx('crunch',false);
          if(gore&&e.d>0.5){p=anchorA(c);var kb=r.bodies.indexOf(c.b);gL=layerOf(kb);gush(p.x,p.y,0,0.5,Math.round(20+30*e.d),1.6+0.6*e.d,1);addDecal(r,kb,p,0.035);addDecal(r,r.bodies.indexOf(c.a),p,0.025);} // open fracture bleeds; a closed one is just the crack
        }else if(e.t==='crush'){
          sfx('crunch',true);sfx('squelch');
          b=e.k===0?r.bodies[0]:r.rc[e.k].b;p=b.getWorldCenter();
          if(gore){var kc=r.bodies.indexOf(b);gL=layerOf(kc);gush(p.x,p.y,0,1,140,2.4+0.12*e.d,1);gush(p.x,p.y,0,1.5,24,2.8,1,true);r.bl[kc]=1;splats(r,kc,10,0.03);}
          if(e.k===0)r.headCrushed=true;
        }else if(e.t==='crushT'){ // torso crushed (chest caved in, belly burst)
          sfx('crunch',true);sfx('squelch');b=r.bodies[e.k];p=b.getWorldCenter();
          if(gore){gL=layerOf(e.k);gush(p.x,p.y,0,1,180,2.6+0.1*e.d,1);gush(p.x,p.y,0,1.5,30,2.8,1,true);r.bl[e.k]=1;splats(r,e.k,14,0.035);}
        }else if(e.t==='snap'){
          sfx('squelch');
          if(gore){p=anchorA(r.rc[e.k]);gL=layerOf(r.bodies.indexOf(r.rc[e.k].b));gush(p.x,p.y,0,0.5,30,1.6,1);}
        }else if(e.t==='sever'){
          c=r.rc[e.k];sfx('squelch');sfx('crunch',true);
          if(gore){p=anchorA(c);var ka=r.bodies.indexOf(c.a),kb2=r.bodies.indexOf(c.b);gL=layerOf(kb2);gush(p.x,p.y,0,1,120,2.8,1);gush(p.x,p.y,0,1,14,2.2,1,true);
            for(var di=0;di<4;di++){addDecal(r,ka,p,0.04);addDecal(r,kb2,c.b.getWorldPoint(c.j.getLocalAnchorB()),0.04);}}
        }
      });
      if(r.ev)r.ev.length=0;
      if(r.chunks)for(var ci=0;ci<r.chunks.length;ci++){var ch=r.chunks[ci];if(ch.seen)continue;ch.seen=1;ch.t0=performance.now();var ck=r.bodies.length+ci;
        r.dec=r.dec||[];splats(r,ck,6,0.03);if(gore){var cp=ch.b.getPosition();gL=layerOf(ch.k);gush(cp.x,cp.y,0,0.5,20,1.5,1);}}
      // tears: small drops from the eye while they're hurting and awake (not gore, so always on)
      var hurtLvl=r.dead||r.koT>0?0:Math.max(r.pain||0,r.cowering?0.3:0);
      if(hurtLvl>0.12&&!r.headCrushed&&!(r.inj&&r.inj.gone[0])&&Math.random()<dt*(0.8+2.5*Math.min(1,hurtLvl))&&parts.length<MAXP){
        var hd=r.bodies[0],S=r.scale||1,lx=0.055*(r.dir||1)*S,ly=0.03*S,e=hd.getWorldPoint(Vec2(lx,ly));
        var tq={x:e.x,y:e.y,vx:0,vy:0,r:(0.011+Math.random()*0.005)*S,g:false,t:true,life:4,ck:true,L:2};stick(tq,{r:r,k:0,b:hd,f:hd.getFixtureList()});parts.push(tq);
      }
      if(!gore)return;
      // spurting stumps: pulses with the heartbeat, weaker and faster as the blood runs out; a dead heart just oozes
      var bl=r.blood==null?1:r.blood,alive=!r.dead;
      r.hb=(r.hb||0)+dt*(alive?1.2+1.8*(1-bl):0.3);
      var beat=alive?Math.pow(Math.max(0,Math.sin(r.hb*Math.PI*2)),3):0.15,newBeat=alive&&Math.sin(r.hb*Math.PI*2)>0.95&&!r.beatOn;
      r.beatOn=alive&&Math.sin(r.hb*Math.PI*2)>0.95;
      var sp=0;
      stumps(r,function(c,k){
        if(bl<=0.02)return;
        var pa=anchorA(c),ca=c.a.getWorldCenter(),dx=pa.x-ca.x,dy=pa.y-ca.y,dl=Math.hypot(dx,dy)||1;
        var n=Math.round((1+6*beat)*Math.min(1,bl*1.5));
        gL=layerOf(r.bodies.indexOf(c.a));if(n)spray(pa,dx/dl,dy/dl,n,(0.8+3.2*beat)*Math.min(1,0.3+bl));
        var pb=c.b.getWorldPoint(c.j.getLocalAnchorB());if(Math.random()<0.25){gL=layerOf(r.bodies.indexOf(c.b));gush(pb.x,pb.y,0,0,1,0.3,1);} // the torn-off piece drips
        sp++;
      });
      if(r.heart&&alive&&bl>0.02){ // shot through the heart: it pumps out of the hole with every beat
        var ch=r.bodies[1],hp2=ch.getWorldPoint(Vec2(r.heart.x,r.heart.y)),cc=ch.getWorldCenter(),hx=hp2.x-cc.x,hy=hp2.y-cc.y,hl=Math.hypot(hx,hy)||1;
        var hn=Math.round((1+12*beat)*Math.min(1,bl*1.5));gL=layerK(r,1);if(hn)spray(hp2,hx/hl,hy/hl,hn,(0.8+4*beat)*Math.min(1,0.3+bl));
        if(newBeat)sfx('spurt',Math.min(1,bl+0.3));
      }
      if(newBeat&&sp)sfx('spurt',Math.min(1,bl+0.2));
      spreadDecals(r,dt);soak(r,dt);
      // wounds ooze: drips start right on the skin at stumps and open fractures and trickle down the limb
      if(bl>0.05){
        stumps(r,function(c){if(Math.random()<dt*(r.dead?0.6:2.2)*Math.min(1,bl*1.5))ooze(r,r.bodies.indexOf(c.a),anchorA(c));
          if(Math.random()<dt*0.8)ooze(r,r.bodies.indexOf(c.b),c.b.getWorldPoint(c.j.getLocalAnchorB()));});
        if(r.inj)for(var kk=0;kk<RS.NJ;kk++)if(r.inj.broken[kk]&&r.inj.open&&r.inj.open[kk]&&!r.inj.gone[kk]&&Math.random()<dt*0.5)ooze(r,r.bodies.indexOf(r.rc[kk].b),anchorA(r.rc[kk]));
        if(r.chunks)r.chunks.forEach(function(ch,ci){if(!ch.gone&&Math.random()<dt*0.4)ooze(r,r.bodies.length+ci,ch.b.getPosition());});
        if(r.hole&&bl>0.02)r.hole.forEach(function(h){var hb=partOf(r,h.k);if(hb&&Math.random()<dt*(r.dead?0.4:1.4))ooze(r,h.k,hb.getWorldPoint(Vec2(h.x,h.y)));});
      }
      var I=r.inj;if(I)for(var k=0;k<RS.NJ;k++)if(I.broken[k]&&I.open&&I.open[k]&&Math.random()<0.04){var q=anchorA(r.rc[k]);gL=layerOf(r.bodies.indexOf(r.rc[k].b));gush(q.x,q.y,0,0,1,0.2,1);} // open fractures drip
    });
    // particles: fall, land on the ground (stain + tiny splat sound)
    for(var i=parts.length-1;i>=0;i--){
      var q=parts[i];
      if(q.on){ // running over the skin: trickles downhill across the part (leaving a smear), drips off the edge
        var b=q.on,ha=b.getAngle(),c=Math.cos(ha),sn=Math.sin(ha),spd=q.t?0.12:(q.spd||(q.spd=0.07+Math.random()*0.12)),wob=0.35*Math.sin(q.ph+=dt*5);
        q.lx+=(-sn+wob*c)*spd*dt;q.ly+=(-c-wob*sn)*spd*dt;
        var tl=q.tl||(q.tl=[q.lx,q.ly]);if(Math.hypot(tl[tl.length-2]-q.lx,tl[tl.length-1]-q.ly)>0.008){tl.push(q.lx,q.ly);if(tl.length>16)tl.splice(0,2);}
        var wp=b.getWorldPoint(Vec2(q.lx,q.ly));q.x=wp.x;q.y=wp.y;q.life-=q.t?dt:dt*0.4;
        if(!q.t){q.tr=(q.tr||0)+spd*dt;if(q.tr>0.018){q.tr=0;addDecalL(q.rig,q.k,q.lx,q.ly,q.r*0.8);q.vol--;}}
        if(q.life<=0||(!q.t&&q.vol<=0)||rigs.indexOf(q.rig)<0){if(!q.t&&rigs.indexOf(q.rig)>=0)addTrickle(q.rig,q.k,q.tl,q.r*0.7);parts.splice(i,1);continue;}
        if(q.fx!==b.getFixtureList())q.fx=b.getFixtureList(); // the part may have been shattered (new, smaller fixture)
        if(q.fx&&q.fx.testPoint(wp))continue;
        if(!q.t)addTrickle(q.rig,q.k,q.tl,q.r*0.7);
        var hv=b.getLinearVelocityFromWorldPoint(wp);q.vx=hv.x;q.vy=hv.y;q.skip=b;q.skT=0.12;q.on=null;
      }
      q.vy-=10*dt;q.x+=q.vx*dt;q.y+=q.vy*dt;q.life-=dt;
      if(q.ck&&!q.g){if(q.skT>0)q.skT-=dt;var h=hitBody(q);if(h){stick(q,h);continue;}}
      if(q.y<=q.r*0.5){
        if(q.g){q.y=q.r*0.5;q.vy*=-0.25;q.vx*=0.5;if(Math.abs(q.vy)<0.2){stain(q.x,q.r*0.8);q.life=Math.min(q.life,0);}}
        else if(q.t)q.life=0;
        else{stain(q.x,q.r);if(q.r>0.012)sfx('splat',Math.min(1,-q.vy/5));q.life=0;}
      }
      if(q.life<=0)parts.splice(i,1);
    }
  }
  var groundPat=null,skinPat=null;
  function groundPattern(){ // grey concrete, 64 px = 1 m, made once
    if(groundPat)return groundPat;
    var c=document.createElement('canvas');c.width=c.height=64;var g=c.getContext('2d'),i;
    g.fillStyle='#1c1a17';g.fillRect(0,0,64,64);
    for(i=0;i<500;i++){var v=18+Math.random()*26|0;g.fillStyle='rgba('+v+','+(v+3)+','+(v+8)+','+(0.3+Math.random()*0.4)+')';g.fillRect(Math.random()*64|0,Math.random()*64|0,1,1);}
    g.fillStyle='rgba(0,0,0,0.18)';g.fillRect(0,0,1,64);g.fillRect(0,0,64,1); // slab seams
    groundPat=ctx.createPattern(c,'repeat');return groundPat;
  }
  function skinPattern(){ // pixel speckle laid over the flesh colour, 32 px = 0.5 m
    if(skinPat)return skinPat;
    var c=document.createElement('canvas');c.width=c.height=32;var g=c.getContext('2d');
    for(var i=0;i<140;i++){g.fillStyle=Math.random()<0.5?'rgba(0,0,0,0.10)':'rgba(255,255,255,0.08)';g.fillRect(Math.random()*32|0,Math.random()*32|0,1,1);}
    skinPat=ctx.createPattern(c,'repeat');return skinPat;
  }
  function worldPat(pat,px){ // fill with a pattern pinned to the world: px pattern pixels per metre, scales with zoom
    var o=toScreen(Vec2(0,0)),k=cam.z/px;
    if(pat.setTransform)pat.setTransform(new DOMMatrix([k,0,0,k,o.x,o.y]));
    return pat;
  }
  function drawGround(gy){ // the arena: a 1 m concrete shell (floor, two walls, roof) with nothing outside it
    var TH=1,a=toScreen(Vec2(-ARENA-TH,ROOF+TH)),b=toScreen(Vec2(ARENA+TH,-TH)),c=toScreen(Vec2(-ARENA,ROOF)),d=toScreen(Vec2(ARENA,0)),ew=Math.max(2,0.03*cam.z);
    ctx.save();ctx.imageSmoothingEnabled=false;ctx.fillStyle=worldPat(groundPattern(),64);
    ctx.beginPath();ctx.rect(a.x,a.y,b.x-a.x,b.y-a.y);ctx.rect(c.x,c.y,d.x-c.x,d.y-c.y);ctx.fill('evenodd'); // shell = outer box minus the inside
    ctx.fillStyle='#3a3329';ctx.fillRect(c.x,d.y,d.x-c.x,ew);ctx.fillRect(c.x,c.y-ew,d.x-c.x,ew);ctx.fillRect(c.x-ew,c.y,ew,d.y-c.y);ctx.fillRect(d.x,c.y,ew,d.y-c.y); // inner edges
    ctx.strokeStyle='rgba(255,176,32,0.35)';ctx.lineWidth=1;ctx.strokeRect(c.x,c.y,d.x-c.x,d.y-c.y); // thin amber line round the inside
    ctx.fillStyle='#4a0606';
    stains.forEach(function(s){var a=toScreen(Vec2(s.x,0));ctx.globalAlpha=0.92;ctx.beginPath();ctx.ellipse(a.x,a.y+1,Math.max(0.5,s.w*cam.z*0.5),Math.max(0.4,s.h*cam.z),0,0,Math.PI*2);ctx.fill();});
    ctx.restore();
  }
  function mix(a,b,t){return [a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];}
  function rgb(c,m){m=m||1;return 'rgb('+(c[0]*m|0)+','+(c[1]*m|0)+','+(c[2]*m|0)+')';}
  var SKIN=[226,174,132],PALE=[205,196,182],BLOOD=[120,6,8];
  function skinTone(){var t=Math.random(),c=mix([238,192,152],[196,140,102],t),j=function(){return (Math.random()-0.5)*14;};return [c[0]+j(),c[1]+j(),c[2]+j()];}
  function drawCrushedHead(b,base,far){ // flattened stump of a head: jaw and neck left, burst top, skull chips (all in world units so it scales with zoom)
    var p=toScreen(b.getPosition()),z=cam.z,J=[[-0.09,-0.02],[-0.06,0.01],[-0.035,-0.025],[-0.01,0.015],[0.02,-0.03],[0.045,0.005],[0.07,-0.02],[0.09,0.0]];
    ctx.save();ctx.translate(p.x,p.y);ctx.rotate(-b.getAngle());ctx.scale(z,-z);
    function edge(dy){ctx.moveTo(-0.09,-0.115);ctx.lineTo(0.09,-0.115);for(var i=J.length-1;i>=0;i--)ctx.lineTo(J[i][0],J[i][1]+dy);ctx.closePath();}
    ctx.beginPath();ctx.moveTo(-0.11,-0.04);ctx.quadraticCurveTo(-0.13,0.03,-0.07,0.045);ctx.quadraticCurveTo(0,0.06,0.07,0.04);ctx.quadraticCurveTo(0.13,0.02,0.11,-0.045);ctx.closePath();
    ctx.fillStyle=rgb([95,4,6],far?0.72:1);ctx.fill(); // burst mush
    ctx.beginPath();edge(0);ctx.fillStyle=rgb(mix(base,BLOOD,0.45),far?0.72:1);ctx.fill();
    ctx.strokeStyle='#1b1b1d';ctx.lineWidth=0.012;ctx.stroke();
    ctx.beginPath();for(var i=0;i<J.length;i++){var q=J[i];if(i===0)ctx.moveTo(q[0],q[1]);else ctx.lineTo(q[0],q[1]);}
    ctx.strokeStyle=rgb(BLOOD,0.8);ctx.lineWidth=0.02;ctx.stroke(); // raw torn edge
    ctx.fillStyle='#d9c9b0';[[-0.07,0.03,0.03,0.018,0.4],[-0.015,0.04,0.024,0.014,-0.3],[0.05,0.03,0.028,0.016,0.2]].forEach(function(c){ // skull fragments
      ctx.save();ctx.translate(c[0],c[1]);ctx.rotate(c[4]);ctx.fillRect(-c[2]/2,-c[3]/2,c[2],c[3]);ctx.restore();});
    ctx.restore();
  }
  function bodyPat(pat,b,px){ // pattern stuck to a body part (moves and turns with it), px pattern pixels per metre
    var o=toScreen(b.getPosition()),k=cam.z/px,a=b.getAngle(),c=Math.cos(a),sn=Math.sin(a);
    if(pat.setTransform)pat.setTransform(new DOMMatrix([k*c,-k*sn,-k*sn,-k*c,o.x,o.y]));
    return pat;
  }
  var SHADE=[[150,14,18],[118,8,12],[86,4,8],[66,10,8]]; // fresh to old, dark and brownish
  function shade(c,o,far){var q=SHADE[c],m=far?0.75:1;return 'rgba('+(q[0]*m|0)+','+(q[1]*m|0)+','+(q[2]*m|0)+','+o+')';}
  function drawHoles(b,A,far){ // bruised skin, a ragged wet red rim, then the dark hole (exit wounds bigger and more torn)
    if(!A||!A.length)return;var z=cam.z,an=b.getAngle();
    function blob(p,rr,h,sc){ctx.beginPath();for(var q=0;q<10;q++){var t=h.a+an*0+q*0.628,m=rr*(1+(h.j[q]-1)*sc);var x=p.x+Math.cos(t)*m,y=p.y+Math.sin(t)*m;if(q)ctx.lineTo(x,y);else ctx.moveTo(x,y);}ctx.closePath();ctx.fill();}
    for(var i=0;i<A.length;i++){var h=A[i],p=toScreen(b.getWorldPoint(Vec2(h.x,h.y))),rr=Math.max(h.exit?4.5:3.5,h.r*z),e=h.exit?1.6:1;
      if(!h.j){h.j=[1,1,1,1,1,1,1,1,1,1];h.a=0;}
      if(h.cut){ctx.save();ctx.translate(p.x,p.y);ctx.rotate(-(an+h.ca));var L=rr*h.len*1.6; // a slit: red lips around a dark gash
        ctx.fillStyle=far?'#6a1414':'#a82219';ctx.beginPath();ctx.ellipse(0,0,L,rr*0.75,0,0,7);ctx.fill();
        ctx.fillStyle='#120303';ctx.beginPath();ctx.ellipse(0,0,L*0.85,rr*0.28,0,0,7);ctx.fill();ctx.restore();continue;}
      ctx.fillStyle=far?'rgba(40,10,30,0.35)':'rgba(70,20,50,0.35)';ctx.beginPath();ctx.arc(p.x,p.y,rr*2.6,0,7);ctx.fill(); // bruise
      ctx.fillStyle=far?'#5a0f0f':'#8e1a16';blob(p,rr*1.7,h,0.8*e); // torn flesh
      ctx.fillStyle=far?'#7a1a14':'#b8281e';blob(p,rr*1.25,h,0.6*e); // wet red edge
      ctx.fillStyle='#0d0202';blob(p,rr*(h.exit?0.85:0.75),h,0.5*e); // the hole
      ctx.fillStyle='rgba(255,200,190,0.35)';ctx.beginPath();ctx.arc(p.x-rr*0.35,p.y-rr*0.35,Math.max(0.6,rr*0.18),0,7);ctx.fill();} // glint of wet
  }
  function drawDecals(b,D,far,LN){
    var i,d,z=cam.z,o0=toScreen(b.getPosition()),an=b.getAngle(),ca=Math.cos(an),sa=Math.sin(an);
    for(var g=0;g<12;g++){ // 4 shades x 3 opacities, one path each
      var c=g>>2,ol=g&3;if(ol===3)continue;var any=false;
      for(i=0;i<D.length;i++){d=D[i];if(d.c!==c||(d.ol!=null?d.ol:(d.ol=d.o<0.6?0:d.o<0.8?1:2))!==ol)continue;
        if(!any){ctx.fillStyle=shade(c,[0.55,0.72,0.9][ol],far);ctx.beginPath();any=true;}
        var px=o0.x+z*(ca*d.x-sa*d.y),py=o0.y-z*(sa*d.x+ca*d.y),rr=Math.max(0.4,d.s*z);
        ctx.moveTo(px+rr,py);ctx.arc(px,py,rr,0,7);if(rr<4)continue; // satellites only when big enough to see
        var ox=Math.cos(d.a)*rr*0.9,oy=Math.sin(d.a)*rr*0.9;ctx.moveTo(px+ox+rr*0.55,py+oy);ctx.arc(px+ox,py+oy,rr*0.55,0,7);}
      if(any)ctx.fill();}
    if(LN&&LN.length){ctx.lineCap='round';ctx.lineJoin='round';
      for(i=0;i<LN.length;i++){var ln=LN[i];ctx.beginPath();for(var j=0;j<ln.p.length;j+=2){var pp=toScreen(b.getWorldPoint(Vec2(ln.p[j],ln.p[j+1])));if(j)ctx.lineTo(pp.x,pp.y);else ctx.moveTo(pp.x,pp.y);}
        ctx.strokeStyle=shade(ln.c,ln.o,far);ctx.lineWidth=Math.max(0.5,ln.w*cam.z);ctx.stroke();}
      ctx.lineCap='butt';ctx.lineJoin='miter';}
  }
  function drawFlesh(L,far){ // filled, shaded body parts of one layer; paler as it bleeds out, blood splats where hurt
    rigs.forEach(function(r){
      var bl=r.blood==null?1:r.blood,base=mix(r.skin||SKIN,PALE,Math.min(1,(1-bl)*1.4+(r.dead?0.35:0)));
      r.bodies.forEach(function(b,k){
        if(layerOf(k)!==L)return;
        var u=b.getUserData()||{};if(!!u.far!==far)return;
        var f=b.getFixtureList();if(!f)return;
        if(k===0&&r.headCrushed){drawCrushedHead(b,base,far);return;}
        var c=mix(base,BLOOD,Math.min(0.3,(r.bl?r.bl[k]:0)*0.3)); // only a crushed part darkens all over
        ctx.beginPath();polyPath(b,f.getShape());ctx.closePath();ctx.fillStyle=rgb(c,far?0.72:1);ctx.fill();ctx.fillStyle=bodyPat(skinPattern(),b,64);ctx.fill();
        var D=(r.dec&&r.dec[k])||[],LN=r.dln&&r.dln[k];if(D.length||LN&&LN.length){ctx.save();ctx.clip();drawDecals(b,D,far,LN);drawHoles(b,r.holeK&&r.holeK[k],far);ctx.restore();ctx.beginPath();polyPath(b,f.getShape());ctx.closePath();}
        ctx.strokeStyle='#1b1b1d';ctx.lineWidth=Math.max(1,0.012*cam.z);ctx.stroke();
        if(k===1)drawNumber(r);
      });
      if(r.chunks)r.chunks.forEach(function(ch,ci){ // pieces of a shattered limb: raw and bloody
        if(ch.gone||layerOf(ch.k)!==L)return;var b=ch.b,u=b.getUserData()||{};if(!!u.far!==far)return;var f=b.getFixtureList();if(!f)return;
        ctx.beginPath();polyPath(b,f.getShape());ctx.fillStyle=rgb(mix(base,BLOOD,0.3),far?0.72:1);ctx.fill();ctx.fillStyle=bodyPat(skinPattern(),b,64);ctx.fill();
        var D=(r.dec&&r.dec[r.bodies.length+ci])||[],LN=r.dln&&r.dln[r.bodies.length+ci];if(D.length||LN&&LN.length){ctx.save();ctx.clip();drawDecals(b,D,far,LN);drawHoles(b,r.holeK&&r.holeK[r.bodies.length+ci],far);ctx.restore();ctx.beginPath();polyPath(b,f.getShape());}
        ctx.strokeStyle='#1b1b1d';ctx.lineWidth=Math.max(1,0.012*cam.z);ctx.stroke();
      });
    });
  }
  function drawStrands(L){ // red strings of muscle and gut still joining a torn-off limb to the stump
    rigs.forEach(function(r){
      (r.strands||[]).forEach(function(s){
        if(layerOf(s.k+1)!==L)return;
        var a=toScreen(s.A.getWorldPoint(s.la)),b=toScreen(s.B.getWorldPoint(s.lb)),dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy)||1;
        var slack=Math.max(0,s.L*cam.z-d),taut=Math.min(1,d/(s.L*cam.z+1e-6));
        for(var i=0;i<s.n;i++){
          var o=(i-(s.n-1)/2)*0.018*cam.z*(1-0.7*taut),sg=slack*0.5+0.01*cam.z*Math.sin(s.seed+i*2.1);
          var mx=(a.x+b.x)/2-dy/d*o,my=(a.y+b.y)/2+dx/d*o+sg;
          var w=Math.max(s.thin?0.8:1.5,(0.022-0.005*(i%3))*cam.z*(1-0.5*taut)*(s.thin?0.45:1));
          ctx.beginPath();ctx.moveTo(a.x-dy/d*o*0.4,a.y+dx/d*o*0.4);ctx.quadraticCurveTo(mx,my,b.x-dy/d*o*0.4,b.y+dx/d*o*0.4);
          ctx.strokeStyle='#3a0204';ctx.lineWidth=w+2;ctx.lineCap='round';ctx.stroke();
          ctx.strokeStyle=i%2?'#b8323a':'#8e0f14';ctx.lineWidth=w;ctx.stroke();
        }
      });
    });
    ctx.lineCap='butt';
  }
  // liquid drawn as streaks: a falling drop is a short line along its velocity, a drop running over the skin is
  // a trickle through its recent path; widths come from the drop size in metres, so it all scales with zoom
  function drawBlood(L){
    var i,q,sc,rr,z=cam.z,B={},T={},k;
    function seg(bk,w,pts){var key=Math.max(0.5,Math.round(w*2)/2);(bk[key]||(bk[key]=[])).push(pts);}
    for(i=0;i<parts.length;i++){q=parts[i];if(q.g||(q.L==null?2:q.L)!==L)continue;
      var bk=q.t?T:B,w=q.r*z*(q.t?0.75:0.9),pts=[];
      if(q.on&&q.tl){for(k=0;k<q.tl.length;k+=2){sc=toScreen(q.on.getWorldPoint(Vec2(q.tl[k],q.tl[k+1])));pts.push(sc.x,sc.y);}sc=toScreen(q);pts.push(sc.x,sc.y);w*=0.8;}
      else{sc=toScreen(q);var vx=q.vx*0.035*z,vy=-q.vy*0.035*z,vl=Math.hypot(vx,vy),mx=Math.max(w*1.5,0.12*z);if(vl>mx){vx*=mx/vl;vy*=mx/vl;}pts.push(sc.x-vx,sc.y-vy,sc.x,sc.y);}
      seg(bk,w,pts);}
    ctx.lineCap='round';ctx.lineJoin='round';
    function flush(bk,col){for(var key in bk){var a=bk[key];ctx.beginPath();for(var j=0;j<a.length;j++){var p=a[j];ctx.moveTo(p[0],p[1]);for(var m=2;m<p.length;m+=2)ctx.lineTo(p[m],p[m+1]);if(p.length===2)ctx.lineTo(p[0]+0.01,p[1]);}ctx.lineWidth=+key;ctx.strokeStyle=col;ctx.stroke();}}
    flush(B,'rgba(118,6,9,0.93)');
    flush(T,'rgba(186,208,222,0.55)'); // tears: clear, faintly blue-grey
    ctx.lineCap='butt';ctx.lineJoin='miter';
    ctx.fillStyle='#5a0505';ctx.beginPath();
    for(i=0;i<parts.length;i++){q=parts[i];if(!q.g||(q.L==null?2:q.L)!==L)continue;sc=toScreen(q);rr=Math.max(0.6,q.r*z);ctx.moveTo(sc.x+rr,sc.y);ctx.arc(sc.x,sc.y,rr,0,7);}
    ctx.fill();
  }
  function draw(){
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.fillStyle='#0c0b0a';ctx.fillRect(0,0,W,H); // near-black backdrop (matches the menus)
    var fg='#e9e4d8';
    ctx.strokeStyle=fg;ctx.fillStyle=fg;ctx.lineWidth=2;
    var gy=toScreen(Vec2(0,0)).y;
    var g0=toWorld(0,H),g1=toWorld(W,0),gx,gyy; // 1 m grid
    ctx.save();ctx.globalAlpha=0.07;ctx.strokeStyle='#ffb020';ctx.lineWidth=1;ctx.beginPath();
    for(gx=Math.ceil(g0.x);gx<=g1.x;gx++){var sx=toScreen(Vec2(gx,0)).x;ctx.moveTo(sx,0);ctx.lineTo(sx,H);}
    for(gyy=Math.ceil(g0.y);gyy<=g1.y;gyy++){var sy=toScreen(Vec2(0,gyy)).y;ctx.moveTo(0,sy);ctx.lineTo(W,sy);}
    ctx.stroke();ctx.restore();
    drawGround(gy);
    for(var L=0;L<5;L++){drawFlesh(L,true);drawFlesh(L,false);drawStrands(L);drawBlood(L);} // far arm, far leg, body, near leg, near arm on top
    drawItems();
    if(opt('oBox')){ctx.globalAlpha=0.4;drawBodies(true);ctx.globalAlpha=1;drawBodies(false);}
    { // name above the head (only once you've named them)
      ctx.font='500 18px Teko,system-ui,Arial,sans-serif';ctx.textAlign='center';ctx.fillStyle=fg;
      rigs.forEach(function(r){if(!r.name)return;var hp=toScreen(r.bodies[0].getPosition());ctx.fillText(r.name,hp.x,hp.y-0.2*cam.z);});
    }
    if(opt('oSkel')){
      ctx.lineWidth=3;ctx.strokeStyle='#e8590c';
      for(var b=world.getBodyList();b;b=b.getNext()){
        var u=b.getUserData();if(!u||!u.bone)continue;
        ctx.globalAlpha=u.far?0.4:1;
        var p1=toScreen(b.getWorldPoint(u.bone[0])),p2=toScreen(b.getWorldPoint(u.bone[1]));
        ctx.beginPath();ctx.moveTo(p1.x,p1.y);ctx.lineTo(p2.x,p2.y);ctx.stroke();
      }
      ctx.globalAlpha=1;
    }
    if(opt('oJoint')){
      ctx.fillStyle='#1971c2';
      for(var j=world.getJointList();j;j=j.getNext()){var a=toScreen(j.getAnchorA());ctx.beginPath();ctx.arc(a.x,a.y,4,0,Math.PI*2);ctx.fill();}
    }
    if(opt('oCom')){
      rigs.forEach(function(r){
        var mx=0,my=0,M=0;
        r.bodies.forEach(function(b){var c=b.getWorldCenter(),m=b.getMass();mx+=c.x*m;my+=c.y*m;M+=m;});
        var com=Vec2(mx/M,my/M),cs=toScreen(com),gs=toScreen(Vec2(com.x,0));
        var lo=1e9,hi=-1e9;
        r.feet.forEach(function(b){
          var vs=b.getFixtureList().getShape().m_vertices;
          for(var i=0;i<vs.length;i++){var w=b.getWorldPoint(vs[i]);if(w.y<0.03){lo=Math.min(lo,w.x);hi=Math.max(hi,w.x);}}
        });
        if(hi>lo){
          var a=toScreen(Vec2(lo,0)),c=toScreen(Vec2(hi,0));
          ctx.strokeStyle='#2f9e44';ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(c.x,c.y);ctx.stroke();
        }
        ctx.strokeStyle='#e03131';ctx.fillStyle='#e03131';ctx.lineWidth=1.5;
        ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(cs.x,cs.y);ctx.lineTo(gs.x,gs.y);ctx.stroke();ctx.setLineDash([]);
        ctx.beginPath();ctx.arc(cs.x,cs.y,6,0,Math.PI*2);ctx.fill();
      });
    }
  }

  // INJURIES panel: tick boxes show the state of the rigs (any rig) and change it for all of them
  function injSync(){
    var k,b,g,ko=false,dead=false;
    for(k=0;k<RS.NJ;k++){
      b=false;g=rigs.length>0;
      rigs.forEach(function(r){var I=r.inj;if(I&&I.broken[k])b=true;if(!(I&&I.gone[k]))g=false;});
      var e=document.getElementById('iB'+k);e.checked=b;e.disabled=g;e.parentNode.style.opacity=g?0.4:1;
    }
    rigs.forEach(function(r){if(r.koT>0)ko=true;if(r.dead)dead=true;});
    document.getElementById('iKO').checked=ko;document.getElementById('iDead').checked=dead;
  }
  function setBone(k,on){rigs.forEach(function(r){RS.breakBone(r,k,on,0);if(k===0&&!on)r.dead=false;});}
  for(var bi=0;bi<RS.NJ;bi++)(function(k){document.getElementById('iB'+k).onchange=function(){setBone(k,this.checked);injSync();};})(bi);
  document.getElementById('iKO').onchange=function(){var on=this.checked;rigs.forEach(function(r){r.koT=on?1e9:0;if(!on)r.age=0;});};
  document.getElementById('iDead').onchange=function(){var on=this.checked;rigs.forEach(function(r){r.dead=on;if(!on){RS.breakBone(r,0,false);r.koT=0;r.age=0;r.gu.ph=-1;}});};
  document.getElementById('bBreakAll').onclick=function(){for(var k=1;k<RS.NJ;k++)setBone(k,true);injSync();}; // not the neck: that kills it
  document.getElementById('bHealAll').onclick=function(){for(var k=0;k<RS.NJ;k++)setBone(k,false);injSync();};
  function drawInjuries(){ // red dot on broken joints, red stump where a limb came off
    ctx.fillStyle='#e03131';
    rigs.forEach(function(r){
      var I=r.inj;if(!I)return;
      for(var k=0;k<RS.NJ;k++){
        if(!I.broken[k]&&!(I.gone[k]&&(k===3||k===6||k===9||k===12||!I.gone[k-1])))continue;
        var c=r.rc[k],a=toScreen(c.a.getWorldPoint(c.j.getLocalAnchorA()));
        var shut=!I.gone[k]&&!(I.open&&I.open[k]);ctx.fillStyle=shut?'#4a2f52':'#e03131'; // closed break: a bruise, not blood
        ctx.globalAlpha=(k>=3&&k<9?0.5:1)*(shut?0.7:1);ctx.beginPath();ctx.arc(a.x,a.y,Math.max(1.5,(I.gone[k]?0.05:0.035)*cam.z),0,Math.PI*2);ctx.fill();
        if(I.gone[k]){var b=toScreen(c.b.getWorldPoint(c.j.getLocalAnchorB()));ctx.beginPath();ctx.arc(b.x,b.y,Math.max(1.5,0.045*cam.z),0,Math.PI*2);ctx.fill();}
      }
    });
    ctx.globalAlpha=1;
  }

  var last=performance.now(),acc=0,injN=0;
  function frame(t){ // one bad frame must never stop the game: log it and keep going
    try{frameInner(t);}catch(e){if(window.console)console.error(e);}
    requestAnimationFrame(frame);
  }
  function frameInner(t){
    var dt=Math.min((t-last)/1000,0.05);last=t;
    if(world){var gw=gyro.on&&gyro.lis?Vec2(gyro.gx,gyro.gy):Vec2(0,-10),g0=world.getGravity();if(Math.abs(g0.x-gw.x)+Math.abs(g0.y-gw.y)>1e-3){world.setGravity(gw);world.getBodyList()&&rigs.forEach(function(r){r.bodies.forEach(function(b){b.setAwake(true);});});items.forEach(function(g){g.b.setAwake(true);});}}
    cam.x=Math.max(-ARENA+1,Math.min(ARENA-1,cam.x));cam.y=Math.max(-1,Math.min(ROOF+1,cam.y)); // camera stays in the arena
    items.forEach(function(g){var q=g.b.getPosition();if(!g.stuck&&(q.y<-0.03||q.y>ROOF||Math.abs(q.x)>ARENA)){g.b.setTransform(Vec2(Math.max(-ARENA+0.3,Math.min(ARENA-0.3,q.x)),0.2),g.b.getAngle());g.b.setLinearVelocity(Vec2(0,0));}}); // anything that got pushed through the floor or a wall pops back out
    var paused=userPaused||training&&pool.mode!=='workers'; // main-thread training: no time left for the live view
    if(!paused){
      if(opt('oSlow'))dt*=0.25;
      acc+=dt;
      var use=opt('oUse'),pd=opt('oPD'),reflex=opt('oReflex'),i;
      RS.FLAGS.air=RS.FLAGS.land=RS.FLAGS.fall=opt('oFall')?1:0;
      RS.FLAGS.step=opt('oStep')?1:0;RS.FLAGS.bal2=opt('oBal2')?1:0;RS.FLAGS.nn=opt('oNN')?1:0;RS.FLAGS.getup=opt('oNN')&&opt('oGetup')?1:0;RS.FLAGS.cower=opt('oCower')?1:0;RS.FLAGS.die=opt('oDie')?1:0;RS.FLAGS.protect=opt('oProtect')?1:0;RS.FLAGS.inj=opt('oInj')?1:0;var gv=goreV();RS.FLAGS.sever=opt('oSever')&&gv>=4?1:0;RS.FLAGS.shatter=gv>=7?1:0;RS.FLAGS.crush=1;RS.FLAGS.bleed=gv>0?1:0;RS.FLAGS.bleedMul=gv/10;RS.FLAGS.death=gv>0?1:0;if(!gv)RS.FLAGS.die=0;
      RS.FLAGS.realism=+$('rReal').value;RS.FLAGS.crushLimb=+$('rCrush').value/7;RS.FLAGS.crushImp=14*RS.FLAGS.crushLimb; // settings: 7 (default) = real-world speeds
      while(acc>=DT){
        if((use||reflex)&&stepCount%SUB===0)for(i=0;i<rigs.length;i++)RS.act(rigs[i],use?ES.theta:ZERO_TH,liveBufs[i],liveRng,{delay:1,noise:0.005,reflex:reflex}); // reflexes run even when the policy is off (untrained = zero weights)
        if(pd)for(i=0;i<rigs.length;i++)RS.pdRig(rigs[i]);
        applyDrag();world.step(DT,RS.VEL_IT,RS.POS_IT);meleeStep();acc-=DT;stepCount++;
      }
    }
    if(camFollow&&rigs.indexOf(camFollow)<0)camFollow=null;
    if(camFollow){var fp=camFollow.bodies[3].getPosition();cam.x+=(fp.x-cam.x)*0.1;cam.y+=(fp.y+0.3-cam.y)*0.1;}
    if(camT){cam.x+=(camT.x-cam.x)*0.15;cam.y+=(camT.y-cam.y)*0.15;if(Math.abs(camT.x-cam.x)+Math.abs(camT.y-cam.y)<0.005)camT=null;}
    if(!paused){goreStep(Math.min(dt,0.05));fxStep(Math.min(dt,0.05));}
    draw();if(++injN%10===0){injSync();hud();}
  }
  window.EQ={rigs:function(){return rigs;},parts:function(){return parts;},cam:function(){return cam;},follow:function(){return camFollow;},items:function(){return items;},fire:fire,fx:function(){return fx;},toScreen:toScreen,drag:function(){return drag;}}; // for poking at it from the console
  init();updateInfo();requestAnimationFrame(frame);
})();
