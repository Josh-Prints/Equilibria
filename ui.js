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
  function init(){
    world=new planck.World({gravity:Vec2(0,-10)});
    drag=null;group=0;rigs=[];
    ground=world.createBody();ground.createFixture(Box(1000,1,Vec2(0,-1),0),{friction:0.9});
    spawn('stand');
  }
  var liveBufs=[];
  function spawn(kind){
    var x=cam.x+(rigs.length?(Math.random()-0.5)*0.6:-0.2);
    var sc=RS.sampleScenario(RS.rngMake((Math.random()*1e9)|0),{only:kind==='stand'?'stand':kind,pushMax:1.5});
    var o={pose:kind==='stand'?null:sc.pose,rootAng:kind==='stand'?0:sc.rootAng,h:kind==='drop'?Math.max(0.3,sc.h):0,clear:kind==='fallen'?0.01:0.02};
    RS.FLAGS.soft=opt('oSoft')?1:0;
    var rig=RS.buildRig(world,x,0,-(++group),o);
    if(kind==='drop'&&sc.vel)rig.parts.forEach(function(b){b.setLinearVelocity(Vec2(sc.vel.vx,sc.vel.vy));b.setAngularVelocity(sc.vel.w);});
    rigs.push(rig);liveBufs.push(RS.newBuf());
  }
  function hit(p){
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
  function applyDrag(){
    if(!drag)return;
    var b=drag.body;b.setAwake(true);
    var wp=b.getWorldPoint(drag.local),v=b.getLinearVelocityFromWorldPoint(wp);
    var ax=120*(drag.target.x-wp.x)-22*v.x,ay=120*(drag.target.y-wp.y)-22*v.y;
    var mag=Math.sqrt(ax*ax+ay*ay);if(mag>80){ax*=80/mag;ay*=80/mag;}
    var m=Math.max(drag.mass,b.getMass());
    b.applyForce(Vec2(ax*m,ay*m),wp,true);
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
  cv.addEventListener('pointerdown',function(e){
    e.preventDefault();
    try{cv.setPointerCapture(e.pointerId);}catch(_){}
    ptrs[e.pointerId]=pos(e);
    if(ids().length>=2){startPinch();return;}
    var p=ptrs[e.pointerId],h=pick(p.x,p.y);
    if(h){
      var wp=toWorld(h.sx,h.sy);
      drag={body:h.b,local:h.b.getLocalPoint(wp),target:toWorld(p.x,p.y),mass:islandMass(h.b)};
      mode='drag';
    }else{mode='pan';pan={x:p.x,y:p.y};}
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
    }else if(mode==='drag'&&drag){drag.target=toWorld(p.x,p.y);}
    else if(mode==='pan'&&pan){cam.x-=(p.x-pan.x)/cam.z;cam.y+=(p.y-pan.y)/cam.z;pan={x:p.x,y:p.y};}
  });
  function up(e){
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

  document.getElementById('bSpawn').onclick=function(){spawn('stand');};
  document.getElementById('bDrop').onclick=function(){spawn('drop');};
  document.getElementById('bFall').onclick=function(){spawn('fallen');};
  document.getElementById('bReset').onclick=init;
  document.getElementById('bIn').onclick=function(){setZoom(cam.z*1.4,W/2,H/2);};
  document.getElementById('bOut').onclick=function(){setZoom(cam.z/1.4,W/2,H/2);};
  function push(dir){
    var J=parseFloat(document.getElementById('pS').value)*dir;
    rigs.forEach(function(r){var c=r.parts[1];c.applyLinearImpulse(Vec2(J,0),c.getWorldCenter(),true);});
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
  function draw(){
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,W,H);
    var fg=getComputedStyle(document.body).color;
    ctx.strokeStyle=fg;ctx.fillStyle=fg;ctx.lineWidth=2;
    var gy=toScreen(Vec2(0,0)).y;
    var g0=toWorld(0,H),g1=toWorld(W,0),gx,gyy; // 1 m grid
    ctx.save();ctx.globalAlpha=0.15;ctx.lineWidth=1;ctx.beginPath();
    for(gx=Math.ceil(g0.x);gx<=g1.x;gx++){var sx=toScreen(Vec2(gx,0)).x;ctx.moveTo(sx,0);ctx.lineTo(sx,H);}
    for(gyy=Math.ceil(g0.y);gyy<=g1.y;gyy++){var sy=toScreen(Vec2(0,gyy)).y;ctx.moveTo(0,sy);ctx.lineTo(W,sy);}
    ctx.stroke();ctx.restore();
    ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(W,gy);ctx.stroke();
    if(opt('oBox')){ctx.globalAlpha=0.4;drawBodies(true);ctx.globalAlpha=1;drawBodies(false);}
    if(opt('oNN')&&RS.hasDet()){ // what the network thinks is happening, above each head
      ctx.font='600 12px system-ui,Arial,sans-serif';ctx.textAlign='center';ctx.fillStyle=fg;
      rigs.forEach(function(r){
        var p=r.dp,b=0;for(var j=1;j<p.length;j++)if(p[j]>p[b])b=j;var gu=r.gu&&r.gu.ph>=0,cw=r.cowering,dd=r.dead||(r.ko?'knocked out':'');
        var hp=toScreen(r.parts[0].getPosition());
        ctx.globalAlpha=gu||cw||dd?1:0.5+0.5*p[b];ctx.fillText(dd?(r.dead?'dead':dd):cw?'cowering':gu?'getting up':RS.DCLS[b]+' '+Math.round(p[b]*100)+'%',hp.x,hp.y-0.2*cam.z);ctx.globalAlpha=1;
      });
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
        r.parts.forEach(function(b){var c=b.getWorldCenter(),m=b.getMass();mx+=c.x*m;my+=c.y*m;M+=m;});
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
  function setBone(k,on){rigs.forEach(function(r){RS.breakBone(r,k,on);if(k===0&&!on)r.dead=false;});}
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
        var c=r.ctrls[k],a=toScreen(c.a.getWorldPoint(c.j.getLocalAnchorA()));
        ctx.globalAlpha=k>=3&&k<9?0.5:1;ctx.beginPath();ctx.arc(a.x,a.y,I.gone[k]?6:4,0,Math.PI*2);ctx.fill();
        if(I.gone[k]){var b=toScreen(c.b.getWorldPoint(c.j.getLocalAnchorB()));ctx.beginPath();ctx.arc(b.x,b.y,5,0,Math.PI*2);ctx.fill();}
      }
    });
    ctx.globalAlpha=1;
  }

  var last=performance.now(),acc=0,injN=0;
  function frame(t){
    var dt=Math.min((t-last)/1000,0.05);last=t;
    var paused=training&&pool.mode!=='workers'; // main-thread training: no time left for the live view
    if(!paused){
      if(opt('oSlow'))dt*=0.25;
      acc+=dt;
      var use=opt('oUse'),pd=opt('oPD'),reflex=opt('oReflex'),i;
      RS.FLAGS.air=RS.FLAGS.land=RS.FLAGS.fall=opt('oFall')?1:0;
      RS.FLAGS.step=opt('oStep')?1:0;RS.FLAGS.bal2=opt('oBal2')?1:0;RS.FLAGS.nn=opt('oNN')?1:0;RS.FLAGS.getup=opt('oNN')&&opt('oGetup')?1:0;RS.FLAGS.cower=opt('oCower')?1:0;RS.FLAGS.die=opt('oDie')?1:0;RS.FLAGS.protect=opt('oProtect')?1:0;RS.FLAGS.inj=opt('oInj')?1:0;RS.FLAGS.sever=opt('oSever')?1:0;
      while(acc>=DT){
        if((use||reflex)&&stepCount%SUB===0)for(i=0;i<rigs.length;i++)RS.act(rigs[i],use?ES.theta:ZERO_TH,liveBufs[i],liveRng,{delay:1,noise:0.005,reflex:reflex}); // reflexes run even when the policy is off (untrained = zero weights)
        if(pd)for(i=0;i<rigs.length;i++)RS.pdRig(rigs[i]);
        applyDrag();world.step(DT,RS.VEL_IT,RS.POS_IT);acc-=DT;stepCount++;
      }
    }
    draw();drawInjuries();if(++injN%10===0)injSync();requestAnimationFrame(frame);
  }
  init();updateInfo();requestAnimationFrame(frame);
})();
