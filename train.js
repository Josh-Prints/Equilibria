// Ragdoll trainer for GitHub Actions (Node 18+, needs: npm install planck@1.0.0)
// Same simulation, reflexes and evolution strategy as the Ragdoll Trainer page, but uses every CPU core.
// Writes out/policy.json (latest), out/best.json (best held-out score) and out/log.txt.
// The .json files load in the trainer page with "Load file".
'use strict';
const planck=require('planck');
const {Worker,isMainThread,parentPort}=require('worker_threads');
const os=require('os'),fs=require('fs');

var RS=(function(){
  'use strict';
  var Vec2=planck.Vec2,Box=planck.Box,FLAGS={air:0,land:0,fall:0,fallE:0.25,fallA:0.55,landT:0.3,landSL:0.7,landH:0.2,landK:-0.3,step:0,Ts:0.2,land_off:0.0,trig:0.12,clear:0.07}; // step: legs go out when the capture point leaves the feet // fall/landing reflexes: experimental, off by default (they lowered the scores)
  var DT=1/240,LIMF=0.35,TMAX=1,INERTIA_X=16,POLICY_HZ=120,SUB=2,VEL_IT=6,POS_IT=2,W0=3.24,FILT=0.6;

  // =====================================================================
  // RIG (side-on, facing +x). Standing-pose coordinates in meters, feet at y=0.
  // Joint angle = relative angle (child - parent), 0 = standing pose, + = counter-clockwise = swings forward.
  // =====================================================================
  var T={},ORDER=[],JT=[];
  function part(n,p,a,c,hx,hy,d,bone,fl){T[n]={n:n,p:p,a:a,c:c,hx:hx,hy:hy,d:d,bone:bone,far:fl&&fl.far,foot:fl&&fl.foot,hand:fl&&fl.hand};ORDER.push(n);}
  function jnt(child,lo,hi,om,g){T[child].ji=JT.length;JT.push({child:child,lo:lo,hi:hi,om:om,g:g});}
  // g: stiffness group 0=legs 1=spine/neck 2=arms
  part('head','chest',[0,1.61],[0.01,1.735],0.09,0.115,4,[0,1.61,0,1.85]);
  part('chest','abd',[0,1.29],[0,1.45],0.11,0.16,3,[0,1.29,0,1.61]);
  part('abd','pelv',[0,1.09],[0,1.19],0.09,0.10,4,[0,1.09,0,1.29]);
  part('pelv',null,null,[0,1.00],0.10,0.09,6,[0,0.94,0,1.09]);
  [1,0].forEach(function(far){
    var s=far?'f':'n',fl={far:!!far};
    part('ua'+s,'chest',[0,1.55],[0,1.40],0.04,0.15,3,[0,1.55,0,1.25],fl);
    part('fa'+s,'ua'+s,[0,1.25],[0,1.11],0.035,0.14,3,[0,1.25,0,0.97],fl);
    part('hd'+s,'fa'+s,[0,0.97],[0.01,0.92],0.04,0.05,3,[0,0.97,0,0.87],{far:!!far,hand:true});
    part('th'+s,'pelv',[0,0.94],[0,0.73],0.06,0.21,3,[0,0.94,0,0.52],fl);
    part('sh'+s,'th'+s,[0,0.52],[0,0.31],0.045,0.21,3,[0,0.52,0,0.10],fl);
    part('ft'+s,'sh'+s,[0,0.10],[0.03,0.05],0.13,0.05,4,[0,0.10,0.15,0.05],{far:!!far,foot:true});
  });
  jnt('head',-0.5,0.5,15,1);jnt('chest',-0.5,0.9,80,1);jnt('abd',-0.4,0.6,80,1);
  [1,0].forEach(function(far){
    var s=far?'f':'n';
    jnt('ua'+s,-1.2,3.1,15,2);jnt('fa'+s,0,2.6,15,2);jnt('hd'+s,-0.9,0.9,15,2);
    jnt('th'+s,-0.7,2.4,80,0);jnt('sh'+s,-2.5,0,80,0);jnt('ft'+s,-0.7,0.7,80,0);
  });
  var NJ=JT.length; // 15
  var IDX={};ORDER.forEach(function(n,i){IDX[n]=i;});
  // body indices: 0 head, 1 chest, 2 abd, 3 pelvis, 4.. far (ua,fa,hd,th,sh,ft), 10.. near
  var FOOT_IDX=[IDX.ftf,IDX.ftn],HAND_IDX=[IDX.hdf,IDX.hdn];

  function rot(a,x,y){var c=Math.cos(a),s=Math.sin(a);return [x*c-y*s,x*s+y*c];}
  // forward kinematics: world transform of every part for a joint-angle vector and root (pelvis) pose
  function fk(pose,rx,ry,ra){
    var res={pelv:{x:rx,y:ry,a:ra,ax:0,ay:0}};
    function get(n){
      if(res[n])return res[n];
      var t=T[n],P=get(t.p),pc=T[t.p].c,th=pose[t.ji];
      var o=rot(P.a,t.a[0]-pc[0],t.a[1]-pc[1]),axw=P.x+o[0],ayw=P.y+o[1],ca=P.a+th;
      var c=rot(ca,t.c[0]-t.a[0],t.c[1]-t.a[1]);
      return (res[n]={x:axw+c[0],y:ayw+c[1],a:ca,ax:axw,ay:ayw});
    }
    ORDER.forEach(get);return res;
  }

  var GAIN=null; // per-joint {kp,kd,max,f}, computed once from the standing pose
  var ZERO=new Float64Array(NJ);
  function sideInertia(start,skip,an){
    var seen=[start],st=[start],I=0;
    while(st.length){
      var b=st.pop(),wc=b.getWorldCenter(),m=b.getMass();
      var Icom=b.getInertia()-m*b.getLocalCenter().lengthSquared();
      var dx=wc.x-an.x,dy=wc.y-an.y;I+=Icom+m*(dx*dx+dy*dy);
      for(var n=b.getJointList();n;n=n.next){
        if(n.joint===skip)continue;
        var o=n.other;if(o&&o.isDynamic()&&seen.indexOf(o)<0){seen.push(o);st.push(o);}
      }
    }
    return I;
  }

  // o: pose (15 angles), rootAng, h (extra height), clear, mass, fric, gain, partVar[]
  function buildRig(w,x,y,g,o,forGains){
    o=o||{};
    var pose=o.pose||ZERO,ra=o.rootAng||0,ms=o.mass||1,fr=o.fric||0.9,gs=o.gain||1,clear=(o.clear==null?0.02:o.clear);
    var tr=fk(pose,0,1.0,ra),minY=1e9,i;
    ORDER.forEach(function(n){var t=T[n],q=tr[n],lo=q.y-(Math.abs(t.hx*Math.sin(q.a))+Math.abs(t.hy*Math.cos(q.a)));if(lo<minY)minY=lo;});
    var lift=clear-minY+(o.h||0),bodies=[];
    ORDER.forEach(function(n,k){
      var t=T[n],q=tr[n];
      var b=w.createBody({type:'dynamic',position:Vec2(x+q.x,y+q.y+lift),angle:q.a,linearDamping:0.01,angularDamping:0.01});
      var pv=o.partVar?o.partVar[k]:1;
      b.createFixture(Box(t.hx,t.hy),{density:t.d*ms*pv,friction:fr,restitution:0,filterGroupIndex:g});
      var md={mass:0,center:Vec2(),I:0};b.getMassData(md);md.I*=INERTIA_X;b.setMassData(md); // heavier rotational inertia keeps PD stable
      b.setUserData({far:t.far,foot:t.foot,hand:t.hand,bone:[Vec2(t.bone[0]-t.c[0],t.bone[1]-t.c[1]),Vec2(t.bone[2]-t.c[0],t.bone[3]-t.c[1])]});
      bodies.push(b);
    });
    var joints=[],pend=[];
    JT.forEach(function(jt,k){
      var t=T[jt.child],q=tr[jt.child];
      var j=w.createJoint(planck.RevoluteJoint({lowerAngle:jt.lo,upperAngle:jt.hi,enableLimit:true,referenceAngle:0},bodies[IDX[t.p]],bodies[IDX[jt.child]],Vec2(x+q.ax,y+q.ay+lift)));
      joints.push(j);
      if(forGains)pend.push({j:j,a:bodies[IDX[t.p]],b:bodies[IDX[jt.child]],an:Vec2(x+q.ax,y+q.ay+lift),om:jt.om,f:1});
    });
    if(forGains){
      pend.forEach(function(p){
        var IA=sideInertia(p.a,p.j,p.an),IB=sideInertia(p.b,p.j,p.an),Ir=IA*IB/(IA+IB);
        p.kp=p.om*p.om*Ir;p.kd=2*p.om*Ir;
      });
      var wl=LIMF/DT;
      bodies.slice().reverse().forEach(function(b){
        var js=pend.filter(function(p){return p.a===b||p.b===b;});if(!js.length)return;
        var S=0,Sd=0;js.forEach(function(p){S+=p.kp*p.f;Sd+=p.kd*p.f;});
        var I=b.getInertia()-b.getMass()*b.getLocalCenter().lengthSquared();
        var f=Math.min(1,wl*wl*I/S,wl*I/Sd);
        js.forEach(function(p){p.f=Math.min(p.f,f);});
      });
      GAIN=pend.map(function(p){var kp=p.kp*p.f;return {kp:kp,kd:p.kd*p.f,max:kp*TMAX};});
    }
    if(!GAIN){var sw=new planck.World({gravity:Vec2(0,-10)});buildRig(sw,0,0,-1,null,true);}
    var ctrls=JT.map(function(jt,k){
      return {j:joints[k],a:bodies[IDX[T[jt.child].p]],b:bodies[IDX[jt.child]],g:jt.g,kp:GAIN[k].kp*gs,kd:GAIN[k].kd*gs,max:GAIN[k].max*gs,target:0,t:0};
    });
    var feet=[bodies[IDX.ftf],bodies[IDX.ftn]];
    return {parts:bodies,feet:feet,ctrls:ctrls,stiff:[1,1,1],tgt:new Float64Array(NJ),
      lo:-0.10,hi:0.16,noC:0,fc:[false,false],e:0,airT:0,landT:0,fallT:0,rs:[1,1,1],fcT:0,quiet:0,xi:0,st:{ph:0,t:0,leg:0,next:0,hold:[0,0,0],dir:1,xl:0},com:{x:0,y:0.95,vx:0,contact:false},hist:[],prev:new Float64Array(NJ)};
  }

  // PD: torque = kp*(target-angle) - kd*angularVelocity, clamped. Stiffness groups scale kp, kd, and the torque cap.
  function pdRig(r){
    var cs=r.ctrls,st=r.stiff;
    for(var i=0;i<cs.length;i++){
      var c=cs[i],ks=st[c.g];
      var t=c.kp*ks*(c.target-c.j.getJointAngle())-c.kd*Math.sqrt(ks)*c.j.getJointSpeed();
      var mx=c.max*ks;
      if(t>mx)t=mx;else if(t<-mx)t=-mx;
      c.t=t/(mx+1e-9);
      c.b.applyTorque(t,true);c.a.applyTorque(-t,true);
    }
  }

  // =====================================================================
  // POLICY: MLP  NI -> NH -> NH -> NO.   NO = 15 PD target offsets + 3 stiffness groups
  // =====================================================================
  var ASC=[0.4,0.5,0.4, 1.2,1.2,0.5,1.2,1.5,0.6, 1.2,1.2,0.5,1.2,1.5,0.6]; // max PD-target offset per joint (rad)
  var NI=49,NH=40,NO=18;
  var B1=NH*NI,W2=B1+NH,B2=W2+NH*NH,W3=B2+NH,B3=W3+NO*NH,NP=B3+NO;
  function newBuf(){return {h1:new Float64Array(NH),h2:new Float64Array(NH),obs:new Float64Array(NI),out:new Float64Array(NO)};}
  function forward(p,x,out,b){
    var i,j,s,h1=b.h1,h2=b.h2;
    for(j=0;j<NH;j++){s=p[B1+j];for(i=0;i<NI;i++)s+=p[j*NI+i]*x[i];h1[j]=Math.tanh(s);}
    for(j=0;j<NH;j++){s=p[B2+j];for(i=0;i<NH;i++)s+=p[W2+j*NH+i]*h1[i];h2[j]=Math.tanh(s);}
    for(j=0;j<NO;j++){s=p[B3+j];for(i=0;i<NH;i++)s+=p[W3+j*NH+i]*h2[i];out[j]=Math.tanh(s);}
  }
  function rngMake(seed){var s=seed>>>0;return function(){s=(s+0x6D2B79F5)|0;var t=Math.imul(s^(s>>>15),1|s);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296;};}
  function gauss(r){var u=1-r(),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
  function initParams(seed){
    var r=rngMake(seed||1),p=new Float64Array(NP),i;
    for(i=0;i<B1;i++)p[i]=gauss(r)*Math.sqrt(1/NI);
    for(i=W2;i<B2;i++)p[i]=gauss(r)*Math.sqrt(1/NH);
    for(i=W3;i<B3;i++)p[i]=0; // zero outputs: training starts exactly at 'standing pose + reflex' and improves from there
    return p;
  }
  function wrap(a){return a-6.283185307179586*Math.round(a/6.283185307179586);}
  function clampA(v,a){return v>a?a:(v<-a?-a:v);}
  function sig(x){return 1/(1+Math.exp(-x));}

  // lowest world y of a body (box corners)
  function lowY(b){
    var vs=b.getFixtureList().getShape().m_vertices,m=1e9;
    for(var i=0;i<vs.length;i++){var y=b.getWorldPoint(vs[i]).y;if(y<m)m=y;}
    return m;
  }

  // what the policy sees
  function sense(rig,o,rng,noise){
    var cs=rig.ctrls,k,n=0,ps=rig.parts,nz=noise||0;
    for(k=0;k<NJ;k++){o[n++]=cs[k].j.getJointAngle()+(nz?gauss(rng)*nz*0.5:0);o[n++]=cs[k].j.getJointSpeed()*0.1+(nz?gauss(rng)*nz*2:0);}
    var pel=ps[3],ch=ps[1],hd=ps[0],a;
    a=pel.getAngle();o[30]=Math.sin(a);o[31]=Math.cos(a);
    a=ch.getAngle();o[32]=Math.sin(a);o[33]=Math.cos(a);
    a=hd.getAngle();o[34]=Math.sin(a);o[35]=Math.cos(a);
    o[36]=pel.getAngularVelocity()*0.1;o[37]=ch.getAngularVelocity()*0.1;
    var mx=0,my=0,vx=0,vy=0,M=0;
    for(k=0;k<ps.length;k++){
      var b=ps[k],c=b.getWorldCenter(),m=b.getMass(),v=b.getLinearVelocityFromWorldPoint(c);
      mx+=c.x*m;my+=c.y*m;vx+=v.x*m;vy+=v.y*m;M+=m;
    }
    mx/=M;my/=M;vx/=M;vy/=M;
    // foot contact with hysteresis, and a smoothed support span (no flicker when a heel lifts for a moment)
    var any=false,lo=1e9,hi=-1e9,f;
    for(f=0;f<2;f++){
      var fb=rig.feet[f],ly=lowY(fb);
      if(rig.fc[f]){if(ly>0.045)rig.fc[f]=false;}else if(ly<0.02)rig.fc[f]=true;
      if(rig.fc[f]){
        any=true;var vs=fb.getFixtureList().getShape().m_vertices;
        for(var i=0;i<vs.length;i++){var w=fb.getWorldPoint(vs[i]);if(w.y<0.05){if(w.x<lo)lo=w.x;if(w.x>hi)hi=w.x;}}
      }
    }
    if(any&&hi>lo){rig.lo+=(lo-rig.lo)*0.5;rig.hi+=(hi-rig.hi)*0.5;rig.noC=0;}
    else rig.noC++;
    var contact=rig.noC<30; // keep the last support for ~0.12 s of lost contact
    var c0=(rig.lo+rig.hi)/2;
    o[38]=(mx-c0)/0.2;o[39]=vx/0.5;o[40]=(mx+vx/W0-c0)/0.2;o[41]=(my-0.95)*2;
    o[42]=contact?1:0;
    o[43]=rig.fc[0]?1:0;o[44]=rig.fc[1]?1:0;
    o[45]=lowY(ps[HAND_IDX[0]])<0.04?1:0;o[46]=lowY(ps[HAND_IDX[1]])<0.04?1:0;
    o[47]=lowY(pel)<0.06?1:0;o[48]=lowY(hd)<0.06?1:0;
    rig.com={x:mx,y:my,vx:vx,vy:vy,contact:contact};
    rig.xi=mx+vx/W0;
    rig.e=rig.xi-(rig.lo+0.45*(rig.hi-rig.lo));
  }

  // ---- stepping reflex: when the capture point leaves the feet, swing one leg out to catch it, then bring it back ----
  // phases: 0 standing, 1 swinging, 2 planted (hold), 3 bringing the leg back
  // The swing foot follows a smooth path to the landing spot and a 2-link leg IK turns that into hip/knee/ankle targets.
  var LEGK=[[6,7,8],[12,13,14]],L1=0.42,L2=0.42;
  function legIK(hx,hy,ax,ay,ap,out){
    var dx=ax-hx,dy=ay-hy,d=Math.sqrt(dx*dx+dy*dy),dm=(L1+L2)*0.995;
    if(d>dm){dx*=dm/d;dy*=dm/d;d=dm;}
    if(d<0.2){d=0.2;}
    var cg=clampA((L1*L1+L2*L2-d*d)/(2*L1*L2),1),g=Math.acos(cg);
    var beta=Math.asin(clampA(L2*Math.sin(g)/d,1)),psi=Math.atan2(dx,-dy);
    var hr=psi+beta-ap,kn=-(Math.PI-g);
    out[0]=clampA(hr,2.3);out[1]=Math.max(-2.4,Math.min(0,kn));out[2]=clampA(-(ap+hr+kn),0.65);
  }
  var IKO=[0,0,0];
  function beginStep(rig,c0){
    var st=rig.st;
    st.ph=1;st.t=0;st.leg=st.next;st.next=1-st.next;st.dir=rig.xi>c0?1:-1;
    st.xl=rig.xi-st.dir*FLAGS.land_off;
    st.x0=rig.ctrls[LEGK[st.leg][2]].j.getAnchorA().x;
  }
  function outside(rig,m){return rig.xi>rig.hi+m||rig.xi<rig.lo-m;}
  function stepReflex(rig,tt,ca,dt){
    var st=rig.st,hy=rig.parts[0].getPosition().y,upright=Math.cos(ca)>0.7&&hy>1.15,c0=(rig.lo+rig.hi)/2;
    rig.fcT=(rig.fc[0]&&rig.fc[1])?rig.fcT+dt:0;
    if(st.ph===0&&upright&&rig.fcT>0.5&&rig.quiet<=0&&outside(rig,FLAGS.trig))beginStep(rig,c0); // only after standing settled on both feet
    if(st.ph===0)return;
    var K=LEGK[st.leg],hip=rig.ctrls[K[0]].j.getAnchorA(),ap=wrap(rig.parts[3].getAngle());
    st.t+=dt;
    if(st.ph===1){
      var x=Math.min(1,st.t/FLAGS.Ts),sm=x*x*(3-2*x);
      st.xl+=(rig.xi-st.dir*FLAGS.land_off-st.xl)*0.1; // keep tracking the capture point while the leg is in the air
      var xa=st.x0+(st.xl-st.x0)*sm,ya=0.10+FLAGS.clear*Math.sin(Math.PI*x);
      legIK(hip.x,hip.y,xa,ya,ap,IKO);
      tt[K[0]]=IKO[0];tt[K[1]]=IKO[1];tt[K[2]]=IKO[2];
      st.hold[0]=IKO[0];st.hold[1]=IKO[1];st.hold[2]=IKO[2];
      if((st.t>FLAGS.Ts*0.7&&rig.fc[st.leg])||st.t>FLAGS.Ts*2){st.ph=2;st.t=0;}
    }else if(st.ph===2){ // planted: keep the leg as it landed while balance catches up
      tt[K[0]]=st.hold[0];tt[K[1]]=st.hold[1];tt[K[2]]=st.hold[2];
      if(st.t>0.1&&upright&&outside(rig,FLAGS.trig))beginStep(rig,c0);     // still falling: step again with the other leg
      else if(st.t>0.4&&Math.abs(rig.e)<0.1){st.ph=3;st.t=0;}
    }else{ // bring the leg back under the body
      var b=Math.min(1,st.t/0.9);
      tt[K[0]]=st.hold[0]*(1-b);tt[K[1]]=st.hold[1]*(1-b);tt[K[2]]=st.hold[2]*(1-b);
      if(b>=1){st.ph=0;st.t=0;}
      else if(upright&&outside(rig,FLAGS.trig)){st.ph=0;st.t=0;}
    }
    if(!upright&&hy<0.9)st.ph=0; // fell anyway
  }

  // run the policy: stiffness + filtered PD targets, with a fixed balance reflex underneath (only when upright on the feet)
  function act(rig,theta,buf,rng,opt){
    opt=opt||{};
    sense(rig,buf.obs,rng,opt.noise);
    forward(theta,buf.obs,buf.out,buf);
    var d=opt.delay||0,h=rig.hist;
    h.unshift(Array.prototype.slice.call(buf.out));if(h.length>4)h.pop();
    var out=h[Math.min(d,h.length-1)];
    var dt=1/POLICY_HZ,ca=wrap(rig.parts[1].getAngle()),air=!(rig.fc[0]||rig.fc[1]);
    var tA=0,tH=0,upright=Math.cos(ca)>0.8&&rig.parts[0].getPosition().y>1.35;
    var rf=opt.reflex!==false;
    if(rf&&upright&&rig.com.contact){
      var e=rig.e,ae=Math.abs(e)-0.02;
      tA=clampA(-4*e,0.5);
      if(ae>0)tH=clampA((e>0?-1:1)*4*ae,0.7);
    }
    // ---- reflexes for things that go wrong: landing, losing balance, falling ----
    var sL=1,sS=1,sA=1,addK=0,addH=0,neck=0,sh=0,el=0,fall=false;
    if(rf){
      if(air)rig.airT+=dt;
      else{if(rig.airT>0.15&&rig.com.vy<-0.3){rig.landT=FLAGS.landT;rig.quiet=0.8;}rig.airT=0;} // landed after a real fall (not the 2 cm settle at the start): absorb it
      if(FLAGS.air&&air&&rig.airT>0.15){sL=0.7;addH=0.25;addK=-0.3;sA=0.5;sh=0.6;}   // in the air: legs ready to land, arms out a bit
      if(!FLAGS.land)rig.landT=0;
      if(rig.quiet>0)rig.quiet-=dt;
      if(rig.landT>0){rig.landT-=dt;sL=FLAGS.landSL;addH=FLAGS.landH;addK=FLAGS.landK;sS=0.6;}  // landing: go soft and crouch
      var hy0=rig.parts[0].getPosition().y,stepping=rig.st.ph>0;
      var lean=FLAGS.fall&&hy0>1.0&&!stepping&&rig.quiet<=0&&(Math.abs(ca)>FLAGS.fallA||rig.xi>rig.hi+FLAGS.fallE||rig.xi<rig.lo-FLAGS.fallE); // only while going down, and not while a step is trying to catch it
      if(lean&&!air&&rig.landT<=0){
        fall=true;
        var dir=(-ca+2*rig.e)>0?1:-1;           // which way it is going down: +1 forward
        sL=0.4;sS=0.45;sA=0.3;neck=-0.35;      // limp and tuck the chin
        sh=dir>0?1.3:-0.9;el=0.5;               // arms reach out toward the ground
        addH=0.15;addK=-0.2;
      }
    }
    rig.fallT=fall?rig.fallT+dt:0;
    var gl=[sL,sS,sA],g;
    for(g=0;g<3;g++){
      var base=0.2+0.8*sig(2*out[15+g]+2),want=Math.min(base,gl[g]);
      rig.rs[g]+=(want-rig.rs[g])*(want<rig.rs[g]?0.5:0.12); // soften fast, stiffen back quickly
      rig.stiff[g]=rig.rs[g];
    }
    var tt=new Array(NJ);
    for(var k=0;k<NJ;k++){
      var t=out[k]*ASC[k];
      if(k===8||k===14)t+=tA;else if(k===6||k===12)t+=tH+addH;else if(k===7||k===13)t+=addK;
      else if(k===0)t+=neck;else if(k===3||k===9)t+=sh;else if(k===4||k===10)t+=el;
      tt[k]=t;
    }
    if(rf&&FLAGS.step)stepReflex(rig,tt,ca,dt);
    for(k=0;k<NJ;k++){
      rig.tgt[k]+=(tt[k]-rig.tgt[k])*FILT;
      rig.ctrls[k].target=rig.tgt[k];
    }
    return out;
  }

  // =====================================================================
  // SCENARIOS + EPISODE
  //  stand: tiny random perturbation   push: 1-3 pushes anywhere   drop: thrown/dropped with a random pose
  //  fallen: starts lying (back, front or crumpled) and has to get up
  // =====================================================================
  var CASES=['stand','push','drop','fallen'];
  function clampPose(p,s){for(var k=0;k<NJ;k++){var j=JT[k];p[k]=Math.max(j.lo*0.9,Math.min(j.hi*0.9,p[k]));}return p;}
  function sampleScenario(r,cfg){
    var u=r(),type=cfg.only||(u<0.2?'stand':u<0.55?'push':u<0.8?'drop':'fallen');
    var sc={type:type,pose:new Float64Array(NJ),rootAng:0,h:0,T:8,pushes:[],vel:null,
      mass:1+0.15*(2*r()-1),fric:0.7+0.5*r(),gain:0.9+0.2*r(),delay:Math.floor(r()*4),noise:0.01*r(),partVar:ORDER.map(function(){return 1+0.1*(2*r()-1);}),clear:0.02};
    var k;
    if(type==='stand'){
      for(k=0;k<NJ;k++)sc.pose[k]=0.06*(2*r()-1);
      clampPose(sc.pose);sc.rootAng=0.04*(2*r()-1);sc.T=6;
    }else if(type==='push'){
      sc.T=9;var n=1+Math.floor(r()*3),t=0.8;
      for(k=0;k<n;k++){
        t+=0.2+r()*1.6;
        var tgt=[1,1,1,3,0,2,7,13][Math.floor(r()*8)];
        sc.pushes.push({t:t,body:tgt,jx:(r()<0.5?-1:1)*(0.3+r()*(cfg.pushMax-0.3))*(tgt===1||tgt===3?1:0.5),jy:0.3*(2*r()-1)});
      }
      sc.T=Math.max(8,t+3.5);
    }else if(type==='drop'){
      for(k=0;k<NJ;k++)sc.pose[k]=0.5*(2*r()-1)*(JT[k].g===2?1.2:0.8);
      clampPose(sc.pose);sc.rootAng=0.5*(2*r()-1);sc.h=0.05+r()*0.8;
      sc.vel={vx:(2*r()-1)*1.0,vy:(2*r()-1)*0.5,w:(2*r()-1)*1.5};sc.T=9;
    }else{ // fallen
      var side=r()<0.5?1:-1;
      sc.rootAng=side*(1.1+0.5*r());
      var crumple=r()<0.5;
      for(k=0;k<NJ;k++)sc.pose[k]=crumple?(JT[k].lo*0.6+(JT[k].hi-JT[k].lo)*0.6*r()):0.15*(2*r()-1);
      clampPose(sc.pose);sc.clear=0.01;sc.T=11;
    }
    return sc;
  }

  // returns the reward gathered over the episode (about 60 per simulated second at best)
  function runEpisode(theta,seed,cfg){
    cfg=cfg||{};cfg.pushMax=cfg.pushMax||1.5;
    if(cfg.flags)for(var fk_ in cfg.flags)FLAGS[fk_]=cfg.flags[fk_]; // which reflexes are active for this episode
    var r=rngMake(seed),sc=sampleScenario(r,cfg),rr=rngMake(seed^0x9e3779b9);
    var w=new planck.World({gravity:Vec2(0,-10)});
    var gd=w.createBody();gd.createFixture(Box(1000,1,Vec2(0,-1),0),{friction:sc.fric});
    var rig=buildRig(w,0,0,-1,{pose:sc.pose,rootAng:sc.rootAng,h:sc.h,mass:sc.mass,fric:sc.fric,gain:sc.gain,partVar:sc.partVar,clear:sc.clear});
    if(sc.vel)rig.parts.forEach(function(b){b.setLinearVelocity(Vec2(sc.vel.vx,sc.vel.vy));b.setAngularVelocity(sc.vel.w);});
    var buf=newBuf(),hz=POLICY_HZ,steps=Math.round(sc.T*hz),scale=60/hz,total=0,cs=rig.ctrls,head=rig.parts[0],chest=rig.parts[1];
    var pi=0,lastDist=sc.type==='fallen'||sc.type==='drop'?0:-9,landed=false,quietT=0,lieT=0,rateAvg=0,opt={delay:sc.delay,noise:sc.noise,reflex:cfg.noReflex?false:true};
    var done=false,i,k;
    for(i=0;i<steps&&!done;i++){
      var t=i/hz;
      while(pi<sc.pushes.length&&t>=sc.pushes[pi].t){
        var p=sc.pushes[pi++],b=rig.parts[p.body];b.applyLinearImpulse(Vec2(p.jx,p.jy),b.getWorldCenter(),true);lastDist=t;
      }
      var out=act(rig,theta,buf,rr,opt);
      for(var s=0;s<SUB;s++){pdRig(rig);w.step(DT,VEL_IT,POS_IT);}
      var hp=head.getPosition(),hy=hp.y,ca=wrap(chest.getAngle());
      if(!(hy===hy)||Math.abs(hp.x)>50||hy>20)break; // blew up
      if(!landed&&sc.type==='drop'&&(rig.fc[0]||rig.fc[1])){landed=true;lastDist=t;}
      var upright=Math.cos(ca)>0.85&&hy>1.45;
      if(!upright&&sc.type!=='stand')lastDist=Math.max(lastDist,t-0.3);
      var e2=0,le=0,da=0,jv=0,eff=0,q;
      for(k=0;k<NJ;k++){
        var a=cs[k].j.getJointAngle();e2+=a*a;
        q=out[k];da+=Math.abs(q-rig.prev[k]);rig.prev[k]=q;
        jv+=Math.abs(cs[k].j.getJointSpeed());eff+=Math.abs(cs[k].t);
      }
      for(k=6;k<9;k++){var la=cs[k].j.getJointAngle(),lb=cs[k+6].j.getJointAngle();le+=la*la+lb*lb;}
      var split=Math.abs(cs[6].j.getJointAngle()-cs[12].j.getJointAngle())+Math.abs(cs[7].j.getJointAngle()-cs[13].j.getJointAngle())+Math.abs(cs[8].j.getJointAngle()-cs[14].j.getJointAngle());
      var up=Math.exp(-8*ca*ca),pose=Math.exp(-6*e2/15),legs=Math.exp(-25*le/6),hh=Math.exp(-30*(hy-1.735)*(hy-1.735));
      var hn=Math.max(0,Math.min(1,hy/1.735));
      // while a reflex is doing its job (step, landing, bracing) don't punish the posture it needs
      var reactive=rig.st.ph>0||rig.fallT>0||rig.landT>0||rig.airT>0.15;
      if(reactive){pose=1;legs=1;split=0;da=0;jv*=0.25;}
      var stand=0.3*up+0.25*pose+0.25*legs+0.2*hh,prog=0.25*hn*hn+0.15*Math.max(0,Math.cos(ca));
      var calm=Math.min(1,Math.max(0.15,(t-lastDist)/1.2));
      var pen=calm*(2.0*split+2.0*da/NJ+1.0*jv/NJ+0.5*Math.abs(rig.com.vx)+1.0*Math.max(0,Math.abs(rig.com.x)-0.05)+0.1*eff/NJ);
      var rew=scale*(0.6*stand+prog-pen);
      total+=rew;rateAvg=rateAvg*0.98+rew*0.02;
      // early stop: standing still and no disturbance left -> assume it stays that way; lying motionless -> give up
      var lastPushT=sc.pushes.length?sc.pushes[sc.pushes.length-1].t:0;
      if(upright&&jv/NJ<0.05&&Math.abs(rig.com.vx)<0.03&&t>lastPushT+1.5&&(sc.type!=='drop'||landed)){quietT+=1/hz;}else quietT=0;
      if(quietT>1.2){total+=(steps-i-1)*rateAvg/(1-0.98)*0.02;done=true;}
      if(hy<0.6&&jv/NJ<0.05&&Math.abs(rig.com.vx)<0.03&&sc.type!=='fallen')lieT+=1/hz;else if(hy>=0.6)lieT=0;
      if(lieT>2.5){total+=(steps-i-1)*rateAvg/(1-0.98)*0.02;done=true;}
    }
    return total;
  }
  // mean over K scenarios (the same K for every candidate in a generation: common random numbers)
  function evalCandidate(theta,seed,K,cfg){
    var s=0;for(var k=0;k<K;k++)s+=runEpisode(theta,seed*131+k*7+1,cfg);
    return s/K;
  }

  // =====================================================================
  // EVOLUTION STRATEGIES helpers (antithetic sampling, centred ranks, Adam)
  // =====================================================================
  function esNoise(P,n,seed){var r=rngMake(seed),out=[];for(var i=0;i<P;i++){var e=new Float64Array(n);for(var k=0;k<n;k++)e[k]=gauss(r);out.push(e);}return out;}
  function esStep(S,eps,Fp,Fm,sigma,lr){
    var P=eps.length,n=S.theta.length,all=Fp.concat(Fm),i,k;
    var idx=all.map(function(_,j){return j;}).sort(function(a,b){return all[a]-all[b];});
    var rank=new Array(all.length);idx.forEach(function(j,q){rank[j]=q/(all.length-1)-0.5;});
    var g=new Float64Array(n);
    for(i=0;i<P;i++){var d=rank[i]-rank[P+i],e=eps[i];for(k=0;k<n;k++)g[k]+=d*e[k];}
    S.t++;var b1=0.9,b2=0.999;
    for(k=0;k<n;k++){
      g[k]/=(2*P*sigma);
      S.m[k]=b1*S.m[k]+(1-b1)*g[k];S.v[k]=b2*S.v[k]+(1-b2)*g[k]*g[k];
      var mh=S.m[k]/(1-Math.pow(b1,S.t)),vh=S.v[k]/(1-Math.pow(b2,S.t));
      S.theta[k]+=lr*mh/(Math.sqrt(vh)+1e-8);
    }
    var sum=0,best=-1e9;all.forEach(function(f){sum+=f;if(f>best)best=f;});
    return {mean:sum/all.length,best:best};
  }

  return {FLAGS:FLAGS,buildRig:buildRig,pdRig:pdRig,act:act,sense:sense,newBuf:newBuf,forward:forward,initParams:initParams,rngMake:rngMake,gauss:gauss,
    esNoise:esNoise,esStep:esStep,runEpisode:runEpisode,evalCandidate:evalCandidate,sampleScenario:sampleScenario,fk:fk,
    NP:NP,NI:NI,NO:NO,NJ:NJ,DT:DT,SUB:SUB,POLICY_HZ:POLICY_HZ,VEL_IT:VEL_IT,POS_IT:POS_IT,JT:JT,ORDER:ORDER,CASES:CASES,wrap:wrap};
})();


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
  const FLAGS={step:1,air:1,land:1,fall:1};         // all reflexes on, like the page default
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
    if(d.theta&&d.theta.length===NP){S.theta=Float64Array.from(d.theta);gen0=d.gen||0;log('resumed from resume.json at generation '+gen0);}
    else log('resume.json has the wrong size, starting fresh');
  }
  if(num('MINUTES',330)>340)log('MINUTES capped at 340: GitHub stops a job at 6 hours, so longer runs would lose their results');
  log('cpus='+os.cpus().length+' workers='+NW+' pairs='+P+' scenarios='+K+' sigma='+SIGMA+' lr='+LR+' mode='+MODE+' minutes='+MINUTES);

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
    fs.writeFileSync(outDir+'/'+file,JSON.stringify({format:'ragdoll-policy',version:2,np:NP,gen:gen,theta:Array.prototype.slice.call(S.theta),hist:[]}));
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
