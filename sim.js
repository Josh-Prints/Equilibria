var RS=(function(){
  'use strict';
  var Vec2=planck.Vec2,Box=planck.Box,FLAGS={air:0,land:0,fall:0,fallE:0.25,fallA:0.55,landT:0.3,landSL:0.7,landH:0.2,landK:-0.3,step:0,Ts:0.2,land_off:0.0,trig:0.12,clear:0.07,smooth:0,soft:0,softT:0.6,
    bal2:0,trig2:0.03,off2:0.0,max2:0.45,Ts2:0.25,clear2:0.09,lean2:1.0,trunk2:1.0,ank2:4,nn:0}; // step: legs go out when the capture point leaves the feet // fall/landing reflexes: experimental, off by default (they lowered the scores)
  // smooth: reward smooth, calm, human-like motion (jerk/acceleration penalties, stillness bonus)   soft: ease into the pose for softT s after spawning instead of snapping to it
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
    var soft=FLAGS.soft&&Math.abs(ra)<0.3&&(o.h||0)<0.05,tgt0=new Float64Array(NJ); // soft start (only when spawned upright on the feet): PD targets begin at the spawn pose
    if(soft)for(i=0;i<NJ;i++)tgt0[i]=pose[i];
    return {parts:bodies,feet:feet,ctrls:ctrls,stiff:new Float64Array(NJ).fill(1),tgt:tgt0,age:soft?0:1e9,prevO:new Float64Array(NO),prevO2:new Float64Array(NO),prevW:new Float64Array(NJ),
      lo:-0.10,hi:0.16,noC:0,fc:[false,false],e:0,airT:0,landT:0,fallT:0,rs:new Float64Array(NJ).fill(1),fcT:0,quiet:0,xi:0,st:{ph:0,t:0,leg:0,next:0,hold:[0,0,0],dir:1,xl:0},dh:[],dp:new Float64Array([1,0,0,0,0]),wph:0,st2:{ph:0,t:0,leg:0,next:0,calm:0,x0:0,xl:0,Ts:0.25,clr:0.09,dir:0},com:{x:0,y:0.95,vx:0,contact:false},hist:[],prev:new Float64Array(NJ)};
  }

  // PD: torque = kp*(target-angle) - kd*angularVelocity, clamped. Per-joint stiffness scales kp, kd, and the torque cap.
  function pdRig(r){
    var cs=r.ctrls,st=r.stiff;
    for(var i=0;i<cs.length;i++){
      var c=cs[i],ks=st[i];
      var t=c.kp*ks*(c.target-c.j.getJointAngle())-c.kd*Math.sqrt(ks)*c.j.getJointSpeed();
      var mx=c.max*ks;
      if(t>mx)t=mx;else if(t<-mx)t=-mx;
      c.t=t/(mx+1e-9);
      c.b.applyTorque(t,true);c.a.applyTorque(-t,true);
    }
  }

  // =====================================================================
  // POLICY: MLP  NI -> NH -> NH -> NO.
  //  in:  0-48 body state (see sense), 49-63 its own current PD targets (so it can move smoothly from where it is)
  //  out: 0-14 PD target offsets, 15-29 per-joint stiffness, 30/31 ankle/hip balance-reflex gain (0..2x), 32 target smoothing rate
  //  All extra outputs are neutral at 0, so a zero output layer still means 'standing pose + reflexes'.
  // =====================================================================
  var ASC=[0.4,0.5,0.4, 1.2,1.2,0.5,1.2,1.5,0.6, 1.2,1.2,0.5,1.2,1.5,0.6]; // max PD-target offset per joint (rad)
  var NI=64,NH=40,NO=33,O_STIFF=15,O_RFA=30,O_RFH=31,O_FILT=32;
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
  // old weight files (version 2: 49 inputs, 18 outputs, 3 stiffness groups) -> current layout, behaving exactly the same.
  // Returns a Float64Array of NP weights, or null if the size is not recognised.
  var OLD={NI:49,NO:18};OLD.NP=NH*OLD.NI+NH+NH*NH+NH+OLD.NO*NH+OLD.NO;
  function migrate(th){
    if(!th)return null;
    if(th.length===NP)return Float64Array.from(th);
    if(th.length!==OLD.NP)return null;
    var p=new Float64Array(NP),j,i,k,o1=NH*OLD.NI,o2=o1+NH,o3=o2+NH*NH+NH,ob=o3+OLD.NO*NH;
    for(j=0;j<NH;j++)for(i=0;i<OLD.NI;i++)p[j*NI+i]=th[j*OLD.NI+i];  // new inputs get zero weight
    for(j=0;j<NH;j++)p[B1+j]=th[o1+j];
    for(i=0;i<NH*NH+NH;i++)p[W2+i]=th[o2+i];
    function row(dst,src){for(i=0;i<NH;i++)p[W3+dst*NH+i]=th[o3+src*NH+i];p[B3+dst]=th[ob+src];}
    for(k=0;k<NJ;k++){row(k,k);row(O_STIFF+k,15+JT[k].g);} // each joint copies its old group's stiffness output
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
    for(k=0;k<NJ;k++)o[49+k]=rig.tgt[k]/ASC[k];
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

  // ---- balance v2 (FLAGS.bal2): SIMBICON-style stepping that catches real shoves ----
  // Standing legs hold the trunk upright in WORLD frame (hip torque), leaning a little into the fall like a person;
  // ankles keep the feet flat and push the capture point back over the feet. When the capture point leaves the feet
  // (FLAGS.trig2 past toe/heel) the rear leg swings to just beyond it, then the other leg if that wasn't enough.
  // Once settled, the trailing foot is brought back beside the front one. Reflexes keep fighting down to a 55 deg tilt.
  // st2.ph: 0 standing, 1 swinging, 2 landed (double support), 3 bringing the trailing foot in
  var SHIN=[IDX.shf,IDX.shn];
  function ankleX(rig,l){return rig.ctrls[LEGK[l][2]].j.getAnchorA().x;}
  function begin2(rig,l,xl,Ts,clr,dir){var s=rig.st2;s.ph=dir?1:3;s.leg=l;s.t=0;s.x0=ankleX(rig,l);s.xl=xl;s.Ts=Ts;s.clr=clr;s.dir=dir;}
  function bal2(rig,tt,out,ca,dt){
    var s=rig.st2,ps=rig.parts,hy=ps[0].getPosition().y,pa=wrap(ps[3].getAngle()),pw=ps[3].getAngularVelocity();
    var ok=Math.cos(ca)>0.55&&hy>1.1&&(rig.com.contact||s.ph===1);
    if(!ok){s.ph=0;s.calm=0;return false;}
    var e=rig.e,xi=rig.xi,k,l;
    s.t+=dt;
    // lean into the fall (hip strategy, not mid-swing), then hold the trunk upright in world frame with the stance hips
    var pdes=s.ph===1?0:clampA(-FLAGS.lean2*e,0.3),gain=FLAGS.trunk2;
    // which way the capture point has gone, and by how much
    var out2=xi>rig.hi+FLAGS.trig2?1:(xi<rig.lo-FLAGS.trig2?-1:0);
    if(s.ph===0||s.ph===2){
      if(s.ph===2&&s.t<0.12)out2=0; // let the landing settle for a moment
      if(out2){
        // swing the foot that is behind (relative to the fall); tie -> alternate
        var a0=ankleX(rig,0),a1=ankleX(rig,1),lg=Math.abs(a0-a1)<0.03?s.next:(out2>0?(a0<a1?0:1):(a0>a1?0:1));
        s.next=1-lg;
        var hipX=rig.ctrls[LEGK[lg][0]].j.getAnchorA().x,xl=xi+out2*FLAGS.off2;
        xl=Math.max(hipX-FLAGS.max2,Math.min(hipX+FLAGS.max2,xl));
        begin2(rig,lg,xl,FLAGS.Ts2,FLAGS.clear2,out2);
      }else if(s.ph===2){
        s.calm=(Math.abs(e)<0.06&&Math.abs(rig.com.vx)<0.15)?s.calm+dt:0;
        if(s.calm>0.5){ // settled: bring the trailing foot back beside the front one
          var b0=ankleX(rig,0),b1=ankleX(rig,1),cx=rig.com.x;
          if(Math.abs(b0-b1)>0.1){var tl=Math.abs(b0-cx)>Math.abs(b1-cx)?0:1;begin2(rig,tl,(tl?b0:b1)+0.0,0.45,0.05,0);}
          else{s.ph=0;}
          s.calm=0;
        }
      }
    }
    var sw=(s.ph===1||s.ph===3)?s.leg:-1;
    if(sw>=0){
      var x=Math.min(1,s.t/s.Ts),sm=x*x*(3-2*x);
      if(s.ph===1){ // keep tracking the capture point while in the air
        var hx=rig.ctrls[LEGK[sw][0]].j.getAnchorA().x,want=xi+s.dir*FLAGS.off2;
        want=Math.max(hx-FLAGS.max2,Math.min(hx+FLAGS.max2,want));
        s.xl+=(want-s.xl)*0.15;
      }
      var hip=rig.ctrls[LEGK[sw][0]].j.getAnchorA();
      legIK(hip.x,hip.y,s.x0+(s.xl-s.x0)*sm,0.10+s.clr*Math.sin(Math.PI*x),pa,IKO);
      for(k=0;k<3;k++)tt[LEGK[sw][k]]=IKO[k]+out[LEGK[sw][k]]*ASC[LEGK[sw][k]]*0.3;
      if((x>0.55&&rig.fc[sw])||s.t>s.Ts*1.8){
        if(s.ph===3){s.ph=0;}else{s.ph=2;}
        s.t=0;s.calm=0;
      }
    }
    // stance legs: trunk control at the hip, knee nearly straight, foot flat + ankle push toward the support
    var nst=(sw<0?2:1),tA=clampA(-FLAGS.ank2*e,0.5)*(1+out[O_RFA]);
    for(l=0;l<2;l++){
      if(l===sw)continue;
      var K=LEGK[l],hj=rig.ctrls[K[0]].j.getJointAngle();
      tt[K[0]]=hj+(gain*(pa-pdes)+0.02*pw)*(2/nst)*0.5*(1+out[O_RFH])+out[K[0]]*ASC[K[0]]*0.3;
      tt[K[1]]=-0.06+out[K[1]]*ASC[K[1]];
      tt[K[2]]=clampA(-wrap(ps[SHIN[l]].getAngle()),0.7)+tA+out[K[2]]*ASC[K[2]];
    }
    return true;
  }

  // =====================================================================
  // NEURAL EVENT DETECTOR (FLAGS.nn): a small network looks at what the body feels (the same sensors the policy gets,
  // now and 25/50 ms ago) and says what is happening: calm, stumbling, falling, being moved (grabbed/dragged/lifted),
  // or down. Trained offline by supervised learning (tools/det_train.js) on simulated episodes labelled with the truth;
  // the labels are never available at run time, only the network's guess is. Its guess drives the reactions:
  // arms wave when falling or being moved, arms out for balance when stumbling, arms brace just before hitting the ground.
  // =====================================================================
  var DCLS=['calm','stumble','falling','moved','down'],DNO=49,DLAGS=[3,6],DIN=DNO*(1+DLAGS.length);
  var DET=(typeof RS_DETW!=='undefined')?RS_DETW:null;
  function detPush(rig,obs){ // keep the last 7 sensor frames
    var h=rig.dh,f=new Float64Array(DNO);for(var i=0;i<DNO;i++)f[i]=obs[i];
    h.unshift(f);if(h.length>7)h.pop();
  }
  function detFeat(rig,x){ // current frame + how it changed over the last 25 and 50 ms
    var h=rig.dh,c=h[0],i,l;
    for(i=0;i<DNO;i++)x[i]=c[i];
    for(l=0;l<DLAGS.length;l++){var p=h[Math.min(DLAGS[l],h.length-1)];for(i=0;i<DNO;i++)x[DNO*(l+1)+i]=c[i]-p[i];}
    return x;
  }
  function detForward(W,x,out){ // normalise -> tanh -> tanh -> softmax
    var nh=W.b1.length,nh2=W.b2.length,no=W.b3.length,i,j,s,h1=new Float64Array(nh),h2=new Float64Array(nh2);
    for(j=0;j<nh;j++){s=W.b1[j];var r=j*DIN;for(i=0;i<DIN;i++)s+=W.W1[r+i]*(x[i]-W.mu[i])/W.sd[i];h1[j]=Math.tanh(s);}
    for(j=0;j<nh2;j++){s=W.b2[j];for(i=0;i<nh;i++)s+=W.W2[j*nh+i]*h1[i];h2[j]=Math.tanh(s);}
    var m=-1e9;for(j=0;j<no;j++){s=W.b3[j];for(i=0;i<nh2;i++)s+=W.W3[j*nh2+i]*h2[i];out[j]=s;if(s>m)m=s;}
    var z=0;for(j=0;j<no;j++){out[j]=Math.exp(out[j]-m);z+=out[j];}for(j=0;j<no;j++)out[j]/=z;
    return out;
  }
  var DX=new Float64Array(DIN),DO=new Float64Array(5);
  function detect(rig){
    if(!DET||rig.dh.length<2)return rig.dp;
    detForward(DET,detFeat(rig,DX),DO);
    for(var k=0;k<5;k++)rig.dp[k]+=(DO[k]-rig.dp[k])*0.35; // light smoothing so a single odd frame doesn't flicker the arms
    return rig.dp;
  }
  // reactions driven by the detector's guess (arms + neck only; legs stay with the balance/step controller)
  function nnReact(rig,tt,dt){
    var p=rig.dp,ps=rig.parts,hy=ps[0].getPosition().y,ca=wrap(ps[1].getAngle());
    var wave=Math.min(1,Math.max(0,(Math.max(p[2],p[3])-0.35)/0.4)),bal=Math.min(1,Math.max(0,(p[1]-0.3)/0.4));
    var brace=p[2]>0.5&&hy<1.25&&hy>0.5; // about to hit the ground: hands go out toward it
    rig.wph+=dt*2*Math.PI*(2.0+0.8*wave);
    var w=rig.wph,dir=(-ca+2*rig.e)>0?1:-1;
    var sF=0.25+0.35*bal+1.2*wave*Math.sin(w)+1.0*wave,sN=0.25+0.35*bal+1.2*wave*Math.sin(w+Math.PI*0.9)+1.0*wave;
    var eF=0.3+0.7*wave*(0.5+0.5*Math.sin(w+1.2)),eN=0.3+0.7*wave*(0.5+0.5*Math.sin(w+1.2+Math.PI*0.9));
    if(brace){sF=sN=dir>0?1.3:-0.9;eF=eN=0.4;tt[0]=-0.35;}
    var mix=Math.max(wave,bal,brace?1:0);
    tt[3]=tt[3]*(1-mix)+sF*mix;tt[9]=tt[9]*(1-mix)+sN*mix;tt[4]=tt[4]*(1-mix)+eF*mix;tt[10]=tt[10]*(1-mix)+eN*mix;
    tt[5]=tt[5]*(1-mix)+0.3*wave*Math.sin(w*1.3)*mix;tt[11]=tt[11]*(1-mix)+0.3*wave*Math.sin(w*1.3+1)*mix;
  }

  // run the policy: stiffness + filtered PD targets, with a fixed balance reflex underneath (only when upright on the feet)
  function act(rig,theta,buf,rng,opt){
    opt=opt||{};
    sense(rig,buf.obs,rng,opt.noise);
    forward(theta,buf.obs,buf.out,buf);
    detPush(rig,buf.obs);if(FLAGS.nn)detect(rig);
    var d=opt.delay||0,h=rig.hist;
    h.unshift(Array.prototype.slice.call(buf.out));if(h.length>4)h.pop();
    var out=h[Math.min(d,h.length-1)];
    var dt=1/POLICY_HZ,ca=wrap(rig.parts[1].getAngle()),air=!(rig.fc[0]||rig.fc[1]);
    var tA=0,tH=0,upright=Math.cos(ca)>0.8&&rig.parts[0].getPosition().y>1.35;
    var rf=opt.reflex!==false;
    if(rf&&upright&&rig.com.contact){
      var e=rig.e,ae=Math.abs(e)-0.02;
      tA=clampA(-4*e,0.5)*(1+out[O_RFA]);
      if(ae>0&&!FLAGS.bal2)tH=clampA((e>0?-1:1)*4*ae,0.7)*(1+out[O_RFH]);
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
      var hy0=rig.parts[0].getPosition().y,stepping=rig.st.ph>0||rig.st2.ph>0;
      var lean=FLAGS.fall&&hy0>1.0&&!stepping&&!(FLAGS.bal2&&Math.cos(ca)>0.55&&hy0>1.1)&&rig.quiet<=0&&(Math.abs(ca)>FLAGS.fallA||rig.xi>rig.hi+FLAGS.fallE||rig.xi<rig.lo-FLAGS.fallE); // only while going down, and not while a step is trying to catch it
      if(lean&&!air&&rig.landT<=0){
        fall=true;
        var dir=(-ca+2*rig.e)>0?1:-1;           // which way it is going down: +1 forward
        sL=0.4;sS=0.45;sA=0.3;neck=-0.35;      // limp and tuck the chin
        sh=dir>0?1.3:-0.9;el=0.5;               // arms reach out toward the ground
        addH=0.15;addK=-0.2;
      }
    }
    rig.fallT=fall?rig.fallT+dt:0;
    var gl=[sL,sS,sA],k;
    for(k=0;k<NJ;k++){ // per-joint stiffness from the network, capped by what the active reflexes allow for that joint's group
      var base=0.2+0.8*sig(2*out[O_STIFF+k]+2),want=Math.min(base,gl[JT[k].g]);
      rig.rs[k]+=(want-rig.rs[k])*(want<rig.rs[k]?0.5:0.12); // soften fast, stiffen back quickly
      rig.stiff[k]=rig.rs[k];
    }
    var tt=new Array(NJ);
    for(k=0;k<NJ;k++){
      var t=out[k]*ASC[k];
      if(k===8||k===14)t+=tA;else if(k===6||k===12)t+=tH+addH;else if(k===7||k===13)t+=addK;
      else if(k===0)t+=neck;else if(k===3||k===9)t+=sh;else if(k===4||k===10)t+=el;
      tt[k]=t;
    }
    if(rf&&FLAGS.bal2)bal2(rig,tt,out,ca,dt);
    if(rf&&FLAGS.nn)nnReact(rig,tt,dt);
    else if(rf&&FLAGS.step)stepReflex(rig,tt,ca,dt);
    // target smoothing: the network can slow it down (smoother) or speed it up; soft start eases in after spawning
    var al=clampA(FILT*(1+0.6*out[O_FILT]),1);
    if(rig.age<FLAGS.softT){var x=rig.age/FLAGS.softT;al*=0.04+0.96*x*x*(3-2*x);}
    rig.age+=dt;
    for(k=0;k<NJ;k++){
      rig.tgt[k]+=(tt[k]-rig.tgt[k])*al;
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
      // with bal2 the drift is measured from the feet, (while standing) so a recovery step that ends somewhere else isn't punished
      var pen=calm*(2.0*split+2.0*da/NJ+1.0*jv/NJ+0.5*Math.abs(rig.com.vx)+1.0*Math.max(0,Math.abs(rig.com.x-(FLAGS.bal2&&rig.com.contact&&hy>1.2?(rig.lo+rig.hi)/2:0))-0.05)+0.1*eff/NJ);
      var hum=0;
      if(FLAGS.smooth){
        // smooth & human-like: penalise jerky commands and joint accelerations (twitching), hardest just after an upright start;
        // once nothing is happening, pay for calm stillness, a level head and relaxed (not co-contracted) joints
        var jk=0,acc=0,rx=0;
        for(k=0;k<NO;k++){jk+=Math.abs(out[k]-2*rig.prevO[k]+rig.prevO2[k]);rig.prevO2[k]=rig.prevO[k];rig.prevO[k]=out[k];}
        for(k=0;k<NJ;k++){var wv=cs[k].j.getJointSpeed();acc+=Math.abs(wv-rig.prevW[k]);rig.prevW[k]=wv;rx+=rig.stiff[k];}
        if(i<2){jk=0;acc=0;} // no history yet
        var startW=(sc.type==='stand'||sc.type==='push')?1+3*Math.exp(-t/0.8):1;
        hum=-startW*(reactive?0.3:1)*(1.5*jk/NO+3.0*Math.min(acc/NJ,0.05)); // capped: hard impacts are physics, not twitching
        if(upright&&!reactive){
          var ha=wrap(head.getAngle());
          hum+=calm*(0.15*Math.exp(-jv/NJ/0.05)*Math.exp(-Math.abs(rig.com.vx)/0.05)+0.05*Math.exp(-10*ha*ha)+0.1*(1-rx/NJ));
        }
      }
      var rew=scale*(0.6*stand+prog-pen+hum);
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

  return {DCLS:DCLS,DIN:DIN,detFeat:detFeat,detForward:detForward,setDet:function(w){DET=w;},hasDet:function(){return !!DET;},FLAGS:FLAGS,buildRig:buildRig,pdRig:pdRig,act:act,sense:sense,newBuf:newBuf,forward:forward,initParams:initParams,rngMake:rngMake,gauss:gauss,
    esNoise:esNoise,esStep:esStep,migrate:migrate,runEpisode:runEpisode,evalCandidate:evalCandidate,sampleScenario:sampleScenario,fk:fk,
    NP:NP,NI:NI,NO:NO,NJ:NJ,DT:DT,SUB:SUB,POLICY_HZ:POLICY_HZ,VEL_IT:VEL_IT,POS_IT:POS_IT,JT:JT,ORDER:ORDER,CASES:CASES,wrap:wrap};
})();
