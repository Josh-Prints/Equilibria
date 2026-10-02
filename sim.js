var RS=(function(){
  'use strict';
  var Vec2=planck.Vec2,Box=planck.Box,FLAGS={air:0,land:0,fall:0,fallE:0.25,fallA:0.55,landT:0.3,landSL:0.7,landH:0.2,landK:-0.3,step:0,Ts:0.2,land_off:0.0,trig:0.12,clear:0.07,smooth:0,soft:0,softT:0.6,
    bal2:0,trig2:0.03,off2:0.0,max2:0.45,Ts2:0.25,clear2:0.09,lean2:1.0,trunk2:1.0,ank2:4,nn:0,getup:0,guArm:1,cower:0,cowerImp:3.5,die:0,dieImp:12,protect:0,inj:0,sever:0,koImp:5.5,dieHead:8.5,dieTorso:11.5,dieLimb:21,crush:0,crushImp:15,crushLimb:1.25,ripImp:1.6,shatter:1,strands:1,groggyT:5,bleed:0}; // step: legs go out when the capture point leaves the feet // fall/landing reflexes: experimental, off by default (they lowered the scores)
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
    var tr=fk(pose,0,1.0,ra),minY=1e9,i,S=o.scale||1,WD=o.wid||1,M=o.dir===-1?-1:1; // scale: overall size, wid: build (widths), dir -1: faces -x
    if(S!==1)ORDER.forEach(function(n){var q=tr[n];q.x*=S;q.y*=S;q.ax*=S;q.ay*=S;});
    function HX(t){return t.hx*S*WD;}function HY(t){return t.hy*S;}
    ORDER.forEach(function(n){var t=T[n],q=tr[n],lo=q.y-(Math.abs(HX(t)*Math.sin(q.a))+Math.abs(HY(t)*Math.cos(q.a)));if(lo<minY)minY=lo;});
    var lift=clear-minY+(o.h||0),bodies=[];
    ORDER.forEach(function(n,k){
      var t=T[n],q=tr[n];
      var b=w.createBody({type:'dynamic',position:Vec2(x+M*q.x,y+q.y+lift),angle:M*q.a,linearDamping:0.01,angularDamping:0.01});
      var pv=o.partVar?o.partVar[k]:1;
      b.createFixture(Box(HX(t),HY(t)),{density:t.d*ms*pv,friction:fr,restitution:0,filterGroupIndex:g});
      var md={mass:0,center:Vec2(),I:0};b.getMassData(md);md.I*=INERTIA_X;b.setMassData(md); // heavier rotational inertia keeps PD stable
      b.setUserData({far:t.far,foot:t.foot,hand:t.hand,bone:[Vec2(M*S*(t.bone[0]-t.c[0]),S*(t.bone[1]-t.c[1])),Vec2(M*S*(t.bone[2]-t.c[0]),S*(t.bone[3]-t.c[1]))]});
      bodies.push(b);
    });
    var joints=[],pend=[];
    JT.forEach(function(jt,k){
      var t=T[jt.child],q=tr[jt.child];
      var j=w.createJoint(planck.RevoluteJoint({lowerAngle:M>0?jt.lo:-jt.hi,upperAngle:M>0?jt.hi:-jt.lo,enableLimit:true,referenceAngle:0},bodies[IDX[t.p]],bodies[IDX[jt.child]],Vec2(x+M*q.ax,y+q.ay+lift)));
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
    var real=bodies,rjoints=joints;
    if(M<0){bodies=real.map(mirrorBody);joints=rjoints.map(mirrorJoint);} // the controller sees a +x-facing rig; the mirror flips everything it reads and writes
    var rc=JT.map(function(jt,k){return {j:rjoints[k],a:real[IDX[T[jt.child].p]],b:real[IDX[jt.child]]};});
    var ctrls=JT.map(function(jt,k){
      return {j:joints[k],a:bodies[IDX[T[jt.child].p]],b:bodies[IDX[jt.child]],g:jt.g,kp:GAIN[k].kp*gs,kd:GAIN[k].kd*gs,max:GAIN[k].max*gs,target:0,t:0};
    });
    var feet=[bodies[IDX.ftf],bodies[IDX.ftn]];
    var soft=FLAGS.soft&&Math.abs(ra)<0.3&&(o.h||0)<0.05,tgt0=new Float64Array(NJ); // soft start (only when spawned upright on the feet): PD targets begin at the spawn pose
    if(soft)for(i=0;i<NJ;i++)tgt0[i]=pose[i];
    return {parts:bodies,bodies:real,rc:rc,dir:M,scale:S,wid:WD,feet:feet,ctrls:ctrls,stiff:new Float64Array(NJ).fill(1),tgt:tgt0,age:soft?0:1e9,prevO:new Float64Array(NO),prevO2:new Float64Array(NO),prevW:new Float64Array(NJ),
      lo:-0.10,hi:0.16,noC:0,fc:[false,false],e:0,airT:0,landT:0,fallT:0,rs:new Float64Array(NJ).fill(1),fcT:0,quiet:0,xi:0,st:{ph:0,t:0,leg:0,next:0,hold:[0,0,0],dir:1,xl:0},gu:{ph:-1,t:0,downT:0,tries:0,seq:'prone',from:null},dh:[],dp:new Float64Array([1,0,0,0,0]),wph:0,wph2:0,lph:0,wsg:1,b2ok:false,nz:new Float64Array(10),nr:rngMake(12345),st2:{ph:0,t:0,leg:0,next:0,calm:0,x0:0,xl:0,Ts:0.25,clr:0.09,dir:0},com:{x:0,y:0.95,vx:0,contact:false},hist:[],prev:new Float64Array(NJ)};
  }

  // MIRROR: a rig facing -x is built mirrored in the world, and the controller talks to it through these wrappers,
  // which reflect x (positions, velocities, forces) and flip the sign of angles, spins, torques and joint limits.
  // Everything above them (balance, reflexes, get-up, injuries) then works unchanged.
  function mv(v){return Vec2(-v.x,v.y);}
  function mirrorBody(b){
    return {_r:b,
      getPosition:function(){return mv(b.getPosition());},getWorldCenter:function(){return mv(b.getWorldCenter());},
      getLinearVelocity:function(){return mv(b.getLinearVelocity());},getAngle:function(){return -b.getAngle();},
      getAngularVelocity:function(){return -b.getAngularVelocity();},getWorldPoint:function(p){return mv(b.getWorldPoint(mv(p)));},
      getLinearVelocityFromWorldPoint:function(p){return mv(b.getLinearVelocityFromWorldPoint(mv(p)));},getLocalCenter:function(){return mv(b.getLocalCenter());},
      applyTorque:function(t,w){b.applyTorque(-t,w);},applyLinearImpulse:function(J,p,w){b.applyLinearImpulse(mv(J),mv(p),w);},
      applyForce:function(F,p,w){b.applyForce(mv(F),mv(p),w);},
      setLinearVelocity:function(v){b.setLinearVelocity(mv(v));},setAngularVelocity:function(w){b.setAngularVelocity(-w);},
      getMass:function(){return b.getMass();},getInertia:function(){return b.getInertia();},getContactList:function(){return b.getContactList();},
      getFixtureList:function(){return b.getFixtureList();},getUserData:function(){return b.getUserData();},isDynamic:function(){return b.isDynamic();},
      getWorld:function(){return b.getWorld();},setAwake:function(f){b.setAwake(f);},getJointList:function(){return b.getJointList();}};
  }
  function mirrorJoint(j){
    return {_r:j,
      getJointAngle:function(){return -j.getJointAngle();},getJointSpeed:function(){return -j.getJointSpeed();},
      setLimits:function(lo,hi){j.setLimits(-hi,-lo);},getLowerLimit:function(){return -j.getUpperLimit();},getUpperLimit:function(){return -j.getLowerLimit();},
      enableLimit:function(f){j.enableLimit(f);},isLimitEnabled:function(){return j.isLimitEnabled();},
      getReactionForce:function(h){return mv(j.getReactionForce(h));},getAnchorA:function(){return mv(j.getAnchorA());},getAnchorB:function(){return mv(j.getAnchorB());},
      getBodyA:function(){return j.getBodyA();}};
  }

  // PD: torque = kp*(target-angle) - kd*angularVelocity, clamped. Per-joint stiffness scales kp, kd, and the torque cap.
  function pdRig(r){
    var cs=r.ctrls,st=r.stiff;
    for(var i=0;i<cs.length;i++){
      var c=cs[i],ks=st[i];
      if(r.inj&&r.inj.gone[i])continue; // torn off
      var sl=0;
      if(r.inj&&r.inj.blo[i]!=null){ // broken: soft ends to the new range, it slows down (and is pushed back) near them
        var a=c.j.getJointAngle(),w=c.j.getJointSpeed(),m=0.45,lo=r.inj.blo[i],hi=r.inj.bhi[i],p;
        if(a<lo+m){p=(lo+m-a)/m;sl=c.max*0.5*p*p-(w<0?c.kd*3*p*w:0);}
        else if(a>hi-m){p=(a-(hi-m))/m;sl=-c.max*0.5*p*p-(w>0?c.kd*3*p*w:0);}
      }
      if(sl){c.b.applyTorque(sl,true);c.a.applyTorque(-sl,true);}
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
  // Reactions are driven by the detector's guess, and by what the body is actually doing right now (trunk spin,
  // contact), with smooth random variation so no two falls look the same:
  //  falling / being moved -> arms windmill in the direction the body is tipping (that's what fights the rotation),
  //   unevenly and out of step with each other; legs kick and bicycle when the feet are off the ground, and scramble
  //   for footing once the balance controller has given up;  stumbling -> arms come up a little;
  //  about to hit the ground -> hands reach toward it, chin tucks.
  function ou(rig,k,dt,tau){ // smooth random signal, unit size, ~tau s correlation
    var n=rig.nz;n[k]+=-n[k]*dt/tau+Math.sqrt(2*dt/tau)*gauss(rig.nr);return n[k];
  }
  function nnReact(rig,tt,dt){
    var p=rig.dp,ps=rig.parts,hy=ps[0].getPosition().y,ca=wrap(ps[1].getAngle()),om=ps[1].getAngularVelocity();
    var fallP=Math.min(1,Math.max(0,(p[2]-0.35)/0.4)),movP=Math.min(1,Math.max(0,(p[3]-0.35)/0.4)),stP=Math.min(1,Math.max(0,(p[1]-0.3)/0.4));
    var lying=hy<0.65||lowY(ps[3])<0.06||lowY(ps[1])<0.06; // torso on the ground: done flailing, just settle
    var wild=lying?0:Math.max(fallP,movP),dir=(-ca+2*rig.e)>0?1:-1,k;
    if(lying){ // down: go loose and let the knees and hips fold, instead of holding stiff poses (no balancing on the head)
      stP=0;for(k=0;k<NJ;k++)rig.stiff[k]=Math.min(rig.stiff[k],0.3);
      tt[6]+=0.5;tt[12]+=0.4;tt[7]-=0.8;tt[13]-=0.7;
    }
    var n=[];for(k=0;k<10;k++)n.push(ou(rig,k,dt,k<6?0.22:0.35));
    // ---- arms ----
    var spin=clampA(om/2.5,1);                                  // + = tipping backward (counter-clockwise)
    var rate=2*Math.PI*(1.1+1.6*Math.abs(spin)+0.9*movP+0.35*n[0]);
    var sgn=Math.abs(spin)>0.15?(spin>0?1:-1):(rig.wsg||1);rig.wsg=sgn;   // windmill the same way the body tips
    rig.wph+=dt*rate*sgn;rig.wph2+=dt*rate*sgn*(1+0.25*n[1]);         // the two arms drift in and out of step
    var A=1.0+0.35*n[2],c0=1.0+0.4*movP+0.3*n[3];
    var sF=c0+A*Math.sin(rig.wph)-0.5*spin,sN=c0+(A*(0.85+0.2*n[4]))*Math.sin(rig.wph2+2.2)-0.5*spin;
    var eF=0.25+0.9*(0.5+0.5*Math.sin(rig.wph+1.4+0.4*n[5])),eN=0.25+0.9*(0.5+0.5*Math.sin(rig.wph2+3.6));
    // stumbling: arms come up a little, unevenly (bigger arm swings here cost balance: measured)
    var bF=0.3+0.1*n[2],bN=0.22+0.1*n[4],beF=0.45+0.1*n[5],beN=0.5+0.1*n[3];
    var aF,aN,aeF,aeN,mix=Math.max(wild,stP);
    aF=tt[3];aN=tt[9];aeF=tt[4];aeN=tt[10];
    if(mix>0){var wW=wild/(wild+stP+1e-9);aF=sF*wW+bF*(1-wW);aN=sN*wW+bN*(1-wW);aeF=eF*wW+beF*(1-wW);aeN=eN*wW+beN*(1-wW);}
    var brace=p[2]>0.5&&hy<1.2&&hy>0.45&&!(rig.fc[0]&&rig.fc[1]&&rig.b2ok);
    if(brace){aF=(dir>0?1.35:-0.9)+0.2*n[2];aN=(dir>0?1.2:-0.8)+0.2*n[4];aeF=0.35;aeN=0.45;mix=1;tt[0]=-0.35;}
    // hands on the ground (or lying): elbows give and the arms go soft, so they cushion instead of pole-vaulting the body over
    var handDown=lowY(ps[HAND_IDX[0]])<0.05||lowY(ps[HAND_IDX[1]])<0.05;
    if((handDown&&!rig.b2ok)||lying){
      if(brace||handDown){aeF=aeN=1.3;mix=1;}
      for(k=3;k<6;k++){rig.stiff[k]=Math.min(rig.stiff[k],0.35);rig.stiff[k+6]=Math.min(rig.stiff[k+6],0.35);}
    }
    if(mix>0){
      tt[3]=tt[3]*(1-mix)+aF*mix;tt[9]=tt[9]*(1-mix)+aN*mix;tt[4]=tt[4]*(1-mix)+aeF*mix;tt[10]=tt[10]*(1-mix)+aeN*mix;
      tt[5]+=0.35*wild*n[6];tt[11]+=0.35*wild*n[7];
      tt[0]+=0.12*wild*n[8];
    }
    // ---- legs: only when the balance controller isn't using them ----
    if(rig.b2ok||wild<=0)return;
    var air=true;for(k=0;k<ps.length;k++)if(lowY(ps[k])<0.05){air=false;break;} // nothing touching the ground
    rig.lph+=dt*2*Math.PI*(1.4+0.6*movP+0.3*n[9]);
    var L=LEGK,lp=rig.lph;
    if(air){ // kicking / bicycling, uneven
      tt[L[0][0]]+=wild*(0.35+0.55*Math.sin(lp)+0.2*n[0]);tt[L[0][1]]+=wild*(-0.7-0.5*(0.5+0.5*Math.sin(lp+1.3)));
      tt[L[1][0]]+=wild*(0.35+0.55*Math.sin(lp+2.8+0.5*n[1])+0.2*n[3]);tt[L[1][1]]+=wild*(-0.7-0.5*(0.5+0.5*Math.sin(lp+4.0)));
      tt[L[0][2]]+=0.3*wild*n[6];tt[L[1][2]]+=0.3*wild*n[7];
    }else if(hy>0.8){ // feet down but past saving: scramble a leg out toward the fall, knees give
      var lead=Math.sin(lp*0.5)>0?0:1;
      tt[L[lead][0]]+=wild*(dir>0?0.8:-0.45);tt[L[lead][1]]+=wild*-0.45;
      tt[L[1-lead][1]]+=wild*-0.6;tt[L[1-lead][0]]+=wild*0.25;
    }
  }

  // =====================================================================
  // GET UP (FLAGS.getup): when the detector says "down" and the body has settled, get up the way people do.
  //  on the back: rock up to sitting with a leg swing -> tuck the feet in and fold forward over them -> crouch on the
  //  feet -> stand, then hand back to balance once the knees are nearly straight (easing into its targets, no snap).
  //  face down: push up -> hands and knees -> one foot forward into a lunge -> push up off the front foot -> stand.
  //  Each stage moves one side 0.3 s after the other, so hands and knees move one at a time.
  // Stages blend toward a target pose; some hold the trunk at a set angle in WORLD frame with the hips (so it adapts
  // to how it is lying). Stages were found by evolution strategies in the sim (one move at a time: sit, crouch, stand).
  // If it slumps back down or is still low at the end, it starts over once it settles.
  // Stage poses are channels: neck, upper spine, lower spine, shoulder, elbow, wrist, hip, knee, ankle (both sides);
  // n = extra [hip, knee, ankle] for the near leg, m = extra [shoulder, elbow, wrist] for the near arm, lag = seconds one
  // side trails the other (so hands and knees move one at a time); trunk = wanted pelvis angle in world frame
  // (held with the hips, weight tw) or null; T = seconds to blend into the pose. Poses were optimised in the sim.
  // =====================================================================
  // Getting up needs more forward bend than the normal limits allow (to get the weight over the feet from sitting),
  // so while it gets up the hips may flex to 2.8 (normally 2.4) and the spine curl to -0.8/-0.6 (normally -0.5/-0.4).
  var GU_LOOSE={1:-0.8,2:-0.6,6:2.8,12:2.8};
  function guLimits(rig,loose){
    if(!!rig.guLoose===loose)return;
    for(var k in GU_LOOSE){
      var j=rig.ctrls[k].j,jt=JT[k],a=j.getJointAngle();
      if(!loose&&(a<jt.lo||a>jt.hi))return; // wait until it is back inside the normal range (no snap)
    }
    for(k in GU_LOOSE){var lim=GU_LOOSE[k],q=JT[k];if(rig.inj&&rig.inj.blo[k]!=null)continue;rig.ctrls[k].j.setLimits(lim<0&&loose?lim:q.lo,lim>0&&loose?lim:q.hi);}
    rig.guLoose=loose;
  }
  var GU={
    prone:[
      {T:0.42,a:[0.08,0.92,0.79,0.66,1.74,0.61,-0.04,-0.15,0.29],n:[0.04,0.16,0.33],trunk:null,tw:0,lag:0.34,m:[-0.33,-0.05,-0.22]},
      {T:0.43,a:[1.32,0.93,-0.18,1.2,-0.22,-0.21,1.84,-2.02,0.72],n:[-0.16,-0.29,0.09],trunk:0.32,tw:0.13,lag:-0.42,m:[0.07,0.16,0.27]},
      {T:0.43,a:[0.42,0.05,-0.21,1.67,-0.14,0.45,1.7,-1.68,0.51],n:[1.58,-0.4,-0.11],trunk:null,tw:0,lag:0.3,m:[-0.04,0.34,-0.13]},
      {T:1.06,a:[0.06,0.48,-0.07,1.18,0.07,0.14,0.93,-1.47,0.55],n:[0.96,0.3,-0.5],trunk:-0.76,tw:1.03,lag:-0.31,m:[-0.23,-0.35,-0.14]},
      {T:0.91,a:[0.43,-0.2,-0.13,0.56,0.56,0.15,0.68,-1.35,0.14],n:[0.91,0.6,-0.27],trunk:-0.36,tw:0.94,lag:0.43,m:[0.11,-0.04,0.3]},
      {T:0.65,a:[-0.19,0.29,0.29,0.41,0.39,-0.09,0.17,-0.63,0.14],n:[0.87,0.13,-0.05],trunk:-0.37,tw:1.08,lag:0.3,m:[-0.25,0.32,0.08]},
      {T:0.84,a:[-0.01,0.26,0,0.44,0.23,0.22,0.05,-0.45,-0.02],n:[0.32,0.05,-0.12],trunk:0.05,tw:0.93,lag:-0.41,m:[0.01,0.18,0.19]},
      {T:0.95,a:[0.03,0.02,-0.02,0.24,0.36,0.2,0.04,-0.03,0],n:[0.04,0.16,-0.15],trunk:0.06,tw:1.1,lag:0.56,m:[-0.29,0.17,0.2]}
    ],
    supine:[
      {T:0.41,a:[-0.41,0.08,0,1.99,0.04,0.04,0.91,-1.6,-0.69],n:[0.03,-0.39,-0.58],trunk:null,tw:0,lag:0.31,m:[0.11,0.03,0.05]},
      {T:0.46,a:[-0.45,-0.35,-0.28,0.81,-0.08,-0.41,0.91,-0.85,-0.05],n:[-0.38,0.55,-0.16],trunk:null,tw:0,lag:-0.34,m:[0.34,-0.01,-0.2]},
      {T:1.26,a:[0.26,-0.62,-0.48,1.64,-0.09,-0.01,2.06,-1.39,0.66],n:[0.17,-0.08,-0.13],trunk:0.14,tw:0.98,lag:0.31,m:[-0.02,0.24,-0.12]},
      {T:0.42,a:[-0.75,-1.05,-0.29,1.97,0.15,0.24,3.1,-2.35,0.35],n:[0,0.11,0.2],trunk:-0.36,tw:0.96,lag:-0.32,m:[-0.32,-0.02,-0.17]},
      {T:0.41,a:[0.02,-0.15,-0.22,1.26,0,0.08,3.54,-2.49,0.85],n:[-0.06,-0.56,0.3],trunk:-1.52,tw:1.5,lag:0.3,m:[0.08,0.08,0.05]},
      {T:0.41,a:[-0.15,-0.01,-0.89,1.91,-0.1,-0.05,1.82,-2.81,0.84],n:[0,-0.02,0.2],trunk:-0.49,tw:1.42,lag:-0.3,m:[0.17,-0.06,-0.01]},
      {T:0.61,a:[0.25,-0.11,0.2,0.8,0.33,0.36,1.35,-1.03,0.26],n:[-0.68,-0.57,0.22],trunk:-0.21,tw:1.08,lag:0.33,m:[-0.17,0.01,0.05]},
      {T:1.22,a:[-0.4,0.28,-0.06,0.34,0.4,-0.45,0.89,-0.73,0.04],n:[-0.42,0.04,0.25],trunk:0.02,tw:1.09,lag:-0.32,m:[-0.12,-0.06,0.12]},
      {T:0.84,a:[-0.2,0.07,0.1,0.01,0.25,0.46,0.15,0.14,0],n:[0.14,-0.17,0.08],trunk:0.06,tw:1.06,lag:0.34,m:[0,0.07,-0.01]}
    ]};
  function getUp(rig,tt,dt){
    var g=rig.gu,ps=rig.parts,hy=ps[0].getPosition().y,pa=wrap(ps[3].getAngle()),k;
    if(g.ph<0){
      var slow=Math.abs(rig.com.vx)<0.25&&Math.abs(ps[3].getAngularVelocity())<1.0;
      g.downT=(rig.dp[4]>0.6&&slow&&hy<0.9)?g.downT+dt:0;
      if(g.downT<0.8)return false;
      g.seq=pa<0?'prone':'supine';if(!GU[g.seq].length)return false;g.ph=0;g.t=0;g.from=Float64Array.from(rig.tgt);g.base=new Float64Array(NJ);g.downT=0;g.peak=hy;
    }
    if(hy>1.5&&Math.cos(ps[1].getAngle())>0.9&&(rig.fc[0]||rig.fc[1])&&Math.abs(rig.ctrls[7].j.getJointAngle())<0.35&&Math.abs(rig.ctrls[13].j.getJointAngle())<0.35){
      g.ph=-1;g.tries=0;rig.age=Math.min(rig.age,0.3);return false; // up on its feet: balance takes over, easing into its targets (no snap)
    }
    var S=GU[g.seq],st=S[g.ph];g.t+=dt;
    var lag=st.lag||0,sm=0;
    for(k=0;k<NJ;k++){ // one side moves a beat after the other (lag > 0: far side late, < 0: near side late)
      var late=k>=3&&(lag>0?k<9:k>=9),u=Math.min(1,Math.max(0,(g.t-(late?Math.abs(lag):0))/st.T)),s1=u*u*(3-2*u);
      if(!late)sm=s1;
      var c=k<3?k:3+((k-3)%6),v=st.a[c];if(st.n&&k>=12)v+=st.n[k-12];if(st.m&&k>=9&&k<12)v+=st.m[k-9];
      tt[k]=g.base[k]=g.from[k]+(v-g.from[k])*s1;
    }
    if(st.trunk!=null){ // hips hold the pelvis at the wanted world angle (both legs that are on the ground)
      var pw=ps[3].getAngularVelocity(),corr=clampA(1.2*(pa-st.trunk)+0.03*pw,0.8)*sm*(st.tw==null?1:st.tw);
      tt[6]+=corr;tt[12]+=corr;
    }
    for(k=0;k<NJ;k++){rig.rs[k]=1;rig.stiff[k]=(JT[k].g===2&&(k%6)!==5)?FLAGS.guArm:1;} // arms push harder while getting up
    if(g.t>=st.T+Math.abs(st.lag||0)+0.25){ // stage done: next one (or hand back once standing)
      if(g.ph===S.length-1){g.ph=-1;if(hy<1.4){g.tries++;}else{g.tries=0;rig.age=Math.min(rig.age,0.3);}return hy>=1.4;}
      g.from=Float64Array.from(g.base);g.ph++;g.t=0;
    }
    g.peak=Math.max(g.peak,hy);
    if(hy<g.peak-0.5){g.ph=-1;g.tries++;} // slumped back down: start over once it settles
    return true;
  }

  // COWER (FLAGS.cower): if the head, chest or pelvis hits the ground hard (velocity change in one tick while touching
  // the ground above cowerImp m/s, e.g. toppling over from a big shove or landing flat from high up), curl up and cover
  // the head for a moment (longer for harder hits), trembling a little, before trying to get up.
  // Pushes from the buttons don't count: the chest isn't touching the ground when it's shoved.
  var COWER=[-0.45,-0.5,-0.4, 2.3,2.4,0.6, 1.7,-2.3,0.3]; // chin tucked, back curled, arms folded over the head, knees up
  function onGround(b){for(var c=b.getContactList();c;c=c.next)if(c.contact.isTouching()&&c.other.isStatic())return true;return false;}
  // ground impacts this tick (velocity change of a part while it touches the ground; pushes don't count because the
  // shoved chest isn't on the ground): impT = worst of head/chest/pelvis (cowering), impD = how close any part came to a
  // fatal hit (dying). Real falls (feet-first ~6% die vs ~45-57% head/front/side first; overall LD50 ~4 storeys):
  // head hits kill from ~4-5 m head-first (dieHead), flat torso hits from ~12-16 m (dieTorso), feet-first only from
  // ~16-20 m (dieLimb, internal injuries) - lower falls on the feet break legs/back instead
  function impacts(rig){
    var ps=rig.parts,n=ps.length,k;
    if(!rig.cpv)rig.cpv=new Float64Array(2*n);
    rig.impS=0;if(!rig.impB)rig.impB=new Float64Array(n);
    var it=0,id=0,live=(rig.ticks=(rig.ticks||0)+1)>12; // ignore the settle right after spawning
    for(k=0;k<n;k++){
      var v=ps[k].getLinearVelocity(),d=0;
      if(live&&onGround(ps[k]))d=Math.hypot(v.x-rig.cpv[2*k],v.y-rig.cpv[2*k+1]);
      rig.cpv[2*k]=v.x;rig.cpv[2*k+1]=v.y;
      rig.impB[k]=d;if(d>2.5)ev(rig,'hit',k,d);
      if(k===0)rig.impH=d;
      if(k>=1&&k<=3&&d>rig.impS){rig.impS=d;rig.impSk=k;} // worst torso hit (spine fractures)
      if(k===0||k===1||k===3)it=Math.max(it,d);
      id=Math.max(id,d/(k===0?FLAGS.dieHead:k<4?FLAGS.dieTorso:FLAGS.dieLimb));
    }
    rig.impT=it;rig.impD=id*FLAGS.dieImp; // impD > dieImp = dead
  }
  function cower(rig,tt,dt){
    var imp=FLAGS.cower?rig.impT||0:0,k;
    if(hurt(rig)){if(!(rig.cowT>0))rig.cowA=0;rig.cowT=Math.max(rig.cowT||0,1);rig.gu.ph=-1;} // a broken bone: stays curled up
    if(imp>FLAGS.cowerImp&&(rig.cowT||0)<0.5){rig.cowT=Math.min(3.5,1.5+0.6*(imp-FLAGS.cowerImp));rig.cowA=0;rig.gu.ph=-1;}
    if(!(rig.cowT>0))return false;
    rig.cowT-=dt;rig.cowA=(rig.cowA||0)+dt;
    if(rig.cowT<=0)rig.age=0; // done: ease back into whatever comes next (no snap)
    var tr=0.06*Math.min(1,rig.cowT)*Math.sin(rig.cowA*55); // a little shaking, fading out at the end
    for(k=0;k<NJ;k++){
      var c=k<3?k:3+((k-3)%6);tt[k]=COWER[c]+((c===3||c===4)?tr:0);
      rig.rs[k]=rig.stiff[k]=0.75;
    }
    return true;
  }

  // DIE (FLAGS.die): an impact above dieImp kills it: every joint goes limp (just a little friction) for good.
  function goLimp(rig,out){
    for(var k=0;k<NJ;k++){rig.rs[k]=rig.stiff[k]=0.003;var a=rig.ctrls[k].j.getJointAngle();rig.tgt[k]=a;rig.ctrls[k].target=a;}
    rig.gu.ph=-1;rig.cowering=false;return out;
  }

  // INJURIES (FLAGS.inj: bones break, head knocks cause knockouts; FLAGS.sever: limbs can be torn off).
  // Load on each joint = its constraint force smoothed over ~0.1 s (single ticks are too spiky to use). A bone breaks
  // when that load passes BRK for the joint (arms are weakest); a broken joint loses its limits (it can spin all the
  // way round) and it and everything beyond it go limp. Broken neck = dead. Broken back = legs paralysed.
  // Any broken bone makes it cower for as long as it is conscious. A limb comes off when the smoothed force pulling
  // the joint apart (not pushing it together) passes SEV: only hard pulls (dragging) do that, never falls or pushes.
  // Measured (smoothed load): standing <10 N, big shoves <45, 3 m drop on the feet 60-90, 8 m drop 110-170 (legs/back),
  // dragging a hand/foot/head away 120-130. Pulling apart: everything <35 except dragging (115-130).
  // Head hits (velocity change on the ground) above koImp knock it out for a few seconds: a big shove onto its back
  // gives ~4.5, a 3 m head-first drop ~6.6. Dying still uses dieImp (where head hits count 1.4x).
  var BRK=[130,115,115, 60,60,60,105,100,80, 60,60,60,105,100,80],SEV=95;
  var JNAMES=['neck','upper back','lower back','far shoulder','far elbow','far wrist','far hip','far knee','far ankle','near shoulder','near elbow','near wrist','near hip','near knee','near ankle'];
  var KIDS=[[0],[1],[2],[3,4,5],[4,5],[5],[6,7,8],[7,8],[8],[9,10,11],[10,11],[11],[12,13,14],[13,14],[14]];
  function injOf(rig){return rig.inj||(rig.inj={load:new Float64Array(NJ),pull:new Float64Array(NJ),broken:new Uint8Array(NJ),gone:new Uint8Array(NJ),blo:new Array(NJ).fill(null),bhi:new Array(NJ).fill(null),t:0});}
  function hurt(rig){var I=rig.inj;if(!I)return false;for(var k=0;k<NJ;k++)if(I.broken[k]||I.gone[k])return true;return false;}
  function breakBone(rig,k,on){
    var I=injOf(rig);if(I.gone[k]||!!I.broken[k]===!!on)return;
    I.broken[k]=on?1:0;
    if(on){ // a broken joint gets a random amount of extra range each way (not a free spin); pdRig softens the new ends
      var jt=JT[k],ex1=0.5+Math.random()*1.3,ex2=0.5+Math.random()*1.3,lo=jt.lo-ex1,hi=jt.hi+ex2,sp=hi-lo;
      if(sp>5.6){lo+=(sp-5.6)/2;hi-=(sp-5.6)/2;}
      I.blo[k]=lo;I.bhi[k]=hi;rig.ctrls[k].j.setLimits(lo,hi);rig.ctrls[k].j.enableLimit(true);
      if(k===0&&FLAGS.death!==0)rig.dead=true;ev(rig,'break',k);}
    else rig.age=Math.min(rig.age,0.3); // healed: the limit comes back once the joint is inside its range again (below)
  }
  function addStrand(rig,k,A,B,la,lb,wt,o){ // a bundle of strings (one rope) between two bodies
    o=o||{};var L0=o.L||0.06,w=A.getWorld(),rp=w.createJoint(planck.RopeJoint({maxLength:L0,localAnchorA:Vec2(la.x,la.y),localAnchorB:Vec2(lb.x,lb.y)},A,B));
    (rig.strands||(rig.strands=[])).push({j:rp,k:k,A:A,B:B,la:Vec2(la.x,la.y),lb:Vec2(lb.x,lb.y),L:L0,max:o.max||0.3+0.15*Math.random(),wt:Math.max(o.minW||5,wt),fs:0,n:o.n||2+(Math.random()*3|0),thin:!!o.thin,seed:Math.random()*6.28});
  }
  function sever(rig,k,noStr){
    var I=injOf(rig);if(I.gone[k]||k<3)return; // limbs only
    var j=rig.ctrls[k].j,rj=j._r||j,w=rj.getBodyA().getWorld(),A=rj.getBodyA(),B=rj.getBodyB(),la=rj.getLocalAnchorA(),lb=rj.getLocalAnchorB();
    w.destroyJoint(rj);
    if(FLAGS.strands&&!noStr){ // still hanging on by strands of muscle, tendon and gut: they stretch under a pull and snap
      var wt=0;KIDS[k].forEach(function(q){if(!I.gone[q])wt+=rig.parts[q+1].getMass()*10;}); // weight of what's hanging off it
      addStrand(rig,k,A,B,la,lb,wt);
    }
    KIDS[k].forEach(function(q){I.gone[q]=1;I.broken[q]=0;});
    ev(rig,'sever',k);
  }
  // SHATTER (FLAGS.shatter): a crushed limb part breaks into 2-4 chunks along its length. Each gap is either clean
  // (the chunks fly apart) or still held by a few thin strings; same for the stump end and whatever hung off the far end.
  // The part's own body becomes the chunk at the joint end; the others are new bodies listed in rig.chunks {b,k}.
  function shatter(rig,j,hard){
    var I=injOf(rig);if(I.gone[j])return;
    var rj=rig.ctrls[j].j,rj=rj._r||rj,B=rj.getBodyB(),lb=rj.getLocalAnchorB(),w=B.getWorld(),k=j+1;
    var nx=j%3<2&&!I.gone[j+1]?rig.ctrls[j+1].j:null,nr=nx?(nx._r||nx):null,C=nr?nr.getBodyB():null,ncA=nr?nr.getLocalAnchorA():null,ncB=nr?nr.getLocalAnchorB():null;
    var old=(rig.strands||[]).filter(function(s){return s.A===B||s.B===B;}); // strings already tied to this part (e.g. a hand torn off earlier)
    sever(rig,j,hard&&Math.random()<0.5);
    if(nr){w.destroyJoint(nr);}
    var f=B.getFixtureList(),vs=f.getShape().m_vertices,hx=0,hy=0,i;for(i=0;i<vs.length;i++){hx=Math.max(hx,Math.abs(vs[i].x));hy=Math.max(hy,Math.abs(vs[i].y));}
    var ax=hx>hy?0:1,half=ax?hy:hx,thick=ax?hx:hy,sgn=(ax?lb.y:lb.x)>=0?1:-1; // long axis, and which end the joint is at
    var n=2+(Math.random()*(hard?3:2)|0),cuts=[0],t=0;for(i=1;i<n;i++){t+=(1/n)*(0.75+0.5*Math.random());cuts.push(Math.min(0.9,t));}cuts.push(1);
    var den=f.getDensity(),fr=f.getFriction(),g=f.getFilterGroupIndex(),ud=B.getUserData(),ang=B.getAngle(),av=B.getAngularVelocity();
    function seg(a,b){var c=sgn*half*(1-(a+b)),h=half*(b-a);return {c:c,h:h};} // along the axis, from the joint end
    var prev=null,prevEnd=null,pieces=[],sgs=[];
    for(i=0;i<n;i++){
      var sg=seg(cuts[i],cuts[i+1]),lc=ax?Vec2(0,sg.c):Vec2(sg.c,0),hb=ax?Box(thick*(0.85+0.15*Math.random()),sg.h*0.92,lc,0):Box(sg.h*0.92,thick*(0.85+0.15*Math.random()),lc,0),body;
      if(i===0){B.destroyFixture(f);B.createFixture(hb,{density:den,friction:fr,restitution:0,filterGroupIndex:g});body=B;}
      else{
        var wp=B.getWorldPoint(lc);body=w.createBody({type:'dynamic',position:wp,angle:ang});
        body.createFixture(ax?Box(thick*(0.85+0.15*Math.random()),sg.h*0.92):Box(sg.h*0.92,thick*(0.85+0.15*Math.random())),{density:den,friction:fr,restitution:0,filterGroupIndex:g});
        var v=B.getLinearVelocityFromWorldPoint(wp),kick=hard?2.5:1.2;
        body.setLinearVelocity(Vec2(v.x+(Math.random()-0.5)*kick,v.y+Math.random()*kick));body.setAngularVelocity(av+(Math.random()-0.5)*8);
        body.setUserData(ud);(rig.chunks||(rig.chunks=[])).push({b:body,k:k});
        lc=Vec2(0,0);
      }
      var near=ax?Vec2(lc.x,lc.y+sgn*sg.h*0.9):Vec2(lc.x+sgn*sg.h*0.9,lc.y),far=ax?Vec2(lc.x,lc.y-sgn*sg.h*0.9):Vec2(lc.x-sgn*sg.h*0.9,lc.y);
      if(prev&&Math.random()>(hard?0.55:0.3))addStrand(rig,j,prev,body,prevEnd,near,Math.min(prev.getMass(),body.getMass())*10,{L:0.03,n:3+(Math.random()*3|0),thin:true,minW:1.5,max:0.15+0.15*Math.random()});
      pieces.push(body);prev=body;prevEnd=far;sgs.push(sg);
    }
    // re-tie the old strings to whichever piece now has their end (else they'd hang off empty space where the part was)
    old.forEach(function(s){
      var aSide=s.A===B,la=aSide?s.la:s.lb,u=ax?la.y:la.x,bi=0,bd=1e9;
      sgs.forEach(function(g,q){var d=Math.abs(u-g.c)-g.h;if(d<bd){bd=d;bi=q;}});
      var g=sgs[bi],uc=Math.max(g.c-g.h*0.9,Math.min(g.c+g.h*0.9,u)),P=pieces[bi],wp=B.getWorldPoint(ax?Vec2(la.x,uc):Vec2(uc,la.y)),nl=P.getLocalPoint(wp);
      if(P===B&&uc===u)return;
      try{w.destroyJoint(s.j);}catch(e){}
      s.age=0;if(aSide){s.A=P;s.la=Vec2(nl.x,nl.y);}else{s.B=P;s.lb=Vec2(nl.x,nl.y);}
      s.j=w.createJoint(planck.RopeJoint({maxLength:s.L,localAnchorA:s.la,localAnchorB:s.lb},s.A,s.B));
    });
    if(C){ // whatever hung off the far end (hand, foot, forearm...) stays in one piece
      var last=pieces[pieces.length-1];
      if(Math.random()>(hard?0.5:0.25))addStrand(rig,j,last,C,prevEnd,ncB,C.getMass()*10,{L:0.04,n:3+(Math.random()*2|0),thin:true,minW:2});
    }
  }
  // events for the page (blood, sounds): {t:'break'|'sever'|'crush'|'hit', k: joint (or body for hits), d: strength}
  function ev(rig,t,k,d){
    (rig.ev||(rig.ev=[])).push({t:t,k:k,d:d||0});if(rig.ev.length>64)rig.ev.shift();
    var sp=t==='break'?(k<3?0.6:0.45):t==='sever'?0.8:t==='crush'?0.7:t==='hit'&&d>6?(d-6)*0.08:0; // pain spike
    if(sp&&!rig.dead)rig.painS=Math.min(2,(rig.painS||0)+sp);
  }
  // PAIN (with FLAGS.inj): sharp pain from each new injury (fades over ~10 s) on top of a dull ache from every bone
  // still broken or limb missing. Past 1 it passes out for a few seconds; after coming round it gets a few seconds'
  // grace before it can pass out again (so "break everything" leaves it drifting in and out).
  function pain(rig,dt){
    var I=rig.inj,base=0,k;
    if(I)for(k=0;k<NJ;k++){if(I.broken[k])base+=k<3?0.25:k%3===0?0.18:0.12;else if(I.gone[k]&&(k%3===0||!I.gone[k-1]))base+=0.3;}
    rig.painS=Math.max(0,(rig.painS||0)-0.1*dt);rig.wakeT=Math.max(0,(rig.wakeT||0)-dt);
    rig.pain=Math.min(2,0.6*base+rig.painS);
    if(rig.pain>1&&!rig.dead&&!(rig.koT>0)&&rig.wakeT<=0){rig.koT=6+4*Math.min(1,rig.pain-1);rig.koWhy='pain';rig.painS*=0.5;}
  }
  // CRUSH (FLAGS.crush): a limb part slammed into the ground hard enough is crushed (its joint breaks); harder still
  // and it's ripped off (with FLAGS.sever). A head slammed that hard is crushed: dead.
  // BLEED (FLAGS.bleed): each torn-off limb bleeds ~5%/s of its blood (heart pumping, slower as the blood runs out),
  // broken bones a little. Under 45% it passes out, under 20% it's dead.
  // per-part crush thresholds (upper arm, forearm, hand, thigh, shin, foot): big parts barely change speed when they
  // hit (the rest of the body carries them), hands and feet slam hard, so each is set so all get crushed about as often
  var CRUSHL=[4,7,11,5,4.5,11];
  function crushBleed(rig,dt){
    var I=injOf(rig),ps=rig.parts,k;
    if(FLAGS.crush&&I.t>0.2&&rig.impB){
      for(k=0;k<ps.length;k++){
        var lim=k===0?FLAGS.crushImp:k<4?1e9:CRUSHL[(k-4)%6]*FLAGS.crushLimb;
        var d=rig.impB[k];if(!(d>lim))continue;
        if(k===0){if(!rig.dead&&FLAGS.death!==0){rig.dead=true;ev(rig,'crush',0,d);}continue;}
        if(k<4)continue;
        var j=k-1;if(I.gone[j])continue;
        if(FLAGS.shatter&&FLAGS.sever){ev(rig,'crush',j,d);shatter(rig,j,d>lim*FLAGS.ripImp);}
        else if(FLAGS.sever&&d>lim*FLAGS.ripImp){ev(rig,'crush',j,d);sever(rig,j);}
        else if(!I.broken[j]){ev(rig,'crush',j,d);breakBone(rig,j,true);}
      }
    }
    if(rig.blood==null)rig.blood=1;
    if(FLAGS.bleed&&rig.blood>0){
      var w=0;for(k=3;k<NJ;k++){if(I.gone[k]&&(k===3||k===6||k===9||k===12||!I.gone[k-1]))w+=k%3===0?1:0.7;else if(I.broken[k])w+=0.05;}
      if(I.broken[1]||I.broken[2])w+=0.1;
      rig.bleedW=w;
      rig.blood=Math.max(0,rig.blood-(FLAGS.bleedMul==null?1:FLAGS.bleedMul)*0.05*w*(0.3+0.7*rig.blood)*(rig.dead?0.3:1)*dt);
      if(rig.blood<0.2&&!rig.dead&&FLAGS.death!==0){rig.dead=true;ev(rig,'bledout',0);}
      else if(rig.blood<0.45&&!rig.dead){rig.koT=Math.max(rig.koT||0,0.5);rig.koWhy='blood';}
    }
  }
  function strandStep(rig,dt){ // strands stretch while pulled hard (and slowly sag under a dangling limb's weight), then snap
    var S=rig.strands;if(!S)return;
    for(var i=S.length-1;i>=0;i--){
      var s=S[i],f=0;try{var F=s.j.getReactionForce(1/DT);f=Math.hypot(F.x,F.y)||0;}catch(e){} // (planck throws before its first step)
      // only just holds the limb's own weight: it sags under it, any real tug (or a jolt) snaps it
      s.fs+=(f-s.fs)*0.06;s.f=f;if(s.fs>0.7*s.wt)s.L+=Math.min(1,(s.fs-0.7*s.wt)/(2*s.wt))*dt;
      s.age=(s.age||0)+dt;if(s.L>=s.max||(s.fs>1.8*s.wt&&s.age>0.3)){s.j.getBodyA().getWorld().destroyJoint(s.j);S.splice(i,1);ev(rig,'snap',s.k,s.fs/s.wt);}
      else s.j.setMaxLength(s.L);
    }
  }
  function injuries(rig,dt){
    var I=injOf(rig),k;I.t+=dt;strandStep(rig,dt);
    for(k=0;k<NJ;k++){
      if(I.gone[k])continue;
      var c=rig.ctrls[k],j=c.j,F=j.getReactionForce(1/DT),an=j.getAnchorB(),cb=c.b.getWorldCenter(),dx=cb.x-an.x,dy=cb.y-an.y,dl=Math.hypot(dx,dy)||1;
      I.load[k]+=(Math.hypot(F.x,F.y)-I.load[k])*0.08;
      I.pull[k]+=(Math.max(0,-(F.x*dx+F.y*dy)/dl)-I.pull[k])*0.08;
      if(I.t>0.2){ // not during the settle after spawning
        if(FLAGS.inj&&!I.broken[k]&&I.load[k]>BRK[k])breakBone(rig,k,true);
        if(FLAGS.sever&&k>=3&&I.pull[k]>SEV){sever(rig,k);continue;}
      }
      if(!I.broken[k]&&I.blo[k]!=null){var a0=j.getJointAngle();if(a0>=JT[k].lo&&a0<=JT[k].hi){j.setLimits(JT[k].lo,JT[k].hi);I.blo[k]=I.bhi[k]=null;}else{I.blo[k]=Math.min(JT[k].lo,a0);I.bhi[k]=Math.max(JT[k].hi,a0);j.setLimits(I.blo[k],I.bhi[k]);}} // healed: normal range once back inside it
      if(!I.broken[k]&&!j.isLimitEnabled()){var a=j.getJointAngle();if(a>=j.getLowerLimit()&&a<=j.getUpperLimit())j.enableLimit(true);}
    }
    if(FLAGS.inj&&I.t>0.2&&(rig.impS||0)>9)breakBone(rig,rig.impSk===1?1:2,true); // landing flat and hard on the back/front: spine fracture
    if(FLAGS.inj&&(rig.impH||0)>FLAGS.koImp&&!rig.dead){rig.koT=Math.max(rig.koT||0,5+3*(rig.impH-FLAGS.koImp));rig.koWhy='head';}
    if(FLAGS.inj)pain(rig,dt);
  }
  function limp1(rig,q){rig.rs[q]=rig.stiff[q]=0.003;var a=rig.ctrls[q].j.getJointAngle();rig.tgt[q]=a;rig.ctrls[q].target=a;}
  function injLimp(rig){ // broken or severed joints (and the limb beyond them) can't be moved
    var I=rig.inj;if(!I)return;
    for(var k=0;k<NJ;k++)if(I.broken[k]||I.gone[k])KIDS[k].forEach(function(q){limp1(rig,q);});
    if(I.broken[1]||I.broken[2])[6,7,8,12,13,14].forEach(function(q){limp1(rig,q);});
  }

  // PROTECT HEAD (FLAGS.protect): while falling toward the ground (detector says falling, or it's in the air), with the
  // head coming down, bring the forearms up around the head and tuck the chin; held a moment after it lands.
  // falling forward: forearms up in front of the face (a guard); otherwise: hands behind the head
  var COVER_F={0:-0.4,3:1.5,4:2.2,9:1.4,10:2.3},COVER_B={0:-0.45,3:2.6,4:2.3,9:2.5,10:2.4};
  function protectHead(rig,tt,dt){
    var h=rig.parts[0],hp=h.getPosition(),hv=h.getLinearVelocity();
    var falling=(rig.dp[2]>0.4||rig.airT>0.2)&&hv.y<-0.8&&hp.y<1.45&&hp.y>0.3;
    if(falling){rig.protT=0.35;rig.protF=wrap(rig.parts[1].getAngle())<-0.3||hv.x>0.5;}
    else rig.protT=Math.max(0,(rig.protT||0)-dt);
    var want=rig.protT>0?1:0;rig.prot=(rig.prot||0)+(want-(rig.prot||0))*(want?0.3:0.08);
    if(rig.prot<0.01)return;
    var C=rig.protF?COVER_F:COVER_B;
    for(var k in C){tt[k]+=(C[k]-tt[k])*rig.prot;if(k>0){rig.rs[k]=Math.max(rig.rs[k],0.6*rig.prot);rig.stiff[k]=rig.rs[k];}}
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
    if(rf&&(FLAGS.cower||FLAGS.die||FLAGS.inj||FLAGS.crush))impacts(rig);
    if(FLAGS.inj||FLAGS.sever||rig.inj)injuries(rig,dt);
    if(rf&&(FLAGS.crush||FLAGS.bleed))crushBleed(rig,dt);
    if(rf&&FLAGS.die&&FLAGS.death!==0&&rig.impD>FLAGS.dieImp)rig.dead=true;
    if(rig.dead)return goLimp(rig,out);
    if(rig.koT>0){rig.koT-=dt;if(rig.koT<=0){rig.age=0;rig.wakeT=4;rig.groggy=FLAGS.groggyT;}goLimp(rig,out);rig.ko=true;return out;} // knocked out: limp until it comes round
    rig.ko=false;
    // coming round after being out: muscles come back slowly (weak and floppy at first), no getting up until halfway
    var gw=1;if(rig.groggy>0){rig.groggy-=dt;var gx=Math.max(0,1-rig.groggy/FLAGS.groggyT);gw=0.03+0.97*gx*gx;}
    var cowering=rf&&(FLAGS.cower||hurt(rig))&&cower(rig,tt,dt);rig.cowering=!!cowering;
    var gettingUp=!cowering&&rf&&FLAGS.getup&&gw>0.3&&getUp(rig,tt,dt);
    guLimits(rig,!!gettingUp);
    rig.b2ok=!cowering&&!gettingUp&&rf&&FLAGS.bal2?bal2(rig,tt,out,ca,dt):false;
    if(!cowering&&!gettingUp&&rf&&!FLAGS.bal2&&FLAGS.step)stepReflex(rig,tt,ca,dt);
    if(!cowering&&!gettingUp&&rf&&FLAGS.nn)nnReact(rig,tt,dt);
    if(!cowering&&!gettingUp&&rf&&FLAGS.protect)protectHead(rig,tt,dt);
    // target smoothing: the network can slow it down (smoother) or speed it up; soft start eases in after spawning
    var al=clampA(FILT*(1+0.6*out[O_FILT]),1);
    if(rig.age<FLAGS.softT){var x=rig.age/FLAGS.softT;al*=0.04+0.96*x*x*(3-2*x);}
    rig.age+=dt;
    for(k=0;k<NJ;k++){
      rig.tgt[k]+=(tt[k]-rig.tgt[k])*al;
      rig.ctrls[k].target=rig.tgt[k];
    }
    if(gw<1)for(k=0;k<NJ;k++){rig.rs[k]=Math.min(rig.rs[k],gw);rig.stiff[k]=rig.rs[k];}
    injLimp(rig);
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

  return {JNAMES:JNAMES,breakBone:breakBone,sever:sever,shatter:shatter,hurt:hurt,DCLS:DCLS,DIN:DIN,detFeat:detFeat,detForward:detForward,setDet:function(w){DET=w;},hasDet:function(){return !!DET;},GU:GU,FLAGS:FLAGS,buildRig:buildRig,pdRig:pdRig,act:act,sense:sense,newBuf:newBuf,forward:forward,initParams:initParams,rngMake:rngMake,gauss:gauss,
    esNoise:esNoise,esStep:esStep,migrate:migrate,runEpisode:runEpisode,evalCandidate:evalCandidate,sampleScenario:sampleScenario,fk:fk,
    NP:NP,NI:NI,NO:NO,NJ:NJ,DT:DT,SUB:SUB,POLICY_HZ:POLICY_HZ,VEL_IT:VEL_IT,POS_IT:POS_IT,JT:JT,ORDER:ORDER,CASES:CASES,wrap:wrap};
})();
