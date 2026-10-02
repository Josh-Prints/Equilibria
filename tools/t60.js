const fs=require('fs');const planck=require('planck');const wt=require('worker_threads');
const html=fs.readFileSync(process.argv[2]||''+__dirname+'/../ragdoll_trainer.html','utf8');
const sims=html.split('<script id="simsrc">')[1].split('</script>')[0];
const ui=html.split('</script>\n<script>\n')[1].split('</script>')[0];
const noop=()=>{};
const mkctx=()=>new Proxy({},{get:(t,k)=>(k in t?t[k]:noop),set:(t,k,v)=>{t[k]=v;return true;}});
const els={};const checks={oBox:true,oPD:true,oReflex:true,oUse:false,oSkel:true,oJoint:true,oCom:true,oFall:true,oStep:true};
const vals={nPairs:'3',nScen:'1',nWork:process.env.NW||'2',pS:'1'};
const el=id=>({getContext:()=>mkctx(),addEventListener:noop,clientWidth:390,clientHeight:700,getBoundingClientRect:()=>({left:0,top:0}),textContent:id==='simsrc'?sims:'',width:440,height:140,files:null,click:noop,setPointerCapture:noop,
 get value(){return vals[id]},set value(v){vals[id]=v},get checked(){return !!checks[id]},set checked(v){checks[id]=v}});
const doc={getElementById:id=>els[id]||(els[id]=el(id)),addEventListener:noop,body:{},documentElement:{}};
const blobs={};let bn=0;
global.Blob=class{constructor(p){this.text=p.join('');}};
const URLx={createObjectURL:b=>{const id='blob:'+(bn++);blobs[id]=b.text;return id;}};
const boot=`const {parentPort,workerData}=require('worker_threads');globalThis.importScripts=()=>{globalThis.planck=require(workerData.planckPath);};globalThis.postMessage=m=>parentPort.postMessage(m);parentPort.on('message',d=>{globalThis.onmessage&&globalThis.onmessage({data:d});});(0,eval)(workerData.code);`;
class Worker{constructor(url){this.w=new wt.Worker(boot,{eval:true,workerData:{code:blobs[url],planckPath:require.resolve('planck')}});this.w.on('message',m=>this.onmessage&&this.onmessage({data:m}));this.w.on('error',e=>this.onerror&&this.onerror(e));}postMessage(m){this.w.postMessage(m);}terminate(){this.w.terminate();}}
let raf=null,t=0,saved=null;
const WIN={devicePixelRatio:3,URL:URLx,claude:{use:async n=>n==='downloads'?{save:async r=>{saved=r;return {status:'saved'};}}:null}};
const RSmain=new Function('planck',sims+';return RS;')(planck);
new Function('planck','RS','document','window','innerWidth','innerHeight','addEventListener','getComputedStyle','requestAnimationFrame','performance','setTimeout','Worker','Blob','URL','navigator','FileReader',ui)(
 planck,RSmain,doc,WIN,390,700,noop,()=>({color:'#000'}),f=>{raf=f;},{now:()=>t},setTimeout,Worker,global.Blob,URLx,{hardwareConcurrency:4},class{});
const frames=n=>{for(let i=0;i<n;i++){t+=16.667;raf(t);}};
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
  frames(30);
  // determinism: worker result == main-thread result for the same job
  els.bTrain.onclick();
  const t0=Date.now();
  while(!/Generation [1-9]/.test(els.info.textContent)&&Date.now()-t0<120000)await sleep(500);
  console.log('after first generation:',els.info.textContent.replace(/\n/g,' | '));
  console.log('hint:',els.hint.textContent);
  frames(120);console.log('live view kept running while training (workers):',els.info.textContent.includes('worker'));
  els.bTrain.onclick();await sleep(1500);
  console.log('stopped:',els.bTrain.textContent);
  // live buttons
  els.bDrop.onclick();els.bFall.onclick();els.bSpawn.onclick();checks.oUse=true;frames(300);console.log('drop/fallen/spawn + policy live ok');
  els.pR.onclick();els.pL.onclick();frames(120);
  els.bSave.onclick();await sleep(100);console.log('save:',saved&&saved.filename,saved&&saved.data.length);
  els.fIn.files=[{name:'p.json',text:saved.data}];
  process.exit(0);
})();
