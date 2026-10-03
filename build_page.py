sim=open(__import__('os').path.join(__import__('os').path.dirname(__file__),'sim.js')).read()
ui=open(__import__('os').path.join(__import__('os').path.dirname(__file__),'ui.js')).read()
detw=open(__import__('os').path.join(__import__('os').path.dirname(__file__),'det_weights.js')).read()
html='''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover">
<title>Equilibria (beta)</title>
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#0c0b0a">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Teko:wght@400;500;600&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
:root{box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px);--bg:#0c0b0a;--fg:#e9e4d8;--btn:#151310;--panel:#0f0e0ce6}
html,body{height:100%;margin:0;overflow:hidden;background:var(--bg);color:var(--fg);font-family:system-ui,Arial,sans-serif;-webkit-user-select:none;user-select:none;-webkit-touch-callout:none}
canvas#c{display:block;width:100%;height:100%;touch-action:none}
#bar{position:fixed;left:0;right:0;bottom:calc(10px + env(safe-area-inset-bottom,0px));display:flex;gap:6px;justify-content:center;flex-wrap:nowrap;padding:0 8px}
button{font:600 15px system-ui,Arial,sans-serif;padding:11px 14px;border:0;border-radius:10px;background:var(--btn);color:var(--fg)}
#hint{position:fixed;bottom:calc(112px + env(safe-area-inset-bottom,0px));left:0;right:0;text-align:center;font-size:12px;opacity:0;color:#dfe3e8;pointer-events:none;padding:0 8px}
#panels{position:fixed;top:calc(8px + env(safe-area-inset-top,0px));left:8px;display:flex;flex-direction:column;gap:6px;max-height:calc(100% - 150px);overflow:auto;max-width:calc(100% - 16px)}
details{background:var(--panel);border-radius:10px;padding:6px 10px;font-size:14px;backdrop-filter:blur(6px)}
summary{font-weight:600;padding:2px 0}
label{display:flex;align-items:center;gap:8px;padding:5px 0;white-space:nowrap}
#iTab{border-collapse:collapse}#iTab td{padding:3px 10px 3px 0}#iTab input{width:22px;height:22px}
details input[type=checkbox]{width:18px;height:18px;margin:0}
details input[type=number]{width:54px;font-size:14px;padding:3px}
.row{display:flex;gap:8px;padding:5px 0;flex-wrap:wrap}
details button{font-size:14px;padding:8px 12px}
#info{font-size:12px;line-height:1.45;font-variant-numeric:tabular-nums;white-space:pre-wrap;max-width:300px}
#chart{width:220px;height:70px;display:block;margin-top:4px}
button{-webkit-tap-highlight-color:transparent}button:active{filter:brightness(1.3)}
#bar button{min-width:44px;white-space:nowrap;box-shadow:0 2px 0 #0006}#bar button.on{background:#c0392b}
#hint{transition:opacity 1s}
#hud{position:fixed;top:calc(8px + env(safe-area-inset-top,0px));right:8px;display:flex;flex-direction:column;gap:4px;pointer-events:none;font:600 12px system-ui,Arial,sans-serif}
.hc{background:var(--panel);border-radius:8px;padding:5px 8px;min-width:110px;pointer-events:auto;cursor:pointer}.hc:active{filter:brightness(1.4)}
#hmenu{position:fixed;z-index:8;display:flex;flex-direction:column;gap:4px;background:var(--panel);padding:6px;border-radius:12px;box-shadow:0 4px 16px #0008}#hmenu[hidden]{display:none}#hmenu button{font-size:14px;padding:9px 14px;text-align:left}
.ver{position:absolute;right:12px;bottom:calc(10px + env(safe-area-inset-bottom,0px));font:600 12px system-ui,Arial,sans-serif;opacity:.55}
#rnIn{font:600 18px system-ui,Arial,sans-serif;padding:10px 12px;border-radius:10px;border:0;width:220px;background:#eef0f3;color:#111}
.ov .row button{min-width:100px}
.hb{height:6px;border-radius:3px;background:#0006;margin-top:4px;overflow:hidden}.hb i{display:block;height:100%;background:#d63031}
#splash[hidden]{display:none}
#splash{position:fixed;inset:0;background:radial-gradient(circle at 50% 40%,#3a4048,#1c1f23);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;z-index:10;text-align:center;padding:24px}
#splash h1{margin:0;font:800 40px system-ui,Arial,sans-serif;letter-spacing:6px}
#splash .tag{display:inline-block;background:#c0392b;color:#fff;border-radius:6px;padding:2px 8px;font-size:12px;letter-spacing:2px;vertical-align:middle}
#splash p{margin:0;opacity:.75;font-size:14px;line-height:1.6;max-width:300px}
#splash #bPlay{font-size:18px;padding:14px 46px;background:#c0392b;color:#fff;border-radius:12px}
#splash .sm{font-size:13px;padding:8px 12px}
.ov{position:fixed;inset:0;background:#000a;backdrop-filter:blur(4px);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;z-index:9}
.ov[hidden]{display:none}.ov h2{margin:0 0 6px;font:800 26px system-ui,Arial,sans-serif;letter-spacing:3px}
.ov button{min-width:220px;font-size:16px}
#smenu{z-index:11}.sl{width:260px;max-width:86vw}.slh{display:flex;justify-content:space-between;font:600 15px system-ui,Arial,sans-serif;margin-bottom:4px}
.sl input{width:100%;accent-color:#c0392b;height:28px}.sld{display:flex;justify-content:space-between;font-size:11px;opacity:.6}.slx{font-size:12px;opacity:.7;max-width:260px;text-align:center;min-height:32px}
/* in-game: same dark CRT look as the menus */
body::after{content:"";position:fixed;inset:0;pointer-events:none;z-index:5;background:repeating-linear-gradient(0deg,rgba(0,0,0,.22) 0 1px,transparent 1px 3px);box-shadow:inset 0 0 90px 10px #000c}
#bar button{font:500 20px/1 Teko,system-ui,sans-serif;letter-spacing:1px;text-transform:uppercase;padding:9px 10px 6px;flex:0 1 auto;min-width:0;border-radius:2px;background:#11100ee6;border:1px solid #3a342a;color:#c9c2b4;box-shadow:none}
#bar button:active{color:#ffb020;border-color:#ffb020;filter:none;text-shadow:0 0 10px #ffb02088}
#bar button.on{background:#ffb020;color:#0c0b0a;border-color:#ffb020}
#hint{font:400 12px 'Share Tech Mono',monospace;color:#a9a397}
#hud{font:400 12px 'Share Tech Mono',monospace}
.hc{border-radius:2px;border:1px solid #2c2720;border-left:2px solid #ffb020;background:#0f0e0ce6;color:#d8d2c4}
.hb{border-radius:0;background:#2a2520}
#hmenu{border-radius:2px;border:1px solid #3a342a;background:#0f0e0cf2}
#hmenu button{font:500 22px/1 Teko,system-ui,sans-serif;letter-spacing:1.5px;text-transform:uppercase;background:none;border-radius:0;color:#c9c2b4;padding:8px 14px 5px}
#hmenu button:active{color:#ffb020;filter:none}
#spmenu{position:fixed;z-index:8;left:8px;bottom:calc(64px + env(safe-area-inset-bottom));display:flex;flex-direction:column;gap:2px;padding:6px;border:1px solid #3a342a;background:#0f0e0cf2;box-shadow:0 4px 16px #0008}#spmenu[hidden]{display:none}
#spmenu button{font:500 22px/1 Teko,system-ui,sans-serif;letter-spacing:1.5px;text-transform:uppercase;background:none;border-radius:0;color:#c9c2b4;padding:8px 14px 5px;text-align:left}
#spmenu button:active{color:#ffb020;filter:none}
#rename.gm{align-items:flex-start!important}#rename .gt{font-size:clamp(34px,10vw,52px)}#rename .row{gap:16px}
#rename .row button{all:unset;cursor:pointer;font:500 30px/1.1 Teko,system-ui,sans-serif;letter-spacing:2px;text-transform:uppercase;color:#a9a397}#rename .row button:active{color:#ffb020}
#rnIn{font:400 18px 'Share Tech Mono',monospace;border-radius:0;background:#14120f;color:#ffb020;border:1px solid #3a342a;outline:none}
#hlist{z-index:9}#hlc{display:flex;flex-direction:column;gap:5px;width:300px;max-width:82vw;max-height:58vh;overflow-y:auto;font:400 13px 'Share Tech Mono',monospace;-webkit-overflow-scrolling:touch}
#hlc .hc{padding:7px 10px;cursor:pointer}
.hdd{font-size:13px}.hdd .dda{color:#ffb020;margin-left:4px}
/* glitchy dark menus (main, pause, settings): CRT scanlines, film noise, chromatic split title, flicker */
.gm{background:radial-gradient(ellipse at 30% 40%,#16130f 0%,#070707 55%,#000 100%)!important;backdrop-filter:none!important;align-items:flex-start!important;justify-content:center!important;text-align:left!important;
  padding:24px 24px 24px max(28px,10vw)!important;gap:8px!important;font-family:Teko,'Share Tech Mono',system-ui,sans-serif;color:#e9e4d8;overflow:hidden;animation:gmShake 7s infinite steps(1)}
#pmenu.gm{background:radial-gradient(ellipse at 30% 40%,#16130fee 0%,#070707f2 55%,#000 100%)!important}
.gm::before{content:"";position:absolute;inset:-50%;pointer-events:none;opacity:.09;z-index:0;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");animation:gmNoise .4s infinite steps(4)}
.gm::after{content:"";position:absolute;inset:0;pointer-events:none;z-index:3;
  background:repeating-linear-gradient(0deg,rgba(0,0,0,.38) 0 1px,transparent 1px 3px),linear-gradient(90deg,rgba(255,0,0,.025),rgba(0,255,255,.02));
  box-shadow:inset 0 0 120px 30px #000;animation:gmFlick 5s infinite}
.gm>*{position:relative;z-index:1}.gm>.ver{position:absolute}
.gm .gt{margin:0;font:600 clamp(54px,17vw,92px)/0.85 Teko,system-ui,sans-serif;letter-spacing:5px;color:#f2ede1;text-shadow:0 0 18px #ffb02033}
#pmenu .gt,#smenu .gt,#hlist .gt{font-size:clamp(46px,13vw,70px)}
.gt::before,.gt::after{content:attr(data-t);position:absolute;left:0;top:0;width:100%;overflow:hidden;pointer-events:none}
.gt::before{color:#ff2a1f;mix-blend-mode:screen;transform:translate(-2px,0);clip-path:inset(0 0 60% 0);animation:gmT1 2.6s infinite steps(1)}
.gt::after{color:#19e6ff;mix-blend-mode:screen;transform:translate(2px,0);clip-path:inset(55% 0 0 0);animation:gmT2 3.1s infinite steps(1)}
.gm .sub{font:400 15px 'Share Tech Mono',monospace;letter-spacing:1px;opacity:.6;text-transform:lowercase;margin:-2px 0 18px}
#splash.gm .tag{text-transform:uppercase;letter-spacing:2px;vertical-align:1px;background:none;border:1px solid #ffb020;color:#ffb020;border-radius:0;padding:0 5px;font-size:11px;margin-right:6px}
.gm nav{display:flex;flex-direction:column;gap:2px;margin:6px 0}
.gm nav button,#splash #bPlay{all:unset;cursor:pointer;font:500 34px/1.1 Teko,system-ui,sans-serif;letter-spacing:2px;text-transform:uppercase;color:#a9a397;padding:2px 0;white-space:nowrap;position:relative;transition:color .08s,transform .08s;-webkit-tap-highlight-color:transparent}
.gm nav button::before{content:">";position:absolute;left:-22px;color:#ffb020;opacity:0;transition:opacity .08s}
.gm nav button:hover,.gm nav button:active,.gm nav button:focus-visible{color:#ffb020;transform:translateX(8px);text-shadow:0 0 12px #ffb02088,-2px 0 #ff2a1f66,2px 0 #19e6ff55}
.gm nav button:hover::before,.gm nav button:active::before{opacity:1}
.gm nav button.on{color:#e9e4d8}
#splash #bPlay{font-size:46px;color:#f2ede1}
#pmenu nav button{font-size:30px}
.gm .tip{font:400 12px 'Share Tech Mono',monospace;opacity:.4;margin-top:14px;max-width:300px}
.gm .ver{font:400 13px 'Share Tech Mono',monospace;opacity:.45;color:#ffb020}
.gm .sl{width:300px;max-width:78vw;margin:4px 0}.gm .slh{font:500 26px/1 Teko,system-ui,sans-serif;letter-spacing:1px;text-transform:uppercase;color:#a9a397}.gm .slh b{color:#ffb020;font-weight:500}
.gm .sl input{accent-color:#ffb020}.gm .sld{font:400 11px 'Share Tech Mono',monospace}.gm .slx{font:400 12px 'Share Tech Mono',monospace;text-align:left;max-width:300px;opacity:.55}
@keyframes gmNoise{0%{transform:translate(0,0)}25%{transform:translate(-8%,5%)}50%{transform:translate(6%,-7%)}75%{transform:translate(-4%,-3%)}}
@keyframes gmFlick{0%,100%{opacity:1}47%{opacity:1}48%{opacity:.75}49%{opacity:1}82%{opacity:.9}83%{opacity:1}}
@keyframes gmShake{0%,100%{transform:none;filter:none}62%{transform:translate(-3px,1px) skewX(-2deg);filter:hue-rotate(20deg) contrast(1.3)}62.6%{transform:translate(4px,0);filter:none}63.2%{transform:none}91%{transform:translate(2px,-1px)}91.5%{transform:none}}
@keyframes gmT1{0%{clip-path:inset(0 0 60% 0);transform:translate(-2px,0)}20%{clip-path:inset(20% 0 50% 0);transform:translate(-5px,0)}22%{clip-path:inset(0 0 60% 0);transform:translate(-2px,0)}60%{clip-path:inset(70% 0 5% 0);transform:translate(4px,0)}62%{clip-path:inset(0 0 60% 0);transform:translate(-2px,0)}}
@keyframes gmT2{0%{clip-path:inset(55% 0 0 0);transform:translate(2px,0)}35%{clip-path:inset(10% 0 75% 0);transform:translate(6px,0)}37%{clip-path:inset(55% 0 0 0);transform:translate(2px,0)}80%{clip-path:inset(40% 0 40% 0);transform:translate(-4px,0)}82%{clip-path:inset(55% 0 0 0);transform:translate(2px,0)}}
@media (prefers-reduced-motion:reduce){.gm,.gm::before,.gm::after,.gt::before,.gt::after{animation:none}}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="panels">
<div hidden><!-- training, policy and behaviour switches: off the page; behaviours stay on (checked) -->
<details id="trn" open>
  <summary>Train (evolution strategies)</summary>
  <div class="row"><button id="bTrain">Start training</button><button id="bResetP">Reset policy</button><button id="bTest">Test</button></div>
  <div id="info">Generation 0</div>
  <canvas id="chart" width="440" height="140"></canvas>
  <label>Pairs <input type="number" id="nPairs" value="16" min="2" max="64"> Scenarios <input type="number" id="nScen" value="3" min="1" max="8"></label>
  <label>Workers <input type="number" id="nWork" value="-1" min="-1" max="16"> <span style="opacity:.6">-1 = auto, 0 = none</span></label>
  <label><input type="checkbox" id="oNN" checked> Neural reactions (shows what it senses)</label>
  <label><input type="checkbox" id="oGetup" checked> Get up when down (needs Neural reactions)</label>
  <label><input type="checkbox" id="oCower" checked> Cower after hard hits</label>
  <label><input type="checkbox" id="oProtect" checked> Protect head when falling</label>
  <label><input type="checkbox" id="oDie" checked> Can die from really hard hits</label>
  <label><input type="checkbox" id="oUse"> Use old ES policy (watch it)</label>
  <label><input type="checkbox" id="oReflex" checked> Balance reflex</label>
  <label><input type="checkbox" id="oBal2" checked> New balance + recovery steps</label>
  <label><input type="checkbox" id="oStep" checked> Old step reflex (if New balance off)</label>
  <label><input type="checkbox" id="oFall" checked> Fall &amp; landing reflex (experimental)</label>
  <label><input type="checkbox" id="oSmooth" checked> Train for smooth, human-like motion</label>
  <label><input type="checkbox" id="oSoft" checked> Soft start: ease into standing</label>
  <div class="row"><button id="bSave">Save file</button><button id="bLoad">Load file</button></div>
  <input type="file" id="fIn" accept=".json,application/json" style="display:none">
</details>
<details id="injp">
  <summary>Injuries</summary>
  <label><input type="checkbox" id="oInj" checked> Bones can break, head hits knock it out</label>
  <label><input type="checkbox" id="oSever" checked> Limbs come off if pulled hard enough</label>
  <div class="row"><label><input type="checkbox" id="iKO"> Knocked out</label><label><input type="checkbox" id="iDead"> Dead</label></div>
  <div style="opacity:.7">Broken bones (tick to break, untick to heal):</div>
  <div class="row"><label><input type="checkbox" id="iB0"> Neck</label><label><input type="checkbox" id="iB1"> Upper back</label><label><input type="checkbox" id="iB2"> Lower back</label></div>
  <table id="iTab"><tr><td></td><td>Far</td><td>Near</td></tr>
    <tr><td>Shoulder</td><td><input type="checkbox" id="iB3"></td><td><input type="checkbox" id="iB9"></td></tr>
    <tr><td>Elbow</td><td><input type="checkbox" id="iB4"></td><td><input type="checkbox" id="iB10"></td></tr>
    <tr><td>Wrist</td><td><input type="checkbox" id="iB5"></td><td><input type="checkbox" id="iB11"></td></tr>
    <tr><td>Hip</td><td><input type="checkbox" id="iB6"></td><td><input type="checkbox" id="iB12"></td></tr>
    <tr><td>Knee</td><td><input type="checkbox" id="iB7"></td><td><input type="checkbox" id="iB13"></td></tr>
    <tr><td>Ankle</td><td><input type="checkbox" id="iB8"></td><td><input type="checkbox" id="iB14"></td></tr>
  </table>
  <div class="row"><button id="bBreakAll">Break all</button><button id="bHealAll">Heal all</button></div>
</details>
</div>
<div hidden><details id="dbg">
  <summary>Debug</summary>
  <div class="row"><button id="pL">&#9664; Push</button><button id="pR">Push &#9654;</button></div>
  <label>Strength <input type="range" id="pS" min="0.1" max="3" step="0.1" value="1" style="width:110px;height:auto"></label>
  <label><input type="checkbox" id="oGore" checked> Blood &amp; gore</label>
  <label><input type="checkbox" id="oSound" checked> Sound</label>
  <label><input type="checkbox" id="oBox"> Boxes</label>
  <label><input type="checkbox" id="oSkel"> Skeleton</label>
  <label><input type="checkbox" id="oJoint"> Joints</label>
  <label><input type="checkbox" id="oCom"> COM &amp; support</label>
  <label><input type="checkbox" id="oPD" checked> Pose (PD on)</label>
  <label><input type="checkbox" id="oSlow"> Slow-mo</label>
</details></div>
</div>
<div id="hint">1 finger: drag ragdoll or pan · 2 fingers: zoom</div>
<div id="bar">
  <button id="bSpawn">Spawn</button>
  <button id="bPause">Pause</button>
  <button id="bReset2">Reset</button>
  <button id="bIn">+</button>
  <button id="bOut">&minus;</button>
</div>
<div id="hud"></div>
<div id="hlist" class="ov gm" hidden>
  <h2 class="gt" data-t="HUMANS">HUMANS</h2>
  <div id="hlc"></div>
  <nav><button id="hlClose">Close</button></nav>
</div>
<div id="spmenu" hidden><button id="spHuman">Human</button><button id="spGun">Handgun</button><button id="spHammer">Sledgehammer</button><button id="spKnife">Knife</button></div>
<div id="hmenu" hidden>
  <button id="hmRename">Rename</button>
  <button id="hmFollow">Follow</button>
  <button id="hmTurn">Turn around</button>
  <button id="hmHeal">Heal</button>
  <button id="hmKill">Kill</button>
  <button id="hmRemove">Remove</button>
</div>
<div id="rename" class="ov gm" hidden>
  <h2 class="gt" data-t="NAME THIS HUMAN">NAME THIS HUMAN</h2>
  <input id="rnIn" maxlength="16" placeholder="Name" autocomplete="off">
  <div class="row"><button id="rnOk">Save</button><button id="rnCancel">Cancel</button></div>
</div>
<div id="pmenu" class="ov gm" hidden>
  <h2 class="gt" data-t="PAUSED">PAUSED</h2>
  <nav>
  <button id="mResume">Resume</button>
  <button id="bSlowB" class="tg">Slow motion: off</button>
  <button id="mSound" class="tg">Sound: on</button>
  <button id="mSet">Settings</button>
  <button id="bClean">Clean up blood</button>
  <button id="bReset">Reset scene</button>
  <button id="mMain">Main menu</button>
  </nav>
  <div class="ver">beta v0.2.21</div>
</div>

<div id="smenu" class="ov gm" hidden>
  <h2 class="gt" data-t="SETTINGS">SETTINGS</h2>
  <div class="sl"><div class="slh"><span>Realism</span><b id="vReal">Default</b></div><input type="range" id="rReal" min="0" max="2" step="1" value="1"><div class="sld"><span>glass</span><span>realistic</span></div><div class="slx" id="rDesc"></div></div>
  <div class="sl"><div class="slh"><span>Crush threshold</span><b id="vCrush">7</b></div><input type="range" id="rCrush" min="1" max="10" step="1" value="7"><div class="sld"><span>crushes easily</span><span>almost never</span></div></div>
  <div class="sl"><div class="slh"><span>Gore</span><b id="vGore">Full</b></div><input type="range" id="rGore" min="0" max="4" step="1" value="4"><div class="sld"><span>none</span><span>full</span></div></div>
  <div class="slx" id="gDesc"></div>
  <nav><button id="sDone">Done</button></nav>
</div>
<div id="splash" class="gm">
  <h1 class="gt" data-t="EQUILIBRIA">EQUILIBRIA</h1>
  <div class="sub"><span class="tag">BETA</span> a ragdoll that tries to stay alive</div>
  <nav>
  <button id="bPlay">Play</button>
  <button id="sSet">Settings</button>
  <button id="sSound" class="tg">Sound: on</button>
  </nav>
  <p class="tip">drag a body part to throw it &middot; pinch to zoom</p>
  <div class="ver">beta v0.2.21</div>
</div>
<script src="https://cdn.jsdelivr.net/npm/planck@1.0.0/dist/planck.min.js"></script>
<script id="simsrc">
'''+detw+sim+'''
</script>
<script>
'''+ui+'''
</script>
</body>
</html>
'''
open(__import__('os').path.join(__import__('os').path.dirname(__file__),'ragdoll_trainer.html'),'w').write(html)
print(len(html))
