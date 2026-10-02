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
<meta name="theme-color" content="#2c3035">
<style>
:root{box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px);--bg:#2c3035;--fg:#e6e9ed;--btn:#3d434b;--panel:#23272cdd}
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
  <button id="bSpawn">&#x2795; Spawn</button>
  <button id="bPause">Pause</button>
  <button id="bReset2">Reset</button>
  <button id="bIn">+</button>
  <button id="bOut">&minus;</button>
</div>
<div id="hud"></div>
<div id="hmenu" hidden>
  <button id="hmRename">Rename</button>
  <button id="hmFollow">Follow</button>
  <button id="hmTurn">Turn around</button>
  <button id="hmHeal">Heal</button>
  <button id="hmKill">Kill</button>
  <button id="hmRemove">Remove</button>
</div>
<div id="rename" class="ov" hidden>
  <h2>Name this human</h2>
  <input id="rnIn" maxlength="16" placeholder="Name" autocomplete="off">
  <div class="row"><button id="rnOk">Save</button><button id="rnCancel">Cancel</button></div>
</div>
<div id="pmenu" class="ov" hidden>
  <h2>Paused</h2>
  <button id="mResume">Resume</button>
  <button id="bSlowB" class="tg">Slow motion: off</button>
  <button id="mSound" class="tg">Sound: on</button>
  <button id="mGore" class="tg">Blood &amp; gore: on</button>
  <button id="bClean">Clean up blood</button>
  <button id="bReset">Reset scene</button>
  <button id="mMain">Main menu</button>
  <div class="ver">beta v0.1.16</div>
</div>

<div id="splash">
  <h1>EQUILIBRIA</h1>
  <div><span class="tag">BETA</span></div>
  <p>A ragdoll that tries to stay alive. It balances, catches itself, gets back up and bleeds.</p>
  <p>Drag a body part to throw it around. Pinch to zoom. Use the buttons to spawn, drop or knock them over.</p>
  <div class="ver">beta v0.1.16</div>
  <button id="bPlay">Play</button>
  <div class="row" style="justify-content:center"><button id="sSound" class="tg sm">Sound: on</button><button id="sGore" class="tg sm">Blood &amp; gore: on</button></div>
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
