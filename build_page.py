sim=open(__import__('os').path.join(__import__('os').path.dirname(__file__),'sim.js')).read()
ui=open(__import__('os').path.join(__import__('os').path.dirname(__file__),'ui.js')).read()
html='''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover">
<title>Equilibria</title>
<style>
:root{box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px);--bg:#f2f2f2;--fg:#111;--btn:#ddd;--panel:#ffffffd9}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#111;--fg:#eee;--btn:#333;--panel:#222d}}
:root[data-theme="dark"]{--bg:#111;--fg:#eee;--btn:#333;--panel:#222d}
html,body{height:100%;margin:0;overflow:hidden;background:var(--bg);color:var(--fg);font-family:system-ui,Arial,sans-serif;-webkit-user-select:none;user-select:none;-webkit-touch-callout:none}
canvas#c{display:block;width:100%;height:100%;touch-action:none}
#bar{position:fixed;left:0;right:0;bottom:calc(10px + env(safe-area-inset-bottom,0px));display:flex;gap:6px;justify-content:center;flex-wrap:wrap;padding:0 8px}
button{font:600 15px system-ui,Arial,sans-serif;padding:11px 14px;border:0;border-radius:10px;background:var(--btn);color:var(--fg)}
#hint{position:fixed;bottom:calc(112px + env(safe-area-inset-bottom,0px));left:0;right:0;text-align:center;font-size:12px;opacity:.7;pointer-events:none;padding:0 8px}
#panels{position:fixed;top:calc(8px + env(safe-area-inset-top,0px));left:8px;display:flex;flex-direction:column;gap:6px;max-height:calc(100% - 150px);overflow:auto;max-width:calc(100% - 16px)}
details{background:var(--panel);border-radius:10px;padding:6px 10px;font-size:14px;backdrop-filter:blur(6px)}
summary{font-weight:600;padding:2px 0}
label{display:flex;align-items:center;gap:8px;padding:5px 0;white-space:nowrap}
details input[type=checkbox]{width:18px;height:18px;margin:0}
details input[type=number]{width:54px;font-size:14px;padding:3px}
.row{display:flex;gap:8px;padding:5px 0;flex-wrap:wrap}
details button{font-size:14px;padding:8px 12px}
#info{font-size:12px;line-height:1.45;font-variant-numeric:tabular-nums;white-space:pre-wrap;max-width:300px}
#chart{width:220px;height:70px;display:block;margin-top:4px}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="panels">
<details id="trn" open>
  <summary>Train (evolution strategies)</summary>
  <div class="row"><button id="bTrain">Start training</button><button id="bResetP">Reset policy</button><button id="bTest">Test</button></div>
  <div id="info">Generation 0</div>
  <canvas id="chart" width="440" height="140"></canvas>
  <label>Pairs <input type="number" id="nPairs" value="16" min="2" max="64"> Scenarios <input type="number" id="nScen" value="3" min="1" max="8"></label>
  <label>Workers <input type="number" id="nWork" value="-1" min="-1" max="16"> <span style="opacity:.6">-1 = auto, 0 = none</span></label>
  <label><input type="checkbox" id="oUse"> Use policy (watch it)</label>
  <label><input type="checkbox" id="oReflex" checked> Balance reflex</label>
  <label><input type="checkbox" id="oBal2" checked> New balance + recovery steps</label>
  <label><input type="checkbox" id="oStep" checked> Old step reflex (if New balance off)</label>
  <label><input type="checkbox" id="oFall"> Fall &amp; landing reflex (experimental)</label>
  <label><input type="checkbox" id="oSmooth" checked> Train for smooth, human-like motion</label>
  <label><input type="checkbox" id="oSoft" checked> Soft start: ease into standing</label>
  <div class="row"><button id="bSave">Save file</button><button id="bLoad">Load file</button></div>
  <input type="file" id="fIn" accept=".json,application/json" style="display:none">
  <div class="row"><button id="pL">&#9664; Push</button><button id="pR">Push &#9654;</button></div>
  <label>Strength <input type="range" id="pS" min="0.1" max="3" step="0.1" value="1" style="width:110px;height:auto"></label>
</details>
<details id="dbg">
  <summary>Debug</summary>
  <label><input type="checkbox" id="oBox" checked> Boxes</label>
  <label><input type="checkbox" id="oSkel"> Skeleton</label>
  <label><input type="checkbox" id="oJoint"> Joints</label>
  <label><input type="checkbox" id="oCom"> COM &amp; support</label>
  <label><input type="checkbox" id="oPD" checked> Pose (PD on)</label>
  <label><input type="checkbox" id="oSlow"> Slow-mo</label>
</details>
</div>
<div id="hint">1 finger: drag ragdoll or pan · 2 fingers: zoom</div>
<div id="bar">
  <button id="bSpawn">Spawn</button>
  <button id="bDrop">Drop</button>
  <button id="bFall">Fallen</button>
  <button id="bReset">Reset</button>
  <button id="bIn">+</button>
  <button id="bOut">&minus;</button>
</div>
<script src="https://cdn.jsdelivr.net/npm/planck@1.0.0/dist/planck.min.js"></script>
<script id="simsrc">
'''+sim+'''
</script>
<script>
'''+ui+'''
</script>
</body>
</html>
'''
open(__import__('os').path.join(__import__('os').path.dirname(__file__),'ragdoll_trainer.html'),'w').write(html)
print(len(html))
