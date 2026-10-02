# Handoff: Equilibria, a 2D active-ragdoll trainer (for a Claude Code session)

Paste this whole file into the Claude Code chat. The source is in `ragdoll-src.zip` (unzip into the repo root; see "Repo layout").

## 1. What this project is

The project is called **Equilibria** (the user chose the name; it was previously "Ragdoll Trainer"). It is a 2D, side-on, People-Playground-style **active ragdoll** that balances and reacts to pushes using physics only (joint torques; no external forces or "motors" that look fake). Inspired by Endorphin/Euphoria. The user wants it **as realistic and human-like as possible**, with a neural network that learns to balance and look human, plus hand-coded reflexes that make it look human (stepping, bracing).

- Physics: **Planck.js 1.0.0** (Box2D port). Rig is all boxes. No graphics polish wanted yet.
- Deliverable surfaces:
  - A single-file HTML **Claude Artifact**: https://claude.ai/artifact/GXqVBmLVjSyRHibRgth1zH ("Equilibria"; live view + in-browser training + save/load file). An older rig-only page is https://claude.ai/artifact/XTLKsUWLyeLGyygW9LmRCa (no training).
  - A Node script `train.js` for multi-core training on GitHub Actions.
- The user only has an **iPhone** (no computer) and dictates messages (expect typos). Everything must be doable from a phone.

## 2. Current state in one paragraph

Rig, PD control, reflexes, multi-scenario training environment, worker-parallel ES trainer, save/load of weights, and the GitHub Actions trainer all work. **The neural network does not meaningfully beat the reflex-only baseline** (ES gives noise-level gains, except on the "drop" scenario). The **ragdoll "falls over super easily"** (user's latest complaint) because push robustness regressed during the rewrite. A **get-up from lying down is only half solved** (sits up, can't stand from sitting). See section 8 for the prioritised to-do list.

## 3. Files (repo layout)

Unzip `ragdoll-src.zip` at the repo root:

| File | What it is |
|---|---|
| `sim.js` | **The whole simulation** as one global `var RS = (function(){...})()`: rig (forward-kinematics posed), PD controller, MLP policy, observation, reflexes, scenarios, reward/episode runner, ES helpers. No DOM. Needs a global `planck`. |
| `ui.js` | Page script: live world, drag/pan/pinch input, Web Worker pool for parallel training, ES loop, Test button, save/load file, debug drawing. |
| `build_page.py` | Assembles `ragdoll_trainer.html` from `sim.js` + `ui.js` (HTML shell/CSS lives inside this script). Output is what was published as the Artifact. |
| `ragdoll_trainer.html` | Built output (identical to the published Artifact). |
| `build_train.py`, `train.head.js`, `train.tail.js` | Regenerate `train.js` = head + `sim.js` + tail. **Never hand-edit `train.js`; edit `sim.js` and run `python3 build_train.py`.** |
| `train.js` | Node multi-core ES trainer (`worker_threads`). Already in the user's GitHub repo. |
| `.github/workflows/train.yml` | `workflow_dispatch` workflow (inputs: minutes, pairs, scenarios, sigma, lr, resume). |
| `tools/` | Headless test/eval scripts (see section 9). |
| `getup/` | Get-up optimisation experiments + the working sit-up solution JSON. |
| `history/` | Older `sim.js` versions (v1, v3, v5) for regression hunting. |

Setup: `npm install planck@1.0.0` at the repo root (all tools `require('planck')`).

**Caveat:** the user's existing GitHub repo only has `train.js` and the workflow. The user also has **two GitHub Pages sites in other repos. Do not touch those.** The training repo is public (needed for free 4-vCPU runners).

## 4. Physics / rig design (and why)

Constants in `sim.js`: `DT=1/240` (240 Hz substeps), `LIMF=0.35`, `TMAX=1`, `INERTIA_X=16`, `POLICY_HZ=120` (`SUB=2` substeps per policy step), solver iterations `VEL_IT=6, POS_IT=2`, `FILT=0.6` (target smoothing), `W0=3.24` (capture-point time constant).

- **Side-on, faces +x.** Joint angle = child − parent, 0 = standing, **positive = counter-clockwise = swings forward**. Knees bend negative; elbows positive.
- **15 joints, fixed order:** neck, upper spine, lower spine, then far limb set (shoulder, elbow, wrist, hip, knee, ankle), then near limb set (same). Index 6/12 hips, 7/13 knees, 8/14 ankles, 3/9 shoulders.
- **16 bodies:** head, chest, abdomen, pelvis (indices 0-3), then far set (ua, fa, hand, thigh, shin, foot), near set. Both legs overlap in the side view (far set drawn faded).
- **Posed by forward kinematics** (`fk`) from any joint-angle vector + root pose, using `referenceAngle: 0` on every RevoluteJoint so "angle 0" always means standing. This is how lying/crumpled/dropped starts are built.
- Joint limits (rad): neck ±0.5, upper spine -0.5..0.9, lower spine -0.4..0.6, shoulder -1.2..3.1, elbow 0..2.6, wrist ±0.9, hip -0.7..2.4, knee -2.5..0, ankle ±0.7.
- **Foot: 0.26 m long** (half-length 0.13, centre +0.03 from ankle), 0.10 m tall. The user asked for smaller feet; measured trade-off (balance-only push survival, impulse units): 0.30 m → 0.7 fwd / 0.5 back; 0.26 → same; 0.24 → 0.5 / 0.3; 0.20 → 0.3 / 0.3.
- **PD:** `torque = kp*(target-angle) - kd*angularVelocity`, clamped to `±kp*TMAX*stiffness`; equal and opposite torque on both bodies.
- **Explicit PD explodes on light bodies.** Gains are derived from inertia: `kp = w^2*I_reduced`, `kd = 2*w*I_reduced` (w = 15 for neck/arms, 80 for spine/legs), computed once from the standing pose, then **capped per body** so the sum of gains on any one body stays below `(LIMF/DT)^2 * I` (leaves-first order matters: core keeps leftover stiffness). **`INERTIA_X=16` inflates every body's rotational inertia (mass unchanged)** or the PD buzzes. `LIMF` 0.5 buzzes/explodes once mass/gain randomisation is on; 0.35 is stable across the randomisation range.
- **Stiffness groups (legs, spine/neck, arms):** scale `kp`, `kd*sqrt(s)`, and torque cap. Network output sets them: `s = 0.2 + 0.8*sigmoid(2*out+2)`. At zero output `s = 0.905` (slightly below full stiffness).
- Densities: head 4, chest 3, abdomen 4, pelvis 6, limbs 3, feet 4, hands 3 (tuned so no body is too light for its joints).

## 5. Policy and reflexes

**Network** (`NI=64, NH=40, NO=33, NP=5593` since the smooth-motion change; was `49/40/18/4378`, tanh, plain JS arrays; weights file format below). Inputs 49-63 are its own current PD targets; outputs 15-29 are per-joint stiffness (was 3 groups), 30/31 ankle/hip balance-reflex gain (0..2x), 32 target smoothing rate. All neutral at zero. Output layer initialised to **zero**, so an untrained net = standing pose + reflexes.
- Observation: 0-29 joint angle and speed*0.1 (interleaved); 30-35 sin/cos of pelvis, chest, head; 36-37 pelvis/chest angular velocity*0.1; 38 (COM x − support centre)/0.2; 39 COM vx/0.5; 40 capture-point offset/0.2; 41 (COM y − 0.95)*2; 42 support contact; 43-44 foot contact; 45-46 hand-on-ground; 47 pelvis-on-ground; 48 head-on-ground.
- Output: 15 PD target offsets (tanh × `ASC` per joint: neck .4, spines .5/.4, shoulder 1.2, elbow 1.2, wrist .5, hip 1.2, knee 1.5, ankle .6) + 3 stiffness values.
- Domain randomisation per episode: mass ±15%, per-part ±10%, friction 0.7-1.2, gain 0.9-1.1, action delay 0-3 policy steps, observation noise ≤0.01.

**Balance reflex** (fixed, always under the net, flag `opt.reflex`): only when upright (cos(chest)>0.8, head>1.35 m) and on the feet. Capture point `xi = comX + vx/W0`; `e = xi − (lo + 0.45*(hi−lo))` where lo/hi = smoothed foot support span (contact has hysteresis: on <0.02 m, off >0.045 m; support held ~0.12 s through flicker). Ankles: target += `clamp(-4e, ±0.5)`. Hips: target += `clamp(-4*(|e|-0.02)*sign(e), ±0.7)`.

**Extra reflexes, controlled by `RS.FLAGS`** (all default **off** in `sim.js`; the page's tick boxes and `train.js` turn them on; training episodes read them from `cfg.flags`):
- `step`: stepping reflex. Phases 0 stand, 1 swing, 2 planted, 3 bring leg back. Swing foot follows a smoothstep path to a landing spot tracking the capture point; **2-link leg IK** (thigh/shin 0.42 m) gives hip/knee/ankle targets. Params: `Ts` (swing time, 0.2), `land_off` (0), `trig` (0.12 m outside support), `clear` (0.07). Only starts after >0.5 s standing on both feet and outside the 0.8 s "quiet" window after a landing.
- `air`, `land`, `fall`: experimental (legs ready in the air; soft crouch on landing; limp + tuck chin + arms out when falling). `fall` only triggers on the way down (head>1.0, not while stepping, capture point >0.25 m outside support or chest tilt >0.55).
- User explicitly **wants these reflexes on** because "they make it look more human". Training and Test use whichever are ticked (`reflexCfg()` in `ui.js`).

**Balance v2 (`FLAGS.bal2`, page tick box "New balance + recovery steps", on by default; `train.js` env `BAL2`, on unless `false`):** replaces the old hip reflex and step reflex when on. Stance hips hold the pelvis upright in world frame (SIMBICON-style), leaning up to 0.3 rad into the fall except mid-swing; ankles keep the foot flat and push the capture point back. A step starts when the capture point is `trig2` (0.03 m) past toe/heel: the rear foot swings (2-link IK, `Ts2` 0.25 s, `clear2` 0.09) to the live capture point (`off2` 0, capped `max2` 0.45 m from the hip), then the other leg if needed; once settled the trailing foot is brought beside the front one. Gives up below 55 deg chest tilt / head 1.1 m, and the fall reflex only fires after that. With bal2 the reward's COM-drift penalty is measured from the feet while standing (recovery steps aren't punished). The page now runs the reflexes even when "Use policy" is unticked (zero weights).

## 6. Training environment, reward, ES

- **Scenarios** (`sampleScenario`): stand 20%, push 35% (1-3 pushes, random body incl. head/pelvis/thigh/arm, random direction, strength up to `pushMax`), drop 25% (random pose, tilt, height 0.05-0.85 m, random velocities), fallen 20% (supine/prone/crumpled start). Early stop when stably standing (extrapolates remaining reward) or lying motionless.
- **Reward per step (scaled to 60/POLICY_HZ):** `0.6*stand + prog − calm*penalties`. `stand = 0.3*up + 0.25*pose + 0.25*legs + 0.2*head-height`; `prog = 0.25*hn^2 + 0.15*cos(chest)` (dense shaping for getting up); penalties: leg split, action change, joint speed, COM velocity/drift, torque effort, all scaled by `calm` (reduced for 1.2 s after a disturbance). **While a reflex is active (`reactive`) posture/leg/jerk penalties are switched off**, otherwise the reward punishes the reflexes.
- **The reward is blind to "human-like"**: stepping/bracing earn nothing extra. Planned fix (not built): pay for foot landing under the capture point when falling, arms reaching toward the fall, and recovery.
- **ES** (`esNoise`, `esStep` in `sim.js`): antithetic sampling, centred ranks, Adam. Defaults `sigma=0.01, lr=0.004, pairs=16, scenarios=3`, curriculum `pushMax = min(1.5, 0.7 + 0.02*gen)`. All candidates in a generation see the same scenarios (common random numbers). `sigma=0.1` diverges; `0.05`/`0.01` stable but not improving much.
- Weights file (page "Save/Load file" and `train.js` output): `{"format":"ragdoll-policy","version":2,"np":4378,"gen":N,"theta":[...],"hist":[]}`. Version 3 = current layout. `RS.migrate` converts version-2 (4378-weight) files exactly, and the page's Load file and `train.js` resume use it. Any other change to NI/NH/NO needs a new migration.

- **Smooth/human reward (`FLAGS.smooth`, on by default in `train.js` and the page):** penalises command jerk (2nd difference of outputs) and joint acceleration (capped at 0.05 rad/s per step so impacts don't dominate). The penalty is 4x right after an upright start and 0.3x while a reflex is active. It pays for calm stillness, a level head and relaxed joints once nothing is happening. **`FLAGS.soft`** eases PD targets in over `softT`=0.6 s after an upright spawn. Measure twitch with `tools/t120.js`.

## 7. Measurements so far (so you don't repeat them)

All numbers are rewards where about 300 is the max for "stand".

- **Reflex-only baseline, all reflexes on** (GitHub run, 8 episodes/scenario): stand 313, push 307, drop 15, fallen 163, **total 797**. On other seeds: balance-only 282/247/50/166; all reflexes 282/258/16/161.
- **GitHub Actions (4 vCPU) ES run, 16 pairs × 3 scenarios:** 3.15 s/gen. Test totals: gen 440 → 852 (stand 299, push 291, drop 119, fallen 143); gen 660 → 802. Test noise is about ±40 per scenario, so only the **drop** gain looks real.
- Sandbox (1 core): ~4.5 s/gen. Output-layer-only ES (738 params) did not help either.
- **Push survival** (balance-only, impulse on chest at t=1 s, success = head >1.45 m and chest tilt <0.5 rad after 7 s; script `tools/t110.js`): current `sim.js` **0.75 fwd / 0.56 back**. `history/sim_v1..v3` (foot 0.30): ~1.1 / 0.69. An older rig version (before the multi-scenario rewrite, `LIMF` 0.5, foot 0.30, hand-coded ankle+hip reflex) survived **2.7 fwd / 1.9 back**. That regression is the user's "falls over super easily".
- Step-reflex sweep (`tools/t115.js`, solver iterations 8/3): best `trig 0.12, Ts 0.22, land_off 0` gave **2.13 fwd / 0.56 back**. Results are **chaotic**: neighbouring settings swing between 0.6 and 2.0, so any change must be judged on many seeds/pushes, not one trial. Backward pushes are the weak side (heel margin ~0.10 m).
- Reflex ablation (`tools/t95.js`): early versions of the reflexes misfired (landing reflex fired on the 2 cm settle at spawn; fall reflex stayed on after falling) and wrecked the baseline (stand 306 → 39 on the user's seeds). Fixed (real air time >0.15 s, fall only on the way down, quiet window, reward exemption). Still weak: **drop** with all reflexes together is worse than balance-only.
- **Get-up** (`getup/`): a 6-keyframe supine routine optimised by low-dimensional ES (`getup/getup_situp_solution.json`, ~1.7 s/gen) reliably **sits up** (head 0.84 m, stable, ~5 s). Stage 2 (sitting → standing; 5-6 more keyframes) never stood. A launch exploit appeared (body flung up then landed sitting) and was removed from the score. Suspect rig strength/spine range (forward spine flexion limited to ~0.9+0.4 rad), not just the search. Untried: widen spine/hip limits or torque cap and retest standing, prone variant, handing over to the network/reflexes after standing.

- **Balance v2 (2026-10-02):** chest-push survival (6 push timings, page loop) forward ~1.8 / back ~1.5 (old: ~0.7 / 0.5 balance-only; the old step reflex made 0.7-1.5 worse). Push-scenario episodes ending upright 14/40 -> 38/40, drop 4/40 -> 21/40. Reflex-only reward (16 eps): stand 343, push 292, drop 151, fallen 178 = 965 (bal2 alone); with air/land/fall too 935 (old all-reflex 691). Old notes on the cause: the rig weighs 1.8 kg, so push 1 = 0.56 m/s, past what feet-in-place can hold; the old hip reflex leaned away from the fall.

## 8. Prioritised to-do list

1. ~~Fix push robustness~~ (done with Balance v2, section 5/7; pushes above ~2 still fall). Start from `tools/t110.js`. Candidate levers: solver iterations 8/3, step-reflex `Ts 0.22`, longer/heavier heel for backward pushes (user wants small feet, so trade-offs must be explained), higher-gain ankle/hip reflex, torque cap. Evaluate on many seeds; keep the page's stand scenario at ≥300.
2. Make the reward value human-like reactions (step landing, arms bracing, limp recovery) so the net can learn them; retest.
3. Get-up: finish stand-up (see section 7), then wire it in as a "fallen" behaviour.
4. Network quality: ES is the weak link. Options: PPO off-browser (Node or Python), smaller policy, train only on drop/fallen, bigger populations on GitHub Actions.
5. Drop scenario with all reflexes together (currently worse than balance-only).
6. Show seconds/generation in the page (user asked about speed); optional.

## 9. How to run things

```
npm install planck@1.0.0
node tools/t61.js                 # baseline per-scenario scores (reflex only), args: [simPath] [N]
node tools/t95.js 12              # ablation: each reflex alone vs all
node tools/t110.js                # push-survival bisection (balance only / all reflexes)
node tools/t115.js                # step-reflex parameter sweep
node tools/t100.js name full 100 12 3 0.01 0.004   # headless ES trainer, env HN=16 for more held-out episodes
node tools/t60.js                 # page-level test with a worker_threads Worker polyfill (NW=0 for main-thread path)
node tools/t90.js                 # worker/main determinism check for reflex flags
python3 build_page.py             # rebuild ragdoll_trainer.html from sim.js + ui.js
python3 build_train.py            # rebuild train.js from sim.js
node getup/getup.js               # evaluate the hand-made get-up guess
```
Tools were written against absolute paths in my sandbox and patched to `__dirname`-relative ones; if one breaks, fix the path first. `tools/` are headless-only: **nothing here has been run on the user's phone except the published page**, which the user did run (3 Web Workers worked on their iPhone).

GitHub Actions: Actions tab → `train` → Run workflow (inputs: minutes ≤340, pairs, scenarios, sigma, lr, resume). Output artifact `ragdoll-policy` contains `best.json` (best held-out total), `policy.json` (latest), `log.txt` (TEST lines compare net vs baseline). The same files are also committed to the `results` branch under `runs/<run number>/` (cloud Claude sessions cannot download artifacts). Job limit 6 h; `train.js` caps minutes at 340. To resume, commit a previous `policy.json` as `resume.json` and set `resume=true`.

## 10. Gotchas learned the hard way

- **iOS cannot run web pages in the background**; the Artifact viewer pauses when closed. Long training must run elsewhere (GitHub Actions, or Claude Code cloud sessions: untested for compute/duration).
- A document-level `touchstart` `preventDefault` kills button clicks on iOS; only call it when `e.target === canvas`.
- Size the canvas from `clientWidth/clientHeight`, not `innerWidth/innerHeight` (safe-area padding made shapes look skewed).
- Planck `MouseJoint` was dropped; dragging uses a custom spring force on the grabbed point, scaled by the whole connected body's mass. Grab uses finger-sized tolerance (rings of 12/24/36 px).
- Published Artifact pages: CSP only allows scripts from cdnjs / jsdelivr npm / tailwind / jquery; Web Workers are created from a Blob built from `<script id="simsrc">` text; file save uses the `downloads` capability (`claude.use('downloads').save(...)`, must be declared at publish); file load uses `<input type=file>` + `FileReader`.
- Planck notes: `getInertia()` is about the body origin; `setMassData` with inflated `I` is how `INERTIA_X` works; `RevoluteJoint` accepts `referenceAngle`.
- Do not trust single-episode or single-push comparisons: chaotic dynamics and ±40 test noise.

## 11. How the user likes to work

- Concise and direct; dislikes over-explanation. Blunt feedback is welcome. Ready-to-use files, not "steps for you to do".
- Likes to **toggle** new behaviour (tick boxes) so they can compare before/after. When I changed balance behaviour without a toggle they said "go back to what you had before", so **keep reverts easy and changes switchable**.
- Wants honesty about whether something actually worked; I over-claimed once (said they could untick reflexes to read Test numbers; the flags did not apply). Report measured numbers and what was not verified.
- Often asks a question first ("don't fix yet, just tell me"); respect that.
- Phone-only: prefer solutions that run on GitHub Actions or in the Claude app, and give iPhone-friendly steps.
