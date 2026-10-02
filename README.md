# Equilibria

A 2D, side-on **active ragdoll** that balances, takes pushes and tries to stay upright using joint torques only, with no puppet strings or fake forces.

It is inspired by Euphoria, the procedural character-physics technology used in Rockstar games like GTA IV and Red Dead Redemption. This is an independent, from-scratch experiment. It is **not affiliated with or endorsed by Rockstar, NaturalMotion or Take-Two**, and it uses none of their code.

## What it does
- Box-based rig on [Planck.js](https://github.com/piqnt/planck) (a Box2D port), run at 240 Hz.
- PD controller on every joint, with gains derived from body inertia so it stays stable.
- Hand-written reflexes: ankle/hip balance, a stepping reflex, and experimental fall and landing reflexes.
- A small neural network that adds a learned correction, trained with evolution strategies on random pushes, drops and falls.
- A browser trainer, plus `train.js` for multi-core training on GitHub Actions.

## Status
Experimental. It stands, recovers from moderate pushes and sits up from lying down. The network does not yet clearly beat the reflexes alone, it falls over under hard pushes, and getting fully back to standing is unsolved. See `HANDOFF.md` for measurements and the to-do list.

Training rewards smooth, human-like motion (no twitching, calm stillness) and the ragdoll eases into standing after spawning. Both are switchable: tick boxes on the page, `smooth`/`soft` inputs on the GitHub Actions workflow.

## Run it
