// Ragdoll trainer for GitHub Actions (Node 18+, needs: npm install planck@1.0.0)
// Same simulation, reflexes and evolution strategy as the Ragdoll Trainer page, but uses every CPU core.
// Writes out/policy.json (latest), out/best.json (best held-out score) and out/log.txt.
// The .json files load in the trainer page with "Load file".
'use strict';
const planck=require('planck');
const {Worker,isMainThread,parentPort}=require('worker_threads');
const os=require('os'),fs=require('fs');

