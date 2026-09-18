// Record the existing controller and game against real inference, at wall-clock speed.
// No move substitution, accelerated physics, or retries to select a better run.
'use strict';
const fs = require('node:fs'), vm = require('node:vm'), path = require('node:path');
const {TetrisGame, ACTIONS} = require('./tetris_game.js');
const args = Object.fromEntries(process.argv.slice(2).map(arg => arg.replace(/^--/, '').split('=')));
const port = Number(args.port || 8773), seed = Number(args.seed || 42);
const gravity = Number(args.gravity || 500), seconds = Number(args.seconds || 60);
const scenario = args.scenario || 'empty';
const output = path.resolve(args.output || 'tetris-demo.json');
if (![port,seed,gravity,seconds].every(Number.isFinite) || seconds <= 0 || seconds > 300 || gravity < 50 ||
    !['empty','stack','tetris'].includes(scenario)) throw new Error('Invalid recording parameters');

const nodes = new Map(), trace = [], frames = [], timers = new Set();
let stopped = false, started = 0, lastCapture = -Infinity, serverInfo;
const paint = new Proxy({}, {get: () => () => {}, set: () => true});
function element() { return {textContent:'',value:'',hidden:false,disabled:false,style:{},dataset:{},children:[],
  getContext:()=>paint,setAttribute(){},append(...n){this.children.push(...n);},
  prepend(...n){this.children.unshift(...n);},replaceChildren(...n){this.children=n;},
  get lastElementChild(){return {remove:()=>this.children.pop()};}}; }
const $ = id => {if (!nodes.has(id)) nodes.set(id,element());return nodes.get(id);};
$('speed').value=String(gravity);$('limit').value='50';$('scenario').value=scenario;
const buttons=ACTIONS.filter(a=>a!=='none').map(a=>Object.assign(element(),{dataset:{action:a}}));
const snapshotCode = `JSON.stringify({board:game.board,active:game.active,cells:game.cells(),
  ghost:game.over?[]:game.cells(game.landing()),queue:game.queue.slice(0,4),names:game.names,
  pieces:game.pieces,lines:game.lines,score:game.score,over:game.over,running:game.running,
  fallElapsed:game.fallElapsed,grounded:game.grounded(),target,pending:!!pending,calls,applied,
  idle,blocked,discarded,errors,inputTokens,outputTokens})`;
function capture(force=false) {
  const elapsed=performance.now()-started;
  if (!started || (!force && elapsed-lastCapture<100)) return;
  lastCapture=elapsed;
  frames.push({at_ms:elapsed,...JSON.parse(vm.runInContext(snapshotCode,context)),
    decision:$('decision').textContent,detail:$('detail').textContent,
    latency:$('latency').textContent,status:$('game-status').textContent,error:$('error').textContent});
}
const context = vm.createContext({
  document:{getElementById:$,createElement:element,querySelectorAll:()=>buttons,addEventListener(){},hidden:false},
  TetrisGame,TETRIS_ACTIONS:ACTIONS,AbortController,performance,console,setTimeout,clearTimeout,
  requestAnimationFrame:callback=>{if(stopped)return;const h=setTimeout(()=>{
    timers.delete(h);callback(performance.now());if(started)capture();},16);timers.add(h);},
  fetch:async(route,options)=>{
    const begin=performance.now();
    const response=await fetch(`http://127.0.0.1:${port}${route}`,options);
    const result=await response.json();
    if(options)trace.push({at_ms:begin-started,duration_ms:performance.now()-begin,
      snapshot:JSON.parse(options.body),http_status:response.status,result});
    else serverInfo=result;
    return {ok:response.ok,json:async()=>result};
  }
});
vm.runInContext(fs.readFileSync(path.join(__dirname,'tetris.js'),'utf8'),context);
vm.runInContext(`seed=${seed}`,context);
const startup=setTimeout(()=>{console.error('Tetris service did not become ready');process.exit(1);},30000);
let paused=false,finishedAt=0;
const poll=setInterval(()=>{
  if(!vm.runInContext('!!game',context))return;
  if(!started){clearTimeout(startup);started=performance.now();$('play').onclick();capture(true);
    console.log(JSON.stringify({recording:true,seed,gravity_ms:gravity,seconds,scenario}));return;}
  const elapsed=performance.now()-started;
  const running=vm.runInContext('game.running',context);
  if(!paused && (!running || elapsed>=seconds*1000)){
    capture(true);finishedAt=elapsed;
    if(running)$('play').onclick();paused=true;
  }
  if(!paused || (vm.runInContext('!!pending',context) && elapsed<finishedAt+27000))return;
  stopped=true;clearInterval(poll);for(const timer of timers)clearTimeout(timer);
  const state=JSON.parse(vm.runInContext(snapshotCode,context));
  const latencies=trace.filter(x=>x.http_status===200).map(x=>x.result.latency_ms).sort((a,b)=>a-b);
  const summary={seed,scenario,gravity_ms:gravity,elapsed_seconds:finishedAt/1000,
    recorded_frames:frames.length,goal_lines:10,won:state.lines>=10,
    pieces:state.pieces,lines:state.lines,score:state.score,game_over:state.over,
    requests:state.calls,model_evaluations:trace.reduce((n,x)=>n+(x.result.api_requests||0),0),
    applied:state.applied,idle:state.idle,blocked:state.blocked,discarded:state.discarded,
    errors:state.errors,reported_output_tokens:state.outputTokens,
    output_token_note:'Qwen generated zero answer tokens. Jev output_tokens is API-reported usage; it is not interpreted as autoregressive text generation.',
    decision_latency_ms:latencies.length?{min:latencies[0],median:(latencies[Math.floor((latencies.length-1)/2)]+latencies[Math.floor(latencies.length/2)])/2,max:latencies.at(-1)}:null};
  fs.mkdirSync(path.dirname(output),{recursive:true});
  fs.writeFileSync(output,JSON.stringify({recorded_at:new Date().toISOString(),
    recording:'Actual controller and game at wall-clock speed; frames sampled every 100 ms; no manual moves.',
    server:serverInfo,summary,frames,trace})+'\n');
  console.log(JSON.stringify(summary,null,2));process.exit(state.errors?1:0);
},100);
const progress=setInterval(()=>{if(started&&!stopped)console.log(JSON.stringify({seconds:Math.round((performance.now()-started)/1000),
  pieces:$('pieces').textContent,lines:$('lines').textContent,requests:trace.length,error:$('error').textContent}));},15000);
progress.unref();
