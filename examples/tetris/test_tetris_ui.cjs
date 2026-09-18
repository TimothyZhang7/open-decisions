// Real physics + real UI controller, with a fake clock and deferred network.
const test=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm'),{execFileSync}=require('node:child_process');
const {TetrisGame,ACTIONS}=require('./tetris_game.js');
const {orientations,shapes}=JSON.parse(execFileSync(process.env.PYTHON||`${__dirname}/../../.venv/bin/python`,['-c','import json; from tetris_realtime import ORIENTATIONS; from tetris_engine import SHAPES; print(json.dumps(dict(orientations=ORIENTATIONS,shapes=SHAPES)))'],{cwd:__dirname}));
const source=fs.readFileSync(`${__dirname}/tetris.js`,'utf8');
const flush=()=>new Promise(resolve=>setImmediate(resolve));
function setup(){
  const elements=new Map(),pending=[],frames=[],timers=new Map();let now=0,nextTimer=0;
  const paint=new Proxy({}, {get:()=>()=>{},set:()=>true});
  function node(){return {textContent:'',value:'',hidden:false,disabled:false,style:{},dataset:{},children:[],getContext:()=>paint,setAttribute(){},append(...nodes){this.children.push(...nodes);},prepend(...nodes){this.children.unshift(...nodes);},replaceChildren(...nodes){this.children=nodes;},get lastElementChild(){return {remove:()=>this.children.pop()};}};}
  const $=id=>{if(!elements.has(id))elements.set(id,node());return elements.get(id);};
  $('speed').value='500';$('limit').value='10';$('scenario').value='empty';
  const buttons=ACTIONS.filter(action=>action!=='none').map(action=>Object.assign(node(),{dataset:{action}}));
  const document={getElementById:$,createElement:node,querySelectorAll:()=>buttons,addEventListener(){},hidden:false};
  const context=vm.createContext({document,TetrisGame,TETRIS_ACTIONS:ACTIONS,AbortController,performance:{now:()=>now},console,
    setTimeout:(callback,delay)=>{const id=++nextTimer;timers.set(id,{callback,at:now+delay});return id;},clearTimeout:id=>timers.delete(id),
    requestAnimationFrame:callback=>frames.push(callback),
    fetch:(path,options)=>path==='/api/info'?Promise.resolve({ok:true,json:async()=>({model:'qwen-test',orientations,shapes})}):new Promise((resolve,reject)=>{
      pending.push({resolve,reject,body:JSON.parse(options.body)});
      options.signal.addEventListener('abort',()=>{const error=new Error('aborted');error.name='AbortError';reject(error);});
    }),
  });
  vm.runInContext(source,context);
  return {context,$,pending,buttons,
    state:()=>JSON.parse(vm.runInContext('JSON.stringify({active:game.active,board:game.board,pieces:game.pieces,lines:game.lines,running:game.running,calls,applied,idle,blocked,discarded,errors,pending:!!pending})',context)),
    advance(ms){const target=now+ms;while(now<target){now=Math.min(target,now+16);for(const[id,timer]of [...timers])if(timer.at<=now){timers.delete(id);timer.callback();}const batch=frames.splice(0);for(const callback of batch)callback(now);}},
  };
}
function resolveAction(request,{action='left',ok=true,error,target=null}={}){
  request.resolve({ok,json:async()=>error?{error}:{action,piece_id:request.body.active.id,generation:request.body.generation,request_id:request.body.request_id,probabilities:Object.fromEntries(ACTIONS.map(key=>[key,Number(key===action)])),confidence:1,latency_ms:100,model:'qwen-test',usage:{input_tokens:100,output_tokens:0},snapshot_y:request.body.active.y,target}});
}
async function playing(){const ui=setup();await flush();ui.$('play').onclick();ui.advance(16);return ui;}

test('physics permits four single inputs, gravity alone drops and locks',()=>{
  const game=new TetrisGame(orientations);game.running=true;const before={...game.active};
  assert.equal(game.action('left'),true);assert.equal(game.active.x,before.x-1);assert.equal(game.active.y,before.y);
  for(const illegal of ['drop','wait','hold',['left','right']])assert.throws(()=>game.action(illegal));
  game.tick(20000,1);assert.equal(game.pieces,1);assert.equal(game.board.flat().filter(Boolean).length,4);
});
test('four line clear requires rotation and repeated one-column inputs',()=>{
  const game=new TetrisGame(orientations,{scenario:'tetris'});game.running=true;
  game.action('clockwise');for(let n=0;n<4;n++)game.action('right');
  assert.equal(game.active.x,7);assert.equal(game.active.rotation,1);assert.equal(game.pieces,0);
  game.tick(20000,1);assert.equal(game.pieces,1);assert.equal(game.lines,4);assert.equal(game.score,800);assert.ok(game.board.flat().every(cell=>cell===0));
});
test('rotating on ground cannot reset lock delay',()=>{
  const game=new TetrisGame(orientations);game.running=true;game.active={id:1,piece:'T',x:3,y:18,rotation:0};
  for(let i=0;i<50&&game.pieces===0;i++){game.action(i%2?'clockwise':'counterclockwise');game.tick(100,1);}
  assert.equal(game.pieces,1);
});
test('no action preserves piece and timers, so gravity and locking continue',()=>{
  const game=new TetrisGame(orientations);game.running=true;game.tick(200);
  const active={...game.active},fall=game.fallElapsed;
  assert.equal(game.action('none'),false);assert.deepEqual(game.active,active);assert.equal(game.fallElapsed,fall);
  game.tick(400);assert.equal(game.active.y,active.y+1);
  game.active={id:1,piece:'T',x:3,y:18,rotation:0};game.groundedElapsed=400;
  game.action('none');game.tick(116,1);assert.equal(game.pieces,1);
});
test('no-action response is shown and counted separately from blocked buttons',async()=>{
  const ui=await playing(),before=ui.state().active;
  resolveAction(ui.pending[0],{action:'none'});await flush();
  assert.deepEqual(ui.state().active,before);assert.equal(ui.state().idle,1);assert.equal(ui.state().applied,0);assert.equal(ui.state().blocked,0);
  assert.equal(ui.$('decision').textContent,'No input');assert.match(ui.$('detail').textContent,/gravity continues/);
  ui.advance(1100);assert.ok(ui.state().active.y>before.y);assert.equal(ui.state().running,true);
});
test('gravity runs during inference; response presses one button at live row',async()=>{
  const ui=await playing(),start=ui.state();ui.advance(2200);
  assert.equal(ui.pending.length,1);assert.equal(ui.state().active.y,start.active.y+4);
  const live=ui.state().active;resolveAction(ui.pending[0]);await flush();
  assert.equal(ui.state().active.x,live.x-1);assert.equal(ui.state().active.y,live.y);assert.equal(ui.state().applied,1);
});
test('gravity locks while inference is pending; old response cannot move next piece',async()=>{
  const ui=await playing();ui.advance(11000);assert.ok(ui.state().pieces>=1);const active=ui.state().active;
  resolveAction(ui.pending[0]);await flush();assert.deepEqual(ui.state().active,active);assert.equal(ui.state().discarded,1);assert.equal(ui.state().applied,0);
});
test('pause and resume preserve one pending request and invalidate old input',async()=>{
  const ui=await playing();ui.advance(1000);ui.$('play').onclick();const paused=ui.state().active;
  ui.advance(2000);assert.deepEqual(ui.state().active,paused);ui.$('play').onclick();assert.equal(ui.pending.length,1);
  resolveAction(ui.pending[0]);await flush();assert.equal(ui.state().applied,0);ui.advance(200);assert.equal(ui.pending.length,2);
});
test('reset invalidates pending response and prevents request queue',async()=>{
  const ui=await playing();ui.$('reset').onclick();const reset=ui.state().active;ui.$('play').onclick();assert.equal(ui.pending.length,1);
  resolveAction(ui.pending[0]);await flush();assert.deepEqual(ui.state().active,reset);assert.equal(ui.state().applied,0);
});
test('API failures and request timeouts leave gravity running',async()=>{
  const ui=await playing();resolveAction(ui.pending[0],{ok:false,error:'Unavailable'});await flush();const row=ui.state().active.y;
  ui.advance(1000);assert.equal(ui.state().running,true);assert.ok(ui.state().active.y>row);assert.match(ui.$('error').textContent,/Gravity keeps running/);
  ui.advance(25050);await flush();assert.equal(ui.state().errors,2);assert.equal(ui.state().running,true);assert.ok(ui.state().pieces>=2);
});
test('piece budget stops physics even after a delayed render frame',async()=>{
  const ui=await playing();ui.$('play').onclick();ui.$('limit').value='1';ui.$('play').onclick();
  ui.advance(30000);assert.equal(ui.state().pieces,1);assert.equal(ui.state().running,false);assert.equal(ui.$('game-status').textContent,'RUN COMPLETE');
});

test('a model-selected target travels with subsequent requests and is cleared on reset',async()=>{
  const ui=await playing();
  const target={piece_id:ui.pending[0].body.active.id,cells:[[0,18],[1,18],[1,19],[2,19]],outcome:{lines:0,holes:0}};
  resolveAction(ui.pending[0],{target});await flush();ui.advance(200);
  assert.deepEqual(ui.pending[1].body.target,target);
  ui.$('reset').onclick();assert.equal(vm.runInContext('target',ui.context),null);
});
test('reaching ten lines wins and prevents further model requests',async()=>{
  const ui=await playing();vm.runInContext('game.lines=10',ui.context);ui.advance(16);
  assert.equal(ui.state().running,false);assert.equal(ui.$('game-status').textContent,'CHALLENGE WON');
  assert.equal(ui.$('play').disabled,true);const count=ui.pending.length;ui.advance(1000);assert.equal(ui.pending.length,count);
});

test('the physics stops at the line goal immediately after a real clear',()=>{
  const game=new TetrisGame(orientations,{scenario:'tetris'});game.running=true;game.lines=9;
  game.action('clockwise');for(let n=0;n<4;n++)game.action('right');
  game.tick(20000,50,10);assert.equal(game.lines,13);assert.equal(game.pieces,1);assert.equal(game.active.y,-1);
});
