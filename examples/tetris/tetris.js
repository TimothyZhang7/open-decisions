'use strict';
const $=id=>document.getElementById(id),CELL=30;
const COLORS=['','#62d6eb','#f4d777','#bfa0ef','#7be8ba','#ef7f93','#7baaf0','#ffa775'];
const LABELS={left:'← Left',right:'Right →',clockwise:'↻ Clockwise',counterclockwise:'↺ Counterclockwise',none:'No input'};
function providerName(){return info?.backend==='typesafe'?'TypeSafe API':'Open Decisions';}
let target=null,game=null,info=null,seed=42,generation=0,requestId=0,pending=null,runUntil=0,previousTime=null,nextRequestAt=0;
let calls=0,applied=0,idle=0,blocked=0,discarded=0,errors=0,inputTokens=0,outputTokens=0,recentActions=[],lastLocked=0;
const canvas=$('board'),ctx=canvas.getContext('2d');ctx.scale(2,2);
function block(context,x,y,size,color,alpha=1){context.globalAlpha=alpha;context.fillStyle=color;context.fillRect(x+1,y+1,size-2,size-2);context.fillStyle='#ffffff35';context.fillRect(x+2,y+2,size-4,2);context.fillStyle='#00000020';context.fillRect(x+2,y+size-4,size-4,2);context.globalAlpha=1;}
function color(piece){return COLORS[game.names.indexOf(piece)+1];}
function draw(){
  ctx.clearRect(0,0,300,600);ctx.fillStyle='#080f18';ctx.fillRect(0,0,300,600);ctx.strokeStyle='#20304166';ctx.lineWidth=.5;
  for(let x=0;x<=10;x++){ctx.beginPath();ctx.moveTo(x*CELL,0);ctx.lineTo(x*CELL,600);ctx.stroke();}
  for(let y=0;y<=20;y++){ctx.beginPath();ctx.moveTo(0,y*CELL);ctx.lineTo(300,y*CELL);ctx.stroke();}
  if(!game)return;
  game.board.forEach((row,y)=>row.forEach((value,x)=>{if(value)block(ctx,x*CELL,y*CELL,CELL,COLORS[value]);}));
  if(game.over)return;
  if(target?.piece_id===game.active.id){ctx.strokeStyle='#f4d777';ctx.setLineDash([3,3]);for(const[x,y]of target.cells)ctx.strokeRect(x*CELL+2,y*CELL+2,26,26);ctx.setLineDash([]);}
  const landing=game.landing();ctx.strokeStyle=color(game.active.piece);ctx.lineWidth=1;
  for(const[x,y]of game.cells(landing)){if(y<0)continue;ctx.strokeRect(x*CELL+3,y*CELL+3,24,24);}
  const fraction=game.grounded()?0:Math.min(.95,game.fallElapsed/game.gravityMs);
  for(const[x,y]of game.cells())block(ctx,x*CELL,(y+fraction)*CELL,CELL,color(game.active.piece));
}
function drawQueue(){
  $('queue').replaceChildren();for(const piece of game.queue.slice(0,4)){const c=document.createElement('canvas');c.width=104;c.height=70;c.setAttribute('aria-label',piece+' piece');c.title=piece;const g=c.getContext('2d');g.scale(2,2);const cells=info.shapes[piece],w=Math.max(...cells.map(p=>p[0]))+1,h=Math.max(...cells.map(p=>p[1]))+1;for(const[x,y]of cells)block(g,(52-w*11)/2+x*11,(35-h*11)/2+y*11,11,color(piece));$('queue').append(c);}
}
function boardHoles(){let holes=0;for(let x=0;x<10;x++){let filled=false;for(let y=0;y<20;y++){if(game.board[y][x])filled=true;else if(filled)holes++;}}return holes;}
function update(){
  if(!game)return;
  $('score').textContent=game.score.toLocaleString();$('lines').textContent=game.lines;$('pieces').textContent=game.pieces;
  $('level').textContent=`LEVEL ${String(1+Math.floor(game.lines/10)).padStart(2,'0')}`;
  $('play').textContent=game.running?'Ⅱ Pause':'▶ Play';$('play').disabled=game.over||game.lines>=(info.goal_lines||10);
  $('calls').textContent=`${calls} REQUESTS`;$('holes').textContent=boardHoles();
  $('async-status').textContent=pending?'MODEL REQUEST IN FLIGHT':'MODEL READY';
  $('discarded').textContent=discarded;$('applied').textContent=applied;$('blocked').textContent=blocked;$('idle').textContent=idle;
  $('position').textContent=game.over?'—':`${game.active.piece} #${game.active.id} · x ${game.active.x} · y ${game.active.y}`;
  $('gravity-value').textContent=`${game.gravityMs} ms / row`;$('goal-progress').textContent=`${game.lines} / ${info.goal_lines||10} lines`;$('target').textContent=target?`Target: columns ${Math.min(...target.cells.map(c=>c[0]))+1}–${Math.max(...target.cells.map(c=>c[0]))+1} · ${target.outcome.lines} projected clears · ${target.outcome.holes} holes`:'Choosing a landing for this piece…';
  $('usage').textContent=`${inputTokens.toLocaleString()} input + ${outputTokens.toLocaleString()} ${info.backend==='typesafe'?'API-reported output tokens':'generated answer tokens'} · ${errors} inference errors.`;
  for(const button of document.querySelectorAll('[data-action]'))button.disabled=!game.running||game.over;
  drawQueue();draw();
}
function overlay(title,subtitle){$('overlay').hidden=false;$('overlay').replaceChildren();const b=document.createElement('b'),s=document.createElement('span');b.textContent=title;s.textContent=subtitle;$('overlay').append(b,s);}
function pause(message='PAUSED'){
  if(!game)return;game.running=false;generation++;previousTime=null;
  // Let the old request occupy its slot until it settles, preventing a queue.
  // Its generation can no longer apply input after a pause or reset.
  $('game-status').textContent=message;update();
}
function newGame(){
  generation++;target=null;game=new TetrisGame(info.orientations,{seed,gravityMs:Number($('speed').value),scenario:$('scenario').value});
  previousTime=null;runUntil=0;nextRequestAt=0;calls=0;applied=0;idle=0;blocked=0;discarded=0;errors=0;inputTokens=0;outputTokens=0;recentActions=[];lastLocked=0;
  $('error').textContent='';$('decision').textContent='Waiting for the first decision.';$('detail').textContent='Gravity runs independently of inference.';
  $('latency').textContent='—';$('confidence').textContent='—';$('raw').textContent='No request yet.';$('candidates').replaceChildren();$('history').replaceChildren();
  $('seed-label').textContent=`SEED ${seed}`;$('game-status').textContent='READY';$('budget').textContent='Each decision sends one input or none while gravity continues.';
  overlay('Ready when you are.','Press Play to start gravity and the agent.');update();
}
function recordAction(action,changed){
  recentActions.push(action);recentActions=recentActions.slice(-8);if(action==='none')idle++;else{applied++;if(!changed)blocked++;}
  const item=document.createElement('span');item.textContent=LABELS[action]+(changed||action==='none'?'':' · blocked');$('history').prepend(item);
  while($('history').children.length>12)$('history').lastElementChild.remove();
}
function showDecision(result,changed){
  $('decision').textContent=LABELS[result.action];$('detail').textContent=`${result.action==='none'?'No input sent; gravity continues':changed?'Applied one button':'Button had no effect'} · snapshot row ${result.snapshot_y} → live row ${game.active.y}`;
  $('latency').textContent=`${Math.round(result.latency_ms)} ms`;$('confidence').textContent=result.confidence.toFixed(2);
  $('connection').textContent=`${result.model} / ${providerName()}`;$('raw').textContent=JSON.stringify({...result,disposition:'applied'},null,2);
  $('candidates').replaceChildren();for(const action of TETRIS_ACTIONS){const row=document.createElement('div');row.className='candidate';const label=document.createElement('span'),track=document.createElement('div'),bar=document.createElement('div'),prob=document.createElement('span');label.textContent=LABELS[action];track.className='track';bar.className='bar';bar.style.width=`${result.probabilities[action]*100}%`;if(action===result.action)bar.style.background='var(--mint)';prob.textContent=`${Math.round(result.probabilities[action]*100)}%`;track.append(bar);row.append(label,track,prob);$('candidates').append(row);}
}
async function askModel(now=performance.now()){
  if(!game?.running||game.over||pending||now<nextRequestAt)return;
  const snapshot={...game.snapshot(generation,++requestId,recentActions),target},flight={generation,requestId,started:now,game};pending=flight;calls++;update();
  const abort=new AbortController(),timeout=setTimeout(()=>abort.abort(),25000);
  try{
    const response=await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(snapshot),signal:abort.signal});const result=await response.json();
    if(!response.ok)throw new Error(result.error||'Model action failed.');
    if(!TETRIS_ACTIONS.includes(result.action)||result.request_id!==snapshot.request_id||result.generation!==snapshot.generation||result.piece_id!==snapshot.active.id)throw new Error('Invalid action response; no input applied.');
    if(flight.game===game){inputTokens+=result.usage.input_tokens;outputTokens+=result.usage.output_tokens;}
    if(flight.generation!==generation||!game.running||game.over||game.active.id!==result.piece_id){if(flight.game===game)discarded++;return;}
    target=result.target||null;const changed=game.action(result.action);recordAction(result.action,changed);showDecision(result,changed);$('error').textContent='';if(result.action==='none'&&result.steps_to_target===0)nextRequestAt=performance.now()+Math.min(game.gravityMs,500);
  }catch(error){
    if(flight.generation===generation){errors++;$('error').textContent=(error.name==='AbortError'?'Model request timed out.':error.message)+' Gravity keeps running.';nextRequestAt=performance.now()+1000;}
  }finally{
    clearTimeout(timeout);if(pending===flight)pending=null;
    nextRequestAt=Math.max(nextRequestAt,performance.now()+120);update();
  }
}
function frame(now){
  if(game?.running){
    if(previousTime!==null)game.tick(now-previousTime,runUntil,info.goal_lines||10);previousTime=now;
    if(game.pieces!==lastLocked){lastLocked=game.pieces;recentActions=[];target=null;update();}
    if(game.lines>=(info.goal_lines||10)){pause('CHALLENGE WON');overlay('Challenge won!',`${game.lines} lines · ${game.score} points · ${game.pieces} pieces.`);}
    else if(game.over){pause('GAME OVER');overlay('No room left.',`${game.lines} lines · ${game.pieces} pieces.`);}
    else if(game.pieces>=runUntil){pause('RUN COMPLETE');$('budget').textContent='Run complete. Press Play to continue.';}
    else{$('game-status').textContent='GRAVITY RUNNING';$('position').textContent=`${game.active.piece} #${game.active.id} · x ${game.active.x} · y ${game.active.y}`;askModel(now);}
  }else previousTime=null;
  draw();requestAnimationFrame(frame);
}
$('play').onclick=()=>{if(!game||game.over||game.lines>=(info.goal_lines||10))return;if(game.running){pause();return;}generation++;game.running=true;runUntil=game.pieces+Number($('limit').value);previousTime=null;$('overlay').hidden=true;$('budget').textContent=`Stops after ${runUntil-game.pieces} more locked pieces. Gravity continues during every request.`;update();askModel();};
$('reset').onclick=()=>{if(info){seed++;newGame();}};
$('speed').onchange=()=>{if(game){game.gravityMs=Number($('speed').value);update();}};
for(const button of document.querySelectorAll('[data-action]'))button.onclick=()=>{if(!game?.running)return;generation++;target=null;const action=button.dataset.action,changed=game.action(action);recordAction(action,changed);update();};
document.addEventListener('visibilitychange',()=>{if(document.hidden&&game?.running)pause('PAUSED WHILE AWAY');});
async function init(){
  try{const response=await fetch('/api/info');if(!response.ok)throw new Error('Cannot connect to the game server.');info=await response.json();$('connection').textContent=`${info.model} / ${providerName()}`;$('backend-note').textContent=info.backend==='typesafe'?'Hosted Jev · typed decisions':'Open Decisions · 0 generated tokens';$('model-tag').textContent=info.backend==='typesafe'?'HOSTED JEV':'LOCAL MODEL';newGame();requestAnimationFrame(frame);}
  catch(error){$('connection').textContent='Server unavailable';$('error').textContent=error.message;}
}
init();
