// Run the actual browser controller and physics against the live API, with a
// minimal DOM in Node. Real wall clock, real model latency, no accelerated drops.
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const {TetrisGame,ACTIONS}=require('./tetris_game.js');
const params=Object.fromEntries(process.argv.slice(2).map(arg=>arg.replace(/^--/,'').split('=')));
const port=Number(params.port||8773),seed=Number(params.seed||42),gravity=Number(params.gravity||500),seconds=Number(params.seconds||120);
const output=params.output||`results-persistent-target-seed${seed}.json`;
const nodes=new Map(),trace=[],frames=new Set();let stopped=false,started=performance.now(),requests=0;
const paint=new Proxy({}, {get:()=>()=>{},set:()=>true});
function element(){return {textContent:'',value:'',hidden:false,disabled:false,style:{},dataset:{},children:[],getContext:()=>paint,setAttribute(){},append(...nodes){this.children.push(...nodes);},prepend(...nodes){this.children.unshift(...nodes);},replaceChildren(...nodes){this.children=nodes;},get lastElementChild(){return{remove:()=>this.children.pop()};}};}
const $=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);};
$('speed').value=String(gravity);$('limit').value='50';$('scenario').value='empty';
const buttons=ACTIONS.filter(a=>a!=='none').map(a=>Object.assign(element(),{dataset:{action:a}}));
const context=vm.createContext({document:{getElementById:$,createElement:element,querySelectorAll:()=>buttons,addEventListener(){},hidden:false},TetrisGame,TETRIS_ACTIONS:ACTIONS,AbortController,performance,console,setTimeout,clearTimeout,
 requestAnimationFrame:callback=>{if(stopped)return;const handle=setTimeout(()=>{frames.delete(handle);callback(performance.now());},16);frames.add(handle);},
 fetch:async(route,options)=>{const begin=performance.now();const response=await fetch(`http://127.0.0.1:${port}${route}`,options);const result=await response.json();if(options){requests++;trace.push({at_ms:begin-started,duration_ms:performance.now()-begin,snapshot:JSON.parse(options.body),http_status:response.status,result});}return {ok:response.ok,json:async()=>result};}
});
vm.runInContext(fs.readFileSync(path.join(__dirname,'tetris.js'),'utf8'),context);vm.runInContext(`seed=${seed}`,context);
const poll=setInterval(()=>{
 if(!$('play').onclick||$('connection').textContent==='Server unavailable')return;
 const exists=vm.runInContext('!!game',context);if(!exists)return;
 if(!context.didStart){context.didStart=true;started=performance.now();$('play').onclick();console.log(JSON.stringify({started:true,seed,gravity_ms:gravity,goal_lines:10}));return;}
 const state=JSON.parse(vm.runInContext('JSON.stringify({board:game.board,active:game.active,pieces:game.pieces,lines:game.lines,score:game.score,over:game.over,running:game.running,pending:!!pending,calls,applied,idle,blocked,discarded,errors,inputTokens,outputTokens})',context));
 const elapsed=(performance.now()-started)/1000;
 if(state.running&&elapsed<seconds)return;
 if(state.running){$('play').onclick();state.running=false;}
 if(state.pending&&elapsed<seconds+26)return;
 stopped=true;clearInterval(poll);for(const frame of frames)clearTimeout(frame);
 const summary={seed,gravity_ms:gravity,elapsed_seconds:elapsed,goal_lines:10,won:state.lines>=10,...state,status:$('game-status').textContent,requests};
 fs.writeFileSync(output,JSON.stringify({summary,trace},null,2)+'\n');console.log(JSON.stringify(summary));process.exit(0);
},100);
const progress=setInterval(()=>{if(context.didStart&&!stopped)console.log(JSON.stringify({seconds:Math.round((performance.now()-started)/1000),pieces:$('pieces').textContent,lines:$('lines').textContent,goal:$('goal-progress').textContent,error:$('error').textContent,requests}));},15000);progress.unref();
