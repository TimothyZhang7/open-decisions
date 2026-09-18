/* Pure continuous-time game mechanics. No model calls or move-selection policy. */
(function(root){
'use strict';
const ACTIONS=['left','right','clockwise','counterclockwise','none'];
class TetrisGame {
  constructor(orientations,{seed=42,gravityMs=500,lockDelayMs=500,scenario='empty'}={}){
    this.orientations=orientations;this.names=Object.keys(orientations);this.seed=seed;this.randomState=seed;
    this.board=Array.from({length:20},()=>Array(10).fill(0));this.queue=[];this.active=null;this.nextId=1;
    this.gravityMs=gravityMs;this.lockDelayMs=lockDelayMs;this.fallElapsed=0;this.groundedElapsed=0;
    this.elapsed=0;this.running=false;this.over=false;this.pieces=0;this.lines=0;this.score=0;this.lastClear=[];this.locks=[];
    if(scenario==='tetris')for(let y=16;y<20;y++)for(let x=0;x<9;x++)this.board[y][x]=(x+y)%7+1;
    if(scenario==='stack')[3,2,4,3,2,1,3,2,1,0].forEach((height,x)=>{for(let y=20-height;y<20;y++)this.board[y][x]=(x+y)%7+1;});
    this.refill();if(scenario==='tetris')this.queue.unshift('I');this.spawn();
  }
  random(){let t=this.randomState+=0x6D2B79F5;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;}
  refill(){while(this.queue.length<8){const bag=[...this.names];for(let i=bag.length-1;i>0;i--){const j=Math.floor(this.random()*(i+1));[bag[i],bag[j]]=[bag[j],bag[i]];}this.queue.push(...bag);}}
  shape(active=this.active){return this.orientations[active.piece][active.rotation];}
  cells(active=this.active){return this.shape(active).map(([x,y])=>[x+active.x,y+active.y]);}
  collision(active){return this.cells(active).some(([x,y])=>x<0||x>=10||y>=20||(y>=0&&this.board[y][x]!==0));}
  grounded(){return this.active&&!this.over&&this.collision({...this.active,y:this.active.y+1});}
  spawn(){this.refill();const piece=this.queue.shift();this.active={id:this.nextId++,piece,x:piece==='O'?4:3,y:-1,rotation:0};this.fallElapsed=0;this.groundedElapsed=0;if(this.collision(this.active)){this.over=true;this.running=false;}}
  action(action){
    if(!ACTIONS.includes(action))throw new Error('Choose one of the four movement actions or none.');
    if(!this.running||this.over||action==='none')return false;
    const before=this.active;
    if(action==='left'||action==='right'){
      const moved={...before,x:before.x+(action==='left'?-1:1)};
      if(this.collision(moved))return false;this.active=moved;return true;
    }
    if(before.piece==='O')return false;
    const rotation=(before.rotation+(action==='clockwise'?1:3))%4;
    for(const[dx,dy]of [[0,0],[-1,0],[1,0],[-2,0],[2,0],[0,-1],[0,-2]]){
      const moved={...before,rotation,x:before.x+dx,y:before.y+dy};
      if(!this.collision(moved)){this.active=moved;return true;}
    }
    return false;
  }
  lock(){
    const cells=this.cells();if(cells.some(([,y])=>y<0)){this.over=true;this.running=false;return;}
    for(const[x,y]of cells)this.board[y][x]=this.names.indexOf(this.active.piece)+1;
    const cleared=[];this.board.forEach((row,y)=>{if(row.every(Boolean))cleared.push(y);});
    const remaining=this.board.filter(row=>!row.every(Boolean));
    this.board=[...cleared.map(()=>Array(10).fill(0)),...remaining];
    this.score+=[0,100,300,500,800][cleared.length]*(1+Math.floor(this.lines/10));this.lines+=cleared.length;this.pieces++;
    this.lastClear=cleared;this.locks.unshift({piece:this.active.piece,lines:cleared.length});this.locks=this.locks.slice(0,16);this.spawn();
  }
  tick(ms,maxPieces=Infinity,maxLines=Infinity){
    if(!this.running||this.over||!Number.isFinite(ms)||ms<=0)return;
    // Small physics steps make locking independent of render frame rate. Inputs
    // never reset the accumulated ground-contact timer, preventing endless spins.
    while(ms>0&&this.running&&!this.over&&this.pieces<maxPieces&&this.lines<maxLines){
      const dt=Math.min(ms,16);ms-=dt;this.elapsed+=dt;
      if(this.grounded()){this.fallElapsed=0;this.groundedElapsed+=dt;if(this.groundedElapsed>=this.lockDelayMs)this.lock();}
      else{this.fallElapsed+=dt;if(this.fallElapsed>=this.gravityMs){this.fallElapsed-=this.gravityMs;this.active={...this.active,y:this.active.y+1};}}
    }
  }
  landing(){if(!this.active||this.over)return null;let result={...this.active};while(!this.collision({...result,y:result.y+1}))result={...result,y:result.y+1};return result;}
  snapshot(generation,requestId,recentActions=[]){return {board:this.board.map(row=>[...row]),active:{...this.active},next_piece:this.queue[0],generation,request_id:requestId,gravity_ms:this.gravityMs,recent_actions:recentActions.slice(-8)};}
}
if(typeof module!=='undefined'&&module.exports)module.exports={TetrisGame,ACTIONS};
root.TetrisGame=TetrisGame;root.TETRIS_ACTIONS=ACTIONS;
})(typeof globalThis!=='undefined'?globalThis:this);
