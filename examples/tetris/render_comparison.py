"""Render a synchronized replay of two real, separately recorded Tetris runs."""
import argparse
import bisect
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[2]/'docs/media')
parser.add_argument('--output-dir',type=Path,default=Path('tetris-video'))
args=parser.parse_args()
OUT=args.output_dir
OUT.mkdir(parents=True,exist_ok=True)
RUNS=[json.loads((args.data_dir/f'tetris-{n}.json').read_text()) for n in ['qwen','jev']]
W,H,FPS=1600,1100,20
BG='#0D141D'; INK='#EDF3FA'; MUTED='#A0B0C1'; LINE='#2A3A4B'; GREEN='#7BE8BA'
COLORS=['','#62d6eb','#f4d777','#bfa0ef','#7be8ba','#ef7f93','#7baaf0','#ffa775']
LABELS={'left':'Left','right':'Right','clockwise':'Clockwise','counterclockwise':'Counterclockwise','none':'No input'}
TIMES=[[f['at_ms'] for f in run['frames']] for run in RUNS]
for run in RUNS:
    assert (run['summary']['seed'],run['summary']['gravity_ms'],run['summary']['scenario'])==(42,500,'empty')
assert RUNS[0]['server']['orientations']==RUNS[1]['server']['orientations']

@lru_cache(None)
def font(size,bold=False):
    return ImageFont.truetype('/System/Library/Fonts/Supplemental/'+('Arial Bold.ttf' if bold else 'Arial.ttf'),size)
def text(d,x,y,s,size=25,color=INK,bold=False):
    d.text((x,y),str(s),font=font(size,bold),fill=color)
def wrap(d,x,y,s,width,size=24,color=MUTED):
    line=''
    for word in s.split():
        candidate=(line+' '+word).strip()
        if line and d.textlength(candidate,font=font(size))>width:
            text(d,x,y,line,size,color);y+=size+8;line=word
        else: line=candidate
    if line:text(d,x,y,line,size,color)

def frame_at(i,ms):
    return RUNS[i]['frames'][max(0,bisect.bisect_right(TIMES[i],ms)-1)]
def result_at(i,ms):
    ready=[t for t in RUNS[i]['trace'] if t['at_ms']+t['duration_ms']<=ms and t['http_status']==200]
    return ready[-1]['result'] if ready else None

def render(ms,end=False):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    text(d,48,31,'Tetris: Qwen vs. Jev',48,bold=True)
    text(d,48,97,'Seed 42  ·  Empty board  ·  500 ms gravity  ·  Recorded playback at 1×',26,MUTED)
    text(d,1375,46,f'{min(ms/1000,60):04.1f} / 60 s',26,GREEN)
    d.line((800,166,800,987),fill=LINE,width=2)
    for i,offset in enumerate([48,848]):
        f=frame_at(i,ms);run=RUNS[i];summary=run['summary'];res=result_at(i,ms)
        title='Open Decisions + Qwen3-VL-4B' if i==0 else 'Jev 1.13.0'
        text(d,offset,160,title,30,bold=True)
        subtitle='Local MLX 4-bit · M4 Pro · 0 generated tokens' if i==0 else 'Hosted TypeSafe API · network time included'
        text(d,offset,207,subtitle,23,MUTED)
        bx,by,cell=offset,262,29
        d.rectangle((bx-1,by-1,bx+290+1,by+580+1),fill='#080F18',outline='#456070',width=2)
        for x in range(11):d.line((bx+x*cell,by,bx+x*cell,by+580),fill='#182534')
        for y in range(21):d.line((bx,by+y*cell,bx+290,by+y*cell),fill='#182534')
        def block(x,y,c,outline=False):
            if not (0<=x<10 and 0<=y<20):return
            box=(bx+x*cell+1,by+y*cell+1,bx+(x+1)*cell-1,by+(y+1)*cell-1)
            if outline:d.rectangle(box,outline=c,width=2)
            else:
                d.rectangle(box,fill=c)
                d.line((box[0]+2,box[1]+2,box[2]-2,box[1]+2),fill='#FFFFFF',width=1)
        for y,row in enumerate(f['board']):
            for x,value in enumerate(row):
                if value:block(x,y,COLORS[value])
        for x,y in f['ghost']:block(x,y,'#445966',True)
        if f['target'] and f['target']['piece_id']==f['active']['id']:
            for x,y in f['target']['cells']:block(x,y,'#E7C565',True)
        if not f['over']:
            col=COLORS[f['names'].index(f['active']['piece'])+1]
            for x,y in f['cells']:block(x,y,col)
        sx=offset+322
        text(d,sx,267,'Latest decision',23,MUTED)
        action=LABELS.get(res['action'],'Waiting') if res else 'Waiting'
        text(d,sx,306,action,31,GREEN,True)
        text(d,sx,357,f"{res['latency_ms']:.0f} ms" if res else '—',30)
        text(d,sx,400,'Controller request time',21,MUTED)
        text(d,sx,449,'Request in flight' if f['pending'] else 'Waiting for next request',22,MUTED)
        counts=summary if end else f
        for j,(label,key) in enumerate([('Inputs applied','applied'),('No-input choices','idle'),('Stale discarded','discarded'),('Request errors','errors')]):
            y=510+j*65
            text(d,sx,y,label,23,MUTED)
            text(d,offset+665,y,counts[key],26,INK,True)
        if f['error']:wrap(d,sx,779,f['error'],365,19,'#FFBAA1')
        values=[('Lines',f['lines']),('Pieces',f['pieces']),('Score',f['score'])]
        for j,(label,value) in enumerate(values):
            x=offset+j*125
            text(d,x,874,value,37,GREEN,True);text(d,x,923,label,22,MUTED)
        if end:
            med=summary['decision_latency_ms']['median']
            text(d,sx,874,f'{med:.0f} ms median',28,bold=True)
            text(d,sx,925,'Neither run reached 10 lines.',21,MUTED)
        else:
            text(d,sx,874,f"{f['calls']} requests",27,bold=True)
            text(d,sx,925,'Paused at 60 s' if end else 'Gravity runs during inference.',21,MUTED)
    d.line((48,986,1552,986),fill=LINE,width=2)
    text(d,48,1010,'Structured state + game-computed legal landings. One seed; no gameplay or speed-parity claim.',24,MUTED)
    text(d,48,1054,'github.com/TimothyZhang7/open-decisions',24,GREEN)
    text(d,1065,1054,'Sequential runs · Qwen scores uncalibrated',21,MUTED)
    return im

path=OUT/'tetris-qwen-vs-jev.mp4'
writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'avc1'),FPS,(W,H))
assert writer.isOpened()
try:
    for idx in range(60*FPS):
        ms=idx*1000/FPS
        im=render(ms)
        if idx==30*FPS:im.save(OUT/'tetris-comparison-poster.png')
        writer.write(cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR))
    end=render(60000,end=True)
    end.save(OUT/'tetris-comparison-results.png')
    frame=cv2.cvtColor(np.array(end),cv2.COLOR_RGB2BGR)
    for _ in range(5*FPS):writer.write(frame)
finally:writer.release()

# A lightweight README preview covers the same complete minute at 1×, sampled at 2 Hz.
palette=render(30000).resize((960,660)).quantize(colors=96)
gif_frames=[render(ms).resize((960,660)).quantize(palette=palette) for ms in range(0,60000,500)]
gif_frames[0].save(OUT/'tetris-comparison.gif',save_all=True,append_images=gif_frames[1:],duration=500,loop=0,optimize=True)
capture=cv2.VideoCapture(str(path))
meta={k:capture.get(v) for k,v in {'fps':cv2.CAP_PROP_FPS,'frames':cv2.CAP_PROP_FRAME_COUNT,'width':cv2.CAP_PROP_FRAME_WIDTH,'height':cv2.CAP_PROP_FRAME_HEIGHT}.items()}
for t in [0,15,30,45,59,63]:
    capture.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=capture.read();assert ok
capture.release()
meta.update(duration_seconds=meta['frames']/meta['fps'],content='60 seconds of recorded game states at 1×, plus a five-second result card',filesize=path.stat().st_size)
(OUT/'tetris-video-info.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
