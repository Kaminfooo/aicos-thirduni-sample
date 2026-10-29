"""Real local inference integration checks, separate from recognition accuracy."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from fastapi.testclient import TestClient
from deploy.judge_workspace import create_app,isolated
from deploy.judge_capabilities import IDS


def encode(frame):
    h,w=frame.shape[:2];scale=min(480/w,480/h,1)
    frame=cv2.resize(frame,(round(w*scale),round(h*scale)))
    ok,data=cv2.imencode('.png',frame)
    if not ok:raise RuntimeError('PNG encoding failed')
    return base64.b64encode(data).decode()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'.runtime'):raise ValueError('Choose a new runtime directory')
    out.mkdir(parents=True);report={'complete':False,'cases':[],'physical_camera_tested':False,
        'scope':'Fresh local API inference and evidence integration; not new independent accuracy'}
    started=time.monotonic()
    def save(): (out/'report.json').write_text(json.dumps(report,indent=2))
    save()
    with TestClient(create_app()) as client:
        settings={'detectors':list(IDS),'region':[.1,.1,.9,.9]}
        client.post('/judge/api/run',json={'mode':'file',**settings}).raise_for_status()
        measurements=[]
        for second in (0,.34):
            tick=time.monotonic()
            response=client.post('/judge/api/frame',json={'image':encode(np.zeros((270,480,3),np.uint8)),'seconds':second})
            response.raise_for_status();data=response.json()
            assert {r['id'] for r in data['detectors']}==set(IDS)
            assert not [r for r in data['detectors'] if r['status']=='error'],data
            assert not data['alert'],data
            measurements.append({'wall_seconds':time.monotonic()-tick,'detectors':data['detectors']})
        client.post('/judge/api/finish',json={}).raise_for_status()
        report['cases'].append({'case':'all twelve adapters, real weights, two blank frames',
            'passed':True,'measurements':measurements});save()
        cases=[
            ('person_down_reference',isolated.ASSETS/'controlled-fall.mp4',['person_down','fire_smoke','crowd'],'person_down',{}),
            ('walking_negative',isolated.ASSETS/'walking.mp4',['person_down','fire_smoke'],None,{}),
            ('fire_reference',ROOT/'data/capability_validation_20261027/firesense/subset/fire-positive-posVideo11.870.avi',['fire_smoke'],'fire_smoke',{}),
            ('smoke_reference',ROOT/'data/capability_validation_20261027/firesense/subset/smoke-positive-testpos06.822.avi',['fire_smoke'],'fire_smoke',{}),
            ('zone_presence',ROOT/'data/capability_validation_20261027/site/Walk1.mpg',['intrusion','loitering','crowd'],'intrusion',{'region':[0,0,1,1],'loiter_seconds':5,'crowd_warn':2}),
            ('violence_known_failure',ROOT/'data/capability_validation_20261027/airtlab/violent-1.mp4',['violence'],'record_only',{})]
        for name,path,detectors,target,overrides in cases:
            client.post('/judge/api/run',json={'mode':'file','detectors':detectors,**overrides}).raise_for_status()
            capture=cv2.VideoCapture(str(path));next_sample=0;rows=[];incidents=[]
            try:
                if not capture.isOpened():raise ValueError('Cannot decode '+str(path))
                while True:
                    if time.monotonic()-started>720:raise RuntimeError('Twelve-minute integration budget exceeded')
                    ok,frame=capture.read()
                    if not ok:break
                    second=capture.get(cv2.CAP_PROP_POS_MSEC)/1000
                    if second<next_sample:continue
                    next_sample=second+1/3
                    response=client.post('/judge/api/frame',json={'image':encode(frame),'seconds':second})
                    response.raise_for_status();data=response.json()
                    assert not [r for r in data['detectors'] if r['status']=='error'],data
                    rows.append({'seconds':second,'detectors':data['detectors']})
                    incidents=data['incidents']
                    if target not in (None,'record_only') and any(r['capability']==target for r in incidents):break
            finally:capture.release()
            client.post('/judge/api/finish',json={}).raise_for_status()
            assert rows,'No frames analysed'
            if target is None:assert not incidents,incidents
            elif target!='record_only':assert any(r['capability']==target for r in incidents),(name,incidents)
            evidence=[]
            for incident in incidents:
                query='?event_id='+str(incident['event_id'])
                before=client.get('/judge/api/incident'+query);before.raise_for_status()
                assert before.json()['verification']['ok']
                assert client.post('/judge/api/decision'+query,json={'status':'approved'}).status_code==409
                client.post('/judge/api/review'+query,json={'decision':'accept','note':'Automated integration test using attributed reference footage; model interpretation unverified.'}).raise_for_status()
                client.post('/judge/api/decision'+query,json={'status':'approved'}).raise_for_status()
                after=client.get('/judge/api/incident'+query).json()
                assert after['verification']['ok'] and after['event']['status']=='approved'
                for kind in ('snapshot','clip'):client.get('/judge/api/media/'+kind+query).raise_for_status()
                evidence.append({'id':incident['event_id'],'verified':True,'actions':[r['action'] for r in after['audit']]})
            record={'case':name,'path':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'passed':True,'sampled_frames':len(rows),'incidents':incidents,'evidence':evidence,
                'frames':rows,'recognition_gate':target!='record_only'}
            report['cases'].append(record);save();print(name,len(rows),incidents,flush=True)
    report.update(complete=True,elapsed_seconds=time.monotonic()-started);save()
    (out/'exit.txt').write_text('0');print('Integration checks passed; recognition limitations unchanged.',flush=True)


if __name__=='__main__':main()
