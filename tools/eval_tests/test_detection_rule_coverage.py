"""Controlled rule regressions. Injected predictions are not model accuracy evidence."""
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'edge'))
sys.path.insert(0, str(ROOT / 'backend'))
import cv2
import numpy as np
from detectors import YoloBehaviorSuite, _in_active_hours
from falldet import FallModelDetector
from violence import ViolenceDetector
from rtsp_worker import MotionDetector
from site_behaviors import SiteBehaviorSuite
from camera_health import ObservedDetector
from camera_integrity import compare_frames, IntegrityMonitor
from camera_registry import CameraRegistry
from persistent_hazards import PersistentHazardDetector

START = 10000.0

class Engine:
    rows = []
    def infer(self, frame):
        return self.rows

def person(x=200, y=180, height=100, width=40, wrists=None):
    points = None
    if wrists is not None:
        points = [[0, 0, 0] for _ in range(17)]
        points[9] = [x + wrists, y, .9]
        points[10] = [x - wrists, y, .9]
    return {'cls': 0, 'conf': .99, 'box': [x-width/2, y-height/2, x+width/2, y+height/2], 'kpts': points}

def vehicle(x=200, y=200):
    return {'cls': 2, 'conf': .99, 'box': [x-20, y-15, x+20, y+15]}

def bag(x=200, y=200):
    return {'cls': 24, 'conf': .99, 'box': [x-10, y-10, x+10, y+10]}

def suite(capability, config=None, camera=None, site=False):
    names = ('person_down','fight','crowd','panic','intrusion','loitering','abandoned_object','snatch','vehicles')
    cfg = {name: {'enabled': name == capability} for name in names}
    cfg[capability].update(config or {})
    engine = Engine()
    implementation = SiteBehaviorSuite if site else YoloBehaviorSuite
    return implementation(camera=camera, config=cfg, engine=engine), engine

class ControlledRules(unittest.TestCase):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    def step(self, detector, engine, seconds, rows):
        engine.rows = rows
        return detector.detect(self.frame, now=START + seconds)
    def types(self, events):
        return [row['type'] for row in events]

    def test_legacy_person_down_transition_requires_hold_and_cools_down(self):
        d,e = suite('person_down', {'seconds': 2})
        self.step(d,e,0,[person()])
        alerts=[]
        for n in range(1,9):
            events=self.step(d,e,n*.5,[person(height=40,width=100)])
            if n < 5: self.assertEqual(events,[])
            alerts += events
        self.assertEqual(self.types(alerts), ['Person down / possible medical emergency'])
        self.assertEqual(alerts[0]['severity'],4)

    def test_legacy_person_down_static_prone_box_is_not_a_transition(self):
        d,e=suite('person_down',{'seconds':2})
        for n in range(12): self.assertEqual(self.step(d,e,n,[person(height=40,width=100)]),[])

    def test_fight_sustained_striking_stationary_pair_and_cooldown(self):
        d,e=suite('fight',{'seconds':2})
        alerts=[]
        for n in range(16):
            rows=[person(200,wrists=40 if n%2 else -40),person(245,wrists=20 if n%2 else -20)]
            events=self.step(d,e,n*.5,rows)
            if n<4: self.assertEqual(events,[])
            alerts+=events
        self.assertEqual(self.types(alerts),['Possible fight / altercation'])

    def test_fight_conversation_and_parallel_walking_are_quiet(self):
        for walking in (False,True):
            with self.subTest(walking=walking):
                d,e=suite('fight',{'seconds':2})
                for n in range(16):
                    x=100+n*15 if walking else 200
                    self.assertEqual(self.step(d,e,n*.5,[person(x),person(x+45)]),[])

    def test_fight_one_stationary_spectator_does_not_count_as_reacting(self):
        d,e=suite('fight',{'seconds':2})
        for n in range(16):
            self.assertEqual(self.step(d,e,n*.5,[person(200,wrists=40 if n%2 else -40),person(245,wrists=0)]),[])

    def test_legacy_crowd_below_warn_warn_critical_and_repeat_gate(self):
        d,e=suite('crowd',{'warn':2,'critical':4})
        rows=[person(70+n*100) for n in range(4)]
        self.assertEqual(self.step(d,e,0,rows[:1]),[])
        self.assertEqual(self.types(self.step(d,e,1,rows[:2])),['Crowd build-up'])
        self.assertEqual(self.step(d,e,2,rows[:2]),[])
        self.assertEqual(self.types(self.step(d,e,3,rows)),['Large crowd forming'])

    def test_panic_three_fast_tracks_need_two_observations_not_one(self):
        d,e=suite('panic')
        self.assertEqual(self.step(d,e,0,[person(60,y=80),person(60,y=240),person(60,y=400)]),[])
        self.assertEqual(self.step(d,e,.5,[person(160,y=80),person(160,y=240),person(160,y=400)]),[])
        event=self.step(d,e,1,[person(260,y=80),person(260,y=240),person(260,y=400)])[0]
        self.assertEqual(event['type'],'Panic movement detected')
        self.assertEqual(event['extra']['runners'],3)
        self.assertEqual(self.step(d,e,1.5,[person(360,y=80),person(360,y=240),person(360,y=400)]),[])

    def test_panic_two_runners_or_slow_walkers_are_quiet(self):
        for count,step in ((2,100),(3,10)):
            d,e=suite('panic')
            for n in range(5):
                self.assertEqual(self.step(d,e,n*.5,[person(60+n*step,y=80+i*160) for i in range(count)]),[])

    def test_intrusion_uses_feet_and_zone_and_repeat_gate(self):
        camera={'zones':[{'name':'Yard','type':'intrusion','points':[[0,0],[.5,0],[.5,1],[0,1]]}]}
        d,e=suite('intrusion',camera=camera)
        self.assertEqual(self.step(d,e,0,[person(500)]),[])
        events=self.step(d,e,1,[person(200)])
        self.assertEqual(self.types(events),['Intrusion — Yard'])
        self.assertEqual(self.step(d,e,2,[person(200)]),[])

    def test_loiter_exit_resets_dwell(self):
        camera={'zones':[{'name':'Lobby','type':'loiter','points':[[0,0],[.5,0],[.5,1],[0,1]]}]}
        d,e=suite('loitering',{'seconds':2},camera=camera)
        self.assertEqual(self.step(d,e,0,[person(200)]),[])
        self.assertEqual(self.step(d,e,1,[person(200)]),[])
        # Move gradually across the boundary so the same track remains matched.
        self.assertEqual(self.step(d,e,1.5,[person(340)]),[])
        self.assertEqual(self.step(d,e,2,[person(200)]),[])
        self.assertEqual(self.step(d,e,3,[person(200)]),[])
        self.assertEqual(self.types(self.step(d,e,4,[person(200)])),['Loitering — Lobby'])

    def test_unattended_bag_needs_owner_absence_for_full_hold(self):
        d,e=suite('abandoned_object',{'seconds':2})
        for t in (0,1,2): self.assertEqual(self.step(d,e,t,[bag(),person()]),[])
        self.assertEqual(self.step(d,e,3,[bag()]),[])
        self.assertEqual(self.types(self.step(d,e,4,[bag()])),['Unattended object left behind'])
        self.assertEqual(self.step(d,e,5,[bag()]),[])

    def test_unattended_moving_bag_does_not_accumulate_hold(self):
        d,e=suite('abandoned_object',{'seconds':2})
        for n in range(6): self.assertEqual(self.step(d,e,n,[bag(60+n*60)]),[])

    def snatch_sequence(self,contact_samples=2,delay=.5):
        d,e=suite('snatch',{'contact_samples':contact_samples})
        out=[]
        out+=self.step(d,e,0,[person(100),person(150)])
        if contact_samples==2: out+=self.step(d,e,.1,[person(100),person(150)])
        for n,x in enumerate((250,350,450)):
            out+=self.step(d,e,delay+n*.5,[person(100),person(x)])
        return out

    def test_snatch_contact_then_one_fast_departure_and_cooldown(self):
        self.assertEqual(self.types(self.snatch_sequence()),['Possible snatch theft (contact then flight)'])

    def test_snatch_single_contact_or_old_contact_is_quiet(self):
        # Two contacts are required, even when the later flight is fast.
        d,e=suite('snatch')
        self.step(d,e,0,[person(100),person(150)])
        for n,x in enumerate((250,350,450)):
            self.assertEqual(self.step(d,e,.5+n*.5,[person(100),person(x)]),[])
        self.assertEqual(self.snatch_sequence(delay=7),[])

    def test_stopped_vehicle_requires_full_dwell_movement_resets_and_cools_down(self):
        d,e=suite('vehicles',{'stopped_seconds':2})
        self.assertEqual(self.step(d,e,0,[vehicle()]),[])
        self.assertEqual(self.step(d,e,1,[vehicle()]),[])
        self.assertEqual(self.types(self.step(d,e,2,[vehicle()])),['Vehicle stopped on the road'])
        self.assertEqual(self.step(d,e,3,[vehicle()]),[])
        moving,em=suite('vehicles',{'stopped_seconds':2})
        for n in range(8): self.assertEqual(self.step(moving,em,n,[vehicle(100+n*20)]),[])

    def test_congestion_needs_count_and_more_than_five_seconds_stillness(self):
        d,e=suite('vehicles',{'stopped_seconds':45,'congestion_count':3})
        cars=[vehicle(x) for x in (100,200,300)]
        for t in (0,1,2,3,4,5): self.assertEqual(self.step(d,e,t,cars),[])
        events=self.step(d,e,6,cars)
        self.assertEqual(self.types(events),['Traffic congestion building'])
        self.assertEqual(events[0]['extra']['stationary_vehicles'],3)
        low,el=suite('vehicles',{'stopped_seconds':45,'congestion_count':3})
        for t in range(8): self.assertEqual(self.step(low,el,t,cars[:2]),[])

    def test_wrong_way_requires_configured_opposite_motion_and_speed(self):
        for step,expected in ((-20,True),(20,False),(-2,False)):
            with self.subTest(step=step):
                d,e=suite('vehicles',{'wrong_way':{'enabled':True,'direction':'right','min_speed':.05}})
                self.assertEqual(self.step(d,e,0,[vehicle(300)]),[])
                events=self.step(d,e,.5,[vehicle(300+step)])
                self.assertEqual('Vehicle travelling against traffic' in self.types(events),expected)

    def test_site_wrong_way_without_direction_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'explicit allowed traffic direction'):
            SiteBehaviorSuite(config={'vehicles':{'enabled':True,'wrong_way':{'enabled':True}},
                'road_region':[[0,0],[1,0],[1,1],[0,1]]},engine=Engine())

    def test_site_crowd_count_drop_and_frame_gap_reset_hold(self):
        for gap in (False,True):
            with self.subTest(frame_gap=gap):
                engine=Engine()
                d=SiteBehaviorSuite(config={'crowd':{'enabled':True,'warn':2,'critical':3,'seconds':2},
                    'crowd_region':[[0,0],[1,0],[1,1],[0,1]]},engine=engine)
                rows=[person(100),person(250),person(400)]
                for t in (0,.5,1):self.assertEqual(self.step(d,engine,t,rows),[])
                if not gap:self.assertEqual(self.step(d,engine,1.5,rows[:1]),[])
                for t in (3,3.5,4,4.5):self.assertEqual(self.step(d,engine,t,rows),[])
                events=self.step(d,engine,5,rows)
                self.assertEqual(self.types(events),['Large crowd occupancy'])
                self.assertTrue(events[0]['extra']['human_review_required'])

    def test_intrusion_inactive_hours_and_disabled_zone_are_quiet(self):
        camera={'zones':[{'name':'Night yard','type':'intrusion','active_hours':'21:00-06:00',
            'points':[[0,0],[1,0],[1,1],[0,1]]}]}
        d,e=suite('intrusion',camera=camera)
        with patch('detectors._in_active_hours',return_value=False):
            self.assertEqual(self.step(d,e,0,[person()]),[])
        with patch('detectors._in_active_hours',return_value=True):
            event=self.step(d,e,1,[person()])[0]
        self.assertEqual(event['severity'],4)

    def test_smoke_persistence_minimum_observations_and_cooldown(self):
        engine=Engine();engine.rows=[{'label':'smoke','conf':.9,'box':[100,100,200,180]}]
        d=PersistentHazardDetector({'seconds':2,'min_observations':4,'gap_tolerance':2},engine=engine)
        for t in (0,1,2):self.assertEqual(d.detect(self.frame,now=START+t),[])
        event=d.detect(self.frame,now=START+3)[0]
        self.assertEqual((event['type'],event['severity']),('Possible smoke detected',3))
        self.assertTrue(event['extra']['human_review_required'])
        self.assertEqual(d.detect(self.frame,now=START+4),[])

    def test_hazard_invalid_box_and_spatial_handoff_do_not_join_hold(self):
        engine=Engine();d=PersistentHazardDetector(engine=engine)
        engine.rows=[{'label':'fire','conf':.99,'box':[100,100,90,180]}]
        for t in (0,.5,1,1.5,2):self.assertEqual(d.detect(self.frame,now=START+t),[])
        engine.rows=[{'label':'fire','conf':.99,'box':[10,10,40,40]}]
        for t in (3,3.5,4,4.5):self.assertEqual(d.detect(self.frame,now=START+t),[])
        engine.rows=[{'label':'fire','conf':.99,'box':[400,10,430,40]}]
        self.assertEqual(d.detect(self.frame,now=START+5),[])
        for t in (5.5,6,6.5):self.assertEqual(d.detect(self.frame,now=START+t),[])
        self.assertEqual(d.detect(self.frame,now=START+7)[0]['type'],'Possible fire detected')

    def test_motion_initial_frame_static_negative_and_changed_positive(self):
        detector=MotionDetector(.02)
        black=np.zeros((100,100,3),dtype=np.uint8)
        white=np.full_like(black,255)
        self.assertEqual(detector.detect(black,None),[])
        self.assertEqual(detector.detect(black,black.copy()),[])
        event=detector.detect(white,black)[0]
        self.assertEqual((event['type'],event['severity']),('Unusual activity (motion)',2))

    def test_night_wraparound_start_included_end_excluded_and_day_quiet(self):
        from types import SimpleNamespace
        for hour,minute,expected in ((20,59,False),(21,0,True),(0,0,True),(5,59,True),(6,0,False)):
            with patch('detectors.time.localtime',return_value=SimpleNamespace(tm_hour=hour,tm_min=minute)):
                self.assertEqual(_in_active_hours('21:00-06:00',START),expected)

    def test_night_security_only_escalation_clamps_severity(self):
        d,e=suite('crowd')
        d.cfg['night']={'hours':'21:00-06:00','severity_boost':3}
        rows=[{'agent':'security','severity':4},{'agent':'traffic','severity':3},{'agent':'emergency','severity':4}]
        with patch('detectors._in_active_hours',return_value=True): events=d._apply_night(rows,START)
        self.assertEqual([r['severity'] for r in events],[5,3,4])
        self.assertTrue(events[0]['extra']['night_escalated'])

    def test_legacy_fall_adapter_needs_seconds_signal_reset_and_cooldown(self):
        class Model:
            score=.9
            def best_fall(self,frame): return self.score,[10,10,50,30]
        model=Model();d=FallModelDetector({'seconds':2},model=model)
        for t in (0,1): self.assertEqual(d.detect(self.frame,now=START+t),[])
        model.score=0; self.assertEqual(d.detect(self.frame,now=START+1.5),[])
        model.score=.9
        for t in (2,3): self.assertEqual(d.detect(self.frame,now=START+t),[])
        event=d.detect(self.frame,now=START+4)[0]
        self.assertEqual(event['severity'],4)
        self.assertEqual(d.detect(self.frame,now=START+5),[])

    def test_legacy_fall_adapter_failure_is_explicit(self):
        class Broken:
            def best_fall(self,frame): raise RuntimeError('controlled inference failure')
        d=FallModelDetector(model=Broken())
        with patch('falldet.log.error'):
            self.assertEqual(d.detect(self.frame,now=START),[])
        self.assertTrue(d._disabled)

    def test_legacy_violence_adapter_stride_streak_reset_and_cooldown(self):
        class Model:
            score=.9
            calls=0
            def best_score(self,frame,class_id): self.calls+=1; return self.score
        model=Model();d=ViolenceDetector({'stride':2,'persistence':2},model=model)
        self.assertEqual(d.detect(self.frame,now=START),[])
        self.assertEqual(model.calls,0)
        self.assertEqual(d.detect(self.frame,now=START+.5),[])
        model.score=0
        d.detect(self.frame,now=START+1);d.detect(self.frame,now=START+1.5)
        model.score=.9
        for t in (2,2.5,3): self.assertEqual(d.detect(self.frame,now=START+t),[])
        event=d.detect(self.frame,now=START+3.5)[0]
        self.assertEqual(event['type'],'Possible violence detected (model)')
        for t in (4,4.5,5,5.5): self.assertEqual(d.detect(self.frame,now=START+t),[])

    def test_legacy_violence_adapter_failure_is_explicit(self):
        class Broken:
            def best_score(self,frame,class_id): raise RuntimeError('controlled inference failure')
        d=ViolenceDetector({'stride':1},model=Broken())
        with patch('violence.log.error'):
            self.assertEqual(d.detect(self.frame,now=START),[])
        self.assertTrue(d._disabled)

    def test_integrity_black_white_blurred_and_empty_frame(self):
        rng=np.random.default_rng(1)
        base=rng.integers(0,256,(160,200,3),dtype=np.uint8)
        self.assertEqual(compare_frames(base,np.zeros_like(base))['status'],'obscured')
        self.assertEqual(compare_frames(base,np.full_like(base,255))['status'],'obscured')
        self.assertEqual(compare_frames(base,np.full_like(base,100))['status'],'degraded')
        with self.assertRaisesRegex(ValueError,'empty frame'): compare_frames(base,None)

    def test_integrity_no_automatic_baseline_is_uncalibrated(self):
        with tempfile.TemporaryDirectory() as td:
            result=IntegrityMonitor('C',td,auto_create=False).assess(self.frame)
            self.assertEqual(result['status'],'uncalibrated')
            self.assertEqual(list(Path(td).iterdir()),[])

    def test_camera_health_exception_and_recovery_are_explicit(self):
        class Detector:
            name='controlled'
            def detect(self,frame,previous): return []
        detector=Detector();observed=ObservedDetector(detector)
        with patch.object(detector,'detect',side_effect=RuntimeError('controlled failure')):
            with self.assertRaises(RuntimeError): observed.detect(None,None)
        self.assertEqual(observed.status,'failed')
        self.assertEqual(observed.detect(None,None),[])
        self.assertEqual(observed.status,'running')

    def test_camera_registry_heartbeat_loss_overrides_detector_and_freshness(self):
        conn=sqlite3.connect(':memory:',check_same_thread=False);conn.row_factory=sqlite3.Row
        self.addCleanup(conn.close)
        registry=CameraRegistry(conn,threading.RLock())
        with patch('camera_registry.time.time',return_value=START):
            registry.heartbeat('C',{'extra':{'detector_status':'failed','frame_age_seconds_at_heartbeat':0}})
        with patch('camera_registry.time.time',return_value=START+registry.stale_seconds+1):
            row=registry.get('C')
        self.assertFalse(row['online']);self.assertEqual(row['operational_status'],'offline')

class MissingObservationBoundaries(unittest.TestCase):
    """Targeted gap reproductions; separate class lets failures remain reviewable."""
    frame = ControlledRules.frame
    step = ControlledRules.step
    def test_stopped_vehicle_reappearance_after_stale_gap_requires_new_hold(self):
        d,e=suite('vehicles',{'stopped_seconds':10})
        self.assertEqual(self.step(d,e,0,[vehicle()]),[])
        self.assertEqual(self.step(d,e,1,[vehicle()]),[])
        self.assertEqual(self.step(d,e,20,[vehicle()]),[],'Unobserved 19 seconds must not count as stopped dwell')

    def test_unattended_reappearance_after_stale_gap_requires_new_hold(self):
        d,e=suite('abandoned_object',{'seconds':10})
        self.assertEqual(self.step(d,e,0,[bag()]),[])
        self.assertEqual(self.step(d,e,1,[bag()]),[])
        self.assertEqual(self.step(d,e,20,[bag()]),[],'Unobserved 19 seconds must not count as unattended dwell')

    def test_reacquired_vehicle_and_bag_can_alert_after_a_new_full_hold(self):
        for capability,config,row,expected in (
                ('vehicles',{'stopped_seconds':10},vehicle(),'Vehicle stopped on the road'),
                ('abandoned_object',{'seconds':10},bag(),'Unattended object left behind')):
            with self.subTest(capability=capability):
                d,e=suite(capability,config)
                for t in (0,1,20,21,22,23,24,25,26,27,28,29):
                    self.assertEqual(self.step(d,e,t,[row]),[])
                events=self.step(d,e,30,[row])
                self.assertEqual([event['type'] for event in events],[expected])

    def test_loiter_inactive_schedule_resets_active_dwell(self):
        camera={'zones':[{'name':'Timed lobby','type':'loiter','active_hours':'21:00-06:00',
                          'points':[[0,0],[1,0],[1,1],[0,1]]}]}
        d,e=suite('loitering',{'seconds':2},camera=camera)
        with patch('detectors._in_active_hours',return_value=True):
            self.assertEqual(self.step(d,e,0,[person()]),[])
            self.assertEqual(self.step(d,e,1,[person()]),[])
        with patch('detectors._in_active_hours',return_value=False):
            for t in (2,3,4):
                self.assertEqual(self.step(d,e,t,[person()]),[])
        with patch('detectors._in_active_hours',return_value=True):
            self.assertEqual(self.step(d,e,5,[person()]),[],'Inactive time must not satisfy an active-zone dwell')
            self.assertEqual(self.step(d,e,6,[person()]),[])
            self.assertEqual(self.step(d,e,7,[person()])[0]['type'],'Loitering — Timed lobby')

if __name__=='__main__': unittest.main(verbosity=2)
