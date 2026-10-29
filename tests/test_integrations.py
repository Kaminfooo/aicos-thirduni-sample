import os, sqlite3, sys, threading, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from integrations import IntegrationBus

class T(unittest.TestCase):
    def test_outbox_is_durable_and_deduplicated(self):
        old={k:os.environ.get(k) for k in ['AICOS_INTEGRATION_WEBHOOK_URLS','AICOS_INTEGRATION_STAGES']}
        try:
            os.environ['AICOS_INTEGRATION_WEBHOOK_URLS']='https://vms.local/aicos'
            os.environ['AICOS_INTEGRATION_STAGES']='approved'
            c=sqlite3.connect(':memory:',check_same_thread=False); c.row_factory=sqlite3.Row
            bus=IntegrationBus(c,threading.RLock())
            bus._stop.set()
            event={'id':7,'event_uuid':'u7','source_id':'CAM-1','status':'approved','extra':{}}
            self.assertEqual(bus.queue('approved',event),1)
            self.assertEqual(bus.queue('approved',event),0)
            self.assertEqual(bus.status()['pending'],1)
            sent=[]
            def fake(row): sent.append(row['id']); return True,''
            self.assertEqual(bus.flush_once(sender=fake),1)
            self.assertEqual(bus.status()['pending'],0)
            bus.close(); c.close()
        finally:
            for k,v in old.items():
                if v is None: os.environ.pop(k,None)
                else: os.environ[k]=v
if __name__=='__main__': unittest.main()
