// Deterministic controller checks. DOM doubles do not replace browser testing.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync('dashboard/judge-workspace.html','utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
const elements=new Map();
function element(tag='div'){
 const classes=new Set();const x={tag,tagName:tag.toUpperCase(),children:[],value:'',checked:false,disabled:false,textContent:'',files:[],attrs:{},listeners:{},
 style:{setProperty(key,value){this[key]=value}},classList:{add:v=>classes.add(v),remove:v=>classes.delete(v),contains:v=>classes.has(v),toggle(v){if(classes.has(v)){classes.delete(v);return false}classes.add(v);return true}},
 setAttribute(key,value){this.attrs[key]=String(value)},getAttribute(key){return this.attrs[key]},focus(){this.focused=true},scrollIntoView(){this.scrolled=true},click(){this.clicked=true},
 append(...items){this.children.push(...items)},replaceChildren(...items){this.children=[...items]},
 removeAttribute(){},querySelectorAll(selector){let all=[];function walk(n){for(const c of n.children||[]){if(c.tag==='input')all.push(c);walk(c)}}walk(this);return selector==='input:checked'?all.filter(c=>c.checked):all},
 addEventListener(name,handler){this.listeners[name]=handler},removeEventListener(){},pause(){},load(){},getContext(){return {clearRect(){},strokeRect(){},setLineDash(){},fillText(){},drawImage(){}}}};
 return x;
}
const $=id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id)};
for(const [id,v] of Object.entries({'region-left':'.1','region-top':'.1','region-right':'.9','region-bottom':'.9','crowd-warn':'8','loiter-seconds':'120','stopped-seconds':'45','direction':''}))$(id).value=v;
const ids=['person_down','fire_smoke','crowd','intrusion','loitering','vehicles','abandoned_object','violence','fight','panic','snatch','motion'];
const catalog=ids.map(id=>({id,title:id,status:'Development',limitation:'Test only'}));
let requests=[],nextResponse=null,tracksStopped=0,mobile=false;
const context=vm.createContext({console,setTimeout,clearTimeout,AbortSignal,performance,URL,Blob,encodeURIComponent,
 document:{getElementById:$,createElement:element,createElementNS:(_,tag)=>element(tag),createTextNode:t=>({textContent:t}),querySelectorAll:()=>[]},
 window:{addEventListener(){},matchMedia:query=>({matches:query.includes('max-width')?mobile:false})},navigator:{},fetch:async(url,options)=>{requests.push({url,options});if(nextResponse)return nextResponse;return {ok:true,headers:{get:()=> 'application/json'},json:async()=>url==='/health'?{ok:true,mode:'local-fresh-inference'}:{capabilities:catalog}}}});
function run(js){return vm.runInContext(js,context)}
(async()=>{
 vm.runInContext(source,context);await new Promise(setImmediate);
 let passed=0;
 assert.equal($('capabilities').children.length,12);assert.equal(run('selected().length'),1);passed++;
 $('select-all').onclick();assert.equal(run('selected().length'),12);assert.equal($('pause-alert').checked,false);passed++;
 $('select-core').onclick();assert.equal(run('selected().join(",")'),'person_down,fire_smoke,crowd');assert.equal($('pause-alert').checked,true);passed++;
 $('region-left').value='.95';assert.throws(()=>run('settings()'),/rectangle/);$('region-left').value='.1';passed++;
 $('crowd-warn').value='2.5';assert.throws(()=>run('settings()'),/rule values/);$('crowd-warn').value='8';passed++;
 for(const input of $('capabilities').querySelectorAll('input'))input.checked=false;
 assert.throws(()=>run('settings()'),/at least one/);$('select-reference').onclick();passed++;
 const before=requests.length;await run('start("file")');assert.equal(requests.length,before);assert.match($('status').textContent,/Select a video file/);assert.equal(run('active'),false);passed++;
 nextResponse={ok:true,headers:{get:()=> 'text/html'}};const writes=requests.length;
 await assert.rejects(run('api("decision",{status:"approved"})'),/page instead of data/);assert.equal(requests.length,writes+1);passed++;
 nextResponse={ok:true,headers:{get:()=> 'application/json'},json:async()=>({})};run('selectedIncident=42');await run('api("decision",{status:"dismissed"})');assert.match(requests.at(-1).url,/decision\?event_id=42$/);passed++;
 context.testStream={getTracks:()=>[{stop:()=>tracksStopped++},{stop:()=>tracksStopped++}]};run('stream=testStream;cleanup()');assert.equal(tracksStopped,2);assert.equal(run('stream'),null);passed++;
 run('runAlerts=[{event_id:42}];active=true;buttons()');assert.equal($('incident-picker').disabled,true);assert.equal($('start-camera').disabled,true);run('active=false;buttons()');assert.equal($('incident-picker').disabled,false);passed++;
 run('runAlerts=[];detectorFailures=["fire_smoke"]');await run('finishRun("Run ended")');assert.match($('status').textContent,/unavailable/);assert.equal($('status').className,'error');passed++;
 run('runAlerts=[];resetDashboard("scenario");recordSample({frames:1,seconds:.33,detectors:[{id:"person_down",status:"monitoring",elapsed_ms:100},{id:"fire_smoke",status:"monitoring",elapsed_ms:20}]});recordSample({frames:2,seconds:.66,detectors:[{id:"person_down",status:"alert",elapsed_ms:200},{id:"fire_smoke",status:"monitoring",elapsed_ms:60}]})');
 assert.equal(run('summarizeRun().average_processing_ms'),190);assert.equal($('metric-frames').textContent,'2');assert.equal($('metric-time').textContent,'0.7 s');assert.equal($('activity-chart').attrs['data-samples'],'2');passed++;
 run('runAlerts=[{event_id:42,capability:"person_down",type:"Test fall"},{event_id:43,capability:"fire_smoke",type:"Test fire"}];rememberIncident({event:{id:42,type:"Test fall",status:"approved"},draft:{review_status:"accepted_for_demo"},verification:{ok:true}});rememberIncident({event:{id:43,type:"Test fire",status:"detected"},draft:{review_status:"pending"},verification:{ok:false}})');
 assert.equal($('metric-alerts').textContent,'2');assert.equal($('metric-reviews').textContent,'1');assert.equal($('metric-decisions').textContent,'1');assert.equal($('metric-verified').textContent,'1');passed++;
 $('donut-mode').value='alerts';run('renderDonut()');assert.equal(run('donutData().total'),2);assert.equal(run('donutData().segments.length'),2);assert.equal($('mix-caption').textContent,'alerts retained');passed++;
 $('donut-mode').value='selection';run('renderDonut()');assert.equal(run('donutData().total'),1);assert.match($('mix-note').textContent,/not recognition accuracy/);passed++;
 $('chart-range').value='30';run('resetDashboard("file");for(let i=0;i<35;i++)recordSample({frames:i+1,seconds:i,detectors:[]})');assert.equal($('activity-chart').attrs['data-samples'],'30');$('chart-range').value='all';run('renderTimeline()');assert.equal($('activity-chart').attrs['data-samples'],'35');passed++;
 run('clearIncident();resetDashboard("camera")');assert.equal($('metric-frames').textContent,'0');assert.equal($('metric-reviews').textContent,'0');assert.equal($('metric-verified').textContent,'0');assert.equal($('chart-empty').hidden,false);passed++;
 mobile=true;run('updateNavigationAccess()');assert.equal($('sidebar').inert,true);$('menu-toggle').onclick();assert.equal($('sidebar').inert,false);assert.equal($('nav-scrim').hidden,false);assert.equal($('menu-toggle').attrs['aria-expanded'],'true');run('closeNavigation(true)');assert.equal($('sidebar').inert,true);assert.equal($('menu-toggle').focused,true);passed++;
 $('evaluation-scope').tagName='DETAILS';run('navigateSection("evaluation-scope")');assert.equal($('evaluation-scope').open,true);assert.equal($('evaluation-scope').scrolled,true);assert.equal($('evaluation-scope').focused,true);mobile=false;run('updateNavigationAccess()');assert.equal($('sidebar').inert,false);passed++;
 nextResponse={ok:true,headers:{get:()=> 'application/json'},json:async()=>({ok:true,mode:'local-fresh-inference'})};await run('checkConnection()');assert.equal($('service-status').textContent,'Inference connected');passed++;
 nextResponse={ok:true,headers:{get:()=> 'text/html'}};const checks=requests.length;await run('checkConnection()');assert.equal(requests.length,checks+1);assert.match($('service-status').textContent,/unavailable/);passed++;
 run('capabilitiesReady=false');nextResponse=null;await run('checkConnection()');assert.equal(run('capabilitiesReady'),true);assert.equal($('start-scenario').disabled,false);passed++;
 // Surveillance must forward the complete selection and its own camera metadata.
 $('select-all').onclick();context.wallAnalysisSource=()=>({id:'source-test',url:'https://camera.example.test/stream.mp4',location:{camera_name:'Selected camera',location_label:'Synthetic site',latitude:0,longitude:0}});
 Object.assign($('video'),{play:async()=>{},currentTime:0,paused:false,ended:false,videoWidth:0});
 nextResponse={ok:true,headers:{get:()=> 'application/json'},json:async()=>({})};const priorWallRequests=requests.length;
 await run('start("wall")');const wallRequest=requests.slice(priorWallRequests).find(r=>r.url==='/judge/api/run');assert.ok(wallRequest);const wallBody=JSON.parse(wallRequest.options.body);
 assert.equal(wallBody.mode,'camera');assert.equal(wallBody.detectors.length,12);assert.equal(wallBody.camera_name,'Selected camera');assert.equal(wallBody.latitude,0);assert.equal(wallBody.longitude,0);assert.equal(run('active'),false);passed++;
 console.log(JSON.stringify({passed,failed:0,scope:'Controller/DOM doubles; actual browser acceptance recorded separately'}));
})().catch(e=>{console.error(e);process.exitCode=1});
