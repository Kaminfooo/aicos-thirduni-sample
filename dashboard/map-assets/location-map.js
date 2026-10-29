'use strict';
function parseCameraCoordinate(raw,axis){
 const limit=axis==='latitude'?90:180,title=axis==='latitude'?'Latitude':'Longitude';
 let text=String(raw).trim().toUpperCase().replace(/[−–]/g,'-').replace(/[′’‘]/g,"'").replace(/[″“”]/g,'"');
 if(!text)return null;
 const invalid=()=>Error(title+': use decimal degrees or degrees/minutes/seconds, with a valid '+(axis==='latitude'?'N/S':'E/W')+' direction.');
 if((text.match(/,/g)||[]).length===1&&!text.includes('.'))text=text.replace(',','.');
 let hemisphere='';
 if(/^[NSEW]/.test(text)){hemisphere=text[0];text=text.slice(1).trim()}
 if(/[NSEW]$/.test(text)){if(hemisphere)throw invalid();hemisphere=text.slice(-1);text=text.slice(0,-1).trim()}
 if(hemisphere&&!(axis==='latitude'?['N','S']:['E','W']).includes(hemisphere))throw invalid();
 if(!text||!/^[+\-\d.\s°º'":]+$/.test(text))throw invalid();
 const negative=text.startsWith('-');
 let tokens=text.replace(/[°º'":]/g,' ').trim().split(/\s+/);
 if(tokens.length===1&&hemisphere&&Math.abs(Number(tokens[0]))>limit){
  const compact=tokens[0].match(/^([+-]?)(\d{1,3})(\d{2})(\d{2}(?:\.\d+)?)$/);
  if(!compact)throw invalid();tokens=[compact[1]+compact[2],compact[3],compact[4]];
 }
 // Some pasted coordinates omit the degree symbol: DDMM'SS.s.
 if(tokens.length===2&&Math.abs(Number(tokens[0]))>limit&&/^[+-]?\d{3,5}$/.test(tokens[0])){
  const prefix=tokens[0];tokens=[prefix.slice(0,-2),prefix.slice(-2),tokens[1]];
 }
 if(tokens.length>3||tokens.some(t=>!/^[-+]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(t)))throw invalid();
 const degrees=Number(tokens[0]),minutes=Number(tokens[1]||0),seconds=Number(tokens[2]||0);
 if(![degrees,minutes,seconds].every(Number.isFinite)||tokens.length>1&&!Number.isInteger(degrees)||tokens.length===3&&!Number.isInteger(minutes)||minutes<0||minutes>=60||seconds<0||seconds>=60)throw invalid();
 if(negative&&['N','E'].includes(hemisphere))throw invalid();
 const value=(Math.abs(degrees)+minutes/60+seconds/3600)*((negative||['S','W'].includes(hemisphere))?-1:1);
 if(Math.abs(value)>limit)throw Error(title+' must be between '+(-limit)+' and '+limit+' degrees.');
 return value;
}
// Camera positions are explicit metadata. A browser's position is never inferred
// to be the location of a remote camera or an event in an example recording.
let locationMap=null,locationLayers=null,streetLayer=null,streetTilesFailed=false;
let cameraLocation={camera_name:'Current input',location_label:'',latitude:null,longitude:null};
const mapEl=id=>document.getElementById(id);
function locationSettings(){return {...cameraLocation}}
function locationMessage(text,error=false){mapEl('map-message').textContent=text;mapEl('map-message').className='map-message'+(error?' error':'')}
function streetMessage(text){mapEl('map-tile-status').textContent=text}
function readLocationForm(){
 const name=mapEl('camera-name').value.trim(),label=mapEl('camera-location-label').value.trim();
 const a=mapEl('camera-latitude').value.trim(),b=mapEl('camera-longitude').value.trim();
 if(!name||name.length>80||label.length>120)throw Error('Enter a camera name of 1–80 characters and a site label of up to 120.');
 if((a==='')!==(b===''))throw Error('Enter both latitude and longitude, or leave both blank.');
 const latitude=parseCameraCoordinate(a,'latitude'),longitude=parseCameraCoordinate(b,'longitude');
 if(latitude!==null&&(!Number.isFinite(latitude)||latitude< -90||latitude>90||!Number.isFinite(longitude)||longitude< -180||longitude>180))throw Error('Latitude must be −90 to 90 and longitude −180 to 180.');
 return {camera_name:name,location_label:label,latitude,longitude};
}
function validLocation(p){return p&&typeof p.latitude==='number'&&Number.isFinite(p.latitude)&&Math.abs(p.latitude)<=90&&typeof p.longitude==='number'&&Number.isFinite(p.longitude)&&Math.abs(p.longitude)<=180}
function coordinateText(p){return validLocation(p)?p.latitude.toFixed(5)+', '+p.longitude.toFixed(5):'Location unknown'}
function locationPopup(title,detail){const box=document.createElement('div'),strong=document.createElement('strong'),p=document.createElement('span');strong.textContent=title;p.textContent=detail;box.append(strong,p);return box}
function focusLocation(p){if(locationMap&&validLocation(p))locationMap.setView([p.latitude,p.longitude],16)}
function showMapIncident(id){if(active||writing)return;mapEl('incident-picker').value=String(id);mapEl('incident-picker').onchange();navigateSection('review-panel')}
function syncLocationMap(){
 if(!mapEl('camera-map'))return;
 for(const id of ['camera-name','camera-location-label','camera-latitude','camera-longitude','save-camera-location','clear-camera-location','use-device-location'])mapEl(id).disabled=active||writing;
 const source=runSettings||cameraLocation;
 mapEl('map-source').textContent=(runSettings?'Run camera: ':'Next input: ')+(source.camera_name||'Current input');
 mapEl('map-coordinate').textContent=coordinateText(source);
 if(locationLayers)locationLayers.clearLayers();
 if(locationMap&&validLocation(source))L.circleMarker([source.latitude,source.longitude],{radius:11,color:'#fff',weight:3,fillColor:'#1675ce',fillOpacity:1}).addTo(locationLayers).bindPopup(locationPopup(source.camera_name,'Configured camera position · '+coordinateText(source)));
 if(locationMap&&validLocation(cameraLocation)&&(source.latitude!==cameraLocation.latitude||source.longitude!==cameraLocation.longitude))L.circleMarker([cameraLocation.latitude,cameraLocation.longitude],{radius:10,color:'#fff',weight:3,fillColor:'#1675ce',fillOpacity:1}).addTo(locationLayers).bindPopup(locationPopup(cameraLocation.camera_name,'Next analysis camera position · '+coordinateText(cameraLocation)));
 if(locationMap&&typeof cameraWallLocations==='function')for(const p of cameraWallLocations().filter(validLocation))L.circleMarker([p.latitude,p.longitude],{radius:9,color:'#fff',weight:2,fillColor:'#1675ce',fillOpacity:.9}).addTo(locationLayers).bindPopup(locationPopup(p.camera_name,(p.location_label||'Camera wall source')+' · '+coordinateText(p)));
 const list=mapEl('map-events');list.replaceChildren();let located=0;
 for(const row of runAlerts){const p=row.location;const button=document.createElement('button');button.className='location-event';button.type='button';button.disabled=active||writing;button.setAttribute('aria-current',String(String(selectedIncident)===String(row.event_id)));const title=document.createElement('strong'),detail=document.createElement('small');title.textContent='#'+row.event_id+' · '+row.type;detail.textContent=(p?.camera_name||'Current input')+' · '+coordinateText(p);button.append(title,detail);button.onclick=()=>{focusLocation(p);showMapIncident(row.event_id)};list.append(button);
  if(validLocation(p)){located++;if(locationMap){const marker=L.circleMarker([p.latitude,p.longitude],{radius:6,color:'#fff',weight:2,fillColor:'#ec7d36',fillOpacity:1}).addTo(locationLayers);const box=locationPopup(row.type,(p.label||p.camera_name)+' · Camera position; exact incident coordinates unavailable.');const open=document.createElement('button');open.type='button';open.className='secondary';open.textContent='Review evidence';open.onclick=()=>showMapIncident(row.event_id);box.append(document.createElement('br'),open);marker.bindPopup(box)}}
 }
 mapEl('map-count').textContent=located+' located / '+runAlerts.length+' alerts';
 if(!runAlerts.length){const empty=document.createElement('p');empty.className='small-note';empty.textContent='Detected events will appear here. No incident locations have been seeded.';list.append(empty)}
 mapEl('map-events-note').textContent=runAlerts.length-located?String(runAlerts.length-located)+' alert(s) have no configured location.':'Event markers use the position saved when analysis starts.';
}
function fitLocations(){if(!locationMap)return;const points=[runSettings||cameraLocation,cameraLocation,...runAlerts.map(r=>r.location),...(typeof cameraWallLocations==='function'?cameraWallLocations():[])].filter(validLocation).map(p=>[p.latitude,p.longitude]);if(points.length)locationMap.fitBounds(L.latLngBounds(points),{padding:[45,45],maxZoom:16});else locationMessage('Set a camera location before centring the map.');}
function initLocationMap(){
 if(typeof L!=='undefined'){
  locationMap=L.map('camera-map',{scrollWheelZoom:false,worldCopyJump:true}).setView([28,3],4);locationLayers=L.layerGroup().addTo(locationMap);
  locationMap.on('click',e=>{if(active||writing){locationMessage('Finish the current analysis before editing camera position.');return}mapEl('camera-latitude').value=e.latlng.lat.toFixed(6);mapEl('camera-longitude').value=e.latlng.wrap().lng.toFixed(6);locationMessage('Pin selected. Apply camera location to use it for the next run.');});
  if(typeof ResizeObserver!=='undefined')new ResizeObserver(()=>locationMap.invalidateSize()).observe(mapEl('camera-map'));
 }else locationMessage('Map library unavailable. Coordinates and event list remain usable; reload to retry.');
 mapEl('save-camera-location').onclick=()=>{if(active||writing)return;try{cameraLocation=readLocationForm();mapEl('camera-latitude').value=cameraLocation.latitude===null?'':cameraLocation.latitude.toFixed(6);mapEl('camera-longitude').value=cameraLocation.longitude===null?'':cameraLocation.longitude.toFixed(6);locationMessage('Camera location applied: '+coordinateText(cameraLocation)+'. Existing event locations remain unchanged.');syncLocationMap();if(validLocation(cameraLocation))focusLocation(cameraLocation)}catch(error){locationMessage(error.message,true)}};
 mapEl('clear-camera-location').onclick=()=>{if(active||writing)return;cameraLocation={...cameraLocation,latitude:null,longitude:null};mapEl('camera-latitude').value='';mapEl('camera-longitude').value='';syncLocationMap();locationMessage('Location cleared for the next analysis. Existing events retain their saved location.')};
 mapEl('fit-map').onclick=fitLocations;
 mapEl('load-street-map').onclick=()=>{if(!locationMap){locationMessage('Map library unavailable; reload to retry.');return}if(streetLayer){locationMap.removeLayer(streetLayer);streetLayer=null}streetTilesFailed=false;streetMessage('Loading street map…');streetLayer=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,noWrap:true,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'});streetLayer.on('tileerror',()=>{streetTilesFailed=true;streetMessage('Street tiles could not load. Check the internet connection and select Reload map; local pins and coordinates remain available.')});streetLayer.on('load',()=>{if(!streetTilesFailed)streetMessage('Street map connected. Click the map to select a camera position; then apply it.')});streetLayer.addTo(locationMap);};
 mapEl('use-device-location').onclick=()=>{if(active||writing)return;if(!navigator.geolocation){locationMessage('Device location is unavailable. Enter camera coordinates manually.');return}locationMessage('Waiting for location permission. Use only if this device is physically at the camera.');navigator.geolocation.getCurrentPosition(p=>{if(active||writing){locationMessage('Analysis started; device position was not applied.');return}mapEl('camera-latitude').value=p.coords.latitude.toFixed(6);mapEl('camera-longitude').value=p.coords.longitude.toFixed(6);locationMessage('Device position received (reported accuracy ±'+Math.round(p.coords.accuracy)+' m). Apply only if this is the camera location.');},()=>locationMessage('Device location was denied or unavailable. Enter coordinates or choose a point on the map.'),{enableHighAccuracy:false,timeout:10000,maximumAge:0})};
 syncLocationMap();
 if(locationMap)mapEl('load-street-map').onclick();
}
initLocationMap();
