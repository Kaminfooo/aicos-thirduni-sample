const fs=require('fs'),vm=require('vm'),assert=require('assert');
const context=vm.createContext({console});vm.runInContext(fs.readFileSync('dashboard/map-assets/location-map.js','utf8').split('let locationMap=')[0],context);
const parse=(raw,axis)=>vm.runInContext(`parseCameraCoordinate(${JSON.stringify(raw)},${JSON.stringify(axis)})`,context);let passed=0;
for(const [raw,axis,expected] of [
 ['','latitude',null],['0','latitude',0],['0','longitude',0],['.5','latitude',.5],['12.5','latitude',12.5],['12,5','latitude',12.5],['-12.5','latitude',-12.5],['90','latitude',90],['-180','longitude',-180],
 ['12°30\'00"N','latitude',12.5],['12 30 00 S','latitude',-12.5],['12°30.0\'','latitude',12.5],['S12:30:00','latitude',-12.5],['12°30′00″N','latitude',12.5],['12°30’00”N','latitude',12.5],['123000N','latitude',12.5],['1230\'00','latitude',12.5],
 ['2°15\'00"E','longitude',2.25],['21500E','longitude',2.25],['215\'00E','longitude',2.25],['2 15 00 W','longitude',-2.25],['-0 30 00','longitude',-.5],['180°0\'0"E','longitude',180]
 ]){const actual=parse(raw,axis);expected===null?assert.equal(actual,null):assert.ok(Math.abs(actual-expected)<1e-9,raw);passed++}
for(const [raw,axis] of [['91','latitude'],['181','longitude'],['90 0 1 N','latitude'],['180 0 1 E','longitude'],['12 60 0 N','latitude'],['12 0 60 N','latitude'],['12 -1 0 N','latitude'],['12 1.5 2 N','latitude'],['12.5 1 N','latitude'],['12 N','longitude'],['12 W','latitude'],['-12 N','latitude'],['NaN','latitude'],['Infinity','longitude'],['1e2','longitude'],['N12S','latitude'],['hello','latitude'],['123000','latitude']]){assert.throws(()=>parse(raw,axis));passed++}
console.log(JSON.stringify({passed,failed:0,scope:'Decimal and DMS formats, compact notation, hemisphere/sign handling, zero coordinates, range/minute/second bounds and invalid text'}));
