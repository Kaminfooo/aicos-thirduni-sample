"""Read the native AVI video stream timebase; never infer FPS from content."""
import struct


def read_clock(path):
    data=path.read_bytes()
    if data[:4]!=b'RIFF' or data[8:12]!=b'AVI ':raise ValueError('Not a native AVI file')
    headers=[]
    def walk(start,end):
        offset=start
        while offset+8<=end:
            tag=data[offset:offset+4];size=struct.unpack_from('<I',data,offset+4)[0];body=offset+8;stop=body+size
            if stop>end:raise ValueError('Invalid AVI chunk boundary')
            if tag==b'LIST' and data[body:body+4] in (b'hdrl',b'strl'):walk(body+4,stop)
            elif tag==b'strh' and data[body:body+4]==b'vids':
                if size<56:raise ValueError('Short video stream header')
                scale,rate,start_sample,length=struct.unpack_from('<IIII',data,body+20)
                if not scale or not rate or not length or start_sample:raise ValueError('Unsupported AVI clock')
                headers.append(dict(scale=scale,rate=rate,frames=length))
            offset=stop+(size&1)
    walk(12,min(len(data),struct.unpack_from('<I',data,4)[0]+8))
    if len(headers)!=1:raise ValueError('Require exactly one video stream')
    return dict(headers[0],source='AVI stream dwScale/dwRate with actual decoded-frame count validation',
        specification='https://learn.microsoft.com/en-us/previous-versions/ms779638(v=vs.85)')
