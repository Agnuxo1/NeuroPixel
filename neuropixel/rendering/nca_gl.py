"""Native local NCA arithmetic through RGBA32F framebuffer rendering.

Torch-free backend. Read/write textures are distinct at each draw. Channel banks
are laid vertically; no quantization, clipping, blending or wraparound is applied.
This module supplies execution, not a performance or biological claim.
"""
from __future__ import annotations
import hashlib,time
import numpy as np

VERTEX='''#version 330
void main() {
 vec2 p[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));
 gl_Position=vec4(p[gl_VertexID],0,1);
}'''

def pack(a):
    a=np.asarray(a,dtype=np.float32);c,h,w=a.shape;b=(c+3)//4
    p=np.zeros((b*4,h,w),dtype=np.float32);p[:c]=a
    return p.reshape(b,4,h,w).transpose(0,2,3,1).reshape(b*h,w,4).copy()

def unpack(raw,c,h,w):
    return np.frombuffer(raw,dtype=np.float32).reshape((c+3)//4,h,w,4).transpose(0,3,1,2).reshape(-1,h,w)[:c].copy()

def weight_texture(ctx,a):
    flat=np.asarray(a,dtype=np.float32).reshape(-1);p=np.zeros(((len(flat)+3)//4)*4,np.float32);p[:len(flat)]=flat
    tex=ctx.texture((len(p)//4,1),4,p.tobytes(),dtype='f4');tex.filter=(9728,9728);return tex

def field_texture(ctx,c,h,w):
    tex=ctx.texture((w,h*((c+3)//4)),4,dtype='f4');tex.filter=(9728,9728);tex.repeat_x=False;tex.repeat_y=False;return tex

COMMON='''
uniform sampler2D x0; uniform sampler2D x1; uniform sampler2D x2;
uniform sampler2D weights; uniform sampler2D bias; uniform sampler2D mask;
float scalar(sampler2D t,int index) {return texelFetch(t,ivec2(index/4,0),0)[index%4];}
float cell(sampler2D t,ivec2 p,int c) {
 if (p.x<0||p.x>=WIDTH||p.y<0||p.y>=HEIGHT) return 0.0;
 return texelFetch(t,ivec2(p.x,p.y+(c/4)*HEIGHT),0)[c%4];
}
out vec4 colour;
'''

class Pass:
    def __init__(self,ctx,w,h,channels,expression,weight,bias,body='',relu=False,mask=False):
        self.ctx,self.w,self.h,self.channels=ctx,w,h,channels
        self.weights=weight_texture(ctx,weight);self.bias=weight_texture(ctx,bias)
        self.texture=field_texture(ctx,channels,h,w);self.fbo=ctx.framebuffer([self.texture])
        source=('#version 330\n#define WIDTH '+str(w)+'\n#define HEIGHT '+str(h)+'\n'+COMMON+body+'\nvoid main(){\n'
            'ivec2 pixel=ivec2(gl_FragCoord.xy);int bank=pixel.y/HEIGHT;pixel.y=pixel.y%HEIGHT;vec4 value=vec4(0);\n'
            'for(int lane=0;lane<4;lane++){int output_c=bank*4+lane;if(output_c<'+str(channels)+'){\n'
            'float v=scalar(bias,output_c);\n'+expression+'\n'+('v=max(v,0.0);\n' if relu else '')
            +('v*=texelFetch(mask,pixel,0).r;\n' if mask else '')+'value[lane]=v;}}colour=value;}')
        self.shader_sha256=hashlib.sha256(source.encode()).hexdigest();start=time.perf_counter()
        self.program=ctx.program(vertex_shader=VERTEX,fragment_shader=source);self.vao=ctx.vertex_array(self.program,[])
        self.compile_seconds=time.perf_counter()-start
    def draw(self,x0,x1=None,x2=None,mask=None,target=None):
        textures={'x0':(x0,0),'x1':(x1,1),'x2':(x2,2),'weights':(self.weights,3),'bias':(self.bias,4),'mask':(mask,5)}
        target=self.fbo if target is None else target
        for name,(tex,unit) in textures.items():
            if name in self.program:
                if tex is None:raise ValueError('Required texture '+name)
                tex.use(unit);self.program[name].value=unit
        target.use();self.ctx.viewport=(0,0,self.w,self.h*((self.channels+3)//4));self.vao.render(vertices=3)
        return target.color_attachments[0]
    def release(self):
        for obj in [self.vao,self.program,self.fbo,self.texture,self.weights,self.bias]:obj.release()

class ConvPass(Pass):
    def __init__(self,ctx,w,h,weight,bias,dilation=1,relu=False,present=False):
        weight=np.asarray(weight,np.float32);out_c,in_c,k,_=weight.shape
        expression=f'''for(int input_c=0;input_c<{in_c};input_c++){{
 for(int ky=0;ky<{k};ky++)for(int kx=0;kx<{k};kx++){{
  int index=((output_c*{in_c}+input_c)*{k}+ky)*{k}+kx;
  v+=scalar(weights,index)*cell(x0,pixel+ivec2((kx-{k//2})*{dilation},(ky-{k//2})*{dilation}),input_c);
 }}}}'''
        super().__init__(ctx,w,h,out_c,expression,weight,bias,relu=relu,mask=present)

class GLNCA:
    """Three render passes implement one unchanged eval NCA state update."""
    def __init__(self,ctx,width,height,weights):
        self.ctx=ctx;self.w=width;self.h=height;self.weights=weights
        f1=np.asarray(weights['f1.weight'],np.float32);f2=np.asarray(weights['f2.weight'],np.float32)
        self.c=f2.shape[0];self.hidden=f2.shape[1];self.cid=f1.shape[1]-3*self.c
        if self.cid<=0:raise ValueError('Native concatenation dimensions')
        self.ids=field_texture(ctx,self.cid,height,width)
        self.present=ctx.texture((width,height),1,dtype='f4');self.present.filter=(9728,9728)
        self.fire=ctx.texture((width,height),1,np.ones((height,width),np.float32).tobytes(),dtype='f4');self.fire.filter=(9728,9728)
        self.seed=ConvPass(ctx,width,height,weights['seed.weight'],weights['seed.bias'],present=True)
        expression='''int source_c=output_c/2;
for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++) {
 v+=scalar(weights,output_c*9+ky*3+kx)*cell(x0,pixel+ivec2(kx-1,ky-1),source_c);
}'''
        self.perceive=Pass(ctx,width,height,2*self.c,expression,weights['perceive.weight'],np.zeros(2*self.c,np.float32))
        expression=f'''for(int input_c=0;input_c<{3*self.c+self.cid};input_c++){{
 float input_v;
 if(input_c<{self.c})input_v=cell(x0,pixel,input_c);
 else if(input_c<{3*self.c})input_v=cell(x1,pixel,input_c-{self.c});
 else input_v=cell(x2,pixel,input_c-{3*self.c});
 v+=scalar(weights,output_c*{3*self.c+self.cid}+input_c)*input_v;
}}'''
        self.f1=Pass(ctx,width,height,self.hidden,expression,f1,weights['f1.bias'],relu=True)
        expression=f'''for(int hidden_c=0;hidden_c<{self.hidden};hidden_c++){{
 v+=scalar(weights,output_c*{self.hidden}+hidden_c)*cell(x0,pixel,hidden_c);
}}
v=cell(x1,pixel,output_c)+v*texelFetch(mask,pixel,0).r;'''
        self.f2=Pass(ctx,width,height,self.c,expression,f2,weights['f2.bias'])
        self.states=[field_texture(ctx,self.c,height,width) for _ in range(2)];self.fbos=[ctx.framebuffer([t]) for t in self.states];self.current=0
        self.passes=[self.seed,self.perceive,self.f1,self.f2]
        # ModernGL uses capability bit flags, not raw GL enum values.
        import moderngl
        self.ctx.disable(moderngl.BLEND);self.ctx.disable(moderngl.DEPTH_TEST)
        self.fire_changed=False
    def initialize(self,ids,present):
        ids=np.asarray(ids,np.float32);present=np.asarray(present,np.float32)
        if ids.shape!=(self.cid,self.h,self.w) or present.shape!=(self.h,self.w):raise ValueError('Input geometry differs')
        self.ids.write(pack(ids).tobytes());self.present.write(present.tobytes());self.current=0
        self.fire.write(np.ones((self.h,self.w),np.float32).tobytes());self.fire_changed=False
        self.seed.draw(self.ids,mask=self.present,target=self.fbos[0])
    def set_state(self,state):
        if np.asarray(state).shape!=(self.c,self.h,self.w):raise ValueError('State dimensions')
        self.current=0;self.states[0].write(pack(state).tobytes())
    def step(self,fire=None):
        if fire is not None:
            fire=np.asarray(fire,np.float32)
            if fire.shape!=(self.h,self.w):raise ValueError('Mask geometry')
            self.fire.write(fire.tobytes())
            self.fire_changed=True
        elif self.fire_changed:
            self.fire.write(np.ones((self.h,self.w),np.float32).tobytes());self.fire_changed=False
        p=self.perceive.draw(self.states[self.current]);h=self.f1.draw(self.states[self.current],p,self.ids)
        nxt=1-self.current;self.f2.draw(h,self.states[self.current],mask=self.fire,target=self.fbos[nxt]);self.current=nxt
    def state(self):return unpack(self.states[self.current].read(),self.c,self.h,self.w)
    def metadata(self):return dict(state_channels=self.c,identity_channels=self.cid,hidden=self.hidden,passes_per_update=3,storage='RGBA32F banked textures',shader_sha256=[p.shader_sha256 for p in self.passes],compile_seconds=sum(p.compile_seconds for p in self.passes))
    def release(self):
        for p in self.passes:p.release()
        for obj in self.fbos+self.states+[self.ids,self.present,self.fire]:obj.release()

def numpy_update(state,ids,weights,fire=None):
    """Independent scalar-kernel indexing; vectorized only over cell positions."""
    state=np.asarray(state,np.float64);ids=np.asarray(ids,np.float64);c,h,w=state.shape
    padded=np.pad(state,((0,0),(1,1),(1,1)));perception=np.zeros((2*c,h,w),np.float64)
    kernels=np.asarray(weights['perceive.weight'],np.float64).reshape(2*c,3,3)
    for k in range(2*c):
        for y in range(3):
            for x in range(3):perception[k]+=kernels[k,y,x]*padded[k//2,y:y+h,x:x+w]
    features=np.concatenate([state,perception,ids],axis=0).reshape(3*c+ids.shape[0],-1)
    hidden=np.maximum(0,np.asarray(weights['f1.weight'],np.float64).reshape(-1,features.shape[0])@features+np.asarray(weights['f1.bias'])[:,None])
    delta=(np.asarray(weights['f2.weight'],np.float64).reshape(c,-1)@hidden+np.asarray(weights['f2.bias'])[:,None]).reshape(c,h,w)
    return state+delta*(1 if fire is None else np.asarray(fire,np.float64)[None])
