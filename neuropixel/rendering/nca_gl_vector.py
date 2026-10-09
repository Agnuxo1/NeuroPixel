"""New RGBA-vector render variant; original scalar renderer stays immutable."""
import hashlib,time
import numpy as np
from .nca_gl import GLNCA,VERTEX,field_texture,pack,unpack

def blocks(weight):
    w=np.asarray(weight,np.float32);oc,ic,kh,kw=w.shape;ob=(oc+3)//4;ib=(ic+3)//4
    a=np.zeros((ob*4,ib*4,kh,kw),np.float32);a[:oc,:ic]=w
    return a.reshape(ob,4,ib,4,kh,kw).transpose(0,2,4,5,1,3).reshape(ob,ib*kh*kw*4,4).copy()

class VectorPass:
    def __init__(self,ctx,width,height,channels,weight,bias,body,code):
        self.ctx=ctx;self.w=width;self.h=height;self.channels=channels
        data=np.asarray(weight,np.float32);self.weights=ctx.texture((data.shape[1],data.shape[0]),4,data.tobytes(),dtype='f4');self.weights.filter=(9728,9728)
        b=np.zeros(((channels+3)//4,4),np.float32);b.reshape(-1)[:channels]=bias;self.bias=ctx.texture((len(b),1),4,b.tobytes(),dtype='f4');self.bias.filter=(9728,9728)
        self.texture=field_texture(ctx,channels,height,width);self.fbo=ctx.framebuffer([self.texture])
        source=f'''#version 330
#define WIDTH {width}
#define HEIGHT {height}
uniform sampler2D x0;uniform sampler2D x1;uniform sampler2D x2;
uniform sampler2D weights;uniform sampler2D bias;uniform sampler2D mask;
vec4 bank(sampler2D t,ivec2 p,int b){{
 if(p.x<0||p.x>=WIDTH||p.y<0||p.y>=HEIGHT)return vec4(0);
 return texelFetch(t,ivec2(p.x,p.y+b*HEIGHT),0);
}}
float scalar_cell(sampler2D t,ivec2 p,int c){{return bank(t,p,c/4)[c%4];}}
vec4 product(ivec2 p,vec4 input_v){{
 return vec4(dot(texelFetch(weights,p,0),input_v),dot(texelFetch(weights,p+ivec2(1,0),0),input_v),dot(texelFetch(weights,p+ivec2(2,0),0),input_v),dot(texelFetch(weights,p+ivec2(3,0),0),input_v));
}}
out vec4 colour;
{body}
void main(){{ivec2 pixel=ivec2(gl_FragCoord.xy);int output_bank=pixel.y/HEIGHT;pixel.y%=HEIGHT;
vec4 value=texelFetch(bias,ivec2(output_bank,0),0);
{code}
colour=value;}}
'''
        self.shader_sha256=hashlib.sha256(source.encode()).hexdigest();start=time.perf_counter();self.program=ctx.program(vertex_shader=VERTEX,fragment_shader=source);self.vao=ctx.vertex_array(self.program,[]);self.compile_seconds=time.perf_counter()-start
    def draw(self,x0,x1=None,x2=None,mask=None,target=None):
        for name,tex,unit in [('x0',x0,0),('x1',x1,1),('x2',x2,2),('weights',self.weights,3),('bias',self.bias,4),('mask',mask,5)]:
            if name in self.program:
                if tex is None:raise ValueError('Required texture '+name)
                tex.use(unit);self.program[name].value=unit
        target=self.fbo if target is None else target;target.use();self.ctx.viewport=(0,0,self.w,self.h*((self.channels+3)//4));self.vao.render(vertices=3);return target.color_attachments[0]
    def release(self):
        for obj in [self.vao,self.program,self.fbo,self.texture,self.weights,self.bias]:obj.release()

class VectorConvPass(VectorPass):
    def __init__(self,ctx,w,h,weight,bias,dilation=1,relu=False,present=False,pad_output=False):
        a=np.asarray(weight,np.float32);oc,ic,k,_=a.shape;ib=(ic+3)//4
        code=f'''for(int input_bank=0;input_bank<{ib};input_bank++)for(int ky=0;ky<{k};ky++)for(int kx=0;kx<{k};kx++){{
 vec4 src=bank(x0,pixel+ivec2((kx-{k//2})*{dilation},(ky-{k//2})*{dilation}),input_bank);
 int col=((input_bank*{k}+ky)*{k}+kx)*4;
 value+=product(ivec2(col,output_bank),src);
}}
'''+('value=max(value,vec4(0));\n' if relu else '')+('value*=texelFetch(mask,pixel,0).r;\n' if present else '')+('if(output_bank==0)value.x=-10000.0;\n' if pad_output else '')
        super().__init__(ctx,w,h,oc,blocks(a),bias,'',code)

class VectorGLNCA(GLNCA):
    def __init__(self,ctx,width,height,weights):
        super().__init__(ctx,width,height,weights)
        for p in self.passes:p.release()
        self.seed=VectorConvPass(ctx,width,height,weights['seed.weight'],weights['seed.bias'],present=True)
        c=self.c;cid=self.cid;hidden=self.hidden
        kernels=np.asarray(weights['perceive.weight'],np.float32).reshape(2*c,9);p=np.zeros(((2*c+3)//4,9,4),np.float32)
        for out_c in range(2*c):p[out_c//4,:,out_c%4]=kernels[out_c]
        code='''for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++) {
 vec4 src=bank(x0,pixel+ivec2(kx-1,ky-1),output_bank/2);
 vec4 paired=(output_bank%2==0)?src.xxyy:src.zzww;
 value+=paired*texelFetch(weights,ivec2(ky*3+kx,output_bank),0);
}'''
        self.perceive=VectorPass(ctx,width,height,2*c,p,np.zeros(2*c,np.float32),'',code)
        if c%4==0 and cid%4==0:
            body=f'''vec4 features(ivec2 p,int b){{if(b<{c//4})return bank(x0,p,b);else if(b<{3*c//4})return bank(x1,p,b-{c//4});return bank(x2,p,b-{3*c//4});}}'''
        else:
            body=f'''float feature(ivec2 p,int k){{if(k<{c})return scalar_cell(x0,p,k);else if(k<{3*c})return scalar_cell(x1,p,k-{c});else if(k<{3*c+cid})return scalar_cell(x2,p,k-{3*c});return 0;}}
vec4 features(ivec2 p,int b){{int k=b*4;return vec4(feature(p,k),feature(p,k+1),feature(p,k+2),feature(p,k+3));}}'''
        code=f'''for(int b=0;b<{(3*c+cid+3)//4};b++)value+=product(ivec2(b*4,output_bank),features(pixel,b));
value=max(value,vec4(0));'''
        self.f1=VectorPass(ctx,width,height,hidden,blocks(weights['f1.weight']),weights['f1.bias'],body,code)
        code=f'''for(int b=0;b<{(hidden+3)//4};b++)value+=product(ivec2(b*4,output_bank),bank(x0,pixel,b));
value=bank(x1,pixel,output_bank)+value*texelFetch(mask,pixel,0).r;'''
        self.f2=VectorPass(ctx,width,height,c,blocks(weights['f2.weight']),weights['f2.bias'],'',code);self.passes=[self.seed,self.perceive,self.f1,self.f2]
    def install_readout(self,read_weight,read_bias,dictionary):
        if hasattr(self,'read_pass'):raise ValueError('Readout already installed')
        self.vocab=len(dictionary);self.read_pass=VectorConvPass(self.ctx,self.w,self.h,np.asarray(read_weight)[:,:,None,None],read_bias)
        self.raw_decode_pass=VectorConvPass(self.ctx,self.w,self.h,np.asarray(dictionary)[:,:,None,None],np.zeros(self.vocab,np.float32))
        self.decode_pass=VectorConvPass(self.ctx,self.w,self.h,np.asarray(dictionary)[:,:,None,None],np.zeros(self.vocab,np.float32),pad_output=True);self.passes.extend([self.read_pass,self.raw_decode_pass,self.decode_pass])
    def install_retina(self,weights):
        if hasattr(self,'retina_passes'):raise ValueError('Retina already installed')
        self.rgb=field_texture(self.ctx,3,self.h,self.w);self.retina_passes=[]
        for index,dilation in [(0,1),(2,2),(4,4),(6,8),(8,1)]:self.retina_passes.append(VectorConvPass(self.ctx,self.w,self.h,weights[f'retina.net.{index}.weight'],weights[f'retina.net.{index}.bias'],dilation=dilation,relu=index!=8))
        self.passes.extend(self.retina_passes)
