"""OpenGL 4.6 graphics-compute backend: SSBO weights and shared pixel tiles.

Separate candidate. It processes float images with compute shaders rather than
fragment rasterization; benchmark labels retain that distinction.
"""
import hashlib,time
import ctypes,ctypes.util,sys
import numpy as np
from .nca_gl import GLNCA,field_texture
from .nca_gl_vector import blocks

def shared_memory_limit(ctx):
    """Query the active driver's actual limit when ModernGL omits this key."""
    if 'GL_MAX_COMPUTE_SHARED_MEMORY_SIZE' in ctx.info:return ctx.info['GL_MAX_COMPUTE_SHARED_MEMORY_SIZE']
    library=ctypes.WinDLL('opengl32') if sys.platform=='win32' else ctypes.CDLL(ctypes.util.find_library('GL'))
    query=library.glGetIntegerv;query.argtypes=[ctypes.c_uint,ctypes.POINTER(ctypes.c_int)];query.restype=None
    value=ctypes.c_int();query(0x8262,ctypes.byref(value))
    if value.value<32768:raise ValueError('Cannot verify actual compute shared-memory limit')
    return value.value

class ComputePass:
    def __init__(self,ctx,w,h,channels,weight,bias,declarations,load,code):
        self.ctx,self.w,self.h,self.channels=ctx,w,h,channels
        self.weights=ctx.buffer(np.asarray(weight,np.float32).reshape(-1,4).tobytes());b=np.zeros(((channels+3)//4)*4,np.float32);b[:channels]=bias;self.bias=ctx.buffer(b.tobytes())
        self.texture=field_texture(ctx,channels,h,w);self.fbo=ctx.framebuffer([self.texture]);ob=(channels+3)//4
        source=f'''#version 460
layout(local_size_x=4,local_size_y=4,local_size_z=4)in;
layout(rgba32f,binding=0)uniform readonly image2D x0;
layout(rgba32f,binding=1)uniform readonly image2D x1;
layout(rgba32f,binding=2)uniform readonly image2D x2;
layout(r32f,binding=3)uniform readonly image2D mask;
layout(rgba32f,binding=6)uniform writeonly image2D target;
layout(std430,binding=4)readonly buffer Coefficients{{vec4 coefficients[];}};
layout(std430,binding=5)readonly buffer BiasValues{{vec4 biases[];}};
#define WIDTH {w}
#define HEIGHT {h}
vec4 product(int k,vec4 v){{return vec4(dot(coefficients[k],v),dot(coefficients[k+1],v),dot(coefficients[k+2],v),dot(coefficients[k+3],v));}}
{declarations}
void main(){{
ivec2 pixel=ivec2(gl_GlobalInvocationID.xy);int output_bank=int(gl_GlobalInvocationID.z);
int local_pixel=int(gl_LocalInvocationID.y)*4+int(gl_LocalInvocationID.x);
{load}
barrier();
if(pixel.x>=WIDTH||pixel.y>=HEIGHT||output_bank>={ob})return;
vec4 value=biases[output_bank];
{code}
imageStore(target,ivec2(pixel.x,pixel.y+output_bank*HEIGHT),value);
}}
'''
        # Keep explicit image formats visible to drivers that reject opaque
        # image parameters (NVIDIA image2D_bindless overload ambiguity).
        helpers=''
        for i in range(3):
            helpers+=f'\nvec4 bank{i}(ivec2 p,int b){{if(p.x<0||p.x>=WIDTH||p.y<0||p.y>=HEIGHT)return vec4(0);return imageLoad(x{i},ivec2(p.x,p.y+b*HEIGHT));}}\nfloat cell{i}(ivec2 p,int c){{return bank{i}(p,c/4)[c%4];}}\n'
            source=source.replace(f'bank(x{i},',f'bank{i}(').replace(f'cell(x{i},',f'cell{i}(')
        source=source.replace('vec4 product(',helpers+'\nvec4 product(')
        start=time.perf_counter();self.program=ctx.compute_shader(source);self.compile_seconds=time.perf_counter()-start;self.shader_sha256=hashlib.sha256(source.encode()).hexdigest()
    def draw(self,x0,x1=None,x2=None,mask=None,target=None):
        for name,tex,unit in [('x0',x0,0),('x1',x1,1),('x2',x2,2),('mask',mask,3)]:
            if name in self.program:
                if tex is None:raise ValueError('Required compute image '+name)
                tex.bind_to_image(unit,read=True,write=False)
        out=self.texture if target is None else target.color_attachments[0];out.bind_to_image(6,read=False,write=True);self.weights.bind_to_storage_buffer(4);self.bias.bind_to_storage_buffer(5)
        self.program.run(group_x=(self.w+3)//4,group_y=(self.h+3)//4,group_z=((self.channels+3)//4+3)//4);self.ctx.memory_barrier();return out
    def release(self):
        for obj in [self.program,self.fbo,self.texture,self.weights,self.bias]:obj.release()

class ComputeConvPass(ComputePass):
    def __init__(self,ctx,w,h,weight,bias,dilation=1,relu=False,present=False,residual=False,pad_output=False):
        a=np.asarray(weight,np.float32);oc,ic,k,_=a.shape;ib=(ic+3)//4;r=(k//2)*dilation;span=4+2*r;area=span*span;count=ib*area
        declarations=f'shared vec4 tile[{count}];'
        load=f'''for(int i=int(gl_LocalInvocationIndex);i<{count};i+=64){{int b=i/{area};int q=i%{area};ivec2 p=ivec2(gl_WorkGroupID.xy)*4+ivec2(q%{span}-{r},q/{span}-{r});tile[i]=bank(x0,p,b);}}'''
        code=f'''for(int b=0;b<{ib};b++)for(int ky=0;ky<{k};ky++)for(int kx=0;kx<{k};kx++){{
 int tx=int(gl_LocalInvocationID.x)+kx*{dilation};int ty=int(gl_LocalInvocationID.y)+ky*{dilation};
 int index=output_bank*{ib*k*k*4}+((b*{k}+ky)*{k}+kx)*4;
 value+=product(index,tile[b*{area}+ty*{span}+tx]);
}}'''+('value=max(value,vec4(0));' if relu else '')+('value*=imageLoad(mask,pixel).r;' if present else '')+('value=bank(x1,pixel,output_bank)+value*imageLoad(mask,pixel).r;' if residual else '')+('if(output_bank==0)value.x=-10000.0;' if pad_output else '')
        if count*16>shared_memory_limit(ctx):raise ValueError('Shared tile exceeds actual hardware')
        super().__init__(ctx,w,h,oc,blocks(a),bias,declarations,load,code)

class ComputeGLNCA(GLNCA):
    def __init__(self,ctx,width,height,weights):
        if ctx.version_code<460:raise ValueError('Actual OpenGL4.6 required')
        self.ctx,self.w,self.h,self.weights=ctx,width,height,weights;f1=np.asarray(weights['f1.weight']);f2=np.asarray(weights['f2.weight']);self.c=f2.shape[0];self.hidden=f2.shape[1];self.cid=f1.shape[1]-3*self.c;c=self.c;cid=self.cid
        self.ids=field_texture(ctx,cid,height,width);self.active_ids=self.ids;self.present=ctx.texture((width,height),1,dtype='f4');self.fire=ctx.texture((width,height),1,np.ones((height,width),np.float32).tobytes(),dtype='f4');self.fire_changed=False
        self.states=[field_texture(ctx,c,height,width) for _ in range(2)];self.fbos=[ctx.framebuffer([t]) for t in self.states];self.current=0
        self.seed=ComputeConvPass(ctx,width,height,weights['seed.weight'],weights['seed.bias'],present=True)
        cb=(c+3)//4;span=6;count=cb*36;decl=f'shared vec4 state_tile[{count}];';load=f'''for(int i=int(gl_LocalInvocationIndex);i<{count};i+=64){{int b=i/36;int q=i%36;ivec2 p=ivec2(gl_WorkGroupID.xy)*4+ivec2(q%6-1,q/6-1);state_tile[i]=bank(x0,p,b);}}'''
        kernels=np.asarray(weights['perceive.weight'],np.float32).reshape(2*c,9);coeff=np.zeros(((2*c+3)//4,9,4),np.float32)
        for out_c in range(2*c):coeff[out_c//4,:,out_c%4]=kernels[out_c]
        code='''for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++){int q=(int(gl_LocalInvocationID.y)+ky)*6+int(gl_LocalInvocationID.x)+kx;vec4 src=state_tile[(output_bank/2)*36+q];vec4 paired=(output_bank%2==0)?src.xxyy:src.zzww;value+=paired*coefficients[output_bank*9+ky*3+kx];}'''
        self.perceive=ComputePass(ctx,width,height,2*c,coeff,np.zeros(2*c,np.float32),decl,load,code)
        ib=(3*c+cid+3)//4
        if c%4==0 and cid%4==0:feature=f'''vec4 feature(ivec2 p,int b){{if(b<{c//4})return bank(x0,p,b);else if(b<{3*c//4})return bank(x1,p,b-{c//4});return bank(x2,p,b-{3*c//4});}}'''
        else:feature=f'''float feature_scalar(ivec2 p,int k){{if(k<{c})return cell(x0,p,k);else if(k<{3*c})return cell(x1,p,k-{c});else if(k<{3*c+cid})return cell(x2,p,k-{3*c});return 0;}}vec4 feature(ivec2 p,int b){{int k=b*4;return vec4(feature_scalar(p,k),feature_scalar(p,k+1),feature_scalar(p,k+2),feature_scalar(p,k+3));}}'''
        decl=feature+f'\nshared vec4 features_tile[{ib*16}];';load=f'''for(int i=int(gl_LocalInvocationIndex);i<{ib*16};i+=64){{int b=i/16;int q=i%16;ivec2 p=ivec2(gl_WorkGroupID.xy)*4+ivec2(q%4,q/4);features_tile[i]=feature(p,b);}}''';code=f'''for(int b=0;b<{ib};b++)value+=product(output_bank*{ib*4}+b*4,features_tile[b*16+local_pixel]);value=max(value,vec4(0));'''
        self.f1=ComputePass(ctx,width,height,self.hidden,blocks(weights['f1.weight']),weights['f1.bias'],decl,load,code)
        self.f2=ComputeConvPass(ctx,width,height,weights['f2.weight'],weights['f2.bias'],residual=True);self.passes=[self.seed,self.perceive,self.f1,self.f2]
    def install_readout(self,read_weight,read_bias,dictionary):
        if hasattr(self,'read_pass'):raise ValueError('Readout already installed')
        self.vocab=len(dictionary);self.read_pass=ComputeConvPass(self.ctx,self.w,self.h,np.asarray(read_weight)[:,:,None,None],read_bias);self.raw_decode_pass=ComputeConvPass(self.ctx,self.w,self.h,np.asarray(dictionary)[:,:,None,None],np.zeros(self.vocab,np.float32));self.decode_pass=ComputeConvPass(self.ctx,self.w,self.h,np.asarray(dictionary)[:,:,None,None],np.zeros(self.vocab,np.float32),pad_output=True);self.passes.extend([self.read_pass,self.raw_decode_pass,self.decode_pass])
    def install_retina(self,weights):
        if hasattr(self,'retina_passes'):raise ValueError('Retina already installed')
        self.rgb=field_texture(self.ctx,3,self.h,self.w);self.retina_passes=[ComputeConvPass(self.ctx,self.w,self.h,weights[f'retina.net.{i}.weight'],weights[f'retina.net.{i}.bias'],dilation=d,relu=i!=8) for i,d in [(0,1),(2,2),(4,4),(6,8),(8,1)]];self.passes.extend(self.retina_passes)
    def metadata(self):
        m=super().metadata();m.update(execution='OpenGL4.6 compute shaders',coefficient_storage='SSBO vec4',workgroup=[4,4,4],shared_pixel_tiles=True);return m
