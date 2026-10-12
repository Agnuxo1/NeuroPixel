"""Complete scalar SSBO graphics inference for the separate stable-field candidate.

This is a correctness reference, not an optimized acceleration claim. The call
includes buffer creation, transfers and readback. Physical GPU parity is mandatory.
"""
from __future__ import annotations

import numpy as np

from .stable_field import StableField, recurrent_bound


UPDATE_SHADER = r'''#version 430
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer S { float state[]; };
layout(std430,binding=1) readonly buffer U { float input_field[]; };
layout(std430,binding=2) readonly buffer K { float kernel[]; };
layout(std430,binding=3) readonly buffer M { float mix[]; };
layout(std430,binding=4) readonly buffer D { float drive[]; };
layout(std430,binding=5) readonly buffer B { float bias[]; };
layout(std430,binding=6) writeonly buffer O { float next_state[]; };
uniform int batches, channels, inputs, height, width;
uniform float alpha;
void main() {
    uint index=gl_GlobalInvocationID.x;
    uint pixels=uint(height*width);
    if (index>=uint(batches*channels)*pixels) return;
    int x=int(index%uint(width));
    int y=int((index/uint(width))%uint(height));
    int output_channel=int((index/pixels)%uint(channels));
    int batch=int(index/(pixels*uint(channels)));
    float z=bias[output_channel];
    for (int c=0;c<channels;c++) {
        float perception=0.0;
        for (int ky=0;ky<3;ky++) for (int kx=0;kx<3;kx++) {
            int yy=y+ky-1, xx=x+kx-1;
            if (yy>=0 && yy<height && xx>=0 && xx<width)
                perception+=kernel[c*9+ky*3+kx]*state[((batch*channels+c)*height+yy)*width+xx];
        }
        z+=mix[output_channel*channels+c]*perception;
    }
    for (int c=0;c<inputs;c++)
        z+=drive[output_channel*inputs+c]*input_field[((batch*inputs+c)*height+y)*width+x];
    next_state[index]=(1.0-alpha)*state[index]+alpha*tanh(z);
}
'''

DECODE_SHADER = r'''#version 430
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer S { float state[]; };
layout(std430,binding=1) writeonly buffer O { float logits[]; };
layout(std430,binding=2) readonly buffer R { float decoder[]; };
layout(std430,binding=3) readonly buffer B { float output_bias[]; };
uniform int batches, channels, classes, height, width;
void main() {
    uint index=gl_GlobalInvocationID.x;
    if (index>=uint(batches*classes)) return;
    int batch=int(index)/classes, label=int(index)%classes;
    float value=output_bias[label];
    for (int c=0;c<channels;c++)
        value+=decoder[label*channels+c]*state[((batch*channels+c)*height+height/2)*width+width/2];
    logits[index]=value;
}
'''


class GraphicsField:
    """Manage a standalone GL context and resident learned coefficients."""
    def __init__(self, model: StableField, *, backend: str | None = None):
        import moderngl
        self.context = moderngl.create_standalone_context(require=430, **({"backend": backend} if backend else {}))
        self.model = model
        self.update_shader = self.context.compute_shader(UPDATE_SHADER)
        self.decode_shader = self.context.compute_shader(DECODE_SHADER)
        self.parameters = {name: np.ascontiguousarray(value, dtype="f4") for name, value in model.parameters.items()}
        self.weights = {name: self.context.buffer(value.tobytes()) for name, value in self.parameters.items()}
        self.metadata = dict(renderer=self.context.info["GL_RENDERER"], vendor=self.context.info["GL_VENDOR"],
                             gl_version=self.context.info["GL_VERSION"], physical_gpu_verified=False,
                             rounded_weight_recurrent_bound=str(recurrent_bound(self.parameters["mix"], self.parameters["kernels"])),
                             scope="Allocation and transfer inclusive correctness reference")

    def forward(self, input_field: np.ndarray, *, steps: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Execute every update and decoder on graphics buffers, then read outputs."""
        inputs = np.ascontiguousarray(input_field, dtype="f4")
        depth = self.model.steps if steps is None else steps
        if (inputs.ndim != 4 or len(inputs) < 1 or not np.isfinite(inputs).all()
                or inputs.shape[1] != self.parameters["drive"].shape[1]
                or not isinstance(depth, int) or depth < 0):
            raise ValueError("Invalid graphics input or depth")
        batch, _, height, width = inputs.shape
        channels, classes = len(self.parameters["mix"]), len(self.parameters["decoder"])
        shape = (batch, channels, height, width)
        state_bytes = np.zeros(shape, dtype="f4").tobytes()
        buffers = []
        try:
            current = self.context.buffer(state_bytes); buffers.append(current)
            following = self.context.buffer(state_bytes); buffers.append(following)
            image = self.context.buffer(inputs.tobytes()); buffers.append(image)
            logits_buffer = self.context.buffer(reserve=batch*classes*4); buffers.append(logits_buffer)
            for name, value in dict(batches=batch, channels=channels, inputs=inputs.shape[1], height=height, width=width,
                                    alpha=float(np.float32(self.model.alpha))).items():
                self.update_shader[name].value = value
            image.bind_to_storage_buffer(1)
            for binding, name in enumerate(("kernels", "mix", "drive", "bias"), start=2):
                self.weights[name].bind_to_storage_buffer(binding)
            for _ in range(depth):
                current.bind_to_storage_buffer(0)
                following.bind_to_storage_buffer(6)
                self.update_shader.run(group_x=(int(np.prod(shape))+63)//64)
                self.context.memory_barrier()
                current, following = following, current
            current.bind_to_storage_buffer(0)
            logits_buffer.bind_to_storage_buffer(1)
            self.weights["decoder"].bind_to_storage_buffer(2)
            self.weights["output_bias"].bind_to_storage_buffer(3)
            for name, value in dict(batches=batch, channels=channels, classes=classes, height=height, width=width).items():
                self.decode_shader[name].value = value
            self.decode_shader.run(group_x=(batch*classes+63)//64)
            self.context.memory_barrier()
            self.context.finish()
            state = np.frombuffer(current.read(), dtype="f4").reshape(shape).copy()
            logits = np.frombuffer(logits_buffer.read(), dtype="f4").reshape(batch, classes).copy()
            return logits, state
        finally:
            for buffer in reversed(buffers):
                buffer.release()

    def close(self) -> None:
        """Release every resident resource; call exactly once after inference."""
        for buffer in self.weights.values():
            buffer.release()
        self.update_shader.release()
        self.decode_shader.release()
        self.context.release()
