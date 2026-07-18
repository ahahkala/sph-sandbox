// Small WebGL2 helpers: shader compilation, render targets, data textures.

export function createProgram(gl, vsSrc, fsSrc) {
  const compile = (type, src) => {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error("Shader compile error:\n" + gl.getShaderInfoLog(sh) + "\n--- source ---\n" + src);
    }
    return sh;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error("Program link error: " + gl.getProgramInfoLog(prog));
  }
  return prog;
}

export function getUniforms(gl, prog) {
  const uniforms = {};
  const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const info = gl.getActiveUniform(prog, i);
    uniforms[info.name.replace("[0]", "")] = gl.getUniformLocation(prog, info.name);
  }
  return uniforms;
}

export function createTexture(gl, width, height, internalFormat, format, type, filter) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

// Offscreen render target. Returns null if the format can't be rendered to.
export function createTarget(gl, width, height, internalFormat, format, type, filter, depthStencil = false) {
  const tex = createTexture(gl, width, height, internalFormat, format, type, filter);
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  let rb = null;
  if (depthStencil) {
    rb = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH24_STENCIL8, width, height);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, rb);
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
  }
  const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  if (!ok) {
    gl.deleteTexture(tex);
    gl.deleteFramebuffer(fbo);
    if (rb) gl.deleteRenderbuffer(rb);
    return null;
  }
  return { tex, fbo, rb, width, height };
}

export function deleteTarget(gl, target) {
  if (!target) return;
  gl.deleteTexture(target.tex);
  gl.deleteFramebuffer(target.fbo);
  if (target.rb) gl.deleteRenderbuffer(target.rb);
}

// Growable RGBA32F data texture holding one obstacle per texel in 256-wide
// rows (xy = center, z = radius) — matches the texelFetch layout in the
// collision and overlay shaders. Capacity doubles on demand; no count limit.
export class ObstacleTexture {
  constructor(gl) {
    this.gl = gl;
    this.cap = 256; // texels; always a multiple of the 256-texel row width
    this.tex = createTexture(gl, 256, 1, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST);
    this.data = new Float32Array(this.cap * 4);
  }

  // Uploads on the given texture unit and leaves the texture bound there,
  // ready for a sampler uniform pointing at that unit.
  upload(obstacles, unit) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    if (obstacles.length > this.cap) {
      while (this.cap < obstacles.length) this.cap *= 2;
      gl.deleteTexture(this.tex);
      this.tex = createTexture(gl, 256, this.cap / 256, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST);
      this.data = new Float32Array(this.cap * 4);
    }
    for (let k = 0; k < obstacles.length; k++) {
      this.data[k * 4] = obstacles[k].x;
      this.data[k * 4 + 1] = obstacles[k].y;
      this.data[k * 4 + 2] = obstacles[k].r;
    }
    const rows = Math.max(1, Math.ceil(obstacles.length / 256));
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 256, rows, gl.RGBA, gl.FLOAT,
      this.data.subarray(0, rows * 256 * 4));
  }
}

// Attach an existing depth-stencil renderbuffer to a target's framebuffer
// (used to share one stencil buffer between the two grid-slot targets).
export function attachDepthStencil(gl, target, rb) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, rb);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  target.rb = null; // not owned
}
