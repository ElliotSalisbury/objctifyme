import cv2
import moderngl
import numpy as np

ctx, prog = None,None

def init():
    global ctx, prog

    if ctx is None:
        # Context creation
        ctx = moderngl.create_standalone_context()
        ctx.enable(moderngl.DEPTH_TEST)

        prog = ctx.program(
            vertex_shader='''
                    #version 330

                    uniform mat4 proj;
                    uniform mat4 modelview;

                    in vec3 in_vert;
                    in vec3 in_color;

                    out vec3 v_color;

                    void main() {
                        v_color = in_color;
                        gl_Position = proj * modelview * vec4(in_vert, 1.0);
                    }
                ''',
            fragment_shader='''
                    #version 330

                    in vec3 v_color;

                    out vec3 f_color;

                    void main() {
                        f_color = v_color;
                    }
                ''',
        )

def render_mesh(im, mesh, pose):
    global ctx, prog

    init()

    modelview = np.array(pose.get_modelview())
    proj = np.array(pose.get_projection())
    proj[2, 2] = -0.001

    prog['proj'].value = tuple(proj.T.flatten().tolist())
    prog['modelview'].value = tuple(modelview.T.flatten().tolist())

    vertices = np.array(mesh.vertices)
    colors = (np.array(mesh.colors) / 255)
    indices = np.array(mesh.tvi)

    buffer = []
    for tri in indices:
        for i in tri:
            v = vertices[i]
            c = colors[i]
            buffer.append([v[0], v[1], v[2], c[2], c[1], c[0]])
    vertices = np.array(buffer)

    vbo = ctx.buffer(vertices.astype('f4').tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'in_vert', 'in_color')

    # Rendering
    fbo = ctx.simple_framebuffer((im.shape[1], im.shape[0]))
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    vao.render(moderngl.TRIANGLES)

    farr = np.fromstring(fbo.read(), dtype=np.uint8)
    farr = farr.reshape((im.shape[0], im.shape[1], 3), order='c')
    farr = cv2.flip(farr, 0)

    return farr
