import cv2
import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *


def printOpenGLError():
    err = glGetError()
    if (err != GL_NO_ERROR):
        print('GLERROR: ', gluErrorString(err))
        # sys.exit()


class Shader(object):
    def initShader(self, vertex_shader_source, fragment_shader_source):
        # create program
        self.program = glCreateProgram()
        print('create program')
        printOpenGLError()

        # vertex shader
        print('compile vertex shader...')
        self.vs = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(self.vs, [vertex_shader_source])
        glCompileShader(self.vs)
        glAttachShader(self.program, self.vs)
        printOpenGLError()

        # fragment shader
        print('compile fragment shader...')
        self.fs = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(self.fs, [fragment_shader_source])
        glCompileShader(self.fs)
        glAttachShader(self.program, self.fs)
        printOpenGLError()

        print('link...')
        glLinkProgram(self.program)
        printOpenGLError()

    def begin(self):
        if glUseProgram(self.program):
            printOpenGLError()

    def end(self):
        glUseProgram(0)


shader = None
window_size = None
curr_window = None
def get_window(im):
    global curr_window, window_size, shader

    if curr_window is None:
        glfw.init()

    if curr_window is not None and im.shape[0] == window_size[0] and im.shape[1] == window_size[1]:
        return curr_window
    else:
        if curr_window is not None:
            glfw.destroy_window(curr_window)

        window_size = (im.shape[0], im.shape[1])
        glfw.window_hint(glfw.VISIBLE, False)
        curr_window = glfw.create_window(window_size[1], window_size[0], "", None, None)
        glfw.make_context_current(curr_window)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)

        shader = Shader()
        shader.initShader(
            '''
                void main()
                {
                    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
                    gl_FrontColor = gl_Color;
                }
            ''',
            '''
                void main()
                {
                    gl_FragColor = gl_Color;
                }
            ''')

    return curr_window

def render_mesh(im, mesh, pose):
    global shader

    offscreen_context = get_window(im)

    modelview = np.matrix(pose.get_modelview())
    proj = np.matrix(pose.get_projection())
    proj[2, 2] = -0.001

    glMatrixMode(GL_PROJECTION)
    glLoadMatrixd(proj.T)

    # view
    glMatrixMode(GL_MODELVIEW)
    glLoadMatrixd(modelview.T)

    # create buffers for mesh
    vertices = np.array(mesh.vertices).flatten().tolist()
    colors = (np.array(mesh.colors) / 255).flatten().tolist()
    indices = np.zeros((len(mesh.tvi), 6), dtype=np.int)
    indices[:, :3] = mesh.tvi
    indices[:, 3:] = mesh.tvi
    indices = indices.flatten().tolist()

    buffers = glGenBuffers(3)
    glBindBuffer(GL_ARRAY_BUFFER, buffers[0])
    glBufferData(GL_ARRAY_BUFFER,
                 len(vertices) * 4,  # byte size
                 (ctypes.c_float * len(vertices))(*vertices),
                 GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, buffers[1])
    glBufferData(GL_ARRAY_BUFFER,
                 len(colors) * 4,  # byte size
                 (ctypes.c_float * len(colors))(*colors),
                 GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2])
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 len(indices) * 4,  # byte size
                 (ctypes.c_uint * len(indices))(*indices),
                 GL_STATIC_DRAW)

    # glBindFramebuffer(GL_FRAMEBUFFER, FBO)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    shader.begin()
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, buffers[0])
    glVertexPointer(3, GL_FLOAT, 0, None)
    glBindBuffer(GL_ARRAY_BUFFER, buffers[1])
    glColorPointer(3, GL_FLOAT, 0, None)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2])
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)
    shader.end()

    frame = glReadPixels(0, 0, im.shape[1], im.shape[0], GL_BGR, type=GL_UNSIGNED_BYTE)
    farr = np.fromstring(frame, dtype=np.uint8)
    farr = farr.reshape((im.shape[0], im.shape[1], 3), order='c')
    farr = cv2.flip(farr, 0)

    # glBindFramebuffer(GL_FRAMEBUFFER, 0)

    glfw.swap_buffers(offscreen_context)

    return farr
