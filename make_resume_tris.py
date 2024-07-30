import moderngl
import numpy as np
import glfw
from pyrr import Matrix44
from itertools import product
import random
from PIL import Image
import sys
import math as m

vertex_shader_code = '''
#version 330
in vec2 in_position;
in vec3 in_color;
out vec2 fragPos;
out vec3 vertColor;
void main() {
    vertColor = in_color;
    fragPos = in_position;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
'''

fragment_shader_code = '''
#version 330

uniform float swayPower;
uniform float twistPower;
uniform float twistFreq;
uniform float swayFreq;
uniform float xRibbonPos;

uniform vec3 leftBandColor; // vec3(0.94, 0.85, 0.45)
uniform vec3 rightBandColor;

in vec3 vertColor;
in vec2 fragPos;
out vec4 fragColor;
void main() {
    float signedTilt = sin((fragPos.y - 0.4) * twistFreq + 10);
    float baseTilt = abs(signedTilt);
    float phong = baseTilt + 0.3;

    float fuzzPower = 0.0000;
    float fuzz = (cos(fragPos.y * 1000) + 0.8 * cos(fragPos.y * 102342 + 12)) * fuzzPower;

    float depth = (baseTilt + 0.02) * twistPower + fuzz;
    float center = (sin(fragPos.y * swayFreq + 33) * swayPower) + xRibbonPos;
    float signedDist = fragPos.x - center;

    float onRibbon = step(abs(signedDist), depth);
    float onLeftStrip = step(-sign(signedTilt) * signedDist, -depth * 0.7);
    float onRightStrip = step(depth * 0.7, -sign(signedTilt) * signedDist);

    vec3 ribbonCol = mix(vec3(1, 0, 0), leftBandColor, onLeftStrip);
    ribbonCol = mix(ribbonCol, rightBandColor, onRightStrip);
    fragColor = vec4(mix(vertColor, ribbonCol * phong, onRibbon), 1.0);
}
'''

if not glfw.init():
    raise Exception("GLFW can't be initialized")
window = glfw.create_window(827, 1169, "Triangle Grid", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window can't be created")


glfw.make_context_current(window)
ctx = moderngl.create_context()

def accept_by_y_range_up(tri, y_min, y_max):
    center_y = ((tri[0] + tri[1] + tri[2]) / 3.0)[1]
    check_val = max(0, 1.0 - ((center_y - y_min) / (y_max - y_min)))
    return check_val >= random.random()

class TriangleGrid:
    def __init__(self, delaunay, shifted_grid):

        self.default_color = [253/255,250/255,212/255]
        self.on_color_choices = [[255/255,207/255,92/255], [255/255,227/255,160/255], [249/255,233/255,142/255]]
        self.on_color_shifts = {'min': [0.0,0.0,0.0], 'max': [0.2, 0.2, 0.2]}

        self.delaunay = delaunay
        self.shifted_grid = shifted_grid
        # REMEMBER THE THREE RULE!
        triangles = shifted_grid[delaunay.simplices]
        colors = np.repeat(np.repeat(np.array([[self.default_color]], dtype='f4'), len(triangles), axis=0), 3, axis=1)
        self.vertices = np.concatenate([triangles, colors], axis=-1).astype('f4')
        self.vbo = ctx.buffer(self.vertices.reshape(-1, self.vertices.shape[-1]).tobytes())

        self.mouse_drag_left_click = False
        self.mouse_dragging = False

    def get_vbo(self):
        return self.vbo

    def write_to_pos(self, x, y, col):
        tid = self.delaunay.find_simplex([x, y])
        self.vertices[tid,:,2:] = [col]        

    def commit(self):
        self.vbo.write(self.vertices.reshape(-1, self.vertices.shape[-1]).tobytes())

    def pick_color(self):
        if (not self.mouse_drag_left_click): return self.default_color
        chosen_color = random.choice(self.on_color_choices).copy()
        mi, ma = self.on_color_shifts['min'],self.on_color_shifts['max']
        for i in range(3): chosen_color[i] += random.uniform(mi[i], ma[i])
        return chosen_color

    def process_mouse_click(self, action, button, x, y):
        if action == glfw.PRESS:
            self.mouse_dragging = True
            self.mouse_drag_left_click = (button == glfw.MOUSE_BUTTON_LEFT)
        elif action == glfw.RELEASE:
            col = self.pick_color()
            self.write_to_pos(x, y, col)
            self.commit()
            self.mouse_dragging = False

    def process_mouse_move(self, x, y):
        if self.mouse_dragging:
            col = self.pick_color()
            self.write_to_pos(x, y, col)
            self.commit()

class RibbonParams:
    def __init__(self, parent_program):
        self.program = parent_program
        self.main_color = [1, 0, 0]
        self.right_ribbon = [1.0, 0, 0] # 0.55, 0.04]
        self.left_ribbon = [1.0, 0.55, 0.04]


        self.twist_freq = 2
        self.sway_freq = 4
        self.twist_power = 0.1
        self.sway_power = 0.2
        self.x_pos = -0.7

        #a = {'twistFreq': 2, 'twistPower': 0.1, 'swayFreq': 4.893325118400387, 'swayPower': 0.10868622764246555, 'xRibbonPos': -0.7825962264556438}
        #a = {'twistFreq': 2, 'twistPower': 0.06700844994629734, 'swayFreq': 4.893325118400387, 'swayPower': 0.052035090868594244, 'xRibbonPos': -0.8593212885898536}
        #self.twist_freq = a['twistFreq']
        #self.twist_power = a['twistPower']
        #self.sway_freq = a['swayFreq']
        #self.sway_power = a['swayPower']
        #self.x_pos = a['xRibbonPos']

        self.program['leftBandColor'].value = self.left_ribbon
        self.program['rightBandColor'].value = self.right_ribbon
        self.program['twistFreq'].value = self.twist_freq
        self.program['swayFreq'].value = self.sway_freq
        self.program['twistPower'].value = self.twist_power
        self.program['swayPower'].value = self.sway_power
        self.program['xRibbonPos'].value = self.x_pos

    def __str__(self):
        return str({'twistFreq': self.twist_freq, 'twistPower': self.twist_power,
                'swayFreq': self.sway_freq, 'swayPower': self.sway_power,
                'xRibbonPos': self.x_pos})

    def __repr__(self):
        return str({'twistFreq': self.twist_freq, 'twistPower': self.twist_power,
                'swayFreq': self.sway_freq, 'swayPower': self.sway_power,
                'xRibbonPos': self.x_pos})

    def process_input(self, key, action, delta_time):
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_1:
                self.twist_freq -= 0.9 * delta_time
            elif key == glfw.KEY_2:
                self.twist_freq += 0.9 * delta_time

            if key == glfw.KEY_3:
                self.twist_power -= 0.1 * delta_time
            elif key == glfw.KEY_4:
                self.twist_power += 0.1 * delta_time

            if key == glfw.KEY_5:
                self.sway_freq -= 0.9 * delta_time
            elif key == glfw.KEY_6:
                self.sway_freq += 0.9 * delta_time

            if key == glfw.KEY_7:
                self.sway_power -= 0.1 * delta_time
            elif key == glfw.KEY_8:
                self.sway_power += 0.1 * delta_time
   
            if key == glfw.KEY_9:
                self.x_pos -= 0.9 * delta_time
            elif key == glfw.KEY_0:
                self.x_pos += 0.9 * delta_time

            self.program['twistFreq'].value = self.twist_freq
            self.program['swayFreq'].value = self.sway_freq
            self.program['twistPower'].value = self.twist_power
            self.program['swayPower'].value = self.sway_power
            self.program['xRibbonPos'].value = self.x_pos
            print(str(self))


def make_rect_grid(min_pt, max_pt, num_x, num_y):
    x_min, y_min = min_pt
    x_max, y_max = max_pt
    x = np.linspace(x_min, x_max, num_x)
    y = np.linspace(y_min, y_max, num_y)
    x_grid, y_grid = np.meshgrid(x, y)
    return np.concatenate([x_grid[:,:,None], y_grid[:,:,None]], axis=-1, dtype='f4')

def shift_rect_grid(grid, num_shifts, col_size):
    if (num_shifts <= 1): return grid
    shift_size = col_size / (num_shifts)
    shifts = np.array([shift_size * (i % num_shifts) for i in range(grid.shape[0])], dtype='f4')
    grid[:,:,0] += shifts[:,None]
    return grid

from scipy.spatial import Delaunay
def get_delaunay_triangles(grid):
    pts = grid.reshape(-1, grid.shape[-1])
    tris_simps = Delaunay(pts)
    return TriangleGrid(tris_simps, pts)

def make_triangle_grid(num_rows, num_cols, num_shifts):
    col_size = 2 / num_cols
    row_size = 2 / num_rows
    grid = make_rect_grid([(-1) - col_size, (-1) - row_size], [1 + col_size, 1 + row_size], num_cols + 3, num_rows + 3)
    grid = shift_rect_grid(grid, num_shifts, col_size)
    return get_delaunay_triangles(grid)

rows = 40
h2w_ratio = (827 / 1169) # this is allegedly A4

tri_grid = make_triangle_grid(rows, round(rows * h2w_ratio), 2)
program = ctx.program(vertex_shader=vertex_shader_code,
                      fragment_shader=fragment_shader_code,)
vao = ctx.vertex_array(program, [
    (tri_grid.get_vbo(), '2f 3f', 'in_position', 'in_color')
])
ribbon = RibbonParams(program)


def screen_to_view_via_window(window):
    x, y = glfw.get_cursor_pos(window)
    width, height = glfw.get_window_size(window)

    x_view = (x / width) * 2.0 - 1.0
    y_view = (y / height) * 2.0 - 1.0
    return x_view, -y_view

def screen_to_view(x, y, window):
    width, height = glfw.get_window_size(window)

    x_view = (x / width) * 2.0 - 1.0
    y_view = (y / height) * 2.0 - 1.0
    return x_view, -y_view 

def mouse_button_callback(window, button, action, mods):
    global tri_grid
    x, y = screen_to_view_via_window(window)
    tri_grid.process_mouse_click(action, button, x, y)
    
def cursor_position_callback(window, x, y):
    global tri_grid
    x, y = screen_to_view(x, y, window)
    tri_grid.process_mouse_move(x, y)



def save_render_as_image(filename, width, height):
    framebuffer = ctx.framebuffer(
        color_attachments=[ctx.texture((width, height), 4)]
    )
    framebuffer.use()
    ctx.clear(1.0, 1.0, 1.0)
    vao.render(moderngl.TRIANGLES)
    data = framebuffer.read(components=4)
    image = Image.frombytes('RGBA', (width, height), data, 'raw', 'RGBA', 0, -1)
    image.save(filename)

    # Bind the default framebuffer
    ctx.screen.use()


from timeit import default_timer as timer
last_time = timer()
save_id = 0
def key_callback(window, key, scancode, action, mods):
    global save_id, out_fn, last_time
    this_time = timer()
    delta_time = float(this_time - last_time)
    ribbon.process_input(key, action, delta_time)

    if key == glfw.KEY_SPACE and action == glfw.PRESS:
        filename = f'{sys.argv[1]}_{save_id}.png'
        save_render_as_image(filename, 827*2, 1169*2)
        print("Saved image as", filename)
        save_id += 1

glfw.set_key_callback(window, key_callback)
glfw.set_mouse_button_callback(window, mouse_button_callback)
glfw.set_cursor_pos_callback(window, cursor_position_callback)






while not glfw.window_should_close(window):
    vao.render(moderngl.TRIANGLES)
    glfw.swap_buffers(window)
    glfw.poll_events()
    last_time = timer()

glfw.terminate()

