import argparse
from collections.abc import Callable, Sequence
import ctypes as c
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'  # nopep8

import pygame

from rlp import api, constants

def wrap_function(lib: c.PyDLL, funcname: str,
                  restype, argtypes) -> Callable:
    """Simplify wrapping generic ctypes functions"""
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func


class Puzzle:
    """ 
    Class encapsulating Simon Tatham's Portable Puzzles.
    """

    def __init__(self,
                 puzzle: str,
                 width: int = 512,
                 height: int = 512,
                 arg: str | None = None,
                 headless: bool = False):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        self._build_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        f"../puzzles/build/")
        self._lib = c.PyDLL(os.path.join(
            self._build_path, f"lib{puzzle}.so"))
        api.specific.set_api_structures(puzzle)
        self._puzzle_name = puzzle

        # Register C library functions as callable instance objects
        self._init_python = wrap_function(self._lib, "init_python", None, None)
        self._destroy_window = wrap_function(
            self._lib, "destroy_window", api.FRONTEND_PTR, None)

        self._new_window = wrap_function(self._lib, "new_window",
                                         api.FRONTEND_PTR,
                                         [c.py_object, c.c_int, c.c_int,
                                          c.c_char_p, c.c_int])

        get_py_calls = wrap_function(
            self._lib, "get_py_calls", api.PYCALLS_PTR, None)
        self.calls = get_py_calls()

        # Text
        draw_text_prototype = c.PYFUNCTYPE(None, api.PYCALLS_PTR,
                                           c.py_object, c.py_object,
                                           c.c_int, c.c_int,
                                           api.Colour, c.c_char_p)
        self._draw_text = draw_text_prototype(("c_draw_text", self._lib))
        # Rectangle
        draw_rect_prototype = c.PYFUNCTYPE(None, api.PYCALLS_PTR,
                                           c.py_object, c.c_int, c.c_int,
                                           c.c_int, c.c_int, api.Colour)
        self._draw_rect = draw_rect_prototype(("c_draw_rect", self._lib))
        # Circle
        draw_circle_prototype = c.PYFUNCTYPE(None, api.PYCALLS_PTR,
                                             c.py_object, c.c_int, c.c_int,
                                             c.c_int, api.Colour, api.Colour)
        self._draw_circle = draw_circle_prototype(("c_draw_circle", self._lib))
        # Line
        draw_line_prototype = c.PYFUNCTYPE(None, api.PYCALLS_PTR,
                                           c.py_object, c.c_int, c.c_int,
                                           c.c_int, c.c_int, api.Colour)
        self._draw_line = draw_line_prototype(("c_draw_line", self._lib))
        # Thick Line
        draw_thick_line_prototype = c.PYFUNCTYPE(None, api.PYCALLS_PTR,
                                                 c.py_object, c.c_float, c.c_float,
                                                 c.c_float, c.c_float, c.c_float,
                                                 api.Colour)
        self._draw_thick_line = draw_thick_line_prototype(
            ("c_draw_thick_line", self._lib))
        # Polygon
        draw_polygon_prototype = c.PYFUNCTYPE(None, api.PYCALLS_PTR,
                                              c.py_object, c.POINTER(c.c_int),
                                              c.c_int, api.Colour, api.Colour)
        self._draw_polygon = draw_polygon_prototype(
            ("c_draw_polygon", self._lib))

        # Clipping functions
        clip_prototype = c.PYFUNCTYPE(None, api.PYCALLS_PTR,
                                      c.py_object, c.c_int,
                                      c.c_int, c.c_int, c.c_int)
        self._clip = clip_prototype(("c_clip", self._lib))

        unclip_prototype = c.PYFUNCTYPE(None, api.PYCALLS_PTR,
                                        c.py_object)
        self._unclip = unclip_prototype(("c_unclip", self._lib))

        timer_func_prototype = c.PYFUNCTYPE(None, api.FRONTEND_PTR)
        self._timer_func = timer_func_prototype(("timer_func", self._lib))

        self._process_key = wrap_function(self._lib, "process_key",
                                          c.c_bool, [api.FRONTEND_PTR,
                                                     c.c_int, c.c_int,
                                                     c.c_int])

        self._game_status = wrap_function(self._lib, "game_status",
                                          c.c_int, [api.FRONTEND_PTR])

        new_game_prototype = c.PYFUNCTYPE(None, api.FRONTEND_PTR, c.c_char_p)
        self._new_game = new_game_prototype(("c_new_game", self._lib))

        # actual setup of window & instance variables
        pygame.init()
        self._init_python()

        if not headless:
            self.screen = pygame.display.set_mode(size=(width, height))
        self.surf = pygame.Surface((width, height))
        if arg:
            self.arg = arg.encode()
            self.fe = self._new_window(self.surf,
                                       self.surf.get_width(),
                                       self.surf.get_height(),
                                       self.arg, 2)
        else:
            self.fe = self._new_window(self.surf,
                                       self.surf.get_width(),
                                       self.surf.get_height(),
                                       None, 0)

        self._shift = 0
        self._ctrl = 0
        self._alt = False
        self._mouse_pressed = (False, False, False)

    def destroy(self):
        self._destroy_window(self.fe)

    def draw_text(self, x: int, y: int, fonttype: int, fontsize: int, align: int, colour: api.Colour, text: str):
        self._draw_text(self.calls, self.surf, x, y,
                        fonttype, fontsize, align, colour, text)

    def draw_rect(self, x: int, y: int, w: int, h: int,
                  colour: api.Colour):
        self._draw_rect(self.calls, self.surf, x, y, w, h, colour)

    def draw_circle(self, x: int, y: int, radius: int,
                    fillcolour: api.Colour, outlinecolour: api.Colour):
        self._draw_circle(self.calls, self.surf, x, y,
                          radius, fillcolour, outlinecolour)

    def draw_line(self, x1: int, y1: int, x2: int, y2: int,
                  colour: api.Colour):
        self._draw_line(self.calls, self.surf, x1, y1, x2, y2, colour)

    def draw_thick_line(self, thickness, x1: int, y1: int, x2: int, y2: int,
                        colour: api.Colour):
        self._draw_thick_line(self.calls, self.surf,
                              thickness, x1, y1, x2, y2, colour)

    def draw_polygon(self, points: Sequence[int],
                     fillcolour: api.Colour, outlinecolour: api.Colour):
        # cast "points" to a c_int array
        coords = (c.c_int*len(points))(*points)
        self._draw_polygon(self.calls, self.surf, coords,
                           len(points)//2, fillcolour, outlinecolour)

    def clip(self, x: int, y: int, w: int, h: int):
        self._clip(self.calls, self.surf, x, y, w, h)

    def unclip(self):
        self._unclip(self.calls, self.surf)

    def proceed_animation(self):
        self._timer_func(self.fe)
        if self.screen:
            self.screen.fill(0)
            self.screen.blit(self.surf, (0,0))
            pygame.display.flip()

    def process_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
            self.process_key_event(event)
        elif event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEBUTTONUP:
            self.process_button_event(event)
        elif event.type == pygame.MOUSEMOTION:
            self.process_motion_event()

    def process_key_event(self, event: pygame.event.Event):
        self.process_key(event.type, 
                         event.key, 
                         event.mod)
        
    def process_key(self, type: int, key: int, mod: int):
        self._shift = constants.MOD_SHFT if mod & pygame.KMOD_SHIFT else 0
        self._ctrl = constants.MOD_CTRL if mod & pygame.KMOD_CTRL else 0

        self._alt = True if mod & pygame.KMOD_ALT else False
        if type == pygame.KEYUP:
            return

        if key == pygame.K_UP:
            keyval = self._shift | self._ctrl | constants.CURSOR_UP
        elif key == pygame.K_KP8 or key == pygame.K_KP_8:
            keyval = constants.MOD_NUM_KEYPAD | ord('8')
        elif key == pygame.K_DOWN:
            keyval = self._shift | self._ctrl | constants.CURSOR_DOWN
        elif key == pygame.K_KP2 or key == pygame.K_KP_2:
            keyval = constants.MOD_NUM_KEYPAD | ord('2')
        elif key == pygame.K_LEFT:
            keyval = self._shift | self._ctrl | constants.CURSOR_LEFT
        elif key == pygame.K_KP4 or key == pygame.K_KP_4:
            keyval = constants.MOD_NUM_KEYPAD | ord('4')
        elif key == pygame.K_RIGHT:
            keyval = self._shift | self._ctrl | constants.CURSOR_RIGHT
        elif key == pygame.K_KP6 or key == pygame.K_KP_6:
            keyval = constants.MOD_NUM_KEYPAD | ord('6')
        elif key == pygame.K_KP7 or key == pygame.K_KP_7:
            keyval = constants.MOD_NUM_KEYPAD | ord('7')
        elif key == pygame.K_KP1 or key == pygame.K_KP_1:
            keyval = constants.MOD_NUM_KEYPAD | ord('1')
        elif key == pygame.K_KP9 or key == pygame.K_KP_9:
            keyval = constants.MOD_NUM_KEYPAD | ord('9')
        elif key == pygame.K_KP3 or key == pygame.K_KP_3:
            keyval = constants.MOD_NUM_KEYPAD | ord('3')
        elif key == pygame.K_KP0 or key == pygame.K_KP_0:
            keyval = constants.MOD_NUM_KEYPAD | ord('0')
        elif key == pygame.K_KP5 or key == pygame.K_KP_5:
            keyval = constants.MOD_NUM_KEYPAD | ord('5')
        elif key == pygame.K_BACKSPACE or key == pygame.K_DELETE or key == pygame.K_KP_PERIOD:
            keyval = ord('\177')
        elif key == pygame.K_RETURN or key == pygame.K_KP_ENTER:
            keyval = constants.CURSOR_SELECT
        elif key == pygame.K_SPACE:
            keyval = constants.CURSOR_SELECT2
        elif key == pygame.K_z:
            keyval = constants.UI_REDO
        elif len(pygame.key.name(key)) == 1:
            keyval = ord(pygame.key.name(key))
        else:
            keyval = -1

        self._process_key(self.fe, 0, 0, keyval)
        

    def process_button_event(self, event: pygame.event.Event):
        if (event.type != pygame.MOUSEBUTTONDOWN
                and event.type != pygame.MOUSEBUTTONUP):
            return True

        # detect which button was actually changed
        new_mouse_pressed = pygame.mouse.get_pressed(num_buttons=3)[0:3]
        changed_button = -1
        for i in range(len(new_mouse_pressed)):
            if new_mouse_pressed[i] ^ self._mouse_pressed[i]:
                changed_button = i
        self._mouse_pressed = new_mouse_pressed

        # assign correct value for key processing
        if changed_button == 1 or self._shift:
            button = constants.MIDDLE_BUTTON
        elif changed_button == 2 or self._alt:
            button = constants.RIGHT_BUTTON
        elif changed_button == 0:
            button = constants.LEFT_BUTTON
        else:
            return False

        if event.type == pygame.MOUSEBUTTONUP and button >= constants.LEFT_BUTTON:
            button += constants.LEFT_RELEASE - constants.LEFT_BUTTON
        x, y = pygame.mouse.get_pos()
        self._process_key(self.fe, x, y, button)

    def process_motion_event(self):
        if self._mouse_pressed[1] or self._shift:
            button = constants.MIDDLE_DRAG
        elif self._mouse_pressed[0]:
            button = constants.LEFT_DRAG
        elif self._mouse_pressed[2] or self._alt:
            button = constants.RIGHT_DRAG
        else:
            return False

        x, y = pygame.mouse.get_pos()
        self._process_key(self.fe, x, y, button)

    def game_status(self):
        return self._game_status(self.fe)
    
    def get_puzzle_state(self):
        return api.specific.make_puzzle_state(self._puzzle_name, 
            self.fe.contents.me.contents.states[self.fe.contents.me.contents.nstates-1].state.contents)

    def new_game(self):
        self._new_game(self.fe, self.arg)

def py_add_font(fonttype: int, fontsize: int):
    fonttype_str = "monospace" if fonttype == constants.FONT_FIXED else "sans"
    font = pygame.font.SysFont(fonttype_str, fontsize, bold=True)
    return font


def py_string_metrics(font: pygame.font.Font, text: str):
    width, height = font.size(text)
    return [width, height]


def py_draw_text(surf: pygame.Surface, x: int, y: int, font: pygame.font.Font,
                 colour: tuple[int, int, int], text: str):
    text_surf = font.render(text, True, colour)
    surf.blit(text_surf, (x, y))


def py_draw_rect(surf: pygame.Surface, x: int, y: int, w: int, h: int,
                 colour: tuple[int, int, int]):
    pygame.draw.rect(surface=surf, color=colour, rect=[x, y, w, h], width=0)


def py_draw_circle(surf: pygame.Surface, x: int, y: int, radius: int,
                   fillcolour: tuple[int, int, int],
                   outlinecolour: tuple[int, int, int]):
    if fillcolour != (-1, -1, -1):
        pygame.draw.circle(surface=surf, color=fillcolour,
                           center=(x+0.5, y+0.5), radius=radius, width=0)
    pygame.draw.circle(surface=surf, color=outlinecolour,
                       center=(x+0.5, y+0.5), radius=radius, width=1)


def py_draw_line(surf: pygame.Surface, x1: int, y1: int, x2: int, y2: int,
                 colour: tuple[int, int, int]):
    pygame.draw.line(surface=surf, color=colour, start_pos=(
        x1, y1), end_pos=(x2, y2), width=1)


def py_draw_thick_line(surf: pygame.Surface, thickness: float,
                       x1: int, y1: int, x2: int, y2: int,
                       colour: tuple[int, int, int]):
    if thickness < 1.0:
        thickness = 1.0

    p1 = pygame.math.Vector2(x1, y1)
    p2 = pygame.math.Vector2(x2, y2)
    line = (p2 - p1).normalize()
    orth_line = pygame.math.Vector2(-line.y, line.x) * (thickness // 2)
    points = [p1 - orth_line, p1 + orth_line,
              p2 + orth_line, p2 - orth_line]
    pygame.draw.polygon(surf, colour, points)


def py_draw_polygon(surf: pygame.Surface, points: Sequence[Sequence[float]],
                    fillcolour: tuple[int, int, int],
                    outlinecolour: tuple[int, int, int]):
    if fillcolour != (-1, -1, -1):
        pygame.draw.polygon(surf, fillcolour, points, 0)
    pygame.draw.polygon(surf, outlinecolour, points, 1)


def py_clip(surf: pygame.Surface, x: int, y: int, w: int, h: int):
    surf.set_clip(pygame.Rect(x, y, w, h))


def py_unclip(surf: pygame.Surface):
    surf.set_clip(None)


def py_blitter_new(w: int, h: int):
    return pygame.Surface((w, h))


def py_blitter_free(surf: pygame.Surface):
    del surf


def py_blitter_save(surf: pygame.Surface, blitter_surf: pygame.Surface,
                    x: int, y: int):
    blit_area = pygame.Rect(x, y,
                            blitter_surf.get_width(),
                            blitter_surf.get_height())
    blitter_surf.blit(surf, (0, 0), blit_area)


def py_blitter_load(surf: pygame.Surface, blitter_surf: pygame.Surface,
                    x: int, y: int):
    surf.blit(blitter_surf, (x, y))


def py_set_window_title(title: str):
    pygame.display.set_caption(title)


def py_set_window_icon(icon_prefix: str):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        f"../puzzles/build/icons/{icon_prefix}-96d24.png")
    icon = pygame.image.load(path)
    pygame.display.set_icon(icon)


def make_puzzle_parser():
    parser = argparse.ArgumentParser(
        description="Run one of Simon Tatham's Portable Puzzles.")
    parser.add_argument('-p', '--puzzle', metavar='PUZZLE_NAME', type=str,
                        help="Choose which puzzle to play.", default='net')
    parser.add_argument('-s', '--size', type=int, metavar=('WIDTH', 'HEIGHT'),
                        help="Screen width & height in pixels.", default=(880, 880), nargs=2)
    parser.add_argument('-a', '--arg', type=str, metavar='PARAMETERS',
                        help="Choose the parameters for a non-randomly generated puzzle.\n Format: 'params', 'params:description' or 'params#seed'\n Example:'8x8de#12' for a 8x8 easy difficulty hard Slant puzzle with seed 12.")
    return parser
