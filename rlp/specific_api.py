from collections.abc import Callable
import ctypes as c

from pygame import locals as pl

CT_PTR = c.POINTER
CT_FUNC = c.CFUNCTYPE
CT_UCHAR = c.c_ubyte
CT_UCHAR_PTR = CT_PTR(CT_UCHAR)
CT_SCHAR = c.c_byte
CT_SCHAR_PTR = CT_PTR(CT_SCHAR)
CT_USHORT_PTR = CT_PTR(c.c_ushort)
CT_INT_PTR = CT_PTR(c.c_int)
CT_UINT_PTR = CT_PTR(c.c_uint)
CT_LONG_PTR = CT_PTR(c.c_long)
CT_ULONG_PTR = CT_PTR(c.c_ulong)
CT_BOOL_PTR = CT_PTR(c.c_bool)
# As defined in latin.h
Digit = CT_UCHAR
CT_DIGIT_PTR = CT_PTR(Digit)


class Midend(c.Structure):
    pass


class Frontend(c.Structure):
    pass


class Blitter(c.Structure):
    pass


class GameParams(c.Structure):
    pass


class GameUi(c.Structure):
    pass


class GameDrawState(c.Structure):
    pass


class GameState(c.Structure):
    pass


class SolverState(c.Structure):
    pass


class RandomState(c.Structure):
    pass


class Grid(c.Structure):
    '''As defined in grid.h'''
    def _as_dict(self):
        faces_ptr = c.cast(self.faces, c.c_void_p).value
        edges_ptr = c.cast(self.edges, c.c_void_p).value
        dots_ptr = c.cast(self.dots, c.c_void_p).value
        return {
            'num_faces': self.num_faces,
            'faces': _get_substruct_array_elements(self.faces, self.num_faces, edges_ptr, dots_ptr),
            'num_edges': self.num_edges,
            'edges': _get_substruct_array_elements(self.edges, self.num_edges, dots_ptr, faces_ptr),
            'num_dots': self.num_dots,
            'dots': _get_substruct_array_elements(self.dots, self.num_dots, edges_ptr, faces_ptr),
            'lowest_x': self.lowest_x,
            'lowest_y': self.lowest_y,
            'highest_x': self.highest_x,
            'highest_y': self.highest_y,
            'tilesize': self.tilesize,
        }



MIDEND_PTR = CT_PTR(Midend)
FRONTEND_PTR = CT_PTR(Frontend)
BLITTER_PTR = CT_PTR(Blitter)
GAMEPARAMS_PTR = CT_PTR(GameParams)
GAMEUI_PTR = CT_PTR(GameUi)
GAMEDRAWSTATE_PTR = CT_PTR(GameDrawState)
GAMESTATE_PTR = CT_PTR(GameState)
SOLVERSTATE_PTR = CT_PTR(SolverState)
RANDOMSTATE_PTR = CT_PTR(RandomState)
GRID_PTR = CT_PTR(Grid)

# for converting gamestates into dicts
def _get_array_elements(pointer, len):
    return [e for e in pointer[:len]]
def _get_substruct_array_elements(pointer, len, *arg):
    return [e._as_dict(*arg) for e in pointer[:len]]
def _get_substruct(pointer, *arg):
    return pointer[0]._as_dict(*arg)


def set_grid_structure() -> None:

    class GridFace(c.Structure):
        def _as_dict(self, edges_ptr, dots_ptr):
            return {
                'order': self.order,
                'edges_indices': [(c.cast(self.edges[i], c.c_void_p).value-edges_ptr)//c.sizeof(GridEdge) for i in range(self.order)],
                'dots_indices': [(c.cast(self.dots[i], c.c_void_p).value-dots_ptr)//c.sizeof(GridDot) for i in range(self.order)],
                'has_incentre': self.has_incentre,
                'ix': self.ix if self.has_incentre else -1,
                'iy': self.iy if self.has_incentre else -1,
            }

    class GridEdge(c.Structure):
        def _as_dict(self, dots_ptr, faces_ptr):
            return {
                'dot1_index': (c.cast(self.dot1, c.c_void_p).value-dots_ptr)//c.sizeof(GridDot),
                'dot2_index': (c.cast(self.dot2, c.c_void_p).value-dots_ptr)//c.sizeof(GridDot),
                'face1_index': (c.cast(self.face1, c.c_void_p).value-faces_ptr)//c.sizeof(GridFace) if self.face1 else None,
                'face2_index': (c.cast(self.face2, c.c_void_p).value-faces_ptr)//c.sizeof(GridFace) if self.face2 else None,
            }

    class GridDot(c.Structure):
        def _as_dict(self, edges_ptr, faces_ptr):
            return {
                'order': self.order,
                'edges_indices': [(c.cast(self.edges[i], c.c_void_p).value-edges_ptr)//c.sizeof(GridEdge) for i in range(self.order)],
                'faces_indices': [((c.cast(self.faces[i], c.c_void_p).value-faces_ptr)//c.sizeof(GridFace) if self.faces[i] else None) for i in range(self.order)],
                'x': self.x,
                'y': self.y,
                }

    GridFace._fields_ = [
        ('order', c.c_int),
        ('edges', CT_PTR(CT_PTR(GridEdge))),
        ('dots', CT_PTR(CT_PTR(GridDot))),
        ('has_incentre', c.c_bool),
        ('ix', c.c_int),
        ('iy', c.c_int),
    ]

    GridEdge._fields_ = [
        ('dot1', CT_PTR(GridDot)),
        ('dot2', CT_PTR(GridDot)),
        ('face1', CT_PTR(GridFace)),
        ('face2', CT_PTR(GridFace)),
    ]

    GridDot._fields_ = [
        ('order', c.c_int),
        ('edges', CT_PTR(CT_PTR(GridEdge))),
        ('faces', CT_PTR(CT_PTR(GridFace))),
        ('x', c.c_int),
        ('y', c.c_int),
    ]

    Grid._fields_ = [
        ('num_faces', c.c_int),
        ('faces', CT_PTR(GridFace)),
        ('num_edges', c.c_int),
        ('edges', CT_PTR(GridEdge)),
        ('num_dots', c.c_int),
        ('dots', CT_PTR(GridDot)),
        ('lowest_x', c.c_int),
        ('lowest_y', c.c_int),
        ('highest_x', c.c_int),
        ('highest_y', c.c_int),
        ('tilesize', c.c_int),
    ]

# Set the structures individiually for each Puzzle.
# This may not be needed overall but correctly setting the API
# should prevent some random segfaults


def set_api_structures_blackbox() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('minballs', c.c_int),
        ('maxballs', c.c_int),
    ]

    GameUi._fields_ = [
        ('flash_laserno', c.c_int),
        ('errors', c.c_int),
        ('newmove', c.c_bool),
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('cur_visible', c.c_bool),
        ('flash_laser', c.c_int),
    ]

    GameDrawState._fields_ = [
        ('tilesize', c.c_int),
        ('crad', c.c_int),
        ('rrad', c.c_int),
        ('w', c.c_int),
        ('h', c.c_int),
        ('grid', CT_UINT_PTR),
        ('started', c.c_bool),
        ('reveal', c.c_bool),
        ('isflash', c.c_bool),
        ('flash_laserno', c.c_int),
    ]

    GameState._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('minballs', c.c_int),
        ('maxballs', c.c_int),
        ('nballs', c.c_int),
        ('nlasers', c.c_int),
        ('grid', CT_UINT_PTR),
        ('exits', CT_UINT_PTR),
        ('done', c.c_bool),
        ('laserno', c.c_int),
        ('nguesses', c.c_int),
        ('nright', c.c_int),
        ('nwrong', c.c_int),
        ('nmissed', c.c_int),
        ('reveal', c.c_bool),
        ('justwrong', c.c_bool),
    ]


def set_api_structures_bridges() -> None:
    class _SurroundsPoints(c.Structure):
        _fields_ = [
            ('x', c.c_int),
            ('y', c.c_int),
            ('dx', c.c_int),
            ('dy', c.c_int),
            ('off', c.c_int),
        ]
        def _as_dict(self):
            return {
                'x': self.x,
                'y': self.y,
                'dx': self.dx,
                'dy': self.dy,
                'off': self.off,
            }

    class Surrounds(c.Structure):
        _fields_ = [
            ('points', _SurroundsPoints*4),
            ('npoints', c.c_int),
            ('nislands', c.c_int),
        ]
        def _as_dict(self):
            return {
                'npoints': self.npoints,
                'nislands': self.nislands,
                'points': [self.points[i]._as_dict() for i in range(4)]
            }

    class Island(c.Structure):
        _fields_ = [
            ('state', GAMESTATE_PTR),
            ('x', c.c_int),
            ('y', c.c_int),
            ('count', c.c_int),
            ('adj', Surrounds),
        ]
        def _as_dict(self):
            return {
                'x': self.x,
                'y': self.y,
                'count': self.count,
                'adj': self.adj._as_dict()
            }

    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('maxb', c.c_int),
        ('islands', c.c_int),
        ('expansion', c.c_int),
        ('allowloops', c.c_bool),
        ('difficulty', c.c_int),
    ]

    GameUi._fields_ = [
        ('dragx_src', c.c_int),
        ('dragy_src', c.c_int),
        ('dragx_dst', c.c_int),
        ('dragy_dst', c.c_int),
        ('todraw', c.c_uint),
        ('dragging', c.c_bool),
        ('drag_is_noline', c.c_bool),
        ('nlines', c.c_int),
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('cur_visible', c.c_bool),
        ('show_hints', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('tilesize', c.c_int),
        ('w', c.c_int),
        ('h', c.c_int),
        ('grid', CT_ULONG_PTR),
        ('newgrid', CT_ULONG_PTR),
        ('lv', CT_INT_PTR),
        ('lh', CT_INT_PTR),
        ('started', c.c_bool),
        ('dragging', c.c_bool),
    ]

    GameState._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('maxb', c.c_int),
        ('completed', c.c_bool),
        ('solved', c.c_bool),
        ('allowloops', c.c_bool),
        ('grid', CT_UINT_PTR),
        ('islands', CT_PTR(Island)),
        ('n_islands', c.c_int),
        ('n_islands_alloc', c.c_int),
        ('params', GameParams),
        ('wha', CT_PTR(c.c_char)),
        ('possv', CT_PTR(c.c_char)),
        ('possh', CT_PTR(c.c_char)),
        ('lines', CT_PTR(c.c_char)),
        ('maxv', CT_PTR(c.c_char)),
        ('maxh', CT_PTR(c.c_char)),
        ('gridi', CT_PTR(CT_PTR(Island))),
        ('solver', SOLVERSTATE_PTR),
    ]

    SolverState._fields_ = [
        ('dsf', CT_INT_PTR),
        ('comptspaces', CT_INT_PTR),
        ('tmpdsf', CT_INT_PTR),
        ('tmpcompspaces', CT_INT_PTR),
        ('refcount', c.c_int),
    ]


def set_api_structures_cube() -> None:
    GameParams._fields_ = [
        ('solid', c.c_int),
        ('d1', c.c_int),
        ('d2', c.c_int),
    ]

    GameUi._fields_ = [
    ]

    GameDrawState._fields_ = [
        ('gridscale', c.c_float),
        ('ox', c.c_int),
        ('oy', c.c_int),
    ]
    MAXVERTICES = 20
    MAXFACES = 20
    MAXORDER = 4

    class Solid(c.Structure):
        _fields_ = [
            ('nvertices', c.c_int),
            ('vertices', c.c_float*(MAXVERTICES * 3)),
            ('order', c.c_int),
            ('nfaces', c.c_int),
            ('faces', c.c_int*(MAXFACES * MAXORDER)),
            ('normals', c.c_float*(MAXFACES * 3)),
            ('shear', c.c_float),
            ('border', c.c_float),
        ]
        def _as_dict(self):
            return {
                'nvertices': self.nvertices,
                'vertices': [self.vertices[i] for i in range(MAXVERTICES*3)],
                'order': self.order,
                'nfaces': self.nfaces,
                'faces': [self.faces[i] for i in range(MAXFACES * MAXORDER)],
                'normals': [self.normals[i] for i in range(MAXFACES * 3)],
                'shear': self.shear,
                'border': self.border,
            }

    class GridSquare(c.Structure):
        _fields_ = [
            ('x', c.c_float),
            ('y', c.c_float),
            ('npoints', c.c_int),
            ('points', c.c_float*8),
            ('directions', c.c_int*8),
            ('flip', c.c_bool),
            ('tetra_class', c.c_int),
        ]
        def _as_dict(self):
            return {
                'x': self.x,
                'y': self.y,
                'npoints': self.npoints,
                'points': [self.points[i] for i in range(8)],
                'directions': [self.directions[i] for i in range(8)],
                'flip': self.flip,
                'tetra_class': self.tetra_class,
            }

    class GameGrid(c.Structure):
        _fields_ = [
            ('refcount', c.c_int),
            ('squares', CT_PTR(GridSquare)),
            ('nsquares', c.c_int),
        ]
        def _as_dict(self):
            return {
                'squares': [e._as_dict() for e in self.squares[:self.nsquares]],
                'nsquares': self.nsquares,
            }

    GameState._fields_ = [
        ('params', GameParams),
        ('solid', CT_PTR(Solid)),
        ('facecolours', CT_INT_PTR),
        ('grid', CT_PTR(GameGrid)),
        ('bluemask', CT_ULONG_PTR),
        ('current', c.c_int),
        ('sgkey', c.c_int*2),
        ('dgkey', c.c_int*2),
        ('spkey', c.c_int*2),
        ('dpkey', c.c_int*2),
        ('previous', c.c_int),
        ('angle', c.c_float),
        ('completed', c.c_int),
        ('movecount', c.c_int),
    ]


def set_api_structures_dominosa() -> None:
    GameParams._fields_ = [
        ('n', c.c_int),
        ('diff', c.c_int),
    ]

    GameUi._fields_ = [
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('highlight_1', c.c_int),
        ('highlight_2', c.c_int),
        ('cur_visible', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('tilesize', c.c_int),
        ('visible', CT_ULONG_PTR),
    ]

    class GameNumbers(c.Structure):
        _fields_ = [
            ('refcount', c.c_int),
            ('numbers', CT_INT_PTR),
        ]
        def _as_dict(self, wh):
            return {
                'numbers': _get_array_elements(self.numbers, wh)
            }

    GameState._fields_ = [
        ('params', GameParams),
        ('w', c.c_int),
        ('h', c.c_int),
        ('numbers', CT_PTR(GameNumbers)),
        ('grid', CT_INT_PTR),
        ('edges', CT_USHORT_PTR),
        ('completed', c.c_bool),
        ('cheated', c.c_bool),
    ]


def set_api_structures_fifteen() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
    ]

    GameUi._fields_ = [
    ]

    GameDrawState._fields_ = [
        ('started', c.c_bool),
        ('w', c.c_int),
        ('h', c.c_int),
        ('bgcolour', c.c_int),
        ('tiles', CT_INT_PTR),
        ('tilesize', c.c_int),
    ]

    GameState._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('n', c.c_int),
        ('tiles', CT_INT_PTR),
        ('gap_pos', c.c_int),
        ('completed', c.c_int),
        ('used_solve', c.c_bool),
        ('movecount', c.c_int),
    ]


def set_api_structures_filling() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
    ]

    GameUi._fields_ = [
        ('sel', CT_BOOL_PTR),
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('cur_visible', c.c_bool),
        ('keydragging', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('params', GameParams),
        ('tilesize', c.c_int),
        ('started', c.c_bool),
        ('v', CT_INT_PTR),
        ('flags', CT_INT_PTR),
        ('dsf_scratch', CT_INT_PTR),
        ('border_scratch', CT_INT_PTR),
    ]

    class SharedState(c.Structure):
        _fields_ = [
            ('params', GameParams),
            ('clues', CT_INT_PTR),
            ('refcnt', c.c_int),
        ]
        def _as_dict(self, arg):
            return {
                'clues': _get_array_elements(self.clues, arg)
            }

    GameState._fields_ = [
        ('board', CT_INT_PTR),
        ('shared', CT_PTR(SharedState)),
        ('completed', c.c_bool),
        ('cheated', c.c_bool),
    ]


def set_api_structures_flip() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('matrixtype', c.c_int),
    ]

    GameUi._fields_ = [
        ('cx', c.c_int),
        ('cy', c.c_int),
        ('cdraw', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('started', c.c_bool),
        ('tiles', CT_UCHAR_PTR),
        ('tilesize', c.c_int),

    ]

    class Matrix(c.Structure):
        _fields_ = [
            ('refcount', c.c_int),
            ('matrix', CT_UCHAR_PTR),
        ]
        def _as_dict(self, arg):
            return {
                'matrix': _get_array_elements(self.matrix, arg)
            }

    GameState._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('moves', c.c_int),
        ('completed', c.c_bool),
        ('cheated', c.c_bool),
        ('hints_active', c.c_bool),
        ('grid', CT_UCHAR_PTR),
        ('matrix', CT_PTR(Matrix)),
    ]


def set_api_structures_flood() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('colours', c.c_int),
        ('leniency', c.c_int),
    ]

    GameUi._fields_ = [
        ('cursor_visible', c.c_bool),
        ('cx', c.c_int),
        ('cy', c.c_int),
        ('flash_type', c.c_int),
    ]

    GameDrawState._fields_ = [
        ('started', c.c_bool),
        ('tilesize', c.c_int),
        ('grid', CT_INT_PTR),

    ]

    class Soln(c.Structure):
        _fields_ = [
            ('refcount', c.c_int),
            ('nmoves', c.c_int),
            ('moves', CT_PTR(c.c_char)),
        ]
        def _as_dict(self):
            return {
                'nmoves': self.nmoves,
                'moves': _get_array_elements(self.moves, self.nmoves),
            }

    GameState._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('colours', c.c_int),
        ('moves', c.c_int),
        ('movelimit', c.c_int),
        ('complete', c.c_bool),
        ('grid', CT_PTR(c.c_char)),
        ('cheated', c.c_bool),
        ('solnpos', c.c_int),
        ('soln', CT_PTR(Soln)),
    ]


def set_api_structures_galaxies() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('diff', c.c_int),
    ]

    GameUi._fields_ = [
        ('dragging', c.c_bool),
        ('dx', c.c_int),
        ('dy', c.c_int),
        ('dotx', c.c_int),
        ('doty', c.c_int),
        ('srcx', c.c_int),
        ('srcy', c.c_int),
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('cur_visible', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('started', c.c_bool),
        ('w', c.c_int),
        ('h', c.c_int),
        ('tilesize', c.c_int),
        ('grid', CT_ULONG_PTR),
        ('dx', CT_INT_PTR),
        ('dy', CT_INT_PTR),
        ('bl', Blitter),
        ('blmirror', Blitter),
        ('dragging', c.c_bool),
        ('dragx', c.c_int),
        ('dragy', c.c_int),
        ('oppx', c.c_int),
        ('oppy', c.c_int),
        ('colour_scratch', CT_INT_PTR),
        ('cx', c.c_int),
        ('cy', c.c_int),
        ('cur_visible', c.c_bool),
        ('cur_bl', Blitter),

    ]

    class Space(c.Structure):
        _fields_ = [
            ('x', c.c_int),
            ('y', c.c_int),
            ('type', c.c_int),
            ('flags', c.c_uint),
            ('dotx', c.c_int),
            ('doty', c.c_int),
            ('nassoc', c.c_int),
        ]
        def _as_dict(self):
            return {
                'x': self.x,
                'y': self.y,
                'type': self.type,
                'flags': self.flags,
                'dotx': self.dotx,
                'doty': self.doty,
                'nassoc': self.nassoc,
            }

    GameState._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('sx', c.c_int),
        ('sy', c.c_int),
        ('grid', CT_PTR(Space)),
        ('completed', c.c_bool),
        ('used_solve', c.c_bool),
        ('ndots', c.c_int),
        ('dots', CT_PTR(CT_PTR(Space))),
        ('me', MIDEND_PTR),
        ('cdiff', c.c_int),
    ]


def set_api_structures_guess() -> None:
    GameParams._fields_ = [
        ('ncolours', c.c_int),
        ('npegs', c.c_int),
        ('nguesses', c.c_int),
        ('allow_blank', c.c_bool),
        ('allow_multiple', c.c_bool),
    ]

    class PegRow(c.Structure):
        _fields_ = [
            ('npegs', c.c_int),
            ('pegs', CT_INT_PTR),
            ('feedback', CT_INT_PTR),
        ]
        def _as_dict(self):
            return {
                'npegs': self.npegs,
                'pegs': _get_array_elements(self.pegs, self.npegs),
                'feedback': _get_array_elements(self.feedback, self.npegs),
            }

    GameUi._fields_ = [
        ('params', GameParams),
        ('curr_pegs', PegRow),
        ('holds', CT_BOOL_PTR),
        ('colour_curr', c.c_int),
        ('peg_cur', c.c_int),
        ('display_cur', c.c_bool),
        ('markable', c.c_bool),
        ('drag_col', c.c_int),
        ('drag_x', c.c_int),
        ('drag_y', c.c_int),
        ('drag_opeg', c.c_int),
        ('show_labels', c.c_bool),
        ('hint', PegRow),
    ]

    GameDrawState._fields_ = [
        ('nguesses', c.c_int),
        ('guesses', CT_PTR(PegRow)),
        ('solution', PegRow),
        ('colours', PegRow),
        ('pegsz', c.c_int),
        ('hintsz', c.c_int),
        ('gapsz', c.c_int),
        ('pegrad', c.c_int),
        ('hintrad', c.c_int),
        ('border', c.c_int),
        ('colx', c.c_int),
        ('coly', c.c_int),
        ('guessx', c.c_int),
        ('guessy', c.c_int),
        ('solnx', c.c_int),
        ('solny', c.c_int),
        ('hintw', c.c_int),
        ('w', c.c_int),
        ('h', c.c_int),
        ('started', c.c_bool),
        ('solved', c.c_int),
        ('next_go', c.c_int),
        ('blit_peg', BLITTER_PTR),
        ('drag_col', c.c_int),
        ('blit_ox', c.c_int),
        ('blit_oy', c.c_int),
    ]

    GameState._fields_ = [
        ('params', GameParams),
        ('guesses', CT_PTR(CT_PTR(PegRow))),
        ('holds', CT_BOOL_PTR),
        ('solution', CT_PTR(PegRow)),
        ('next_go', c.c_int),
        ('solved', c.c_int),
    ]


def set_api_structures_inertia() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
    ]

    GameUi._fields_ = [
        ('anim_length', c.c_float),
        ('flashtype', c.c_int),
        ('deaths', c.c_int),
        ('just_made_move', c.c_bool),
        ('just_died', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('params', GameParams),
        ('tilesize', c.c_int),
        ('started', c.c_bool),
        ('grid', CT_USHORT_PTR),
        ('player_background', BLITTER_PTR),
        ('player_bg_saved', c.c_bool),
        ('pbgx', c.c_int),
        ('pbgy', c.c_int),
    ]

    class Soln(c.Structure):
        _fields_ = [
            ('refcount', c.c_int),
            ('len', c.c_int),
            ('list', CT_UCHAR_PTR),
        ]
        def _as_dict(self):
            return {
                'len': self.len,
                'list': _get_array_elements(self.list, self.len)
            }

    GameState._fields_ = [
        ('params', GameParams),
        ('px', c.c_int),
        ('py', c.c_int),
        ('gems', c.c_int),
        ('grid', CT_PTR(c.c_char)),
        ('distance_moved', c.c_int),
        ('dead', c.c_bool),
        ('cheated', c.c_bool),
        ('solnpos', c.c_int),
        ('soln', CT_PTR(Soln)),
    ]


def set_api_structures_keen() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('diff', c.c_int),
        ('multiplication_only', c.c_bool),
    ]

    GameUi._fields_ = [
        ('hx', c.c_int),
        ('hy', c.c_int),
        ('hpencil', c.c_bool),
        ('hshow', c.c_bool),
        ('hcursor', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('tilesize', c.c_int),
        ('started', c.c_bool),
        ('tiles', CT_LONG_PTR),
        ('errors', CT_LONG_PTR),
        ('minus_sign', c.c_char_p),
        ('times_sign', c.c_char_p),
        ('divide_sign', c.c_char_p),
    ]

    class Clues(c.Structure):
        _fields_ = [
            ('refcount', c.c_int),
            ('w', c.c_int),
            ('dsf', CT_INT_PTR),
            ('clues', CT_LONG_PTR),
        ]
        def _as_dict(self, a):
            return {
                'w': self.w,
                'dsf': _get_array_elements(self.dsf, a),
                'clues': _get_array_elements(self.clues, a),
            }

    GameState._fields_ = [
        ('par', GameParams),
        ('clues', CT_PTR(Clues)),
        ('grid', CT_DIGIT_PTR),
        ('pencil', CT_INT_PTR),
        ('completed', c.c_bool),
        ('cheated', c.c_bool),
    ]


def set_api_structures_lightup() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('plackpc', c.c_int),
        ('symm', c.c_int),
        ('difficulty', c.c_int),
    ]

    GameUi._fields_ = [
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('cur_visible', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('tilesize', c.c_int),
        ('crad', c.c_int),
        ('w', c.c_int),
        ('h', c.c_int),
        ('flags', CT_UINT_PTR),
        ('started', c.c_bool),
    ]

    GameState._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('nlights', c.c_int),
        ('lights', CT_INT_PTR),
        ('flags', CT_UINT_PTR),
        ('completed', c.c_bool),
        ('used_solve', c.c_bool),
    ]


def set_api_structures_loopy() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('diff', c.c_int),
        ('type', c.c_int),
    ]

    GameUi._fields_ = [
    ]

    GameDrawState._fields_ = [
        ('started', c.c_bool),
        ('tilesize', c.c_int),
        ('flashing', c.c_bool),
        ('textx', CT_INT_PTR),
        ('texty', CT_INT_PTR),
        ('lines', c.c_char_p),
        ('clue_error', CT_BOOL_PTR),
        ('clue_satisfied', CT_BOOL_PTR),
    ]

    set_grid_structure()

    GameState._fields_ = [
        ('game_grid', GRID_PTR),
        ('clues', CT_SCHAR_PTR),
        ('lines', CT_PTR(c.c_char)),
        ('line_errors', CT_BOOL_PTR),
        ('exactly_one_loop', c.c_bool),
        ('solved', c.c_bool),
        ('cheated', c.c_bool),
        ('grid_type', c.c_int),
    ]


def set_api_structures_magnets() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('diff', c.c_int),
        ('stripclues', c.c_bool),
    ]

    GameUi._fields_ = [
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('cur_visible', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('tilesize', c.c_int),
        ('started', c.c_bool),
        ('solved', c.c_bool),
        ('w', c.c_int),
        ('h', c.c_int),
        ('what', CT_ULONG_PTR),
        ('colwhat', CT_ULONG_PTR),
        ('rowwhat', CT_ULONG_PTR),
    ]

    class GameCommon(c.Structure):
        _fields_ = [
            ('dominoes', CT_INT_PTR),
            ('rowcount', CT_INT_PTR),
            ('colcount', CT_INT_PTR),
            ('refcount', c.c_int),
        ]
        def _as_dict(self, w, h):
            return {
                'dominoes': _get_array_elements(self.dominoes, w*h),
                'rowcount': _get_array_elements(self.rowcount, 3*h),
                'colcount': _get_array_elements(self.colcount, 3*w),
            }

    GameState._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('wh', c.c_int),
        ('grid', CT_INT_PTR),
        ('flags', CT_UINT_PTR),
        ('solved', c.c_bool),
        ('completed', c.c_bool),
        ('numbered', c.c_bool),
        ('counts_done', CT_BOOL_PTR),
        ('common', CT_PTR(GameCommon)),
    ]


def set_api_structures_map() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('n', c.c_int),
        ('diff', c.c_int),
    ]

    GameUi._fields_ = [
        ('drag_colour', c.c_int),
        ('drag_pencil', c.c_int),
        ('dragx', c.c_int),
        ('dragy', c.c_int),
        ('show_numbers', c.c_bool),
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('cur_lastmove', c.c_int),
        ('cur_visible', c.c_bool),
        ('cur_moved', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('tilesize', c.c_int),
        ('drawn', CT_ULONG_PTR),
        ('todraw', CT_ULONG_PTR),
        ('started', c.c_bool),
        ('dragx', c.c_int),
        ('dragy', c.c_int),
        ('drag_visible', c.c_bool),
        ('bl', BLITTER_PTR),
    ]

    class Map(c.Structure):
        _fields_ = [
            ('refcount', c.c_int),
            ('map', CT_INT_PTR),
            ('graph', CT_INT_PTR),
            ('n', c.c_int),
            ('ngraph', c.c_int),
            ('immutable', CT_BOOL_PTR),
            ('edgex', CT_INT_PTR),
            ('edgey', CT_INT_PTR),
            ('regionx', CT_INT_PTR),
            ('regiony', CT_INT_PTR),
        ]
        def _as_dict(self, wh):
            return {
                'map': _get_array_elements(self.map, 4 * wh),
                'graph': _get_array_elements(self.graph, self.n * self.n),
                'n': self.n,
                'ngraph': self.ngraph,
                'immutable': _get_array_elements(self.immutable, self.n),
                'edgex': _get_array_elements(self.edgex, self.ngraph),
                'regionx': _get_array_elements(self.regionx, self.n),
                'regiony': _get_array_elements(self.regiony, self.n),
            }

    GameState._fields_ = [
        ('p', GameParams),
        ('map', CT_PTR(Map)),
        ('colouring', CT_INT_PTR),
        ('pencil', CT_INT_PTR),
        ('completed', c.c_bool),
        ('cheated', c.c_bool),
    ]


def set_api_structures_mines() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('n', c.c_int),
        ('unique', c.c_bool),
    ]

    GameUi._fields_ = [
        ('hx', c.c_int),
        ('hy', c.c_int),
        ('hradius', c.c_int),
        ('validradius', c.c_int),
        ('flash_is_death', c.c_bool),
        ('deaths', c.c_int),
        ('completed', c.c_bool),
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('cur_visible', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('tilesize', c.c_int),
        ('bg', c.c_int),
        ('started', c.c_bool),
        ('grid', CT_SCHAR_PTR),
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
    ]

    class MineLayout(c.Structure):
        _fields_ = [
            ('refcount', c.c_int),
            ('mines', CT_BOOL_PTR),
            ('n', c.c_int),
            ('unique', c.c_bool),
            ('rs', RANDOMSTATE_PTR),
            ('me', MIDEND_PTR),
        ]
        def _as_dict(self, wh):
            return {
                'mines': [self.mines[i] for i in range(wh)] if self.mines else [],
                'n': self.n,
                'unique': self.unique,
            }

    GameState._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('n', c.c_int),
        ('dead', c.c_bool),
        ('won', c.c_bool),
        ('used_solve', c.c_bool),
        ('layout', CT_PTR(MineLayout)),
        ('grid', CT_SCHAR_PTR),
    ]


def set_api_structures_mosaic() -> None:
    GameParams._fields_ = [
        ('width', c.c_int),
        ('height', c.c_int),
        ('aggressive', c.c_bool),
    ]

    GameUi._fields_ = [
        ('solved', c.c_bool),
        ('in_progress', c.c_bool),
        ('last_x', c.c_int),
        ('last_y', c.c_int),
        ('last_state', c.c_int),
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('prev_cur_x', c.c_int),
        ('prev_cur_y', c.c_int),
        ('cur_visible', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('tilesize', c.c_int),
        ('state', CT_INT_PTR),
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('prev_cur_x', c.c_int),
        ('prev_cur_y', c.c_int),
    ]

    class BoardCell(c.Structure):
        _fields_ = [
            ('clue', CT_SCHAR),
            ('shown', c.c_bool),
        ]
        def _as_dict(self):
            return {
                'clue': self.clue,
                'shown': self.shown
            }

    class BoardState(c.Structure):
        _fields_ = [
            ('references', c.c_uint),
            ('actual_board', CT_PTR(BoardCell)),
        ]
        def _as_dict(self, wh):
            return {
                'actual_board': _get_substruct_array_elements(self.actual_board, wh)
            }

    GameState._fields_ = [
        ('cheating', c.c_bool),
        ('not_completed_clues', c.c_int),
        ('width', c.c_int),
        ('height', c.c_int),
        ('cells_contents', CT_PTR(c.c_char)),
        ('board', CT_PTR(BoardState)),
    ]


def set_api_structures_net() -> None:
    GameParams._fields_ = [
        ('width', c.c_int),
        ('height', c.c_int),
        ('wrapping', c.c_bool),
        ('unique', c.c_bool),
        ('barrier_probability', c.c_float),
    ]

    GameUi._fields_ = [
        ('org_x', c.c_int),
        ('org_y', c.c_int),
        ('cx', c.c_int),
        ('cy', c.c_int),
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('cur_visible', c.c_bool),
        ('rs', RANDOMSTATE_PTR),
    ]

    GameDrawState._fields_ = [
        ('width', c.c_int),
        ('height', c.c_int),
        ('tilesize', c.c_int),
        ('visible', CT_ULONG_PTR),
        ('to_draw', CT_ULONG_PTR),
    ]

    class GameImmutableState(c.Structure):
        _fields_ = [
            ('refcount', c.c_int),
            ('barriers', CT_UCHAR_PTR),
        ]
        def _as_dict(self, wh):
            return {
                'barriers': _get_array_elements(self.barriers, wh)
            }

    GameState._fields_ = [
        ('width', c.c_int),
        ('height', c.c_int),
        ('wrapping', c.c_bool),
        ('completed', c.c_bool),
        ('last_rotate_x', c.c_int),
        ('last_rotate_y', c.c_int),
        ('last_rotate_dir', c.c_int),
        ('used_solve', c.c_bool),
        ('tiles', CT_UCHAR_PTR),
        ('imm', CT_PTR(GameImmutableState)),
    ]


def set_api_structures_netslide() -> None:
    GameParams._fields_ = [
        ('width', c.c_int),
        ('height', c.c_int),
        ('wrapping', c.c_bool),
        ('barrier_probability', c.c_float),
        ('movetarget', c.c_int),
    ]

    GameUi._fields_ = [
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('cur_visible', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('started', c.c_bool),
        ('width', c.c_int),
        ('height', c.c_int),
        ('tilesize', c.c_int),
        ('visible', CT_UCHAR_PTR),
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
    ]

    GameState._fields_ = [
        ('width', c.c_int),
        ('height', c.c_int),
        ('cx', c.c_int),
        ('cy', c.c_int),
        ('completed', c.c_int),
        ('wrapping', c.c_bool),
        ('used_solve', c.c_bool),
        ('move_count', c.c_int),
        ('movetarget', c.c_int),
        ('last_move_row', c.c_int),
        ('last_move_col', c.c_int),
        ('last_move_dir', c.c_int),
        ('tiles', CT_UCHAR_PTR),
        ('barriers', CT_UCHAR_PTR),
    ]


def set_api_structures_palisade() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('k', c.c_int),
    ]

    GameUi._fields_ = [
        ('x', c.c_int),
        ('y', c.c_int),
        ('show', c.c_bool),
    ]
    DSFlags = c.c_ushort

    GameDrawState._fields_ = [
        ('tilesize', c.c_int),
        ('grid', CT_PTR(DSFlags)),
    ]

    Clue = CT_SCHAR
    BorderFlag = CT_UCHAR

    class SharedState(c.Structure):
        _fields_ = [
            ('params', GameParams),
            ('clues', CT_PTR(Clue)),
            ('refcount', c.c_int),
        ]
        def _as_dict(self):
            return {
                'params': {
                    'w': self.params.w,
                    'h': self.params.h,
                    'k': self.params.k,
                },
                'clues': _get_array_elements(self.clues, self.params.w * self.params.h)
            }

    GameState._fields_ = [
        ('shared', CT_PTR(SharedState)),
        ('borders', CT_PTR(BorderFlag)),
        ('completed', c.c_bool),
        ('cheated', c.c_bool),
    ]


def set_api_structures_pattern() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
    ]

    GameUi._fields_ = [
        ('dragging', c.c_bool),
        ('drag_start_x', c.c_int),
        ('drag_start_y', c.c_int),
        ('drag_end_x', c.c_int),
        ('drag_end_y', c.c_int),
        ('drag', c.c_int),
        ('release', c.c_int),
        ('state', c.c_int),
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('cur_visible', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('started', c.c_bool),
        ('w', c.c_int),
        ('h', c.c_int),
        ('tilesize', c.c_int),
        ('visible', CT_UCHAR_PTR),
        ('numcolours', CT_UCHAR_PTR),
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('strbuf', c.c_char_p),
    ]

    class GameStateCommon(c.Structure):
        _fields_ = [
            ('w', c.c_int),
            ('h', c.c_int),
            ('rowsize', c.c_int),
            ('rowdata', CT_INT_PTR),
            ('rowlen', CT_INT_PTR),
            ('immutable', CT_BOOL_PTR),
            ('refcount', c.c_int),
            ('fontsize', c.c_int),
        ]
        def _as_dict(self):
            wph = self.w + self.h
            return {
                'w': self.w,
                'h': self.h,
                'rowsize': self.rowsize,
                'rowdata': [e if abs(e) <= self.rowsize else 0 for e in self.rowdata[:self.rowsize * wph]],
                'rowlen': _get_array_elements(self.rowlen, wph),
                'immutable': _get_array_elements(self.immutable, self.w * self.h),
                'fontsize': self.fontsize,
            }

    GameState._fields_ = [
        ('common', CT_PTR(GameStateCommon)),
        ('grid', CT_UCHAR_PTR),
        ('completed', c.c_bool),
        ('cheated', c.c_bool),
    ]


def set_api_structures_pearl() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('difficulty', c.c_int),
        ('nosolve', c.c_bool),
    ]

    GameUi._fields_ = [
        ('dragcoords', CT_INT_PTR),
        ('ndragcoords', c.c_int),
        ('clickx', c.c_int),
        ('clicky', c.c_int),
        ('curx', c.c_int),
        ('cury', c.c_int),
        ('cursor_active', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('halfsz', c.c_int),
        ('started', c.c_bool),
        ('w', c.c_int),
        ('h', c.c_int),
        ('sz', c.c_int),
        ('lflags', CT_UINT_PTR),
        ('draglines', c.c_char_p),
    ]

    class SharedState(c.Structure):
        _fields_ = [
            ('w', c.c_int),
            ('h', c.c_int),
            ('sz', c.c_int),
            ('clues', CT_PTR(c.c_char)),
            ('refcnt', c.c_int),
        ]
        def _as_dict(self):
            return {
                'w': self.w,
                'h': self.h,
                'sz': self.sz,
                'clues': _get_array_elements(self.clues, self.w * self.h),
            }

    GameState._fields_ = [
        ('shared', CT_PTR(SharedState)),
        ('lines', CT_PTR(c.c_char)),
        ('errors', CT_PTR(c.c_char)),
        ('marks', CT_PTR(c.c_char)),
        ('completed', c.c_bool),
        ('used_solve', c.c_bool),
    ]


def set_api_structures_pegs() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('type', c.c_int),
    ]

    GameUi._fields_ = [
        ('dragging', c.c_bool),
        ('sx', c.c_int),
        ('sy', c.c_int),
        ('dx', c.c_int),
        ('dy', c.c_int),
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('cur_visible', c.c_bool),
        ('cur_jumping', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('tilesize', c.c_int),
        ('drag_background', BLITTER_PTR),
        ('dragging', c.c_bool),
        ('dragx', c.c_int),
        ('dragy', c.c_int),
        ('w', c.c_int),
        ('h', c.c_int),
        ('grid', CT_UCHAR_PTR),
        ('started', c.c_bool),
        ('bgcolour', c.c_int),
    ]

    GameState._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('completed', c.c_bool),
        ('grid', CT_UCHAR_PTR),
    ]


def set_api_structures_range() -> None:
    PuzzleSize = CT_SCHAR
    GameParams._fields_ = [
        ('w', PuzzleSize),
        ('h', PuzzleSize),
    ]

    GameUi._fields_ = [
        ('r', PuzzleSize),
        ('c', PuzzleSize),
        ('cursor_show', c.c_bool),
    ]

    class DrawCell(c.Structure):
        _fields_ = [
            ('value', PuzzleSize),
            ('error', c.c_bool),
            ('cursor', c.c_bool),
            ('flash', c.c_bool),
        ]

    GameDrawState._fields_ = [
        ('tilesize', c.c_int),
        ('grid', CT_PTR(DrawCell)),
    ]

    GameState._fields_ = [
        ('params', GameParams),
        ('has_cheated', c.c_bool),
        ('was_solved', c.c_bool),
        ('grid', CT_PTR(PuzzleSize)),
    ]


def set_api_structures_rect() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('expandfactor', c.c_float),
        ('unique', c.c_bool),
    ]

    GameUi._fields_ = [
        ('drag_start_x', c.c_int),
        ('drag_start_y', c.c_int),
        ('drag_end_x', c.c_int),
        ('drag_end_y', c.c_int),
        ('dragged', c.c_bool),
        ('erasing', c.c_bool),
        ('x1', c.c_int),
        ('y1', c.c_int),
        ('x2', c.c_int),
        ('y2', c.c_int),
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('cur_visible', c.c_bool),
        ('cur_dragging', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('started', c.c_bool),
        ('w', c.c_int),
        ('h', c.c_int),
        ('tilesize', c.c_int),
        ('visible', CT_ULONG_PTR),
    ]

    GameState._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('grid', CT_INT_PTR),
        ('vedge', CT_UCHAR_PTR),
        ('hedge', CT_UCHAR_PTR),
        ('completed', c.c_bool),
        ('cheated', c.c_bool),
        ('correct', CT_UCHAR_PTR),
    ]


def set_api_structures_samegame() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('ncols', c.c_int),
        ('scoresub', c.c_int),
        ('soluble', c.c_bool),
    ]

    GameUi._fields_ = [
        ('params', GameParams),
        ('tiles', CT_INT_PTR),
        ('nselected', c.c_int),
        ('xsel', c.c_int),
        ('ysel', c.c_int),
        ('displaysel', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('started', c.c_bool),
        ('bgcolour', c.c_int),
        ('tileinner', c.c_int),
        ('tilegap', c.c_int),
        ('tiles', CT_INT_PTR),
    ]

    GameState._fields_ = [
        ('params', GameParams),
        ('n', c.c_int),
        ('tiles', CT_INT_PTR),
        ('score', c.c_int),
        ('complete', c.c_bool),
        ('impossible', c.c_bool),
    ]


def set_api_structures_signpost() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('force_corner_start', c.c_bool),
    ]

    GameUi._fields_ = [
        ('cx', c.c_int),
        ('cy', c.c_int),
        ('cshow', c.c_bool),
        ('dragging', c.c_bool),
        ('drag_is_from', c.c_bool),
        ('sx', c.c_int),
        ('sy', c.c_int),
        ('dx', c.c_int),
        ('dy', c.c_int),
    ]

    GameDrawState._fields_ = [
        ('tilesize', c.c_int),
        ('started', c.c_bool),
        ('solved', c.c_bool),
        ('w', c.c_int),
        ('h', c.c_int),
        ('n', c.c_int),
        ('nums', CT_INT_PTR),
        ('dirp', CT_INT_PTR),
        ('f', CT_UINT_PTR),
        ('angle_offset', c.c_double),
        ('dragging', c.c_bool),
        ('dx', c.c_int),
        ('dy', c.c_int),
        ('dragb', BLITTER_PTR),
    ]

    GameState._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('n', c.c_int),
        ('completed', c.c_bool),
        ('used_solve', c.c_bool),
        ('impossible', c.c_bool),
        ('dirs', CT_INT_PTR),
        ('nums', CT_INT_PTR),
        ('flags', CT_UINT_PTR),
        ('next', CT_INT_PTR),
        ('prev', CT_INT_PTR),
        ('dsf', CT_INT_PTR),
        ('numsi', CT_INT_PTR),
    ]


def set_api_structures_singles() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('diff', c.c_int),
    ]

    GameUi._fields_ = [
        ('cx', c.c_int),
        ('cy', c.c_int),
        ('cshow', c.c_bool),
        ('show_black_nums', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('tilesize', c.c_int),
        ('started', c.c_bool),
        ('solved', c.c_bool),
        ('w', c.c_int),
        ('h', c.c_int),
        ('n', c.c_int),
    ]

    GameState._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('n', c.c_int),
        ('o', c.c_int),
        ('completed', c.c_bool),
        ('used_solve', c.c_bool),
        ('impossible', c.c_bool),
        ('nums', CT_INT_PTR),
        ('flags', CT_UINT_PTR),
    ]


def set_api_structures_sixteen() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('movetarget', c.c_int),
    ]

    GameUi._fields_ = [
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('cur_visible', c.c_bool),
        ('cur_mode', c.c_int),
    ]

    GameDrawState._fields_ = [
        ('started', c.c_bool),
        ('w', c.c_int),
        ('h', c.c_int),
        ('bgcolour', c.c_int),
        ('tiles', CT_INT_PTR),
        ('tilesize', c.c_int),
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
    ]

    GameState._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('n', c.c_int),
        ('tiles', CT_INT_PTR),
        ('completed', c.c_int),
        ('used_solve', c.c_bool),
        ('movecount', c.c_int),
        ('movetarget', c.c_int),
        ('last_movement_sense', c.c_int),
    ]


def set_api_structures_slant() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('diff', c.c_int),
    ]

    GameUi._fields_ = [
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('cur_visible', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('tilesize', c.c_int),
        ('grid', CT_LONG_PTR),
        ('todraw', CT_LONG_PTR),
    ]

    class GameClues(c.Structure):
        _fields_ = [
            ('w', c.c_int),
            ('h', c.c_int),
            ('clues', CT_SCHAR_PTR),
            ('tmpdsf', CT_INT_PTR),
            ('refcount', c.c_int),
        ]
        def _as_dict(self, WH):
            return {
                'w': self.w,
                'h': self.h,
                'clues': _get_array_elements(self.clues, WH),
                'tmpdsf': _get_array_elements(self.clues, WH * 2 + self.w + self.h + 2),
            }

    GameState._fields_ = [
        ('p', GameParams),
        ('clues', CT_PTR(GameClues)),
        ('soln', CT_SCHAR_PTR),
        ('errors', CT_UCHAR_PTR),
        ('completed', c.c_bool),
        ('used_solve', c.c_bool),
    ]


def set_api_structures_solo() -> None:
    GameParams._fields_ = [
        ('c', c.c_int),
        ('r', c.c_int),
        ('symm', c.c_int),
        ('diff', c.c_int),
        ('kdiff', c.c_int),
        ('xtype', c.c_bool),
        ('killer', c.c_bool),
    ]

    GameUi._fields_ = [
        ('hx', c.c_int),
        ('hy', c.c_int),
        ('hpencil', c.c_bool),
        ('hshow', c.c_bool),
        ('hcursor', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('started', c.c_bool),
        ('xtype', c.c_bool),
        ('cr', c.c_int),
        ('tilesize', c.c_int),
        ('grid', CT_DIGIT_PTR),
        ('pencil', CT_UCHAR_PTR),
        ('hl', CT_UCHAR_PTR),
        ('nregions', c.c_int),
        ('entered_items', CT_INT_PTR),
    ]

    class BlockStructure(c.Structure):
        _fields_ = [
            ('refcount', c.c_int),
            ('c', c.c_int),
            ('r', c.c_int),
            ('area', c.c_int),
            ('whichblock', CT_INT_PTR),
            ('blocks', CT_PTR(CT_INT_PTR)),
            ('nr_squares', CT_INT_PTR),
            ('blocks_data', CT_INT_PTR),
            ('nr_blocks', c.c_int),
            ('max_nr_squares', c.c_int),
        ]
        def _as_dict(self):
            return {
                'c': self.c,
                'r': self.r,
                'area': self.area,
                'whichblock': _get_array_elements(self.whichblock, self.area),
                'blocks': [_get_array_elements(self.blocks[i], self.nr_blocks) for i in range(self.nr_blocks)],
                'nr_squares': _get_array_elements(self.nr_squares, self.nr_blocks),
                'blocks_data': _get_array_elements(self.blocks_data, 
                                self.nr_blocks * self.max_nr_squares),
                'nr_blocks': self.nr_blocks,
                'max_nr_squares': self.max_nr_squares,
            }

    GameState._fields_ = [
        ('cr', c.c_int),
        ('blocks', CT_PTR(BlockStructure)),
        ('kblocks', CT_PTR(BlockStructure)),
        ('xtype', c.c_bool),
        ('killer', c.c_bool),
        ('grid', CT_DIGIT_PTR),
        ('kgrid', CT_DIGIT_PTR),
        ('pencil', CT_BOOL_PTR),
        ('immutable', CT_BOOL_PTR),
        ('completed', c.c_bool),
        ('cheated', c.c_bool),
    ]


def set_api_structures_tents() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('diff', c.c_int),
    ]

    GameUi._fields_ = [
        ('dsx', c.c_int),
        ('dsy', c.c_int),
        ('dex', c.c_int),
        ('dey', c.c_int),
        ('drag_button', c.c_int),
        ('drag_ok', c.c_bool),
        ('cx', c.c_int),
        ('cy', c.c_int),
        ('cdisp', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('tilesize', c.c_int),
        ('started', c.c_bool),
        ('p', GameParams),
        ('drawn', CT_INT_PTR),
        ('numbersdrawn', CT_INT_PTR),
        ('cx', c.c_int),
        ('cy', c.c_int),
    ]

    class Numbers(c.Structure):
        _fields_ = [
            ('refcount', c.c_int),
            ('numbers', CT_INT_PTR),
        ]
        def _as_dict(self, wh):
            return {
                'numbers': _get_array_elements(self.numbers, wh)
            }

    GameState._fields_ = [
        ('p', GameParams),
        ('grid', CT_PTR(c.c_char)),
        ('numbers', CT_PTR(Numbers)),
        ('completed', c.c_bool),
        ('used_solve', c.c_bool),
    ]


def set_api_structures_towers() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('diff', c.c_int),
    ]

    GameUi._fields_ = [
        ('hx', c.c_int),
        ('hy', c.c_int),
        ('hpencil', c.c_bool),
        ('hshow', c.c_bool),
        ('hcursor', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('tilesize', c.c_int),
        ('three_d', c.c_bool),
        ('tiles', CT_LONG_PTR),
        ('drawn', CT_LONG_PTR),
        ('errtmp', CT_BOOL_PTR),
    ]

    class Clues(c.Structure):
        _fields_ = [
            ('refcount', c.c_int),
            ('w', c.c_int),
            ('clues', CT_INT_PTR),
            ('immutable', CT_DIGIT_PTR),
        ]
        def _as_dict(self, a, w4):
            return {
                'w': self.w,
                'clues': _get_array_elements(self.clues, w4),
                'immutable': _get_array_elements(self.immutable, a),
            }

    GameState._fields_ = [
        ('par', GameParams),
        ('clues', CT_PTR(Clues)),
        ('clues_done', CT_BOOL_PTR),
        ('grid', CT_DIGIT_PTR),
        ('pencil', CT_INT_PTR),
        ('completed', c.c_bool),
        ('cheated', c.c_bool),
    ]


def set_api_structures_tracks() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('diff', c.c_int),
        ('single_ones', c.c_bool),
    ]

    GameUi._fields_ = [
        ('dragcoords', CT_INT_PTR),
        ('ndragcoords', c.c_int),
        ('clickx', c.c_int),
        ('clicky', c.c_int),
        ('curx', c.c_int),
        ('cury', c.c_int),
        ('cursor_active', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('sz6', c.c_int),
        ('border', c.c_int),
        ('grid_line_all', c.c_int),
        ('grid_line_tl', c.c_int),
        ('grid_line_br', c.c_int),
        ('started', c.c_bool),
        ('w', c.c_int),
        ('h', c.c_int),
        ('sz', c.c_int),
        ('flags', CT_UINT_PTR),
        ('flags_drag', CT_UINT_PTR),
        ('num_errors', CT_INT_PTR),
    ]

    class Numbers(c.Structure):
        _fields_ = [
            ('refcount', c.c_int),
            ('numbers', CT_INT_PTR),
            ('row_s', c.c_int),
            ('col_s', c.c_int),
        ]
        def _as_dict(self, wph):
            return {
                'numbers': _get_array_elements(self.numbers, wph),
                'row_s': self.row_s,
                'col_s': self.col_s,
            }

    GameState._fields_ = [
        ('p', GameParams),
        ('sflags', CT_UINT_PTR),
        ('numbers', CT_PTR(Numbers)),
        ('num_errors', CT_INT_PTR),
        ('completed', c.c_bool),
        ('used_solve', c.c_bool),
        ('impossible', c.c_bool),
    ]


def set_api_structures_twiddle() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('n', c.c_int),
        ('rowsonly', c.c_bool),
        ('orientable', c.c_bool),
        ('movetarget', c.c_int),
    ]

    GameUi._fields_ = [
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
        ('cur_visible', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('started', c.c_bool),
        ('w', c.c_int),
        ('h', c.c_int),
        ('bgcolour', c.c_int),
        ('grid', CT_INT_PTR),
        ('tilesize', c.c_int),
        ('cur_x', c.c_int),
        ('cur_y', c.c_int),
    ]

    GameState._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('n', c.c_int),
        ('orientable', c.c_bool),
        ('grid', CT_INT_PTR),
        ('completed', c.c_int),
        ('used_solve', c.c_bool),
        ('movecount', c.c_int),
        ('movetarget', c.c_int),
        ('lastx', c.c_int),
        ('lasty', c.c_int),
        ('lastr', c.c_int),
    ]


def set_api_structures_undead() -> None:
    GameParams._fields_ = [
        ('w', c.c_int),
        ('h', c.c_int),
        ('diff', c.c_int),
    ]

    GameUi._fields_ = [
        ('hx', c.c_int),
        ('hy', c.c_int),
        ('hshow', c.c_bool),
        ('hpencil', c.c_bool),
        ('hcursor', c.c_bool),
        ('ascii', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('tilesize', c.c_int),
        ('started', c.c_bool),
        ('solved', c.c_bool),
        ('w', c.c_int),
        ('h', c.c_int),
        ('monsters', CT_INT_PTR),
        ('pencils', CT_UCHAR_PTR),
        ('count_errors', c.c_bool*3),
        ('cell_errors', CT_BOOL_PTR),
        ('hint_errors', CT_BOOL_PTR),
        ('hints_done', CT_BOOL_PTR),
        ('hx', c.c_int),
        ('hy', c.c_int),
        ('hshow', c.c_bool),
        ('hpencil', c.c_bool),
        ('hflash', c.c_bool),
        ('ascii', c.c_bool),
    ]

    class Path(c.Structure):
        _fields_ = [
            ('length', c.c_int),
            ('p', CT_INT_PTR),
            ('grid_start', c.c_int),
            ('grid_end', c.c_int),
            ('num_monsters', c.c_int),
            ('mapping', CT_INT_PTR),
            ('sightings_start', c.c_int),
            ('sightings_end', c.c_int),
            ('xy', CT_INT_PTR),
        ]
        def _as_dict(self, wh):
            return {
                'length': self.length,
                'p': _get_array_elements(self.p, self.length),
                'grid_start': self.grid_start,
                'grid_end': self.grid_end,
                'num_monsters': self.num_monsters,
                'mapping': _get_array_elements(self.mapping, wh),
                'sightings_start': self.sightings_start,
                'sightings_end': self.sightings_end,
                'xy': _get_array_elements(self.xy, self.length),
            }

    class GameCommon(c.Structure):
        _fields_ = [
            ('refcount', c.c_int),
            ('params', GameParams),
            ('wh', c.c_int),
            ('num_ghosts', c.c_int),
            ('num_vampires', c.c_int),
            ('num_zombies', c.c_int),
            ('num_total', c.c_int),
            ('num_paths', c.c_int),
            ('paths', CT_PTR(Path)),
            ('grid', CT_INT_PTR),
            ('xinfo', CT_INT_PTR),
            ('fixed', CT_BOOL_PTR),
        ]
        def _as_dict(self):
            return {
                'params': {
                    'w': self.params.w,
                    'h': self.params.h,
                    'diff': self.params.diff,
                },
                'wh': self.wh,
                'num_ghosts': self.num_ghosts,
                'num_vampires': self.num_vampires,
                'num_zombies': self.num_zombies,
                'num_total': self.num_total,
                'num_paths': self.num_paths,
                'paths': _get_substruct_array_elements(self.paths, self.num_paths, self.wh),
                'grid': _get_array_elements(self.grid, self.wh),
                'xinfo': _get_array_elements(self.xinfo, self.wh),
                'fixed': _get_array_elements(self.fixed, self.num_total),
            }

    GameState._fields_ = [
        ('common', CT_PTR(GameCommon)),
        ('guess', CT_INT_PTR),
        ('pencils', CT_UCHAR_PTR),
        ('cell_errors', CT_BOOL_PTR),
        ('hint_errors', CT_BOOL_PTR),
        ('hints_done', CT_BOOL_PTR),
        ('count_errors', c.c_bool*3),
        ('solved', c.c_bool),
        ('cheated', c.c_bool),
    ]


def set_api_structures_unequal() -> None:
    GameParams._fields_ = [
        ('order', c.c_int),
        ('diff', c.c_int),
        ('mode', c.c_int),
    ]

    GameUi._fields_ = [
        ('hx', c.c_int),
        ('hy', c.c_int),
        ('hshow', c.c_bool),
        ('hpencil', c.c_bool),
        ('hcursor', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('tilesize', c.c_int),
        ('order', c.c_int),
        ('started', c.c_bool),
        ('mode', c.c_int),
        ('nums', CT_DIGIT_PTR),
        ('hints', CT_UCHAR_PTR),
        ('flags', CT_UINT_PTR),
        ('hx', c.c_int),
        ('hy', c.c_int),
        ('hshow', c.c_bool),
        ('hpencil', c.c_bool),
        ('hflash', c.c_bool),
    ]

    GameState._fields_ = [
        ('order', c.c_int),
        ('completed', c.c_bool),
        ('cheated', c.c_bool),
        ('mode', c.c_int),
        ('nums', CT_DIGIT_PTR),
        ('hints', CT_UCHAR_PTR),
        ('flags', CT_UINT_PTR),
    ]


def set_api_structures_unruly() -> None:
    GameParams._fields_ = [
        ('w2', c.c_int),
        ('h2', c.c_int),
        ('unique', c.c_bool),
        ('diff', c.c_int),
    ]

    GameUi._fields_ = [
        ('cx', c.c_int),
        ('cy', c.c_int),
        ('cursor', c.c_bool),
    ]

    GameDrawState._fields_ = [
        ('tilesize', c.c_int),
        ('w2', c.c_int),
        ('h2', c.c_int),
        ('started', c.c_bool),
        ('gridfs', CT_INT_PTR),
        ('rowfs', CT_BOOL_PTR),
        ('grid', CT_INT_PTR),
    ]

    class UnrulyCommon(c.Structure):
        _fields_ = [
            ('refcount', c.c_int),
            ('immutable', CT_BOOL_PTR),
        ]
        def _as_dict(self, s):
            return {
                'immutable': _get_array_elements(self.immutable, s)
            }

    GameState._fields_ = [
        ('w2', c.c_int),
        ('h2', c.c_int),
        ('unique', c.c_bool),
        ('grid', CT_PTR(c.c_char)),
        ('common', CT_PTR(UnrulyCommon)),
        ('completed', c.c_bool),
        ('cheated', c.c_bool),
    ]


def set_api_structures_untangle() -> None:
    GameParams._fields_ = [
        ('n', c.c_int),
    ]

    class Point(c.Structure):
        _fields_ = [
            ('x', c.c_long),
            ('y', c.c_long),
            ('d', c.c_long),
        ]
        def _as_dict(self):
            return {
                'x': self.x,
                'y': self.y,
                'd': self.d,
            }

    class Edge(c.Structure):
        _fields_ = [
            ('a', c.c_int),
            ('b', c.c_int),
        ]
        def _as_dict(self):
            return {
                'a': self.a,
                'b': self.b,
            }

    GameUi._fields_ = [
        ('dragpoint', c.c_int),
        ('newpoint', Point),
        ('just_dragged', c.c_bool),
        ('just_moved', c.c_bool),
        ('anim_length', c.c_float),
    ]

    GameDrawState._fields_ = [
        ('tilesize', c.c_long),
        ('bg', c.c_int),
        ('dragpoint', c.c_int),
        ('x', c.c_long),
        ('y', c.c_long),
    ]

    class Node234(c.Structure):
            pass

    Node234._fields_ = [
        ('parent', CT_PTR(Node234)),
        ('kids', CT_PTR(Node234)),
        ('counts', c.c_int*4),
        ('elems', c.c_void_p*3),
    ]

    class Tree234(c.Structure):
        _fields_ = [
            ('root', CT_PTR(Node234)),
            ('cmp', CT_FUNC(c.c_int, CT_PTR(Edge), CT_PTR(Edge))),
        ]

    class Graph(c.Structure):
        _fields_ = [
            ('refcount', c.c_int),
            ('edges', Tree234),
        ]
        def _as_dict(self):
            return {
                'edges': self.edges._as_dict()
            }

    GameState._fields_ = [
        ('params', GameParams),
        ('w', c.c_int),
        ('h', c.c_int),
        ('pts', CT_PTR(Point)),
        ('graph', CT_PTR(Graph)),
        ('completed', c.c_bool),
        ('cheated', c.c_bool),
        ('just_solved', c.c_bool),
    ]


set_api_structures_methods: dict[str, Callable[[], None]] = {
    'blackbox': set_api_structures_blackbox,
    'bridges': set_api_structures_bridges,
    'cube': set_api_structures_cube,
    'dominosa': set_api_structures_dominosa,
    'fifteen': set_api_structures_fifteen,
    'filling': set_api_structures_filling,
    'flip': set_api_structures_flip,
    'flood': set_api_structures_flood,
    'galaxies': set_api_structures_galaxies,
    'guess': set_api_structures_guess,
    'inertia': set_api_structures_inertia,
    'keen': set_api_structures_keen,
    'lightup': set_api_structures_lightup,
    'loopy': set_api_structures_loopy,
    'magnets': set_api_structures_magnets,
    'map': set_api_structures_map,
    'mines': set_api_structures_mines,
    'mosaic': set_api_structures_mosaic,
    'net': set_api_structures_net,
    'netslide': set_api_structures_netslide,
    'palisade': set_api_structures_palisade,
    'pattern': set_api_structures_pattern,
    'pearl': set_api_structures_pearl,
    'pegs': set_api_structures_pegs,
    'range': set_api_structures_range,
    'rect': set_api_structures_rect,
    'samegame': set_api_structures_samegame,
    'signpost': set_api_structures_signpost,
    'singles': set_api_structures_singles,
    'sixteen': set_api_structures_sixteen,
    'slant': set_api_structures_slant,
    'solo': set_api_structures_solo,
    'tents': set_api_structures_tents,
    'towers': set_api_structures_towers,
    'tracks': set_api_structures_tracks,
    'twiddle': set_api_structures_twiddle,
    'undead': set_api_structures_undead,
    'unequal': set_api_structures_unequal,
    'unruly': set_api_structures_unruly,
    'untangle': set_api_structures_untangle,
}


def set_api_structures(puzzle: str) -> None:
    if puzzle in set_api_structures_methods:
        if not hasattr(GameParams, '_fields_'):
            set_api_structures_methods[puzzle]()
    else:
        raise Exception(f'Puzzle {puzzle} does not have valid API Structures.')

def make_puzzle_state(puzzle: str, state: GameState) -> dict:
    if puzzle == 'blackbox':
        return {
            'w': state.w,
            'h': state.h,
            'minballs': state.minballs,
            'maxballs': state.maxballs,
            'nballs': state.nballs,
            'nlasers': state.nlasers,
            'grid': _get_array_elements(state.grid, (state.w+2)*(state.h+2)),
            'exits': _get_array_elements(state.exits, state.nlasers),
            'done': state.done,
            'laserno': state.laserno,
            'nguesses': state.nguesses,
            'nright': state.nright,
            'nwrong': state.nwrong,
            'nmissed': state.nmissed,
            'reveal': state.reveal,
            'justwrong': state.justwrong,
        }
    elif puzzle == 'bridges':
        wh = state.w*state.h
        return {
            'w': state.w,
            'h': state.h,
            'completed': state.completed,
            'solved': state.solved,
            'allowloops': state.allowloops,
            'grid': _get_array_elements(state.grid,wh),
            'islands': _get_substruct_array_elements(state.islands, state.n_islands),
            'n_islands': state.n_islands,
            'n_islands_alloc': state.n_islands_alloc,
            'params': {
                'w': state.params.w,
                'h': state.params.h,
                'maxb': state.params.maxb,
                'islands': state.params.islands,
                'expansion': state.params.expansion,
                'allowloops': state.params.allowloops,
                'difficulty': state.params.difficulty,
            },
            'wha': _get_array_elements(state.wha,wh),
            'possv': _get_array_elements(state.possv,wh),
            'possh': _get_array_elements(state.possh,wh),
            'lines': _get_array_elements(state.lines,wh),
            'maxv': _get_array_elements(state.maxv,wh),
            'maxh': _get_array_elements(state.maxh,wh),
        }
    elif puzzle == 'cube':
        return {
            'params': {
                'solid': state.params.solid,
                'd1': state.params.d1,
                'd2': state.params.d2,
            },
            'solid': _get_substruct(state.solid),
            'facecolours': _get_array_elements(state.facecolours,state.solid.contents.nfaces),
            'grid': _get_substruct(state.grid),
            'bluemask': _get_array_elements(state.bluemask,int((state.grid.contents.nsquares+31)/32)),
            'current': state.current,
            'sgkey': _get_array_elements(state.sgkey,2),
            'dgkey': _get_array_elements(state.dgkey,2),
            'spkey': _get_array_elements(state.spkey,2),
            'dpkey': _get_array_elements(state.dpkey,2),
            'previous': state.previous,
            'angle': state.angle,
            'completed': state.completed,
            'movecount': state.movecount,
        }
    elif puzzle == 'dominosa':
        wh = state.w * state.h
        return {
            'params': {
                'n': state.params.n,
                'diff': state.params.diff,
            },
            'w': state.w,
            'h': state.h,
            'numbers': _get_substruct(state.numbers, wh),
            'grid': _get_array_elements(state.grid, wh),
            'edges': _get_array_elements(state.edges, wh),
            'completed': state.completed,
            'cheated': state.cheated,
        }
    elif puzzle == 'fifteen':
        return {
            'w': state.w,
            'h': state.h,
            'n': state.n,
            'tiles': _get_array_elements(state.tiles, state.n),
            'gap_pos': state.gap_pos,
            'completed': state.completed,
            'used_solve': state.used_solve,
            'movecount': state.movecount,
        }
    elif puzzle == 'filling':
        wh = state.shared.contents.params.w * state.shared.contents.params.h
        return {
            'w': state.shared.contents.params.w,
            'h': state.shared.contents.params.h,
            'board': _get_array_elements(state.board, wh),
            'shared': _get_substruct(state.shared, wh),
            'completed': state.completed,
            'cheated': state.cheated,
        }
    elif puzzle == 'flip':
        wh = state.w * state.h
        return {
            'w': state.w,
            'h': state.h,
            'moves': state.moves,
            'completed': state.completed,
            'cheated': state.cheated,
            'hints_active': state.hints_active,
            'grid': _get_array_elements(state.grid, wh),
            'matrix': _get_substruct(state.matrix, wh),
        }
    elif puzzle == 'flood':
        return {
            'w': state.w,
            'h': state.h,
            'colours': state.colours,
            'moves': state.moves,
            'movelimit': state.movelimit,
            'complete': state.complete,
            'grid': _get_array_elements(state.grid, state.w * state.h),
            'cheated': state.cheated,
            'solnpos': state.solnpos,
            'soln': _get_substruct(state.soln) if state.soln else {},
        }
    elif puzzle == 'galaxies':
        return {
            'w': state.w,
            'h': state.h,
            'sx': state.sx,
            'sy': state.sy,
            'grid': _get_substruct_array_elements(state.grid, state.sx*state.sy),
            'completed': state.completed,
            'used_solve': state.used_solve,
            'ndots': state.ndots,
            'dots': _get_substruct_array_elements(state.dots[0], state.ndots),
            'cdiff': state.cdiff,
        }
    elif puzzle == 'guess':
        return {
            'params': {
                'ncolours': state.params.ncolours,
                'npegs': state.params.npegs,
                'nguesses': state.params.nguesses,
                'allow_blank': state.params.allow_blank,
                'allow_multiple': state.params.allow_multiple,
            }, 
            'guesses': [e[0]._as_dict() for e in state.guesses[:state.params.nguesses]],
            'holds': _get_array_elements(state.holds, state.params.npegs),
            'solution': _get_substruct(state.solution),
            'next_go': state.next_go,
            'solved': state.solved,
        }
    elif puzzle == 'inertia':
        return {
            'params': {
                'w': state.params.w,
                'h': state.params.h,
            }, 
            'px': state.px,
            'py': state.py,
            'gems': state.gems,
            'grid': _get_array_elements(state.grid, state.params.w * state.params.h),
            'distance_moved': state.distance_moved,
            'dead': state.dead,
            'cheated': state.cheated,
            'solnpos': state.solnpos,
            'soln': _get_substruct(state.soln) if state.soln else {},
        }
    elif puzzle == 'keen':
        a = state.par.w * state.par.w
        return {
            'par': {
                'w': state.par.w,
                'diff': state.par.diff,
                'multiplication_only': state.par.multiplication_only,
            }, 
            'clues': _get_substruct(state.clues, a),
            'grid': _get_array_elements(state.grid, a),
            'pencil': _get_array_elements(state.pencil, a),
            'completed': state.completed,
            'cheated': state.cheated,
        }
    elif puzzle == 'lightup':
        wh = state.w * state.h
        return {
            'w': state.w,
            'h': state.h,
            'nlights': state.nlights,
            'lights': _get_array_elements(state.lights, wh),
            'flags': _get_array_elements(state.flags, wh),
            'completed': state.completed,
            'used_solve': state.used_solve,
        }
    elif puzzle == 'loopy':
        return {
            'game_grid': _get_substruct(state.game_grid),
            'clues': _get_array_elements(state.clues, 
                                            state.game_grid.contents.num_faces),
            'lines': _get_array_elements(state.lines, 
                                            state.game_grid.contents.num_edges),
            'line_errors': _get_array_elements(state.line_errors, 
                                                state.game_grid.contents.num_edges),
            'exactly_one_loop': state.exactly_one_loop,
            'solved': state.solved,
            'cheated': state.cheated,
            'grid_type': state.grid_type,
        }
    elif puzzle == 'magnets':
        return {
            'w': state.w,
            'h': state.h,
            'wh': state.wh,
            'grid': _get_array_elements(state.grid, state.wh),
            'flags': _get_array_elements(state.flags, state.wh),
            'solved': state.solved,
            'completed': state.completed,
            'numbered': state.numbered,
            'counts_done': _get_array_elements(state.counts_done, (state.w + state.h) * 2),
            'common': _get_substruct(state.common, state.w, state.h),
        }
    elif puzzle == 'map':
        wh = state.p.w * state.p.h
        return {
            'p': {
                'w': state.p.w,
                'h': state.p.h,
                'n': state.p.n,
                'diff': state.p.diff,
            },
            'map': _get_substruct(state.map, wh),
            'colouring': _get_array_elements(state.colouring, state.p.n),
            'pencil': _get_array_elements(state.pencil, state.p.n),
            'completed': state.completed,
            'cheated': state.cheated,
        }
    elif puzzle == 'mines':
        wh = state.w * state.h
        return {
            'w': state.w,
            'h': state.h,
            'n': state.n,
            'dead': state.dead,
            'won': state.won,
            'used_solve': state.used_solve,
            'layout': _get_substruct(state.layout, wh),
            'grid': _get_array_elements(state.grid, wh),
        }
    elif puzzle == 'mosaic':
        wh = state.width * state.height
        return {
            'cheating': state.cheating,
            'not_completed_clues': state.not_completed_clues,
            'width': state.width,
            'height': state.height,
            'cells_contents': _get_array_elements(state.cells_contents, wh),
            'board': _get_substruct(state.board, wh),
        }
    elif puzzle == 'net':
        wh = state.width * state.height
        return {
            'width': state.width,
            'height': state.height,
            'wrapping': state.wrapping,
            'completed': state.completed,
            'last_rotate_x': state.last_rotate_x,
            'last_rotate_y': state.last_rotate_y,
            'last_rotate_dir': state.last_rotate_dir,
            'used_solve': state.used_solve,
            'tiles': _get_array_elements(state.tiles, wh),
            'imm': _get_substruct(state.imm, wh),
        }
    elif puzzle == 'netslide':
        wh = state.width * state.height
        return {
            'width': state.width,
            'height': state.height,
            'cx': state.cx,
            'cy': state.cy,
            'completed': state.completed,
            'wrapping': state.wrapping,
            'used_solve': state.used_solve,
            'move_count': state.move_count,
            'movetarget': state.movetarget,
            'last_move_row': state.last_move_row,
            'last_move_col': state.last_move_col,
            'last_move_dir': state.last_move_dir,
            'tiles': _get_array_elements(state.tiles, wh),
            'barriers': _get_array_elements(state.barriers, wh),
        }
    elif puzzle == 'palisade':
        return {
            'shared': _get_substruct(state.shared),
            'borders': _get_array_elements(state.borders, 
                                            state.shared.contents.params.w*state.shared.contents.params.h),
            'completed': state.completed,
            'cheated': state.cheated,
        }
    elif puzzle == 'pattern':
        return {
            'common': _get_substruct(state.common),
            'grid': _get_array_elements(state.grid, 
                                        state.common.contents.w*state.common.contents.h),
            'completed': state.completed,
            'cheated': state.cheated,
        }
    elif puzzle == 'pearl':
        wh = state.shared.contents.w * state.shared.contents.h
        return {
            'shared': _get_substruct(state.shared),
            'lines': _get_array_elements(state.lines, wh),
            'errors': _get_array_elements(state.errors, wh),
            'marks': _get_array_elements(state.marks, wh),
            'completed': state.completed,
            'used_solve': state.used_solve,
        }
    elif puzzle == 'pegs':
        return {
            'w': state.w,
            'h': state.h,
            'completed': state.completed,
            'grid': _get_array_elements(state.grid, state.w * state.h),
        }
    elif puzzle == 'range':
        return {
            'params': {
                'w': state.params.w,
                'h': state.params.h,
            },
            'has_cheated': state.has_cheated,
            'was_solved': state.was_solved,
            'grid': _get_array_elements(state.grid, 
                                        state.params.w * state.params.h),
        }
    elif puzzle == 'rect':
        wh = state.w * state.h
        return {
            'w': state.w,
            'h': state.h,
            'grid': _get_array_elements(state.grid, wh),
            'vedge': _get_array_elements(state.vedge, wh+state.h),
            'hedge': _get_array_elements(state.hedge, wh+state.w),
            'completed': state.completed,
            'cheated': state.cheated,
            'correct': _get_array_elements(state.correct, wh),
        }
    elif puzzle == 'samegame':
        return {
            'params': {
                'w': state.params.w,
                'h': state.params.h,
                'ncols': state.params.ncols,
                'scoresub': state.params.scoresub,
                'soluble': state.params.soluble,
            },
            'n': state.n,
            'tiles': _get_array_elements(state.tiles, state.n),
            'score': state.score,
            'complete': state.complete,
            'impossible': state.impossible,
        }
    elif puzzle == 'signpost':
        return {
            'w': state.w,
            'h': state.h,
            'n': state.n,
            'completed': state.completed,
            'used_solve': state.used_solve,
            'impossible': state.impossible,
            'dirs': _get_array_elements(state.dirs, state.n),
            'nums': _get_array_elements(state.nums, state.n),
            'flags': _get_array_elements(state.flags, state.n),
            'next': _get_array_elements(state.next, state.n),
            'prev': _get_array_elements(state.prev, state.n),
            'dsf': _get_array_elements(state.dsf, state.n),
            'numsi': _get_array_elements(state.numsi, state.n+1),
        }
    elif puzzle == 'singles':
        return {
            'w': state.w,
            'h': state.h,
            'n': state.n,
            'o': state.o,
            'completed': state.completed,
            'used_solve': state.used_solve,
            'impossible': state.impossible,
            'nums': _get_array_elements(state.nums, state.n),
            'flags': _get_array_elements(state.flags, state.n),
        }
    elif puzzle == 'sixteen':
        return {
            'w': state.w,
            'h': state.h,
            'n': state.n,
            'tiles': _get_array_elements(state.tiles, state.n),
            'completed': state.completed,
            'used_solve': state.used_solve,
            'movecount': state.movecount,
            'movetarget': state.movetarget,
            'last_movement_sense': state.last_movement_sense,
        }
    elif puzzle == 'slant':
        WH = (state.p.w+1)*(state.p.h+1)
        return {
            'p': {
                'w': state.p.w,
                'h': state.p.h,
                'diff': state.p.diff,
            },
            'clues': _get_substruct(state.clues, WH),
            'soln': _get_array_elements(state.soln, state.p.w * state.p.h),
            'errors': _get_array_elements(state.errors, WH),
            'completed': state.completed,
            'used_solve': state.used_solve,
        }
    elif puzzle == 'solo':
        area  = state.cr * state.cr
        return {
            'cr': state.cr,
            'blocks': _get_substruct(state.blocks),
            'kblocks': _get_substruct(state.kblocks) if state.killer else 'type is not killer',
            'xtype': state.xtype,
            'killer': state.killer,
            'grid': _get_array_elements(state.grid, area),
            'kgrid': _get_array_elements(state.kgrid, area) if state.killer else 'type is not killer',
            'pencil': _get_array_elements(state.pencil, area * state.cr),
            'immutable': _get_array_elements(state.immutable, area),
            'completed': state.completed,
            'cheated': state.cheated,
        }
    elif puzzle == 'tents':
        return {
            'p': {
                'w': state.p.w,
                'h': state.p.h,
                'diff': state.p.diff,
            },
            'grid': _get_array_elements(state.grid, 
                                            state.p.w * state.p.h),
            'numbers': _get_substruct(state.numbers, 
                                        state.p.w + state.p.h),
            'completed': state.completed,
            'used_solve': state.used_solve,
        }
    elif puzzle == 'towers':
        a = state.par.w * state.par.w
        w4 = state.par.w*4
        return {
            'par': {
                'w': state.par.w,
                'diff': state.par.diff,
            },
            'clues': _get_substruct(state.clues, a, w4),
            'clues_done': _get_array_elements(state.clues_done, w4),
            'grid': _get_array_elements(state.grid, a),
            'pencil': _get_array_elements(state.pencil, a),
            'completed': state.completed,
            'cheated': state.cheated,
        }
    elif puzzle == 'tracks':
        wph = state.p.w + state.p.h
        return {
            'p': {
                'w': state.p.w,
                'h': state.p.h,
                'diff': state.p.diff,
                'single_ones': state.p.single_ones,
            },
            'sflags': _get_array_elements(state.sflags,
                                            state.p.w * state.p.h),
            'numbers': _get_substruct(state.numbers, wph),
            'num_errors': _get_array_elements(state.num_errors, wph),
            'completed': state.completed,
            'used_solve': state.used_solve,
            'impossible': state.impossible,
        }
    elif puzzle == 'twiddle':
        return {
            'w': state.w,
            'h': state.h,
            'n': state.n,
            'orientable': state.orientable,
            'grid': _get_array_elements(state.grid, 
                                        state.w * state.h),
            'completed': state.completed,
            'used_solve': state.used_solve,
            'movecount': state.movecount,
            'movetarget': state.movetarget,
            'lastx': state.lastx,
            'lasty': state.lasty,
            'lastr': state.lastr,
        }
    elif puzzle == 'undead':
        np2 = 2 * state.common.contents.num_paths
        return {
            'common': _get_substruct(state.common),
            'guess': _get_array_elements(state.guess, 
                                            state.common.contents.num_total),
            'pencils': _get_array_elements(state.pencils, 
                                            state.common.contents.num_total),
            'cell_errors': _get_array_elements(state.cell_errors, 
                                                state.common.contents.wh),
            'hint_errors': _get_array_elements(state.hint_errors, np2),
            'hints_done': _get_array_elements(state.hints_done, np2),
            'count_errors': _get_array_elements(state.count_errors, 3),
            'solved': state.solved,
            'cheated': state.cheated,
        }
    elif puzzle == 'unequal':
        o2 = state.order * state.order
        return {
            'order': state.order,
            'completed': state.completed,
            'cheated': state.cheated,
            'mode': state.mode,
            'nums': _get_array_elements(state.nums, o2),
            'hints': _get_array_elements(state.hints, o2 * state.order),
            'flags': _get_array_elements(state.flags, o2),
        }
    elif puzzle == 'unruly':
        s = state.w2 * state.h2
        return {
            'w2': state.w2,
            'h2': state.h2,
            'unique': state.unique,
            'grid': _get_array_elements(state.grid, s),
            'common': _get_substruct(state.common, s),
            'completed': state.completed,
            'cheated': state.cheated,
        }
    elif puzzle == 'untangle':
        return {
            'params': {
                'n': state.params.n,
            },
            'w': state.w,
            'h': state.h,
            'pts': _get_substruct_array_elements(state.pts, state.params.n),
            'graph': _get_substruct(state.graph),
            'completed': state.completed,
            'cheated': state.cheated,
            'just_solved': state.just_solved,
        }
    else:
        raise Exception(f'Extracting the internal state of {puzzle} is not supported.')


def get_action_keys_blackbox() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
    ]


def get_action_keys_bridges() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
    ]


def get_action_keys_cube() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
    ]


def get_action_keys_dominosa() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
    ]


def get_action_keys_fifteen() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
    ]


def get_action_keys_filling() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_KP_1,
        pl.K_KP_2,
        pl.K_KP_3,
        pl.K_KP_4,
        pl.K_KP_5,
        pl.K_KP_6,
        pl.K_KP_7,
        pl.K_KP_8,
        pl.K_KP_9,
    ]


def get_action_keys_flip() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
    ]


def get_action_keys_flood() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
    ]


def get_action_keys_galaxies() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
    ]


def get_action_keys_guess() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
    ]


def get_action_keys_inertia() -> list[int]:
    return [
        pl.K_KP_1,
        pl.K_KP_2,
        pl.K_KP_3,
        pl.K_KP_4,
        pl.K_KP_6,
        pl.K_KP_7,
        pl.K_KP_8,
        pl.K_KP_9,
        pl.K_u,
    ]


def get_action_keys_keen() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_SPACE,
        pl.K_KP_1,
        pl.K_KP_2,
        pl.K_KP_3,
        pl.K_KP_4,
        pl.K_KP_5,
        pl.K_KP_6,
        pl.K_KP_7,
        pl.K_KP_8,
        pl.K_KP_9,
    ]


def get_action_keys_lightup() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
    ]


def get_action_keys_magnets() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
        pl.K_SPACE,
    ]


def get_action_keys_map() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
    ]


def get_action_keys_mines() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
        pl.K_SPACE,
        pl.K_u,
    ]


def get_action_keys_mosaic() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
        pl.K_SPACE,
    ]


def get_action_keys_net() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
    ]


def get_action_keys_netslide() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
    ]


def get_action_keys_palisade() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_LCTRL,
    ]


def get_action_keys_pattern() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
        pl.K_SPACE,
    ]


def get_action_keys_pearl() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
    ]


def get_action_keys_pegs() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
        pl.K_u,
    ]


def get_action_keys_range() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
    ]


def get_action_keys_rect() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
    ]


def get_action_keys_samegame() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
        pl.K_u,
    ]


def get_action_keys_signpost() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
        pl.K_SPACE,
    ]


def get_action_keys_singles() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
        pl.K_SPACE,
        ]


def get_action_keys_sixteen() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
        pl.K_SPACE,
    ]


def get_action_keys_slant() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
        pl.K_SPACE,
    ]


def get_action_keys_solo() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_KP_1,
        pl.K_KP_2,
        pl.K_KP_3,
        pl.K_KP_4,
        pl.K_KP_5,
        pl.K_KP_6,
        pl.K_KP_7,
        pl.K_KP_8,
        pl.K_KP_9,
    ]


def get_action_keys_tents() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
        pl.K_SPACE,
    ]


def get_action_keys_towers() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_SPACE,
        pl.K_KP_1,
        pl.K_KP_2,
        pl.K_KP_3,
        pl.K_KP_4,
        pl.K_KP_5,
        pl.K_KP_6,
        pl.K_KP_7,
        pl.K_KP_8,
        pl.K_KP_9,
    ]


def get_action_keys_tracks() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
    ]


def get_action_keys_twiddle() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
        pl.K_SPACE,
    ]


def get_action_keys_undead() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_SPACE,
        pl.K_1,
        pl.K_2,
        pl.K_3,
    ]


def get_action_keys_unequal() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_KP_1,
        pl.K_KP_2,
        pl.K_KP_3,
        pl.K_KP_4,
        pl.K_KP_5,
        pl.K_KP_6,
        pl.K_KP_7,
        pl.K_KP_8,
        pl.K_KP_9,
    ]


def get_action_keys_unruly() -> list[int]:
    return [
        pl.K_UP,
        pl.K_DOWN,
        pl.K_LEFT,
        pl.K_RIGHT,
        pl.K_RETURN,
        pl.K_SPACE,
    ]


get_action_keys_methods: dict[str, Callable[[], list[int]]] = {
    'blackbox': get_action_keys_blackbox,
    'bridges': get_action_keys_bridges,
    'cube': get_action_keys_cube,
    'dominosa': get_action_keys_dominosa,
    'fifteen': get_action_keys_fifteen,
    'filling': get_action_keys_filling,
    'flip': get_action_keys_flip,
    'flood': get_action_keys_flood,
    'galaxies': get_action_keys_galaxies,
    'guess': get_action_keys_guess,
    'inertia': get_action_keys_inertia,
    'keen': get_action_keys_keen,
    'lightup': get_action_keys_lightup,
    'magnets': get_action_keys_magnets,
    'map': get_action_keys_map,
    'mines': get_action_keys_mines,
    'mosaic': get_action_keys_mosaic,
    'net': get_action_keys_net,
    'netslide': get_action_keys_netslide,
    'palisade': get_action_keys_palisade,
    'pattern': get_action_keys_pattern,
    'pearl': get_action_keys_pearl,
    'pegs': get_action_keys_pegs,
    'range': get_action_keys_range,
    'rect': get_action_keys_rect,
    'samegame': get_action_keys_samegame,
    'signpost': get_action_keys_signpost,
    'singles': get_action_keys_singles,
    'sixteen': get_action_keys_sixteen,
    'slant': get_action_keys_slant,
    'solo': get_action_keys_solo,
    'tents': get_action_keys_tents,
    'towers': get_action_keys_towers,
    'tracks': get_action_keys_tracks,
    'twiddle': get_action_keys_twiddle,
    'undead': get_action_keys_undead,
    'unequal': get_action_keys_unequal,
    'unruly': get_action_keys_unruly,
}


def get_action_keys(puzzle: str) -> list[int]:
    if puzzle in get_action_keys_methods:
        return get_action_keys_methods[puzzle]()
    else:
        raise Exception(f'Puzzle {puzzle} does not have a valid action space.')

# end of puzzle-specific section
