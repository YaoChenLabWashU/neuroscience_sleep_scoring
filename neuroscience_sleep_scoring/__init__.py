"""Sleep-scoring package.

NOTE: the matplotlib backend is pinned on macOS BEFORE anything imports pyplot
(the `from .extract_data import *` below does). See _pin_gui_backend().
"""
import sys as _sys


def _pin_gui_backend():
	"""On macOS, force matplotlib onto TkAgg.

	matplotlib auto-selects in the order ['macosx', 'qtagg', ..., 'tkagg'], so on
	a Mac it picks the Cocoa 'macosx' backend. This GUI is built around a single
	Tk interpreter: the launcher owns the Tk mainloop, every dialog is parented to
	matplotlib's Tk root (New_SWS._mpl_root), and the saved window layout is
	applied with Tk-only wm_geometry(). With the macosx backend those quietly stop
	working, and matplotlib's Cocoa event loop ends up nested inside Tk's -- both
	driving the one NSApplication run loop, which aborts the interpreter.

	Linux/Windows are left alone: whatever matplotlib picks there (QtAgg here) is
	already proven and measurably faster at blitting, and X11/Win32 have no
	single shared run loop for the toolkits to fight over.
	"""
	if _sys.platform != 'darwin':
		return
	try:
		import matplotlib
		matplotlib.use('TkAgg', force=True)
	except Exception as e:  # pragma: no cover - never block import over this
		print(f'Could not pin the TkAgg backend ({e}); '
			'the GUI may be unstable on macOS.')


_pin_gui_backend()

from .extract_data import *
from .New_SWS import *
