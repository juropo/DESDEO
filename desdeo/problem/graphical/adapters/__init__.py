"""Backend-specific adapters for `desdeo.problem.graphical`.

Each adapter module is the only place its backend is imported from within
`desdeo/problem/graphical/`; this `__init__.py` intentionally does not re-export from them, so
importing `desdeo.problem.graphical.adapters` alone pulls in no backend.
"""
