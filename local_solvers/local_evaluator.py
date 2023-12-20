"""
borrows machinery of LocalSolver
but solvers f(A1 * x1, A2 * x2, ...) using projectors
out = f(P1 A1 P1^* P1 x1, P2 A2 P2^* P2 x2, ...)
where Ai are the "operators", xi are the "targets"
"""

