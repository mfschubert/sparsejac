"""sparsejac - Efficient forward- and reverse-mode sparse Jacobians using Jax."""

__version__ = "v0.2.0"
__author__ = "Martin Schubert <mfschubert@gmail.com>"

from sparsejac.sparsejac import ForwardModeColoring as ForwardModeColoring
from sparsejac.sparsejac import ReverseModeColoring as ReverseModeColoring
from sparsejac.sparsejac import jacfwd as jacfwd
from sparsejac.sparsejac import jacfwd_coloring as jacfwd_coloring
from sparsejac.sparsejac import jacrev as jacrev
from sparsejac.sparsejac import jacrev_coloring as jacrev_coloring
