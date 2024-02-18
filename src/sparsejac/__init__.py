"""sparsejac - Efficient forward- and reverse-mode sparse Jacobians using Jax."""

__version__ = "0.1.1"
__author__ = "Martin Schubert <mfschubert@gmail.com>"

from sparsejac.sparsejac import jacfwd as jacfwd
from sparsejac.sparsejac import jacrev as jacrev
