"""
emission.py - Stabilizer tableau simulation for photon emission protocols.

Efficient Python implementation of the MATLAB stabilizer-based circuit solver.
All qubit indices are 0-based throughout this module.

The stabilizer state of n qubits is represented by two n×n binary matrices
``xs`` (X part) and ``zs`` (Z part), plus a length-n binary sign vector.
Row i encodes the generator  (-1)^{sign[i]} · X^{xs[i]} · Z^{zs[i]}
where X^{xs[i]} is shorthand for the tensor product of single-qubit Paulis.
"""

import numpy as np


# ============================================================
# Data Structure
# ============================================================

class StabilizerState:
    """Stabilizer state stored as separate X and Z binary matrices.

    Attributes
    ----------
    xs : ndarray, shape (n, n), dtype int8
        Binary matrix for the X part of each generator.
    zs : ndarray, shape (n, n), dtype int8
        Binary matrix for the Z part of each generator.
    sign_vector : ndarray, shape (n,), dtype int8
        Binary vector.  0 → '+', 1 → '-'.
    """

    def __init__(self, xs, zs, sign_vector):
        self.xs = np.asarray(xs, dtype=np.int8) % 2
        self.zs = np.asarray(zs, dtype=np.int8) % 2
        self.sign_vector = np.asarray(sign_vector, dtype=np.int8).flatten() % 2

    @property
    def n(self):
        return self.xs.shape[0]

    def copy(self):
        return StabilizerState(self.xs.copy(), self.zs.copy(),
                               self.sign_vector.copy())


# ============================================================
# Clifford Operations  (efficient, column-based on xs / zs)
# ============================================================

_COMPOSITE_GATES = frozenset({
    'HP', 'PH', 'HPH',
    'HX', 'PX', 'HPX', 'PHX', 'HPHX',
    'HY', 'PY', 'HPY', 'PHY', 'HPHY',
    'HZ', 'PZ', 'HPZ', 'PHZ', 'HPHZ',
})


def clifford_operation(state, gate, site):
    """Apply a Clifford gate to *state* in-place and return it.

    Instead of constructing a full 2n×2n symplectic matrix and doing matrix
    multiplication (as the MATLAB version does), each gate directly updates
    the relevant columns of xs / zs and the sign vector.  This is O(n) per
    gate rather than O(n²).

    Parameters
    ----------
    state : StabilizerState
    gate  : str  – 'I','H','P','X','Y','Z','CNOT'/'CX', or composite.
    site  : int for 1-qubit gates, or [control, target] for CNOT.

    Sign-update rules
    -----------------
    After the tableau columns are updated, the sign vector is corrected so
    that the represented Pauli operators stay consistent.  The formulae come
    from the Aaronson–Gottesman CHP formalism (PRA 70, 052328, 2004):

    * **H(i)**: swap xs[:,i] ↔ zs[:,i], then  sgn ^= xs[:,i] · zs[:,i].
    * **P(i)**: zs[:,i] += xs[:,i], then  sgn ^= xs[:,i] · zs[:,i].
    * **X(i)**: sgn ^= zs[:,i]   (Z anticommutes with X).
    * **Y(i)**: sgn ^= xs[:,i] + zs[:,i].
    * **Z(i)**: sgn ^= xs[:,i]   (X anticommutes with Z).
    * **CNOT(c,t)**: xs[:,t] += xs[:,c],  zs[:,c] += zs[:,t],
      then  sgn ^= xs[:,c] · zs[:,t] · (xs[:,t] + zs[:,c] + 1).
    """
    xs = state.xs
    zs = state.zs
    sgn = state.sign_vector

    if gate in ('I', '', None):
        return state

    # Hadamard on qubit i: X↔Z swap.
    if gate == 'H':
        i = site
        tmp = xs[:, i].copy()
        xs[:, i] = zs[:, i]
        zs[:, i] = tmp
        sgn[:] = (sgn + xs[:, i] * zs[:, i]) % 2
        return state

    # Phase gate on qubit i: Z_i ← Z_i ⊕ X_i  (S gate).
    if gate == 'P':
        i = site
        zs[:, i] = (zs[:, i] + xs[:, i]) % 2
        sgn[:] = (sgn + xs[:, i] * zs[:, i]) % 2
        return state

    # Pauli X on qubit i: flips sign wherever generator has Z_i = 1.
    if gate == 'X':
        sgn[:] = (sgn + zs[:, site]) % 2
        return state

    # Pauli Y on qubit i: flips sign wherever generator has X_i=1 or Z_i=1.
    if gate == 'Y':
        sgn[:] = (sgn + xs[:, site] + zs[:, site]) % 2
        return state

    # Pauli Z on qubit i: flips sign wherever generator has X_i = 1.
    if gate == 'Z':
        sgn[:] = (sgn + xs[:, site]) % 2
        return state

    # CNOT(control c, target t):
    #   X propagates forward  (X_c ⊗ I  →  X_c ⊗ X_t)
    #   Z propagates backward (I ⊗ Z_t  →  Z_c ⊗ Z_t)
    if gate in ('CNOT', 'CX'):
        c, t = int(site[0]), int(site[1])
        xs[:, t] = (xs[:, t] + xs[:, c]) % 2
        zs[:, c] = (zs[:, c] + zs[:, t]) % 2
        sgn[:] = (sgn + xs[:, c] * zs[:, t]
                  * (xs[:, t] + zs[:, c] + 1)) % 2
        return state

    # Composite gates (e.g. 'PH', 'HPZ'): decompose left-to-right into
    # single characters and apply each in sequence.
    if gate in _COMPOSITE_GATES:
        for ch in gate:
            clifford_operation(state, ch, site)
        return state

    raise ValueError(f"Unknown gate: {gate}")


# ============================================================
# Gauge Transformation  (Aaronson–Gottesman row-sum)
# ============================================================

def gauge_transformation(state, R):
    """Row transformation with Aaronson–Gottesman phase tracking.

    Computes  new_generator[k] = product of old generators selected by R[k,:].
    The phase arising from multiplying Pauli operators is computed via the
    row-sum formula from PRA 70, 052328 (2004).

    Modifies *state* in-place and returns it.
    """
    R = np.asarray(R, dtype=np.int8) % 2
    n = state.n
    xs = state.xs
    zs = state.zs
    sgn = state.sign_vector

    # ------------------------------------------------------------------
    # Step 1: linear part of sign (XOR of selected original signs).
    # If new generator k is the product of old generators {i : R[k,i]=1},
    # the "naive" new sign is the XOR of those original signs.  This
    # ignores the extra phase that arises from multiplying non-commuting
    # Pauli operators; that correction is computed in step 2.
    # ------------------------------------------------------------------
    sign_temp = (R.astype(int) @ sgn.astype(int)) % 2

    # ------------------------------------------------------------------
    # Step 2: phase correction from Pauli multiplication.
    # When generators g1, g2, ... are multiplied in sequence, each qubit
    # contributes a phase factor of i^f where f depends on the two Pauli
    # types being multiplied.  The total extra phase (mod 4) is converted
    # to a sign flip (mod 2).  This is the "row-sum" procedure described
    # in Aaronson & Gottesman, PRA 70, 052328 (2004), Table III.
    #
    # For each qubit position the per-step contribution is:
    #   Y · Y: 0,   Y · Z: +1/2,  Y · X: -1/2,   Y · I: 0
    #   X · Y: +1/2,  X · X: 0,   X · Z: -1/2,   X · I: 0
    #   Z · X: +1/2,  Z · Z: 0,   Z · Y: -1/2,   Z · I: 0
    #   I · *: 0
    # These are encoded compactly in the three-branch formula below.
    # ------------------------------------------------------------------
    new_sign = np.zeros(n, dtype=np.int8)
    for k in range(n):
        rows = np.where(R[k] != 0)[0]
        m = len(rows)
        if m <= 1:
            # Only zero or one generator contributes → no multiplication phase.
            new_sign[k] = int(sign_temp[k])
            continue

        x_mat = xs[rows].astype(np.float64)
        z_mat = zs[rows].astype(np.float64)
        extra_sign = 0.0

        # Running product: accumulate row-by-row.
        x_rr = x_mat[0].copy()
        z_rr = z_mat[0].copy()
        for idx in range(1, m):
            x2 = x_mat[idx]
            z2 = z_mat[idx]
            # Vectorised per-qubit phase (sum over all qubit positions at once).
            extra_sign += np.sum(
                (x_rr == 1) * (z_rr == 1) * (z2 - x2)
                + (x_rr == 1) * (z_rr == 0) * (z2 * (2 * x2 - 1))
                + (x_rr == 0) * (z_rr == 1) * (x2 * (1 - 2 * z2))
            ) / 2.0
            # Update running product (binary addition = XOR).
            x_rr = (x_rr + x2) % 2
            z_rr = (z_rr + z2) % 2
        new_sign[k] = (int(sign_temp[k]) + int(round(extra_sign))) % 2

    state.sign_vector = new_sign

    # ------------------------------------------------------------------
    # Step 3: apply the row transformation to the X and Z matrices.
    # new_xs = R · xs (mod 2),  new_zs = R · zs (mod 2).
    # Each new generator row is the XOR of the selected old rows.
    # ------------------------------------------------------------------
    state.xs = ((R.astype(int) @ xs.astype(int)) % 2).astype(np.int8)
    state.zs = ((R.astype(int) @ zs.astype(int)) % 2).astype(np.int8)
    return state


# ============================================================
# Echelon Form
# ============================================================

def echelon_tableau(xs, zs):
    """Row-echelon form using the alternating-column ordering from NJP 7, 170.

    The NJP convention interleaves X and Z columns as [x0,z0,x1,z1,...].
    Gaussian elimination is then performed on this 2n-wide matrix using
    column-major pivot selection (matching MATLAB's ``find(...,1)``).
    The result is an upper-echelon form that respects the physical pairing
    of X and Z on each qubit.

    Parameters
    ----------
    xs, zs : ndarray (n, n) – X and Z binary matrices.

    Returns
    -------
    xs_echelon, zs_echelon : ndarray (n, n) – echelon-form X / Z matrices.
    R : ndarray (n, n) – binary row-transformation matrix such that
        (xs_ech, zs_ech) = R applied to the input rows.
    """
    xs = np.asarray(xs, dtype=np.int8).copy()
    zs = np.asarray(zs, dtype=np.int8).copy()
    n = xs.shape[0]
    R = np.eye(n, dtype=np.int8)

    # Pre-sort: push all-zero rows to the bottom so the elimination loop
    # only touches non-trivial generators.
    nz_mask = ~(np.all(xs == 0, axis=1) & np.all(zs == 0, axis=1))
    perm = np.concatenate([np.where(nz_mask)[0],
                           np.where(~nz_mask)[0]]).astype(int)
    xs = xs[perm]; zs = zs[perm]; R = R[perm]
    nz_row_num = int(np.sum(nz_mask))

    # Build the alternating-column matrix for pivot search.
    tab_alt = np.empty((n, 2 * n), dtype=np.int8)
    tab_alt[:, 0::2] = xs   # even columns ← X
    tab_alt[:, 1::2] = zs   # odd  columns ← Z

    # Forward elimination: for each working row i, find the first 1 in the
    # remaining sub-matrix (column-major scan), swap it into position, then
    # XOR-eliminate all rows below in that pivot column.
    for i in range(max(0, nz_row_num - 1)):
        sub = tab_alt[i:nz_row_num, :]
        # Transposing converts column-major to row-major for argwhere.
        hits = np.argwhere(sub.T == 1)
        if hits.size == 0:
            break
        cpiv = int(hits[0, 0])           # pivot column (in alternating order)
        rpiv = i + int(hits[0, 1])       # pivot row    (in full matrix)

        if rpiv != i:
            tab_alt[[i, rpiv]] = tab_alt[[rpiv, i]]
            R[[i, rpiv]] = R[[rpiv, i]]

        # XOR-eliminate all rows below that have a 1 in the pivot column.
        elim = np.where(tab_alt[i + 1:nz_row_num, cpiv] == 1)[0] + (i + 1)
        if elim.size:
            tab_alt[elim] = (tab_alt[elim] + tab_alt[i]) % 2
            R[elim] = (R[elim] + R[i]) % 2

    # De-interleave back to separate X / Z matrices.
    return tab_alt[:, 0::2].astype(np.int8), tab_alt[:, 1::2].astype(np.int8), R


# ============================================================
# Bigram
# ============================================================

def tableau_to_bigram(xs, zs):
    """Return bigram array (nz, 2) with 0-indexed leftmost/rightmost positions. (Phys. Rev. B 100, 134306 (2019))

    A generator's *support* is the set of qubit indices where the Pauli is
    non-identity (i.e. xs[i,j]+zs[i,j] != 0).  The bigram records the
    leftmost and rightmost support positions for each non-zero row.
    """
    nz_mask = ~(np.all(xs == 0, axis=1) & np.all(zs == 0, axis=1))
    nz = int(np.sum(nz_mask))
    bigram = np.zeros((nz, 2), dtype=int)
    for i in range(nz):
        support = (xs[i] + zs[i]) != 0
        positions = np.where(support)[0]
        bigram[i, 0] = positions[0]
        bigram[i, 1] = positions[-1]
    return bigram


# ============================================================
# Height Function
# ============================================================

def height_function(xs, zs):
    """Compute the height function h(x) from stabilizer generators (NJP 7, 170).

    The height function measures how many entanglement bonds cross each
    bipartition of the qubit chain.  Its maximum value equals the minimum
    number of emitter qubits required to produce the photonic state.

    Algorithm
    ---------
    1. Reverse the qubit ordering (so the echelon form is computed from
       the right-hand side of the chain).
    2. Put the reversed generators into echelon form and extract bigrams.
    3. For each cut position x = 1..n, count how many bigrams start at or
       beyond the corresponding threshold — the height is (x minus that count).

    Parameters
    ----------
    xs, zs : ndarray (n, n)

    Returns
    -------
    h : ndarray of length n+1.  h[0] = 0 (by definition), h[i] = h(x = i)
        for i = 1..n.
    """
    n = xs.shape[0]
    rv = np.arange(n - 1, -1, -1)
    xs_rev = xs[:, rv]
    zs_rev = zs[:, rv]

    xs_ech, zs_ech, _ = echelon_tableau(xs_rev, zs_rev)
    B = tableau_to_bigram(xs_ech, zs_ech)

    # Vectorised: for each cut i, count generators whose left endpoint in
    # the reversed ordering is >= (n - 1 - i), i.e. they span past the cut.
    thresholds = n - 1 - np.arange(n)
    counts = np.sum(B[:, 0].reshape(1, -1) >= thresholds.reshape(-1, 1), axis=1)
    h_body = np.arange(1, n + 1, dtype=int) - counts
    return np.concatenate([[0], h_body])  # prepend h(0) = 0


# ============================================================
# Module-level helpers for parallel permutation search
# (must be at module level so they are picklable by the
#  'spawn' start method used on macOS and Windows)
# ============================================================

_worker_xs = None
_worker_zs = None


def _init_worker(xs, zs):
    """Pool-initializer: store shared matrices in each worker process."""
    global _worker_xs, _worker_zs
    _worker_xs = xs
    _worker_zs = zs


def _eval_perm(perm):
    """Worker function: compute hmax for a single column permutation.

    Returns (hmax, h, perm_array) so the caller can track the best.
    """
    p = np.asarray(perm, dtype=int)
    h = height_function(_worker_xs[:, p], _worker_zs[:, p])
    return int(np.max(h)), h, p


def optimal_emission_order(xs, zs, n_samples=10000, seed=None):
    """Find a qubit ordering (permutation) that minimises max(h).

    The height function—and therefore the number of emitters needed—depends
    on the order in which photons are emitted.  This function searches over
    qubit permutations to find one that yields the smallest peak height.

    The search is parallelised across multiple CPU cores using
    ``multiprocessing.Pool``.  The implementation is compatible with both
    macOS and Windows (which use the *spawn* start method).

    Strategy
    --------
    * If n <= 10, an **exhaustive search** over all n! permutations is used.
    * If n > 10, a **random search** over *n_samples* uniformly random
      permutations is performed (the identity is always included).

    Parameters
    ----------
    xs, zs     : ndarray (n, n) – graph-state X / Z matrices.
    n_samples  : int – number of random permutations to try when n > 10.
    seed       : int or None – random seed for reproducibility.

    Returns
    -------
    best_perm  : ndarray of length n – the best permutation found.
    best_h     : ndarray of length n+1 – the height function under that
                 permutation.
    best_hmax  : int – the peak height, i.e. the minimum number of emitters.
    """
    import multiprocessing as mp
    import os
    from itertools import permutations as _perms

    n = xs.shape[0]
    rng = np.random.default_rng(seed)

    best_hmax = n + 1  # upper bound
    best_perm = np.arange(n)
    best_h = height_function(xs, zs)

    # Determine the number of worker processes.
    n_workers = max(1, os.cpu_count() or 1)

    if n <= 10:
        # Exhaustive search over all n! permutations in parallel.
        with mp.Pool(processes=n_workers,
                     initializer=_init_worker,
                     initargs=(xs, zs)) as pool:
            for hmax, h, perm in pool.imap_unordered(
                    _eval_perm, _perms(range(n)), chunksize=512):
                if hmax < best_hmax:
                    best_hmax = hmax
                    best_perm = perm
                    best_h = h
                    if best_hmax == 1:
                        break  # can't do better than 1
    else:
        # Random search: pre-generate all permutations, then evaluate
        # them in parallel.  The identity is always tried first.
        perms_to_try = [np.arange(n)]
        for _ in range(n_samples):
            perms_to_try.append(rng.permutation(n))

        with mp.Pool(processes=n_workers,
                     initializer=_init_worker,
                     initargs=(xs, zs)) as pool:
            for hmax, h, perm in pool.imap_unordered(
                    _eval_perm, perms_to_try, chunksize=128):
                if hmax < best_hmax:
                    best_hmax = hmax
                    best_perm = perm
                    best_h = h
                    if best_hmax == 1:
                        break

    return best_perm, best_h, best_hmax


# ============================================================
# Local Measurement
# ============================================================

def local_measurement(state, sign, direction, site):
    """Perform a projective Pauli measurement on a single qubit.

    The measurement projects onto the +1 or -1 eigenstate of the chosen
    Pauli operator (X, Y, or Z) at the given qubit.  The stabilizer
    formalism handles this in three cases:

    * **No anticommuting generator** (n_A = 0): the measurement outcome is
      deterministic and the state is unchanged.
    * **Exactly one anticommuting generator** (n_A = 1): replace that
      generator with the measured Pauli and set its sign to the outcome.
    * **Multiple anticommuting generators** (n_A > 1): first gauge-transform
      (multiply rows) so that only one generator anticommutes, then recurse.

    Parameters
    ----------
    state     : StabilizerState (modified in-place)
    sign      : int 0 or 1 – measurement outcome (0 → '+', 1 → '-')
    direction : 'x', 'y', 'z'
    site      : int – qubit index (0-indexed)
    """
    n = state.n

    # Represent the measured Pauli as separate x / z indicator vectors.
    gx = np.zeros(n, dtype=np.int8)
    gz = np.zeros(n, dtype=np.int8)
    if direction == 'x':
        gx[site] = 1
    elif direction == 'y':
        gx[site] = 1
        gz[site] = 1
    elif direction == 'z':
        gz[site] = 1

    # Symplectic inner product (mod 2) checks which generators anticommute
    # with the measured Pauli:  anti[i] = (xs[i]·gz + zs[i]·gx) mod 2.
    # Two Paulis anticommute iff their symplectic inner product is 1.
    anti = (
        state.xs.astype(np.int16) @ gz.astype(np.int16)
        + state.zs.astype(np.int16) @ gx.astype(np.int16)
    ) % 2
    anti = anti.astype(np.int8)
    anti_bool = anti.astype(bool)
    n_A = int(np.sum(anti_bool))

    if n_A == 0:
        pass  # deterministic outcome — state unchanged
    elif n_A == 1:
        # Replace the unique anticommuting generator with the measured Pauli.
        idx = int(np.where(anti_bool)[0][0])
        state.xs[idx] = gx
        state.zs[idx] = gz
        state.sign_vector[idx] = sign
    else:
        # Gauge-transform: multiply all anticommuting generators into the
        # first one, so that exactly one generator anticommutes, then recurse.
        R = np.eye(n, dtype=np.int8)
        ind1st = int(np.where(anti_bool)[0][0])
        R[:, ind1st] = anti
        state = gauge_transformation(state, R)
        state = local_measurement(state, sign, direction, site)
    return state


# ============================================================
# Helpers for the circuit solver
# ============================================================

def _make_op_record(n_p):
    """Create an empty operation/inv-operation record (dict of lists)."""
    return {
        'photons_Up_type': [None] * n_p,
        'Ue_type': [[] for _ in range(n_p)],
        'Ue_site': [[] for _ in range(n_p)],
        'W_type':  [[] for _ in range(n_p)],
        'W_site':  [[] for _ in range(n_p)],
        'W0_type': [],
        'W0_site': [],
        'EmissionSite':    [None] * n_p,
        'MeasurementSite': [None] * n_p,
    }


def _get_pauli(x_row, z_row, site):
    """Return (x, z) Pauli at *site* from separate X and Z row vectors."""
    return int(x_row[site]), int(z_row[site])


def _emitter_positions(x_row, z_row, n_p, n_e):
    """Return emitter indices (0-based inside emitter block) with non-trivial Pauli."""
    return np.where((x_row[n_p:n_p + n_e] + z_row[n_p:n_p + n_e]) != 0)[0]


def _pauli_to_gate_pair(xz, mode, *, strict=False):
    """Map local Pauli (x,z) to (forward_gate, inverse_gate) by mode.

    Modes
    -----
    'to_z' : turn Pauli to Z.  X→H, Y→PH, Z→I  (photon LC / Ue / W0-single)
    'to_x' : turn Pauli to X.  X→I, Y→P,  Z→H  (W / W0-multi branch)
    """
    if mode == 'to_z':
        mapping = {(1, 0): ('H', 'H'), (1, 1): ('PH', 'HPZ'), (0, 1): ('I', 'I')}
    elif mode == 'to_x':
        mapping = {(1, 0): ('I', 'I'), (1, 1): ('P', 'PZ'), (0, 1): ('H', 'H')}
    else:
        raise ValueError(f"Unknown mapping mode: {mode}")
    if xz in mapping:
        return mapping[xz]
    if strict:
        raise RuntimeError(f"Unexpected Pauli value: {xz} in mode={mode}")
    return mapping[(0, 1)]


# ============================================================
# Circuit Solver
# ============================================================

def circuit_solver(generators):
    """Find the photon-emission protocol for a given graph-state.

    The algorithm works backwards through the photon chain (j = n_p−1 down
    to 0).  At each step it identifies the emitter that will emit photon j,
    determines the required Clifford gates and (optionally) an intermediate
    measurement, and records both the "forward" operation (for the circuit
    diagram) and the "inverse" operation (for verification via simulation).

    The key steps per photon round are:
      1. Put the current generators in echelon gauge.
      2. Compute the height function; if dh = −1 a measurement-assisted
         emitter reset is needed (the "W" branch).
      3. Find g_a (the generator whose left endpoint is the current photon),
         apply a local Clifford on the photon to make g_a's Pauli there = Z.
      4. Apply Clifford gates on emitters (U_e) to reduce g_a to a single
         Z on one emitter, then perform the absorption CNOT.
      5. Gauge-transform to eliminate photon-j from other generators.

    After all photon rounds, a final "W0" stage resets the emitters to +Z.

    Parameters
    ----------
    generators : StabilizerState  (n_p qubits, photon-only)

    Returns
    -------
    operation, inv_operation : dict – forward / inverse operation records
    stat : dict – statistics (PhotonsNumber, EmittersNumber, HeightFunc, …)
    """
    xs_phi = generators.xs.copy()
    zs_phi = generators.zs.copy()
    sign_vec = generators.sign_vector.copy()
    n_p = generators.n
    h0 = height_function(xs_phi, zs_phi)
    n_e = int(np.max(h0))
    n = n_p + n_e

    op = _make_op_record(n_p)
    inv_op = _make_op_record(n_p)

    # Build expanded state (photons + emitters).
    # Photon generators keep their X/Z; emitter generators start as +Z.
    xs_temp = np.zeros((n, n), dtype=np.int8)
    zs_temp = np.zeros((n, n), dtype=np.int8)
    xs_temp[:n_p, :n_p] = xs_phi
    zs_temp[:n_p, :n_p] = zs_phi
    zs_temp[n_p:n, n_p:n] = np.eye(n_e, dtype=np.int8)

    g_temp = StabilizerState(xs_temp, zs_temp,
                             np.concatenate([sign_vec, np.zeros(n_e, dtype=np.int8)]))

    # ---- main loop (photon rounds, reverse order) ----
    for k in range(n_p):
        j = n_p - 1 - k  # current photon index (0-based)

        _, _, R = echelon_tableau(g_temp.xs, g_temp.zs)
        g_temp = gauge_transformation(g_temp, R)
        bigram = tableau_to_bigram(g_temp.xs, g_temp.zs)
        h = height_function(g_temp.xs, g_temp.zs)

        # h[0] = 0 by convention; photon j (0-based) corresponds to h(x=j+1).
        dh = h[j + 1] - h[j]

        if dh in (0, 1):
            pass  # nothing to do
        elif dh == -1:
            # Height drops: measurement-assisted emitter update needed.
            # Find g_b (first generator starting in emitter region).
            b = int(np.where(bigram[:, 0] >= n_p)[0][0])
            xb = g_temp.xs[b].copy()
            zb = g_temp.zs[b].copy()

            e_v_b = _emitter_positions(xb, zb, n_p, n_e)
            mu = n_p + int(e_v_b[0])

            if len(e_v_b) == 1:
                s = mu
                xz = _get_pauli(xb, zb, s)
                gs, gsi = _pauli_to_gate_pair(xz, 'to_x')
                clifford_operation(g_temp, gs, s)
                op['W_type'][j].append(gs);  op['W_site'][j].append(s)
                inv_op['W_type'][j].append(gsi); inv_op['W_site'][j].append(s)
            else:
                for ie in range(len(e_v_b)):
                    s = n_p + int(e_v_b[ie])
                    xz = _get_pauli(xb, zb, s)
                    gs, gsi = _pauli_to_gate_pair(xz, 'to_x')
                    clifford_operation(g_temp, gs, s)
                    op['W_type'][j].append(gs);  op['W_site'][j].append(s)
                    inv_op['W_type'][j].append(gsi); inv_op['W_site'][j].append(s)

                e_v_b_remain = np.setdiff1d(e_v_b, [e_v_b[0]])
                for ie in range(len(e_v_b_remain)):
                    s = n_p + int(e_v_b_remain[ie])
                    clifford_operation(g_temp, 'CNOT', [mu, s])
                    op['W_type'][j].append('CNOT'); op['W_site'][j].append([mu, s])
                    inv_op['W_type'][j].append('CNOT'); inv_op['W_site'][j].append([mu, s])

            # Fix sign of g_b.
            if g_temp.sign_vector[b] == 1:
                clifford_operation(g_temp, 'Z', mu)
                op['W_type'][j].append('Z');  op['W_site'][j].append(mu)
                inv_op['W_type'][j].append('Z'); inv_op['W_site'][j].append(mu)

            # CNOT to boost h(x); record measurement site.
            op['MeasurementSite'][j] = mu
            inv_op['MeasurementSite'][j] = mu
            clifford_operation(g_temp, 'CNOT', [mu, j])

            # Refresh echelon gauge.
            _, _, R = echelon_tableau(g_temp.xs, g_temp.zs)
            g_temp = gauge_transformation(g_temp, R)
            bigram = tableau_to_bigram(g_temp.xs, g_temp.zs)

        # --- Find g_a (last generator with leftmost position == j) ---
        a = int(np.where(bigram[:, 0] == j)[0][-1])
        xa = g_temp.xs[a].copy()
        za = g_temp.zs[a].copy()

        # LC on photon → turn Pauli at j to Z.
        xz = _get_pauli(xa, za, j)
        gs, gsi = _pauli_to_gate_pair(xz, 'to_z')
        clifford_operation(g_temp, gs, j)
        op['photons_Up_type'][j] = gs
        inv_op['photons_Up_type'][j] = gsi

        # Non-trivial emitter positions on g_a.
        e_v_a = _emitter_positions(xa, za, n_p, n_e)
        eta = n_p + int(e_v_a[0])
        op['EmissionSite'][j] = eta
        inv_op['EmissionSite'][j] = eta

        # Clifford operations on emitters: U_e (turn to Z).
        if len(e_v_a) == 1:
            s = eta
            xz = _get_pauli(xa, za, s)
            gs, gsi = _pauli_to_gate_pair(xz, 'to_z')
            clifford_operation(g_temp, gs, s)
            op['Ue_type'][j].append(gs);  op['Ue_site'][j].append(s)
            inv_op['Ue_type'][j].append(gsi); inv_op['Ue_site'][j].append(s)
        else:
            for ie in range(len(e_v_a)):
                s = n_p + int(e_v_a[ie])
                xz = _get_pauli(xa, za, s)
                gs, gsi = _pauli_to_gate_pair(xz, 'to_z')
                clifford_operation(g_temp, gs, s)
                op['Ue_type'][j].append(gs);  op['Ue_site'][j].append(s)
                inv_op['Ue_type'][j].append(gsi); inv_op['Ue_site'][j].append(s)

            e_v_a_remain = np.setdiff1d(e_v_a, [e_v_a[0]])
            for ie in range(len(e_v_a_remain)):
                s = n_p + int(e_v_a_remain[ie])
                clifford_operation(g_temp, 'CNOT', [s, eta])
                op['Ue_type'][j].append('CNOT'); op['Ue_site'][j].append([s, eta])
                inv_op['Ue_type'][j].append('CNOT'); inv_op['Ue_site'][j].append([s, eta])

        # Fix sign of g_a.
        if g_temp.sign_vector[a] == 1:
            clifford_operation(g_temp, 'X', eta)
            op['Ue_type'][j].append('X');  op['Ue_site'][j].append(eta)
            inv_op['Ue_type'][j].append('X'); inv_op['Ue_site'][j].append(eta)

        # Absorption (inverted emission): CNOT from emitter to photon.
        clifford_operation(g_temp, 'CNOT', [eta, j])

        # Row transformation: eliminate photon-j support from other generators.
        R = np.eye(n, dtype=np.int8)
        col_support = ((g_temp.xs[:, j].astype(int)
                        + g_temp.zs[:, j].astype(int)) != 0).astype(np.int8)
        R[:, a] = col_support
        g_temp = gauge_transformation(g_temp, R)

    # ---- W0: final emitter reset ----
    _, _, R = echelon_tableau(g_temp.xs, g_temp.zs)
    g_temp = gauge_transformation(g_temp, R)

    for i_r in range(n_p, n):
        xc = g_temp.xs[i_r].copy()
        zc = g_temp.zs[i_r].copy()
        e_v_c = _emitter_positions(xc, zc, n_p, n_e)
        kappa = n_p + int(e_v_c[0])

        if len(e_v_c) == 1:
            xz = _get_pauli(xc, zc, kappa)
            gs, gsi = _pauli_to_gate_pair(xz, 'to_z', strict=True)
            clifford_operation(g_temp, gs, kappa)
            op['W0_type'].append(gs);  op['W0_site'].append(kappa)
            inv_op['W0_type'].append(gsi); inv_op['W0_site'].append(kappa)
        else:
            for ie in range(len(e_v_c)):
                s = n_p + int(e_v_c[ie])
                xz = _get_pauli(xc, zc, s)
                gs, gsi = _pauli_to_gate_pair(xz, 'to_x')
                clifford_operation(g_temp, gs, s)
                op['W0_type'].append(gs);  op['W0_site'].append(s)
                inv_op['W0_type'].append(gsi); inv_op['W0_site'].append(s)

            e_v_c_remain = np.setdiff1d(e_v_c, [e_v_c[0]])
            for ie in range(len(e_v_c_remain)):
                s = n_p + int(e_v_c_remain[ie])
                clifford_operation(g_temp, 'CNOT', [kappa, s])
                op['W0_type'].append('CNOT'); op['W0_site'].append([kappa, s])
                inv_op['W0_type'].append('CNOT'); inv_op['W0_site'].append([kappa, s])

            # Recover X → Z on the anchor emitter.
            clifford_operation(g_temp, 'H', kappa)
            op['W0_type'].append('H');  op['W0_site'].append(kappa)
            inv_op['W0_type'].append('H'); inv_op['W0_site'].append(kappa)

        # Row transformation: eliminate all other Zs in the kappa column.
        R = np.eye(n, dtype=np.int8)
        R[:, i_r] = ((g_temp.xs[:, kappa] == 0) &
                      (g_temp.zs[:, kappa] == 1)).astype(np.int8)
        g_temp = gauge_transformation(g_temp, R)

    # Final echelon.
    _, _, R = echelon_tableau(g_temp.xs, g_temp.zs)
    g_temp = gauge_transformation(g_temp, R)

    # Fix remaining emitter signs.
    for i_e in range(n_p, n):
        if g_temp.sign_vector[i_e] == 1:
            gs, gsi = 'X', 'X'
        else:
            gs, gsi = 'I', 'I'
        clifford_operation(g_temp, gs, i_e)
        op['W0_type'].append(gs);  op['W0_site'].append(i_e)
        inv_op['W0_type'].append(gsi); inv_op['W0_site'].append(i_e)

    # Verify: all generators should now be +Z_i.
    if (np.all(g_temp.xs == 0) and
            np.all(g_temp.zs == np.eye(n, dtype=np.int8)) and
            np.all(g_temp.sign_vector == 0)):
        print('\t** The protocol is SOLVED correctly! **')
    else:
        raise RuntimeError('\t** The protocol is NOT solved correctly! **')

    # ---- Statistics ----
    def _count_nontrivial(type_list):
        cnt = 0
        for item in type_list:
            if isinstance(item, list):
                cnt += _count_nontrivial(item)
            elif item is not None and item not in ('I', ''):
                cnt += 1
        return cnt

    def _count_cnot(type_list):
        cnt = 0
        for item in type_list:
            if isinstance(item, list):
                cnt += _count_cnot(item)
            elif item == 'CNOT':
                cnt += 1
        return cnt

    stat = {
        'PhotonsNumber': n_p,
        'EmittersNumber': n_e,
        'HeightFunc': h0,
        'OperationNumber': {
            'Ue': _count_nontrivial(op['Ue_type']),
            'W':  _count_nontrivial(op['W_type']),
            'W0': _count_nontrivial(op['W0_type']),
            'Measurement': sum(1 for s in op['MeasurementSite'] if s is not None),
            'CNOT': (_count_cnot(op['Ue_type'])
                     + _count_cnot(op['W_type'])
                     + _count_cnot(op['W0_type'])),
        },
    }
    stat['OperationNumber']['AllEmitterUnitaries'] = (
        stat['OperationNumber']['Ue']
        + stat['OperationNumber']['W']
        + stat['OperationNumber']['W0']
    )
    return op, inv_op, stat


# ============================================================
# Protocol Executor  (verification by forward simulation)
# ============================================================

def protocol_executor(inv_op, n_e, n_p):
    """Verify the protocol by running it forward on the standard |0…0⟩ state.

    Applies the inverse-operation record produced by :func:`circuit_solver`
    in chronological order: W0 (emitter preparation), then for each photon
    round the emission CNOT, photon LC, emitter U_e, and (if present) a
    measurement with classical feed-forward followed by W gates.

    Measurement outcomes are chosen uniformly at random; the resulting
    photon-only generators should always be gauge-equivalent to the original
    target state regardless of the random outcomes.

    Returns (g_f_recover, g_phi_recover) where g_phi_recover is photon-only.
    """
    n = n_e + n_p
    # Standard initial state: all generators are +Z.
    g_f = StabilizerState(
        np.zeros((n, n), dtype=np.int8),
        np.eye(n, dtype=np.int8),
        np.zeros(n, dtype=np.int8),
    )

    # W0 (reversed order).
    w0_list = inv_op['W0_type']
    w0_sites = inv_op['W0_site']
    for q in range(len(w0_list)):
        idx = len(w0_list) - 1 - q
        gs = w0_list[idx]
        if gs is not None and gs not in ('I', ''):
            clifford_operation(g_f, gs, w0_sites[idx])

    # Photon rounds.
    for j in range(n_p):
        clifford_operation(g_f, 'CNOT', [inv_op['EmissionSite'][j], j])

        gs = inv_op['photons_Up_type'][j]
        if gs is not None and gs not in ('I', ''):
            clifford_operation(g_f, gs, j)

        ue_list = inv_op['Ue_type'][j]
        ue_sites = inv_op['Ue_site'][j]
        for q in range(len(ue_list)):
            idx = len(ue_list) - 1 - q
            gs = ue_list[idx]
            if gs is not None and gs not in ('I', ''):
                clifford_operation(g_f, gs, ue_sites[idx])

        mu = inv_op['MeasurementSite'][j]
        if mu is not None:
            s = np.random.randint(0, 2)
            g_f = local_measurement(g_f, s, 'z', mu)
            if s == 1:
                clifford_operation(g_f, 'X', mu)
                clifford_operation(g_f, 'X', j)
            clifford_operation(g_f, 'H', mu)

            w_list = inv_op['W_type'][j]
            w_sites = inv_op['W_site'][j]
            for q in range(len(w_list)):
                idx = len(w_list) - 1 - q
                gs = w_list[idx]
                if gs is not None and gs not in ('I', ''):
                    clifford_operation(g_f, gs, w_sites[idx])

    g_f_recover = g_f

    # Extract photon-only generators.
    _, _, R = echelon_tableau(g_f.xs, g_f.zs)
    g_f = gauge_transformation(g_f, R)
    g_phi_recover = StabilizerState(
        g_f.xs[:n_p, :n_p].copy(),
        g_f.zs[:n_p, :n_p].copy(),
        g_f.sign_vector[:n_p].copy(),
    )
    return g_f_recover, g_phi_recover


# ============================================================
# Generators Equivalence
# ============================================================

def rref_gf2(A):
    """Reduced row echelon form (RREF) over GF(2).

    Standard Gaussian elimination with full back-substitution, where all
    arithmetic is modulo 2.  Used by :func:`generators_equivalence` to find
    the gauge transformation relating two equivalent stabilizer sets.

    Returns (Arref, pivot_cols) where pivot_cols lists the column indices
    that contain pivot positions.
    """
    A = np.asarray(A, dtype=np.int8).copy() % 2
    m, nc = A.shape
    pivot_row = 0
    pivot_cols = []
    for col in range(nc):
        if pivot_row >= m:
            break
        found = -1
        for r in range(pivot_row, m):
            if A[r, col] == 1:
                found = r
                break
        if found == -1:
            continue
        if found != pivot_row:
            A[[pivot_row, found]] = A[[found, pivot_row]]
        pivot_cols.append(col)
        for r in range(m):
            if r != pivot_row and A[r, col] == 1:
                A[r] = (A[r] + A[pivot_row]) % 2
        pivot_row += 1
    return A, pivot_cols


def generators_equivalence(g1, g2):
    """Check if two stabilizer generator sets are gauge-equivalent.

    Two sets are gauge-equivalent if one can be obtained from the other by
    invertible binary row operations (i.e. they stabilise the same state).

    The check proceeds in three stages:
      1. **Symplectic orthogonality** — all generators of g1 must commute
         with all generators of g2 (necessary condition for same stabiliser
         group).  Computed as (xs1·zs2^T + zs1·xs2^T) mod 2 == 0.
      2. **Find the row transformation R** — solve g1^T · X = g2^T over
         GF(2) using RREF on the augmented matrix [g1^T | g2^T].
      3. **Sign check** — apply the gauge transformation R to g1 and verify
         that the resulting sign vector matches g2's.

    Returns dict with keys 'Overall', 'Tableau', 'SignVector' (all bool).
    """
    result = {'Overall': False, 'Tableau': False, 'SignVector': False}
    n = g1.n
    if g1.xs.shape != g2.xs.shape:
        print('Warning: generator sets have different tableau sizes!')
        return result

    # Symplectic orthogonality: all generators must commute pairwise.
    # (xs1 · zs2^T  +  zs1 · xs2^T) mod 2 == 0.
    comm = (g1.xs.astype(int) @ g2.zs.astype(int).T
            + g1.zs.astype(int) @ g2.xs.astype(int).T) % 2
    if not np.all(comm == 0):
        return result
    result['Tableau'] = True

    # Find row transformation R such that R @ g1 ≡ g2 (mod 2).
    # Stack [xs; zs] to form the 2n×n "full tableau transpose".
    A = np.vstack([g1.xs.T, g1.zs.T]).astype(np.int8)   # (2n × n)
    B = np.vstack([g2.xs.T, g2.zs.T]).astype(np.int8)   # (2n × n)
    M = np.hstack([A, B])                                # (2n × 2n)
    Mrref, _ = rref_gf2(M)
    X = Mrref[:n, n:2 * n].astype(np.int8)

    if not np.all((A.astype(int) @ X.astype(int)) % 2 == B.astype(int) % 2):
        print('AX != B (mod 2)')
        return result

    R = X.T
    g1_copy = g1.copy()
    g1_g = gauge_transformation(g1_copy, R)
    if np.all(g1_g.sign_vector == g2.sign_vector):
        result['SignVector'] = True

    if result['Tableau'] and result['SignVector']:
        result['Overall'] = True
    return result


# ============================================================
# Operation → quantikz (LaTeX circuit)
# ============================================================

def operation_to_quantikz(operation, n_p, n_e, filename):
    """Generate a quantikz LaTeX file for the quantum circuit.

    Translates the operation record from :func:`circuit_solver` into a
    ``quantikz`` environment (TikZ-based quantum circuit drawing package).
    The circuit is laid out on a sparse grid ``CC[(row, col)]`` where each
    row is a qubit wire (photons on top, emitters below) and columns advance
    left to right in time order.

    Layout phases:
      1. Wire labels and |0⟩ initialization boxes.
      2. W0 gates (emitter preparation, reversed iteration).
      3. For each photon round: emission CNOT, U_e + U_p gates, and
         (if present) measurement → re-init → H → W gates.
      4. Fill empty cells with ``\\qw``.
      5. Trim redundant double-H after re-initialization.
      6. Trim long ``\\qw`` tails to keep the output compact.

    Writes ``quantikz_<filename>.txt`` in the current directory.
    """
    n = n_p + n_e
    INI_P = '\\gate[style={fill=red!20}]{\\ket{0}}'
    INI_E = '\\gate[style={fill=green!20}]{\\ket{0}}'

    # ---- grid helpers ----
    CC = {}
    _max_col = [0]

    def _set(r, c, val):
        CC[(r, c)] = val
        if c > _max_col[0]:
            _max_col[0] = c

    def _get(r, c):
        return CC.get((r, c))

    def _col_has_content(c):
        return any(CC.get((r, c)) is not None for r in range(n))

    # ---- column pointer ----
    c = 0

    # Col 0: labels
    for i in range(n_p):
        _set(i, c, f'\\lstick{{$p_{{{i + 1}}}$}}')
    for i in range(n_p, n):
        _set(i, c, f'\\lstick{{$e_{{{i - n_p + 1}}}$}}')

    # Col 1: initialization
    c += 1
    for i in range(n_p):
        _set(i, c, INI_P)
    for i in range(n_p, n):
        _set(i, c, INI_E)

    def _place_split_gate(gate_type, site):
        nonlocal c
        if _get(site, c) is not None:
            c += 1
        L = len(gate_type)
        for s in range(L):
            ch = gate_type[L - 1 - s]
            _set(site, c + s, f'\\gate{{\\texttt{{{ch}}}}}')
        c = c + L - 1

    def _place_emitter_cnot(ctrl, tgt):
        nonlocal c
        if _col_has_content(c):
            c += 1
        d = tgt - ctrl
        for i in range(n_p, n):
            if i == ctrl:
                _set(i, c, f'\\ctrl{{{d}}}')
            elif i == tgt:
                _set(i, c, '\\targ{}')
            else:
                _set(i, c, '\\qw')
        c += 1

    # ======== W0 ========
    c += 1
    w0t = operation['W0_type']
    w0s = operation['W0_site']
    for k in range(len(w0t) - 1, -1, -1):
        gt = w0t[k]
        gs = w0s[k]
        if gt is not None and gt not in ('I', ''):
            if gt in ('CNOT', 'CX'):
                _place_emitter_cnot(gs[0], gs[1])
            else:
                _place_split_gate(gt, gs)

    # ======== Photon rounds ========
    for j in range(n_p):
        if _col_has_content(c):
            c += 1
        eta = operation['EmissionSite'][j]
        for i in range(n):
            if i == eta:
                _set(i, c, f'\\ctrl{{{j - i}}}')
            elif i == j:
                _set(i, c, '\\targ{}')
            else:
                _set(i, c, '\\qw')
        c += 1

        ue_t = list(operation['Ue_type'][j])
        ue_s = list(operation['Ue_site'][j])
        up_t = operation['photons_Up_type'][j]
        op_t = ue_t + [up_t]
        op_s = ue_s + [j]
        for k in range(len(op_t) - 1, -1, -1):
            gt = op_t[k]
            gs_site = op_s[k]
            if gt is not None and gt not in ('I', ''):
                if gt in ('CNOT', 'CX'):
                    _place_emitter_cnot(gs_site[0], gs_site[1])
                else:
                    _place_split_gate(gt, gs_site)

        mu = operation['MeasurementSite'][j]
        if mu is not None:
            if _col_has_content(c):
                c += 1
            d = j - mu
            for i in range(n):
                if i == mu:
                    _set(i, c, f'\\meter{{}} \\vcw{{{d}}}')
                elif i == j:
                    _set(i, c, '\\gate{\\texttt{X}}')
                else:
                    _set(i, c, '\\qw')
            c += 1

            for i in range(n):
                _set(i, c, INI_E if i == mu else '\\qw')
            c += 1
            for i in range(n):
                _set(i, c, '\\gate{\\texttt{H}}' if i == mu else '\\qw')
            c += 1

            wt = operation['W_type'][j]
            ws = operation['W_site'][j]
            for k in range(len(wt) - 1, -1, -1):
                gt = wt[k]
                gs_site = ws[k]
                if gt is not None and gt not in ('I', ''):
                    if gt in ('CNOT', 'CX'):
                        _place_emitter_cnot(gs_site[0], gs_site[1])
                    else:
                        if _get(gs_site, c) is not None:
                            c += 1
                        _set(gs_site, c, f'\\gate{{\\texttt{{{gt}}}}}')

    # ======== Fill empty cells with \qw ========
    total_cols = _max_col[0] + 1
    for r in range(n):
        for ci in range(total_cols):
            if _get(r, ci) is None:
                _set(r, ci, '\\qw')

    # ======== Trim around re-initialized emitter ========
    for r in range(n):
        for ci in range(total_cols):
            if _get(r, ci) == INI_E:
                if (ci + 2 < total_cols
                        and _get(r, ci + 1) == '\\gate{\\texttt{H}}'
                        and _get(r, ci + 2) == '\\gate{\\texttt{H}}'):
                    _set(r, ci + 1, '\\qw')
                    _set(r, ci + 2, '\\qw')

    # ======== Trim long \qw tails ========
    for r in range(n):
        last_non_qw = -1
        for ci in range(total_cols):
            if _get(r, ci) != '\\qw':
                last_non_qw = ci
        for ci in range(last_non_qw + 2, total_cols):
            _set(r, ci, '')

    # ======== Write file ========
    path = f'quantikz_{filename}.txt'
    with open(path, 'w') as f:
        f.write('\\begin{tikzpicture} \n')
        f.write('\t\\node[scale = 0.5]{\t\n')
        f.write('\t\t\\begin{quantikz} [row sep={0.7cm,between origins},'
                'column sep=0.12cm]\n')
        for r in range(n):
            f.write('\t\t\t')
            for ci in range(total_cols):
                cell = _get(r, ci)
                if cell is None:
                    cell = ''
                f.write(f' & {cell}')
            if r < n - 1:
                f.write(' \\\\   \n')
            else:
                f.write(' \n')
        f.write('\t\t\\end{quantikz} \n')
        f.write('\t}; \n')
        f.write('\\end{tikzpicture}')

    print(f'\t** The quantikz code for latex has been saved as: '
          f'quantikz_{filename}.txt **')


# ============================================================
# Plotting utilities
# ============================================================

def circle_tree_layout(bv):
    """Compute the circular tree layout from a branching vector.

    This is a Python translation of the MATLAB ``CircleTreeXY`` function.
    Nodes at each depth level are evenly distributed on a circle whose
    radius grows with depth, producing a visually balanced tree plot.

    Parameters
    ----------
    bv : list of int – branching vector (e.g. [3, 3, 3]).

    Returns
    -------
    pos : dict mapping 1-indexed node labels to (x, y) arrays,
          suitable for passing directly to :func:`plot_graph`.
    """
    bv = list(bv)

    # Reconstruct the 1-indexed parent-pointer vector (same as build_tree).
    nv = np.array([0] + [1] * bv[-1], dtype=int)
    for i in range(len(bv) - 2, -1, -1):
        L = len(nv) - 1
        b = bv[i]
        w = np.arange(1, b + 1) + np.arange(0, b) * L
        non_root = nv[1:]
        M = np.empty((L + 1, b), dtype=int)
        M[0, :] = 1
        for j in range(b):
            M[1:, j] = non_root + w[j]
        nv = np.concatenate([[0], M.flatten(order='F')])

    n_nodes = len(nv)
    X = np.zeros(n_nodes)
    Y = np.zeros(n_nodes)
    n_levels = len(bv)

    # Cumulative product of branching factors gives #nodes at each depth.
    p_v = np.cumprod(bv)

    # The root (index 0) stays at the origin; start from its children.
    inner_layer = np.array([0])  # 0-indexed
    phi = 0.0
    golden = 0.618

    for m in range(n_levels):
        # Children of the current inner layer.
        current_layer = []
        for parent_idx in inner_layer:
            children = np.where(nv == parent_idx + 1)[0]  # nv is 1-indexed
            current_layer.extend(children.tolist())
        current_layer = np.array(current_layer, dtype=int)

        if m == 0:
            phi = 0.0
        else:
            phi = phi + 2 * np.pi * ((bv[m] - 1) / 2) / p_v[m]

        r = 1.0 - golden ** (m + 1)
        angles = np.arange(p_v[m]) / p_v[m] * 2 * np.pi - phi
        X[current_layer] = r * np.cos(angles)
        Y[current_layer] = r * np.sin(angles)

        inner_layer = current_layer

    # Return as {1-indexed label: array([x, y])} for networkx.
    return {i + 1: np.array([X[i], Y[i]]) for i in range(n_nodes)}


def plot_graph(n, edges, title=None, pos=None):
    """Plot a graph with evenly distributed vertices and equal axis ratio.

    Parameters
    ----------
    n     : int – number of vertices (0-indexed internally; labels shown 1-indexed).
    edges : list of (i, j) tuples (0-indexed).
    title : str or None – optional plot title.
    pos   : dict or None – pre-computed node positions keyed by 1-indexed label.
            If *None*, the Kamada-Kawai layout is used automatically.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    G.add_edges_from([(i + 1, j + 1) for i, j in edges])

    # Use supplied positions or fall back to Kamada-Kawai.
    if pos is None:
        pos = nx.kamada_kawai_layout(G)

    # Adapt figure size to the layout's bounding box (clamp aspect 1:3 … 3:1).
    coords = np.array(list(pos.values()))
    dx = max(coords[:, 0].max() - coords[:, 0].min(), 1e-6)
    dy = max(coords[:, 1].max() - coords[:, 1].min(), 1e-6)
    ratio = min(max(dy / dx, 1 / 3), 3)
    base = max(6.0, n * 0.2)
    w, h_fig = (base, base * ratio) if dx >= dy else (base / ratio, base)

    node_sz = 400 if n <= 20 else 200
    font_sz = 14 if n <= 20 else 7

    fig, ax = plt.subplots(figsize=(w, h_fig))
    nx.draw(
        G, pos=pos, with_labels=True, ax=ax,
        node_color='red', node_size=node_sz,
        edge_color=[0.5, 0.5, 0.5], width=2,
        font_size=font_sz, font_color='white',
    )
    ax.set_aspect('equal')
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def plot_height_function(h, title=None):
    """Plot the height function h(x).

    Parameters
    ----------
    h     : 1-d array of length n+1 (as returned in stat['HeightFunc']),
            where h[0] = 0 by convention.
    title : str or None – optional plot title.
    """
    import matplotlib.pyplot as plt

    n = len(h) - 1  # h includes h(0)=0, so number of qubits is len-1
    x_vals = np.arange(0, n + 1)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.3), 4))
    ax.plot(x_vals, h, ':ob')
    ax.grid(True)
    ax.set_xlabel(r'$x$', fontsize=14)
    ax.set_ylabel(r'$h_p(x)$', fontsize=14)
    ax.set_xticks(np.arange(0, n + 1))
    ax.set_yticks(np.arange(0, int(np.max(h)) + 2))
    ax.set_xlim(0, n)
    ax.set_ylim(0, int(np.max(h)) + 1)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def compile_quantikz_pdf(filename):
    """Compile a quantikz txt file to PDF using lualatex.

    Reads ``quantikz_<filename>.txt``, wraps it in a minimal LaTeX document,
    compiles with ``lualatex``, and removes auxiliary files.
    Produces ``quantikz_<filename>.pdf`` in the current directory.
    """
    import subprocess, os

    txt_path = f'quantikz_{filename}.txt'
    tex_path = f'quantikz_{filename}.tex'
    pdf_path = f'quantikz_{filename}.pdf'

    with open(txt_path, 'r') as f:
        content = f.read()

    tex_src = (
        '% !TeX program = lualatex\n'
        '\\documentclass{standalone}\n'
        '\\usepackage{tikz}\n'
        '\\usetikzlibrary{quantikz}\n'
        '\\begin{document}\n'
        + content + '\n'
        '\\end{document}\n'
    )
    with open(tex_path, 'w') as f:
        f.write(tex_src)

    result = subprocess.run(
        ['lualatex', '-interaction=nonstopmode', tex_path],
        capture_output=True, text=True,
    )

    # Clean up auxiliary files regardless of success.
    for ext in ('.aux', '.log', '.tex'):
        p = f'quantikz_{filename}{ext}'
        if os.path.exists(p):
            os.remove(p)

    if result.returncode != 0 or not os.path.exists(pdf_path):
        print(f'\t** ERROR compiling {tex_path} **')
        # Print the last 30 lines of lualatex output for diagnosis.
        lines = result.stdout.splitlines()
        for line in lines[-30:]:
            print(line)
        raise RuntimeError(f'lualatex failed for {tex_path}')

    print(f'\t** PDF compiled: {pdf_path} **')

    # Display the PDF inline when running inside a Jupyter notebook.
    try:
        from pdf2image import convert_from_path
        from IPython.display import display as _display
        images = convert_from_path(pdf_path, dpi=200)
        if images:
            _display(images[0])
    except ImportError:
        pass
    except Exception:
        pass
