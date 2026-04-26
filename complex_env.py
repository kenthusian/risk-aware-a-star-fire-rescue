import numpy as np

# ── Public constants ──────────────────────────────────────────────────────────
N        = 50
START    = (5,  23)
GOAL     = (44, 25)
ALPHA    = 0.8
W_WEIGHT = 2.0

LOBBY_R_LO, LOBBY_R_HI = 14, 35
LOBBY_C_LO, LOBBY_C_HI = 15, 34

# Class A multi-core distributed fire (cen_r, cen_c, peak, half_r, half_c)
CLASS_A = [
    (24.0, 24.0, 500,  6,  8),   # Main lobby inferno
    (20.0, 26.0, 480,  4,  5),   # Spreading north-east (pulled back from door)
    (29.0, 20.0, 460,  5,  4),   # Spreading south-west
]

# Class B fire zones  (cen_r, cen_c, peak, half_r, half_c)
CLASS_B = [
    (24.5, 16.5, 450,  2.5,  1.5),  # Inner Left Hallway (Fast dash through fire)
    (24.5, 32.5, 350,  2.5,  1.5),  # Inner Right Hallway (Medium dash)
]

# Spot fires  (cen_r, cen_c, peak, sigma)
SPOTS = [
    ( 8, 12, 110, 2.5),
    ( 8, 40,  95, 2.5),
    (42, 10, 105, 2.5),
    (42, 38,  90, 2.5),
]


# ── Wall builder ──────────────────────────────────────────────────────────────
def build_walls():
    w = np.zeros((N, N), bool)
    w[0, :] = w[N-1, :] = w[:, 0] = w[:, N-1] = True

    # -- NORTH WING (Executive Offices) --
    w[14, 1:49] = True
    for c in [9, 18, 27, 36, 45]:
        w[1:14, c] = True
    for c in [5, 14, 23, 32, 41]:
        w[14, c] = False

    # -- SOUTH WING (Standard Offices) --
    w[35, 1:49] = True
    for c in [7, 14, 21, 28, 35, 42]:
        w[35:49, c] = True
    for c in [4, 11, 18, 25, 32, 39, 46]:
        w[35, c] = False

    # -- WEST CORE (Elevator & Emergency Stairwell) --
    w[18:32, 5:15] = True   # Solid elevator bank (enclosing the hallway perfectly)
    w[18:32, 4] = True      # Stairwell inner wall
    w[18, 1:5] = True; w[18, 2] = False  # Stairwell North Door
    w[31, 1:5] = True; w[31, 2] = False  # Stairwell South Door

    # -- EAST WING (Large Conference Rooms) --
    w[18:32, 35] = True     # Glass corridor wall
    w[24, 35:49] = True     # Room divider
    w[21, 35] = False       # Door to Conf Room A
    w[29, 35] = False       # Door to Conf Room B
    w[24, 45] = False       # Internal connecting door

    # -- CENTRAL LOBBY (Reception Desk) --
    w[23:26, 21:28] = True
    w[24, 22:27] = False    # Hollow out desk for realism

    w[START] = False
    w[GOAL] = False
    return w


# ── Temperature builder ───────────────────────────────────────────────────────
def build_temperature(walls, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    temp = rng.uniform(14, 22, (N, N))
    temp[walls] = 0

    R, C = np.ogrid[:N, :N]
    
    mask = (temp > 0)
    
    lmask = (R >= LOBBY_R_LO) & (R <= LOBBY_R_HI) & (C >= LOBBY_C_LO) & (C <= LOBBY_C_HI)
    temp[lmask] = np.clip(rng.uniform(60, 100, (N, N))[lmask], 60, 500)
    
    for (ar, ac, apk, hr, hc) in CLASS_A:
        nd = np.sqrt(((R - ar) / hr)**2 + ((C - ac) / hc)**2)
        ragged_nd = nd + rng.uniform(-0.3, 0.3, (N, N))
        core = apk * np.exp(-1.3 * ragged_nd**2) + rng.uniform(-15, 15, (N, N))
        temp[mask] = np.clip(np.maximum(temp[mask], core[mask]), 1, 500)

    # Class B blobs
    for (br, bc, bpk, bhr, bhc) in CLASS_B:
        nd = np.sqrt(((R - br) / bhr)**2 + ((C - bc) / bhc)**2)
        ragged_nd = nd + rng.uniform(-0.4, 0.4, (N, N))
        core = bpk * np.exp(-1.5 * ragged_nd**2) + rng.uniform(-8, 8, (N, N))
        temp[mask] = np.clip(np.maximum(temp[mask], core[mask]), 1, 500)

    # Spot fires
    for (sr, sc, spk, sigma) in SPOTS:
        nd = np.sqrt((R - sr)**2 + (C - sc)**2)
        ragged_nd = nd + rng.uniform(-0.5, 0.5, (N, N))
        core = spk * np.exp(-0.8 * (ragged_nd / sigma)**2) + rng.uniform(-5, 5, (N, N))
        temp[mask] = np.clip(np.maximum(temp[mask], core[mask]), 1, 500)

    # Protect the entrances so detours are cleanly distinguished by their bottleneck
    temp[14:17, 20:26] = np.clip(temp[14:17, 20:26], 1, 85)
    temp[33:36, 22:28] = np.clip(temp[33:36, 22:28], 1, 85)
    
    # Establish perfectly uniform architectural corridor risks to prevent peak heat inversion
    temp[18:32, 2] = np.maximum(temp[18:32, 2], 120 + rng.uniform(-5, 5, 14))         # Far Left Stair (Peak ~125C, very long)
    temp[18:32, 35:49] = np.maximum(temp[18:32, 35:49], 200 + rng.uniform(-5, 5, (14, 14))) # East Conference (Peak ~205C, shorter detour)

    temp[walls] = 0.0
    return temp


def get_environment():
    """Returns (temperature_array, walls_array) for the 50x50 layout."""
    walls = build_walls()
    temp  = build_temperature(walls)
    return temp, walls


def build_env_scaled(N_sc, rng_seed=42):
    k = N_sc / 50.0
    rng = np.random.RandomState(rng_seed)
    w = np.zeros((N_sc, N_sc), bool)
    t = np.full((N_sc, N_sc), 20.0)

    w[0,:] = w[N_sc-1,:] = w[:,0] = w[:,N_sc-1] = True

    def r(val): return int(val * k)

    # -- NORTH WING --
    w[r(14), 1:N_sc-1] = True
    for c in [9, 18, 27, 36, 45]:
        cc = r(c); 
        if 0 < cc < N_sc-1: w[1:r(14), cc] = True
    for c in [5, 14, 23, 32, 41]:
        cc = r(c); 
        if 0 < cc < N_sc-1: w[r(14), cc] = False

    # -- SOUTH WING --
    w[r(35), 1:N_sc-1] = True
    for c in [7, 14, 21, 28, 35, 42]:
        cc = r(c); 
        if 0 < cc < N_sc-1: w[r(35):N_sc-1, cc] = True
    for c in [4, 11, 18, 25, 32, 39, 46]:
        cc = r(c); 
        if 0 < cc < N_sc-1: w[r(35), cc] = False

    # -- WEST CORE --
    w[r(18):r(32), r(5):r(15)] = True
    w[r(18):r(32), r(4)] = True
    w[r(18), 1:r(5)] = True; w[r(18), r(2)] = False
    w[r(31), 1:r(5)] = True; w[r(31), r(2)] = False

    # -- EAST CONFERENCE --
    w[r(18):r(32), r(35)] = True
    w[r(24), r(35):N_sc-1] = True
    w[r(21), r(35)] = False
    w[r(29), r(35)] = False
    w[r(24), r(45)] = False

    # -- CENTRAL LOBBY --
    w[r(23):r(26), r(21):r(28)] = True
    w[r(24), r(22):r(27)] = False

    start = (r(5), r(23))
    goal  = (r(44), r(25))
    w[start] = w[goal] = False

    t[:,:] = rng.uniform(14, 22, (N_sc, N_sc))
    R, C = np.ogrid[:N_sc, :N_sc]
    
    lmask = (R>=r(14))&(R<=r(35))&(C>=r(15))&(C<=r(34))
    t[lmask] = np.clip(rng.uniform(60, 100, (N_sc, N_sc))[lmask], 60, 500)
    
    mask = ~w
    for (ar, ac, apk, hr, hc) in CLASS_A:
        nd = np.sqrt(((R - r(ar)) / r(hr))**2 + ((C - r(ac)) / r(hc))**2)
        ragged_nd = nd + rng.uniform(-0.3, 0.3, (N_sc, N_sc))
        core = apk * np.exp(-1.3 * ragged_nd**2) + rng.uniform(-15, 15, (N_sc, N_sc))
        t[mask] = np.clip(np.maximum(t[mask], core[mask]), 1, 500)

    for (br, bc, bpk, bhr, bhc) in CLASS_B:
        nd = np.sqrt(((R - r(br)) / r(bhr))**2 + ((C - r(bc)) / r(bhc))**2)
        ragged_nd = nd + rng.uniform(-0.4, 0.4, (N_sc, N_sc))
        core = bpk * np.exp(-1.5 * ragged_nd**2) + rng.uniform(-8, 8, (N_sc, N_sc))
        t[mask] = np.clip(np.maximum(t[mask], core[mask]), 1, 500)

    t[w] = 0.0
    return start, goal, w, t
