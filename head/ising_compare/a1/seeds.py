"""Seeds and namespaces of the readout ablation A1 (revision/REVISION_PLAN.md 2.1, "Matched conditions", "Data").

Training seed s (1 and 2; 3 in the full set) fixes, apart from --run-id (then a label only):
  trunk    torch.manual_seed(s), then the trunk is drawn (arch.net.ConvLines / bb.net.ConvLinesBB built on the CPU);
           a1.trunk saves it with its sha256 and every arm of seed s loads it (a trunk-only --init)
  heads    torch.manual_seed(s + HEAD_OFFSET) before each head module, so arms that share a head module start it alike
  stream   surface: arch.stream.LineStream's 8 virtual streams, seeds 10^8 + 10^6 K + v, K = surface_stream(s); a ramp
           run reads arch.stream.RampLineStream, chunk c seeded 10^8 + 10^6 K + c, K = surface_ramp_stream(s);
           BB: bb.shots.chunk_seed(K, p_c, c), K = bb_stream(s) (a ramp run too: its chunks at the ramp's noise
           levels have their own seeds, those at p equal the main run's)
  shuffle  arch.train's batch shuffle: numpy default_rng([SHUFFLE_TAG, s]) (bb.train has no shuffle: chunk order)
So every arm of seed s reads the same shots in the same order. The training-stream namespace of the seed registry
(the research repository's ising_compare/data/study2/seed_namespaces.json) covers K <= 996, the hub's largest training id is 921
and bb.shots allows K < 5,000; K = 1,001-1,013 is new (a1/seedcheck.sh checks it on every branch). --run-id labels:
RUN_IDS (1,100-1,299), also unused.
Evaluation sets (a1.sets; drawn after the freeze, never by a dry run): surface cal 71 M and test 72 M (d = 13: 73 M,
74 M), seed base + 10,000 d + j, 10,000-shot chunks, cal j < 20, test j < 100; BB72 the new kinds a1cal and a1test of
bb.shots' scheme (2^41 + 10^8 kind + 10^7 code + 10^6 p_idx + j), cal j < 20, test j < 50. Dry runs: surface 79 M
and BB kind a1dry, synthetic stand-ins that never enter a result."""

HEAD_OFFSET = 1000
SHUFFLE_TAG = 7101
SEEDS = (1, 2)                      # the minimal set; the full set adds 3


def head_seed(s):
    return s + HEAD_OFFSET


def surface_stream(s):
    return 1000 + s


def surface_ramp_stream(s):
    return 1010 + s


def bb_stream(s):
    return 1000 + s


def shuffle_seed(s):
    return [SHUFFLE_TAG, s]


SURFACE_ARMS = ("S-VA", "S-V", "S-L", "S-P", "S-PA")
BB_ARMS = ("B-JT", "B-J", "B-P", "B-PA")
DEPLOYED = {"surface": "S-VA", "bb72": "B-JT"}
RUN_IDS = {"surface": 1100, "bb72": 1200, "ramp": 50, "dry": 90}


def run_id(code, arm, s, ramp=False, dry=False):
    """The --run-id label of an arm's run: 1,100 + 10 s + i (surface), 1,200 + 10 s + i (BB72), + 50 for a ramp run;
    dry runs 1,190 + i / 1,290 + i (s ignored)."""
    arms = SURFACE_ARMS if code == "surface" else BB_ARMS
    base = RUN_IDS["surface" if code == "surface" else "bb72"]
    return base + (RUN_IDS["dry"] if dry else 10 * s + (RUN_IDS["ramp"] if ramp else 0)) + arms.index(arm)


SURFACE_SETS = {"cal": 71_000_000, "test": 72_000_000, "cal13": 73_000_000, "test13": 74_000_000, "dry": 79_000_000}
SURFACE_CHUNKS = {"cal": 20, "test": 100, "dry": 10}
BB_KINDS = {"a1cal": 4, "a1test": 5, "a1dry": 6}            # bb.shots.SPLIT entries
BB_CHUNKS = {"a1cal": 20, "a1test": 50, "a1dry": 10}


def surface_seed(kind, d, j):
    if not 0 <= j < 10_000:
        raise ValueError(j)
    return SURFACE_SETS[kind] + 10_000 * d + j
