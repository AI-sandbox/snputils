"""
CLI wrapper for OnlineSimulator.

Example
-------
python3 /home/users/geleta/snputils/snputils/simulation/simulator_cli.py \
    --vcf /oak/stanford/projects/tobias/rita/1000G/ref_final_beagle_phased_1kg_hgdp_sgdp_chr22_hg19.vcf.gz \
    --metadata /oak/stanford/projects/tobias/rita/1000G/reference_panel_metadata_v2.tsv \
    --genetic-map /oak/stanford/projects/tobias/rita/1000G/allchrs.b37.gmap \
    --chromosome 22 \
    --window-size 1000 \
    --store-latlon-as-nvec \
    --batch-size 512 \
    --num-generations 64 \
    --n-batches 10 \
    --output-dir /tmp
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from snputils.snp.io.read.vcf import VCFReaderPolars
from snputils.simulation.simulator import OnlineSimulator

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("simulator_cli")
    
def parse_sim_args(cli_args=None) -> argparse.Namespace:
    """Parse command-line flags for the simulator CLI."""
    p = argparse.ArgumentParser(
        prog="simulator_cli",
        description="Batch-simulate admixed haplotypes with OnlineSimulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required I/O
    p.add_argument("--vcf", required=True,
                   help="Path to the phased VCF/VCF-gz file.")
    p.add_argument("--metadata", required=True,
                   help="TSV/CSV file with at least Sample / Population / Latitude / Longitude.")
    p.add_argument("--output-dir", required=True,
                   help="Directory in which to save the simulated batches.")
    # Optional genetic map
    p.add_argument("--genetic-map", default=None,
                   help="Genetic map table with columns: chrom, pos, cM.")
    p.add_argument("--chromosome", type=int, default=None,
                   help="If provided, restrict genetic map rows to this chromosome id.")

    # Simulator hyper-parameters
    p.add_argument("--window-size", type=int, default=1000,
                   help="#SNPs per window.")
    p.add_argument("--store-latlon-as-nvec", action="store_true",
                   help="Convert lat/lon to unit n-vectors (x,y,z).")
    p.add_argument("--make-haploid", action="store_true",
                   help="Flatten diploid genotypes into haplotypes.")
    p.add_argument("--device", default="cpu",
                   help="torch device string, e.g. 'cuda:0'.")

    # Admixture parameters
    p.add_argument("--batch-size", type=int, default=256,
                   help="#simulated haplotypes per batch.")
    p.add_argument("--num-generations", type=int, default=10,
                   help="Upper bound on random generations since admixture.")
    p.add_argument("--n-batches", type=int, default=1,
                   help="#separate batches to generate & save.")

    # Misc
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print additional debugging info.")

    args = p.parse_args(cli_args)

    if args.verbose:
        log.setLevel(logging.DEBUG)

    return args

def main(cli_args=None):
    args = parse_sim_args(cli_args)

    # 1) Sanity checks / output directory
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir.resolve())

    # 2) Read inputs
    log.info("Reading VCF …")
    vcf_reader = VCFReaderPolars(args.vcf)
    vcf_data   = vcf_reader.read()

    log.info("Reading metadata table …")
    meta = pd.read_csv(args.metadata, sep=None, engine="python")
    # Subset to single-ancestry samples if that column exists
    if "Single_Ancestry" in meta.columns:
        meta = meta[meta.Single_Ancestry == True]

    # Keep only the required columns
    cols_needed = ["Sample", "Population", "Latitude", "Longitude"]
    missing = [c for c in cols_needed if c not in meta.columns]
    if missing:
        log.error("Metadata is missing columns: %s", ", ".join(missing))
        sys.exit(1)
    meta = meta[cols_needed]

    # Genetic map (optional)
    genetic_map = None
    if args.genetic_map:
        log.info("Reading genetic map …")
        gm = pd.read_csv(args.genetic_map, sep=None, engine="python",
                         names=["chm", "pos", "cM"])
        if args.chromosome is not None:
            gm = gm[gm.chm == args.chromosome]
        genetic_map = gm

    # 3) Instantiate simulator
    log.info("Initialising OnlineSimulator …")
    simulator = OnlineSimulator(
        vcf_data           = vcf_data,
        meta               = meta,
        genetic_map        = genetic_map,
        make_haploid       = args.make_haploid,
        window_size        = args.window_size,
        store_latlon_as_nvec = args.store_latlon_as_nvec,
    )

    # 4) Run batches
    log.info("Generating %d batch(es)…", args.n_batches)
    for b in range(1, args.n_batches + 1):
        snps, labels_d, labels_c, cp = simulator.simulate(
            batch_size        = args.batch_size,
            num_generation_max= args.num_generations,
            pool_method       = "mode",
            device            = args.device
        )

        # 5) Save
        out_path = out_dir / f"batch_{b:04d}.npz"
        np.savez_compressed(
            out_path,
            snps      = snps.cpu().numpy(),
            labels_d  = (labels_d.cpu().numpy()
                         if labels_d is not None else np.empty(0)),
            labels_c  = (labels_c.cpu().numpy()
                         if labels_c is not None else np.empty(0)),
            cp        = (cp.cpu().numpy()
                         if cp is not None else np.empty(0)),
        )
        log.info("Saved %s", out_path.name)

    log.info("[✓] All done. %d files written to %s", args.n_batches, out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.exception("Fatal error: %s", exc)
        sys.exit(1)