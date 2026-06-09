import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from snputils.snp.io.read import SNPReader
from snputils.simulation._validation import validate_phased_simulation_input
from snputils.simulation.simulator import OnlineSimulator

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("simulator_cli")


def add_simulator_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--snp", required=True,
                   help="Path to phased SNP input (VCF, PGEN, or BGEN fileset). PLINK1 BED is not supported because it cannot store phase.")
    p.add_argument("--metadata", required=True,
                   help="TSV/CSV file with at least Sample/IID and Population columns.")
    p.add_argument("--output-dir", required=True,
                   help="Directory in which to save the simulated batches.")
    p.add_argument("--genetic-map", default=None,
                   help="Genetic map table with columns: chrom, pos, cM.")
    p.add_argument("--chromosome", type=int, default=None,
                   help="If provided, restrict genetic map rows to this chromosome id.")
    p.add_argument("--window-size", type=int, default=1000,
                   help="#SNPs per window.")
    p.add_argument("--store-latlon-as-nvec", action="store_true",
                   help="Convert lat/lon to unit n-vectors (x,y,z).")
    p.add_argument("--make-haploid", action="store_true",
                   help="Flatten diploid genotypes into haplotypes.")
    p.add_argument("--device", default="cpu",
                   help="torch device string, e.g. 'cuda:0'.")
    p.add_argument("--batch-size", type=int, default=256,
                   help="#simulated haplotypes per batch.")
    p.add_argument("--diploid-output", action="store_true",
                   help="Save simulated diploid samples. When set, --batch-size is the number of samples.")
    p.add_argument("--num-generations", type=int, default=10,
                   help="Upper bound on random generations since admixture.")
    p.add_argument("--fixed-generations", action="store_true",
                   help="Use exactly --num-generations instead of drawing uniformly from 0..num-generations.")
    p.add_argument("--ancestry-proportions", default=None,
                   help="Comma-separated population proportions, e.g. YRI:0.8,CEU:0.2.")
    p.add_argument("--n-batches", type=int, default=1,
                   help="#separate batches to generate & save.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print additional debugging info.")


def parse_sim_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="simulator_cli",
        description="Batch-simulate admixed haplotypes with OnlineSimulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_simulator_arguments(p)
    args = p.parse_args(argv)

    if args.verbose:
        log.setLevel(logging.DEBUG)

    return args


def _parse_ancestry_proportions(value: str | None) -> dict[str, float] | None:
    if value is None:
        return None

    proportions = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("Expected ancestry proportions like YRI:0.8,CEU:0.2.")
        pop, prob = item.split(":", 1)
        proportions[pop.strip()] = float(prob)

    if not proportions:
        raise ValueError("No ancestry proportions were provided.")
    return proportions


def _read_genetic_map(path: str, chromosome: int | None = None) -> pd.DataFrame:
    gm = pd.read_csv(path, sep=None, engine="python")
    lower_cols = {str(c).lower(): c for c in gm.columns}

    if {"pos", "cm"}.issubset(lower_cols) and ("chr" in lower_cols or "chm" in lower_cols or "chrom" in lower_cols):
        chrom_col = lower_cols.get("chm", lower_cols.get("chr", lower_cols.get("chrom")))
        gm = gm.rename(columns={chrom_col: "chm", lower_cols["pos"]: "pos", lower_cols["cm"]: "cM"})
    else:
        gm = pd.read_csv(path, sep=None, engine="python", names=["chm", "pos", "cM"])

    if chromosome is not None:
        gm = gm[gm.chm.astype(str) == str(chromosome)]

    return gm[["chm", "pos", "cM"]]


def run_simulator_command(args: argparse.Namespace) -> int:
    if getattr(args, "verbose", False):
        log.setLevel(logging.DEBUG)

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir.resolve())

    validate_phased_simulation_input(args.snp)

    log.info("Reading SNP input...")
    snp_data = SNPReader(args.snp).read(sum_strands=False)

    log.info("Reading metadata table...")
    meta = pd.read_csv(args.metadata, sep=None, engine="python")
    if "Sample" not in meta.columns and "IID" in meta.columns:
        meta = meta.rename(columns={"IID": "Sample"})
    if "Single_Ancestry" in meta.columns:
        meta = meta[meta.Single_Ancestry == True]

    cols_needed = ["Sample", "Population"]
    missing = [c for c in cols_needed if c not in meta.columns]
    if missing:
        log.error("Metadata is missing columns: %s", ", ".join(missing))
        sys.exit(1)
    keep_cols = ["Sample", "Population"]
    if {"Latitude", "Longitude"}.issubset(meta.columns):
        keep_cols.extend(["Latitude", "Longitude"])
    meta = meta[keep_cols]

    genetic_map = None
    if args.genetic_map:
        log.info("Reading genetic map...")
        genetic_map = _read_genetic_map(args.genetic_map, args.chromosome)

    ancestry_proportions = _parse_ancestry_proportions(args.ancestry_proportions)

    log.info("Initialising OnlineSimulator...")
    simulator = OnlineSimulator(
        snp_data             = snp_data,
        meta                 = meta,
        genetic_map          = genetic_map,
        make_haploid         = args.make_haploid,
        window_size          = args.window_size,
        store_latlon_as_nvec = args.store_latlon_as_nvec,
        ancestry_proportions = ancestry_proportions,
    )

    log.info("Generating %d batch(es)...", args.n_batches)
    for b in range(1, args.n_batches + 1):
        simulate_batch_size = args.batch_size * 2 if args.diploid_output else args.batch_size
        snps, labels_d, labels_c, cp = simulator.simulate(
            batch_size         = simulate_batch_size,
            num_generation_max = args.num_generations,
            num_generations     = args.num_generations if args.fixed_generations else None,
            pool_method        = "mode",
            device             = args.device
        )

        output_ploidy = 1
        if args.diploid_output:
            output_ploidy = 2
            snps = snps.reshape(args.batch_size, output_ploidy, snps.shape[-1])
            if labels_d is not None:
                labels_d = labels_d.reshape(args.batch_size, output_ploidy, labels_d.shape[-1])
            if labels_c is not None:
                labels_c = labels_c.reshape(args.batch_size, output_ploidy, labels_c.shape[-2], labels_c.shape[-1])
            if cp is not None:
                cp = cp.reshape(args.batch_size, output_ploidy, cp.shape[-1])

        out_path = out_dir / f"batch_{b:04d}.npz"
        np.savez_compressed(
            out_path,
            snps     = snps.cpu().numpy(),
            labels_d = (labels_d.cpu().numpy()
                        if labels_d is not None else np.empty(0)),
            labels_c = (labels_c.cpu().numpy()
                        if labels_c is not None else np.empty(0)),
            cp       = (cp.cpu().numpy()
                        if cp is not None else np.empty(0)),
            population_names = (np.asarray(simulator.population_names, dtype=str)
                                if simulator.population_names is not None else np.empty(0, dtype=str)),
            output_ploidy = np.asarray(output_ploidy, dtype=np.int8),
        )
        log.info("Saved %s", out_path.name)

    log.info("[✓] All done. %d files written to %s", args.n_batches, out_dir)
    return 0


def main(argv=None) -> int:
    args = parse_sim_args(argv)
    return run_simulator_command(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log.exception("Fatal error: %s", exc)
        sys.exit(1)
