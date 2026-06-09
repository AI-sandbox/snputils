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
    p.add_argument("--output-prefix", default=None,
                   help="Output prefix for cohort mode. Defaults to <output-dir>/simulated when --n-individuals is used.")
    p.add_argument("--output-format", default="same",
                   choices=("same", "pgen", "vcf", "vcf.gz", "bgen"),
                   help="Genotype output format for --n-individuals cohort output.")
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
    p.add_argument("--n-individuals", type=int, default=None,
                   help="Simulate this many diploid individuals and write a single fileset instead of NPZ batches.")
    p.add_argument("--sample-prefix", default="SIM",
                   help="Sample ID prefix for --n-individuals cohort output.")
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


def _copy_array(obj, name: str):
    value = getattr(obj, name, None)
    if value is None:
        return None
    return np.asarray(value).copy()


def _infer_output_format(input_path: str, requested_format: str) -> str:
    if requested_format != "same":
        return requested_format

    suffixes = [suffix.lower() for suffix in Path(input_path).suffixes]
    if len(suffixes) >= 2 and suffixes[-2:] == [".vcf", ".gz"]:
        return "vcf.gz"
    if not suffixes:
        raise ValueError("Cannot infer output format from input path without an extension.")
    suffix = suffixes[-1]
    if suffix in {".pgen", ".pvar", ".psam", ".pvar.zst"}:
        return "pgen"
    if suffix == ".vcf":
        return "vcf"
    if suffix == ".bgen":
        return "bgen"
    raise ValueError(f"Cannot infer cohort output format from input path: {input_path}")


def _output_genotype_path(output_prefix: Path, output_format: str) -> Path:
    if output_format == "pgen":
        return output_prefix.with_suffix(".pgen")
    if output_format == "vcf":
        return output_prefix.with_suffix(".vcf")
    if output_format == "vcf.gz":
        return output_prefix.with_suffix(".vcf.gz")
    if output_format == "bgen":
        return output_prefix.with_suffix(".bgen")
    raise ValueError(f"Unsupported output format: {output_format}")


def _build_simulated_snpobject(snp_data, genotypes: np.ndarray, samples: np.ndarray):
    from snputils.snp.genobj.snpobj import SNPObject

    return SNPObject(
        genotypes=genotypes,
        samples=samples,
        sample_sex=np.repeat("NA", len(samples)),
        variants_ref=_copy_array(snp_data, "variants_ref"),
        variants_alt=_copy_array(snp_data, "variants_alt"),
        variants_chrom=_copy_array(snp_data, "variants_chrom"),
        variants_cm=_copy_array(snp_data, "variants_cm"),
        variants_filter_pass=_copy_array(snp_data, "variants_filter_pass"),
        variants_id=_copy_array(snp_data, "variants_id"),
        variants_pos=_copy_array(snp_data, "variants_pos"),
        variants_qual=_copy_array(snp_data, "variants_qual"),
        variants_info=_copy_array(snp_data, "variants_info"),
    )


def _write_genotypes_like_input(
    snp_data,
    genotypes: np.ndarray,
    samples: np.ndarray,
    output_prefix: Path,
    output_format: str,
) -> Path:
    snpobj = _build_simulated_snpobject(snp_data, genotypes, samples)
    output_path = _output_genotype_path(output_prefix, output_format)

    if output_format == "pgen":
        from snputils.snp.io.write.pgen import PGENWriter

        PGENWriter(snpobj=snpobj, filename=str(output_path)).write()
    elif output_format in {"vcf", "vcf.gz"}:
        from snputils.snp.io.write.vcf import VCFWriter

        VCFWriter(snpobj=snpobj, filename=str(output_path), phased=True).write(
            rename_missing_values=False,
            variants_info=snpobj.variants_info,
        )
    elif output_format == "bgen":
        from snputils.snp.io.write.bgen import BGENWriter

        BGENWriter(snpobj=snpobj, filename=output_path).write(phased=True)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    return output_path


def _write_exact_msp(
    path: Path,
    samples: np.ndarray,
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    snp_data,
    cm_per_snp,
    ancestry_map: dict[str, str],
) -> None:
    n_haplotypes = len(segments)
    if n_haplotypes != len(samples) * 2:
        raise ValueError("Expected two haplotype segment lists per simulated sample.")

    breakpoints = {0, len(snp_data.variants_pos)}
    for starts, ends, _ in segments:
        breakpoints.update(int(x) for x in starts)
        breakpoints.update(int(x) for x in ends)
    breakpoints = np.asarray(sorted(breakpoints), dtype=np.int64)

    variant_pos = np.asarray(snp_data.variants_pos)
    variant_chrom = np.asarray(snp_data.variants_chrom)
    if cm_per_snp is None:
        cm_per_snp = np.full(len(variant_pos), np.nan)
    else:
        cm_per_snp = np.asarray(cm_per_snp)

    haplotypes = [f"{sample}.{strand}" for sample in samples for strand in range(2)]
    segment_ptrs = np.zeros(n_haplotypes, dtype=np.int64)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as out:
        out.write(
            "#Subpopulation order/codes: "
            + "\t".join(f"{name}={code}" for code, name in ancestry_map.items())
            + "\n"
        )
        out.write("#chm\tspos\tepos\tsgpos\tegpos\tn snps\t" + "\t".join(haplotypes) + "\n")

        for start, end in zip(breakpoints[:-1], breakpoints[1:]):
            if start == end:
                continue
            codes = []
            for hap_idx, (seg_starts, seg_ends, seg_codes) in enumerate(segments):
                ptr = segment_ptrs[hap_idx]
                while ptr + 1 < len(seg_ends) and seg_ends[ptr] <= start:
                    ptr += 1
                segment_ptrs[hap_idx] = ptr
                codes.append(str(int(seg_codes[ptr])))

            row = [
                str(variant_chrom[start]),
                str(int(variant_pos[start])),
                str(int(variant_pos[end - 1])),
                str(cm_per_snp[start]),
                str(cm_per_snp[end - 1]),
                str(int(end - start)),
            ]
            out.write("\t".join(row + codes) + "\n")


def _run_cohort_output(args, simulator, snp_data, out_dir: Path) -> int:
    if args.ancestry_proportions is None:
        raise ValueError("--n-individuals requires --ancestry-proportions.")

    output_prefix = Path(args.output_prefix).expanduser() if args.output_prefix else out_dir / "simulated"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    log.info("Simulating %d diploid individual(s)...", args.n_individuals)
    genotypes, samples, segments = simulator.simulate_diploid_population(
        n_individuals=args.n_individuals,
        num_generation_max=args.num_generations,
        num_generations=args.num_generations if args.fixed_generations else None,
        sample_prefix=args.sample_prefix,
    )

    output_format = _infer_output_format(args.snp, args.output_format)
    genotype_path = _output_genotype_path(output_prefix, output_format)
    log.info("Writing simulated %s genotype output to %s", output_format, genotype_path)
    _write_genotypes_like_input(snp_data, genotypes, samples, output_prefix, output_format)

    ancestry_map = {str(i): str(pop) for i, pop in enumerate(simulator.population_names)}
    msp_path = output_prefix.with_suffix(".msp")
    log.info("Writing exact ground-truth MSP to %s", msp_path)
    _write_exact_msp(
        path=msp_path,
        samples=samples,
        segments=segments,
        snp_data=snp_data,
        cm_per_snp=simulator.cm_per_snp,
        ancestry_map=ancestry_map,
    )

    log.info("[✓] Wrote simulated cohort files with prefix %s", output_prefix)
    return 0


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

    if args.n_individuals is not None:
        return _run_cohort_output(args, simulator, snp_data, out_dir)

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
