# Simulation

Haplotype-based simulation of admixed mosaics from phased founder panels. Requires PyTorch (`pip install "snputils[torch]"`). `OnlineSimulator` draws crossover breakpoints from an optional genetic map and returns batched haplotype labels with SNP matrices on CPU or GPU.

Simulation inputs must preserve haplotype phase. Phased VCF, PGEN, or BGEN inputs are appropriate; PLINK1 BED/BIM/FAM files are rejected because they cannot store chromosome-scale phase.

## OnlineSimulator

```{eval-rst}
.. autoclass:: snputils.simulation.simulator.simulator.OnlineSimulator
   :members:
```

See {doc}`../user_guide/analysis` for a Python example and the {doc}`cli` `simulate` subcommand for file-backed batch generation.
