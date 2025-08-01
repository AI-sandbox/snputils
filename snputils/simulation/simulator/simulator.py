import logging
import torch
import numpy as np

log = logging.getLogger(__name__)
    

def latlon_to_nvector(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Convert latitude and longitude (in degrees) to 3D unit n-vector (x, y, z).

    Args:
        lat (np.ndarray): Latitude values in degrees.
        lon (np.ndarray): Longitude values in degrees.

    Returns:
        np.ndarray: Array of shape (..., 3) containing corresponding x, y, z unit vectors.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=-1)


def nvector_to_latlon(nvec):
    """
    Convert an n-vector (x,y,z) back to latitude/longitude in DEGREES.

    Parameters
    ----------
    nvec : np.ndarray of shape (3,) or (n_samples,3)
        x, y, z coordinates of the n-vector.

    Returns
    -------
    (lat_deg, lon_deg) : tuple of floats or np.ndarrays
        Latitude(s) and longitude(s) in degrees.
    """
    if nvec.ndim == 1:
        x, y, z = nvec
        lat_rad = np.arcsin(z)
        lon_rad = np.arctan2(y, x)
        return (np.degrees(lat_rad), np.degrees(lon_rad))
    else:
        x = nvec[..., 0]
        y = nvec[..., 1]
        z = nvec[..., 2]
        lat_rad = np.arcsin(z)
        lon_rad = np.arctan2(y, x)
        return (np.degrees(lat_rad), np.degrees(lon_rad))
    
def approximate_mode_per_row(
    row_2d: torch.Tensor,   # shape (batch_size, W) 
    nbins=32
) -> torch.Tensor:
    """
    row_2d: shape (batch_size, W), continuous data on GPU
    nbins:  number of histogram bins

    Returns: shape (batch_size,) approximate mode for each row 
             (i.e. each row in row_2d).
    """
    device = row_2d.device
    batch_size, W = row_2d.shape

    # row-wise min/max
    row_min = row_2d.min(dim=1).values  # (batch_size,)
    row_max = row_2d.max(dim=1).values  # (batch_size,)

    out_modes = torch.zeros(batch_size, device=device, dtype=torch.float32)

    # We'll do a simple loop over batch_size rows,
    # because torch.histc only handles 1D at a time
    for i in range(batch_size):
        data_i = row_2d[i]  # shape (W,)
        vmin = row_min[i].item()
        vmax = row_max[i].item()

        # if all the same => mode is that value
        if vmax == vmin:
            out_modes[i] = data_i[0]
            continue

        # hist => shape (nbins,)
        hist = torch.histc(data_i, bins=nbins, min=vmin, max=vmax)
        # bin_idx in [0..nbins-1]
        bin_idx = hist.argmax().item()
        bin_width = (vmax - vmin)/nbins
        # approximate midpoint
        bin_mid = vmin + (bin_idx+0.5)*bin_width
        out_modes[i] = bin_mid

    return out_modes


def _chunk_label_array(
    labels: torch.Tensor,
    window_size: int,
    descriptor="continuous",
    pool_method="mode",
    nbins=32
):
    """
    labels: shape (batch_size, n_snps) or (batch_size, n_snps, dim) after final crossovers
      - If descriptor="continuous" => shape (batch_size, n_snps[, dim])
      - If descriptor="discrete"   => shape (batch_size, n_snps)
    window_size: # of SNPs per window
    descriptor:  "continuous" or "discrete"
    pool_method: "mean" or "mode"
      - If continuous + "mode" => uses an approximate histogram-based mode (GPU-friendly)
    nbins: # of bins if using approximate mode for continuous

    Returns:
      chunked => shape (batch_size, n_win, dim), (batch_size, n_win, 1), or (batch_size, n_win)
        depending on descriptor + dimension
    """
    device = labels.device

    if labels.ndim == 2:
        # shape => (batch_size, n_snps) => discrete or continuous w/ dim=1
        batch_size, n_snps = labels.shape
        label_dim = None
    else:
        # shape => (batch_size, n_snps, dim)
        batch_size, n_snps, label_dim = labels.shape

    n_full = n_snps // window_size
    leftover = n_snps % window_size

    chunks = []
    start = 0
    for _ in range(n_full):
        end = start + window_size
        # slice => (batch_size, window_size[, dim])
        window_segment = (
            labels[:, start:end, ...]
            if label_dim else labels[:, start:end]
        )

        if descriptor == "continuous":
            if pool_method == "mean":
                # normal PyTorch .mean(...)
                if label_dim:
                    # shape => (batch_size, window_size, dim)
                    mean_vals = window_segment.mean(dim=1)  # => (batch_size, dim)
                    chunks.append(mean_vals)
                else:
                    # shape => (batch_size, window_size)
                    mean_vals = window_segment.float().mean(dim=1)  # => (batch_size,)
                    chunks.append(mean_vals.unsqueeze(-1))

            elif pool_method == "mode":
                # approximate GPU mode
                if label_dim:
                    # shape => (batch_size, window_size, dim)
                    # do dimension by dimension
                    # => we'll gather a list of (batch_size,) for each dim, then stack
                    mode_vals_list = []
                    for d_i in range(label_dim):
                        # slice => (batch_size, window_size)
                        slice_2d = window_segment[:, :, d_i]
                        # approximate mode => (batch_size,)
                        approx_m = approximate_mode_per_row(slice_2d, nbins=nbins)
                        mode_vals_list.append(approx_m)
                    # stack => (batch_size, label_dim)
                    mode_vals_cat = torch.stack(mode_vals_list, dim=1)
                    chunks.append(mode_vals_cat)
                else:
                    # shape => (batch_size, window_size)
                    approx_m = approximate_mode_per_row(window_segment, nbins=nbins) # => (batch_size,)
                    chunks.append(approx_m.unsqueeze(-1))
            else:
                raise ValueError(f"pool_method '{pool_method}' not implemented for continuous.")

        else:
            # descriptor == "discrete" => use built-in .mode(dim=1)
            # shape => (batch_size, window_size)
            # mode along dimension=1 => shape (batch_size,)
            mode_vals = window_segment.mode(dim=1).values
            chunks.append(mode_vals)

        start = end

    # leftover
    if leftover > 0:
        window_segment = (
            labels[:, start:, ...]
            if label_dim else labels[:, start:]
        )
        if descriptor == "continuous":
            if label_dim:
                if pool_method == "mean":
                    mean_vals = window_segment.mean(dim=1)  # => (batch_size, dim)
                    chunks.append(mean_vals)
                else:
                    # pool_method == "mode" => approximate
                    mode_vals_list = []
                    for d_i in range(label_dim):
                        slice_2d = window_segment[:, :, d_i]
                        approx_m = approximate_mode_per_row(slice_2d, nbins=nbins)
                        mode_vals_list.append(approx_m)
                    mode_vals_cat = torch.stack(mode_vals_list, dim=1)
                    chunks.append(mode_vals_cat)
            else:
                if pool_method == "mean":
                    mean_vals = window_segment.float().mean(dim=1)  # (batch_size,)
                    chunks.append(mean_vals.unsqueeze(-1))
                else:
                    # approximate mode
                    approx_m = approximate_mode_per_row(window_segment, nbins=nbins)
                    chunks.append(approx_m.unsqueeze(-1))
        else:
            # discrete => leftover => .mode(dim=1)
            mode_vals = window_segment.mode(dim=1).values
            chunks.append(mode_vals)

    if len(chunks) == 0:
        # if window_size >= n_snps => no chunk
        return None

    # Now stack => shape (batch_size, n_windows[, dim]) or (batch_size, n_windows)
    if descriptor == "continuous":
        cat_res = torch.stack(chunks, dim=1)  
        return cat_res
    else:
        # discrete => shape => (batch_size,) in each chunk => stack => (batch_size, n_windows)
        cat_res = torch.stack(chunks, dim=1)
        return cat_res


def _chunk_changepoints(cp_mask, window_size):
    """
    cp_mask: shape (batch_size, n_snps), a boolean (or 0/1) array indicating
             breakpoint positions at the SNP level.

    Returns: shape (batch_size, n_windows), with 1 if any SNP in that window
             was a breakpoint, else 0.
    """
    batch_size, n_snps = cp_mask.shape
    n_full = n_snps // window_size
    leftover = n_snps % window_size

    chunks = []
    start = 0
    for _ in range(n_full):
        end = start + window_size
        # if any True in that window => 1
        any_cp = cp_mask[:, start:end].any(axis=1)
        chunks.append(any_cp)
        start = end

    if leftover > 0:
        any_cp = cp_mask[:, start:].any(axis=1)
        chunks.append(any_cp)

    if len(chunks) == 0:
        return None

    # stack along new dim => (batch_size, n_windows)
    out = np.stack(chunks, axis=1).astype(np.int8)
    return torch.tensor(out)
    
class OnlineSimulator:
    """
    A refactored 'OnlineSimulator' for haplotype simulation with window-based SNP data.
    -------------------------------------------------------------------------------
    Core Functionality:
      - Simulates admixed haplotypes.
      - Supports:
         (a) discrete labels (e.g., population codes), or
         (b) lat/lon (converted to n-vectors) stored per window of SNPs.

    Example usage:
    -------------
        sim = OnlineSimulator(
            vcf_data=my_species_chrX_vcf_data,
            meta=metadata_df,
            genetic_map=genetic_map_df,  # optional
            ...
        )
        # Then to simulate:
        snps, labels_discrete, labels_continuous, changepoints = sim.simulate(batch_size=32)
    """
    
    def __init__(
        self,
        vcf_data,
        meta,
        genetic_map = None,
        make_haploid = True,
        window_size = None,
        store_latlon_as_nvec = False,
        cp_tolerance = 0,
    ):
        self.vcf_data = vcf_data
        self.meta = meta
        self.genetic_map = genetic_map
        self.make_haploid = make_haploid
        self.window_size = window_size
        self.store_latlon_as_nvec = store_latlon_as_nvec
        self.cp_tolerance = cp_tolerance
        
        # We will keep discrete and continuous labels separately
        self.labels_discrete = None
        self.labels_continuous = None

        # Load everything
        self._check_sample_metadata()
        self._intersect_vcf_metadata()
        self._build_descriptors()
        self._broadcast_labels_across_snps() 

    def _check_sample_metadata(self):
        """
        Ensures the DataFrame `self.meta` has the necessary columns.
        - If 'discrete', we expect 'Population' column
        - If 'continuous', we expect 'Latitude' and 'Longitude'
        """
        if 'Sample' not in self.meta.columns:
            raise ValueError("Expected 'Sample' column in sample metadata.")
            
        # We'll just check presence:
        # If 'Population' in columns => we'll do discrete
        # If 'Latitude'/'Longitude' in columns => we'll do continuous
        # It's fine if only one is present
        needed_for_continuous = {'Latitude', 'Longitude'}
        self.has_discrete = ('Population' in self.meta.columns)
        self.has_continuous = needed_for_continuous.issubset(self.meta.columns)
        
        if not (self.has_discrete or self.has_continuous):
            raise ValueError(
                "No recognized columns for descriptors. Need 'Population' for discrete "
                "and/or 'Latitude','Longitude' for continuous."
            )

        # Drop rows that lack the necessary fields
        # For discrete, require 'Population'
        if self.has_discrete:
            self.meta = self.meta.dropna(subset=['Sample', 'Population'])
            log.info("Discrete labeling: found 'Population' column in metadata.")
        # For continuous, require lat/lon
        if self.has_continuous:
            self.meta = self.meta.dropna(subset=['Sample', 'Latitude', 'Longitude'])
            log.info("Continuous labeling: found 'Latitude'/'Longitude' columns in metadata.")

        log.info('Metadata OK.')
            
    def _intersect_vcf_metadata(self):
        """
        Intersects VCF samples with metadata samples.
        Produces:
          self.snps: shape (n_samples, n_snps) or (n_samples,2,n_snps) if not yet flattened
          self.samples: array of sample names
        If self.make_haploid is True, flattens to haplotype level => shape (n_samples*2, n_snps).
        """
        # Intersect VCF samples with metadata samples
        vcf_samples = self.vcf_data["samples"]
        log.info(f"VCF has {len(vcf_samples)} samples total.")
        meta_samples = self.meta["Sample"].values
        # Return intersection array plus index arrays
        #   isamples = intersected sample IDs
        #   iidx = indices in vcf_samples that match
        inter = np.intersect1d(vcf_samples, meta_samples, assume_unique=False, return_indices=True)
        isamples, iidx = inter[0], inter[1]
        log.info(f"{len(isamples)} samples found in both VCF and metadata.")
        if len(isamples) == 0:
            raise ValueError("No overlap between VCF samples and metadata samples. Check your paths or sample naming.")
        
        # Reindex the metadata so it lines up with 'intersect_samples'
        # idx_meta is the array of indices in self.metadata that correspond
        # to the intersected sample set
        #self.meta = self.meta.iloc[iidx].copy().reset_index(drop=True)
        samp2idx = {s: idx for idx, s in enumerate(meta_samples)}
        meta_idxs = [samp2idx[s] for s in isamples]
        self.meta = self.meta.iloc[meta_idxs].copy().reset_index(drop=True)
        
        # Load genotype data: shape (variants, samples, ploidy)
        snps = self.vcf_data["calldata_gt"].transpose(1,2,0)[iidx, ...] 
        n_samples, ploidy, n_snps = snps.shape
        
        # Note that if we flatten into haploid, we need to repeat rows
        if self.make_haploid:
            # Flatten into haploid if requested
            snps = snps.reshape(n_samples * ploidy, n_snps)
            # If we flattened from (samples, 2, snps) => (samples*2, snps)
            # we must also repeat the metadata rows for the 2 haplotypes
            isamples = np.repeat(isamples, ploidy)
            self.meta = self.meta.loc[self.meta.index.repeat(2)].reset_index(drop=True)
            
        # Convert to torch
        self.snps = torch.tensor(snps, dtype=torch.int8)
        self.samples = np.array(isamples)
        log.info(f"snps shape = {self.snps.shape}, sample length = {len(self.samples)}")
                                 
        # Read genetic map
        if self.genetic_map is not None:
            cm_interp = np.interp(self.vcf_data["variants_pos"], self.genetic_map['pos'], self.genetic_map['cM'])
            self.rate_per_snp = np.gradient(cm_interp/100.0)
            log.info(f"rate/snp shape = {self.rate_per_snp.shape}")
        else:
            self.rate_per_snp = None
        

    def _build_descriptors(self):
        """
        Build the self.labels array. If discrete, we'll store integer-coded labels.
        If continuous, we store lat/lon or x,y,z for each sample.
        """
        if len(self.samples) != self.snps.shape[0]:
            raise ValueError("Metadata subset mismatch in length after flattening haplotypes.")

        # 1) Discrete
        if self.has_discrete:
            pop_values = self.meta['Population'].values
            unique_pops = sorted(np.unique(pop_values))
            pop2code = {p: i for i, p in enumerate(unique_pops)}
            self.ancestry_map = {str(v): k for k, v in pop2code.items()}
            discrete_arr = np.array([pop2code[p] for p in pop_values], dtype=np.int16)
            # shape => (n_samples,1)
            discrete_arr = discrete_arr[:, None]
            self.labels_discrete = torch.tensor(discrete_arr, dtype=torch.int16)
            log.info(f"Built discrete labels => shape {self.labels_discrete.shape}")

        # 2) Continuous
        if self.has_continuous:
            lat_vals = self.meta["Latitude"].values
            lon_vals = self.meta["Longitude"].values
            if self.store_latlon_as_nvec:
                coords = latlon_to_nvector(lat_vals, lon_vals)  # (n_samples, 3)
            else:
                coords = np.stack([lat_vals, lon_vals], axis=-1)  # (n_samples, 2)
            self.labels_continuous = torch.tensor(coords, dtype=torch.float32)
            log.info(f"Built continuous labels => shape {self.labels_continuous.shape}")


    def _broadcast_labels_across_snps(self):
        """
        Make self.labels have shape (n_samples, n_snps) for discrete or (n_samples, n_snps, coord_dim) for continuous
        so we can do per-SNP crossovers that also scramble the labels.
        """
        n_samples, n_snps = self.snps.shape
        
        # Discrete
        if self.labels_discrete is not None:
            # shape => (n_samples,1) => broadcast => (n_samples, n_snps)
            arr = self.labels_discrete.cpu().numpy()  # shape (n_samples,1)
            arr_bcast = np.repeat(arr, n_snps, axis=1)     # (n_samples, n_snps)
            self.labels_discrete = torch.tensor(arr_bcast, dtype=torch.int16)
            log.info(f"Broadcast discrete => {self.labels_discrete.shape}")

        # Continuous
        if self.labels_continuous is not None:
            arr = self.labels_continuous.cpu().numpy()   # shape (n_samples, 2 or 3)
            coord_dim = arr.shape[1]
            arr_bcast = np.zeros((n_samples, n_snps, coord_dim), dtype=arr.dtype)
            for i in range(coord_dim):
                arr_bcast[:,:,i] = np.repeat(arr[:, i][:, None], n_snps, axis=1)
            self.labels_continuous = torch.tensor(arr_bcast, dtype=torch.float32)
            log.info(f"Broadcast continuous => {self.labels_continuous.shape}")
       
    def _simulate_from_pool(
        self,
        batch_snps,
        batch_labels_discrete,
        batch_labels_continuous,
        num_generation_max,
        device='cpu',
    ):
        """
        Shuffle segments for admixture on snps, discrete labels, continuous labels (if they exist).
        Each has shape: 
          snps => (batch_size, n_snps)
          batch_labels_discrete => (batch_size, n_snps) or None
          batch_labels_continuous => (batch_size, n_snps, cdim) or None
        """
        if device != 'cpu':
            batch_snps = batch_snps.to(device)
            if batch_labels_discrete is not None:
                batch_labels_discrete = batch_labels_discrete.to(device)
            if batch_labels_continuous is not None:
                batch_labels_continuous = batch_labels_continuous.to(device)


        # 1) Pick the number of generations
        G = np.random.randint(0, num_generation_max+1)
        # 2) If we have a rate_per_snp array, we can do random binomial switch at each SNP
        #    or if cM is provided, uniform. 
        #    We'll keep it simple: we do g random switch points
        batch_size, n_snps = batch_snps.shape 
        

        if self.rate_per_snp is not None:
            switch = np.random.binomial(G, self.rate_per_snp) % 2
            split_points = np.flatnonzero(switch)
        else:
            split_points = torch.randint(n_snps, (G,))

        for sp in split_points:
            perm = torch.randperm(batch_size, device=batch_snps.device)
            # Swap SNPs
            batch_snps[:, sp:] = batch_snps[perm, sp:]
            # Swap discrete
            if batch_labels_discrete is not None:
                batch_labels_discrete[:, sp:] = batch_labels_discrete[perm, sp:]
            # Swap continuous
            if batch_labels_continuous is not None:
                batch_labels_continuous[:, sp:, :] = batch_labels_continuous[perm, sp:, :]


        return batch_snps, batch_labels_discrete, batch_labels_continuous

    def simulate(
        self,
        batch_size=256,
        num_generation_max=10,
        device='cpu',
        pool_method='mode'
    ):
        """
        Returns a tuple of:
          ( batch_snps, final_discrete_labels_window, final_continuous_labels_window )

        where:
          - batch_snps.shape == (batch_size, n_snps)
          - final_discrete_labels_window == (batch_size, n_windows) if discrete was present, else None
          - final_continuous_labels_window == (batch_size, n_windows, cdim) if continuous was present, else None
        """
        # pick random subset of samples
        n_samples = self.snps.shape[0]
        idx = torch.randint(n_samples, (batch_size,))
        batch_snps = self.snps[idx, :].clone()  # shape (batch_size, n_snps)

        # Subset discrete
        if self.labels_discrete is not None:
            batch_discrete = self.labels_discrete[idx, :].clone()  # shape (batch_size, n_snps)
        else:
            batch_discrete = None

        # Subset continuous
        if self.labels_continuous is not None:
            batch_continuous = self.labels_continuous[idx, :, :].clone() # (batch_size, n_snps, dim)
        else:
            batch_continuous = None
        
        # Crossovers
        batch_snps, batch_discrete, batch_continuous = self._simulate_from_pool(
            batch_snps, batch_discrete, batch_continuous,
            num_generation_max=num_generation_max,
            device=device
        )

        # Window-chunk each label array if window_size is specified
        discrete_out = None
        continuous_out = None
        final_cp_window = None
        
        if self.window_size is not None and self.window_size > 0:
            if batch_discrete is not None:
                discrete_out = _chunk_label_array(
                    labels=batch_discrete,
                    window_size=self.window_size,
                    descriptor="discrete",
                    pool_method=None,
                )
                # shape => (batch_size, n_snps)
                # find SNP-level breakpoints
                # Compare label[i] vs label[i-1]
                # We'll do that in NumPy or Torch
                lab_np = batch_discrete.cpu().numpy()  # (batch_size, n_snps)
                # define a shift comparison
                # cp_mask[:,i] = True if lab[:,i] != lab[:,i-1]
                # We'll do it for i from 1..n_snps-1
                cp_mask = np.zeros_like(lab_np, dtype=bool)  # (batch_size, n_snps)
                cp_mask[:, 1:] = (lab_np[:, 1:] != lab_np[:, :-1])
                
                # Now chunk that cp_mask into windows
                cp_mask_t = torch.from_numpy(cp_mask)
                final_cp_window = _chunk_changepoints(cp_mask_t, self.window_size)
            
            if batch_continuous is not None:
                continuous_out = _chunk_label_array(
                    labels=batch_continuous,
                    window_size=self.window_size,
                    descriptor="continuous",
                    pool_method=pool_method,
                )

        return batch_snps.float(), discrete_out, continuous_out, final_cp_window
