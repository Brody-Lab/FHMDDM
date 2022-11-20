# temporal basis vectors

The matrix $U$ is built by first separately computing an $U_h$ corresponding to the post-spike filter, an $U_t$ corresponding to the post-stereoclick filter, an $U_m$ corresponding to the pre-movment filter, and an $U_p$ corresponding to the post-photostimulus filter. These matrices are horizontally concatenated to form $U$. If a filter is not being used, the corresponding component $U$ is empty. For example, the photostimulus filter is often not used, and in that case, $U_p$ is empty. You will notice in the code that $U$ is never created. This is because to be computationally efficient, $U$ is immediately concatenated with the time-varying weight of the accumulated evidence $(V)$ and state-dependent gain $(G)$ to form the design matrix $X$. You can think of $V$ and $G$ as a column with the same values. All computations are done with the design matrix $X$.

(I have tested that $U*u .+ V*v[k] .+ G*g[k]$ is substantially slower than using the design matrix to compute the linear predictor: $X* vcat(u, v[k], g[k]$)

To build each component matrix of $U$, such as for the matrix corresponding to the post-spike filter $U_h$, I first build the matrix of temporal basis vectors $\Phi_h$ $(T \times B$ matrix, where $T$ is the maximum number of time steps across all trials in a trialset, and $B$ is the number of temporal basis vectors). $U_h$ is a $(Q \times B$ matrix, where $Q$ is the total number of time steps summed across trials in a trialset).

The functions to build each $\Phi$ and each $U$ are in the file `temporal_basis_functions.jl`. For example, for the spike history filter, I call `spikehistorybasis(model.options)`. Note that the features of the filter are specified in the field options. The relevant fields are:

* `begins0`: whether the first time step of the filter is constrained to be zero
* `ends0`: whether the last time step of the filter is constrained to be zero
* `dur_s`: length of the filter, in seconds.
* `hz`: number of temporal basis vectors per second. The number of basis vectors is equal to `ceil(hz*dur_s)`
* `scalefactor`: scale factor to multiply by the temporal basis vector so that the hessian of the log-likelihood is less ill-conditioned
* `stretch`: the degree to which temporal basis functions centered at later times in the trial are stretched. Larger values indicates greater stretch. This value must be positive.

Using these [options](/src/types.jl), I first create a set of temporal basis vectors corresponding to overlapping cosine bumps, and then I rotate the matrix such that each temporal basis vector has the same norm, while spanning the same temporal space. The result is $\Phi$, which has the property that 

$\Phi' * \Phi = s \cdot I$

where $I$ is an $B\times B$ identity matrix (where $B$ is the number of temporal basis functions), ands is a scalar equal to $s = (N * \text{scalefactor})^2$, where $N$ is the number of neurons, and `scalefactor` is an additional coefficient to keep the hessian from being ill-conditioned. So, $(1/s)*\Phi$ is unitary :slightly_smiling_face:.

The scale factors that I typically use are:
* post-spike filter: 1
* pre-movement filter: 1
* post-stereoclick filter: 2
* weight of accumulated evidence: 5
* post-photostimulus filter (not typically used): 10
 