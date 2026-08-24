## Compare accuracy of superresolution with low-fidelity to high fidelity interpolation.

## Accuracy-constrained break-even

Need to generate specific test data h5 file that solves the same case (all `Constant`) at multiple resolutions.

$$
N_{\mathrm{break}}(r)
\left\lceil
\frac{
T_{\mathrm{data}}+
T_{\mathrm{tune}}+
T_{\mathrm{final\ train}}
}{
t_{\mathrm{num}}(r)-t_{\mathrm{FNO}}(r)
}
\right\rceil.
$$

Simple break-even formuala compare both numerical and FNO methods at the same requested resolution. That may not be fair if their errors differ substantially.

Stronger comparison:
> What is the cheapest numerical resolution that achieves at least the Fourier neural operator’s accuracy?

For each Fourier neural operator prediction resolution ($r_f$), calculate its error

$$
e_{\mathrm{FNO}}(r_f).
$$

Then find the coarsest numerical resolution ($r_n^\star$) satisfying

$$
e_{\mathrm{num}}(r_n^\star)
\leq
e_{\mathrm{FNO}}(r_f).
$$

Use its runtime in the break-even calculation:

$$
N_{\mathrm{break}}^{\mathrm{matched}}
\left\lceil
\frac{T_{\mathrm{offline}}}
{
t_{\mathrm{num}}(r_n^\star)
t_{\mathrm{FNO}}(r_f)
}
\right\rceil.
$$

This prevents a misleading claim such as:

* comparing the Fourier neural operator at relative error (0.8),
* against a fine numerical solution with relative error (0.01).
