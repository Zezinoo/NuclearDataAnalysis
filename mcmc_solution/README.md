# MCMC homework solution

Open `MCMC_HomeWork_2026_solved.ipynb`. It contains answers to all nine questions, derivations, code, saved figures, parameter estimates, and discussion. The notebook has been executed from a fresh kernel.

Keep these files in the same directory:

- `MCMC_HomeWork_2026_solved.ipynb`: main deliverable.
- `mcmc_analysis.py`: model implementation and numerical helpers.
- `mcmc_sampler.py`: original supplied generic sampler, unchanged.
- `sample_data.txt`: original data, unchanged.

The `provided/` directory preserves the original exercise notebook and plotting module for reference. The plotting module is not imported by the solution.

To rerun, install the packages in `requirements.txt`, start Jupyter, and choose **Restart Kernel and Run All Cells**. Alternatively, from this directory:

```bash
python -m pip install -r requirements.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 MCMC_HomeWork_2026_solved.ipynb
```

The notebook regenerates `figures/` and `results_summary.json`. Seeds are fixed. Small floating-point differences may occur across library versions. Execution was checked with Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1, and Matplotlib 3.11.1.

The supplied data have no declared time unit. The notebook uses `u` for that unit and `1/u` for rates. It does not assume seconds.

The submission follows the slides: a Jupyter notebook with Markdown and code, explicit justification of modeling and numerical choices, and discussion of results. Helper scripts keep the notebook organized.
