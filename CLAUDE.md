# Working agreements

- Never start a long-running test or script that uses a big chunk of CPU/GPU without asking first and giving a time estimate. The user works on this same PC while these run, and an unannounced heavy job lags their machine. This applies even to "quick" probes if they involve a model load or multiple diffusion/inference steps -- give the estimate, wait for a go-ahead, then run (foreground or background as agreed).
