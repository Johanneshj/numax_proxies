# Numax Proxies

**Numax Proxies** fetches and prepares stellar light curves from the *Kepler* and *TESS* missions.  
By various methods $\nu_\text{max}$ estimates are calculated.

---
## 🤓 Methods
  1. **2D ACF:**
    - Follows the methodology of Huber+ 2009 and Viani+ 2019
    - Computes 2D ACF from PSD $\rightarrow$ collapsed the ACF to 1D $\rightarrow$ fits Gaussian to estimate $\nu_\text{max}$.
## 🌟 Features

- **Flexible input options** — works with:
  - a simple **target ID** (`KIC` or `TIC`),
  - a detailed **JSON configuration file**,
  - direct **time and flux arrays**, or
  - a **default example star** if no input is given.
- **Automatic source prioritization** — intelligently selects the best available data author (e.g., KASOC, Kepler, TASOC, SPOC, etc.).
- **LightKurve backend** — leverages [Lightkurve](https://docs.lightkurve.org/) for searching and downloading mission data.
- **Custom metadata support** — supports adding extra fields like `logg`, `Teff`, or `sector` for future extensions or proxy modeling.

---

## ⚙️ Installation

```bash
pip install lightkurve numpy
