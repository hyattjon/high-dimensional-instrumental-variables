* Run Stata's JIVE commands on the shared data sets and write validation/results_stata.csv.
*
* Run from the repo root:   do validation/run_stata.do
* Needs (once):
*   ssc install sjive                      // Frandsen, Lefgren, Leslie, McIntyre; `noshrink` gives Kolesar's UJIVE
*   Poi (2006), "Jackknife instrumental variables estimation in Stata", Stata Journal 6(3): the `jive` command.
*   It is NOT on SSC. Find it with `findit jive` or `search jackknife instrumental` and install from the
*   Stata Journal link. If `jive` is missing, that block is skipped and the rest still runs.
*
* STATUS: run successfully (Stata 19 MP); results are in validation/results_stata.csv and results_stata.log.
* Findings, checked against our estimators in validation/compare.py and tests/test_stata_reference.py:
*   jive, ujive1 / ujive2 (robust)  == our UJIVE1 / UJIVE2 (coefficient and SE)
*   jive, jive1 / jive2             == our JIVE1 / JIVE2 (coefficient; the SEs differ because of a bug in jive.ado's CalcJIVE, see validation/README.md)
*   sjive, noshrink                 != R ujive() (Kolesar 2013): it also leaves observation i out of Y's partialling
* sjive with noshrink needed a one-line fix to run (see validation/README.md).

clear all
set more off

tempname h
tempfile res
postfile `h' str30 dataset str40 source double beta double se using `res', replace

foreach ds in n300_k10_nocontrols n300_k10_controls n60_k8_controls {
    import delimited "validation/data/`ds'.csv", clear varnames(1)
    quietly ds z*
    local zs `r(varlist)'
    capture confirm variable w1
    if _rc == 0 {
        local ws "w1 w2"
    }
    else {
        local ws ""
    }

    display as text _newline "=== `ds' ==="

    * sjive without shrinkage = Kolesar's UJIVE (Stata's ivregress 2sls y (t = P_hat) controls)
    capture noisily sjive y `ws' (t = `zs'), noshrink
    if _rc == 0 {
        capture post `h' ("`ds'") ("Stata sjive, noshrink") (_b[t]) (_se[t])
    }

    * Poi's jive command, all four variants; with and without robust standard errors
    foreach opt in jive1 jive2 ujive1 ujive2 {
        capture noisily jive y `ws' (t = `zs'), `opt'
        if _rc == 0 {
            capture post `h' ("`ds'") ("Stata jive, `opt'") (_b[t]) (_se[t])
            if _rc != 0 display as error "jive, `opt' ran but _b[t] was not found; check ereturn list"
        }
        capture noisily jive y `ws' (t = `zs'), `opt' robust
        if _rc == 0 {
            capture post `h' ("`ds'") ("Stata jive, `opt' robust") (_b[t]) (_se[t])
        }
    }
}

postclose `h'
use `res', clear
list, clean noobs
export delimited using "validation/results_stata.csv", replace
display as text _newline "Wrote validation/results_stata.csv. Send it back (or run: python validation/compare.py)."
