* Clean-sample demonstration .ado package command
program define winsorize_simple
    syntax varlist [if] [in] , GENerate(string) [LOw(real 0) HIgh(real 100)]
    marksample touse
    local v `varlist'
    quietly summarize `v' if `touse', detail
    local vmin = r(min)
    local vmax = r(max)
    if `low' > 0 {
        local vmin = r(p5)
    }
    if `high' < 100 {
        local vmax = r(p95)
    }
    generate `generate' = `v' if `touse'
    replace `generate' = `vmin' if `touse' & `v' < `vmin'
    replace `generate' = `vmax' if `touse' & `v' > `vmax'
    return scalar lower = `vmin'
    return scalar upper = `vmax'
end
