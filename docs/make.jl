using Documenter
using GMRFs
using LatentGaussianModels
using INLASPDE
using INLASPDERasters

DocMeta.setdocmeta!(GMRFs, :DocTestSetup, :(using GMRFs); recursive=true)
DocMeta.setdocmeta!(LatentGaussianModels, :DocTestSetup,
    :(using LatentGaussianModels); recursive=true)
DocMeta.setdocmeta!(INLASPDE, :DocTestSetup, :(using INLASPDE); recursive=true)
DocMeta.setdocmeta!(INLASPDERasters, :DocTestSetup,
    :(using INLASPDERasters); recursive=true)

makedocs(
    sitename="Julia INLA Ecosystem",
    authors="Julia INLA contributors",
    repo="https://github.com/HaavardHvarnes/INLA.jl/blob/{commit}{path}#{line}",
    modules=[GMRFs, LatentGaussianModels, INLASPDE, INLASPDERasters],
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://haavardhvarnes.github.io/INLA.jl/",
        edit_link="main",
        assets=String[]
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Vignettes" => [
            "Areal — Scotland BYM2" => "vignettes/scotland-bym2.md",
            "Temporal — Tokyo rainfall" => "vignettes/tokyo-rainfall.md",
            "Spatial — Meuse SPDE" => "vignettes/meuse-spde.md",
            "Survival — Cox PH and Weibull" => "vignettes/coxph-weibull-survival.md"
        ],
        "Benchmarks" => [
            "Quality vs R-INLA" => "benchmarks/quality.md"
        ],
        "Packages" => [
            "GMRFs.jl" => "packages/gmrfs.md",
            "LatentGaussianModels.jl" => "packages/lgm.md",
            "INLASPDE.jl" => "packages/inlaspde.md",
            "INLASPDERasters.jl" => "packages/inlaspderasters.md"
        ],
        "References" => "references.md"
    ],
    warnonly=[:missing_docs, :cross_references]
)

deploydocs(;
    repo="github.com/HaavardHvarnes/INLA.jl.git",
    devbranch="main",
    push_preview=true
)
