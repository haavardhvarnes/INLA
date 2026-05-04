# R-INLA oracle fixture regeneration.
#
# Each fixture is a two-step pipeline:
#   1. R script produces a JSON dump in fixtures/<pkg>/<name>.json.
#   2. Julia `convert_to_jld2.jl` converts it to a JLD2 under the
#      destination package's test/oracle/fixtures/ directory.
#
# `make all` regenerates everything. `make lgm/scotland_bym2` regenerates
# a single fixture (both steps).

R          ?= Rscript
JULIA      ?= julia
ROOT       := $(realpath $(CURDIR)/../..)
JSON_DIR   := $(CURDIR)/fixtures
CONVERT    := $(JULIA) --project=$(CURDIR) $(CURDIR)/convert_to_jld2.jl

# -- Fixture registry ------------------------------------------------
# Each entry: <category>/<name> => (R script, destination JLD2 path)

GMRFS_FIXTURES := qinv_rw2
LGM_FIXTURES   := scotland_bym2 pennsylvania_bym2 synthetic_nbinomial synthetic_gamma scotland_bym synthetic_disconnected_besag synthetic_generic0 synthetic_generic1 synthetic_seasonal synthetic_leroux synthetic_exponential_survival synthetic_weibull_survival synthetic_lognormal_survival synthetic_gamma_survival synthetic_coxph synthetic_joint_gauss_pois synthetic_zip1 synthetic_baghfalaki synthetic_iid2d synthetic_replicate_ar1 synthetic_beta synthetic_betabinomial
SPDE_FIXTURES  := fmesher_unit_square fmesher_lshape fmesher_meuse_hull meuse_spde

GMRFS_DEST := $(ROOT)/packages/GMRFs.jl/test/oracle/fixtures
LGM_DEST   := $(ROOT)/packages/LatentGaussianModels.jl/test/oracle/fixtures
SPDE_DEST  := $(ROOT)/packages/INLASPDE.jl/test/oracle/fixtures

GMRFS_JSONS := $(addprefix $(JSON_DIR)/gmrfs/,$(addsuffix .json,$(GMRFS_FIXTURES)))
LGM_JSONS   := $(addprefix $(JSON_DIR)/lgm/,$(addsuffix .json,$(LGM_FIXTURES)))
SPDE_JSONS  := $(addprefix $(JSON_DIR)/spde/,$(addsuffix .json,$(SPDE_FIXTURES)))

GMRFS_JLD2  := $(addprefix $(GMRFS_DEST)/,$(addsuffix .jld2,$(GMRFS_FIXTURES)))
LGM_JLD2    := $(addprefix $(LGM_DEST)/,$(addsuffix .jld2,$(LGM_FIXTURES)))
SPDE_JLD2   := $(addprefix $(SPDE_DEST)/,$(addsuffix .jld2,$(SPDE_FIXTURES)))

.PHONY: all gmrfs lgm spde clean setup

all: gmrfs lgm spde

gmrfs: $(GMRFS_JLD2)

lgm: $(LGM_JLD2)

spde: $(SPDE_JLD2)

# -- Pattern rules ---------------------------------------------------

$(JSON_DIR)/gmrfs/%.json: gmrfs/%.R _helpers.R
	@mkdir -p $(@D)
	cd $(<D) && $(R) $(<F)

$(JSON_DIR)/lgm/%.json: lgm/%.R _helpers.R
	@mkdir -p $(@D)
	cd $(<D) && $(R) $(<F)

$(JSON_DIR)/spde/%.json: spde/%.R _helpers.R
	@mkdir -p $(@D)
	cd $(<D) && $(R) $(<F)

$(GMRFS_DEST)/%.jld2: $(JSON_DIR)/gmrfs/%.json convert_to_jld2.jl
	@mkdir -p $(@D)
	$(CONVERT) $< $@

$(LGM_DEST)/%.jld2: $(JSON_DIR)/lgm/%.json convert_to_jld2.jl
	@mkdir -p $(@D)
	$(CONVERT) $< $@

$(SPDE_DEST)/%.jld2: $(JSON_DIR)/spde/%.json convert_to_jld2.jl
	@mkdir -p $(@D)
	$(CONVERT) $< $@

# -- Setup / maintenance ---------------------------------------------

setup:
	$(JULIA) --project=$(CURDIR) -e 'using Pkg; Pkg.instantiate()'

clean:
	rm -rf $(JSON_DIR)
