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
LGM_FIXTURES   := scotland_bym2

GMRFS_DEST := $(ROOT)/packages/GMRFs.jl/test/oracle/fixtures
LGM_DEST   := $(ROOT)/packages/LatentGaussianModels.jl/test/oracle/fixtures

GMRFS_JSONS := $(addprefix $(JSON_DIR)/gmrfs/,$(addsuffix .json,$(GMRFS_FIXTURES)))
LGM_JSONS   := $(addprefix $(JSON_DIR)/lgm/,$(addsuffix .json,$(LGM_FIXTURES)))

GMRFS_JLD2  := $(addprefix $(GMRFS_DEST)/,$(addsuffix .jld2,$(GMRFS_FIXTURES)))
LGM_JLD2    := $(addprefix $(LGM_DEST)/,$(addsuffix .jld2,$(LGM_FIXTURES)))

.PHONY: all gmrfs lgm clean setup

all: gmrfs lgm

gmrfs: $(GMRFS_JLD2)

lgm: $(LGM_JLD2)

# -- Pattern rules ---------------------------------------------------

$(JSON_DIR)/gmrfs/%.json: gmrfs/%.R _helpers.R
	@mkdir -p $(@D)
	cd $(<D) && $(R) $(<F)

$(JSON_DIR)/lgm/%.json: lgm/%.R _helpers.R
	@mkdir -p $(@D)
	cd $(<D) && $(R) $(<F)

$(GMRFS_DEST)/%.jld2: $(JSON_DIR)/gmrfs/%.json convert_to_jld2.jl
	@mkdir -p $(@D)
	$(CONVERT) $< $@

$(LGM_DEST)/%.jld2: $(JSON_DIR)/lgm/%.json convert_to_jld2.jl
	@mkdir -p $(@D)
	$(CONVERT) $< $@

# -- Setup / maintenance ---------------------------------------------

setup:
	$(JULIA) --project=$(CURDIR) -e 'using Pkg; Pkg.instantiate()'

clean:
	rm -rf $(JSON_DIR)
