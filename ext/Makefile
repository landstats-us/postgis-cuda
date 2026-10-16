MODULE_big = gpu_spatial
OBJS = gpu_spatial.o
EXTENSION = gpu_spatial
DATA = gpu_spatial--0.1.sql
PG_CPPFLAGS = -I$(srcdir)/../daemon
PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
