# Builds the gpu_spatial extension in ext/ (what `pgxn install postgis_cuda` runs).
# The daemon is built separately: scripts/build_daemon.sh.
.PHONY: all install installcheck uninstall clean
all install installcheck uninstall clean:
	$(MAKE) -C ext $@
