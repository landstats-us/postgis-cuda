#pragma once
// Host-side flat polygon set (see scripts/dump_gset.py for the on-disk layout).
#include "common.h"

struct HostSet {
  uint64_t n = 0, nr = 0, nv = 0;
  std::vector<uint64_t> gro;   // [n+1]  geometry -> first ring
  std::vector<uint64_t> rvo;   // [nr+1] ring -> first vertex (rings closed)
  std::vector<double> xy;      // [nv*2]
  std::vector<std::string> ids;
};

inline HostSet load_gset(const char* path) {
  HostSet s;
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  char magic[4]; uint32_t ver;
  if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GSET", 4) || fread(&ver, 4, 1, f) != 1 ||
      fread(&s.n, 8, 1, f) != 1 || fread(&s.nr, 8, 1, f) != 1 || fread(&s.nv, 8, 1, f) != 1) {
    fprintf(stderr, "bad gset header in %s\n", path); exit(1);
  }
  s.gro.resize(s.n + 1); s.rvo.resize(s.nr + 1); s.xy.resize(s.nv * 2);
  if (fread(s.gro.data(), 8, s.n + 1, f) != s.n + 1 || fread(s.rvo.data(), 8, s.nr + 1, f) != s.nr + 1 ||
      fread(s.xy.data(), 8, s.nv * 2, f) != s.nv * 2) { fprintf(stderr, "short read %s\n", path); exit(1); }
  fclose(f);
  std::ifstream in(std::string(path) + ".ids");
  std::string line;
  while (std::getline(in, line)) s.ids.push_back(line);
  return s;
}

// Flat point set (scripts/dump_pset.py): u32 'PSET', u32 version, u64 n, f64 xy[2n]; ids in .ids
struct HostPoints { uint64_t n = 0; std::vector<double> xy; std::vector<std::string> ids; };
inline HostPoints load_pset(const char* path) {
  HostPoints p; FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  char magic[4]; uint32_t ver;
  if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "PSET", 4) || fread(&ver, 4, 1, f) != 1 || fread(&p.n, 8, 1, f) != 1) { fprintf(stderr, "bad pset %s\n", path); exit(1); }
  p.xy.resize(2 * p.n);
  if (fread(p.xy.data(), 8, 2 * p.n, f) != 2 * p.n) { fprintf(stderr, "short read %s\n", path); exit(1); }
  fclose(f);
  std::ifstream in(std::string(path) + ".ids"); std::string line;
  while (std::getline(in, line)) p.ids.push_back(line);
  return p;
}
