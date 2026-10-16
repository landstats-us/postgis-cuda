/* gsp_cat: send one request to gpujoind and drain the result stream (throughput test / debugging).
 *   gsp_cat SOCKET OP ARG...      e.g. gsp_cat run/gpujoind.sock 6 parcels parcels 1 distinct */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include "../daemon/protocol.h"
static char buf[1 << 22]; static size_t pos, len; static int fd; static uint64_t bytes;
static void fill(void) { memmove(buf, buf + pos, len - pos); len -= pos; pos = 0; ssize_t k = read(fd, buf + len, sizeof buf - len); if (k <= 0) { fprintf(stderr, "eof\n"); exit(1); } len += k; bytes += k; }
static const char* take(size_t n) { while (len - pos < n) fill(); const char* p = buf + pos; pos += n; return p; }
static uint32_t u32(void) { uint32_t v; memcpy(&v, take(4), 4); return v; }
int main(int argc, char** argv) {
  struct sockaddr_un sa = {0}; sa.sun_family = AF_UNIX; strncpy(sa.sun_path, argv[1], sizeof sa.sun_path - 1);
  fd = socket(AF_UNIX, SOCK_STREAM, 0); if (connect(fd, (struct sockaddr*)&sa, sizeof sa)) { perror("connect"); return 1; }
  const char* tok = getenv("GSP_TOKEN") ? getenv("GSP_TOKEN") : "";
  uint32_t op = atoi(argv[2]), n = argc - 3 + 1, h[3] = {GSP_MAGIC, op, n}; write(fd, h, 12);
  { uint32_t l = strlen(tok); write(fd, &l, 4); write(fd, tok, l); }
  for (int i = 3; i < argc; i++) { uint32_t l = strlen(argv[i]); write(fd, &l, 4); write(fd, argv[i], l); }
  struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
  if (u32() != GSP_OK) { uint32_t l = u32(); fprintf(stderr, "error: %.*s\n", l, take(l)); return 1; }
  uint64_t rows = 0;
  if (op == GSP_OP_JOIN || op == GSP_OP_CLUSTER || op == GSP_OP_DWITHIN) {
    for (;;) {
      uint32_t k = u32(); if (k == 0) break;
      if (k == GSP_FRAME_ERR) { uint32_t l = u32(); fprintf(stderr, "stream error: %.*s\n", l, take(l)); return 1; }
      for (uint32_t i = 0; i < k; i++) {
        uint16_t la; memcpy(&la, take(2), 2); take(la);
        if (op != GSP_OP_CLUSTER) { uint16_t lb; memcpy(&lb, take(2), 2); take(lb); take(8); } else take(4);
      }
      rows += k;
    }
  }
  uint32_t l = u32(); const char* js = take(l);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
  printf("rows=%llu bytes=%.1f MB time=%.3f s (%.1f Mrows/s, %.0f MB/s)\n%.*s\n", (unsigned long long)rows, bytes / 1e6, s, rows / s / 1e6, bytes / s / 1e6, l, js);
  return 0;
}
