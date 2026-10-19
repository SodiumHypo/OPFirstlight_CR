/*
 * cr_battle_stream.c — read live Clash Royale battle state from a Null's Royale
 * client process on MuMu x86_64 (ARM64 libg.so under translation).
 *
 * Rebuilt for pipeline/ from two read-only sources:
 *   - FirstLight_CR native_runner/probe/cr_replay_probe.cpp: struct chains
 *     calibrated for libg.so sha256 110aa2b5… / build id 90e6f351… (v15).
 *   - cr-memory-reader x86_64 helper: /proc/<pid>/mem pread scaffolding and
 *     entity-record canaries.
 *
 * Root discovery is a runtime semantic scan over libg.so writable globals:
 * a slot S wins when *S (or *( *S + 0x90 ), or *( *( *S + 0x20 ) + 0x90 ))
 * validates as a battle GameState (manager):
 *   manager+0x60 tick i32, manager+0xa8 world,
 *   world+0xe0/0xe8 players, world+0x30/0x88 identities/decks,
 *   player+0x2f8 elixir_raw (/10000), player+0x220 hand vec, player+0x230 cycle vec,
 *   player+0x10 object_manager, object_manager+0x08/0x10/0x14 object vector,
 *   object+0x08 nativeObjectId, +0x30 kind, +0x78 owner, +0x7c/0x80 pos,
 *   +0x84/0x88 target, +0xac cardId, +0x120 level idx,
 *   object+0x18 components vec, components[2] hitpoint comp (+0x10 hp, +0x14 max,
 *   +0x28 shield, +0x2c maxShield).
 *
 * Output: one JSON object per frame on stdout (NDJSON). No battle: short
 * battle_active:false frames. Exit codes: 2 usage, 3 no libg, 4 no mem,
 * 5 fingerprint mismatch.
 *
 * Build (see build_native_stream.sh): static Linux x86_64 ELF via WSL gcc
 *   (cross-compile from Windows; MSVC cannot target Linux ELF).
 * Run on device (root required, PID via `ps -A | grep nulls`): cr_battle_stream PID [INTERVAL_MS]
 */
#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* Fingerprint gate: this helper's struct offsets are calibrated for exactly
 * this libg build. Refuse anything else (GNU build id). */
static const char EXPECTED_BUILD_ID[] = "90e6f351f0dec4a28b494c3812d2629fd45d5a83";

#define MAX_OBJECTS 2048
#define ENTITY_RECORD 0x128
#define MAX_COMPONENTS 16
#define MAX_REGIONS 1024
#define MAX_HEAP_RANGES 2048
#define BSS_ADJACENT_BYTES (8ULL * 1024 * 1024)
#define MAX_TICK 9000
#define MAX_ELIXIR_RAW 100001

/* ------------------------------------------------------------------ memory */

typedef struct {
  uint64_t start, end;
} Range;

static Range g_heap_ranges[MAX_HEAP_RANGES];
static int g_heap_range_count = 0;

typedef struct {
  uint64_t start, end;
} Region;

static Region g_rw_regions[MAX_REGIONS];
static int g_rw_region_count = 0;
static uint64_t g_libg_base = 0;
static char g_libg_path[512] = {0};

static int read_exact(int fd, uint64_t address, void *output, size_t size) {
  uint8_t *cursor = (uint8_t *)output;
  size_t done = 0;
  while (done < size) {
    ssize_t got = pread(fd, cursor + done, size - done, (off_t)(address + done));
    if (got <= 0) return 0;
    done += (size_t)got;
  }
  return 1;
}

static int read_u64(int fd, uint64_t address, uint64_t *out) {
  return read_exact(fd, address, out, 8);
}
static int read_i32(int fd, uint64_t address, int32_t *out) {
  return read_exact(fd, address, out, 4);
}

static int in_heap(uint64_t ptr) {
  if (ptr < 0x10000ULL || ptr > 0x7fffffffffffULL) return 0;
  int lo = 0, hi = g_heap_range_count - 1;
  while (lo <= hi) {
    int mid = (lo + hi) / 2;
    if (ptr < g_heap_ranges[mid].start) hi = mid - 1;
    else if (ptr >= g_heap_ranges[mid].end) lo = mid + 1;
    else return 1;
  }
  return 0;
}

/* Parse /proc/<pid>/maps: libg.so load base + path, libg rw regions (plus a
 * bounded anonymous window after the last mapping for .bss), and plausible
 * heap ranges (anonymous rw, [heap], [anon:...]) for pointer pruning. */
static int parse_maps(int pid) {
  char path[64];
  snprintf(path, sizeof(path), "/proc/%d/maps", pid);
  FILE *maps = fopen(path, "r");
  if (!maps) return 0;
  g_heap_range_count = 0;
  g_rw_region_count = 0;
  g_libg_base = 0;
  g_libg_path[0] = '\0';
  char line[1024];
  uint64_t last_libg_end = 0;
  while (fgets(line, sizeof(line), maps)) {
    uint64_t start = 0, end = 0, fileoff = 0;
    char perms[8] = {0};
    char name[512] = {0};
    if (sscanf(line, "%llx-%llx %7s %llx %*s %*s %511[^\n]", (unsigned long long *)&start,
               (unsigned long long *)&end, perms, (unsigned long long *)&fileoff, name) < 4)
      continue;
    const char *trimmed = name;
    while (*trimmed == ' ' || *trimmed == '\t') ++trimmed;
    const int is_libg = strstr(trimmed, "/libg.so") != NULL;
    if (is_libg) {
      uint64_t back = start >= fileoff ? start - fileoff : 0;
      if (g_libg_base == 0 || back < g_libg_base) g_libg_base = back;
      if (strlen(trimmed) < sizeof(g_libg_path)) strcpy(g_libg_path, trimmed);
      if (end > last_libg_end) last_libg_end = end;
      if (perms[1] == 'w' && g_rw_region_count < MAX_REGIONS) {
        g_rw_regions[g_rw_region_count].start = start;
        g_rw_regions[g_rw_region_count].end = end;
        ++g_rw_region_count;
      }
      g_heap_ranges[g_heap_range_count].start = start; /* libg rw is readable heap too */
      g_heap_ranges[g_heap_range_count].end = end;
      if (g_heap_range_count < MAX_HEAP_RANGES) ++g_heap_range_count;
    } else if (trimmed[0] == '\0' || strstr(trimmed, "[heap]") == trimmed ||
               strstr(trimmed, "[anon:") == trimmed) {
      if (perms[1] == 'w') {
        /* anonymous window right after libg holds .bss; the rest is heap */
        if (last_libg_end != 0 && start >= last_libg_end && start < last_libg_end + BSS_ADJACENT_BYTES &&
            g_rw_region_count < MAX_REGIONS) {
          g_rw_regions[g_rw_region_count].start = start;
          g_rw_regions[g_rw_region_count].end = end;
          ++g_rw_region_count;
        }
        if (end - start <= (256ULL * 1024 * 1024) && g_heap_range_count < MAX_HEAP_RANGES) {
          g_heap_ranges[g_heap_range_count].start = start;
          g_heap_ranges[g_heap_range_count].end = end;
          ++g_heap_range_count;
        }
      }
    }
  }
  fclose(maps);
  /* sort heap ranges for binary search (insertion sort; ranges are few) */
  for (int i = 1; i < g_heap_range_count; ++i) {
    Range key = g_heap_ranges[i];
    int j = i - 1;
    while (j >= 0 && g_heap_ranges[j].start > key.start) {
      g_heap_ranges[j + 1] = g_heap_ranges[j];
      --j;
    }
    g_heap_ranges[j + 1] = key;
  }
  int merged = 0;
  for (int i = 0; i < g_heap_range_count; ++i) {
    if (merged > 0 && g_heap_ranges[i].start <= g_heap_ranges[merged - 1].end) {
      if (g_heap_ranges[i].end > g_heap_ranges[merged - 1].end)
        g_heap_ranges[merged - 1].end = g_heap_ranges[i].end;
    } else {
      g_heap_ranges[merged++] = g_heap_ranges[i];
    }
  }
  g_heap_range_count = merged;
  return g_libg_base != 0;
}

/* ------------------------------------------------------------- fingerprint */

/* Read the GNU build id out of an ELF file (PT_NOTE, NT_GNU_BUILD_ID). */
static int read_build_id(const char *path, char *out, size_t out_size) {
  FILE *f = fopen(path, "rb");
  if (!f) return 0;
  uint8_t ehdr[64];
  if (fread(ehdr, 1, sizeof(ehdr), f) != sizeof(ehdr) || memcmp(ehdr, "\177" "ELF", 4) != 0) {
    fclose(f);
    return 0;
  }
  uint64_t phoff;
  uint16_t phentsize, phnum;
  memcpy(&phoff, ehdr + 0x20, 8);
  memcpy(&phentsize, ehdr + 0x36, 2);
  memcpy(&phnum, ehdr + 0x38, 2);
  for (uint16_t i = 0; i < phnum; ++i) {
    uint8_t ph[56];
    if (fseek(f, (long)(phoff + (uint64_t)i * phentsize), SEEK_SET) != 0) break;
    if (fread(ph, 1, sizeof(ph), f) != sizeof(ph)) break;
    uint32_t p_type;
    uint64_t p_offset, p_filesz;
    memcpy(&p_type, ph + 0, 4);
    memcpy(&p_offset, ph + 8, 8);
    memcpy(&p_filesz, ph + 32, 8);
    if (p_type != 4 /* PT_NOTE */ || p_filesz < 16 || p_filesz > 65536) continue;
    /* clamp to the real file end: some note segments overstate their size */
    long file_end = -1;
    if (fseek(f, 0, SEEK_END) == 0) file_end = ftell(f);
    if (file_end > 0) {
      if ((uint64_t)file_end <= p_offset) continue;
      if ((uint64_t)file_end < p_offset + p_filesz) p_filesz = (uint64_t)file_end - p_offset;
    }
    uint8_t *note = (uint8_t *)malloc(p_filesz);
    if (!note) break;
    size_t note_got = 0;
    if (fseek(f, (long)p_offset, SEEK_SET) == 0) note_got = fread(note, 1, p_filesz, f);
    if (note_got >= 16) {
      uint64_t pos = 0;
      while (pos + 12 <= note_got) {
        uint32_t namesz, descsz, ntype;
        memcpy(&namesz, note + pos, 4);
        memcpy(&descsz, note + pos + 4, 4);
        memcpy(&ntype, note + pos + 8, 4);
        const uint64_t total = 12 + (uint64_t)namesz + descsz;
        if (pos + total > note_got) break; /* truncated entry: stop walking */
        if (namesz > 64 || descsz > 128) {
          /* oversized entry (e.g. the Android note): skip, keep walking */
          pos = (pos + total + 3) & ~(uint64_t)3;
          continue;
        }
        if (ntype == 3 && namesz >= 4 && memcmp(note + pos + 12, "GNU\0", 4) == 0) {
          const uint8_t *desc = note + pos + 12 + namesz;
          size_t used = 0;
          for (uint32_t b = 0; b < descsz && used + 2 < out_size; ++b) {
            static const char hexd[] = "0123456789abcdef";
            out[used++] = hexd[desc[b] >> 4];
            out[used++] = hexd[desc[b] & 0xf];
          }
          out[used] = '\0';
          free(note);
          fclose(f);
          return 1;
        }
        pos = (pos + total + 3) & ~(uint64_t)3;
      }
    }
    free(note);
  }
  fclose(f);
  return 0;
}

static int fingerprint_ok(void) {
  if (!g_libg_path[0]) {
    fprintf(stderr, "fingerprint: libg path not resolved from maps\n");
    return 0;
  }
  char build_id[128] = {0};
  if (!read_build_id(g_libg_path, build_id, sizeof(build_id))) {
    fprintf(stderr, "fingerprint: build id not found in %s\n", g_libg_path);
    return 0;
  }
  if (strcmp(build_id, EXPECTED_BUILD_ID) != 0) {
    printf("{\"event\":\"fatal\",\"error\":\"fingerprint_build_id\",\"expected\":\"%s\",\"actual\":\"%s\"}\n",
           EXPECTED_BUILD_ID, build_id);
    return 0;
  }
  return 1;
}

/* ------------------------------------------------------- manager validation */

static int validate_manager(int fd, uint64_t m) {
  if (!in_heap(m)) return 0;
  int32_t tick = -1;
  uint64_t world = 0, p0 = 0, p1 = 0, om = 0, objs = 0;
  int32_t e0 = -1, e1 = -1, cap = -1, cnt = -1;
  if (!read_i32(fd, m + 0x60, &tick) || tick < 0 || tick > MAX_TICK) return 0;
  if (!read_u64(fd, m + 0xa8, &world) || !in_heap(world)) return 0;
  if (!read_u64(fd, world + 0xe0, &p0) || !in_heap(p0)) return 0;
  if (!read_u64(fd, world + 0xe8, &p1) || !in_heap(p1) || p1 == p0) return 0;
  if (!read_i32(fd, p0 + 0x2f8, &e0) || e0 < 0 || e0 > MAX_ELIXIR_RAW) return 0;
  if (!read_i32(fd, p1 + 0x2f8, &e1) || e1 < 0 || e1 > MAX_ELIXIR_RAW) return 0;
  if (!read_u64(fd, p0 + 0x10, &om) || !in_heap(om)) return 0;
  if (!read_u64(fd, om + 0x08, &objs) || !in_heap(objs)) return 0;
  if (!read_i32(fd, om + 0x10, &cap) || !read_i32(fd, om + 0x14, &cnt)) return 0;
  if (cnt < 0 || cap < cnt || cap > MAX_OBJECTS) return 0;
  return 1;
}

/* Root slot state: remembered across frames, rescanned on validation loss. */
static uint64_t g_root_slot = 0;
static int g_root_variant = 0;

static int manager_from_slot(int fd, uint64_t slot, int variant, uint64_t *m_out, int32_t *tick_out) {
  uint64_t v = 0;
  if (!read_u64(fd, slot, &v) || !in_heap(v)) return 0;
  uint64_t m = 0;
  if (variant == 1) {
    m = v;
  } else if (variant == 2) {
    if (!read_u64(fd, v + 0x90, &m) || !in_heap(m)) return 0;
  } else {
    uint64_t c = 0;
    if (!read_u64(fd, v + 0x20, &c) || !in_heap(c)) return 0;
    if (!read_u64(fd, c + 0x90, &m) || !in_heap(m)) return 0;
  }
  if (!m) return 0;
  int32_t tick = -1;
  if (!validate_manager(fd, m)) return 0;
  read_i32(fd, m + 0x60, &tick);
  *m_out = m;
  *tick_out = tick;
  return 1;
}

/* Find a validating GameState root slot. The lobby keeps a frozen (tick==0)
 * state object alive, and a new battle may allocate its own manager — so a
 * manager with an advancing tick is the real match. With prefer_battle, an
 * advancing candidate is returned immediately; a frozen one is only a
 * fallback. */
static int find_root(int fd, uint64_t *slot_out, int *variant_out, uint64_t *m_out, int prefer_battle) {
  uint64_t fb_slot = 0;
  int fb_variant = 0;
  uint64_t fb_manager = 0;
  for (int r = 0; r < g_rw_region_count; ++r) {
    const uint64_t size = g_rw_regions[r].end - g_rw_regions[r].start;
    if (size == 0 || size > (64ULL * 1024 * 1024)) continue;
    uint8_t *buf = (uint8_t *)malloc((size_t)size);
    if (!buf) continue;
    if (read_exact(fd, g_rw_regions[r].start, buf, (size_t)size)) {
      for (uint64_t off = 0; off + 8 <= size; off += 8) {
        const uint64_t slot = g_rw_regions[r].start + off;
        uint64_t v;
        memcpy(&v, buf + off, 8);
        if (!in_heap(v)) continue;
        for (int variant = 1; variant <= 3; ++variant) {
          uint64_t m = 0;
          int32_t tick = -1;
          if (manager_from_slot(fd, slot, variant, &m, &tick)) {
            if (tick > 0 || !prefer_battle) {
              *slot_out = slot;
              *variant_out = variant;
              *m_out = m;
              free(buf);
              return 1;
            }
            if (!fb_slot) {
              fb_slot = slot;
              fb_variant = variant;
              fb_manager = m;
            }
            break;
          }
        }
      }
    }
    free(buf);
  }
  if (fb_slot) {
    *slot_out = fb_slot;
    *variant_out = fb_variant;
    *m_out = fb_manager;
    return 1;
  }
  return 0;
}

/* ------------------------------------------------------------------- reads */

typedef struct {
  uint64_t player, identity, deck_obj;
  uint32_t acct_hi, acct_lo;
  int32_t elixir_raw;
  int32_t hand_slot_count, hand_slots[8];
  int32_t cycle_count, cycle_slots[8];
  int32_t deck_count, deck_card_ids[8];
} PlayerRead;

static int read_vector_i32(int fd, uint64_t obj, uint64_t off, int32_t *count_out, int32_t *values,
                           int max, uint64_t *data_out) {
  uint64_t data = 0;
  int32_t cap = -1, cnt = -1;
  if (!read_u64(fd, obj + off, &data)) return 0;
  if (!read_i32(fd, obj + off + 8, &cap) || !read_i32(fd, obj + off + 0xc, &cnt)) return 0;
  if (cnt < 0 || cap < cnt || cap > max) return 0;
  *count_out = cnt;
  if (data_out) *data_out = data;
  if (cnt == 0) return 1;
  if (!in_heap(data)) return 0;
  return read_exact(fd, data, values, (size_t)cnt * 4);
}

static int read_player(int fd, uint64_t world, int owner, PlayerRead *p) {
  memset(p, 0, sizeof(*p));
  p->elixir_raw = -1;
  if (!read_u64(fd, world + 0xe0 + (uint64_t)owner * 8, &p->player)) return 0;
  if (!read_u64(fd, world + 0x30 + (uint64_t)owner * 8, &p->identity)) return 0;
  if (!read_u64(fd, world + 0x88 + (uint64_t)owner * 8, &p->deck_obj)) return 0;
  if (!in_heap(p->player) || !in_heap(p->identity) || !in_heap(p->deck_obj)) return 0;
  read_exact(fd, p->identity, &p->acct_hi, 4);
  read_exact(fd, p->identity + 4, &p->acct_lo, 4);
  read_i32(fd, p->player + 0x2f8, &p->elixir_raw);
  read_vector_i32(fd, p->player, 0x220, &p->hand_slot_count, p->hand_slots, 8, NULL);
  read_vector_i32(fd, p->player, 0x230, &p->cycle_count, p->cycle_slots, 8, NULL);
  /* deck: deck_obj+0x20 vector of battle_deck_slot pointers */
  uint64_t data = 0;
  int32_t cap = -1, cnt = -1;
  if (read_u64(fd, p->deck_obj + 0x20, &data) && read_i32(fd, p->deck_obj + 0x28, &cap) &&
      read_i32(fd, p->deck_obj + 0x2c, &cnt) && cnt >= 0 && cap >= cnt && cnt <= 8 && in_heap(data)) {
    p->deck_count = cnt;
    for (int32_t i = 0; i < cnt; ++i) {
      p->deck_card_ids[i] = -1;
      uint64_t slot_ptr = 0, card_data = 0;
      int32_t card_id = -1;
      if (read_u64(fd, data + (uint64_t)i * 8, &slot_ptr) && in_heap(slot_ptr) &&
          read_u64(fd, slot_ptr + 0x10, &card_data) && in_heap(card_data) &&
          read_i32(fd, card_data + 0x40, &card_id))
        p->deck_card_ids[i] = card_id;
    }
  } else {
    p->deck_count = 0;
  }
  if (p->elixir_raw < 0 || p->elixir_raw > MAX_ELIXIR_RAW) return 0;
  return 1;
}

typedef struct {
  uint32_t native_object_id;
  int32_t kind, owner, x, y, tx, ty, card_id, level, hp, max_hp, shield, max_shield;
} ObjectRead;

static int sane_object(const uint8_t *raw) {
  int32_t kind, owner, x, y, tx, ty, card_id, level;
  uint32_t nid;
  memcpy(&nid, raw + 0x08, 4);
  memcpy(&kind, raw + 0x30, 4);
  memcpy(&owner, raw + 0x78, 4);
  memcpy(&x, raw + 0x7c, 4);
  memcpy(&y, raw + 0x80, 4);
  memcpy(&tx, raw + 0x84, 4);
  memcpy(&ty, raw + 0x88, 4);
  memcpy(&card_id, raw + 0xac, 4);
  memcpy(&level, raw + 0x120, 4);
  if (nid == 0 || nid < 1000000 || nid > 2000000000) return 0;
  if (kind < 10 || kind > 20 || (owner != 0 && owner != 1)) return 0;
  if (x < 0 || x > 18000 || y < 0 || y > 32000) return 0;
  if (tx < 0 || tx > 18000 || ty < 0 || ty > 32000) return 0;
  if (level < 0 || level > 16) return 0;
  if (card_id != -1 && (card_id < 20000000 || card_id >= 1000000000)) return 0;
  return 1;
}

static int read_object(int fd, uint64_t addr, ObjectRead *o) {
  uint8_t raw[ENTITY_RECORD];
  memset(raw, 0, sizeof(raw));
  if (!read_exact(fd, addr, raw, sizeof(raw))) return 0;
  if (!sane_object(raw)) return 0;
  memcpy(&o->native_object_id, raw + 0x08, 4);
  memcpy(&o->kind, raw + 0x30, 4);
  memcpy(&o->owner, raw + 0x78, 4);
  memcpy(&o->x, raw + 0x7c, 4);
  memcpy(&o->y, raw + 0x80, 4);
  memcpy(&o->tx, raw + 0x84, 4);
  memcpy(&o->ty, raw + 0x88, 4);
  memcpy(&o->card_id, raw + 0xac, 4);
  memcpy(&o->level, raw + 0x120, 4);
  o->level += 1;
  o->hp = o->max_hp = o->shield = o->max_shield = -1;
  /* components: obj+0x18 {data,+8 cap,+0xc count}; hp component is slot 2 */
  uint64_t comp_data = 0;
  int32_t ccap = -1, ccnt = -1;
  if (read_u64(fd, addr + 0x18, &comp_data) && read_i32(fd, addr + 0x20, &ccap) &&
      read_i32(fd, addr + 0x24, &ccnt) && ccnt > 2 && ccap >= ccnt && ccap <= MAX_COMPONENTS && in_heap(comp_data)) {
    /* backlink check: comp+0x08 must equal the object address */
    uint64_t comp = 0;
    if (read_u64(fd, comp_data + 2 * 8, &comp) && in_heap(comp)) {
      uint64_t back = 0;
      if (read_u64(fd, comp + 0x08, &back) && back == addr) {
        int32_t hp = -1, max_hp = -1, shield = -1, max_shield = -1;
        if (read_i32(fd, comp + 0x10, &hp) && read_i32(fd, comp + 0x14, &max_hp) && max_hp >= 0 && hp >= 0 &&
            max_hp <= 50000 && hp <= max_hp) {
          o->hp = hp;
          o->max_hp = max_hp;
          if (read_i32(fd, comp + 0x28, &shield) && read_i32(fd, comp + 0x2c, &max_shield) && max_shield >= 0 &&
              shield >= 0 && shield <= max_shield) {
            o->shield = shield;
            o->max_shield = max_shield;
          }
        }
      }
    }
  }
  return 1;
}

/* -------------------------------------------------------------------- main */

static uint64_t monotonic_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

static uint64_t monotonic_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

int main(int argc, char **argv) {
  signal(SIGPIPE, SIG_DFL);
  if (argc < 2 || argc > 3) {
    fprintf(stderr, "usage: cr_battle_stream PID [INTERVAL_MS]\n");
    return 2;
  }
  const int pid = atoi(argv[1]);
  const int interval_ms = argc == 3 ? atoi(argv[2]) : 100;
  if (pid <= 0 || interval_ms < 10 || interval_ms > 5000) return 2;

  if (!parse_maps(pid)) {
    fprintf(stderr, "libg.so mapping not found\n");
    return 3;
  }
  if (!fingerprint_ok()) return 5;

  char mem_path[64];
  snprintf(mem_path, sizeof(mem_path), "/proc/%d/mem", pid);
  const int fd = open(mem_path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    fprintf(stderr, "open %s failed: %s\n", mem_path, strerror(errno));
    return 4;
  }
  setvbuf(stdout, NULL, _IONBF, 0);

  uint64_t seq = 0;
  for (;;) {
    const uint64_t started_us = monotonic_us();
    const uint64_t t_ms = monotonic_ms();

    uint64_t manager = 0;
    if (g_root_slot != 0) {
      int32_t tracked_tick = -1;
      if (!manager_from_slot(fd, g_root_slot, g_root_variant, &manager, &tracked_tick)) {
        g_root_slot = 0;
        manager = 0;
      }
    }
    /* Periodically re-resolve the root: the lobby keeps a frozen (tick==0)
     * state object that also validates, and a fresh battle may allocate its
     * own manager — prefer one with an advancing tick. */
    if (g_root_slot == 0 || (seq % 100) == 0) {
      parse_maps(pid);
      uint64_t reslot = 0;
      int revariant = 0;
      uint64_t remanager = 0;
      if (find_root(fd, &reslot, &revariant, &remanager, 1)) {
        g_root_slot = reslot;
        g_root_variant = revariant;
        manager = remanager;
      }
    }

    int battle_active = manager != 0;
    int32_t tick = -1;
    if (battle_active) {
      if (!read_i32(fd, manager + 0x60, &tick) || tick < 0 || tick > MAX_TICK) {
        battle_active = 0;
        g_root_slot = 0;
      }
    }

    printf("{\"event\":\"battle_frame\",\"seq\":%" PRIu64 ",\"t_ms\":%" PRIu64 ",\"battle_active\":%s,\"tick\":%" PRId32,
           seq++, t_ms, battle_active ? "true" : "false", battle_active ? tick : -1);

    PlayerRead players[2];
    int players_ok = 0;
    if (battle_active) {
      uint64_t world = 0;
      if (read_u64(fd, manager + 0xa8, &world) && in_heap(world)) {
        players_ok = read_player(fd, world, 0, &players[0]) && read_player(fd, world, 1, &players[1]);
      } else {
        battle_active = 0;
        g_root_slot = 0;
      }
    }

    printf(",\"root_slot\":\"%s\",\"players\":[", g_root_slot ? "tracked" : "none");
    if (players_ok) {
      for (int owner = 0; owner < 2; ++owner) {
        const PlayerRead *p = &players[owner];
        if (owner) putchar(',');
        printf("{\"owner\":%d,\"acctHi\":%u,\"acctLo\":%u,\"elixirRaw\":%d,\"elixir\":%.4f,\"hand\":[", owner,
               p->acct_hi, p->acct_lo, p->elixir_raw, (double)p->elixir_raw / 10000.0);
        for (int32_t i = 0; i < p->hand_slot_count; ++i) {
          if (i) putchar(',');
          const int32_t ds = p->hand_slots[i];
          const int32_t cid = (ds >= 0 && ds < p->deck_count) ? p->deck_card_ids[ds] : -1;
          printf("{\"slot\":%d,\"deckSlot\":%d,\"cardId\":%d}", i, ds, cid);
        }
        printf("],\"cycle\":[");
        for (int32_t i = 0; i < p->cycle_count; ++i) {
          if (i) putchar(',');
          const int32_t ds = p->cycle_slots[i];
          printf("%d", (ds >= 0 && ds < p->deck_count) ? p->deck_card_ids[ds] : -1);
        }
        printf("],\"deck\":[");
        for (int32_t i = 0; i < p->deck_count; ++i) {
          if (i) putchar(',');
          printf("{\"slot\":%d,\"cardId\":%d}", i, p->deck_card_ids[i]);
        }
        printf("]}");
      }
    }

    printf("],\"objects\":[");
    int emitted = 0;
    if (battle_active && players_ok) {
      uint64_t om = 0, data = 0;
      int32_t cap = -1, cnt = -1;
      if (read_u64(fd, players[0].player + 0x10, &om) && in_heap(om) && read_u64(fd, om + 0x08, &data) &&
          in_heap(data) && read_i32(fd, om + 0x10, &cap) && read_i32(fd, om + 0x14, &cnt) && cnt >= 0 &&
          cap >= cnt && cnt <= MAX_OBJECTS) {
        for (int32_t i = 0; i < cnt; ++i) {
          uint64_t addr = 0;
          if (!read_u64(fd, data + (uint64_t)i * 8, &addr) || !in_heap(addr)) continue;
          ObjectRead o;
          if (!read_object(fd, addr, &o)) continue;
          if (emitted++) putchar(',');
          printf("{\"id\":%u,\"kind\":%d,\"owner\":%d,\"x\":%d,\"y\":%d,\"tx\":%d,\"ty\":%d,\"cardId\":%d,"
                 "\"level\":%d,\"hp\":%d,\"maxHp\":%d,\"shield\":%d,\"maxShield\":%d}",
                 o.native_object_id, o.kind, o.owner, o.x, o.y, o.tx, o.ty, o.card_id, o.level, o.hp, o.max_hp,
                 o.shield, o.max_shield);
        }
      }
    }
    printf("],\"read_us\":%" PRIu64 "}\n", monotonic_us() - started_us);

    const uint64_t elapsed_us = monotonic_us() - started_us;
    const uint64_t target_us = (uint64_t)interval_ms * 1000ULL;
    if (elapsed_us < target_us) usleep((useconds_t)(target_us - elapsed_us));
  }
}
