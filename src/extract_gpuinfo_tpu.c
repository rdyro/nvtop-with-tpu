/*
 *
 * Copyright (C) 2022 Maxime Schmitt <maxime.schmitt91@gmail.com>
 *
 * This file is part of Nvtop 
 *
 * Nvtop is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Nvtop is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with nvtop.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "nvtop/extract_gpuinfo_common.h"
#include "nvtop/extract_processinfo_fdinfo.h"
#include "nvtop/time.h"

#include "extract_gpuinfo_tpu.h"

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <uthash.h>

#include <sys/time.h>
#include <time.h>
#include <pthread.h>

static bool gpuinfo_tpu_init(void);
static void gpuinfo_tpu_shutdown(void);
static const char *gpuinfo_tpu_last_error_string(void);
static bool gpuinfo_tpu_get_device_handles(struct list_head *devices, unsigned *count);
static void gpuinfo_tpu_populate_static_info(struct gpu_info *_gpu_info);
static void gpuinfo_tpu_refresh_dynamic_info(struct gpu_info *_gpu_info);
static void gpuinfo_tpu_get_running_processes(struct gpu_info *_gpu_info);
void* query_tpu_data_thread_fn(void* _);
void populate_tpu_data(bool verbose);
int64_t millis(void);

struct gpu_vendor gpu_vendor_tpu = {
    .init = gpuinfo_tpu_init,
    .shutdown = gpuinfo_tpu_shutdown,
    .last_error_string = gpuinfo_tpu_last_error_string,
    .get_device_handles = gpuinfo_tpu_get_device_handles,
    .populate_static_info = gpuinfo_tpu_populate_static_info,
    .refresh_dynamic_info = gpuinfo_tpu_refresh_dynamic_info,
    .refresh_running_processes = gpuinfo_tpu_get_running_processes,
    .name = "TPU",
};

int64_t tpu_chip_count = -1;
static struct gpu_info_tpu *gpu_infos;

#define STRINGIFY(x) STRINGIFY_HELPER_(x)
#define STRINGIFY_HELPER_(x) #x

#define VENDOR_TPU 0x1111
#define VENDOR_TPU_STR STRINGIFY(VENDOR_TPU)

__attribute__((constructor)) static void init_extract_gpuinfo_tpu(void) { register_gpu_vendor(&gpu_vendor_tpu); }

pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
pthread_t tpu_query_thread = 0;
bool thread_should_exit = false;

int64_t millis(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return 1000000LL * tv.tv_sec + tv.tv_usec / 1000LL;
}

char python_script[] = 
    "from tpu_info import device, metrics\n"
    "try:\n"
    "  chip_type, count = device.get_local_chips()\n"
    "  chips_usage = metrics.get_chip_usage(chip_type)\n"
    "  for chip_usage in chips_usage:\n"
    "    print(f\"{chip_usage.device_id:d} {chip_usage.memory_usage:d}\""
    "          f\" {chip_usage.total_memory:d} {chip_usage.duty_cycle_pct:.4f}\""
    "          f\" {chip_type.value.name}\")\n"
    "except:\n"
    "  pass\n";
char *popen_command = NULL;

#define MAX_CHIPS_PER_HOST 256
struct tpu_chip_usage_data latest_chips_usage_data[MAX_CHIPS_PER_HOST];

void populate_tpu_data(bool verbose) {
  if (popen_command == NULL) {
    popen_command = (char*)malloc(2048);
    snprintf(popen_command, 2048, "python3 -c '%s'", python_script);
  }
  FILE* p = popen(popen_command, "r");
  char line[2048];
  int idx = 0; 
  while (fgets(line, 2048 - 1, p) != NULL) {
    line[2048 - 1] = '\0';
    if (idx >= MAX_CHIPS_PER_HOST || (tpu_chip_count > 0 && idx >= tpu_chip_count)) break;
    struct tpu_chip_usage_data usage_data;
    if (sscanf(line, "%ld %ld %ld %lf %7[^,]", 
      &usage_data.device_idx, &usage_data.memory_usage, &usage_data.total_memory, 
      &usage_data.duty_cycle_pct, (char*)&usage_data.name) == 5) {
        if (idx != usage_data.device_idx) {
          printf("Out of order TPU device found: %ld on line %d\n", usage_data.device_idx, idx);
          exit(1);
        }
        size_t len = strcspn(usage_data.name, "\n");
        usage_data.name[len] = '\0';
        pthread_mutex_lock(&m);
        latest_chips_usage_data[idx] = usage_data;
        pthread_mutex_unlock(&m);
    } else {
      printf("Error parsing TPU output line: %s\n", line);
      exit(1);
    }
    idx += 1;
    if (idx >= MAX_CHIPS_PER_HOST) break;
  }
  pclose(p);
  if (tpu_chip_count < 0) tpu_chip_count = idx;
  if (verbose) printf("Found %ld TPU chips\n", tpu_chip_count);
}

void* query_tpu_data_thread_fn(void* _){
  (void)_;
  int64_t t = millis();
  while(!thread_should_exit) {
    populate_tpu_data(false);
    do {
      usleep(10 * 1000);
    } while (!thread_should_exit && millis() - t < 1000);
    t = millis();
  }
  return NULL;
}

bool gpuinfo_tpu_init(void) { 
  populate_tpu_data(true);  // this also discovers the number of TPU chips
  return true; 
}

void gpuinfo_tpu_shutdown(void) { 
  thread_should_exit = true;
  if (tpu_query_thread != 0) {
    pthread_join(tpu_query_thread, NULL);
    tpu_query_thread = 0;
  }
  pthread_mutex_destroy(&m);
  if (gpu_infos != NULL) {
    free(gpu_infos);
    gpu_infos = NULL;
  }
  tpu_chip_count = 0;
}

const char *gpuinfo_tpu_last_error_string(void) { return "Err"; }

static void add_tpu_chip(struct list_head *devices, unsigned *count) {
  struct gpu_info_tpu *this_tpu = &gpu_infos[*count];
  this_tpu->base.vendor = &gpu_vendor_tpu;
  this_tpu->device_idx = *count;
  char pdev_val[PDEV_LEN];
  snprintf(pdev_val, PDEV_LEN - 1, "TPU%u-%s", *count, latest_chips_usage_data[*count].name);
  strncpy(this_tpu->base.pdev, pdev_val, PDEV_LEN - 1);
  list_add_tail(&this_tpu->base.list, devices);

  this_tpu->base.processes_count = 0;
  this_tpu->base.processes = NULL;
  this_tpu->base.processes_array_size = 0;

  *count = *count + 1;
}

bool gpuinfo_tpu_get_device_handles(struct list_head *devices_list, unsigned *count) {
  *count = 0;
  if (tpu_chip_count <= 0) return false;
  pthread_create(&tpu_query_thread, NULL, query_tpu_data_thread_fn, NULL);
  gpu_infos = calloc(tpu_chip_count, sizeof(*gpu_infos));
  if (!gpu_infos)
    return false;
  for (int i = 0; i < tpu_chip_count; i++) {
    add_tpu_chip(devices_list, count);
  }
  return true;
}

void gpuinfo_tpu_populate_static_info(struct gpu_info *_gpu_info) {
  struct gpu_info_tpu *gpu_info = container_of(_gpu_info, struct gpu_info_tpu, base);
  struct gpuinfo_static_info *static_info = &gpu_info->base.static_info;
  static_info->integrated_graphics = false;
  static_info->encode_decode_shared = false;
  RESET_ALL(static_info->valid);
  SET_VALID(gpuinfo_device_name_valid, static_info->valid);
  strncpy(static_info->device_name, gpu_info->base.pdev, sizeof(static_info->device_name));
}

void gpuinfo_tpu_refresh_dynamic_info(struct gpu_info *_gpu_info) {
  struct gpu_info_tpu *gpu_info = container_of(_gpu_info, struct gpu_info_tpu, base);
  // struct gpuinfo_static_info *static_info = &gpu_info->base.static_info; // unused
  struct gpuinfo_dynamic_info *dynamic_info = &gpu_info->base.dynamic_info;

  if (gpu_info->device_idx >= tpu_chip_count) return;

  pthread_mutex_lock(&m);
  struct tpu_chip_usage_data usage_data = latest_chips_usage_data[gpu_info->device_idx];
  pthread_mutex_unlock(&m);

  double mem_util = round(1e2 * (double)(usage_data.memory_usage) / (double)(usage_data.total_memory));
  double tpu_util = round(usage_data.duty_cycle_pct);
  SET_GPUINFO_DYNAMIC(dynamic_info, gpu_util_rate, (int)tpu_util);
  SET_GPUINFO_DYNAMIC(dynamic_info, mem_util_rate, (int)mem_util);
  SET_GPUINFO_DYNAMIC(dynamic_info, total_memory, usage_data.total_memory);
  SET_GPUINFO_DYNAMIC(dynamic_info, used_memory, usage_data.memory_usage);
  SET_GPUINFO_DYNAMIC(dynamic_info, free_memory, usage_data.total_memory - usage_data.memory_usage);
  return;
}

void gpuinfo_tpu_get_running_processes(struct gpu_info *_gpu_info) {
  (void)_gpu_info;
}
