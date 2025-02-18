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
#include <glob.h>
#include <errno.h>

static bool gpuinfo_tpu_init(void);
static void gpuinfo_tpu_shutdown(void);
static const char *gpuinfo_tpu_last_error_string(void);
static bool gpuinfo_tpu_get_device_handles(struct list_head *devices, unsigned *count);
static void gpuinfo_tpu_populate_static_info(struct gpu_info *_gpu_info);
static void gpuinfo_tpu_refresh_dynamic_info(struct gpu_info *_gpu_info);
static void gpuinfo_tpu_get_running_processes(struct gpu_info *_gpu_info);
static void* query_tpu_data_thread_fn(void* _);
static void setup_populate_tpu_data(bool);
static bool populate_tpu_data(bool verbose);
static int64_t millis(void);
static int discover_tpu_devices_num(void);
static void reset_tpu_statistics(bool);

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

#define MAX(x, y) ((x >= y) ? (x) : (y))
#define MIN(x, y) ((x <= y) ? (x) : (y))

__attribute__((constructor)) static void init_extract_gpuinfo_tpu(void) { register_gpu_vendor(&gpu_vendor_tpu); }

pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
pthread_t tpu_query_thread = 0;
bool thread_should_exit = false;

int64_t millis(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return 1000000LL * tv.tv_sec + tv.tv_usec / 1000LL;
}

struct tpu_chip_usage_data *latest_chips_usage_data = NULL;

char python_script[] = 
    "try:\n"
    "  from tpu_info import device, metrics\n"
    "except:\n"
    "  print(\"tpu_info missing\", flush=True)\n"
    "try:\n"
    "  chip_type, count = device.get_local_chips()\n"
    "  chips_usage = metrics.get_chip_usage(chip_type)\n"
    "  for chip_usage in chips_usage:\n"
    "    print(f\"{chip_usage.device_id:d} {chip_usage.memory_usage:d}"
    " {chip_usage.total_memory:d} {chip_usage.duty_cycle_pct:.4f}"
    " {chip_type.value.name}\", flush=True)\n"
    "except:\n"
    "  pass\n";
char *popen_command = NULL;

#define CMD_MAX_LEN 2048

int discover_tpu_devices_num(void) {
  glob_t glob_result;
  const char *pattern = "/dev/{vfio,accel}/[0-9]*";
  int ret = glob(pattern, GLOB_BRACE, NULL, &glob_result);
  if (ret == 0) return glob_result.gl_pathc;
  globfree(&glob_result);
  return 0;
}

void setup_populate_tpu_data(bool avoid_py_compile) {
  bool file_written = false;
  popen_command = (char*)malloc(CMD_MAX_LEN);
  // 1. attempt to create a temporary file to store the python script
  //    this helps with repeated invokations since we're using
  //    $ python3 -m py_compile {source_file.py}
  // 2. if that fails, call the source code via python3 -c '...'
  if (!avoid_py_compile) {
    char tmpfile_template[] = "/tmp/query_tpu_data.py.XXXXXX";
    int fd = mkstemp(tmpfile_template);
    if (fd != -1) {
  #ifdef DEBUG
      printf("Created temporary python script file %s\n", tmpfile_template);
  #endif
      int success = write(fd, python_script, strlen(python_script));
      if (success > 0) {
        snprintf(popen_command, CMD_MAX_LEN, "python3 -m py_compile %s", tmpfile_template);
        file_written = true;
      }
      close(fd);
    }
  }
  if (!file_written) {
#ifdef DEBUG
    printf("Failed to create a temporary python script file, falling back\n");
#endif
    snprintf(popen_command, CMD_MAX_LEN, "python3 -c '%s'", python_script);
  }
#ifdef DEBUG
  printf("popen_command = %s\n", popen_command);
#endif
}

bool populate_tpu_data(bool verbose) {
  if (tpu_chip_count <= 0) return false;
  if (popen_command == NULL) setup_populate_tpu_data(false);

#ifdef DEBUG
  int64_t t = millis();
#endif

  char line[2048];
  int id = 0; 

  FILE* p = popen(popen_command, "r");
  while (fgets(line, sizeof(line), p) != NULL) {
    if (tpu_chip_count == 0 || id >= tpu_chip_count) break;

    // strip newline
    line[sizeof(line) - 1] = '\0';
    int line_len = MIN(MAX(0, strcspn(line, "\n")), sizeof(line) - 1);
    line[line_len] = '\0';

    // check if the script is printing "tpu_info_missing"
    if (id == 0 && strcmp(line, "tpu_info missing") == 0) {
      fprintf(stderr, "tpu_info is not installed\n");
      thread_should_exit = true;
      tpu_chip_count = 0;
      break;
    }

    // parse a data line
    struct tpu_chip_usage_data usage_data;
    if (sscanf(line, "%ld %ld %ld %lf %7[^,]", 
      &usage_data.device_id, &usage_data.memory_usage, &usage_data.total_memory, 
      &usage_data.duty_cycle_pct, (char*)&usage_data.name) == 5) {
        usage_data.name[sizeof(usage_data.name) - 1] = '\0';
        pthread_mutex_lock(&m);
        latest_chips_usage_data[id] = usage_data;
        pthread_mutex_unlock(&m);
    } else {
      fprintf(stderr, "Error parsing TPU output line: %s\n", line);
    }
    id += 1;
  }
  pclose(p);

  if (tpu_chip_count > 0 && id == 0) {
    // py_compile way of running the script is failing, fall back to raw python
    if (popen_command != NULL) free(popen_command);
    setup_populate_tpu_data(true);
  }

  // printing timing information about data query
#ifdef DEBUG
  t = millis() - t;
  printf("Populated TPU data in %ld ms\n", t);
  printf("Found data for %d TPU chips\n", id);
#endif
  if (verbose) printf("Found %ld TPU chips\n", tpu_chip_count);
  return id == tpu_chip_count;
}

void* query_tpu_data_thread_fn(void* _){
  (void)_;
  int64_t t = millis();
  int fails_in_a_row = 0;
  while(!thread_should_exit) {
    bool success = populate_tpu_data(false);
    fails_in_a_row = success ? 0 : MIN(fails_in_a_row + 1, 10);
    if (fails_in_a_row >= 2) reset_tpu_statistics(false);
    do {
      usleep(10 * 1000);
    } while (!thread_should_exit && millis() - t < 1000);
    t = millis();
  }
  return NULL;
}

void reset_tpu_statistics(bool fully) {
  pthread_mutex_lock(&m);
  for (int i = 0; i < tpu_chip_count; i++) {
    latest_chips_usage_data[i].memory_usage = 0;
    latest_chips_usage_data[i].duty_cycle_pct = 0;
    if (fully) {
      snprintf(latest_chips_usage_data[i].name, sizeof(latest_chips_usage_data[i].name), "%s", "N/A");
      latest_chips_usage_data[i].device_id = 0;
      latest_chips_usage_data[i].total_memory = 0;
    }
  }
  pthread_mutex_unlock(&m);
}

bool gpuinfo_tpu_init(void) { 
  //populate_tpu_data(true);  // this also discovers the number of TPU chips
  tpu_chip_count = discover_tpu_devices_num();
  if (tpu_chip_count == 0) {
    printf("Found 0 TPU devices in /dev/{accel,vfio}/*\n");
    return false;
  }
  latest_chips_usage_data = (struct tpu_chip_usage_data*)malloc(tpu_chip_count*sizeof(struct tpu_chip_usage_data));
  reset_tpu_statistics(true);
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
  if (popen_command != NULL) {
    free(popen_command);
    popen_command = NULL;
  }
  tpu_chip_count = -1;
}

const char *gpuinfo_tpu_last_error_string(void) { return "Err"; }

static void add_tpu_chip(struct list_head *devices, unsigned *count) {
  struct gpu_info_tpu *this_tpu = &gpu_infos[*count];
  this_tpu->base.vendor = &gpu_vendor_tpu;
  this_tpu->device_id = *count;
  snprintf(this_tpu->base.pdev, PDEV_LEN, "TPU%u", *count);
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
  if (!gpu_infos) return false;
  for (int i = 0; i < tpu_chip_count; i++) add_tpu_chip(devices_list, count);
  return true;
}

void gpuinfo_tpu_populate_static_info(struct gpu_info *_gpu_info) {
  struct gpu_info_tpu *gpu_info = container_of(_gpu_info, struct gpu_info_tpu, base);
  struct gpuinfo_static_info *static_info = &gpu_info->base.static_info;
  static_info->integrated_graphics = false;
  static_info->encode_decode_shared = false;
  RESET_ALL(static_info->valid);
  snprintf(static_info->device_name, MIN(sizeof(static_info->device_name), PDEV_LEN), "%s", gpu_info->base.pdev);
  SET_VALID(gpuinfo_device_name_valid, static_info->valid);
}

void gpuinfo_tpu_refresh_dynamic_info(struct gpu_info *_gpu_info) {
  struct gpu_info_tpu *gpu_info = container_of(_gpu_info, struct gpu_info_tpu, base);
  struct gpuinfo_static_info *static_info = &gpu_info->base.static_info; // unused
  struct gpuinfo_dynamic_info *dynamic_info = &gpu_info->base.dynamic_info;

  if (gpu_info->device_id >= tpu_chip_count) return;

  pthread_mutex_lock(&m);
  struct tpu_chip_usage_data usage_data = latest_chips_usage_data[gpu_info->device_id];
  pthread_mutex_unlock(&m);

  double mem_util = round(1e2 * (double)(usage_data.memory_usage) / (double)MAX(1, usage_data.total_memory));
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
