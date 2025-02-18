#ifndef EXTRACT_GPUINFO_TPU_H_
#define EXTRACT_GPUINFO_TPU_H_

#include <stdint.h>

struct gpu_info_tpu {
  struct gpu_info base;
  int device_id;
};

struct tpu_chip_usage_data {
  char name[8];
  int64_t device_id;
  int64_t memory_usage;
  int64_t total_memory;
  double duty_cycle_pct;
};

#endif // EXTRACT_GPUINFO_TPU_H_