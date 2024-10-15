#include <string>
#include <map>
#include <set>
#include <iostream>
#include <algorithm>
#include "instruction.h"

enum LOAD_OUTCOME { CRIT=1, NON_CRIT=2, CRIT_HIT=3, CRIT_TYPE_COUNT=4 };

// Miss Pred Structs
struct MISS_PRED_ENTRY {
  uint64_t prev_hits = 0;
  uint64_t curr_hits = 0;
  uint64_t last_ip = 0;
  uint64_t last_cycle = 0;
};

struct MISS_PRED_STATS {
  double MP_total_miss_preds = 0;
  double MP_total_preds = 0;
  double correct_offchip = 0;
  double total_offchip = 0;

  double correct_predictions = 0;
  double total_predictions = 0;
  double total_loads = 0;
};


// Criticality Predictor Structs
class CRITICALITY_STATS {
  public:
    double cnt = 0;
    double crit_cycles = 0;   
    double crit_dist = 0;
    double crit_density = 0;

    CRITICALITY_STATS& operator+=(ooo_model_instr const& instr) {
      crit_cycles += instr.critical_cycles;
      return *this;
    }
};

class CR_STATS {
  public:
  double true_preds_num = 0;
  double correct_preds_num = 0;   // Correct true predictions only
}; 

class per_load_stats {
  public:
  double long_latency_occ = 0;
  double perc_cross_mlp = 0;
  double dupli_occ = 0;
  double perc_self_mlp = 0;
  double min_dist_cross = 0;
  double min_dist = 0;
  double density = 0;
  double crit_cycle_pred_correct = 0;
  double occ = 0;
  double cnt [CRIT_TYPE_COUNT] = {0}; 
  double crit_cycles = 0;
  double crit_pred_true[CRIT_METRIC_COUNT] = {0};
};


// MLP Predictor Structs
class MLP_STATS {
  public:
    double mlp_min_dist = 0;
    double mlp_max_dist = 0;
    double mlp_density = 0;
    double mlp_max_dist_cycles = 0;
    double mlp_density_cycles = 0;
    double mlp_perc_overlapped = 0;
    double mlp_error = 0;
    double mlp_total = 0;
    double mlp_TP = 0;
    double mlp_TN = 0;
    double mlp_FP = 0;
    double mlp_FN = 0;
    double mlp_dist_acc = 0;
};


class llsr_entry {
    public:
    uint64_t exec_idx = 0;
    uint64_t ip = 0;
    uint64_t long_latency = 0;
    uint64_t pred_dist = 0;
    uint64_t exec_cycle = 0;
    uint64_t ret_cycle = 0;
    bool critical_load = false;
    uint64_t critical_cycles = 0;

    llsr_entry& operator=(ooo_model_instr const& instr) {
        ip = instr.ip;
        long_latency = (instr.hit_level == "DRAM");
        exec_cycle = instr.exec_cycle;
        ret_cycle = instr.ret_cycle;
        critical_cycles = instr.critical_cycles;
        critical_load = instr.critical_load;
        return *this;
    }
};

struct crit_pred_entry {
    double crit_count = 0;
    double crit_count_reset = 0;
    double occ = 0;
    double crit_cycles = 0;
};
