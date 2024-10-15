/*
 *    Copyright 2023 The ChampSim Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ooo_cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <iostream>

#include "cache.h"
#include "champsim.h"
#include "deadlock.h"
#include "instruction.h"
#include "util/span.h"
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include "stats_printer.h"
#include <fmt/ostream.h>
#include <random>

std::chrono::seconds elapsed_time();

#define WINDOW_SIZE 250
#define TRANSIENT_STATS 0
#define CRIT_WINDOW 25000000

long O3_CPU::operate()
{
  long progress{0};

  progress += retire_rob();                    // retire
  progress += complete_inflight_instruction(); // finalize execution
  progress += execute_instruction();           // execute instructions
  progress += schedule_instruction();          // schedule instructions
  progress += handle_memory_return();          // finalize memory transactions
  progress += operate_lsq();                   // execute memory transactions

  progress += dispatch_instruction(); // dispatch
  progress += decode_instruction();   // decode
  progress += promote_to_decode();

  progress += fetch_instruction(); // fetch
  progress += check_dib();
  initialize_instruction();

  // heartbeat
  if (show_heartbeat && (num_retired >= next_print_instruction)) {
    auto heartbeat_instr{std::ceil(num_retired - last_heartbeat_instr)};
    auto heartbeat_cycle{std::ceil(current_cycle - last_heartbeat_cycle)};

    auto phase_instr{std::ceil(num_retired - begin_phase_instr)};
    auto phase_cycle{std::ceil(current_cycle - begin_phase_cycle)};

    fmt::print("Heartbeat CPU {} instructions: {} cycles: {} heartbeat IPC: {:.4g} cumulative IPC: {:.4g} (Simulation time: {:%H hr %M min %S sec})\n", cpu,
               num_retired, current_cycle, heartbeat_instr / heartbeat_cycle, phase_instr / phase_cycle, elapsed_time());
    next_print_instruction += STAT_PRINTING_PERIOD;

    last_heartbeat_instr = num_retired;
    last_heartbeat_cycle = current_cycle;
  }

  return progress;
}

void O3_CPU::initialize()
{
  // BRANCH PREDICTOR & BTB
  impl_initialize_branch_predictor();
  impl_initialize_btb();
}

void O3_CPU::begin_phase()
{
  begin_phase_instr = num_retired;
  begin_phase_cycle = current_cycle;

  // Record where the next phase begins
  stats_type stats;
  stats.name = "CPU " + std::to_string(cpu);
  stats.begin_instrs = num_retired;
  stats.begin_cycles = current_cycle;
  sim_stats = stats;
}

void O3_CPU::end_phase(unsigned finished_cpu)
{
  // Record where the phase ended (overwrite if this is later)
  sim_stats.end_instrs = num_retired;
  sim_stats.end_cycles = current_cycle;

  if (finished_cpu == this->cpu) {
    finish_phase_instr = num_retired;
    finish_phase_cycle = current_cycle;

    roi_stats = sim_stats;
  }
}

void O3_CPU::initialize_instruction()
{
  auto instrs_to_read_this_cycle = std::min(FETCH_WIDTH, static_cast<long>(IFETCH_BUFFER_SIZE - std::size(IFETCH_BUFFER)));

  while (current_cycle >= fetch_resume_cycle && instrs_to_read_this_cycle > 0 && !std::empty(input_queue)) {
    instrs_to_read_this_cycle--;

    auto stop_fetch = do_init_instruction(input_queue.front());
    if (stop_fetch)
      instrs_to_read_this_cycle = 0;

    // Add to IFETCH_BUFFER
    IFETCH_BUFFER.push_back(input_queue.front());
    input_queue.pop_front();

    IFETCH_BUFFER.back().event_cycle = current_cycle;
  }
}

namespace
{
void do_stack_pointer_folding(ooo_model_instr& arch_instr)
{
  // The exact, true value of the stack pointer for any given instruction can usually be determined immediately after the instruction is decoded without
  // waiting for the stack pointer's dependency chain to be resolved.
  bool writes_sp = std::count(std::begin(arch_instr.destination_registers), std::end(arch_instr.destination_registers), champsim::REG_STACK_POINTER);
  if (writes_sp) {
    // Avoid creating register dependencies on the stack pointer for calls, returns, pushes, and pops, but not for variable-sized changes in the
    // stack pointer position. reads_other indicates that the stack pointer is being changed by a variable amount, which can't be determined before
    // execution.
    bool reads_other = std::count_if(std::begin(arch_instr.source_registers), std::end(arch_instr.source_registers), [](uint8_t r) {
      return r != champsim::REG_STACK_POINTER && r != champsim::REG_FLAGS && r != champsim::REG_INSTRUCTION_POINTER;
    });
    if ((arch_instr.is_branch != 0) || !(std::empty(arch_instr.destination_memory) && std::empty(arch_instr.source_memory)) || (!reads_other)) {
      auto nonsp_end = std::remove(std::begin(arch_instr.destination_registers), std::end(arch_instr.destination_registers), champsim::REG_STACK_POINTER);
      arch_instr.destination_registers.erase(nonsp_end, std::end(arch_instr.destination_registers));
    }
  }
}
} // namespace

bool O3_CPU::do_predict_branch(ooo_model_instr& arch_instr)
{
  bool stop_fetch = false;

  // handle branch prediction for all instructions as at this point we do not know if the instruction is a branch
  sim_stats.total_branch_types[arch_instr.branch_type]++;
  auto [predicted_branch_target, always_taken] = impl_btb_prediction(arch_instr.ip);
  arch_instr.branch_prediction = impl_predict_branch(arch_instr.ip) || always_taken;
  if (arch_instr.branch_prediction == 0)
    predicted_branch_target = 0;

  if (arch_instr.is_branch) {
    if constexpr (champsim::debug_print) {
      fmt::print("[BRANCH] instr_id: {} ip: {:#x} taken: {}\n", arch_instr.instr_id, arch_instr.ip, arch_instr.branch_taken);
    }

    // call code prefetcher every time the branch predictor is used
    l1i->impl_prefetcher_branch_operate(arch_instr.ip, arch_instr.branch_type, predicted_branch_target);

    if (predicted_branch_target != arch_instr.branch_target
        || (((arch_instr.branch_type == BRANCH_CONDITIONAL) || (arch_instr.branch_type == BRANCH_OTHER))
            && arch_instr.branch_taken != arch_instr.branch_prediction)) { // conditional branches are re-evaluated at decode when the target is computed
      sim_stats.total_rob_occupancy_at_branch_mispredict += std::size(ROB);
      sim_stats.branch_type_misses[arch_instr.branch_type]++;
      if (!warmup) {
        fetch_resume_cycle = std::numeric_limits<uint64_t>::max();
        stop_fetch = true;
        arch_instr.branch_mispredicted = 1;
      }
    } else {
      stop_fetch = arch_instr.branch_taken; // if correctly predicted taken, then we can't fetch anymore instructions this cycle
    }

    impl_update_btb(arch_instr.ip, arch_instr.branch_target, arch_instr.branch_taken, arch_instr.branch_type);
    impl_last_branch_result(arch_instr.ip, arch_instr.branch_target, arch_instr.branch_taken, arch_instr.branch_type);
  }

  return stop_fetch;
}

bool O3_CPU::do_init_instruction(ooo_model_instr& arch_instr)
{
  // fast warmup eliminates register dependencies between instructions branch predictor, cache contents, and prefetchers are still warmed up
  if (warmup) {
    arch_instr.source_registers.clear();
    arch_instr.destination_registers.clear();
  }

  ::do_stack_pointer_folding(arch_instr);
  return do_predict_branch(arch_instr);
}

long O3_CPU::check_dib()
{
  // scan through IFETCH_BUFFER to find instructions that hit in the decoded instruction buffer
  auto begin = std::find_if(std::begin(IFETCH_BUFFER), std::end(IFETCH_BUFFER), [](const ooo_model_instr& x) { return !x.dib_checked; });
  auto [window_begin, window_end] = champsim::get_span(begin, std::end(IFETCH_BUFFER), FETCH_WIDTH);
  std::for_each(window_begin, window_end, [this](auto& ifetch_entry){ this->do_check_dib(ifetch_entry); });
  return std::distance(window_begin, window_end);
}

void O3_CPU::do_check_dib(ooo_model_instr& instr)
{
  // Check DIB to see if we recently fetched this line
  if (auto dib_result = DIB.check_hit(instr.ip); dib_result) {
    // The cache line is in the L0, so we can mark this as complete
    instr.fetched = COMPLETED;

    // Also mark it as decoded
    instr.decoded = COMPLETED;

    // It can be acted on immediately
    instr.event_cycle = current_cycle;
  }

  instr.dib_checked = COMPLETED;
}

long O3_CPU::fetch_instruction()
{
  long progress{0};

  // Fetch a single cache line
  auto fetch_ready = [](const ooo_model_instr& x) {
    return x.dib_checked == COMPLETED && !x.fetched;
  };

  // Find the chunk of instructions in the block
  auto no_match_ip = [](const auto& lhs, const auto& rhs) {
    return (lhs.ip >> LOG2_BLOCK_SIZE) != (rhs.ip >> LOG2_BLOCK_SIZE);
  };

  auto l1i_req_begin = std::find_if(std::begin(IFETCH_BUFFER), std::end(IFETCH_BUFFER), fetch_ready);
  for (auto to_read = L1I_BANDWIDTH; to_read > 0 && l1i_req_begin != std::end(IFETCH_BUFFER); --to_read) {
    auto l1i_req_end = std::adjacent_find(l1i_req_begin, std::end(IFETCH_BUFFER), no_match_ip);
    if (l1i_req_end != std::end(IFETCH_BUFFER))
      l1i_req_end = std::next(l1i_req_end); // adjacent_find returns the first of the non-equal elements

    // Issue to L1I
    auto success = do_fetch_instruction(l1i_req_begin, l1i_req_end);
    if (success) {
      std::for_each(l1i_req_begin, l1i_req_end, [](auto& x) { x.fetched = INFLIGHT; });
      ++progress;
    }

    l1i_req_begin = std::find_if(l1i_req_end, std::end(IFETCH_BUFFER), fetch_ready);
  }

  return progress;
}

bool O3_CPU::do_fetch_instruction(std::deque<ooo_model_instr>::iterator begin, std::deque<ooo_model_instr>::iterator end)
{
  CacheBus::request_type fetch_packet;
  fetch_packet.v_address = begin->ip;
  fetch_packet.instr_id = begin->instr_id;
  fetch_packet.ip = begin->ip;
  fetch_packet.instr_depend_on_me = {begin, end};

  if constexpr (champsim::debug_print) {
    fmt::print("[IFETCH] {} instr_id: {} ip: {:#x} dependents: {} event_cycle: {}\n", __func__, begin->instr_id, begin->ip,
               std::size(fetch_packet.instr_depend_on_me), begin->event_cycle);
  }

  return L1I_bus.issue_read(fetch_packet);
}

long O3_CPU::promote_to_decode()
{
  auto available_fetch_bandwidth = std::min<long>(FETCH_WIDTH, DECODE_BUFFER_SIZE - std::size(DECODE_BUFFER));
  auto [window_begin, window_end] = champsim::get_span_p(std::begin(IFETCH_BUFFER), std::end(IFETCH_BUFFER), available_fetch_bandwidth,
                                                         [cycle = current_cycle](const auto& x) { return x.fetched == COMPLETED && x.event_cycle <= cycle; });
  long progress{std::distance(window_begin, window_end)};

  std::for_each(window_begin, window_end,
                [cycle = current_cycle, lat = DECODE_LATENCY, warmup = warmup](auto& x) { return x.event_cycle = cycle + ((warmup || x.decoded) ? 0 : lat); });
  std::move(window_begin, window_end, std::back_inserter(DECODE_BUFFER));
  IFETCH_BUFFER.erase(window_begin, window_end);

  return progress;
}

long O3_CPU::decode_instruction()
{
  auto available_decode_bandwidth = std::min<long>(DECODE_WIDTH, DISPATCH_BUFFER_SIZE - std::size(DISPATCH_BUFFER));
  auto [window_begin, window_end] = champsim::get_span_p(std::begin(DECODE_BUFFER), std::end(DECODE_BUFFER), available_decode_bandwidth,
                                                         [cycle = current_cycle](const auto& x) { return x.event_cycle <= cycle; });
  long progress{std::distance(window_begin, window_end)};

  // Send decoded instructions to dispatch
  std::for_each(window_begin, window_end, [&, this](auto& db_entry) {
    // We know instruction is a Load here, and it's ip. Do prediction here 
    if (db_entry.source_memory.size() > 0) {  // Load instruction

      ///  ----------  Miss Prediction  ---------------
      uint64_t idx = db_entry.ip & (table_size - 1);
      bool predict_miss = (miss_pred_table[idx].curr_hits == miss_pred_table[idx].prev_hits);
      if (predict_miss) sim_stats.MP_stats.MP_total_miss_preds++;  // Total miss predictions
      sim_stats.MP_stats.MP_total_preds++;      
      db_entry.preds.predict_miss = predict_miss;

      /// ---------- Critical Prediction ---------------
      //If a load has been frequently critical, mark it critical again
      if (crit_pred_table.count(db_entry.ip) > 0) {   // Check if table has that key first
        db_entry.preds.crit_cycles_prediction = sim_stats.per_load_stat[db_entry.ip].crit_cycles/sim_stats.per_load_stat[db_entry.ip].occ;
        
        if (crit_pred_table[db_entry.ip].crit_count/crit_cnt[CRIT] > 0.04) {
          db_entry.preds.crit_predictions[FREQ] = true;

          for (int i=0; i<CRIT_METRIC_COUNT; i++) {
            if (db_entry.preds.crit_predictions[i]) {
              sim_stats.CR_stats[i].true_preds_num++;
              sim_stats.per_load_stat[db_entry.ip].crit_pred_true[i]++;
            }
          }
        }
      }
    }
    this->do_dib_update(db_entry);

    // Resume fetch
    if (db_entry.branch_mispredicted) {
      // These branches detect the misprediction at decode
      if ((db_entry.branch_type == BRANCH_DIRECT_JUMP) || (db_entry.branch_type == BRANCH_DIRECT_CALL)
          || (((db_entry.branch_type == BRANCH_CONDITIONAL) || (db_entry.branch_type == BRANCH_OTHER)) && db_entry.branch_taken == db_entry.branch_prediction)) {
        // clear the branch_mispredicted bit so we don't attempt to resume fetch again at execute
        db_entry.branch_mispredicted = 0;
        // pay misprediction penalty
        this->fetch_resume_cycle = this->current_cycle + BRANCH_MISPREDICT_PENALTY;
      }
    }

    // Add to dispatch
    db_entry.event_cycle = this->current_cycle + (this->warmup ? 0 : this->DISPATCH_LATENCY);
  });

  std::move(window_begin, window_end, std::back_inserter(DISPATCH_BUFFER));
  DECODE_BUFFER.erase(window_begin, window_end);

  return progress;
}

void O3_CPU::do_dib_update(const ooo_model_instr& instr) { DIB.fill(instr.ip); }

long O3_CPU::dispatch_instruction()
{
  auto available_dispatch_bandwidth = DISPATCH_WIDTH;

  // dispatch DISPATCH_WIDTH instructions into the ROB
  while (available_dispatch_bandwidth > 0 && !std::empty(DISPATCH_BUFFER) && DISPATCH_BUFFER.front().event_cycle < current_cycle && std::size(ROB) != ROB_SIZE
         && ((std::size_t)std::count_if(std::begin(LQ), std::end(LQ), [](const auto& lq_entry) { return !lq_entry.has_value(); })
             >= std::size(DISPATCH_BUFFER.front().source_memory))
         && ((std::size(DISPATCH_BUFFER.front().destination_memory) + std::size(SQ)) <= SQ_SIZE)) {
    ROB.push_back(std::move(DISPATCH_BUFFER.front()));
    DISPATCH_BUFFER.pop_front();
    do_memory_scheduling(ROB.back());

    available_dispatch_bandwidth--;
  }

  return DISPATCH_WIDTH - available_dispatch_bandwidth;
}

long O3_CPU::schedule_instruction()
{
  auto search_bw = SCHEDULER_SIZE;
  int progress{0};
  for (auto rob_it = std::begin(ROB); rob_it != std::end(ROB) && search_bw > 0; ++rob_it) {
    if (rob_it->scheduled == 0) {
      do_scheduling(*rob_it);
      ++progress;
    }

    if (rob_it->executed == 0)
      --search_bw;
  }

  return progress;
}

void O3_CPU::do_scheduling(ooo_model_instr& instr)
{
  // Mark register dependencies
  for (auto src_reg : instr.source_registers) {
    if (!std::empty(reg_producers[src_reg])) {
      ooo_model_instr& prior = reg_producers[src_reg].back();
      if (prior.registers_instrs_depend_on_me.empty() || prior.registers_instrs_depend_on_me.back().get().instr_id != instr.instr_id) {
        prior.registers_instrs_depend_on_me.push_back(instr);
        instr.num_reg_dependent++;
      }
    }
  }

  for (auto dreg : instr.destination_registers) {
    auto begin = std::begin(reg_producers[dreg]);
    auto end = std::end(reg_producers[dreg]);
    auto ins = std::lower_bound(begin, end, instr, [](const ooo_model_instr& lhs, const ooo_model_instr& rhs) { return lhs.instr_id < rhs.instr_id; });
    reg_producers[dreg].insert(ins, std::ref(instr));
  }

  instr.scheduled = COMPLETED;
  instr.event_cycle = current_cycle + (warmup ? 0 : SCHEDULING_LATENCY);
}

long O3_CPU::execute_instruction()
{
  auto exec_bw = EXEC_WIDTH;
  for (auto rob_it = std::begin(ROB); rob_it != std::end(ROB) && exec_bw > 0; ++rob_it) {
    if (rob_it->scheduled == COMPLETED && rob_it->executed == 0 && rob_it->num_reg_dependent == 0 && rob_it->event_cycle <= current_cycle) {
      do_execution(*rob_it);
      --exec_bw;
    }
  }

  return EXEC_WIDTH - exec_bw;
}

void O3_CPU::do_execution(ooo_model_instr& rob_entry)
{
  rob_entry.executed = INFLIGHT;
  rob_entry.event_cycle = current_cycle + (warmup ? 0 : EXEC_LATENCY);

  // Mark LQ entries as ready to translate
  for (auto& lq_entry : LQ) {
    if (lq_entry.has_value() && lq_entry->instr_id == rob_entry.instr_id) {
      lq_entry->event_cycle = current_cycle + (warmup ? 0 : EXEC_LATENCY);
      lq_entry->exec_cycle = current_cycle;
    }
  }
  // Mark SQ entries as ready to translate
  for (auto& sq_entry : SQ)
    if (sq_entry.instr_id == rob_entry.instr_id)
      sq_entry.event_cycle = current_cycle + (warmup ? 0 : EXEC_LATENCY);

  if constexpr (champsim::debug_print) {
    fmt::print("[ROB] {} instr_id: {} event_cycle: {}\n", __func__, rob_entry.instr_id, rob_entry.event_cycle);
  }
}

void O3_CPU::do_memory_scheduling(ooo_model_instr& instr)
{
  // load
  for (auto& smem : instr.source_memory) {
    auto q_entry = std::find_if_not(std::begin(LQ), std::end(LQ), [](const auto& lq_entry) { return lq_entry.has_value(); });
    assert(q_entry != std::end(LQ));
    q_entry->emplace(instr.instr_id, smem, instr.ip, instr.asid); // add it to the load queue

    // Check for forwarding
    auto sq_it = std::max_element(std::begin(SQ), std::end(SQ), [smem](const auto& lhs, const auto& rhs) {
      return lhs.virtual_address != smem || (rhs.virtual_address == smem && lhs.instr_id < rhs.instr_id);
    });
    if (sq_it != std::end(SQ) && sq_it->virtual_address == smem) {
      (*q_entry)->hit_level = "fwd";
      instr.hit_level = "fwd";
      if (sq_it->fetch_issued) { // Store already executed
        q_entry->reset();
        ++instr.completed_mem_ops;

        if constexpr (champsim::debug_print)
          fmt::print("[DISPATCH] {} instr_id: {} forwards_from: {}\n", __func__, instr.instr_id, sq_it->event_cycle);
      } else {
        assert(sq_it->instr_id < instr.instr_id);   // The found SQ entry is a prior store
        sq_it->lq_depend_on_me.push_back(*q_entry); // Forward the load when the store finishes
        (*q_entry)->producer_id = sq_it->instr_id;  // The load waits on the store to finish

        if constexpr (champsim::debug_print)
          fmt::print("[DISPATCH] {} instr_id: {} waits on: {}\n", __func__, instr.instr_id, sq_it->event_cycle);
      }
    }
  }

  // store
  for (auto& dmem : instr.destination_memory)
    SQ.emplace_back(instr.instr_id, dmem, instr.ip, instr.asid); // add it to the store queue

  if constexpr (champsim::debug_print) {
    fmt::print("[DISPATCH] {} instr_id: {} loads: {} stores: {}\n", __func__, instr.instr_id, std::size(instr.source_memory),
               std::size(instr.destination_memory));
  }
}

long O3_CPU::operate_lsq()
{
  auto store_bw = SQ_WIDTH;

  const auto complete_id = std::empty(ROB) ? std::numeric_limits<uint64_t>::max() : ROB.front().instr_id;
  auto do_complete = [cycle = current_cycle, complete_id, this](const auto& x) {
    return x.instr_id < complete_id && x.event_cycle <= cycle && this->do_complete_store(x);
  };

  auto unfetched_begin = std::partition_point(std::begin(SQ), std::end(SQ), [](const auto& x) { return x.fetch_issued; });
  auto [fetch_begin, fetch_end] = champsim::get_span_p(unfetched_begin, std::end(SQ), store_bw,
                                                       [cycle = current_cycle](const auto& x) { return !x.fetch_issued && x.event_cycle <= cycle; });
  store_bw -= std::distance(fetch_begin, fetch_end);
  std::for_each(fetch_begin, fetch_end, [cycle = current_cycle, this](auto& sq_entry) {
    this->do_finish_store(sq_entry);
    sq_entry.fetch_issued = true;
    sq_entry.event_cycle = cycle;
  });

  auto [complete_begin, complete_end] = champsim::get_span_p(std::cbegin(SQ), std::cend(SQ), store_bw, do_complete);
  store_bw -= std::distance(complete_begin, complete_end);
  SQ.erase(complete_begin, complete_end);

  auto load_bw = LQ_WIDTH;

  for (auto& lq_entry : LQ) {
    if (load_bw > 0 && lq_entry.has_value() && lq_entry->producer_id == std::numeric_limits<uint64_t>::max() && !lq_entry->fetch_issued
        && lq_entry->event_cycle < current_cycle) {
      auto success = execute_load(*lq_entry);
      if (success) {
        --load_bw;
        lq_entry->fetch_issued = true;
      }
    }
  }

  return (SQ_WIDTH - store_bw) + (LQ_WIDTH - load_bw);
}

void O3_CPU::do_finish_store(const LSQ_ENTRY& sq_entry)
{
  sq_entry.finish(std::begin(ROB), std::end(ROB));

  // Release dependent loads
  for (std::optional<LSQ_ENTRY>& dependent : sq_entry.lq_depend_on_me) {
    assert(dependent.has_value()); // LQ entry is still allocated
    assert(dependent->producer_id == sq_entry.instr_id);

    dependent->finish(std::begin(ROB), std::end(ROB));
    dependent.reset();
  }
}

bool O3_CPU::do_complete_store(const LSQ_ENTRY& sq_entry)
{
  CacheBus::request_type data_packet;
  data_packet.v_address = sq_entry.virtual_address;
  data_packet.instr_id = sq_entry.instr_id;
  data_packet.ip = sq_entry.ip;

  if constexpr (champsim::debug_print) {
    fmt::print("[SQ] {} instr_id: {} vaddr: {:x}\n", __func__, data_packet.instr_id, data_packet.v_address);
  }

  return L1D_bus.issue_write(data_packet);
}

bool O3_CPU::execute_load(const LSQ_ENTRY& lq_entry)
{
  CacheBus::request_type data_packet;
  data_packet.v_address = lq_entry.virtual_address;
  data_packet.instr_id = lq_entry.instr_id;
  data_packet.ip = lq_entry.ip;

  if constexpr (champsim::debug_print) {
    fmt::print("[LQ] {} instr_id: {} vaddr: {:#x}\n", __func__, data_packet.instr_id, data_packet.v_address);
  }

  return L1D_bus.issue_read(data_packet);
}

void O3_CPU::do_complete_execution(ooo_model_instr& instr)
{
  for (auto dreg : instr.destination_registers) {
    auto begin = std::begin(reg_producers[dreg]);
    auto end = std::end(reg_producers[dreg]);
    auto elem = std::find_if(begin, end, [id = instr.instr_id](ooo_model_instr& x) { return x.instr_id == id; });
    assert(elem != end);
    reg_producers[dreg].erase(elem);
  }

  instr.executed = COMPLETED;

  for (ooo_model_instr& dependent : instr.registers_instrs_depend_on_me) {
    dependent.num_reg_dependent--;
    assert(dependent.num_reg_dependent >= 0);

    if (dependent.num_reg_dependent == 0)
      dependent.scheduled = COMPLETED;
  }

  if (instr.branch_mispredicted)
    fetch_resume_cycle = current_cycle + BRANCH_MISPREDICT_PENALTY;
}

long O3_CPU::complete_inflight_instruction()
{
  // update ROB entries with completed executions
  auto complete_bw = EXEC_WIDTH;
  for (auto rob_it = std::begin(ROB); rob_it != std::end(ROB) && complete_bw > 0; ++rob_it) {
    if ((rob_it->executed == INFLIGHT) && (rob_it->event_cycle <= current_cycle) && rob_it->completed_mem_ops == rob_it->num_mem_ops()) {
      do_complete_execution(*rob_it);
      --complete_bw;
    }
  }

  return EXEC_WIDTH - complete_bw;
}

long O3_CPU::handle_memory_return()
{
  long progress{0};

  for (auto l1i_bw = FETCH_WIDTH, to_read = L1I_BANDWIDTH; l1i_bw > 0 && to_read > 0 && !L1I_bus.lower_level->returned.empty(); --to_read) {
    auto& l1i_entry = L1I_bus.lower_level->returned.front();

    while (l1i_bw > 0 && !l1i_entry.instr_depend_on_me.empty()) {
      ooo_model_instr& fetched = l1i_entry.instr_depend_on_me.front();
      if ((fetched.ip >> LOG2_BLOCK_SIZE) == (l1i_entry.v_address >> LOG2_BLOCK_SIZE) && fetched.fetched != 0) {
        fetched.fetched = COMPLETED;
        --l1i_bw;
        ++progress;

        if constexpr (champsim::debug_print) {
          fmt::print("[IFETCH] {} instr_id: {} fetch completed\n", __func__, fetched.instr_id);
        }
      }

      l1i_entry.instr_depend_on_me.erase(std::begin(l1i_entry.instr_depend_on_me));
    }

    // remove this entry if we have serviced all of its instructions
    if (l1i_entry.instr_depend_on_me.empty()) {
      L1I_bus.lower_level->returned.pop_front();
      ++progress;
    }
  }

  auto l1d_it = std::begin(L1D_bus.lower_level->returned);
  for (auto l1d_bw = L1D_BANDWIDTH; l1d_bw > 0 && l1d_it != std::end(L1D_bus.lower_level->returned); --l1d_bw, ++l1d_it) {
    for (auto& lq_entry : LQ) {
      if (lq_entry.has_value() && lq_entry->fetch_issued && lq_entry->virtual_address >> LOG2_BLOCK_SIZE == l1d_it->v_address >> LOG2_BLOCK_SIZE) {
        // Add hit level to lq_entry
        lq_entry->hit_level = l1d_it->hit_level;
        lq_entry->ret_cycle = current_cycle;
        lq_entry->finish(std::begin(ROB), std::end(ROB));
        lq_entry.reset();
        ++progress;
      }
    }
    ++progress;
  }
  L1D_bus.lower_level->returned.erase(std::begin(L1D_bus.lower_level->returned), l1d_it);

  return progress;
}

long O3_CPU::retire_rob()
{
  auto [retire_begin, retire_end] = champsim::get_span_p(std::cbegin(ROB), std::cend(ROB), RETIRE_WIDTH, [](const auto& x) { return x.executed == COMPLETED; });
  if constexpr (champsim::debug_print) {
    std::for_each(retire_begin, retire_end, [](const auto& x) { fmt::print("[ROB] retire_rob instr_id: {} is retired\n", x.instr_id); });
  }
  auto retire_count = std::distance(retire_begin, retire_end);
  num_retired += retire_count;
  
  // Check instruction at the head of the ROB if it is critical
  auto rob_head = std::begin(ROB);
  rob_head->critical_cycles++; // # Cycles this instr has blocked head, if not blocked value is 0/1

  for (auto commit_inst=retire_begin; commit_inst < retire_end; commit_inst++) {
    auto retire_inst = *commit_inst;
    uint64_t idx = retire_inst.ip & (table_size - 1);

    // ------------ Load specific ------------ 
    if (retire_inst.source_memory.size()>0) {
      LOAD_OUTCOME load_outcome;

      // Critical Instruction
      if (retire_inst.critical_cycles > 1) {
        if (retire_inst.hit_level == "DRAM") {  // Critical miss
          load_outcome = CRIT;
          for (int i=0; i<CRIT_METRIC_COUNT; i++) {
            if (retire_inst.preds.crit_predictions[i]) {
              sim_stats.CR_stats[i].correct_preds_num++;
            }
          }
          crit_pred_table[retire_inst.ip].crit_count++;
        } else {  // Critical hit
          load_outcome = CRIT_HIT;
        }
      } 
      else {  // Non critical
        load_outcome = NON_CRIT;
      }

      crit_cnt[load_outcome]++;
      sim_stats.per_load_stat[retire_inst.ip].cnt[load_outcome]++;
      sim_stats.crit_stats[load_outcome] += retire_inst;
      sim_stats.crit_stats[load_outcome].cnt++; 

      crit_pred_table[retire_inst.ip].crit_cycles += retire_inst.critical_cycles;
      crit_pred_table[retire_inst.ip].occ++;
      sim_stats.per_load_stat[retire_inst.ip].occ++;
      if (retire_inst.critical_cycles > retire_inst.preds.crit_cycles_prediction) sim_stats.per_load_stat[retire_inst.ip].crit_cycle_pred_correct++;

      // Miss Pred: Update and Eval
      sim_stats.total_loads++;
      if (retire_inst.hit_level == "DRAM") {   
        miss_pred_table[idx].prev_hits = miss_pred_table[idx].curr_hits;
        miss_pred_table[idx].curr_hits = 0;
        sim_stats.MP_stats.total_offchip++;
        if (retire_inst.preds.predict_miss) {   // Predicted offchip
          sim_stats.MP_stats.correct_offchip++;  // TP
          sim_stats.MP_stats.correct_predictions++;
        }
      } else {
        miss_pred_table[idx].curr_hits++;
        if (!retire_inst.preds.predict_miss) {
          sim_stats.MP_stats.correct_predictions++;
        }
      }
    }
    // ~~~~~~~~~~~~~~ Load specific ~~~~~~~~~~~~~~

    uint64_t llsr_size = 512;
    
    // Predict
    uint64_t pred_dist = mlp_pred_table[idx];
    llsr_entry entry;
    entry = retire_inst;
    entry.pred_dist = pred_dist;

    // Update llsr
    if (llsr.size() == llsr_size) {   
      auto llsr_head = llsr.front();
      
      if (llsr_head.critical_load && llsr_head.long_latency == 1) {  
        bool min_calculated = false;
        bool min_cross_calc = false;
        int i = 0;

        // Crit related metrics
        for (auto llsr_entry_it = std::next(llsr.begin()); llsr_entry_it != llsr.end(); llsr_entry_it++) {
          i++;
          if (llsr_entry_it->ip == llsr_head.ip) {
            if (!min_calculated) {
              sim_stats.per_load_stat[llsr_head.ip].min_dist += i;
              min_calculated = true;
              if (llsr_entry_it->long_latency && (llsr_head.ret_cycle > llsr_entry_it->exec_cycle)) sim_stats.per_load_stat[llsr_head.ip].perc_self_mlp++;
            }
            sim_stats.per_load_stat[llsr_head.ip].density++;
          }
          if (!min_cross_calc) {
            if (llsr_entry_it->long_latency) {
              sim_stats.per_load_stat[llsr_head.ip].min_dist_cross += i;
              min_cross_calc = true;
              if (llsr_head.ret_cycle > llsr_entry_it->exec_cycle) sim_stats.per_load_stat[llsr_head.ip].perc_cross_mlp++;
            }
          }
        }; 
        if (min_calculated) sim_stats.per_load_stat[llsr_head.ip].dupli_occ++;
      }
      
      // Update mlp table
      if (llsr_head.long_latency == 1) {  
        // Get MLP distance
        uint64_t mlp_dist = llsr_size - 1;
        for (auto llsr_it=std::end(llsr)-1; llsr_it > std::begin(llsr); llsr_it--) {
          if ((*llsr_it).long_latency == 1) { 
            if (llsr_head.ret_cycle >= (*llsr_it).exec_cycle) {
              sim_stats.MLP_stats.mlp_perc_overlapped++;
            } 
            break;
          }
          mlp_dist--;
        }
        sim_stats.MLP_stats.mlp_max_dist += mlp_dist;

        // Get MLP distance with cycles
        uint64_t mlp_dist_cycles = llsr_size - 1;
        for (auto llsr_it=std::end(llsr)-1; llsr_it > std::begin(llsr); llsr_it--) {
          if ((*llsr_it).long_latency == 1) { 
            if (llsr_head.ret_cycle >= (*llsr_it).exec_cycle) {
              break;
            } 
          }
          mlp_dist_cycles--;
        }
        sim_stats.MLP_stats.mlp_max_dist_cycles += mlp_dist_cycles;

        // Get MLP Min dist and Density
        bool first = true;
        uint64_t mlp_curr_density = 0;
        for (auto llsr_entry_it = llsr.begin() + 1; llsr_entry_it != llsr.end(); llsr_entry_it++) {  
          if ((*llsr_entry_it).long_latency == 1) {   // Long latency MLP
            sim_stats.MLP_stats.mlp_density++;
            if (first) { 
              sim_stats.MLP_stats.mlp_min_dist += std::distance(llsr.begin(), llsr_entry_it);
              first = false;
            }

            if (llsr_head.ret_cycle >= (*llsr_entry_it).exec_cycle) {   // Cycle overlap MLP
              sim_stats.MLP_stats.mlp_density_cycles++;
              mlp_curr_density++;
            }
          }
        }

        // Get MLP error
        sim_stats.MLP_stats.mlp_error += (abs(mlp_dist - llsr_head.pred_dist));

        // Put it in the MLP table
        uint64_t idx_mlp_table = llsr_head.ip & (table_size - 1);
        mlp_pred_table[idx_mlp_table] = mlp_dist;

        // Evaluate here: Resolution of mlp_dist happens here
        sim_stats.MLP_stats.mlp_total++;
        if ((bool) mlp_dist) {  // There is actual MLP
          if ((bool) llsr_head.pred_dist) { sim_stats.MLP_stats.mlp_TP++;}
          else { sim_stats.MLP_stats.mlp_FN++;}
        } else {  // No actual MLP
          if ((bool) llsr_head.pred_dist) { sim_stats.MLP_stats.mlp_FP++;}
          else { sim_stats.MLP_stats.mlp_TN++;}
        }
        if (llsr_head.pred_dist >= mlp_dist) { sim_stats.MLP_stats.mlp_dist_acc++;}
      }
      
      llsr.pop_front();
      llsr.push_back(entry); 
    } else {
      llsr.push_back(entry);    // Only push initially
    } 

    assert(llsr.size() <= llsr_size);
  }

  ROB.erase(retire_begin, retire_end);

  return retire_count;
}

// LCOV_EXCL_START Exclude the following function from LCOV
void O3_CPU::print_deadlock()
{
  fmt::print("DEADLOCK! CPU {} cycle {}\n", cpu, current_cycle);

  auto instr_pack = [](const auto& entry) {
    return std::tuple{entry.instr_id, +entry.fetched, +entry.scheduled, +entry.executed, +entry.num_reg_dependent, entry.num_mem_ops() - entry.completed_mem_ops, entry.event_cycle};
  };
  std::string_view instr_fmt{"instr_id: {} fetched: {} scheduled: {} executed: {} num_reg_dependent: {} num_mem_ops: {} event: {}"};
  champsim::range_print_deadlock(IFETCH_BUFFER, "cpu" + std::to_string(cpu) + "_IFETCH", instr_fmt, instr_pack);
  champsim::range_print_deadlock(DECODE_BUFFER, "cpu" + std::to_string(cpu) + "_DECODE", instr_fmt, instr_pack);
  champsim::range_print_deadlock(DISPATCH_BUFFER, "cpu" + std::to_string(cpu) + "_DISPATCH", instr_fmt, instr_pack);
  champsim::range_print_deadlock(ROB, "cpu" + std::to_string(cpu) + "_ROB", instr_fmt, instr_pack);

  // print LSQ entries
  auto lq_pack = [](const auto& entry) {
    std::string depend_id{"-"};
    if (entry->producer_id != std::numeric_limits<uint64_t>::max()) {
      depend_id = std::to_string(entry->producer_id);
    }
    return std::tuple{entry->instr_id, entry->virtual_address, entry->fetch_issued, entry->event_cycle, depend_id};
  };
  std::string_view lq_fmt{"instr_id: {} address: {:#x} fetch_issued: {} event_cycle: {} waits on {}"};

  auto sq_pack = [](const auto& entry) {
    std::vector<uint64_t> depend_ids;
    std::transform(std::begin(entry.lq_depend_on_me), std::end(entry.lq_depend_on_me), std::back_inserter(depend_ids),
        [](const std::optional<LSQ_ENTRY>& lq_entry) { return lq_entry->producer_id; });
    return std::tuple{entry.instr_id, entry.virtual_address, entry.fetch_issued, entry.event_cycle, depend_ids};
  };
  std::string_view sq_fmt{"instr_id: {} address: {:#x} fetch_issued: {} event_cycle: {} LQ waiting: {}"};
  champsim::range_print_deadlock(LQ, "cpu" + std::to_string(cpu) + "_LQ", lq_fmt, lq_pack);
  champsim::range_print_deadlock(SQ, "cpu" + std::to_string(cpu) + "_SQ", sq_fmt, sq_pack);
}
// LCOV_EXCL_STOP

LSQ_ENTRY::LSQ_ENTRY(uint64_t id, uint64_t addr, uint64_t local_ip, std::array<uint8_t, 2> local_asid)
    : instr_id(id), virtual_address(addr), ip(local_ip), asid(local_asid)
{
}

void LSQ_ENTRY::finish(std::deque<ooo_model_instr>::iterator begin, std::deque<ooo_model_instr>::iterator end) const
{
  auto rob_entry = std::partition_point(begin, end, [id = this->instr_id](auto x) { return x.instr_id < id; });
  assert(rob_entry != end);
  assert(rob_entry->instr_id == this->instr_id);
  
  rob_entry->hit_level = this->hit_level;
  rob_entry->exec_cycle = this->exec_cycle;
  rob_entry->ret_cycle = this->ret_cycle;

  ++rob_entry->completed_mem_ops;
  assert(rob_entry->completed_mem_ops <= rob_entry->num_mem_ops());

  if constexpr (champsim::debug_print) {
    fmt::print("[LSQ] {} instr_id: {} full_address: {:#x} remain_mem_ops: {} event_cycle: {}\n", __func__, instr_id, virtual_address,
               rob_entry->num_mem_ops() - rob_entry->completed_mem_ops, event_cycle);
  }
}

bool CacheBus::issue_read(request_type data_packet)
{
  data_packet.address = data_packet.v_address;
  data_packet.is_translated = false;
  data_packet.cpu = cpu;
  data_packet.type = access_type::LOAD;

  return lower_level->add_rq(data_packet);
}

bool CacheBus::issue_write(request_type data_packet)
{
  data_packet.address = data_packet.v_address;
  data_packet.is_translated = false;
  data_packet.cpu = cpu;
  data_packet.type = access_type::WRITE;
  data_packet.response_requested = false;

  return lower_level->add_wq(data_packet);
}
