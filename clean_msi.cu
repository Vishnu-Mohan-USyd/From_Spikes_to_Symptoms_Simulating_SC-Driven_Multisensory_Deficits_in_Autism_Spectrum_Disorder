#include "clean_msi.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <climits>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace clean_msi {
namespace {

constexpr float kTwoPowMinus32 = 2.3283064365386962890625e-10f;
constexpr std::uint32_t kPhiloxM0 = 0xD2511F53u;
constexpr std::uint32_t kPhiloxM1 = 0xCD9E8D57u;
constexpr std::uint32_t kPhiloxW0 = 0x9E3779B9u;
constexpr std::uint32_t kPhiloxW1 = 0xBB67AE85u;
constexpr int kAssayThreads = 256;
constexpr int kMaximumCrossings = 64;

inline void check_cuda(cudaError_t status, const char* expression,
                       const char* file, int line) {
  if (status == cudaSuccess) {
    return;
  }
  std::ostringstream message;
  message << file << ':' << line << ": " << expression
          << " failed: " << cudaGetErrorString(status);
  throw std::runtime_error(message.str());
}

#define CLEAN_MSI_CUDA(expression) \
  ::clean_msi::check_cuda((expression), #expression, __FILE__, __LINE__)

struct PhiloxWords {
  std::uint32_t x;
  std::uint32_t y;
  std::uint32_t z;
  std::uint32_t w;
};

__host__ __device__ inline std::uint32_t multiply_high(
    std::uint32_t lhs, std::uint32_t rhs) {
#if defined(__CUDA_ARCH__)
  return __umulhi(lhs, rhs);
#else
  return static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(lhs) * rhs) >> 32);
#endif
}

__host__ __device__ inline PhiloxWords philox_round(
    PhiloxWords counter, std::uint32_t key0, std::uint32_t key1) {
  const std::uint32_t high0 = multiply_high(kPhiloxM0, counter.x);
  const std::uint32_t low0 = kPhiloxM0 * counter.x;
  const std::uint32_t high1 = multiply_high(kPhiloxM1, counter.z);
  const std::uint32_t low1 = kPhiloxM1 * counter.z;
  return {high1 ^ counter.y ^ key0, low1,
          high0 ^ counter.w ^ key1, low0};
}

__host__ __device__ inline PhiloxWords philox10(
    PhiloxWords counter, std::uint32_t key0, std::uint32_t key1) {
#pragma unroll
  for (int round = 0; round < 10; ++round) {
    counter = philox_round(counter, key0, key1);
    key0 += kPhiloxW0;
    key1 += kPhiloxW1;
  }
  return counter;
}

__device__ inline float uniform01(std::uint64_t seed, std::uint32_t stream,
                                  std::uint64_t entity,
                                  std::uint32_t time,
                                  std::uint32_t draw) {
  const PhiloxWords words = philox10(
      {static_cast<std::uint32_t>(entity),
       static_cast<std::uint32_t>(entity >> 32), time, draw},
      static_cast<std::uint32_t>(seed) ^ stream,
      static_cast<std::uint32_t>(seed >> 32) + 0xA511E9B3u * stream);
  return (static_cast<float>(words.x) + 0.5f) * kTwoPowMinus32;
}

__device__ inline int poisson_small(float lambda, std::uint64_t seed,
                                    std::uint32_t stream,
                                    std::uint64_t entity,
                                    std::uint32_t time) {
  if (!(lambda > 0.0f)) {
    return 0;
  }
  const float stopping = expf(-lambda);
  float product = 1.0f;
  int count = 0;
  do {
    product *= uniform01(seed, stream, entity, time,
                         static_cast<std::uint32_t>(count));
    ++count;
  } while (product > stopping && count < 64);
  return count - 1;
}

__host__ __device__ inline float clamp_float(float value, float lower,
                                             float upper) {
  return value < lower ? lower : (value > upper ? upper : value);
}

__device__ inline float nmda_voltage_block_device(float voltage_mv) {
  const float exponent = clamp_float(-0.062f * voltage_mv, -80.0f, 80.0f);
  return 1.0f / (1.0f + expf(exponent) / 3.57f);
}

__device__ inline float receptor_advance(const ReceptorKernel& kernel,
                                         float arrivals,
                                         ReceptorState* state) {
  state->rise = state->rise * kernel.rise_decay + arrivals;
  state->decay = state->decay * kernel.decay_decay + arrivals;
  return fmaxf(kernel.normalization * (state->decay - state->rise), 0.0f);
}

struct Derivative {
  float voltage;
  float recovery;
};

__device__ inline float stage_current(float voltage,
                                      const SolverInput& input) {
  return input.g_ampa * (input.excitatory_reversal_mv - voltage) +
         input.g_nmda * nmda_voltage_block_device(voltage) *
             (input.excitatory_reversal_mv - voltage) +
         input.g_gabaa * (input.gabaa_reversal_mv - voltage) +
         input.additive_current;
}

__device__ inline Derivative derivatives(float voltage, float recovery,
                                         const SolverInput& input) {
  const NeuronParameters& p = input.parameters;
  return {0.04f * voltage * voltage + 5.0f * voltage + 140.0f -
              recovery + stage_current(voltage, input),
          p.a * (p.b * voltage - recovery)};
}

__device__ inline NeuronState safe_heun(float voltage, float recovery,
                                        float interval,
                                        const SolverInput& input) {
  const Derivative k1 = derivatives(voltage, recovery, input);
  const float predicted_voltage = voltage + interval * k1.voltage;
  const float predicted_recovery = recovery + interval * k1.recovery;
  const float bounded_predictor =
      fminf(predicted_voltage, input.threshold_mv);
  const Derivative k2 =
      derivatives(bounded_predictor, predicted_recovery, input);
  return {voltage + 0.5f * interval * (k1.voltage + k2.voltage),
          recovery + 0.5f * interval * (k1.recovery + k2.recovery)};
}

__device__ inline bool crossing_occurs(float voltage, float recovery,
                                       float interval,
                                       const SolverInput& input) {
  const Derivative k1 = derivatives(voltage, recovery, input);
  const float predicted_voltage = voltage + interval * k1.voltage;
  const float predicted_recovery = recovery + interval * k1.recovery;
  const float bounded_predictor =
      fminf(predicted_voltage, input.threshold_mv);
  const Derivative k2 =
      derivatives(bounded_predictor, predicted_recovery, input);
  const float end_voltage =
      voltage + 0.5f * interval * (k1.voltage + k2.voltage);
  return predicted_voltage >= input.threshold_mv ||
         end_voltage >= input.threshold_mv;
}

__device__ SolverOutput event_resolved_step(const SolverInput& input) {
  SolverOutput output{};
  float voltage = input.state.voltage_mv;
  float recovery = input.state.recovery;
  float reported_voltage = voltage;
  float reported_recovery = recovery;
  bool emitted = false;
  int crossing_count = 0;

  const float maximum_conductance =
      input.g_ampa + input.g_nmda + input.g_gabaa;
  const float stability_interval =
      0.5f / fmaxf(1.0f, maximum_conductance);
  const int segment_count =
      max(1, static_cast<int>(ceilf(input.dt_ms / stability_interval)));
  const float segment_dt = input.dt_ms / static_cast<float>(segment_count);
  const float tolerance = fmaxf(segment_dt * 1.0e-7f, 1.0e-8f);

  for (int segment = 0; segment < segment_count; ++segment) {
    float remaining = segment_dt;
    bool segment_emitted = false;
    float segment_pre_voltage = voltage;
    float segment_pre_recovery = recovery;

    for (int event = 0; event < 65 && remaining > tolerance; ++event) {
      const bool immediate = voltage >= input.threshold_mv;
      const bool crosses =
          immediate ||
          crossing_occurs(voltage, recovery, remaining, input);
      if (!crosses) {
        const NeuronState end =
            safe_heun(voltage, recovery, remaining, input);
        voltage = end.voltage_mv;
        recovery = end.recovery;
        if (!segment_emitted) {
          segment_pre_voltage = voltage;
          segment_pre_recovery = recovery;
        }
        remaining = 0.0f;
        break;
      }

      float crossing_time = 0.0f;
      float crossing_recovery = recovery;
      if (!immediate) {
        float lower = 0.0f;
        float upper = remaining;
#pragma unroll
        for (int iteration = 0; iteration < 32; ++iteration) {
          const float midpoint = 0.5f * (lower + upper);
          if (crossing_occurs(voltage, recovery, midpoint, input)) {
            upper = midpoint;
          } else {
            lower = midpoint;
          }
        }
        crossing_time = upper;
        crossing_recovery =
            safe_heun(voltage, recovery, crossing_time, input)
                .recovery;
      }

      emitted = true;
      segment_emitted = true;
      ++crossing_count;
      segment_pre_voltage = input.threshold_mv;
      segment_pre_recovery = crossing_recovery;
      voltage = input.parameters.c_mv;
      recovery = crossing_recovery + input.parameters.d;
      remaining = fmaxf(remaining - crossing_time, 0.0f);
      if (crossing_count > kMaximumCrossings) {
        output.overflow = true;
        remaining = 0.0f;
        break;
      }
    }

    if (remaining > tolerance) {
      output.overflow = true;
    }
    if (segment_emitted) {
      reported_voltage = segment_pre_voltage;
      reported_recovery = segment_pre_recovery;
    }
    if (output.overflow) {
      break;
    }
  }

  output.state = {voltage, recovery};
  output.emitted = emitted;
  output.spike_count = crossing_count;
  output.pre_reset_voltage_mv = emitted ? reported_voltage : voltage;
  output.pre_reset_recovery = emitted ? reported_recovery : recovery;
  return output;
}

__global__ void solver_batch_kernel(const SolverInput* inputs,
                                    SolverOutput* outputs, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    outputs[index] = event_resolved_step(inputs[index]);
  }
}

__global__ void receptor_trace_kernel(ReceptorKernel kernel,
                                      const float* arrivals, float* trace,
                                      int count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  ReceptorState state{};
  for (int index = 0; index < count; ++index) {
    trace[index] = receptor_advance(kernel, arrivals[index], &state);
  }
}

__device__ inline NeuronParameters calibration_parameters(
    bool fast_spiking, std::uint32_t stream, int trial) {
  const float heterogeneity =
      uniform01(0, stream, static_cast<std::uint64_t>(trial), 0, 0);
  if (fast_spiking) {
    return {0.02f + 0.08f * heterogeneity,
            0.25f - 0.05f * heterogeneity, -65.0f, 2.0f};
  }
  const float squared = heterogeneity * heterogeneity;
  return {0.02f, 0.20f, -65.0f + 15.0f * squared,
          8.0f - 6.0f * squared};
}

__device__ inline NeuronState calibration_initial_state(
    const NeuronParameters& parameters, bool fast_spiking,
    std::uint32_t stream, int trial) {
  const float draw =
      uniform01(0, stream, static_cast<std::uint64_t>(trial), 0, 1);
  float voltage = 0.0f;
  if (fast_spiking) {
    voltage = -65.0f + 10.0f * draw;
  } else {
    const float lower = fminf(parameters.c_mv, -55.0f);
    const float upper = fmaxf(parameters.c_mv, -55.0f);
    voltage = lower + draw * (upper - lower);
  }
  return {voltage, parameters.b * voltage};
}

struct ReceptorTriplet {
  ReceptorState ampa{};
  ReceptorState nmda{};
  ReceptorState gabaa{};
};

__device__ inline NeuronParameters production_neuron_parameters(
    std::uint64_t seed, int neuron);
__device__ inline NeuronState production_initial_state(
    std::uint64_t seed, int neuron,
    const NeuronParameters& parameters);

__device__ inline int simulate_external(const Config& config, float quantum,
                                        int rate_hz, int trial) {
  const ReceptorKernel ampa =
      make_receptor_kernel(config.ampa_rise_ms, config.ampa_decay_ms,
                           config.dt_ms);
  const ReceptorKernel nmda =
      make_receptor_kernel(config.nmda_rise_ms, config.nmda_decay_ms,
                           config.dt_ms);
  ReceptorTriplet receptors{};
  const std::uint32_t stream = 100u + static_cast<std::uint32_t>(rate_hz);
  const NeuronParameters parameters =
      calibration_parameters(false, 11u, trial);
  NeuronState state =
      calibration_initial_state(parameters, false, 11u, trial);
  const float nmda_quantum = nmda_quantum_from_ampa(quantum);
  int spikes = 0;
  for (int time = 0; time < 150; ++time) {
    const int events =
        time < 50
            ? poisson_small(rate_hz * config.dt_ms / 1000.0f, 0, stream,
                            static_cast<std::uint64_t>(trial), time)
            : 0;
    const float g_ampa =
        receptor_advance(ampa, events * config.external_ampa_weight,
                         &receptors.ampa);
    const float g_nmda =
        receptor_advance(nmda, events * config.external_nmda_weight,
                         &receptors.nmda);
    SolverInput input{};
    input.state = state;
    input.parameters = parameters;
    input.g_ampa = quantum * g_ampa;
    input.g_nmda = nmda_quantum * g_nmda;
    input.dt_ms = config.dt_ms;
    input.threshold_mv = config.threshold_mv;
    input.excitatory_reversal_mv = config.excitatory_reversal_mv;
    input.gabaa_reversal_mv = config.gabaa_reversal_mv;
    const SolverOutput step = event_resolved_step(input);
    state = step.state;
    spikes += step.emitted ? 1 : 0;
  }
  return spikes;
}

__device__ inline int simulate_background(const Config& config,
                                          float quantum,
                                          bool fast_spiking, int trial) {
  const ReceptorKernel ampa =
      make_receptor_kernel(config.ampa_rise_ms, config.ampa_decay_ms,
                           config.dt_ms);
  const ReceptorKernel nmda =
      make_receptor_kernel(config.nmda_rise_ms, config.nmda_decay_ms,
                           config.dt_ms);
  ReceptorTriplet receptors{};
  const std::uint32_t parameter_stream = fast_spiking ? 21u : 20u;
  const NeuronParameters parameters =
      calibration_parameters(fast_spiking, parameter_stream, trial);
  NeuronState state = calibration_initial_state(
      parameters, fast_spiking, parameter_stream, trial);
  const float nmda_quantum = nmda_quantum_from_ampa(quantum);
  int spikes = 0;
  for (int time = 0; time < 5000; ++time) {
    const int events =
        poisson_small(50.0f * config.dt_ms / 1000.0f, 0, 220u,
                      static_cast<std::uint64_t>(trial), time);
    const float g_ampa =
        receptor_advance(ampa, events * config.background_ampa_weight,
                         &receptors.ampa);
    const float g_nmda =
        receptor_advance(nmda, events * config.background_nmda_weight,
                         &receptors.nmda);
    SolverInput input{};
    input.state = state;
    input.parameters = parameters;
    input.g_ampa = quantum * g_ampa;
    input.g_nmda = nmda_quantum * g_nmda;
    input.dt_ms = config.dt_ms;
    input.threshold_mv = config.threshold_mv;
    input.excitatory_reversal_mv = config.excitatory_reversal_mv;
    input.gabaa_reversal_mv = config.gabaa_reversal_mv;
    const SolverOutput step = event_resolved_step(input);
    state = step.state;
    if (time >= 1000) {
      spikes += step.emitted ? 1 : 0;
    }
  }
  return spikes;
}

__device__ inline int simulate_feedforward(const Config& config,
                                           float quantum,
                                           bool fast_spiking,
                                           int contacts) {
  const ReceptorKernel ampa =
      make_receptor_kernel(config.ampa_rise_ms, config.ampa_decay_ms,
                           config.dt_ms);
  const ReceptorKernel nmda =
      make_receptor_kernel(config.nmda_rise_ms, config.nmda_decay_ms,
                           config.dt_ms);
  ReceptorTriplet receptors{};
  const std::uint32_t stream = fast_spiking ? 31u : 30u;
  const NeuronParameters parameters =
      calibration_parameters(fast_spiking, stream, 0);
  NeuronState state =
      calibration_initial_state(parameters, fast_spiking, stream, 0);
  const float nmda_quantum = nmda_quantum_from_ampa(quantum);
  constexpr float ampa_contact_efficacy =
      kCalibrationExcitatoryContactWeight;
  const float nmda_contact_efficacy =
      calibration_nmda_contact_efficacy(
          ampa_contact_efficacy, true);
  int spikes = 0;
  for (int time = 0; time < 20; ++time) {
    const float events = time == 0 ? static_cast<float>(contacts) : 0.0f;
    const float g_ampa =
        receptor_advance(
            ampa, events * ampa_contact_efficacy, &receptors.ampa);
    const float g_nmda =
        receptor_advance(
            nmda, events * nmda_contact_efficacy, &receptors.nmda);
    SolverInput input{};
    input.state = state;
    input.parameters = parameters;
    input.g_ampa = quantum * g_ampa;
    input.g_nmda = nmda_quantum * g_nmda;
    input.dt_ms = config.dt_ms;
    input.threshold_mv = config.threshold_mv;
    input.excitatory_reversal_mv = config.excitatory_reversal_mv;
    input.gabaa_reversal_mv = config.gabaa_reversal_mv;
    const SolverOutput step = event_resolved_step(input);
    state = step.state;
    spikes += step.emitted ? 1 : 0;
  }
  return spikes;
}

__device__ inline GabaaIpspMetrics simulate_gabaa_ipsp(
    const Config& config, float q_gabaa, int contact_count) {
  const ReceptorKernel gabaa =
      make_receptor_kernel(config.gabaa_rise_ms, config.gabaa_decay_ms,
                           config.dt_ms);
  constexpr int production_e_neuron =
      kAuditoryNeurons + kVisualNeurons;
  const NeuronParameters parameters =
      production_neuron_parameters(
          config.seed, production_e_neuron);
  NeuronState settled = production_initial_state(
      config.seed, production_e_neuron, parameters);

  SolverInput settle_input{};
  settle_input.parameters = parameters;
  settle_input.dt_ms = config.dt_ms;
  settle_input.threshold_mv = config.threshold_mv;
  settle_input.excitatory_reversal_mv =
      config.excitatory_reversal_mv;
  settle_input.gabaa_reversal_mv = config.gabaa_reversal_mv;
  for (int time = 0; time < kGabaaIpspSettleMs; ++time) {
    settle_input.state = settled;
    settled = event_resolved_step(settle_input).state;
  }

  NeuronState control_state = settled;
  NeuronState gabaa_state = settled;
  ReceptorState gabaa_receptor{};
  GabaaIpspMetrics metrics{};
  metrics.contact_count = contact_count;
  for (int time = 0;
       time < kGabaaUnitaryDelayMs + kGabaaIpspWindowMs; ++time) {
    const float arrival =
        time == kGabaaUnitaryDelayMs
            ? q_gabaa * kGabaaMedianInitialContactWeight *
                  static_cast<float>(contact_count)
            : 0.0f;
    const float g_gabaa =
        receptor_advance(gabaa, arrival, &gabaa_receptor);

    SolverInput control_input{};
    control_input.state = control_state;
    control_input.parameters = parameters;
    control_input.dt_ms = config.dt_ms;
    control_input.threshold_mv = config.threshold_mv;
    control_input.excitatory_reversal_mv =
        config.excitatory_reversal_mv;
    control_input.gabaa_reversal_mv =
        config.gabaa_reversal_mv;
    const SolverOutput control_step =
        event_resolved_step(control_input);
    control_state = control_step.state;

    SolverInput gabaa_input = control_input;
    gabaa_input.state = gabaa_state;
    gabaa_input.g_gabaa = g_gabaa;
    const float outward_current =
        g_gabaa *
        (gabaa_state.voltage_mv - config.gabaa_reversal_mv);
    const SolverOutput gabaa_step =
        event_resolved_step(gabaa_input);
    gabaa_state = gabaa_step.state;

    if (time < kGabaaUnitaryDelayMs) {
      continue;
    }
    metrics.control_spikes += control_step.emitted ? 1 : 0;
    metrics.gabaa_spikes += gabaa_step.emitted ? 1 : 0;
    const float signed_difference =
        gabaa_state.voltage_mv - control_state.voltage_mv;
    metrics.signed_nadir_mv =
        fminf(metrics.signed_nadir_mv, signed_difference);
    metrics.amplitude_mv =
        fmaxf(metrics.amplitude_mv, -signed_difference);
    metrics.area_mv_ms +=
        fmaxf(-signed_difference, 0.0f) * config.dt_ms;
    metrics.outward_charge +=
        fmaxf(outward_current, 0.0f) * config.dt_ms;
  }
  return metrics;
}

__device__ inline int simulate_gabaa(
    const Config& config, float q_ff_e, float q_gabaa,
    GabaaRelayAudit* relay_audit = nullptr) {
  const ReceptorKernel ampa =
      make_receptor_kernel(config.ampa_rise_ms, config.ampa_decay_ms,
                           config.dt_ms);
  const ReceptorKernel nmda =
      make_receptor_kernel(config.nmda_rise_ms, config.nmda_decay_ms,
                           config.dt_ms);
  const ReceptorKernel gabaa =
      make_receptor_kernel(config.gabaa_rise_ms, config.gabaa_decay_ms,
                           config.dt_ms);
  ReceptorTriplet receptors{};
  const NeuronParameters parameters =
      calibration_parameters(false, 40u, 0);
  NeuronState state =
      calibration_initial_state(parameters, false, 40u, 0);
  const float e_nmda_quantum = nmda_quantum_from_ampa(q_ff_e);
  const float i_nmda_quantum =
      nmda_quantum_from_ampa(config.q_ff_i);
  constexpr float ampa_contact_efficacy =
      kCalibrationExcitatoryContactWeight;
  const float nmda_contact_efficacy =
      calibration_nmda_contact_efficacy(
          ampa_contact_efficacy, true);

  ReceptorTriplet relay_receptors[kInhibitoryScaffoldInDegree]{};
  NeuronParameters relay_parameters[kInhibitoryScaffoldInDegree]{};
  NeuronState relay_states[kInhibitoryScaffoldInDegree]{};
  float inhibitory_weights[kInhibitoryScaffoldInDegree]{};
  int last_direct_packet_ms[kInhibitoryScaffoldInDegree]{};
  float gabaa_queue[kGabaaDurationMs]{};
  for (int relay = 0; relay < kInhibitoryScaffoldInDegree; ++relay) {
    relay_parameters[relay] =
        calibration_parameters(true, 31u, relay);
    relay_states[relay] = calibration_initial_state(
        relay_parameters[relay], true, 31u, relay);
    const int runtime_contact =
        relay * kExcitatoryNeurons;
    inhibitory_weights[relay] =
        0.02f +
        0.03f *
            uniform01(0, 846u, runtime_contact, 0, 0);
    last_direct_packet_ms[relay] = -1;
  }
  if (relay_audit != nullptr) {
    *relay_audit = GabaaRelayAudit{};
    relay_audit->relay_neurons =
        kInhibitoryScaffoldInDegree;
    relay_audit->excitatory_packets = 2;
  }

  int spikes = 0;
  for (int time = 0; time < kGabaaDurationMs; ++time) {
    const float excitatory_events =
        (time == 20 || time == 35) ? 12.0f : 0.0f;
    const bool relay_packet =
        time == 20 + 3 || time == 35 + 3;
    const float relay_events =
        relay_packet ? 12.0f : 0.0f;

    for (int relay = 0; relay < kInhibitoryScaffoldInDegree;
         ++relay) {
      if (relay_packet) {
        last_direct_packet_ms[relay] = time - 3;
      }
      const float relay_g_ampa = receptor_advance(
          ampa, relay_events * ampa_contact_efficacy,
          &relay_receptors[relay].ampa);
      const float relay_g_nmda = receptor_advance(
          nmda, relay_events * nmda_contact_efficacy,
          &relay_receptors[relay].nmda);
      SolverInput relay_input{};
      relay_input.state = relay_states[relay];
      relay_input.parameters = relay_parameters[relay];
      relay_input.g_ampa = config.q_ff_i * relay_g_ampa;
      relay_input.g_nmda = i_nmda_quantum * relay_g_nmda;
      relay_input.dt_ms = config.dt_ms;
      relay_input.threshold_mv = config.threshold_mv;
      relay_input.excitatory_reversal_mv =
          config.excitatory_reversal_mv;
      relay_input.gabaa_reversal_mv =
          config.gabaa_reversal_mv;
      const SolverOutput relay_step =
          event_resolved_step(relay_input);
      relay_states[relay] = relay_step.state;
      if (!relay_step.emitted) {
        continue;
      }
      if (relay_audit != nullptr) {
        ++relay_audit->relay_spikes;
      }
      const int arrival_time = time + 1;
      if (arrival_time >= kGabaaDurationMs) {
        continue;
      }
      gabaa_queue[arrival_time] += inhibitory_weights[relay];
      if (relay_audit != nullptr) {
        ++relay_audit->gabaa_arrivals;
        const int direct_packet_ms =
            last_direct_packet_ms[relay];
        const int lag = arrival_time - direct_packet_ms;
        if (direct_packet_ms >= 0 && lag < kGabaaDurationMs) {
          ++relay_audit
                ->direct_to_gabaa_lag_counts[lag];
          ++relay_audit->lag_count;
        }
      }
    }

    const float g_ampa = receptor_advance(
        ampa, excitatory_events * ampa_contact_efficacy,
        &receptors.ampa);
    const float g_nmda = receptor_advance(
        nmda, excitatory_events * nmda_contact_efficacy,
        &receptors.nmda);
    const float g_gabaa =
        receptor_advance(
            gabaa, gabaa_queue[time], &receptors.gabaa);
    SolverInput input{};
    input.state = state;
    input.parameters = parameters;
    input.g_ampa = q_ff_e * g_ampa;
    input.g_nmda = e_nmda_quantum * g_nmda;
    input.g_gabaa = q_gabaa * g_gabaa;
    input.dt_ms = config.dt_ms;
    input.threshold_mv = config.threshold_mv;
    input.excitatory_reversal_mv = config.excitatory_reversal_mv;
    input.gabaa_reversal_mv = config.gabaa_reversal_mv;
    const SolverOutput step = event_resolved_step(input);
    state = step.state;
    spikes += step.emitted ? 1 : 0;
  }
  if (relay_audit != nullptr && relay_audit->lag_count > 0) {
    int weighted_lag_sum = 0;
    bool have_lag = false;
    for (int lag = 0; lag < kGabaaDurationMs; ++lag) {
      const int count =
          relay_audit->direct_to_gabaa_lag_counts[lag];
      if (count == 0) {
        continue;
      }
      if (!have_lag) {
        relay_audit->minimum_lag_ms = lag;
        have_lag = true;
      }
      relay_audit->maximum_lag_ms = lag;
      weighted_lag_sum += lag * count;
    }
    relay_audit->mean_lag_ms =
        static_cast<float>(weighted_lag_sum) /
        static_cast<float>(relay_audit->lag_count);
  }
  return spikes;
}

__global__ void gabaa_relay_audit_kernel(
    Config config, float q_gabaa, GabaaRelayAudit* audit) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  GabaaRelayAudit measured{};
  measured.with_gabaa_spikes =
      simulate_gabaa(
          config, config.q_ff_e, q_gabaa, &measured);
  measured.without_gabaa_spikes =
      simulate_gabaa(config, config.q_ff_e, 0.0f);
  *audit = measured;
}

__global__ void gabaa_efficacy_audit_kernel(
    Config config, float q_gabaa, GabaaEfficacyAudit* audit) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  GabaaEfficacyAudit measured{};
  measured.unitary =
      simulate_gabaa_ipsp(config, q_gabaa, 1);
  measured.compound = simulate_gabaa_ipsp(
      config, q_gabaa, kInhibitoryScaffoldInDegree);
  *audit = measured;
}

__device__ int block_lower_median(int value, int valid_count,
                                  int* shared_values) {
  const int thread = threadIdx.x;
  shared_values[thread] =
      thread < valid_count ? value : INT_MAX;
  __syncthreads();
  for (int width = 2; width <= blockDim.x; width <<= 1) {
    for (int stride = width >> 1; stride > 0; stride >>= 1) {
      const int partner = thread ^ stride;
      if (partner > thread) {
        const bool ascending = (thread & width) == 0;
        const int own = shared_values[thread];
        const int other = shared_values[partner];
        if ((ascending && own > other) || (!ascending && own < other)) {
          shared_values[thread] = other;
          shared_values[partner] = own;
        }
      }
      __syncthreads();
    }
  }
  const int median = shared_values[(valid_count - 1) / 2];
  __syncthreads();
  return median;
}

__global__ void assay_candidates_kernel(Config config, AssayKind kind,
                                        const float* candidates,
                                        AssayMetrics* metrics,
                                        int candidate_count) {
  const int candidate = blockIdx.x;
  if (candidate >= candidate_count) {
    return;
  }
  const int thread = threadIdx.x;
  const float quantum = candidates[candidate];
  __shared__ int shared_values[kAssayThreads];

  if (kind == AssayKind::kExternalRs) {
    const int count25 = simulate_external(config, quantum, 25, thread);
    const int count50 = simulate_external(config, quantum, 50, thread);
    const int count100 = simulate_external(config, quantum, 100, thread);
    const int median25 = block_lower_median(count25, 256, shared_values);
    const int median50 = block_lower_median(count50, 256, shared_values);
    const int median100 = block_lower_median(count100, 256, shared_values);
    if (thread == 0) {
      metrics[candidate] = {static_cast<float>(median50),
                            static_cast<float>(median25),
                            static_cast<float>(median100)};
    }
    return;
  }

  if (kind == AssayKind::kBackgroundRs ||
      kind == AssayKind::kBackgroundFs) {
    const bool fast_spiking = kind == AssayKind::kBackgroundFs;
    const int count =
        thread < 128
            ? simulate_background(config, quantum, fast_spiking, thread)
            : 0;
    const int median = block_lower_median(count, 128, shared_values);
    if (thread == 0) {
      metrics[candidate] = {static_cast<float>(median),
                            static_cast<float>(median) / 4.0f, 0.0f};
    }
    return;
  }

  if (thread != 0) {
    return;
  }
  if (kind == AssayKind::kFeedforwardE ||
      kind == AssayKind::kFeedforwardI) {
    const bool fast_spiking = kind == AssayKind::kFeedforwardI;
    const int large =
        simulate_feedforward(config, quantum, fast_spiking, 12);
    const int small = simulate_feedforward(config, quantum, fast_spiking, 4);
    const int baseline =
        fast_spiking ? 0 : simulate_gabaa(config, quantum, 0.0f);
    metrics[candidate] = {static_cast<float>(large),
                          static_cast<float>(small),
                          static_cast<float>(baseline)};
    return;
  }
  const GabaaIpspMetrics unitary =
      simulate_gabaa_ipsp(config, quantum, 1);
  metrics[candidate] = {
      unitary.amplitude_mv, unitary.signed_nadir_mv,
      unitary.area_mv_ms};
}

}  // namespace

std::array<std::uint32_t, 4> philox4x32_10_host(
    std::array<std::uint32_t, 4> counter,
    std::array<std::uint32_t, 2> key) {
  const PhiloxWords result =
      philox10({counter[0], counter[1], counter[2], counter[3]},
               key[0], key[1]);
  return {result.x, result.y, result.z, result.w};
}

DeviceInfo query_device(int device) {
  int count = 0;
  CLEAN_MSI_CUDA(cudaGetDeviceCount(&count));
  if (device < 0 || device >= count) {
    throw std::invalid_argument("CUDA device ordinal is unavailable.");
  }
  cudaDeviceProp properties{};
  CLEAN_MSI_CUDA(cudaGetDeviceProperties(&properties, device));
  DeviceInfo result{};
  result.ordinal = device;
  result.major = properties.major;
  result.minor = properties.minor;
  result.multiprocessors = properties.multiProcessorCount;
  result.cooperative_launch = properties.cooperativeLaunch != 0;
  result.name = properties.name;
  return result;
}

__host__ __device__ ReceptorKernel make_receptor_kernel(
    float rise_ms, float decay_ms, float dt_ms) {
  ReceptorKernel kernel{};
  kernel.rise_ms = rise_ms;
  kernel.decay_ms = decay_ms;
  kernel.rise_decay = expf(-dt_ms / rise_ms);
  kernel.decay_decay = expf(-dt_ms / decay_ms);
  const float peak_time =
      rise_ms * decay_ms * logf(decay_ms / rise_ms) /
      (decay_ms - rise_ms);
  const float unnormalized =
      expf(-peak_time / decay_ms) - expf(-peak_time / rise_ms);
  kernel.normalization = 1.0f / unnormalized;
  return kernel;
}

float nmda_voltage_block_host(float voltage_mv) {
  const float exponent = clamp_float(-0.062f * voltage_mv, -80.0f, 80.0f);
  return 1.0f / (1.0f + std::exp(exponent) / 3.57f);
}

__host__ __device__ float nmda_quantum_from_ampa(float ampa_quantum) {
  const float block =
      1.0f / (1.0f + expf(0.062f * 40.0f) / 3.57f);
  return ampa_quantum / (2.0f * block);
}

__host__ __device__ float clopath_pair_delta(
    float eta, float pre_trace, float post_voltage_mv,
    float post_minus_mv, float post_plus_mv,
    float theta_minus_mv, float theta_plus_mv,
    float post_homeostasis_mv2, bool pre_spike) {
  const float minus_factor =
      fmaxf((post_minus_mv - theta_minus_mv) / 20.0f, 0.0f);
  const float instant_plus =
      fmaxf((post_voltage_mv - theta_plus_mv) / 20.0f, 0.0f);
  const float filtered_plus =
      fmaxf((post_plus_mv - theta_minus_mv) / 20.0f, 0.0f);
  const float homeostasis_multiplier =
      clopath_homeostasis_multiplier(post_homeostasis_mv2);
  const float ltp =
      eta * pre_trace * instant_plus * filtered_plus;
  const float ltd =
      pre_spike
          ? 1.3125f * eta * minus_factor *
                homeostasis_multiplier
          : 0.0f;
  return ltp - ltd;
}

__host__ __device__ float additive_hard_bound_update(
    float weight, float raw_delta, float lower_bound,
    float upper_bound) {
  if (!(upper_bound > lower_bound)) {
    return weight;
  }
  return fminf(
      fmaxf(weight + raw_delta, lower_bound),
      upper_bound);
}

__host__ __device__ float multiplicative_soft_bound_update(
    float weight, float raw_delta, float lower_bound,
    float upper_bound) {
  if (!(upper_bound > lower_bound)) {
    return weight;
  }
  const float range = upper_bound - lower_bound;
  const float bounded_weight =
      fminf(fmaxf(weight, lower_bound), upper_bound);
  const float bound_factor =
      raw_delta >= 0.0f
          ? (upper_bound - bounded_weight) / range
          : (bounded_weight - lower_bound) / range;
  return fminf(
      fmaxf(
          bounded_weight + raw_delta * bound_factor,
          lower_bound),
      upper_bound);
}

__host__ __device__ float ordered_vogels_istdp_update(
    float weight, bool pre_event, bool post_event,
    float pre_trace_after, float post_trace_after,
    float eta, float alpha, float lower_bound,
    float upper_bound) {
  if (!(upper_bound > lower_bound)) {
    return weight;
  }
  float updated =
      fminf(fmaxf(weight, lower_bound), upper_bound);
  if (pre_event) {
    const float post_trace_before_event =
        post_trace_after - (post_event ? 1.0f : 0.0f);
    updated =
        fminf(
            fmaxf(
                updated +
                    eta *
                        (post_trace_before_event - alpha),
                lower_bound),
            upper_bound);
  }
  if (post_event) {
    updated =
        fminf(
            fmaxf(
                updated + eta * pre_trace_after,
                lower_bound),
            upper_bound);
  }
  return updated;
}

__host__ __device__ float shared_nmda_contact_efficacy(
    float learned_weight, float fixed_receptor_ratio,
    bool contact_active) {
  return contact_active
             ? learned_weight * fixed_receptor_ratio
             : 0.0f;
}

__host__ __device__ float calibration_nmda_contact_efficacy(
    float ampa_contact_efficacy, bool contact_active) {
  return shared_nmda_contact_efficacy(
      ampa_contact_efficacy, 1.0f, contact_active);
}

std::vector<SolverOutput> run_solver_batch(
    int device, const std::vector<SolverInput>& inputs) {
  if (inputs.empty()) {
    return {};
  }
  CLEAN_MSI_CUDA(cudaSetDevice(device));
  SolverInput* device_inputs = nullptr;
  SolverOutput* device_outputs = nullptr;
  CLEAN_MSI_CUDA(cudaMalloc(
      reinterpret_cast<void**>(&device_inputs),
      inputs.size() * sizeof(SolverInput)));
  try {
    CLEAN_MSI_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_outputs),
        inputs.size() * sizeof(SolverOutput)));
    CLEAN_MSI_CUDA(cudaMemcpy(
        device_inputs, inputs.data(), inputs.size() * sizeof(SolverInput),
        cudaMemcpyHostToDevice));
    constexpr int threads = 128;
    const int blocks =
        static_cast<int>((inputs.size() + threads - 1) / threads);
    solver_batch_kernel<<<blocks, threads>>>(
        device_inputs, device_outputs, static_cast<int>(inputs.size()));
    CLEAN_MSI_CUDA(cudaGetLastError());
    std::vector<SolverOutput> outputs(inputs.size());
    CLEAN_MSI_CUDA(cudaMemcpy(
        outputs.data(), device_outputs,
        outputs.size() * sizeof(SolverOutput), cudaMemcpyDeviceToHost));
    CLEAN_MSI_CUDA(cudaFree(device_outputs));
    CLEAN_MSI_CUDA(cudaFree(device_inputs));
    return outputs;
  } catch (...) {
    cudaFree(device_outputs);
    cudaFree(device_inputs);
    throw;
  }
}

std::vector<float> run_receptor_trace(
    int device, const ReceptorKernel& kernel,
    const std::vector<float>& arrivals) {
  if (arrivals.empty()) {
    return {};
  }
  CLEAN_MSI_CUDA(cudaSetDevice(device));
  float* device_arrivals = nullptr;
  float* device_trace = nullptr;
  CLEAN_MSI_CUDA(cudaMalloc(
      reinterpret_cast<void**>(&device_arrivals),
      arrivals.size() * sizeof(float)));
  try {
    CLEAN_MSI_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_trace),
        arrivals.size() * sizeof(float)));
    CLEAN_MSI_CUDA(cudaMemcpy(
        device_arrivals, arrivals.data(), arrivals.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    receptor_trace_kernel<<<1, 1>>>(
        kernel, device_arrivals, device_trace,
        static_cast<int>(arrivals.size()));
    CLEAN_MSI_CUDA(cudaGetLastError());
    std::vector<float> trace(arrivals.size());
    CLEAN_MSI_CUDA(cudaMemcpy(
        trace.data(), device_trace, trace.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    CLEAN_MSI_CUDA(cudaFree(device_trace));
    CLEAN_MSI_CUDA(cudaFree(device_arrivals));
    return trace;
  } catch (...) {
    cudaFree(device_trace);
    cudaFree(device_arrivals);
    throw;
  }
}

std::vector<AssayMetrics> evaluate_assay_candidates(
    int device, const Config& config, AssayKind kind,
    const std::vector<float>& candidates) {
  if (candidates.empty()) {
    return {};
  }
  CLEAN_MSI_CUDA(cudaSetDevice(device));
  float* device_candidates = nullptr;
  AssayMetrics* device_metrics = nullptr;
  CLEAN_MSI_CUDA(cudaMalloc(
      reinterpret_cast<void**>(&device_candidates),
      candidates.size() * sizeof(float)));
  try {
    CLEAN_MSI_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_metrics),
        candidates.size() * sizeof(AssayMetrics)));
    CLEAN_MSI_CUDA(cudaMemcpy(
        device_candidates, candidates.data(),
        candidates.size() * sizeof(float), cudaMemcpyHostToDevice));
    assay_candidates_kernel<<<static_cast<int>(candidates.size()),
                              kAssayThreads>>>(
        config, kind, device_candidates, device_metrics,
        static_cast<int>(candidates.size()));
    CLEAN_MSI_CUDA(cudaGetLastError());
    std::vector<AssayMetrics> metrics(candidates.size());
    CLEAN_MSI_CUDA(cudaMemcpy(
        metrics.data(), device_metrics,
        metrics.size() * sizeof(AssayMetrics), cudaMemcpyDeviceToHost));
    CLEAN_MSI_CUDA(cudaFree(device_metrics));
    CLEAN_MSI_CUDA(cudaFree(device_candidates));
    return metrics;
  } catch (...) {
    cudaFree(device_metrics);
    cudaFree(device_candidates);
    throw;
  }
}

GabaaRelayAudit audit_gabaa_relay_timing(
    int device, const Config& config, float q_gabaa) {
  CLEAN_MSI_CUDA(cudaSetDevice(device));
  GabaaRelayAudit* device_audit = nullptr;
  CLEAN_MSI_CUDA(cudaMalloc(
      reinterpret_cast<void**>(&device_audit),
      sizeof(GabaaRelayAudit)));
  try {
    gabaa_relay_audit_kernel<<<1, 1>>>(
        config, q_gabaa, device_audit);
    CLEAN_MSI_CUDA(cudaGetLastError());
    GabaaRelayAudit audit{};
    CLEAN_MSI_CUDA(cudaMemcpy(
        &audit, device_audit, sizeof(GabaaRelayAudit),
        cudaMemcpyDeviceToHost));
    CLEAN_MSI_CUDA(cudaFree(device_audit));
    return audit;
  } catch (...) {
    cudaFree(device_audit);
    throw;
  }
}

GabaaEfficacyAudit audit_gabaa_efficacy(
    int device, const Config& config, float q_gabaa) {
  CLEAN_MSI_CUDA(cudaSetDevice(device));
  GabaaEfficacyAudit* device_audit = nullptr;
  CLEAN_MSI_CUDA(cudaMalloc(
      reinterpret_cast<void**>(&device_audit),
      sizeof(GabaaEfficacyAudit)));
  try {
    gabaa_efficacy_audit_kernel<<<1, 1>>>(
        config, q_gabaa, device_audit);
    CLEAN_MSI_CUDA(cudaGetLastError());
    GabaaEfficacyAudit audit{};
    CLEAN_MSI_CUDA(cudaMemcpy(
        &audit, device_audit, sizeof(GabaaEfficacyAudit),
        cudaMemcpyDeviceToHost));
    CLEAN_MSI_CUDA(cudaFree(device_audit));
    return audit;
  } catch (...) {
    cudaFree(device_audit);
    throw;
  }
}

}  // namespace clean_msi

namespace clean_msi {
namespace {

__global__ void clopath_assay_kernel(
    const float* etas, LearningAssayMetrics* metrics, int candidate_count) {
  const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) {
    return;
  }
  const float eta = etas[candidate];
  const float theta_minus_mv =
      izhikevich_stable_rest_voltage_mv(0.20f);
  const float causal_trace = expf(-10.0f / 15.0f);
  constexpr float kInitialWeight = 0.25f;
  float causal_weight = kInitialWeight;
  float shuffled_weight = kInitialWeight;
  for (int pairing = 0; pairing < kClopathPairings; ++pairing) {
    const float causal_delta = clopath_pair_delta(
        eta, causal_trace, -20.0f, -40.0f, -40.0f,
        theta_minus_mv, kClopathThetaPlusMv,
        kClopathHomeostasisReferenceMv2, false);
    causal_weight =
        additive_hard_bound_update(
            causal_weight, causal_delta, 0.0f, 1.0f);
    const float uniform =
        uniform01(0, 500u, pairing, 0, 0);
    const int lag = min(1000, static_cast<int>(uniform * 1001.0f));
    const float shuffled_trace = expf(-static_cast<float>(lag) / 15.0f);
    const float shuffled_delta = clopath_pair_delta(
        eta, shuffled_trace, -20.0f, -40.0f, -55.0f,
        theta_minus_mv, kClopathThetaPlusMv,
        kClopathHomeostasisReferenceMv2, false);
    shuffled_weight =
        additive_hard_bound_update(
            shuffled_weight, shuffled_delta, 0.0f, 1.0f);
  }
  metrics[candidate].clopath_delta =
      causal_weight - kInitialWeight;
  metrics[candidate].clopath_shuffled_drift =
      shuffled_weight - kInitialWeight;
}

__global__ void oja_assay_kernel(
    const float* etas, LearningAssayMetrics* metrics,
    int candidate_count) {
  const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) {
    return;
  }
  const float eta = etas[candidate];
  float weights[20];
#pragma unroll
  for (int contact = 0; contact < 20; ++contact) {
    weights[contact] = 0.10f;
  }
  const float correlated_trace = expf(-10.0f / 20.0f);
  for (int event = 0; event < kOjaEvents; ++event) {
#pragma unroll
    for (int contact = 0; contact < 20; ++contact) {
      float pre_trace = correlated_trace;
      if (contact >= 5) {
        const float uniform =
            uniform01(0, 510u + static_cast<std::uint32_t>(contact),
                      event, 0, 0);
        const int lag = min(100, static_cast<int>(uniform * 101.0f));
        pre_trace = expf(-static_cast<float>(lag) / 20.0f);
      }
      weights[contact] = clamp_float(
          weights[contact] + eta * (pre_trace - weights[contact]),
          0.0f, 1.0f);
    }
  }
  float correlated_mean = 0.0f;
  float distractor_mean = 0.0f;
  int bound_count = 0;
#pragma unroll
  for (int contact = 0; contact < 20; ++contact) {
    if (contact < 5) {
      correlated_mean += weights[contact] / 5.0f;
    } else {
      distractor_mean += weights[contact] / 15.0f;
    }
    bound_count +=
        (weights[contact] <= 0.0f || weights[contact] >= 1.0f) ? 1 : 0;
  }
  metrics[candidate].oja_selection_ratio =
      correlated_mean / fmaxf(distractor_mean, 1.0e-8f);
  metrics[candidate].oja_bound_fraction =
      static_cast<float>(bound_count) / 20.0f;
}

__global__ void istdp_assay_kernel(
    const float* etas, LearningAssayMetrics* metrics,
    int candidate_count) {
  const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) {
    return;
  }
  const float eta = etas[candidate];
  float weights[2][kIstdpInhibitoryInputs];
  float inhibitory_trace[kIstdpInhibitoryInputs]{};
  float excitatory_trace[2]{};
  float measured_spikes[2]{};
  float bound_events[2][kIstdpInhibitoryInputs]{};
#pragma unroll
  for (int input = 0; input < kIstdpInhibitoryInputs; ++input) {
    weights[0][input] = 0.05f;
    weights[1][input] = 0.80f;
  }
  const float trace_decay =
      expf(-1.0f / kIstdpTraceTauMs);

  for (int time = 0; time < kIstdpSteps; ++time) {
    bool inhibitory_spikes[kIstdpInhibitoryInputs];
    float mean_weight[2]{};
#pragma unroll
    for (int input = 0; input < kIstdpInhibitoryInputs; ++input) {
      inhibitory_spikes[input] =
          uniform01(0, 600u + static_cast<std::uint32_t>(input),
                    0, time, 0) < 0.01f;
      mean_weight[0] +=
          weights[0][input] / static_cast<float>(kIstdpInhibitoryInputs);
      mean_weight[1] +=
          weights[1][input] / static_cast<float>(kIstdpInhibitoryInputs);
    }
    bool excitatory_spikes[2];
#pragma unroll
    for (int branch = 0; branch < 2; ++branch) {
      const float rate_hz = 20.0f * expf(-2.0f * mean_weight[branch]);
      excitatory_spikes[branch] =
          uniform01(0, 620u + static_cast<std::uint32_t>(branch),
                    0, time, 0) < rate_hz / 1000.0f;
    }

#pragma unroll
    for (int input = 0; input < kIstdpInhibitoryInputs; ++input) {
      inhibitory_trace[input] =
          inhibitory_trace[input] * trace_decay +
          (inhibitory_spikes[input] ? 1.0f : 0.0f);
    }
#pragma unroll
    for (int branch = 0; branch < 2; ++branch) {
      excitatory_trace[branch] =
          excitatory_trace[branch] * trace_decay +
          (excitatory_spikes[branch] ? 1.0f : 0.0f);
    }

#pragma unroll
    for (int branch = 0; branch < 2; ++branch) {
#pragma unroll
      for (int input = 0; input < kIstdpInhibitoryInputs; ++input) {
        weights[branch][input] =
            ordered_vogels_istdp_update(
                weights[branch][input],
                inhibitory_spikes[input],
                excitatory_spikes[branch],
                inhibitory_trace[input],
                excitatory_trace[branch],
                eta, kIstdpAlpha, 0.0f, 1.0f);
      }
    }

    if (time >= kIstdpMeasurementStart) {
#pragma unroll
      for (int branch = 0; branch < 2; ++branch) {
        measured_spikes[branch] +=
            excitatory_spikes[branch] ? 1.0f : 0.0f;
#pragma unroll
        for (int input = 0; input < kIstdpInhibitoryInputs; ++input) {
          bound_events[branch][input] +=
              (weights[branch][input] <= 0.0f ||
               weights[branch][input] >= 1.0f)
                  ? 1.0f
                  : 0.0f;
        }
      }
    }
  }

  const float measurement_seconds =
      static_cast<float>(kIstdpSteps - kIstdpMeasurementStart) / 1000.0f;
  float bound_fraction[2]{};
#pragma unroll
  for (int branch = 0; branch < 2; ++branch) {
#pragma unroll
    for (int input = 0; input < kIstdpInhibitoryInputs; ++input) {
      bound_fraction[branch] +=
          bound_events[branch][input] /
          (static_cast<float>(kIstdpInhibitoryInputs) *
           static_cast<float>(kIstdpSteps - kIstdpMeasurementStart));
    }
  }
  metrics[candidate].istdp_weak_rate_hz =
      measured_spikes[0] / measurement_seconds;
  metrics[candidate].istdp_strong_rate_hz =
      measured_spikes[1] / measurement_seconds;
  metrics[candidate].istdp_weak_bound_fraction = bound_fraction[0];
  metrics[candidate].istdp_strong_bound_fraction = bound_fraction[1];
}

std::vector<float> logspace(float lower_exponent, float upper_exponent,
                            int count) {
  std::vector<float> values(static_cast<std::size_t>(count));
  if (count == 1) {
    values[0] = std::pow(10.0f, lower_exponent);
    return values;
  }
  for (int index = 0; index < count; ++index) {
    const float fraction =
        static_cast<float>(index) / static_cast<float>(count - 1);
    values[static_cast<std::size_t>(index)] =
        std::pow(10.0f, lower_exponent +
                           fraction * (upper_exponent - lower_exponent));
  }
  return values;
}

std::vector<LearningAssayMetrics> run_learning_grid(
    int device, const Config& config,
    const std::vector<float>& clopath_etas,
    const std::vector<float>& oja_etas,
    const std::vector<float>& istdp_etas) {
  if (clopath_etas.size() != oja_etas.size() ||
      clopath_etas.size() != istdp_etas.size()) {
    throw std::invalid_argument(
        "Learning-rate candidate grids must have one shared size.");
  }
  if (clopath_etas.empty()) {
    return {};
  }
  CLEAN_MSI_CUDA(cudaSetDevice(device));
  const std::size_t count = clopath_etas.size();
  float* device_clopath = nullptr;
  float* device_oja = nullptr;
  float* device_istdp = nullptr;
  LearningAssayMetrics* device_metrics = nullptr;
  try {
    CLEAN_MSI_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_clopath), count * sizeof(float)));
    CLEAN_MSI_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_oja), count * sizeof(float)));
    CLEAN_MSI_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_istdp), count * sizeof(float)));
    CLEAN_MSI_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_metrics),
        count * sizeof(LearningAssayMetrics)));
    CLEAN_MSI_CUDA(cudaMemcpy(
        device_clopath, clopath_etas.data(), count * sizeof(float),
        cudaMemcpyHostToDevice));
    CLEAN_MSI_CUDA(cudaMemcpy(
        device_oja, oja_etas.data(), count * sizeof(float),
        cudaMemcpyHostToDevice));
    CLEAN_MSI_CUDA(cudaMemcpy(
        device_istdp, istdp_etas.data(), count * sizeof(float),
        cudaMemcpyHostToDevice));
    CLEAN_MSI_CUDA(
        cudaMemset(device_metrics, 0,
                   count * sizeof(LearningAssayMetrics)));
    constexpr int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    clopath_assay_kernel<<<blocks, threads>>>(
        device_clopath, device_metrics, static_cast<int>(count));
    CLEAN_MSI_CUDA(cudaGetLastError());
    oja_assay_kernel<<<blocks, threads>>>(
        device_oja, device_metrics, static_cast<int>(count));
    CLEAN_MSI_CUDA(cudaGetLastError());
    istdp_assay_kernel<<<blocks, threads>>>(
        device_istdp, device_metrics, static_cast<int>(count));
    CLEAN_MSI_CUDA(cudaGetLastError());
    std::vector<LearningAssayMetrics> metrics(count);
    CLEAN_MSI_CUDA(cudaMemcpy(
        metrics.data(), device_metrics,
        count * sizeof(LearningAssayMetrics), cudaMemcpyDeviceToHost));
    CLEAN_MSI_CUDA(cudaFree(device_metrics));
    CLEAN_MSI_CUDA(cudaFree(device_istdp));
    CLEAN_MSI_CUDA(cudaFree(device_oja));
    CLEAN_MSI_CUDA(cudaFree(device_clopath));
    return metrics;
  } catch (...) {
    cudaFree(device_metrics);
    cudaFree(device_istdp);
    cudaFree(device_oja);
    cudaFree(device_clopath);
    throw;
  }
}

bool target_reached(AssayKind kind, const AssayMetrics& metrics) {
  switch (kind) {
    case AssayKind::kExternalRs:
      return metrics.primary >= 3.0f;
    case AssayKind::kFeedforwardE:
    case AssayKind::kFeedforwardI:
      return metrics.primary >= 1.0f;
    case AssayKind::kGabaa:
      return metrics.primary >= kGabaaUnitaryMinimumMv;
    case AssayKind::kBackgroundRs:
    case AssayKind::kBackgroundFs:
      return metrics.primary >= 20.0f;
  }
  return false;
}

bool full_predicate(AssayKind kind, const AssayMetrics& metrics) {
  switch (kind) {
    case AssayKind::kExternalRs:
      return metrics.primary == 3.0f && metrics.secondary <= 2.0f &&
             metrics.tertiary <= 8.0f;
    case AssayKind::kFeedforwardE:
      return metrics.primary == 1.0f && metrics.secondary == 0.0f &&
             metrics.tertiary == 2.0f;
    case AssayKind::kFeedforwardI:
      return metrics.primary == 1.0f && metrics.secondary == 0.0f;
    case AssayKind::kGabaa:
      return metrics.primary >= kGabaaUnitaryMinimumMv &&
             metrics.primary <= kGabaaUnitaryMaximumMv &&
             metrics.secondary < 0.0f && metrics.tertiary > 0.0f;
    case AssayKind::kBackgroundRs:
    case AssayKind::kBackgroundFs:
      return metrics.primary == 20.0f;
  }
  return false;
}

struct SearchResult {
  float quantum = 0.0f;
  AssayMetrics metrics{};
};

SearchResult search_gabaa_unitary_ipsp(
    int device, const Config& config) {
  constexpr float previous_quantum = 0.00349115161f;
  constexpr float previous_amplitude_mv = 0.000818f;
  constexpr float estimated_target_quantum =
      previous_quantum * kGabaaUnitaryTargetMv /
      previous_amplitude_mv;
  constexpr float lower_quantum =
      estimated_target_quantum / 10.0f;
  constexpr float upper_quantum =
      estimated_target_quantum * 10.0f;
  const std::vector<AssayMetrics> endpoints =
      evaluate_assay_candidates(
          device, config, AssayKind::kGabaa,
          {lower_quantum, upper_quantum});
  if (target_reached(AssayKind::kGabaa, endpoints[0])) {
    throw std::runtime_error(
        "Unitary GABAA target is already reached at the "
        "derived lower bracket.");
  }
  if (!target_reached(AssayKind::kGabaa, endpoints[1])) {
    throw std::runtime_error(
        "Unitary GABAA target is not reached at the "
        "derived upper bracket.");
  }

  float lower = lower_quantum;
  float upper = upper_quantum;
  AssayMetrics upper_metrics = endpoints[1];
  constexpr int refinement_steps = 28;
  for (int refinement = 0; refinement < refinement_steps;
       ++refinement) {
    const float midpoint =
        std::exp(0.5f * (std::log(lower) + std::log(upper)));
    if (!(midpoint > lower && midpoint < upper)) {
      break;
    }
    const AssayMetrics metrics =
        evaluate_assay_candidates(
            device, config, AssayKind::kGabaa, {midpoint})[0];
    if (target_reached(AssayKind::kGabaa, metrics)) {
      upper = midpoint;
      upper_metrics = metrics;
    } else {
      lower = midpoint;
    }
  }
  if (!full_predicate(AssayKind::kGabaa, upper_metrics)) {
    throw std::runtime_error(
        "Refined unitary GABAA calibration missed the "
        "accepted 0.95-1.05 mV interval.");
  }
  return {upper, upper_metrics};
}

SearchResult refine_assay_bracket(int device, const Config& config,
                                  AssayKind kind, float lower_q,
                                  float upper_q) {
  std::vector<float> refinement(kCalibrationRefinementNodes);
  const float lower_log = std::log(lower_q);
  const float upper_log = std::log(upper_q);
  for (int node = 0; node < kCalibrationRefinementNodes; ++node) {
    const float fraction =
        static_cast<float>(node + 1) /
        static_cast<float>(kCalibrationRefinementNodes + 1);
    refinement[static_cast<std::size_t>(node)] =
        std::exp(lower_log + fraction * (upper_log - lower_log));
  }
  const std::vector<AssayMetrics> refined_metrics =
      evaluate_assay_candidates(device, config, kind, refinement);
  for (int node = 0; node < kCalibrationRefinementNodes; ++node) {
    const AssayMetrics& value =
        refined_metrics[static_cast<std::size_t>(node)];
    if (full_predicate(kind, value)) {
      return {refinement[static_cast<std::size_t>(node)], value};
    }
  }
  return {};
}

SearchResult search_assay(int device, const Config& config,
                          AssayKind kind) {
  constexpr int chunk_size = 4;
  const std::vector<float> coarse =
      logspace(-6.0f, 3.0f, kCalibrationCoarseNodes);
  int crossing_coarse = -1;
  bool previous_reached = false;
  for (int start = 0; start < kCalibrationCoarseNodes;
       start += chunk_size) {
    const int end = std::min(start + chunk_size,
                             kCalibrationCoarseNodes);
    const std::vector<float> chunk(coarse.begin() + start,
                                   coarse.begin() + end);
    const std::vector<AssayMetrics> metrics =
        evaluate_assay_candidates(device, config, kind, chunk);
    for (int local = 0; local < end - start; ++local) {
      const bool reached =
          target_reached(kind, metrics[static_cast<std::size_t>(local)]);
      const int global = start + local;
      if (reached && (global == 0 || !previous_reached)) {
        if (global == 0 && kind != AssayKind::kGabaa) {
          throw std::runtime_error(
              "Calibration lower bound reaches target unexpectedly.");
        }
        crossing_coarse = global;
        break;
      }
      previous_reached = reached;
    }
    if (crossing_coarse >= 0) {
      break;
    }
  }
  if (crossing_coarse < 0) {
    throw std::runtime_error("Calibration target was never reached.");
  }

  const std::vector<float> fine =
      logspace(-6.0f, 3.0f, kCalibrationFineNodes);
  const int fine_start = std::max(0, (crossing_coarse - 1) * 4);
  bool have_previous = false;
  bool previous_fine_reached = false;
  float previous_q = 0.0f;
  bool have_pass = false;
  float preceding_fail_q = 0.0f;
  float first_pass_q = 0.0f;
  AssayMetrics first_pass_metrics{};
  bool have_loss = false;

  for (int start = fine_start;
       start < kCalibrationFineNodes && !have_loss;
       start += chunk_size) {
    const int end =
        std::min(start + chunk_size, kCalibrationFineNodes);
    const std::vector<float> chunk(fine.begin() + start,
                                   fine.begin() + end);
    const std::vector<AssayMetrics> metrics =
        evaluate_assay_candidates(device, config, kind, chunk);
    for (int local = 0; local < end - start; ++local) {
      const float q = chunk[static_cast<std::size_t>(local)];
      const AssayMetrics& value =
          metrics[static_cast<std::size_t>(local)];
      const bool reached = target_reached(kind, value);
      const bool valid = full_predicate(kind, value);
      if (!have_pass && have_previous && !previous_fine_reached &&
          reached && !valid) {
        const SearchResult refined = refine_assay_bracket(
            device, config, kind, previous_q, q);
        if (refined.quantum > 0.0f) {
          return refined;
        }
      }
      if (!have_pass && valid) {
        have_pass = true;
        preceding_fail_q = have_previous ? previous_q : 0.0f;
        first_pass_q = q;
        first_pass_metrics = value;
      } else if (have_pass && !valid) {
        have_loss = true;
        break;
      }
      previous_q = q;
      previous_fine_reached = reached;
      have_previous = true;
    }
  }
  if (!have_pass) {
    throw std::runtime_error(
        "No valid calibration response interval was found.");
  }
  if (!have_loss &&
      calibration_interval_requires_closure(kind)) {
    throw std::runtime_error(
        "First valid calibration interval did not close.");
  }
  if (!(preceding_fail_q > 0.0f)) {
    return {first_pass_q, first_pass_metrics};
  }
  const SearchResult refined = refine_assay_bracket(
      device, config, kind, preceding_fail_q, first_pass_q);
  if (refined.quantum > 0.0f) {
    return refined;
  }
  return {first_pass_q, first_pass_metrics};
}

bool calibration_criteria(const CalibrationResult& result) {
  return full_predicate(AssayKind::kExternalRs, result.external) &&
         full_predicate(AssayKind::kFeedforwardE,
                        result.feedforward_e) &&
         full_predicate(AssayKind::kFeedforwardI,
                        result.feedforward_i) &&
         full_predicate(AssayKind::kGabaa, result.gabaa) &&
         full_predicate(AssayKind::kBackgroundRs,
                        result.background_rs) &&
         full_predicate(AssayKind::kBackgroundFs,
                        result.background_fs) &&
         result.gabaa_efficacy.unitary.contact_count == 1 &&
         result.gabaa_efficacy.unitary.amplitude_mv >=
             kGabaaUnitaryMinimumMv &&
         result.gabaa_efficacy.unitary.amplitude_mv <=
             kGabaaUnitaryMaximumMv &&
         result.gabaa_efficacy.unitary.signed_nadir_mv < 0.0f &&
         result.gabaa_efficacy.unitary.area_mv_ms > 0.0f &&
         result.gabaa_efficacy.unitary.outward_charge > 0.0f &&
         result.gabaa_efficacy.compound.contact_count ==
             kInhibitoryScaffoldInDegree &&
         result.gabaa_efficacy.compound.amplitude_mv >= 3.0f &&
         result.gabaa_efficacy.compound.signed_nadir_mv < 0.0f &&
         result.gabaa_efficacy.compound.area_mv_ms >
             result.gabaa_efficacy.unitary.area_mv_ms &&
         result.gabaa_efficacy.compound.outward_charge >
             result.gabaa_efficacy.unitary.outward_charge &&
         result.gabaa_relay.relay_spikes > 0 &&
         result.gabaa_relay.gabaa_arrivals > 0 &&
         result.gabaa_relay.lag_count > 0 &&
         result.gabaa_relay.minimum_lag_ms >=
             kGabaaUnitaryDelayMs + 3 &&
         result.gabaa_relay.without_gabaa_spikes >
             result.gabaa_relay.with_gabaa_spikes &&
         std::fabs(result.learning.clopath_delta - 0.25f) <= 0.01f &&
         result.learning.clopath_shuffled_drift <= 0.05f &&
         result.learning.oja_selection_ratio >= 2.0f &&
         result.learning.oja_bound_fraction < 0.05f &&
         result.config.eta_istdp == 1.0e-4f &&
         std::isfinite(result.learning.istdp_weak_rate_hz) &&
         result.learning.istdp_weak_rate_hz >= 0.0f &&
         std::isfinite(result.learning.istdp_strong_rate_hz) &&
         result.learning.istdp_strong_rate_hz >= 0.0f &&
         std::isfinite(
             result.learning.istdp_weak_bound_fraction) &&
         result.learning.istdp_weak_bound_fraction >= 0.0f &&
         result.learning.istdp_weak_bound_fraction < 0.10f &&
         std::isfinite(
             result.learning.istdp_strong_bound_fraction) &&
         result.learning.istdp_strong_bound_fraction >= 0.0f &&
         result.learning.istdp_strong_bound_fraction < 0.10f;
}

}  // namespace

LearningAssayMetrics evaluate_learning_candidates(
    int device, const Config& config, float eta_clopath, float eta_oja,
    float eta_istdp) {
  return run_learning_grid(
      device, config, {eta_clopath}, {eta_oja}, {eta_istdp})[0];
}

CalibrationResult calibrate(int device, const Config& base_config) {
  CLEAN_MSI_CUDA(cudaSetDevice(device));
  CalibrationResult result{};
  result.config = base_config;
  result.config.calibrated = false;

  const auto conductance_start = std::chrono::steady_clock::now();
  const SearchResult external =
      search_assay(device, result.config, AssayKind::kExternalRs);
  result.config.q_external_rs = external.quantum;
  const SearchResult feedforward_e =
      search_assay(device, result.config, AssayKind::kFeedforwardE);
  result.config.q_ff_e = feedforward_e.quantum;
  const SearchResult feedforward_i =
      search_assay(device, result.config, AssayKind::kFeedforwardI);
  result.config.q_ff_i = feedforward_i.quantum;
  const SearchResult gabaa =
      search_gabaa_unitary_ipsp(device, result.config);
  result.config.q_gabaa = gabaa.quantum;
  const SearchResult background_rs =
      search_assay(device, result.config, AssayKind::kBackgroundRs);
  result.config.q_background_rs = background_rs.quantum;
  const SearchResult background_fs =
      search_assay(device, result.config, AssayKind::kBackgroundFs);
  result.config.q_background_fs = background_fs.quantum;

  result.external =
      evaluate_assay_candidates(
          device, result.config, AssayKind::kExternalRs,
          {result.config.q_external_rs})[0];
  result.feedforward_e =
      evaluate_assay_candidates(
          device, result.config, AssayKind::kFeedforwardE,
          {result.config.q_ff_e})[0];
  result.feedforward_i =
      evaluate_assay_candidates(
          device, result.config, AssayKind::kFeedforwardI,
          {result.config.q_ff_i})[0];
  result.gabaa =
      evaluate_assay_candidates(
          device, result.config, AssayKind::kGabaa,
          {result.config.q_gabaa})[0];
  result.gabaa_efficacy =
      audit_gabaa_efficacy(
          device, result.config, result.config.q_gabaa);
  result.gabaa_relay =
      audit_gabaa_relay_timing(
          device, result.config, result.config.q_gabaa);
  result.background_rs =
      evaluate_assay_candidates(
          device, result.config, AssayKind::kBackgroundRs,
          {result.config.q_background_rs})[0];
  result.background_fs =
      evaluate_assay_candidates(
          device, result.config, AssayKind::kBackgroundFs,
          {result.config.q_background_fs})[0];
  const auto conductance_end = std::chrono::steady_clock::now();
  result.conductance_seconds =
      std::chrono::duration<float>(conductance_end - conductance_start)
          .count();

  const auto learning_start = std::chrono::steady_clock::now();
  const std::vector<float> clopath_grid =
      logspace(-4.0f, -2.0f, kLearningRateCandidates);
  const std::vector<float> oja_grid =
      logspace(-6.0f, -1.0f, kLearningRateCandidates);
  const std::vector<float> istdp_grid =
      logspace(-5.0f, 0.0f, kLearningRateCandidates);
  static_assert(kLearningRateCandidates == 81);
  constexpr int istdp_index = 16;
  assert(
      istdp_grid[static_cast<std::size_t>(istdp_index)] ==
      1.0e-4f);
  const std::vector<LearningAssayMetrics> learning_grid =
      run_learning_grid(device, result.config, clopath_grid, oja_grid,
                        istdp_grid);

  int clopath_index = -1;
  float clopath_error = FLT_MAX;
  int oja_index = -1;
  for (int index = 0; index < kLearningRateCandidates; ++index) {
    const LearningAssayMetrics& metrics =
        learning_grid[static_cast<std::size_t>(index)];
    const float candidate_clopath_error =
        std::fabs(metrics.clopath_delta - 0.25f);
    const float clopath_final = 0.25f + metrics.clopath_delta;
    if (metrics.clopath_shuffled_drift <= 0.05f &&
        clopath_final > 0.0f && clopath_final < 1.0f &&
        candidate_clopath_error < clopath_error) {
      clopath_error = candidate_clopath_error;
      clopath_index = index;
    }
    if (oja_index < 0 && metrics.oja_selection_ratio >= 2.0f &&
        metrics.oja_bound_fraction < 0.05f) {
      oja_index = index;
    }
  }
  if (clopath_index < 0 || clopath_error > 0.01f) {
    throw std::runtime_error(
        "No Clopath learning-rate candidate passed.");
  }
  if (oja_index < 0) {
    throw std::runtime_error(
        "No Oja learning-rate candidate passed.");
  }
  result.config.eta_clopath_ff =
      clopath_grid[static_cast<std::size_t>(clopath_index)];
  result.config.eta_oja =
      oja_grid[static_cast<std::size_t>(oja_index)];
  result.config.eta_istdp =
      istdp_grid[static_cast<std::size_t>(istdp_index)];
  result.learning.clopath_delta =
      learning_grid[static_cast<std::size_t>(clopath_index)]
          .clopath_delta;
  result.learning.clopath_shuffled_drift =
      learning_grid[static_cast<std::size_t>(clopath_index)]
          .clopath_shuffled_drift;
  result.learning.oja_selection_ratio =
      learning_grid[static_cast<std::size_t>(oja_index)]
          .oja_selection_ratio;
  result.learning.oja_bound_fraction =
      learning_grid[static_cast<std::size_t>(oja_index)]
          .oja_bound_fraction;
  result.learning.istdp_weak_rate_hz =
      learning_grid[static_cast<std::size_t>(istdp_index)]
          .istdp_weak_rate_hz;
  result.learning.istdp_strong_rate_hz =
      learning_grid[static_cast<std::size_t>(istdp_index)]
          .istdp_strong_rate_hz;
  result.learning.istdp_weak_bound_fraction =
      learning_grid[static_cast<std::size_t>(istdp_index)]
          .istdp_weak_bound_fraction;
  result.learning.istdp_strong_bound_fraction =
      learning_grid[static_cast<std::size_t>(istdp_index)]
          .istdp_strong_bound_fraction;
  const auto learning_end = std::chrono::steady_clock::now();
  result.learning_seconds =
      std::chrono::duration<float>(learning_end - learning_start).count();
  result.config.calibrated = true;
  result.criteria_passed = calibration_criteria(result);
  if (!result.criteria_passed) {
    throw std::runtime_error(
        "Calibrated parameters failed the final scientific criteria.");
  }
  return result;
}

namespace {

constexpr int kBlocksPerTrainingSeed = 20;
constexpr int kTrainingThreads = 256;
constexpr int kTotalNeurons = 600;
constexpr int kAOffset = 0;
constexpr int kVOffset = 180;
constexpr int kEOffset = 360;
constexpr int kIOffset = 540;
constexpr int kReceptorCount = 3;
constexpr int kPathCount = 7;
constexpr int kWeightCount = 140400;
constexpr int kClopathPathCount = 5;
constexpr int kClopathContactCount = 118800;
constexpr int kClopathPostsynapticNeurons =
    kExcitatoryNeurons + kInhibitoryNeurons;
constexpr double kAuditoryScaffoldSigmaDeg =
    kAuditoryScaffoldFwhmDeg / kFwhmToSigmaDivisor;
constexpr double kVisualScaffoldSigmaDeg =
    kVisualScaffoldFwhmDeg / kFwhmToSigmaDivisor;
constexpr double kMinimumScaffoldCoordinateDeg = -89.5;
constexpr double kScaffoldCoordinateSpanDeg = 179.0;

struct PathLayout {
  int weight_offset;
  int pre_offset;
  int pre_size;
  int post_offset;
  int post_size;
  int delay_steps;
  float upper_bound;
  float fixed_nmda_ratio;
  bool prunable;
};

__host__ __device__ inline int population_offset(int population) {
  return population == 0 ? kAOffset
       : population == 1 ? kVOffset
       : population == 2 ? kEOffset
                         : kIOffset;
}

__host__ __device__ inline int population_size(int population) {
  return population == 3 ? kInhibitoryNeurons : kExcitatoryNeurons;
}

__host__ __device__ inline PathLayout path_layout(int path) {
  switch (path) {
    case 0:
      return {0, kAOffset, 180, kEOffset, 180, 3, 1.0f, 1.0f, true};
    case 1:
      return {32400, kVOffset, 180, kEOffset, 180, 3, 1.0f, 1.0f, true};
    case 2:
      return {64800, kEOffset, 180, kEOffset, 180, 2, 0.5f, 1.0f, true};
    case 3:
      return {97200, kAOffset, 180, kIOffset, 60, 3, 1.0f, 1.0f, true};
    case 4:
      return {108000, kVOffset, 180, kIOffset, 60, 3, 1.0f, 1.0f, true};
    case 5:
      return {118800, kEOffset, 180, kIOffset, 60, 2, 0.5f, 1.0f, true};
    default:
      return {129600, kIOffset, 60, kEOffset, 180, 1, 1.0f, 0.0f, false};
  }
}

__host__ __device__ inline int path_contact_count(int path) {
  const PathLayout layout = path_layout(path);
  return layout.pre_size * layout.post_size;
}

struct PresentationSpec {
  int kind = 0;
  float auditory_latent_deg = 0.0f;
  float visual_latent_deg = 0.0f;
  float auditory_noise_deg = 0.0f;
  float visual_noise_deg = 0.0f;
  float auditory_azimuth_deg = 0.0f;
  float visual_azimuth_deg = 0.0f;
  float auditory_salience_hz = 0.0f;
  float visual_salience_hz = 0.0f;
  float physical_soa_ms = 0.0f;
  float auditory_latency_ms = 0.0f;
  float visual_latency_ms = 0.0f;
  float auditory_profile_peak = 1.0f;
  float visual_profile_peak = 1.0f;
  float visual_population_mass_scale = 1.0f;
  int auditory_onset_ms = 0;
  int visual_onset_ms = 0;
  int duration_ms = 0;
  int silent_iti_start_ms = 0;
  bool auditory_active = false;
  bool visual_active = false;
};

struct DeviceTrainingState {
  Config config{};
  int seed_count = 0;
  float* neuron_a = nullptr;
  float* neuron_b = nullptr;
  float* neuron_c = nullptr;
  float* neuron_d = nullptr;
  float* voltage = nullptr;
  float* recovery = nullptr;
  float* pre_reset_voltage = nullptr;
  float* receptor_rise = nullptr;
  float* receptor_decay = nullptr;
  float* conductance = nullptr;
  std::uint8_t* spikes = nullptr;
  std::uint8_t* spike_ring = nullptr;
  float* weights = nullptr;
  float* initial_weights = nullptr;
  std::uint8_t* masks = nullptr;
  std::uint8_t* changed = nullptr;
  int* low_weight_dwell = nullptr;
  float* clopath_contact_pre = nullptr;
  float* clopath_voltage_minus = nullptr;
  float* clopath_voltage_plus = nullptr;
  float* clopath_homeostasis = nullptr;
  float* clopath_rest_voltage = nullptr;
  float* oja_pre = nullptr;
  float* oja_post = nullptr;
  float* istdp_pre = nullptr;
  float* istdp_post = nullptr;
  PresentationSpec* presentations = nullptr;
  int* presentation_index = nullptr;
  std::uint64_t* accepted_steps = nullptr;
  std::uint64_t* silent_iti_steps = nullptr;
  std::uint64_t* silent_iti_afferent_arrivals = nullptr;
  std::uint64_t* recurrent_plasticity_steps = nullptr;
  std::uint64_t* population_spikes = nullptr;
  std::uint64_t* scheduled_spikes = nullptr;
  std::uint64_t* arrived_spikes = nullptr;
  int* pruned_contacts = nullptr;
  unsigned int* seed_barrier_count = nullptr;
  unsigned int* seed_barrier_epoch = nullptr;
};

__device__ inline float normal_draw(std::uint64_t seed,
                                    std::uint32_t stream,
                                    std::uint64_t entity,
                                    std::uint32_t time,
                                    std::uint32_t draw_pair) {
  const float first =
      fmaxf(uniform01(seed, stream, entity, time, 2u * draw_pair),
            1.0e-12f);
  const float second =
      uniform01(seed, stream, entity, time, 2u * draw_pair + 1u);
  return sqrtf(-2.0f * logf(first)) *
         cosf(6.2831853071795864769f * second);
}

__host__ __device__ inline float reflected_azimuth(float value) {
  while (value < -90.0f || value > 90.0f) {
    if (value < -90.0f) {
      value = -180.0f - value;
    }
    if (value > 90.0f) {
      value = 180.0f - value;
    }
  }
  return value;
}

__host__ __device__ inline float reflected_profile_raw(
    float center_deg, float sigma_deg, float coordinate_deg) {
  const float center = reflected_azimuth(center_deg);
  float value = 0.0f;
  for (int period = -1; period <= 1; ++period) {
    const float direct =
        center + 360.0f * static_cast<float>(period);
    const float reflected =
        -180.0f - center + 360.0f * static_cast<float>(period);
    const float direct_z = (coordinate_deg - direct) / sigma_deg;
    const float reflected_z =
        (coordinate_deg - reflected) / sigma_deg;
    value += expf(-0.5f * direct_z * direct_z);
    value += expf(-0.5f * reflected_z * reflected_z);
  }
  return value;
}

__host__ __device__ inline float controlled_profile_raw(
    float center_deg, float sigma_deg, float coordinate_deg) {
  const float z = (coordinate_deg - center_deg) / sigma_deg;
  return expf(-0.5f * z * z);
}

__device__ inline float discrete_reflected_profile_peak(
    float center_deg, float sigma_deg) {
  float peak = 0.0f;
  for (int neuron = 0; neuron < kAuditoryNeurons; ++neuron) {
    const float coordinate = -89.5f + static_cast<float>(neuron);
    peak = fmaxf(
        peak, reflected_profile_raw(center_deg, sigma_deg, coordinate));
  }
  return fmaxf(peak, 1.0e-20f);
}

__host__ __device__ inline float discrete_controlled_profile_peak(
    float center_deg, float sigma_deg) {
  float peak = 0.0f;
  for (int neuron = 0; neuron < kAuditoryNeurons; ++neuron) {
    const float coordinate = -89.5f + static_cast<float>(neuron);
    peak = fmaxf(
        peak, controlled_profile_raw(center_deg, sigma_deg, coordinate));
  }
  return fmaxf(peak, 1.0e-20f);
}

__host__ __device__ inline float discrete_visual_population_mass_scale(
    float center_deg, float auditory_peak, float visual_peak,
    bool reflect_edges) {
  float auditory_mass = 0.0f;
  float visual_mass = 0.0f;
  for (int neuron = 0; neuron < kAuditoryNeurons; ++neuron) {
    const float coordinate = -89.5f + static_cast<float>(neuron);
    if (reflect_edges) {
      auditory_mass +=
          reflected_profile_raw(center_deg, 8.0f, coordinate) /
          auditory_peak;
      visual_mass +=
          reflected_profile_raw(center_deg, 2.0f, coordinate) /
          visual_peak;
    } else {
      auditory_mass +=
          controlled_profile_raw(center_deg, 8.0f, coordinate) /
          auditory_peak;
      visual_mass +=
          controlled_profile_raw(center_deg, 2.0f, coordinate) /
          visual_peak;
    }
  }
  return auditory_mass / fmaxf(visual_mass, 1.0e-20f);
}

__device__ inline float positive_normal(
    std::uint64_t seed, std::uint32_t stream,
    std::uint64_t presentation, float mean, float standard_deviation) {
  for (std::uint32_t attempt = 0; attempt < 64; ++attempt) {
    const float value =
        mean + standard_deviation *
                   normal_draw(seed, stream, presentation, 0, attempt);
    if (value > 0.0f) {
      return value;
    }
  }
  return fmaxf(mean, 1.0f);
}

__device__ inline float truncated_common_soa(
    std::uint64_t seed, std::uint64_t presentation) {
  for (std::uint32_t attempt = 0; attempt < 64; ++attempt) {
    const float value =
        -50.0f + 60.0f *
                     normal_draw(seed, 730u, presentation, 0, attempt);
    if (value >= -250.0f && value <= 250.0f) {
      return value;
    }
  }
  return -50.0f;
}

__host__ __device__ inline float jitter_multiplier(
    JitterScale scale) {
  switch (scale) {
    case JitterScale::kZero:
      return 0.0f;
    case JitterScale::kHalf:
      return 0.5f;
    case JitterScale::kDouble:
      return 2.0f;
    default:
      return 1.0f;
  }
}

__device__ PresentationSpec generate_presentation(
    const Config& config, std::uint64_t seed,
    int presentation_number) {
  const std::uint64_t presentation =
      static_cast<std::uint64_t>(presentation_number);
  PresentationSpec spec{};
  const float kind_draw =
      uniform01(seed, 700u, presentation, 0, 0);
  spec.kind = kind_draw < 0.40f ? 0
            : kind_draw < 0.60f ? 1
            : kind_draw < 0.80f ? 2
                                : 3;
  spec.auditory_active = spec.kind != 3;
  spec.visual_active = spec.kind != 2;
  spec.auditory_onset_ms = -1;
  spec.visual_onset_ms = -1;

  if (spec.kind == 0) {
    const float shared_latent =
        -60.0f + 120.0f *
                        uniform01(seed, 701u, presentation, 0, 0);
    if (config.training_cohort == TrainingCohort::kSpatialShuffle) {
      spec.auditory_latent_deg = shared_latent;
      spec.visual_latent_deg =
          -60.0f + 120.0f *
                          uniform01(seed, 705u, presentation, 0, 0);
    } else if (
        config.training_cohort == TrainingCohort::kFixedOffset) {
      spec.auditory_latent_deg =
          shared_latent + 0.5f * config.fixed_offset_deg;
      spec.visual_latent_deg =
          shared_latent - 0.5f * config.fixed_offset_deg;
    } else {
      spec.auditory_latent_deg = shared_latent;
      spec.visual_latent_deg = shared_latent;
    }
  } else {
    if (spec.auditory_active) {
      spec.auditory_latent_deg =
          -60.0f + 120.0f *
                          uniform01(seed, 704u, presentation, 0, 0);
    }
    if (spec.visual_active) {
      spec.visual_latent_deg =
          -60.0f + 120.0f *
                          uniform01(seed, 705u, presentation, 0, 0);
    }
  }
  const float jitter = jitter_multiplier(config.jitter_scale);
  if (spec.auditory_active) {
    spec.auditory_noise_deg =
        jitter * 8.0f *
        normal_draw(seed, 702u, presentation, 0, 0);
    spec.auditory_azimuth_deg = reflected_azimuth(
        spec.auditory_latent_deg + spec.auditory_noise_deg);
    spec.auditory_profile_peak = discrete_reflected_profile_peak(
        spec.auditory_azimuth_deg, 8.0f);
  }
  if (spec.visual_active) {
    spec.visual_noise_deg =
        jitter * 2.0f *
        normal_draw(seed, 703u, presentation, 0, 0);
    spec.visual_azimuth_deg = reflected_azimuth(
        spec.visual_latent_deg + spec.visual_noise_deg);
    spec.visual_profile_peak = discrete_reflected_profile_peak(
        spec.visual_azimuth_deg, 2.0f);
    const float auditory_reference_peak =
        discrete_reflected_profile_peak(
            spec.visual_azimuth_deg, 8.0f);
    spec.visual_population_mass_scale =
        discrete_visual_population_mass_scale(
            spec.visual_azimuth_deg, auditory_reference_peak,
            spec.visual_profile_peak, true);
  }

  const float log_lower = logf(25.0f);
  const float log_upper = logf(100.0f);
  const float common_salience =
      expf(log_lower +
           (log_upper - log_lower) *
               uniform01(seed, 706u, presentation, 0, 0));
  if (spec.kind == 0) {
    spec.auditory_salience_hz = common_salience;
    spec.visual_salience_hz = common_salience;
  } else {
    if (spec.auditory_active) {
      spec.auditory_salience_hz =
          expf(log_lower +
               (log_upper - log_lower) *
                   uniform01(seed, 707u, presentation, 0, 0));
    }
    if (spec.visual_active) {
      spec.visual_salience_hz =
          expf(log_lower +
               (log_upper - log_lower) *
                   uniform01(seed, 708u, presentation, 0, 0));
    }
  }

  if (spec.kind == 0) {
    spec.physical_soa_ms =
        truncated_common_soa(seed, presentation);
  } else if (spec.kind == 1) {
    spec.physical_soa_ms =
        -600.0f + 1200.0f *
                        uniform01(seed, 709u, presentation, 0, 0);
  }
  if (spec.auditory_active) {
    spec.auditory_latency_ms =
        positive_normal(seed, 710u, presentation, 21.0f, 5.0f);
  }
  if (spec.visual_active) {
    spec.visual_latency_ms =
        positive_normal(seed, 711u, presentation, 69.0f, 10.0f);
  }
  const float raw_a =
      spec.auditory_active ? spec.auditory_latency_ms : FLT_MAX;
  const float raw_v =
      spec.visual_active
          ? spec.physical_soa_ms + spec.visual_latency_ms
          : FLT_MAX;
  const float earliest = fminf(raw_a, raw_v);
  if (spec.auditory_active) {
    spec.auditory_onset_ms =
        static_cast<int>(lrintf(100.0f + raw_a - earliest));
  }
  if (spec.visual_active) {
    spec.visual_onset_ms =
        static_cast<int>(lrintf(100.0f + raw_v - earliest));
  }
  const int latest_onset = max(spec.auditory_onset_ms,
                               spec.visual_onset_ms);
  spec.silent_iti_start_ms = latest_onset + 50 + 250;
  spec.duration_ms = spec.silent_iti_start_ms + 1 + 250;
  return spec;
}

__global__ void developmental_audit_kernel(
    Config config, std::uint64_t seed,
    DevelopmentalSampleAudit* samples,
    int sample_count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= sample_count) {
    return;
  }
  const PresentationSpec spec =
      generate_presentation(config, seed, index + 1);
  DevelopmentalSampleAudit sample{};
  sample.kind = static_cast<PresentationKind>(spec.kind);
  sample.auditory_present = spec.auditory_active;
  sample.visual_present = spec.visual_active;
  sample.auditory_latent_deg = spec.auditory_latent_deg;
  sample.visual_latent_deg = spec.visual_latent_deg;
  sample.auditory_noise_deg = spec.auditory_noise_deg;
  sample.visual_noise_deg = spec.visual_noise_deg;
  sample.auditory_location_deg = spec.auditory_azimuth_deg;
  sample.visual_location_deg = spec.visual_azimuth_deg;
  sample.auditory_rate_hz = spec.auditory_salience_hz;
  sample.visual_rate_hz = spec.visual_salience_hz;
  sample.physical_soa_ms = spec.physical_soa_ms;
  sample.auditory_latency_ms = spec.auditory_latency_ms;
  sample.visual_latency_ms = spec.visual_latency_ms;
  sample.auditory_onset_ms = spec.auditory_onset_ms;
  sample.visual_onset_ms = spec.visual_onset_ms;
  sample.poststimulus_end_ms = spec.silent_iti_start_ms;
  sample.first_zero_afferent_receptor_step =
      spec.silent_iti_start_ms + 1;
  sample.valid_end_ms = spec.duration_ms;
  samples[index] = sample;
}

__global__ void spatial_profile_audit_kernel(
    float center_deg, float sigma_deg, float* developmental,
    float* controlled) {
  __shared__ float developmental_values[kTrainingThreads];
  __shared__ float controlled_values[kTrainingThreads];
  const int index = threadIdx.x;
  float developmental_value = 0.0f;
  float controlled_value = 0.0f;
  if (index < kAuditoryNeurons) {
    const float coordinate = -89.5f + static_cast<float>(index);
    developmental_value =
        reflected_profile_raw(center_deg, sigma_deg, coordinate);
    controlled_value =
        controlled_profile_raw(center_deg, sigma_deg, coordinate);
  }
  developmental_values[index] = developmental_value;
  controlled_values[index] = controlled_value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (index < stride) {
      developmental_values[index] =
          fmaxf(developmental_values[index],
                developmental_values[index + stride]);
      controlled_values[index] =
          fmaxf(controlled_values[index],
                controlled_values[index + stride]);
    }
    __syncthreads();
  }
  if (index < kAuditoryNeurons) {
    developmental[index] =
        developmental_value / fmaxf(developmental_values[0], 1.0e-20f);
    controlled[index] =
        controlled_value / fmaxf(controlled_values[0], 1.0e-20f);
  }
}

struct EvaluationConditionSpec {
  ControlledCondition condition{};
  float auditory_profile_peak = 1.0f;
  float visual_profile_peak = 1.0f;
  float visual_population_mass_scale = 1.0f;
};

struct EvaluationTrialMetadata {
  int seed_index = 0;
  int condition_index = 0;
  int trial_index = 0;
  int label = 0;
  float physical_soa_ms = 0.0f;
  float auditory_latency_ms = 0.0f;
  float visual_latency_ms = 0.0f;
  int auditory_received_onset_ms = -1;
  int visual_received_onset_ms = -1;
  int earlier_received_onset_step = 0;
  int burn_in_steps = 0;
  unsigned long long background_afferent_arrivals_during_burn_in = 0;
  float baseline_rate_hz = 0.0f;
  float response_rate_hz = 0.0f;
  int finite = 1;
};

struct DeviceEvaluationWorkspace {
  float* voltage = nullptr;
  float* recovery = nullptr;
  float* receptor_rise = nullptr;
  float* receptor_decay = nullptr;
  float* conductance = nullptr;
  std::uint8_t* spikes = nullptr;
  std::uint8_t* spike_ring = nullptr;
  unsigned int* excitatory_spikes_per_relative_ms = nullptr;
  std::uint16_t* excitatory_baseline_spikes_per_neuron = nullptr;
  std::uint16_t* excitatory_response_spikes_per_neuron = nullptr;
  float* features = nullptr;
  EvaluationTrialMetadata* metadata = nullptr;
};

__device__ inline float evaluation_positive_normal(
    std::uint64_t seed, std::uint32_t stream, std::uint64_t entity,
    float mean, float standard_deviation) {
  for (std::uint32_t attempt = 0; attempt < 64; ++attempt) {
    const float value =
        mean + standard_deviation *
                   normal_draw(seed, stream, entity, 0, attempt);
    if (value > 0.0f) {
      return value;
    }
  }
  return fmaxf(mean, 1.0f);
}

__device__ inline std::uint8_t evaluation_delayed_spike(
    const DeviceEvaluationWorkspace& workspace, int trial_index,
    int neuron, int delay_steps, int step) {
  if (step < delay_steps) {
    return 0;
  }
  const int slot = (step - delay_steps) % kDelayRing;
  return workspace.spike_ring[
      (trial_index * kDelayRing + slot) * kTotalNeurons + neuron];
}

__device__ void evaluation_recurrent_arrivals(
    const DeviceTrainingState& state,
    const DeviceEvaluationWorkspace& workspace, int trial_index,
    int seed_index, int target_population, int post, int step,
    CausalControl control, bool use_initial_weights,
    float* ampa_arrival, float* nmda_arrival,
    float* gabaa_arrival) {
  *ampa_arrival = 0.0f;
  *nmda_arrival = 0.0f;
  *gabaa_arrival = 0.0f;
  const int first_path = target_population == 2 ? 0 : 3;
  const int final_path = target_population == 2 ? 7 : 6;
  const int weight_base = seed_index * kWeightCount;
  for (int path = first_path; path < final_path; ++path) {
    if (target_population == 2 &&
        (path == 3 || path == 4 || path == 5)) {
      continue;
    }
    if (target_population == 3 && path == 6) {
      continue;
    }
    if ((path == 2 &&
         has_control(
             control, CausalControl::kRecurrentExcitationOff)) ||
        ((path == 3 || path == 4 || path == 6) &&
         has_control(
             control, CausalControl::kRecruitedInhibitionOff)) ||
        (path == 6 &&
         has_control(control, CausalControl::kGabaaOff))) {
      continue;
    }
    const PathLayout layout = path_layout(path);
    if (post >= layout.post_size) {
      continue;
    }
    float path_ampa = 0.0f;
    float path_nmda = 0.0f;
    float path_gabaa = 0.0f;
    for (int pre = 0; pre < layout.pre_size; ++pre) {
      if (!evaluation_delayed_spike(
              workspace, trial_index, layout.pre_offset + pre,
              layout.delay_steps, step)) {
        continue;
      }
      int weight_pre = pre;
      if ((path == 0 || path == 1 || path == 3 || path == 4) &&
          has_control(control, CausalControl::kSourceRowShuffle)) {
        weight_pre =
            (73 * pre + 37 * (path + 1)) % kAuditoryNeurons;
      }
      const int local = weight_pre * layout.post_size + post;
      const int index = weight_base + layout.weight_offset + local;
      const float weight =
          use_initial_weights ? state.initial_weights[index]
                              : state.weights[index];
      const bool active =
          use_initial_weights ? weight > 0.0f
                              : state.masks[index] != 0;
      if (!active) {
        continue;
      }
      if (path == 6) {
        path_gabaa += weight;
      } else {
        path_ampa += weight;
        path_nmda +=
            shared_nmda_contact_efficacy(
                weight, layout.fixed_nmda_ratio, active);
      }
    }
    if (path == 6) {
      *gabaa_arrival += state.config.q_gabaa * path_gabaa;
    } else {
      const float quantum =
          target_population == 2 ? state.config.q_ff_e
                                 : state.config.q_ff_i;
      *ampa_arrival += quantum * path_ampa;
      *nmda_arrival +=
          nmda_quantum_from_ampa(quantum) * path_nmda;
    }
  }
}

__device__ inline int evaluation_sensory_afferent_count(
    const DeviceTrainingState& state,
    const EvaluationConditionSpec& condition, int model_seed_index,
    int random_group, int trial_in_condition, int modality, int neuron,
    int source_time, int source_onset, std::uint64_t evaluation_seed) {
  if (source_time < source_onset || source_time >= source_onset + 50) {
    return 0;
  }
  const bool present =
      modality == 0 ? condition.condition.auditory_present
                    : condition.condition.visual_present;
  if (!present) {
    return 0;
  }
  const float center =
      modality == 0 ? condition.condition.auditory_location_deg
                    : condition.condition.visual_location_deg;
  const float rate =
      modality == 0 ? condition.condition.auditory_rate_hz
                    : condition.condition.visual_rate_hz;
  const float sigma = modality == 0 ? 8.0f : 2.0f;
  const float peak =
      modality == 0 ? condition.auditory_profile_peak
                    : condition.visual_profile_peak;
  const float population_mass_scale =
      modality == 0 ? 1.0f
                    : condition.visual_population_mass_scale;
  const float coordinate = -89.5f + static_cast<float>(neuron);
  const float rate_hz =
      rate * population_mass_scale *
      controlled_profile_raw(center, sigma, coordinate) / peak;
  const std::uint64_t model_seed =
      state.config.seed +
      static_cast<std::uint64_t>(model_seed_index);
  const std::uint64_t group_entity =
      model_seed * 0xD2B74407B1CE6E93ull +
      static_cast<std::uint64_t>(random_group + 1) *
          0x9E3779B97F4A7C15ull +
      static_cast<std::uint64_t>(trial_in_condition);
  const std::uint64_t entity =
      group_entity * 512ull + static_cast<std::uint64_t>(neuron);
  return poisson_small(
      rate_hz * state.config.dt_ms / 1000.0f, evaluation_seed,
      1200u + static_cast<std::uint32_t>(modality), entity,
      static_cast<std::uint32_t>(source_time));
}

__device__ inline int evaluation_background_afferent_count(
    const DeviceTrainingState& state, int model_seed_index,
    int random_group, int trial_in_condition, int population, int neuron,
    int source_time, std::uint64_t evaluation_seed) {
  if (source_time < 0) {
    return 0;
  }
  const std::uint64_t model_seed =
      state.config.seed +
      static_cast<std::uint64_t>(model_seed_index);
  const std::uint64_t group_entity =
      model_seed * 0xD2B74407B1CE6E93ull +
      static_cast<std::uint64_t>(random_group + 1) *
          0x9E3779B97F4A7C15ull +
      static_cast<std::uint64_t>(trial_in_condition);
  const std::uint64_t entity =
      group_entity * 1024ull +
      static_cast<std::uint64_t>(population * 256 + neuron);
  return poisson_small(
      50.0f * state.config.dt_ms / 1000.0f, evaluation_seed,
      1220u + static_cast<std::uint32_t>(population), entity,
      static_cast<std::uint32_t>(source_time));
}

__global__ __launch_bounds__(kTrainingThreads, 1)
void frozen_evaluation_kernel(
    DeviceTrainingState state,
    const EvaluationConditionSpec* conditions, int condition_count,
    int trials_per_condition, int burn_in_steps,
    std::uint64_t evaluation_seed,
    DeviceEvaluationWorkspace workspace) {
  const int flat_trial = blockIdx.x;
  const int trials_per_seed = condition_count * trials_per_condition;
  const int seed_index = flat_trial / trials_per_seed;
  if (seed_index >= state.seed_count) {
    return;
  }
  const int within_seed = flat_trial % trials_per_seed;
  const int condition_index = within_seed / trials_per_condition;
  const int trial_in_condition = within_seed % trials_per_condition;
  const EvaluationConditionSpec condition =
      conditions[condition_index];
  const int random_group =
      condition.condition.random_group >= 0
          ? condition.condition.random_group
          : condition_index;
  const int neuron_base = flat_trial * kTotalNeurons;
  const int receptor_base =
      flat_trial * kReceptorCount * kTotalNeurons;
  const int trained_neuron_base = seed_index * kTotalNeurons;
  const int output_ms_base = flat_trial * kEvaluationRelativeMs;
  const int output_neuron_base =
      flat_trial * kExcitatoryNeurons;
  const int feature_base =
      flat_trial * kEvaluationFeatureBins;
  EvaluationTrialMetadata* metadata =
      &workspace.metadata[flat_trial];

  __shared__ int auditory_source_onset;
  __shared__ int visual_source_onset;
  __shared__ int earlier_received_step;
  __shared__ float auditory_latency_ms;
  __shared__ float visual_latency_ms;

  for (int neuron = threadIdx.x; neuron < kTotalNeurons;
       neuron += blockDim.x) {
    workspace.voltage[neuron_base + neuron] =
        state.voltage[trained_neuron_base + neuron];
    workspace.recovery[neuron_base + neuron] =
        state.recovery[trained_neuron_base + neuron];
    workspace.spikes[neuron_base + neuron] = 0;
    for (int receptor = 0; receptor < kReceptorCount; ++receptor) {
      const int index =
          receptor_base + receptor * kTotalNeurons + neuron;
      workspace.receptor_rise[index] = 0.0f;
      workspace.receptor_decay[index] = 0.0f;
      workspace.conductance[index] = 0.0f;
    }
    for (int slot = 0; slot < kDelayRing; ++slot) {
      workspace.spike_ring[
          (flat_trial * kDelayRing + slot) * kTotalNeurons + neuron] = 0;
    }
  }
  for (int index = threadIdx.x; index < kEvaluationRelativeMs;
       index += blockDim.x) {
    workspace.excitatory_spikes_per_relative_ms[
        output_ms_base + index] = 0u;
  }
  for (int neuron = threadIdx.x; neuron < kExcitatoryNeurons;
       neuron += blockDim.x) {
    workspace.excitatory_baseline_spikes_per_neuron[
        output_neuron_base + neuron] = 0;
    workspace.excitatory_response_spikes_per_neuron[
        output_neuron_base + neuron] = 0;
  }
  for (int feature = threadIdx.x;
       feature < kEvaluationFeatureBins; feature += blockDim.x) {
    workspace.features[feature_base + feature] = 0.0f;
  }
  if (threadIdx.x == 0) {
    const std::uint64_t model_seed =
        state.config.seed + static_cast<std::uint64_t>(seed_index);
    const std::uint64_t trial_entity =
        model_seed * 0xD2B74407B1CE6E93ull +
        static_cast<std::uint64_t>(random_group + 1) *
            0x9E3779B97F4A7C15ull +
        static_cast<std::uint64_t>(trial_in_condition);
    auditory_latency_ms =
        condition.condition.auditory_present
            ? evaluation_positive_normal(
                  evaluation_seed, 1240u, trial_entity, 21.0f, 5.0f)
            : 0.0f;
    visual_latency_ms =
        condition.condition.visual_present
            ? evaluation_positive_normal(
                  evaluation_seed, 1241u, trial_entity, 69.0f, 10.0f)
            : 0.0f;
    const float raw_a =
        condition.condition.auditory_present
            ? auditory_latency_ms
            : FLT_MAX;
    const float raw_v =
        condition.condition.visual_present
            ? condition.condition.physical_soa_ms + visual_latency_ms
            : FLT_MAX;
    const float earliest_received = fminf(raw_a, raw_v);
    const int auditory_received_offset =
        condition.condition.auditory_present
            ? static_cast<int>(
                  lrintf(raw_a - earliest_received))
            : -1;
    const int visual_received_offset =
        condition.condition.visual_present
            ? static_cast<int>(
                  lrintf(raw_v - earliest_received))
            : -1;
    earlier_received_step = burn_in_steps + 100;
    auditory_source_onset =
        auditory_received_offset >= 0
            ? earlier_received_step + auditory_received_offset - 1
            : -1;
    visual_source_onset =
        visual_received_offset >= 0
            ? earlier_received_step + visual_received_offset - 1
            : -1;
    *metadata = {};
    metadata->seed_index = seed_index;
    metadata->condition_index = condition_index;
    metadata->trial_index = trial_in_condition;
    metadata->label = condition.condition.label;
    metadata->physical_soa_ms =
        condition.condition.physical_soa_ms;
    metadata->auditory_latency_ms = auditory_latency_ms;
    metadata->visual_latency_ms = visual_latency_ms;
    metadata->auditory_received_onset_ms =
        auditory_received_offset;
    metadata->visual_received_onset_ms =
        visual_received_offset;
    metadata->earlier_received_onset_step =
        earlier_received_step;
    metadata->burn_in_steps = burn_in_steps;
    metadata->finite = 1;
  }
  __syncthreads();

  const ReceptorKernel receptor_kernels[kReceptorCount] = {
      make_receptor_kernel(
          state.config.ampa_rise_ms, state.config.ampa_decay_ms,
          state.config.dt_ms),
      make_receptor_kernel(
          state.config.nmda_rise_ms, state.config.nmda_decay_ms,
          state.config.dt_ms),
      make_receptor_kernel(
          state.config.gabaa_rise_ms, state.config.gabaa_decay_ms,
          state.config.dt_ms)};
  const int total_steps = burn_in_steps + kEvaluationRelativeMs;
  for (int step = 0; step < total_steps; ++step) {
    const int source_time = step - 1;
    for (int neuron = threadIdx.x; neuron < kTotalNeurons;
         neuron += blockDim.x) {
      const int population =
          neuron < kVOffset ? 0
          : neuron < kEOffset ? 1
          : neuron < kIOffset ? 2
                              : 3;
      const int local = neuron - population_offset(population);
      float arrivals[kReceptorCount]{};
      if (population == 0 || population == 1) {
        const int source_onset =
            population == 0 ? auditory_source_onset
                            : visual_source_onset;
        const int external =
            evaluation_sensory_afferent_count(
                state, condition, seed_index, random_group,
                trial_in_condition, population, local, source_time,
                source_onset, evaluation_seed);
        const int background =
            evaluation_background_afferent_count(
                state, seed_index, random_group, trial_in_condition,
                population, local, source_time, evaluation_seed);
        if (source_time < burn_in_steps && background > 0) {
          atomicAdd(
              &metadata
                   ->background_afferent_arrivals_during_burn_in,
              static_cast<unsigned long long>(background));
        }
        const float q_external = state.config.q_external_rs;
        const float q_background = state.config.q_background_rs;
        arrivals[0] =
            external * state.config.external_ampa_weight * q_external +
            background * state.config.background_ampa_weight *
                q_background;
        arrivals[1] =
            external * state.config.external_nmda_weight *
            nmda_quantum_from_ampa(q_external);
      } else {
        evaluation_recurrent_arrivals(
            state, workspace, flat_trial, seed_index, population,
            local, step, condition.condition.control,
            condition.condition.use_initial_weights,
            &arrivals[0], &arrivals[1], &arrivals[2]);
        const int background =
            evaluation_background_afferent_count(
                state, seed_index, random_group, trial_in_condition,
                population, local, source_time, evaluation_seed);
        if (source_time < burn_in_steps && background > 0) {
          atomicAdd(
              &metadata
                   ->background_afferent_arrivals_during_burn_in,
              static_cast<unsigned long long>(background));
        }
        const float q_background =
            population == 3 ? state.config.q_background_fs
                            : state.config.q_background_rs;
        arrivals[0] +=
            background * state.config.background_ampa_weight *
            q_background;
      }
      if (has_control(
              condition.condition.control,
              CausalControl::kNmdaOff)) {
        arrivals[1] = 0.0f;
      }
      if (has_control(
              condition.condition.control,
              CausalControl::kGabaaOff)) {
        arrivals[2] = 0.0f;
      }
      for (int receptor = 0; receptor < kReceptorCount; ++receptor) {
        const int index =
            receptor_base + receptor * kTotalNeurons + neuron;
        ReceptorState receptor_state{
            workspace.receptor_rise[index],
            workspace.receptor_decay[index]};
        workspace.conductance[index] =
            receptor_advance(
                receptor_kernels[receptor], arrivals[receptor],
                &receptor_state);
        workspace.receptor_rise[index] = receptor_state.rise;
        workspace.receptor_decay[index] = receptor_state.decay;
      }
    }
    __syncthreads();

    for (int neuron = threadIdx.x; neuron < kTotalNeurons;
         neuron += blockDim.x) {
      const int index = neuron_base + neuron;
      SolverInput input{};
      input.state = {
          workspace.voltage[index], workspace.recovery[index]};
      input.parameters = {
          state.neuron_a[trained_neuron_base + neuron],
          state.neuron_b[trained_neuron_base + neuron],
          state.neuron_c[trained_neuron_base + neuron],
          state.neuron_d[trained_neuron_base + neuron]};
      if (neuron >= kEOffset && neuron < kIOffset &&
          has_control(
              condition.condition.control,
              CausalControl::kMsiEAdaptationOff)) {
        input.parameters.d = 0.0f;
      }
      input.g_ampa =
          workspace.conductance[receptor_base + neuron];
      input.g_nmda =
          workspace.conductance[
              receptor_base + kTotalNeurons + neuron];
      input.g_gabaa =
          workspace.conductance[
              receptor_base + 2 * kTotalNeurons + neuron];
      input.dt_ms = state.config.dt_ms;
      input.threshold_mv = state.config.threshold_mv;
      input.excitatory_reversal_mv =
          state.config.excitatory_reversal_mv;
      input.gabaa_reversal_mv = state.config.gabaa_reversal_mv;
      const SolverOutput output = event_resolved_step(input);
      const bool finite =
          !output.overflow &&
          isfinite(output.state.voltage_mv) &&
          isfinite(output.state.recovery);
      workspace.voltage[index] =
          finite ? output.state.voltage_mv : nanf("");
      workspace.recovery[index] =
          finite ? output.state.recovery : nanf("");
      workspace.spikes[index] = output.emitted ? 1 : 0;
      if (!finite) {
        atomicExch(&metadata->finite, 0);
      }
      if (output.emitted && neuron >= kEOffset &&
          neuron < kIOffset) {
        const int relative_ms = step - earlier_received_step;
        if (relative_ms >= -100 && relative_ms < 700) {
          const int output_ms = relative_ms + 100;
          atomicAdd(
              &workspace.excitatory_spikes_per_relative_ms[
                  output_ms_base + output_ms],
              1u);
          const int excitatory_neuron = neuron - kEOffset;
          if (relative_ms < 0) {
            ++workspace.excitatory_baseline_spikes_per_neuron[
                output_neuron_base + excitatory_neuron];
          } else if (relative_ms < 250) {
            ++workspace.excitatory_response_spikes_per_neuron[
                output_neuron_base + excitatory_neuron];
          }
        }
      }
    }
    __syncthreads();

    for (int neuron = threadIdx.x; neuron < kTotalNeurons;
         neuron += blockDim.x) {
      workspace.spike_ring[
          (flat_trial * kDelayRing + step % kDelayRing) *
              kTotalNeurons +
          neuron] = workspace.spikes[neuron_base + neuron];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    unsigned int baseline_spikes = 0;
    unsigned int response_spikes = 0;
    for (int feature = 0; feature < kEvaluationFeatureBins;
         ++feature) {
      unsigned int feature_spikes = 0;
      for (int offset = 0; offset < kEvaluationBinWidthMs; ++offset) {
        feature_spikes +=
            workspace.excitatory_spikes_per_relative_ms[
                output_ms_base +
                feature * kEvaluationBinWidthMs + offset];
      }
      workspace.features[feature_base + feature] =
          static_cast<float>(feature_spikes);
      if (feature < 5) {
        baseline_spikes += feature_spikes;
      }
    }
    for (int relative_ms = 0; relative_ms < 250; ++relative_ms) {
      response_spikes +=
          workspace.excitatory_spikes_per_relative_ms[
              output_ms_base + 100 + relative_ms];
    }
    metadata->baseline_rate_hz =
        static_cast<float>(baseline_spikes) /
        (0.100f * static_cast<float>(kExcitatoryNeurons));
    metadata->response_rate_hz =
        static_cast<float>(response_spikes) /
        (0.250f * static_cast<float>(kExcitatoryNeurons));
  }
}

__device__ inline int scaffold_gcd(int first, int second) {
  while (second != 0) {
    const int remainder = first % second;
    first = second;
    second = remainder;
  }
  return first;
}

__device__ inline int scaffold_in_degree(int path) {
  if (path == 0 || path == 1 || path == 3 || path == 4) {
    return kFeedforwardScaffoldInDegree;
  }
  if (path == 2) {
    return kRecurrentExcitatoryScaffoldInDegree;
  }
  return kInhibitoryScaffoldInDegree;
}

__device__ inline bool position_blind_scaffold_contact(
    std::uint64_t seed, int path, int pre, int post) {
  const PathLayout layout = path_layout(path);
  const bool exclude_self = path == 2;
  if (exclude_self && pre == post) {
    return false;
  }
  const int candidate_count =
      layout.pre_size - (exclude_self ? 1 : 0);
  const int candidate =
      exclude_self && pre > post ? pre - 1 : pre;
  const std::uint64_t entity =
      static_cast<std::uint64_t>(path * 1024 + post);
  const float multiplier_draw =
      uniform01(seed, 830u, entity, 0, 0);
  int multiplier =
      1 + min(
              candidate_count - 2,
              static_cast<int>(
                  multiplier_draw *
                  static_cast<float>(candidate_count - 1)));
  while (scaffold_gcd(multiplier, candidate_count) != 1) {
    ++multiplier;
    if (multiplier >= candidate_count) {
      multiplier = 1;
    }
  }
  const int offset =
      min(
          candidate_count - 1,
          static_cast<int>(
              uniform01(seed, 831u, entity, 0, 0) *
              static_cast<float>(candidate_count)));
  const int rank =
      (multiplier * candidate + offset) % candidate_count;
  return rank < scaffold_in_degree(path);
}

__device__ inline double feedforward_scaffold_gumbel_key(
    std::uint64_t seed, int path, int pre, int post) {
  constexpr std::uint32_t stream = 832u;
  constexpr double two_pow_minus_32 =
      2.3283064365386962890625e-10;
  const PhiloxWords words = philox10(
      {static_cast<std::uint32_t>(pre),
       static_cast<std::uint32_t>(post),
       static_cast<std::uint32_t>(path), 0u},
      static_cast<std::uint32_t>(seed) ^ stream,
      static_cast<std::uint32_t>(seed >> 32) +
          0xA511E9B3u * stream);
  const double uniform =
      (static_cast<double>(words.x) + 0.5) *
      two_pow_minus_32;
  const double source_coordinate =
      kMinimumScaffoldCoordinateDeg +
      static_cast<double>(pre);
  const bool inhibitory_target = path == 3 || path == 4;
  const double target_coordinate =
      inhibitory_target
          ? kMinimumScaffoldCoordinateDeg +
                kScaffoldCoordinateSpanDeg *
                    static_cast<double>(post) /
                    static_cast<double>(
                        kInhibitoryNeurons - 1)
          : kMinimumScaffoldCoordinateDeg +
                static_cast<double>(post);
  const double sigma =
      path == 0 || path == 3
          ? kAuditoryScaffoldSigmaDeg
          : kVisualScaffoldSigmaDeg;
  const double displacement =
      source_coordinate - target_coordinate;
  const double gaussian_log_weight =
      -0.5 * displacement * displacement /
      (sigma * sigma);
  return gaussian_log_weight -
         log(-log(uniform));
}

__device__ inline bool gaussian_feedforward_scaffold_contact(
    std::uint64_t seed, int path, int pre, int post) {
  const PathLayout layout = path_layout(path);
  const double key =
      feedforward_scaffold_gumbel_key(
          seed, path, pre, post);
  int rank = 0;
  for (int candidate_pre = 0;
       candidate_pre < layout.pre_size;
       ++candidate_pre) {
    if (candidate_pre == pre) {
      continue;
    }
    const double candidate_key =
        feedforward_scaffold_gumbel_key(
            seed, path, candidate_pre, post);
    if (candidate_key > key ||
        (candidate_key == key && candidate_pre < pre)) {
      ++rank;
    }
  }
  return rank < kFeedforwardScaffoldInDegree;
}

__device__ inline std::uint64_t msi_e_phenotype_hash(
    std::uint64_t seed, int local_neuron) {
  constexpr std::uint32_t stream = 812u;
  const PhiloxWords words = philox10(
      {static_cast<std::uint32_t>(local_neuron), 0u, 0u, 0u},
      static_cast<std::uint32_t>(seed) ^ stream,
      static_cast<std::uint32_t>(seed >> 32) +
          0xA511E9B3u * stream);
  return (static_cast<std::uint64_t>(words.x) << 32) |
         static_cast<std::uint64_t>(words.y);
}

__device__ inline bool msi_e_marked_phenotype(
    std::uint64_t seed, int local_neuron) {
  const std::uint64_t own_hash =
      msi_e_phenotype_hash(seed, local_neuron);
  int rank = 0;
  for (int candidate = 0; candidate < kExcitatoryNeurons;
       ++candidate) {
    if (candidate == local_neuron) {
      continue;
    }
    const std::uint64_t candidate_hash =
        msi_e_phenotype_hash(seed, candidate);
    if (candidate_hash < own_hash ||
        (candidate_hash == own_hash &&
         candidate < local_neuron)) {
      ++rank;
    }
  }
  return rank < kMsiEMarkedPhenotypesPerSeed;
}

__device__ inline NeuronParameters production_neuron_parameters(
    std::uint64_t seed, int neuron) {
  const bool fast_spiking = neuron >= kIOffset;
  const float heterogeneity =
      uniform01(seed, 810u, neuron, 0, 0);
  if (fast_spiking) {
    return {0.02f + 0.08f * heterogeneity,
            0.25f - 0.05f * heterogeneity,
            -65.0f, 2.0f};
  }
  const float squared = heterogeneity * heterogeneity;
  float reset_voltage_mv = -65.0f + 15.0f * squared;
  float recovery_increment = 8.0f - 6.0f * squared;
  if (neuron >= kEOffset && neuron < kIOffset) {
    recovery_increment =
        msi_e_marked_phenotype(seed, neuron - kEOffset)
            ? kMsiEMarkedRecoveryIncrement
            : kMsiERegularRecoveryIncrement;
  }
  return {0.02f, 0.20f, reset_voltage_mv,
          recovery_increment};
}

__device__ inline NeuronState production_initial_state(
    std::uint64_t seed, int neuron,
    const NeuronParameters& parameters) {
  const bool fast_spiking = neuron >= kIOffset;
  const float voltage_draw =
      uniform01(seed, 811u, neuron, 0, 0);
  const float initial_voltage =
      fast_spiking
          ? -65.0f + 10.0f * voltage_draw
          : fminf(parameters.c_mv, -55.0f) +
                voltage_draw *
                    (fmaxf(parameters.c_mv, -55.0f) -
                     fminf(parameters.c_mv, -55.0f));
  return {initial_voltage, parameters.b * initial_voltage};
}

struct DeviceMsiEIntrinsicPhenotypeSample {
  int assigned_marked = 0;
  int observed_greater_than_two = 0;
  int interval_count = 0;
  float first_to_mean_isi = 0.0f;
  float reset_voltage_mv = 0.0f;
};

__global__ void msi_e_intrinsic_phenotype_audit_kernel(
    std::uint64_t base_seed,
    DeviceMsiEIntrinsicPhenotypeSample* samples) {
  const int seed_index = blockIdx.x;
  const int local_neuron = threadIdx.x;
  if (seed_index >= kMsiEPhenotypeAuditSeeds ||
      local_neuron >= kExcitatoryNeurons) {
    return;
  }
  const std::uint64_t seed =
      base_seed + static_cast<std::uint64_t>(seed_index);
  const int neuron = kEOffset + local_neuron;
  const NeuronParameters parameters =
      production_neuron_parameters(seed, neuron);
  NeuronState state =
      production_initial_state(seed, neuron, parameters);

  SolverInput input{};
  input.parameters = parameters;
  input.dt_ms = 1.0f;
  input.threshold_mv = 30.0f;
  input.excitatory_reversal_mv = 0.0f;
  input.gabaa_reversal_mv = -75.0f;
  for (int time = 0; time < kMsiEPhenotypeSettleMs; ++time) {
    input.state = state;
    input.additive_current = 0.0f;
    state = event_resolved_step(input).state;
  }

  int previous_spike_ms = -1;
  int first_isi_ms = 0;
  int last_isi_ms = 0;
  int interval_sum_ms = 0;
  int interval_count = 0;
  for (int time = 0; time < kMsiEPhenotypeDriveMs; ++time) {
    input.state = state;
    input.additive_current = kMsiEPhenotypeDriveCurrent;
    const SolverOutput output = event_resolved_step(input);
    state = output.state;
    if (!output.emitted) {
      continue;
    }
    if (previous_spike_ms >= 0) {
      const int isi_ms = time - previous_spike_ms;
      if (interval_count == 0) {
        first_isi_ms = isi_ms;
      }
      last_isi_ms = isi_ms;
      interval_sum_ms += isi_ms;
      ++interval_count;
    }
    previous_spike_ms = time;
  }

  DeviceMsiEIntrinsicPhenotypeSample sample{};
  sample.reset_voltage_mv = parameters.c_mv;
  sample.assigned_marked =
      parameters.d == kMsiEMarkedRecoveryIncrement ? 1 : 0;
  sample.interval_count = interval_count;
  if (interval_count > 0 && first_isi_ms > 0) {
    sample.first_to_mean_isi =
        static_cast<float>(first_isi_ms * interval_count) /
        static_cast<float>(interval_sum_ms);
    sample.observed_greater_than_two =
        static_cast<float>(last_isi_ms) /
                    static_cast<float>(first_isi_ms) >
                2.0f
            ? 1
            : 0;
  }
  samples[seed_index * kExcitatoryNeurons + local_neuron] =
      sample;
}

__global__ void initialize_training_state_kernel(
    DeviceTrainingState state) {
  const int seed_index = blockIdx.x;
  if (seed_index >= state.seed_count) {
    return;
  }
  const int thread = threadIdx.x;
  const std::uint64_t seed =
      state.config.seed + static_cast<std::uint64_t>(seed_index);
  const int neuron_base = seed_index * kTotalNeurons;
  const int receptor_base =
      seed_index * kReceptorCount * kTotalNeurons;
  const int weight_base = seed_index * kWeightCount;
  const int trace_base = seed_index * 540;

  for (int neuron = thread; neuron < kTotalNeurons;
       neuron += blockDim.x) {
    const NeuronParameters parameters =
        production_neuron_parameters(seed, neuron);
    const NeuronState initial =
        production_initial_state(seed, neuron, parameters);
    const int index = neuron_base + neuron;
    state.neuron_a[index] = parameters.a;
    state.neuron_b[index] = parameters.b;
    state.neuron_c[index] = parameters.c_mv;
    state.neuron_d[index] = parameters.d;
    state.voltage[index] = initial.voltage_mv;
    state.recovery[index] = initial.recovery;
    state.pre_reset_voltage[index] = initial.voltage_mv;
    state.spikes[index] = 0;
    for (int receptor = 0; receptor < kReceptorCount; ++receptor) {
      const int receptor_index =
          receptor_base + receptor * kTotalNeurons + neuron;
      state.receptor_rise[receptor_index] = 0.0f;
      state.receptor_decay[receptor_index] = 0.0f;
      state.conductance[receptor_index] = 0.0f;
    }
    for (int slot = 0; slot < kDelayRing; ++slot) {
      state.spike_ring[
          (seed_index * kDelayRing + slot) * kTotalNeurons + neuron] = 0;
    }
  }
  __syncthreads();

  for (int trace = thread; trace < 540; trace += blockDim.x) {
    state.oja_pre[trace_base + trace] = 0.0f;
  }
  for (int contact = thread; contact < kClopathContactCount;
       contact += blockDim.x) {
    const int index =
        seed_index * kClopathContactCount + contact;
    state.clopath_contact_pre[index] = 0.0f;
  }
  for (int index = thread;
       index < kClopathPostsynapticNeurons;
       index += blockDim.x) {
    const int neuron_index = neuron_base + kEOffset + index;
    const float initial_voltage =
        state.voltage[neuron_index];
    const int trace_index =
        seed_index * kClopathPostsynapticNeurons + index;
    state.clopath_voltage_minus[trace_index] = initial_voltage;
    state.clopath_voltage_plus[trace_index] = initial_voltage;
    state.clopath_homeostasis[trace_index] =
        kClopathHomeostasisReferenceMv2;
    state.clopath_rest_voltage[trace_index] =
        izhikevich_stable_rest_voltage_mv(
            state.neuron_b[neuron_index]);
  }
  for (int index = thread; index < 180; index += blockDim.x) {
    state.istdp_post[seed_index * 180 + index] = 0.0f;
  }
  for (int index = thread; index < 60; index += blockDim.x) {
    state.oja_post[seed_index * 60 + index] = 0.0f;
    state.istdp_pre[seed_index * 60 + index] = 0.0f;
  }

  for (int flat = thread; flat < kWeightCount;
       flat += blockDim.x) {
    int path = 0;
    while (path + 1 < kPathCount &&
           flat >= path_layout(path + 1).weight_offset) {
      ++path;
    }
    const PathLayout layout = path_layout(path);
    const int local = flat - layout.weight_offset;
    const int pre = local / layout.post_size;
    const int post = local % layout.post_size;
    bool active = true;
    float weight = 0.0f;
    if (path <= 1 || path == 3 || path == 4) {
      active =
          gaussian_feedforward_scaffold_contact(
              seed, path, pre, post);
      weight =
          active
              ? 0.05f +
                    0.10f *
                        uniform01(
                            seed, 840u + path, local, 0, 0)
              : 0.0f;
    } else if (path == 2) {
      active =
          position_blind_scaffold_contact(
              seed, path, pre, post);
      weight =
          active
              ? 0.01f +
                    0.02f *
                        uniform01(seed, 842u, local, 0, 0)
              : 0.0f;
    } else if (path == 5) {
      active = uniform01(seed, 824u, local, 0, 0) < 0.25f;
      weight =
          active
              ? 0.02f +
                    0.03f * uniform01(seed, 825u, local, 0, 0)
              : 0.0f;
    } else {
      active =
          position_blind_scaffold_contact(
              seed, path, pre, post);
      weight =
          active
              ? 0.02f +
                    0.03f *
                        uniform01(seed, 846u, local, 0, 0)
              : 0.0f;
    }
    const int index = weight_base + flat;
    state.weights[index] = weight;
    state.initial_weights[index] = weight;
    state.masks[index] = active ? 1 : 0;
    state.changed[index] = 0;
    state.low_weight_dwell[index] = 0;
  }

  if (thread == 0) {
    state.presentation_index[seed_index] = 0;
    state.accepted_steps[seed_index] = 0;
    state.silent_iti_steps[seed_index] = 0;
    state.silent_iti_afferent_arrivals[seed_index] = 0;
    state.recurrent_plasticity_steps[seed_index] = 0;
    state.pruned_contacts[seed_index] = 0;
    state.seed_barrier_count[seed_index] = 0;
    state.seed_barrier_epoch[seed_index] = 0;
    for (int population = 0; population < kPopulationCount; ++population) {
      state.population_spikes[
          seed_index * kPopulationCount + population] = 0;
    }
    for (int path = 0; path < kPathCount; ++path) {
      state.scheduled_spikes[seed_index * kPathCount + path] = 0;
      state.arrived_spikes[seed_index * kPathCount + path] = 0;
    }
    state.presentations[seed_index] =
        generate_presentation(state.config, seed, 1);
  }
}

__device__ inline int sensory_afferent_count(
    const DeviceTrainingState& state, int seed_index, int modality,
    int neuron, int local_time, int presentation_number,
    const PresentationSpec& spec) {
  const int source_time = local_time - 1;
  if (source_time < 0 || source_time >= spec.silent_iti_start_ms) {
    return 0;
  }
  const bool active =
      modality == 0 ? spec.auditory_active : spec.visual_active;
  const int onset =
      modality == 0 ? spec.auditory_onset_ms : spec.visual_onset_ms;
  if (!active || source_time < onset || source_time >= onset + 50) {
    return 0;
  }
  const float azimuth =
      modality == 0 ? spec.auditory_azimuth_deg
                    : spec.visual_azimuth_deg;
  const float salience =
      modality == 0 ? spec.auditory_salience_hz
                    : spec.visual_salience_hz;
  const float sigma = modality == 0 ? 8.0f : 2.0f;
  const float coordinate = -89.5f + static_cast<float>(neuron);
  const float peak =
      modality == 0 ? spec.auditory_profile_peak
                    : spec.visual_profile_peak;
  const float population_mass_scale =
      modality == 0 ? 1.0f : spec.visual_population_mass_scale;
  const float rate_hz =
      salience * population_mass_scale *
      reflected_profile_raw(azimuth, sigma, coordinate) / peak;
  const std::uint64_t seed =
      state.config.seed + static_cast<std::uint64_t>(seed_index);
  const std::uint64_t entity =
      static_cast<std::uint64_t>(presentation_number) * 512u +
      static_cast<std::uint64_t>(neuron);
  return poisson_small(
      rate_hz * state.config.dt_ms / 1000.0f, seed,
      900u + static_cast<std::uint32_t>(modality), entity,
      static_cast<std::uint32_t>(source_time));
}

__device__ inline int background_afferent_count(
    const DeviceTrainingState& state, int seed_index, int population,
    int neuron, int local_time, int presentation_number,
    const PresentationSpec& spec) {
  const int source_time = local_time - 1;
  if (source_time < 0 || source_time >= spec.silent_iti_start_ms) {
    return 0;
  }
  const std::uint64_t seed =
      state.config.seed + static_cast<std::uint64_t>(seed_index);
  const std::uint64_t entity =
      static_cast<std::uint64_t>(presentation_number) * 1024u +
      static_cast<std::uint64_t>(population * 256 + neuron);
  return poisson_small(
      50.0f * state.config.dt_ms / 1000.0f, seed,
      920u + static_cast<std::uint32_t>(population), entity,
      static_cast<std::uint32_t>(source_time));
}

__device__ inline std::uint8_t delayed_spike(
    const DeviceTrainingState& state, int seed_index, int neuron,
    int delay_steps) {
  const std::uint64_t step = state.accepted_steps[seed_index];
  if (step < static_cast<std::uint64_t>(delay_steps)) {
    return 0;
  }
  const int slot =
      static_cast<int>((step - static_cast<std::uint64_t>(delay_steps)) %
                       kDelayRing);
  return state.spike_ring[
      (seed_index * kDelayRing + slot) * kTotalNeurons + neuron];
}

__device__ inline void recurrent_arrivals(
    const DeviceTrainingState& state, int seed_index, int target_population,
    int post, float* ampa_arrival, float* nmda_arrival,
    float* gabaa_arrival) {
  *ampa_arrival = 0.0f;
  *nmda_arrival = 0.0f;
  *gabaa_arrival = 0.0f;
  const int first_path = target_population == 2 ? 0 : 3;
  const int final_path = target_population == 2 ? 7 : 6;
  const int weight_base = seed_index * kWeightCount;
  for (int path = first_path; path < final_path; ++path) {
    if (target_population == 2 && (path == 3 || path == 4 ||
                                   path == 5)) {
      continue;
    }
    if (target_population == 3 && path == 6) {
      continue;
    }
    const PathLayout layout = path_layout(path);
    if (post >= layout.post_size) {
      continue;
    }
    float path_ampa = 0.0f;
    float path_nmda = 0.0f;
    float path_gabaa = 0.0f;
    for (int pre = 0; pre < layout.pre_size; ++pre) {
      if (!delayed_spike(state, seed_index, layout.pre_offset + pre,
                         layout.delay_steps)) {
        continue;
      }
      const int local = pre * layout.post_size + post;
      const int index = weight_base + layout.weight_offset + local;
      if (!state.masks[index]) {
        continue;
      }
      if (path == 6) {
        path_gabaa += state.weights[index];
      } else {
        path_ampa += state.weights[index];
        path_nmda +=
            shared_nmda_contact_efficacy(
                state.weights[index], layout.fixed_nmda_ratio,
                state.masks[index] != 0);
      }
    }
    if (path == 6) {
      *gabaa_arrival += state.config.q_gabaa * path_gabaa;
    } else {
      const float quantum =
          target_population == 2 ? state.config.q_ff_e
                                 : state.config.q_ff_i;
      *ampa_arrival += quantum * path_ampa;
      *nmda_arrival +=
          nmda_quantum_from_ampa(quantum) * path_nmda;
    }
  }
}

__device__ void update_population_receptors(
    const DeviceTrainingState& state, int seed_index, int population,
    int local_start, int local_count, int local_time,
    int presentation_number, bool active) {
  if (!active) {
    return;
  }
  const PresentationSpec spec = state.presentations[seed_index];
  const int size = population_size(population);
  const int population_begin = population_offset(population);
  const int neuron_base = seed_index * kTotalNeurons;
  const int receptor_base =
      seed_index * kReceptorCount * kTotalNeurons;
  const ReceptorKernel kernels[3] = {
      make_receptor_kernel(state.config.ampa_rise_ms,
                           state.config.ampa_decay_ms,
                           state.config.dt_ms),
      make_receptor_kernel(state.config.nmda_rise_ms,
                           state.config.nmda_decay_ms,
                           state.config.dt_ms),
      make_receptor_kernel(state.config.gabaa_rise_ms,
                           state.config.gabaa_decay_ms,
                           state.config.dt_ms)};
  for (int local = local_start + threadIdx.x;
       local < local_start + local_count && local < size;
       local += blockDim.x) {
    float arrivals[3]{};
    int afferent_arrivals = 0;
    if (population == 0 || population == 1) {
      const int external = sensory_afferent_count(
          state, seed_index, population, local, local_time,
          presentation_number, spec);
      const int background = background_afferent_count(
          state, seed_index, population, local, local_time,
          presentation_number, spec);
      afferent_arrivals = external + background;
      const float q_external = state.config.q_external_rs;
      const float q_background = state.config.q_background_rs;
      arrivals[0] =
          external * state.config.external_ampa_weight * q_external +
          background * state.config.background_ampa_weight * q_background;
      arrivals[1] =
          external * state.config.external_nmda_weight *
          nmda_quantum_from_ampa(q_external);
    } else {
      recurrent_arrivals(state, seed_index, population, local,
                         &arrivals[0], &arrivals[1], &arrivals[2]);
      const int background = background_afferent_count(
          state, seed_index, population, local, local_time,
          presentation_number, spec);
      afferent_arrivals = background;
      const float q_background =
          population == 3 ? state.config.q_background_fs
                          : state.config.q_background_rs;
      arrivals[0] +=
          background * state.config.background_ampa_weight * q_background;
    }
    if (local_time > spec.silent_iti_start_ms &&
        afferent_arrivals != 0) {
      atomicAdd(
          reinterpret_cast<unsigned long long*>(
              &state.silent_iti_afferent_arrivals[seed_index]),
          static_cast<unsigned long long>(afferent_arrivals));
    }
    const int neuron = population_begin + local;
    for (int receptor = 0; receptor < 3; ++receptor) {
      const int receptor_index =
          receptor_base + receptor * kTotalNeurons + neuron;
      ReceptorState receptor_state{
          state.receptor_rise[receptor_index],
          state.receptor_decay[receptor_index]};
      state.conductance[receptor_index] =
          receptor_advance(kernels[receptor], arrivals[receptor],
                           &receptor_state);
      state.receptor_rise[receptor_index] = receptor_state.rise;
      state.receptor_decay[receptor_index] = receptor_state.decay;
    }
  }
}

__device__ void integrate_population(
    const DeviceTrainingState& state, int seed_index, int population,
    bool active) {
  if (!active) {
    return;
  }
  const int size = population_size(population);
  const int population_begin = population_offset(population);
  const int neuron_base = seed_index * kTotalNeurons;
  const int receptor_base =
      seed_index * kReceptorCount * kTotalNeurons;
  for (int local = threadIdx.x; local < size; local += blockDim.x) {
    const int neuron = population_begin + local;
    const int index = neuron_base + neuron;
    SolverInput input{};
    input.state = {state.voltage[index], state.recovery[index]};
    input.parameters = {state.neuron_a[index], state.neuron_b[index],
                        state.neuron_c[index], state.neuron_d[index]};
    input.g_ampa =
        state.conductance[receptor_base + neuron];
    input.g_nmda =
        state.conductance[
            receptor_base + kTotalNeurons + neuron];
    input.g_gabaa =
        state.conductance[
            receptor_base + 2 * kTotalNeurons + neuron];
    input.dt_ms = state.config.dt_ms;
    input.threshold_mv = state.config.threshold_mv;
    input.excitatory_reversal_mv =
        state.config.excitatory_reversal_mv;
    input.gabaa_reversal_mv = state.config.gabaa_reversal_mv;
    const SolverOutput output = event_resolved_step(input);
    const bool finite =
        !output.overflow &&
        isfinite(output.state.voltage_mv) &&
        isfinite(output.state.recovery);
    state.voltage[index] =
        finite ? output.state.voltage_mv : nanf("");
    state.recovery[index] =
        finite ? output.state.recovery : nanf("");
    state.pre_reset_voltage[index] =
        output.pre_reset_voltage_mv;
    state.spikes[index] = output.emitted ? 1 : 0;
    if (output.emitted) {
      atomicAdd(
          reinterpret_cast<unsigned long long*>(
              &state.population_spikes[
                  seed_index * kPopulationCount + population]),
          1ull);
    }
  }
}

__device__ inline std::uint64_t ring_population_spike_count(
    const DeviceTrainingState& state, int seed_index, int population,
    int delay_steps) {
  const std::uint64_t step = state.accepted_steps[seed_index];
  if (step < static_cast<std::uint64_t>(delay_steps)) {
    return 0;
  }
  const int slot =
      static_cast<int>((step - static_cast<std::uint64_t>(delay_steps)) %
                       kDelayRing);
  const int offset = population_offset(population);
  const int size = population_size(population);
  std::uint64_t count = 0;
  for (int neuron = 0; neuron < size; ++neuron) {
    count += state.spike_ring[
        (seed_index * kDelayRing + slot) * kTotalNeurons +
        offset + neuron];
  }
  return count;
}

__device__ void record_arrived_path_spikes(
    const DeviceTrainingState& state, int seed_index, bool active) {
  if (!active || threadIdx.x != 0) {
    return;
  }
  for (int path = 0; path < kPathCount; ++path) {
    const PathLayout layout = path_layout(path);
    const int source_population =
        layout.pre_offset == kAOffset ? 0
        : layout.pre_offset == kVOffset ? 1
        : layout.pre_offset == kEOffset ? 2
                                      : 3;
    state.arrived_spikes[seed_index * kPathCount + path] +=
        ring_population_spike_count(
            state, seed_index, source_population, layout.delay_steps);
  }
}

__device__ void update_clopath_learning_view(
    const DeviceTrainingState& state, int seed_index, bool active,
    int seed_thread, int seed_thread_count) {
  if (!active) {
    return;
  }
  const int weight_base = seed_index * kWeightCount;
  const int contact_trace_base =
      seed_index * kClopathContactCount;
  const float clopath_decay =
      expf(-state.config.dt_ms / 15.0f);
  for (int flat = seed_thread; flat < kClopathContactCount;
       flat += seed_thread_count) {
    int path = 0;
    while (path + 1 < kClopathPathCount &&
           flat >= path_layout(path + 1).weight_offset) {
      ++path;
    }
    const PathLayout layout = path_layout(path);
    const int local = flat - layout.weight_offset;
    const int pre = local / layout.post_size;
    const int weight_index = weight_base + flat;
    const bool arrived =
        state.masks[weight_index] &&
        delayed_spike(
            state, seed_index, layout.pre_offset + pre,
            layout.delay_steps);
    const int trace_index = contact_trace_base + flat;
    state.clopath_contact_pre[trace_index] =
        state.clopath_contact_pre[trace_index] * clopath_decay +
        (arrived ? 1.0f : 0.0f);
  }
}

__device__ void update_event_training_traces(
    const DeviceTrainingState& state, int seed_index, bool active) {
  if (!active) {
    return;
  }
  const int neuron_base = seed_index * kTotalNeurons;
  const int trace_base = seed_index * 540;
  const float oja_decay = expf(-state.config.dt_ms / 20.0f);
  const float istdp_decay =
      expf(-state.config.dt_ms / kIstdpTraceTauMs);
  for (int trace = threadIdx.x; trace < 540; trace += blockDim.x) {
    const float spike = state.spikes[neuron_base + trace] ? 1.0f : 0.0f;
    state.oja_pre[trace_base + trace] =
        state.oja_pre[trace_base + trace] * oja_decay + spike;
  }
  for (int post = threadIdx.x; post < 180; post += blockDim.x) {
    const int e_index = neuron_base + kEOffset + post;
    const int trace_index = seed_index * 180 + post;
    state.istdp_post[trace_index] =
        state.istdp_post[trace_index] * istdp_decay +
        (state.spikes[e_index] ? 1.0f : 0.0f);
  }
  const PathLayout inhibitory_layout = path_layout(6);
  for (int pre = threadIdx.x; pre < 60; pre += blockDim.x) {
    const int i_index = neuron_base + kIOffset + pre;
    const int trace_index = seed_index * 60 + pre;
    state.oja_post[trace_index] =
        state.oja_post[trace_index] * oja_decay +
        (state.spikes[i_index] ? 1.0f : 0.0f);
    const bool inhibitory_arrived =
        delayed_spike(
            state, seed_index,
            inhibitory_layout.pre_offset + pre,
            inhibitory_layout.delay_steps) != 0;
    state.istdp_pre[trace_index] =
        state.istdp_pre[trace_index] * istdp_decay +
        (inhibitory_arrived ? 1.0f : 0.0f);
  }
}

__device__ void update_clopath_voltage_traces(
    const DeviceTrainingState& state, int seed_index, bool active) {
  if (!active) {
    return;
  }
  const int neuron_base = seed_index * kTotalNeurons;
  const float minus_decay = expf(-state.config.dt_ms / 10.0f);
  const float plus_decay = expf(-state.config.dt_ms / 7.0f);
  const float homeostasis_decay =
      expf(-state.config.dt_ms / kClopathHomeostasisTauMs);
  for (int post = threadIdx.x;
       post < kClopathPostsynapticNeurons;
       post += blockDim.x) {
    const int neuron_index = neuron_base + kEOffset + post;
    const float pre_reset =
        state.pre_reset_voltage[neuron_index];
    const int trace_index =
        seed_index * kClopathPostsynapticNeurons + post;
    state.clopath_voltage_minus[trace_index] =
        state.clopath_voltage_minus[trace_index] * minus_decay +
        pre_reset * (1.0f - minus_decay);
    state.clopath_voltage_plus[trace_index] =
        state.clopath_voltage_plus[trace_index] * plus_decay +
        pre_reset * (1.0f - plus_decay);
    const float depolarization =
        pre_reset - state.clopath_rest_voltage[trace_index];
    const float squared_depolarization =
        depolarization * depolarization;
    state.clopath_homeostasis[trace_index] =
        homeostasis_decay *
            state.clopath_homeostasis[trace_index] +
        (1.0f - homeostasis_decay) * squared_depolarization;
  }
}

__device__ void update_plastic_contacts(
    const DeviceTrainingState& state, int seed_index,
    int presentation_number, bool active, int seed_thread,
    int seed_thread_count) {
  if (!active) {
    return;
  }
  const int weight_base = seed_index * kWeightCount;
  const int neuron_base = seed_index * kTotalNeurons;
  const int trace_base = seed_index * 540;
  const int contact_trace_base =
      seed_index * kClopathContactCount;
  for (int flat = seed_thread; flat < kWeightCount;
       flat += seed_thread_count) {
    int path = 0;
    while (path + 1 < kPathCount &&
           flat >= path_layout(path + 1).weight_offset) {
      ++path;
    }
    if (path == 3 || path == 4) {
      continue;
    }
    if ((path == 2 || path == 5) &&
        presentation_number <= 1000) {
      continue;
    }
    const PathLayout layout = path_layout(path);
    const int local = flat - layout.weight_offset;
    const int pre = local / layout.post_size;
    const int post = local % layout.post_size;
    const int index = weight_base + flat;
    if (!state.masks[index]) {
      continue;
    }
    const float previous = state.weights[index];
    float delta = 0.0f;
    float updated = previous;
    if (path <= 2 || path == 3 || path == 4) {
      const bool inhibitory_target = path == 3 || path == 4;
      const int pre_neuron = layout.pre_offset + pre;
      const int post_neuron =
          (inhibitory_target ? kIOffset : kEOffset) + post;
      const int post_trace =
          seed_index * kClopathPostsynapticNeurons +
          (inhibitory_target ? kExcitatoryNeurons + post : post);
      const float eta =
          path == 2 ? 0.01f * state.config.eta_clopath_ff
                    : state.config.eta_clopath_ff;
      const int contact_trace = contact_trace_base + flat;
      const bool pre_arrived =
          delayed_spike(
              state, seed_index, pre_neuron,
              layout.delay_steps) != 0;
      delta = clopath_pair_delta(
          eta, state.clopath_contact_pre[contact_trace],
          state.pre_reset_voltage[neuron_base + post_neuron],
          state.clopath_voltage_minus[post_trace],
          state.clopath_voltage_plus[post_trace],
          state.clopath_rest_voltage[post_trace],
          kClopathThetaPlusMv,
          state.clopath_homeostasis[post_trace],
          pre_arrived);
    } else if (path == 5) {
      const int pre_neuron = layout.pre_offset + pre;
      const float pre_trace =
          state.oja_pre[trace_base + pre_neuron];
      const float post_trace =
          state.oja_post[seed_index * 60 + post];
      delta = state.config.eta_oja * state.config.dt_ms * post_trace *
              (pre_trace - post_trace * previous);
    } else {
      const int pre_neuron = layout.pre_offset + pre;
      const int post_neuron = layout.post_offset + post;
      const bool pre_event =
          delayed_spike(
              state, seed_index, pre_neuron,
              layout.delay_steps) != 0;
      const bool post_event =
          state.spikes[neuron_base + post_neuron] != 0;
      updated =
          ordered_vogels_istdp_update(
              previous, pre_event, post_event,
              state.istdp_pre[seed_index * 60 + pre],
              state.istdp_post[seed_index * 180 + post],
              state.config.eta_istdp, kIstdpAlpha,
              0.0f, layout.upper_bound);
    }
    if (path <= 4) {
      updated =
          additive_hard_bound_update(
              previous, delta, 0.0f, layout.upper_bound);
    } else if (path == 5) {
      updated =
          multiplicative_soft_bound_update(
              previous, delta, 0.0f, layout.upper_bound);
    }
    state.weights[index] = updated;
    if (updated != previous) {
      state.changed[index] = 1;
    }
  }
  if (seed_thread == 0 && presentation_number > 1000) {
    state.recurrent_plasticity_steps[seed_index] += 1;
  }
}

__device__ void update_pruning_partition(
    const DeviceTrainingState& state, int seed_index, int parity,
    int presentation_number, bool enable_pruning) {
  if (!enable_pruning || presentation_number < 5000) {
    return;
  }
  const int weight_base = seed_index * kWeightCount;
  for (int path = parity; path < 6; path += 2) {
    const PathLayout layout = path_layout(path);
    const int contact_count = layout.pre_size * layout.post_size;
    for (int local = threadIdx.x; local < contact_count;
         local += blockDim.x) {
      const int index = weight_base + layout.weight_offset + local;
      if (!state.masks[index]) {
        state.low_weight_dwell[index] = 0;
        continue;
      }
      if (state.weights[index] < 0.02f) {
        const int dwell = state.low_weight_dwell[index] + 1;
        state.low_weight_dwell[index] = dwell;
        if (dwell >= 2000) {
          state.masks[index] = 0;
          state.weights[index] = 0.0f;
          state.changed[index] = 1;
          state.low_weight_dwell[index] = 0;
          atomicAdd(&state.pruned_contacts[seed_index], 1);
        }
      } else {
        state.low_weight_dwell[index] = 0;
      }
    }
  }
}

__device__ void complete_training_step(
    const DeviceTrainingState& state, int seed_index, int local_time,
    bool active) {
  if (!active) {
    return;
  }
  const int neuron_base = seed_index * kTotalNeurons;
  const std::uint64_t step = state.accepted_steps[seed_index];
  const int slot = static_cast<int>(step % kDelayRing);
  for (int neuron = threadIdx.x; neuron < kTotalNeurons;
       neuron += blockDim.x) {
    state.spike_ring[
        (seed_index * kDelayRing + slot) * kTotalNeurons + neuron] =
        state.spikes[neuron_base + neuron];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    std::uint64_t source_counts[kPopulationCount]{};
    for (int population = 0; population < kPopulationCount; ++population) {
      const int offset = population_offset(population);
      const int size = population_size(population);
      for (int neuron = 0; neuron < size; ++neuron) {
        source_counts[population] +=
            state.spikes[neuron_base + offset + neuron];
      }
    }
    state.scheduled_spikes[seed_index * kPathCount + 0] +=
        source_counts[0];
    state.scheduled_spikes[seed_index * kPathCount + 1] +=
        source_counts[1];
    state.scheduled_spikes[seed_index * kPathCount + 2] +=
        source_counts[2];
    state.scheduled_spikes[seed_index * kPathCount + 3] +=
        source_counts[0];
    state.scheduled_spikes[seed_index * kPathCount + 4] +=
        source_counts[1];
    state.scheduled_spikes[seed_index * kPathCount + 5] +=
        source_counts[2];
    state.scheduled_spikes[seed_index * kPathCount + 6] +=
        source_counts[3];
    if (local_time >
        state.presentations[seed_index].silent_iti_start_ms) {
      state.silent_iti_steps[seed_index] += 1;
    }
    state.accepted_steps[seed_index] = step + 1;
  }
}

__device__ void synchronize_training_seed(
    const DeviceTrainingState& state, int seed_index) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned int observed_epoch =
        atomicAdd(&state.seed_barrier_epoch[seed_index], 0u);
    const unsigned int arrival =
        atomicAdd(&state.seed_barrier_count[seed_index], 1u);
    if (arrival ==
        static_cast<unsigned int>(kBlocksPerTrainingSeed - 1)) {
      atomicExch(&state.seed_barrier_count[seed_index], 0u);
      __threadfence();
      atomicAdd(&state.seed_barrier_epoch[seed_index], 1u);
    } else {
      while (atomicAdd(
                 &state.seed_barrier_epoch[seed_index], 0u) ==
             observed_epoch) {
      }
      __threadfence();
    }
  }
  __syncthreads();
}

__global__ __launch_bounds__(kTrainingThreads, 1)
void persistent_training_kernel(DeviceTrainingState state,
                                int chunk_presentations,
                                bool enable_pruning) {
  const int seed_index = blockIdx.x / kBlocksPerTrainingSeed;
  const int role = blockIdx.x % kBlocksPerTrainingSeed;
  const int seed_thread = role * kTrainingThreads + threadIdx.x;
  constexpr int kThreadsPerTrainingSeed =
      kBlocksPerTrainingSeed * kTrainingThreads;
  if (seed_index >= state.seed_count) {
    return;
  }
  const std::uint64_t seed =
      state.config.seed + static_cast<std::uint64_t>(seed_index);

  for (int chunk_index = 0; chunk_index < chunk_presentations;
       ++chunk_index) {
    synchronize_training_seed(state, seed_index);
    const int presentation_number =
        state.presentation_index[seed_index] + 1;
    if (role == 19 && threadIdx.x == 0) {
      state.presentations[seed_index] =
          generate_presentation(
              state.config, seed, presentation_number);
    }
    synchronize_training_seed(state, seed_index);
    const int presentation_duration =
        state.presentations[seed_index].duration_ms;

    for (int local_time = 0;
         local_time < presentation_duration;
         ++local_time) {
      const bool active =
          local_time < state.presentations[seed_index].duration_ms;
      if (role == 0) {
        record_arrived_path_spikes(state, seed_index, active);
        update_population_receptors(
            state, seed_index, 0, 0, 180, local_time,
            presentation_number, active);
      } else if (role == 1) {
        update_population_receptors(
            state, seed_index, 1, 0, 180, local_time,
            presentation_number, active);
      } else if (role == 2) {
        update_population_receptors(
            state, seed_index, 2, 0, 90, local_time,
            presentation_number, active);
      } else if (role == 3) {
        update_population_receptors(
            state, seed_index, 2, 90, 90, local_time,
            presentation_number, active);
      } else if (role == 4) {
        update_population_receptors(
            state, seed_index, 3, 0, 60, local_time,
            presentation_number, active);
      }
      synchronize_training_seed(state, seed_index);

      if (role >= 5 && role <= 8) {
        integrate_population(
            state, seed_index, role - 5, active);
      }
      synchronize_training_seed(state, seed_index);

      if (role == 9) {
        update_event_training_traces(
            state, seed_index, active);
      }
      if (state.config.training_cohort !=
          TrainingCohort::kPlasticityOff) {
        update_clopath_learning_view(
            state, seed_index, active, seed_thread,
            kThreadsPerTrainingSeed);
      }
      synchronize_training_seed(state, seed_index);

      if (state.config.training_cohort !=
          TrainingCohort::kPlasticityOff) {
        update_plastic_contacts(
            state, seed_index, presentation_number, active,
            seed_thread, kThreadsPerTrainingSeed);
      }
      synchronize_training_seed(state, seed_index);

      if (role == 9) {
        update_clopath_voltage_traces(
            state, seed_index, active);
      }
      if (role == 19) {
        complete_training_step(
            state, seed_index, local_time, active);
      }
      synchronize_training_seed(state, seed_index);
    }

    if ((role == 17 || role == 18) &&
        state.config.training_cohort !=
            TrainingCohort::kPlasticityOff) {
      update_pruning_partition(
          state, seed_index, role - 17,
          presentation_number, enable_pruning);
    }
    synchronize_training_seed(state, seed_index);
    if (role == 19 && threadIdx.x == 0) {
      state.presentation_index[seed_index] = presentation_number;
    }
    synchronize_training_seed(state, seed_index);
  }
}

}  // namespace

namespace {

template <typename T>
void allocate_training_array(T** pointer, std::size_t count) {
  CLEAN_MSI_CUDA(cudaMalloc(
      reinterpret_cast<void**>(pointer), count * sizeof(T)));
}

template <typename T>
std::vector<T> copy_training_array(const T* pointer,
                                   std::size_t count) {
  std::vector<T> values(count);
  CLEAN_MSI_CUDA(cudaMemcpy(
      values.data(), pointer, count * sizeof(T),
      cudaMemcpyDeviceToHost));
  return values;
}

double endpoint_homeostasis_quantile(
    const std::vector<double>& sorted_values,
    double probability) {
  const double position =
      probability *
      static_cast<double>(sorted_values.size() - 1);
  const std::size_t lower =
      static_cast<std::size_t>(std::floor(position));
  const std::size_t upper =
      static_cast<std::size_t>(std::ceil(position));
  const double fraction =
      position - static_cast<double>(lower);
  return sorted_values[lower] * (1.0 - fraction) +
         sorted_values[upper] * fraction;
}

void print_endpoint_homeostasis_quantiles(
    const DeviceTrainingState& state, int seed_count) {
  const std::vector<float> homeostasis =
      copy_training_array(
          state.clopath_homeostasis,
          static_cast<std::size_t>(seed_count) *
              kClopathPostsynapticNeurons);
  std::vector<double> excitatory;
  std::vector<double> inhibitory;
  excitatory.reserve(
      static_cast<std::size_t>(seed_count) *
      kExcitatoryNeurons);
  inhibitory.reserve(
      static_cast<std::size_t>(seed_count) *
      kInhibitoryNeurons);
  for (int seed = 0; seed < seed_count; ++seed) {
    const std::size_t base =
        static_cast<std::size_t>(seed) *
        kClopathPostsynapticNeurons;
    for (int post = 0; post < kExcitatoryNeurons; ++post) {
      excitatory.push_back(
          static_cast<double>(homeostasis[base + post]) /
          static_cast<double>(
              kClopathHomeostasisReferenceMv2));
    }
    for (int post = 0; post < kInhibitoryNeurons; ++post) {
      inhibitory.push_back(
          static_cast<double>(
              homeostasis[
                  base + kExcitatoryNeurons + post]) /
          static_cast<double>(
              kClopathHomeostasisReferenceMv2));
    }
  }
  std::sort(excitatory.begin(), excitatory.end());
  std::sort(inhibitory.begin(), inhibitory.end());
  constexpr std::array<double, 5> probabilities{
      0.0, 0.25, 0.5, 0.75, 1.0};
  std::cout << std::setprecision(17)
            << "{\"kind\":\"endpoint_clopath_homeostasis\""
            << ",\"ratio\":\"H_over_Href\""
            << ",\"quantile_probabilities\":[";
  for (std::size_t index = 0; index < probabilities.size();
       ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << probabilities[index];
  }
  std::cout << "],\"E\":[";
  for (std::size_t index = 0; index < probabilities.size();
       ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << endpoint_homeostasis_quantile(
        excitatory, probabilities[index]);
  }
  std::cout << "],\"I\":[";
  for (std::size_t index = 0; index < probabilities.size();
       ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << endpoint_homeostasis_quantile(
        inhibitory, probabilities[index]);
  }
  std::cout << "]}\n";
}

}  // namespace

MsiEIntrinsicPhenotypeAudit audit_msi_e_intrinsic_phenotypes(
    int device, std::uint64_t base_seed) {
  query_device(device);
  CLEAN_MSI_CUDA(cudaSetDevice(device));
  constexpr int sample_count =
      kMsiEPhenotypeAuditSeeds * kExcitatoryNeurons;
  DeviceMsiEIntrinsicPhenotypeSample* device_samples = nullptr;
  CLEAN_MSI_CUDA(cudaMalloc(
      reinterpret_cast<void**>(&device_samples),
      sample_count * sizeof(DeviceMsiEIntrinsicPhenotypeSample)));
  try {
    msi_e_intrinsic_phenotype_audit_kernel
        <<<kMsiEPhenotypeAuditSeeds, 256>>>(
            base_seed, device_samples);
    CLEAN_MSI_CUDA(cudaGetLastError());
    std::vector<DeviceMsiEIntrinsicPhenotypeSample> samples(
        sample_count);
    CLEAN_MSI_CUDA(cudaMemcpy(
        samples.data(), device_samples,
        sample_count * sizeof(DeviceMsiEIntrinsicPhenotypeSample),
        cudaMemcpyDeviceToHost));
    CLEAN_MSI_CUDA(cudaFree(device_samples));

    MsiEIntrinsicPhenotypeAudit audit{};
    audit.seed_count = kMsiEPhenotypeAuditSeeds;
    audit.neuron_count = sample_count;
    int regular_count = 0;
    int marked_count = 0;
    int regular_greater_than_two = 0;
    int marked_greater_than_two = 0;
    double regular_ratio_sum = 0.0;
    double marked_ratio_sum = 0.0;
    double sum_index = 0.0;
    double sum_marked = 0.0;
    double sum_index_squared = 0.0;
    double sum_marked_squared = 0.0;
    double sum_product = 0.0;
    for (int seed_index = 0;
         seed_index < kMsiEPhenotypeAuditSeeds; ++seed_index) {
      for (int local_neuron = 0;
           local_neuron < kExcitatoryNeurons; ++local_neuron) {
        const DeviceMsiEIntrinsicPhenotypeSample& sample =
            samples[static_cast<std::size_t>(
                seed_index * kExcitatoryNeurons + local_neuron)];
        audit.assigned_marked_per_seed[
            static_cast<std::size_t>(seed_index)] +=
            sample.assigned_marked;
        audit.observed_greater_than_two_per_seed[
            static_cast<std::size_t>(seed_index)] +=
            sample.observed_greater_than_two;
        audit.assigned_marked_total += sample.assigned_marked;
        audit.observed_greater_than_two_total +=
            sample.observed_greater_than_two;
        if (sample.assigned_marked != 0) {
          ++marked_count;
          marked_greater_than_two +=
              sample.observed_greater_than_two;
          marked_ratio_sum += sample.first_to_mean_isi;
        } else {
          ++regular_count;
          regular_greater_than_two +=
              sample.observed_greater_than_two;
          regular_ratio_sum += sample.first_to_mean_isi;
        }
        const double index = static_cast<double>(local_neuron);
        const double marked =
            static_cast<double>(sample.assigned_marked);
        sum_index += index;
        sum_marked += marked;
        sum_index_squared += index * index;
        sum_marked_squared += marked * marked;
        sum_product += index * marked;
      }
    }
    audit.greater_than_two_prevalence =
        static_cast<float>(
            audit.observed_greater_than_two_total) /
        static_cast<float>(sample_count);
    audit.regular_first_to_mean_isi =
        static_cast<float>(
            regular_ratio_sum / static_cast<double>(regular_count));
    audit.marked_first_to_mean_isi =
        static_cast<float>(
            marked_ratio_sum / static_cast<double>(marked_count));
    audit.regular_greater_than_two_fraction =
        static_cast<float>(regular_greater_than_two) /
        static_cast<float>(regular_count);
    audit.marked_greater_than_two_fraction =
        static_cast<float>(marked_greater_than_two) /
        static_cast<float>(marked_count);
    const double count = static_cast<double>(sample_count);
    const double covariance =
        count * sum_product - sum_index * sum_marked;
    const double index_variance =
        count * sum_index_squared - sum_index * sum_index;
    const double marked_variance =
        count * sum_marked_squared - sum_marked * sum_marked;
    audit.position_index_correlation =
        static_cast<float>(
            covariance /
            std::sqrt(index_variance * marked_variance));
    return audit;
  } catch (...) {
    cudaFree(device_samples);
    throw;
  }
}

GeneratorProfileAudit audit_developmental_generator(
    int device, std::uint64_t seed, int sample_count,
    float profile_center_deg, float profile_sigma_deg) {
  Config config{};
  return audit_developmental_generator(
      device, config, seed, sample_count,
      profile_center_deg, profile_sigma_deg);
}

GeneratorProfileAudit audit_developmental_generator(
    int device, const Config& config, std::uint64_t seed,
    int sample_count, float profile_center_deg,
    float profile_sigma_deg) {
  if (sample_count <= 0) {
    throw std::invalid_argument(
        "Developmental generator audit requires a positive sample count.");
  }
  if (!std::isfinite(profile_center_deg) ||
      !std::isfinite(profile_sigma_deg) ||
      profile_sigma_deg <= 0.0f) {
    throw std::invalid_argument(
        "Generator profile center and positive sigma must be finite.");
  }
  query_device(device);
  CLEAN_MSI_CUDA(cudaSetDevice(device));

  DevelopmentalSampleAudit* device_samples = nullptr;
  float* device_developmental_profile = nullptr;
  float* device_controlled_profile = nullptr;
  try {
    CLEAN_MSI_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_samples),
        static_cast<std::size_t>(sample_count) *
            sizeof(DevelopmentalSampleAudit)));
    CLEAN_MSI_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_developmental_profile),
        kAuditoryNeurons * sizeof(float)));
    CLEAN_MSI_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_controlled_profile),
        kAuditoryNeurons * sizeof(float)));

    constexpr int audit_threads = 128;
    const int audit_blocks =
        (sample_count + audit_threads - 1) / audit_threads;
    developmental_audit_kernel<<<audit_blocks, audit_threads>>>(
        config, seed, device_samples, sample_count);
    CLEAN_MSI_CUDA(cudaGetLastError());
    spatial_profile_audit_kernel<<<1, kTrainingThreads>>>(
        profile_center_deg, profile_sigma_deg,
        device_developmental_profile, device_controlled_profile);
    CLEAN_MSI_CUDA(cudaGetLastError());

    GeneratorProfileAudit audit{};
    audit.samples.resize(static_cast<std::size_t>(sample_count));
    CLEAN_MSI_CUDA(cudaMemcpy(
        audit.samples.data(), device_samples,
        static_cast<std::size_t>(sample_count) *
            sizeof(DevelopmentalSampleAudit),
        cudaMemcpyDeviceToHost));
    CLEAN_MSI_CUDA(cudaMemcpy(
        audit.developmental_reflected_profile.data(),
        device_developmental_profile,
        kAuditoryNeurons * sizeof(float), cudaMemcpyDeviceToHost));
    CLEAN_MSI_CUDA(cudaMemcpy(
        audit.controlled_exact_profile.data(),
        device_controlled_profile,
        kAuditoryNeurons * sizeof(float), cudaMemcpyDeviceToHost));

    CLEAN_MSI_CUDA(cudaFree(device_controlled_profile));
    CLEAN_MSI_CUDA(cudaFree(device_developmental_profile));
    CLEAN_MSI_CUDA(cudaFree(device_samples));
    return audit;
  } catch (...) {
    cudaFree(device_controlled_profile);
    cudaFree(device_developmental_profile);
    cudaFree(device_samples);
    throw;
  }
}

struct NativeModel::Impl {
  Config config{};
  int device = 0;
  int seed_count = 0;
  int cooperative_capacity = 0;
  bool pruning_ever_enabled = false;
  DeviceTrainingState state{};

  ~Impl() {
    cudaSetDevice(device);
    cudaFree(state.seed_barrier_epoch);
    cudaFree(state.seed_barrier_count);
    cudaFree(state.pruned_contacts);
    cudaFree(state.arrived_spikes);
    cudaFree(state.scheduled_spikes);
    cudaFree(state.population_spikes);
    cudaFree(state.recurrent_plasticity_steps);
    cudaFree(state.silent_iti_afferent_arrivals);
    cudaFree(state.silent_iti_steps);
    cudaFree(state.accepted_steps);
    cudaFree(state.presentation_index);
    cudaFree(state.presentations);
    cudaFree(state.istdp_post);
    cudaFree(state.istdp_pre);
    cudaFree(state.oja_post);
    cudaFree(state.oja_pre);
    cudaFree(state.clopath_rest_voltage);
    cudaFree(state.clopath_homeostasis);
    cudaFree(state.clopath_voltage_plus);
    cudaFree(state.clopath_voltage_minus);
    cudaFree(state.clopath_contact_pre);
    cudaFree(state.low_weight_dwell);
    cudaFree(state.changed);
    cudaFree(state.masks);
    cudaFree(state.initial_weights);
    cudaFree(state.weights);
    cudaFree(state.spike_ring);
    cudaFree(state.spikes);
    cudaFree(state.conductance);
    cudaFree(state.receptor_decay);
    cudaFree(state.receptor_rise);
    cudaFree(state.pre_reset_voltage);
    cudaFree(state.recovery);
    cudaFree(state.voltage);
    cudaFree(state.neuron_d);
    cudaFree(state.neuron_c);
    cudaFree(state.neuron_b);
    cudaFree(state.neuron_a);
  }
};

NativeModel::NativeModel(const Config& config, int device, int seed_count)
    : impl_(std::make_unique<Impl>()) {
  if (!config.calibrated) {
    throw std::invalid_argument(
        "Native training requires a scientifically calibrated config.");
  }
  if (seed_count <= 0) {
    throw std::invalid_argument("seed_count must be positive.");
  }
  const int cohort =
      static_cast<int>(config.training_cohort);
  const int jitter =
      static_cast<int>(config.jitter_scale);
  if (cohort < static_cast<int>(TrainingCohort::kBaseline) ||
      cohort > static_cast<int>(TrainingCohort::kFixedOffset) ||
      jitter < static_cast<int>(JitterScale::kBaseline) ||
      jitter > static_cast<int>(JitterScale::kDouble)) {
    throw std::invalid_argument(
        "Training cohort and jitter scale must be recognized.");
  }
  if (!std::isfinite(config.fixed_offset_deg) ||
      (config.training_cohort == TrainingCohort::kFixedOffset &&
       std::abs(std::abs(config.fixed_offset_deg) - 20.0f) >
           1.0e-6f)) {
    throw std::invalid_argument(
        "Fixed-offset cohorts require a signed +20 or -20 degree offset.");
  }
  const DeviceInfo info = query_device(device);
  if (!info.cooperative_launch) {
    throw std::runtime_error(
        "Selected CUDA device does not support cooperative launch.");
  }
  CLEAN_MSI_CUDA(cudaSetDevice(device));
  impl_->config = config;
  impl_->device = device;
  impl_->seed_count = seed_count;
  DeviceTrainingState& state = impl_->state;
  state.config = config;
  state.seed_count = seed_count;
  const std::size_t seeds = static_cast<std::size_t>(seed_count);
  const std::size_t neurons = seeds * kTotalNeurons;
  const std::size_t receptors =
      seeds * kReceptorCount * kTotalNeurons;
  const std::size_t weights = seeds * kWeightCount;

  allocate_training_array(&state.neuron_a, neurons);
  allocate_training_array(&state.neuron_b, neurons);
  allocate_training_array(&state.neuron_c, neurons);
  allocate_training_array(&state.neuron_d, neurons);
  allocate_training_array(&state.voltage, neurons);
  allocate_training_array(&state.recovery, neurons);
  allocate_training_array(&state.pre_reset_voltage, neurons);
  allocate_training_array(&state.receptor_rise, receptors);
  allocate_training_array(&state.receptor_decay, receptors);
  allocate_training_array(&state.conductance, receptors);
  allocate_training_array(&state.spikes, neurons);
  allocate_training_array(&state.spike_ring, seeds * kDelayRing *
                                               kTotalNeurons);
  allocate_training_array(&state.weights, weights);
  allocate_training_array(&state.initial_weights, weights);
  allocate_training_array(&state.masks, weights);
  allocate_training_array(&state.changed, weights);
  allocate_training_array(&state.low_weight_dwell, weights);
  allocate_training_array(
      &state.clopath_contact_pre,
      seeds * kClopathContactCount);
  allocate_training_array(
      &state.clopath_voltage_minus,
      seeds * kClopathPostsynapticNeurons);
  allocate_training_array(
      &state.clopath_voltage_plus,
      seeds * kClopathPostsynapticNeurons);
  allocate_training_array(
      &state.clopath_homeostasis,
      seeds * kClopathPostsynapticNeurons);
  allocate_training_array(
      &state.clopath_rest_voltage,
      seeds * kClopathPostsynapticNeurons);
  allocate_training_array(&state.oja_pre, seeds * 540);
  allocate_training_array(&state.oja_post, seeds * 60);
  allocate_training_array(&state.istdp_pre, seeds * 60);
  allocate_training_array(&state.istdp_post, seeds * 180);
  allocate_training_array(&state.presentations, seeds);
  allocate_training_array(&state.presentation_index, seeds);
  allocate_training_array(&state.accepted_steps, seeds);
  allocate_training_array(&state.silent_iti_steps, seeds);
  allocate_training_array(
      &state.silent_iti_afferent_arrivals, seeds);
  allocate_training_array(&state.recurrent_plasticity_steps, seeds);
  allocate_training_array(&state.population_spikes,
                          seeds * kPopulationCount);
  allocate_training_array(&state.scheduled_spikes,
                          seeds * kPathCount);
  allocate_training_array(&state.arrived_spikes,
                          seeds * kPathCount);
  allocate_training_array(&state.pruned_contacts, seeds);
  allocate_training_array(&state.seed_barrier_count, seeds);
  allocate_training_array(&state.seed_barrier_epoch, seeds);

  initialize_training_state_kernel<<<seed_count, kTrainingThreads>>>(state);
  CLEAN_MSI_CUDA(cudaGetLastError());
  CLEAN_MSI_CUDA(cudaDeviceSynchronize());

  int active_blocks_per_sm = 0;
  CLEAN_MSI_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm, persistent_training_kernel,
      kTrainingThreads, 0));
  impl_->cooperative_capacity =
      active_blocks_per_sm * info.multiprocessors;
  const int requested_blocks = seed_count * kBlocksPerTrainingSeed;
  if (requested_blocks > impl_->cooperative_capacity) {
    std::ostringstream message;
    message << "Cooperative training requires " << requested_blocks
            << " resident blocks, but device capacity is "
            << impl_->cooperative_capacity << '.';
    throw std::runtime_error(message.str());
  }
}

NativeModel::~NativeModel() = default;
NativeModel::NativeModel(NativeModel&&) noexcept = default;
NativeModel& NativeModel::operator=(NativeModel&&) noexcept = default;

Config NativeModel::config() const {
  return impl_->config;
}

int NativeModel::device() const {
  return impl_->device;
}

int NativeModel::seed_count() const {
  return impl_->seed_count;
}

FrozenTrialBatch NativeModel::run_frozen_trials(
    const std::vector<ControlledCondition>& conditions,
    int trials_per_condition, std::uint64_t evaluation_seed,
    int burn_in_steps) const {
  if (conditions.empty()) {
    throw std::invalid_argument(
        "Frozen evaluation requires at least one condition.");
  }
  if (trials_per_condition <= 0) {
    throw std::invalid_argument(
        "Frozen evaluation trials_per_condition must be positive.");
  }
  if (burn_in_steps < 0) {
    throw std::invalid_argument(
        "Frozen evaluation burn-in cannot be negative.");
  }
  std::vector<EvaluationConditionSpec> condition_specs(
      conditions.size());
  for (std::size_t index = 0; index < conditions.size(); ++index) {
    const ControlledCondition& condition = conditions[index];
    if (!condition.auditory_present && !condition.visual_present) {
      throw std::invalid_argument(
          "A controlled condition must present at least one modality.");
    }
    const auto valid_location = [](float value) {
      return std::isfinite(value) && value >= -89.5f &&
             value <= 89.5f;
    };
    if ((condition.auditory_present &&
         !valid_location(condition.auditory_location_deg)) ||
        (condition.visual_present &&
         !valid_location(condition.visual_location_deg))) {
      throw std::invalid_argument(
          "Controlled probe locations must be finite and in bounds.");
    }
    if (!std::isfinite(condition.auditory_rate_hz) ||
        !std::isfinite(condition.visual_rate_hz) ||
        condition.auditory_rate_hz < 0.0f ||
        condition.visual_rate_hz < 0.0f ||
        !std::isfinite(condition.physical_soa_ms)) {
      throw std::invalid_argument(
          "Controlled rates and physical SOA must be finite.");
    }
    EvaluationConditionSpec& spec = condition_specs[index];
    spec.condition = condition;
    if (condition.auditory_present) {
      spec.auditory_profile_peak =
          discrete_controlled_profile_peak(
              condition.auditory_location_deg, 8.0f);
    }
    if (condition.visual_present) {
      spec.visual_profile_peak =
          discrete_controlled_profile_peak(
              condition.visual_location_deg, 2.0f);
      const float auditory_reference_peak =
          discrete_controlled_profile_peak(
              condition.visual_location_deg, 8.0f);
      spec.visual_population_mass_scale =
          discrete_visual_population_mass_scale(
              condition.visual_location_deg, auditory_reference_peak,
              spec.visual_profile_peak, false);
    }
  }

  const std::size_t condition_count = conditions.size();
  const std::size_t total_trials =
      static_cast<std::size_t>(impl_->seed_count) * condition_count *
      static_cast<std::size_t>(trials_per_condition);
  if (total_trials == 0 ||
      total_trials > static_cast<std::size_t>(INT_MAX)) {
    throw std::invalid_argument(
        "Frozen evaluation trial grid exceeds the CUDA grid limit.");
  }

  CLEAN_MSI_CUDA(cudaSetDevice(impl_->device));
  EvaluationConditionSpec* device_conditions = nullptr;
  DeviceEvaluationWorkspace workspace{};
  try {
    allocate_training_array(
        &device_conditions, condition_specs.size());
    CLEAN_MSI_CUDA(cudaMemcpy(
        device_conditions, condition_specs.data(),
        condition_specs.size() * sizeof(EvaluationConditionSpec),
        cudaMemcpyHostToDevice));
    allocate_training_array(
        &workspace.voltage, total_trials * kTotalNeurons);
    allocate_training_array(
        &workspace.recovery, total_trials * kTotalNeurons);
    allocate_training_array(
        &workspace.receptor_rise,
        total_trials * kReceptorCount * kTotalNeurons);
    allocate_training_array(
        &workspace.receptor_decay,
        total_trials * kReceptorCount * kTotalNeurons);
    allocate_training_array(
        &workspace.conductance,
        total_trials * kReceptorCount * kTotalNeurons);
    allocate_training_array(
        &workspace.spikes, total_trials * kTotalNeurons);
    allocate_training_array(
        &workspace.spike_ring,
        total_trials * kDelayRing * kTotalNeurons);
    allocate_training_array(
        &workspace.excitatory_spikes_per_relative_ms,
        total_trials * kEvaluationRelativeMs);
    allocate_training_array(
        &workspace.excitatory_baseline_spikes_per_neuron,
        total_trials * kExcitatoryNeurons);
    allocate_training_array(
        &workspace.excitatory_response_spikes_per_neuron,
        total_trials * kExcitatoryNeurons);
    allocate_training_array(
        &workspace.features,
        total_trials * kEvaluationFeatureBins);
    allocate_training_array(&workspace.metadata, total_trials);

    const auto start = std::chrono::steady_clock::now();
    frozen_evaluation_kernel<<<
        static_cast<int>(total_trials), kTrainingThreads>>>(
        impl_->state, device_conditions,
        static_cast<int>(condition_specs.size()),
        trials_per_condition, burn_in_steps, evaluation_seed,
        workspace);
    CLEAN_MSI_CUDA(cudaGetLastError());
    CLEAN_MSI_CUDA(cudaDeviceSynchronize());
    const auto finish = std::chrono::steady_clock::now();

    const std::vector<unsigned int> per_ms =
        copy_training_array(
            workspace.excitatory_spikes_per_relative_ms,
            total_trials * kEvaluationRelativeMs);
    const std::vector<std::uint16_t> baseline =
        copy_training_array(
            workspace.excitatory_baseline_spikes_per_neuron,
            total_trials * kExcitatoryNeurons);
    const std::vector<std::uint16_t> response =
        copy_training_array(
            workspace.excitatory_response_spikes_per_neuron,
            total_trials * kExcitatoryNeurons);
    const std::vector<float> features =
        copy_training_array(
            workspace.features,
            total_trials * kEvaluationFeatureBins);
    const std::vector<EvaluationTrialMetadata> metadata =
        copy_training_array(workspace.metadata, total_trials);

    FrozenTrialBatch batch{};
    batch.model_seed_count = impl_->seed_count;
    batch.condition_count =
        static_cast<int>(condition_specs.size());
    batch.trials_per_condition = trials_per_condition;
    batch.burn_in_steps = burn_in_steps;
    batch.kernel_seconds =
        std::chrono::duration<float>(finish - start).count();
    for (int edge = 0; edge <= kEvaluationFeatureBins; ++edge) {
      batch.relative_bin_edges_ms[
          static_cast<std::size_t>(edge)] =
          -100 + edge * kEvaluationBinWidthMs;
    }
    batch.trials.resize(total_trials);
    for (std::size_t trial = 0; trial < total_trials; ++trial) {
      FrozenTrialResult& result = batch.trials[trial];
      const EvaluationTrialMetadata& values = metadata[trial];
      result.seed_index = values.seed_index;
      result.condition_index = values.condition_index;
      result.trial_index = values.trial_index;
      result.label = values.label;
      result.physical_soa_ms = values.physical_soa_ms;
      result.auditory_latency_ms = values.auditory_latency_ms;
      result.visual_latency_ms = values.visual_latency_ms;
      result.auditory_received_onset_ms =
          values.auditory_received_onset_ms;
      result.visual_received_onset_ms =
          values.visual_received_onset_ms;
      result.earlier_received_onset_step =
          values.earlier_received_onset_step;
      result.burn_in_steps = values.burn_in_steps;
      result.background_afferent_arrivals_during_burn_in =
          static_cast<std::uint64_t>(
              values.background_afferent_arrivals_during_burn_in);
      result.baseline_rate_hz = values.baseline_rate_hz;
      result.response_rate_hz = values.response_rate_hz;
      result.finite = values.finite != 0;
      for (int ms = 0; ms < kEvaluationRelativeMs; ++ms) {
        const unsigned int count =
            per_ms[
                trial * kEvaluationRelativeMs +
                static_cast<std::size_t>(ms)];
        if (count > UINT16_MAX) {
          throw std::runtime_error(
              "Frozen trial population spike count overflowed uint16.");
        }
        result.excitatory_spikes_per_relative_ms[
            static_cast<std::size_t>(ms)] =
            static_cast<std::uint16_t>(count);
      }
      for (int neuron = 0; neuron < kExcitatoryNeurons; ++neuron) {
        const std::size_t index =
            trial * kExcitatoryNeurons +
            static_cast<std::size_t>(neuron);
        result.excitatory_baseline_spikes_per_neuron[
            static_cast<std::size_t>(neuron)] = baseline[index];
        result.excitatory_response_spikes_per_neuron[
            static_cast<std::size_t>(neuron)] = response[index];
      }
      for (int feature = 0; feature < kEvaluationFeatureBins;
           ++feature) {
        result.features[static_cast<std::size_t>(feature)] =
            features[
                trial * kEvaluationFeatureBins +
                static_cast<std::size_t>(feature)];
      }
    }

    CLEAN_MSI_CUDA(cudaFree(workspace.metadata));
    CLEAN_MSI_CUDA(cudaFree(workspace.features));
    CLEAN_MSI_CUDA(cudaFree(
        workspace.excitatory_response_spikes_per_neuron));
    CLEAN_MSI_CUDA(cudaFree(
        workspace.excitatory_baseline_spikes_per_neuron));
    CLEAN_MSI_CUDA(cudaFree(
        workspace.excitatory_spikes_per_relative_ms));
    CLEAN_MSI_CUDA(cudaFree(workspace.spike_ring));
    CLEAN_MSI_CUDA(cudaFree(workspace.spikes));
    CLEAN_MSI_CUDA(cudaFree(workspace.conductance));
    CLEAN_MSI_CUDA(cudaFree(workspace.receptor_decay));
    CLEAN_MSI_CUDA(cudaFree(workspace.receptor_rise));
    CLEAN_MSI_CUDA(cudaFree(workspace.recovery));
    CLEAN_MSI_CUDA(cudaFree(workspace.voltage));
    CLEAN_MSI_CUDA(cudaFree(device_conditions));
    return batch;
  } catch (...) {
    cudaFree(workspace.metadata);
    cudaFree(workspace.features);
    cudaFree(workspace.excitatory_response_spikes_per_neuron);
    cudaFree(workspace.excitatory_baseline_spikes_per_neuron);
    cudaFree(workspace.excitatory_spikes_per_relative_ms);
    cudaFree(workspace.spike_ring);
    cudaFree(workspace.spikes);
    cudaFree(workspace.conductance);
    cudaFree(workspace.receptor_decay);
    cudaFree(workspace.receptor_rise);
    cudaFree(workspace.recovery);
    cudaFree(workspace.voltage);
    cudaFree(device_conditions);
    throw;
  }
}

std::vector<FrozenWeightAudit> NativeModel::frozen_weight_audit() const {
  CLEAN_MSI_CUDA(cudaSetDevice(impl_->device));
  const std::size_t seeds =
      static_cast<std::size_t>(impl_->seed_count);
  const std::vector<float> weights =
      copy_training_array(
          impl_->state.weights, seeds * kWeightCount);
  const std::vector<std::uint8_t> masks =
      copy_training_array(
          impl_->state.masks, seeds * kWeightCount);
  std::vector<FrozenWeightAudit> audits(seeds);
  for (int seed = 0; seed < impl_->seed_count; ++seed) {
    FrozenWeightAudit& audit =
        audits[static_cast<std::size_t>(seed)];
    audit.seed_index = seed;
    audit.global_seed =
        impl_->config.seed + static_cast<std::uint64_t>(seed);
    const std::size_t seed_offset =
        static_cast<std::size_t>(seed) * kWeightCount;
    for (int path = 0; path < kPathCount; ++path) {
      const PathLayout layout = path_layout(path);
      const std::size_t begin =
          seed_offset +
          static_cast<std::size_t>(layout.weight_offset);
      const std::size_t count =
          static_cast<std::size_t>(layout.pre_size) *
          static_cast<std::size_t>(layout.post_size);
      FrozenPathAudit& path_audit =
          audit.paths[static_cast<std::size_t>(path)];
      path_audit.weights.assign(
          weights.begin() + static_cast<std::ptrdiff_t>(begin),
          weights.begin() +
              static_cast<std::ptrdiff_t>(begin + count));
      path_audit.masks.assign(
          masks.begin() + static_cast<std::ptrdiff_t>(begin),
          masks.begin() +
              static_cast<std::ptrdiff_t>(begin + count));
    }
  }
  return audits;
}

std::vector<SeedWeightStateAudit>
NativeModel::weight_state_audit() const {
  CLEAN_MSI_CUDA(cudaSetDevice(impl_->device));
  const std::size_t seeds =
      static_cast<std::size_t>(impl_->seed_count);
  const std::vector<float> initial_weights =
      copy_training_array(
          impl_->state.initial_weights, seeds * kWeightCount);
  std::vector<FrozenWeightAudit> trained = frozen_weight_audit();
  std::vector<SeedWeightStateAudit> audits(seeds);
  for (int seed = 0; seed < impl_->seed_count; ++seed) {
    SeedWeightStateAudit& audit =
        audits[static_cast<std::size_t>(seed)];
    audit.seed_index = seed;
    audit.global_seed =
        impl_->config.seed + static_cast<std::uint64_t>(seed);
    audit.trained =
        std::move(trained[static_cast<std::size_t>(seed)]);
    audit.initial.seed_index = seed;
    audit.initial.global_seed = audit.global_seed;
    const std::size_t seed_offset =
        static_cast<std::size_t>(seed) * kWeightCount;
    for (int path = 0; path < kPathCount; ++path) {
      const PathLayout layout = path_layout(path);
      const std::size_t begin =
          seed_offset +
          static_cast<std::size_t>(layout.weight_offset);
      const std::size_t count =
          static_cast<std::size_t>(layout.pre_size) *
          static_cast<std::size_t>(layout.post_size);
      FrozenPathAudit& initial_path =
          audit.initial.paths[static_cast<std::size_t>(path)];
      initial_path.weights.assign(
          initial_weights.begin() +
              static_cast<std::ptrdiff_t>(begin),
          initial_weights.begin() +
              static_cast<std::ptrdiff_t>(begin + count));
      initial_path.masks.resize(count);
      for (std::size_t index = 0; index < count; ++index) {
        initial_path.masks[index] =
            initial_path.weights[index] > 0.0f ? 1u : 0u;
      }
    }
  }
  return audits;
}

namespace {

double stable_logistic(double value) {
  if (value >= 0.0) {
    const double tail = std::exp(-value);
    return 1.0 / (1.0 + tail);
  }
  const double head = std::exp(value);
  return head / (1.0 + head);
}

bool solve_dense_system(
    std::vector<double> matrix, std::vector<double> rhs,
    int dimension, std::vector<double>* solution) {
  for (int column = 0; column < dimension; ++column) {
    int pivot = column;
    double pivot_magnitude =
        std::abs(matrix[
            static_cast<std::size_t>(column) * dimension + column]);
    for (int row = column + 1; row < dimension; ++row) {
      const double magnitude =
          std::abs(matrix[
              static_cast<std::size_t>(row) * dimension + column]);
      if (magnitude > pivot_magnitude) {
        pivot = row;
        pivot_magnitude = magnitude;
      }
    }
    if (!(pivot_magnitude > 1.0e-14) ||
        !std::isfinite(pivot_magnitude)) {
      return false;
    }
    if (pivot != column) {
      for (int inner = column; inner < dimension; ++inner) {
        std::swap(
            matrix[
                static_cast<std::size_t>(column) * dimension + inner],
            matrix[
                static_cast<std::size_t>(pivot) * dimension + inner]);
      }
      std::swap(
          rhs[static_cast<std::size_t>(column)],
          rhs[static_cast<std::size_t>(pivot)]);
    }
    const double diagonal =
        matrix[
            static_cast<std::size_t>(column) * dimension + column];
    for (int row = column + 1; row < dimension; ++row) {
      const std::size_t row_base =
          static_cast<std::size_t>(row) * dimension;
      const double factor =
          matrix[row_base + column] / diagonal;
      matrix[row_base + column] = 0.0;
      for (int inner = column + 1; inner < dimension; ++inner) {
        matrix[row_base + inner] -=
            factor *
            matrix[
                static_cast<std::size_t>(column) * dimension + inner];
      }
      rhs[static_cast<std::size_t>(row)] -=
          factor * rhs[static_cast<std::size_t>(column)];
    }
  }
  solution->assign(static_cast<std::size_t>(dimension), 0.0);
  for (int row = dimension - 1; row >= 0; --row) {
    double value = rhs[static_cast<std::size_t>(row)];
    for (int column = row + 1; column < dimension; ++column) {
      value -=
          matrix[
              static_cast<std::size_t>(row) * dimension + column] *
          (*solution)[static_cast<std::size_t>(column)];
    }
    const double diagonal =
        matrix[
            static_cast<std::size_t>(row) * dimension + row];
    (*solution)[static_cast<std::size_t>(row)] = value / diagonal;
  }
  return true;
}

double standardized_observer_score(
    const ObserverMetrics& observer,
    const std::array<float, kEvaluationFeatureBins>& features) {
  double score = observer.intercept;
  for (int feature = 0; feature < kEvaluationFeatureBins; ++feature) {
    const double standardized =
        (static_cast<double>(
             features[static_cast<std::size_t>(feature)]) -
         observer.feature_mean[static_cast<std::size_t>(feature)]) /
        observer.feature_std[static_cast<std::size_t>(feature)];
    score +=
        observer.coefficients[static_cast<std::size_t>(feature)] *
        standardized;
  }
  return score;
}

float observer_probability(
    const ObserverMetrics& observer,
    const std::array<float, kEvaluationFeatureBins>& features) {
  return static_cast<float>(
      stable_logistic(standardized_observer_score(observer, features)));
}

double logistic_objective(
    const std::vector<FrozenTrialResult>& trials,
    const std::array<double, kEvaluationFeatureBins>& feature_mean,
    const std::array<double, kEvaluationFeatureBins>& feature_std,
    const std::vector<double>& coefficients, double l2) {
  double objective = 0.0;
  for (const FrozenTrialResult& trial : trials) {
    double score = coefficients[0];
    for (int feature = 0; feature < kEvaluationFeatureBins; ++feature) {
      const double standardized =
          (trial.features[static_cast<std::size_t>(feature)] -
           feature_mean[static_cast<std::size_t>(feature)]) /
          feature_std[static_cast<std::size_t>(feature)];
      score +=
          coefficients[static_cast<std::size_t>(feature + 1)] *
          standardized;
    }
    const double label = trial.label != 0 ? 1.0 : 0.0;
    objective +=
        std::max(score, 0.0) - label * score +
        std::log1p(std::exp(-std::abs(score)));
  }
  objective /= static_cast<double>(trials.size());
  for (int feature = 1; feature <= kEvaluationFeatureBins; ++feature) {
    const double value =
        coefficients[static_cast<std::size_t>(feature)];
    objective += 0.5 * l2 * value * value;
  }
  return objective;
}

float binary_auroc(
    const std::vector<std::uint8_t>& labels,
    const std::vector<float>& probabilities) {
  struct RankedValue {
    float probability;
    std::uint8_t label;
    std::size_t order;
  };
  std::vector<RankedValue> ranked(labels.size());
  int positives = 0;
  for (std::size_t index = 0; index < labels.size(); ++index) {
    ranked[index] = {
        probabilities[index], labels[index], index};
    positives += labels[index] != 0 ? 1 : 0;
  }
  const int negatives =
      static_cast<int>(labels.size()) - positives;
  if (positives == 0 || negatives == 0) {
    return 0.5f;
  }
  std::stable_sort(
      ranked.begin(), ranked.end(),
      [](const RankedValue& lhs, const RankedValue& rhs) {
        if (lhs.probability != rhs.probability) {
          return lhs.probability < rhs.probability;
        }
        return lhs.order < rhs.order;
      });
  double positive_rank_sum = 0.0;
  std::size_t begin = 0;
  while (begin < ranked.size()) {
    std::size_t end = begin + 1;
    while (end < ranked.size() &&
           ranked[end].probability == ranked[begin].probability) {
      ++end;
    }
    const double average_rank =
        0.5 * (static_cast<double>(begin + 1) +
               static_cast<double>(end));
    for (std::size_t index = begin; index < end; ++index) {
      if (ranked[index].label != 0) {
        positive_rank_sum += average_rank;
      }
    }
    begin = end;
  }
  const double numerator =
      positive_rank_sum -
      0.5 * positives * static_cast<double>(positives + 1);
  return static_cast<float>(
      numerator /
      (static_cast<double>(positives) * negatives));
}

void fit_calibration_line(
    const std::vector<std::uint8_t>& labels,
    const std::vector<float>& probabilities,
    float* intercept, float* slope) {
  double beta0 = 0.0;
  double beta1 = 1.0;
  for (int iteration = 0; iteration < 50; ++iteration) {
    double gradient0 = 0.0;
    double gradient1 = 0.0;
    double hessian00 = 1.0e-9;
    double hessian01 = 0.0;
    double hessian11 = 1.0e-9;
    for (std::size_t index = 0; index < labels.size(); ++index) {
      const double probability =
          std::min(
              std::max(
                  static_cast<double>(probabilities[index]), 1.0e-6),
              1.0 - 1.0e-6);
      const double logit =
          std::log(probability / (1.0 - probability));
      const double fitted =
          stable_logistic(beta0 + beta1 * logit);
      const double residual =
          fitted - (labels[index] != 0 ? 1.0 : 0.0);
      const double curvature =
          std::max(fitted * (1.0 - fitted), 1.0e-8);
      gradient0 += residual;
      gradient1 += residual * logit;
      hessian00 += curvature;
      hessian01 += curvature * logit;
      hessian11 += curvature * logit * logit;
    }
    const double determinant =
        hessian00 * hessian11 - hessian01 * hessian01;
    if (!(std::abs(determinant) > 1.0e-14)) {
      break;
    }
    const double delta0 =
        (hessian11 * gradient0 - hessian01 * gradient1) /
        determinant;
    const double delta1 =
        (-hessian01 * gradient0 + hessian00 * gradient1) /
        determinant;
    beta0 -= delta0;
    beta1 -= delta1;
    if (std::max(std::abs(delta0), std::abs(delta1)) < 1.0e-8) {
      break;
    }
  }
  *intercept = static_cast<float>(beta0);
  *slope = static_cast<float>(beta1);
}

ObserverMetrics fit_observer(
    const FrozenTrialBatch& training,
    const FrozenTrialBatch& holdout) {
  if (training.trials.empty() || holdout.trials.empty()) {
    throw std::invalid_argument(
        "Observer fitting requires non-empty training and holdout data.");
  }
  constexpr int dimension = kEvaluationFeatureBins + 1;
  constexpr double l2 = 1.0e-3;
  std::array<double, kEvaluationFeatureBins> feature_mean{};
  std::array<double, kEvaluationFeatureBins> feature_std{};
  for (const FrozenTrialResult& trial : training.trials) {
    for (int feature = 0; feature < kEvaluationFeatureBins; ++feature) {
      feature_mean[static_cast<std::size_t>(feature)] +=
          trial.features[static_cast<std::size_t>(feature)] /
          static_cast<double>(training.trials.size());
    }
  }
  for (const FrozenTrialResult& trial : training.trials) {
    for (int feature = 0; feature < kEvaluationFeatureBins; ++feature) {
      const double centered =
          trial.features[static_cast<std::size_t>(feature)] -
          feature_mean[static_cast<std::size_t>(feature)];
      feature_std[static_cast<std::size_t>(feature)] +=
          centered * centered /
          static_cast<double>(training.trials.size());
    }
  }
  for (double& value : feature_std) {
    value = std::sqrt(std::max(value, 0.0));
    if (value < 1.0e-6) {
      value = 1.0;
    }
  }
  int positive_count = 0;
  for (const FrozenTrialResult& trial : training.trials) {
    positive_count += trial.label != 0 ? 1 : 0;
  }
  const int negative_count =
      static_cast<int>(training.trials.size()) - positive_count;
  if (positive_count == 0 || negative_count == 0) {
    throw std::invalid_argument(
        "Observer training labels must contain both classes.");
  }
  std::vector<double> coefficients(
      static_cast<std::size_t>(dimension), 0.0);
  coefficients[0] =
      std::log(
          static_cast<double>(positive_count) /
          static_cast<double>(negative_count));
  bool converged = false;
  int iterations = 0;
  for (; iterations < 100; ++iterations) {
    std::vector<double> gradient(
        static_cast<std::size_t>(dimension), 0.0);
    std::vector<double> hessian(
        static_cast<std::size_t>(dimension) * dimension, 0.0);
    std::array<double, dimension> predictors{};
    for (const FrozenTrialResult& trial : training.trials) {
      predictors[0] = 1.0;
      double score = coefficients[0];
      for (int feature = 0; feature < kEvaluationFeatureBins; ++feature) {
        const double standardized =
            (trial.features[static_cast<std::size_t>(feature)] -
             feature_mean[static_cast<std::size_t>(feature)]) /
            feature_std[static_cast<std::size_t>(feature)];
        predictors[static_cast<std::size_t>(feature + 1)] =
            standardized;
        score +=
            coefficients[static_cast<std::size_t>(feature + 1)] *
            standardized;
      }
      const double probability = stable_logistic(score);
      const double label = trial.label != 0 ? 1.0 : 0.0;
      const double residual =
          (probability - label) /
          static_cast<double>(training.trials.size());
      const double curvature =
          std::max(probability * (1.0 - probability), 1.0e-9) /
          static_cast<double>(training.trials.size());
      for (int row = 0; row < dimension; ++row) {
        gradient[static_cast<std::size_t>(row)] +=
            residual * predictors[static_cast<std::size_t>(row)];
        for (int column = 0; column <= row; ++column) {
          hessian[
              static_cast<std::size_t>(row) * dimension + column] +=
              curvature *
              predictors[static_cast<std::size_t>(row)] *
              predictors[static_cast<std::size_t>(column)];
        }
      }
    }
    for (int row = 0; row < dimension; ++row) {
      for (int column = row + 1; column < dimension; ++column) {
        hessian[
            static_cast<std::size_t>(row) * dimension + column] =
            hessian[
                static_cast<std::size_t>(column) * dimension + row];
      }
      hessian[
          static_cast<std::size_t>(row) * dimension + row] += 1.0e-9;
    }
    for (int feature = 1; feature < dimension; ++feature) {
      gradient[static_cast<std::size_t>(feature)] +=
          l2 * coefficients[static_cast<std::size_t>(feature)];
      hessian[
          static_cast<std::size_t>(feature) * dimension + feature] +=
          l2;
    }
    std::vector<double> step;
    if (!solve_dense_system(
            hessian, gradient, dimension, &step)) {
      break;
    }
    double maximum_step = 0.0;
    for (double value : step) {
      maximum_step = std::max(maximum_step, std::abs(value));
    }
    const double current_objective =
        logistic_objective(
            training.trials, feature_mean, feature_std,
            coefficients, l2);
    double scale = 1.0;
    std::vector<double> candidate(coefficients.size());
    bool accepted = false;
    for (int line_search = 0; line_search < 24; ++line_search) {
      for (std::size_t index = 0; index < coefficients.size(); ++index) {
        candidate[index] =
            coefficients[index] - scale * step[index];
      }
      const double candidate_objective =
          logistic_objective(
              training.trials, feature_mean, feature_std,
              candidate, l2);
      if (candidate_objective <= current_objective) {
        accepted = true;
        break;
      }
      scale *= 0.5;
    }
    if (!accepted) {
      break;
    }
    coefficients.swap(candidate);
    if (scale * maximum_step < 1.0e-8) {
      converged = true;
      ++iterations;
      break;
    }
  }

  ObserverMetrics observer{};
  observer.l2 = static_cast<float>(l2);
  observer.intercept = static_cast<float>(coefficients[0]);
  observer.iterations = iterations;
  observer.converged = converged;
  for (int feature = 0; feature < kEvaluationFeatureBins; ++feature) {
    observer.feature_mean[static_cast<std::size_t>(feature)] =
        static_cast<float>(
            feature_mean[static_cast<std::size_t>(feature)]);
    observer.feature_std[static_cast<std::size_t>(feature)] =
        static_cast<float>(
            feature_std[static_cast<std::size_t>(feature)]);
    observer.coefficients[static_cast<std::size_t>(feature)] =
        static_cast<float>(
            coefficients[static_cast<std::size_t>(feature + 1)]);
  }
  observer.training_labels.reserve(training.trials.size());
  observer.training_probabilities.reserve(training.trials.size());
  for (const FrozenTrialResult& trial : training.trials) {
    observer.training_labels.push_back(
        trial.label != 0 ? 1u : 0u);
    observer.training_probabilities.push_back(
        observer_probability(observer, trial.features));
  }
  observer.holdout_labels.reserve(holdout.trials.size());
  observer.holdout_probabilities.reserve(holdout.trials.size());
  double brier = 0.0;
  for (const FrozenTrialResult& trial : holdout.trials) {
    const std::uint8_t label = trial.label != 0 ? 1u : 0u;
    const float probability =
        observer_probability(observer, trial.features);
    observer.holdout_labels.push_back(label);
    observer.holdout_probabilities.push_back(probability);
    const double residual =
        static_cast<double>(probability) - label;
    brier += residual * residual /
             static_cast<double>(holdout.trials.size());
  }
  observer.auroc =
      binary_auroc(
          observer.holdout_labels, observer.holdout_probabilities);
  observer.brier = static_cast<float>(brier);
  fit_calibration_line(
      observer.holdout_labels, observer.holdout_probabilities,
      &observer.calibration_intercept, &observer.calibration_slope);
  double calibration_error = 0.0;
  for (int bin = 0; bin < 10; ++bin) {
    int count = 0;
    double probability_sum = 0.0;
    double label_sum = 0.0;
    for (std::size_t index = 0;
         index < observer.holdout_probabilities.size(); ++index) {
      const float probability =
          observer.holdout_probabilities[index];
      int probability_bin =
          std::min(
              static_cast<int>(probability * 10.0f), 9);
      if (probability_bin == bin) {
        ++count;
        probability_sum += probability;
        label_sum += observer.holdout_labels[index];
      }
    }
    if (count > 0) {
      calibration_error +=
          static_cast<double>(count) /
          observer.holdout_probabilities.size() *
          std::abs(
              probability_sum / count - label_sum / count);
    }
  }
  observer.expected_calibration_error =
      static_cast<float>(calibration_error);
  return observer;
}

struct MeanSem {
  float mean = 0.0f;
  float sem = 0.0f;
};

MeanSem mean_sem(const std::vector<float>& values) {
  MeanSem result{};
  if (values.empty()) {
    return result;
  }
  double mean = 0.0;
  for (float value : values) {
    mean += value / static_cast<double>(values.size());
  }
  double sum_squared = 0.0;
  for (float value : values) {
    const double centered = value - mean;
    sum_squared += centered * centered;
  }
  result.mean = static_cast<float>(mean);
  if (values.size() > 1) {
    const double variance =
        sum_squared / static_cast<double>(values.size() - 1);
    result.sem = static_cast<float>(
        std::sqrt(variance / values.size()));
  }
  return result;
}

const FrozenTrialResult& batch_trial(
    const FrozenTrialBatch& batch, int seed_index,
    int condition_index, int trial_index) {
  const std::size_t index =
      (static_cast<std::size_t>(seed_index) *
           batch.condition_count +
       static_cast<std::size_t>(condition_index)) *
          batch.trials_per_condition +
      static_cast<std::size_t>(trial_index);
  return batch.trials.at(index);
}

using MsiPopulationBins =
    std::array<float, kEvaluationFeatureBins>;

MsiPopulationBins smoothed_msi_population_bins(
    const FrozenTrialResult& trial) {
  MsiPopulationBins smoothed{};
  constexpr int prestimulus_bins =
      100 / kEvaluationBinWidthMs;
  for (int bin = 0; bin < kEvaluationFeatureBins; ++bin) {
    const std::size_t position = static_cast<std::size_t>(bin);
    if (bin < prestimulus_bins) {
      smoothed[position] = trial.features[position];
      continue;
    }
    const float previous =
        trial.features[static_cast<std::size_t>(
            bin > 0 ? bin - 1 : bin)];
    const float current = trial.features[position];
    const float next =
        trial.features[static_cast<std::size_t>(
            bin + 1 < kEvaluationFeatureBins ? bin + 1 : bin)];
    smoothed[position] =
        0.25f * previous + 0.50f * current + 0.25f * next;
  }
  return smoothed;
}

MsiPopulationBins pooled_control_mean(
    const FrozenTrialBatch& batch, int condition_index,
    bool smooth) {
  MsiPopulationBins mean{};
  const int trial_count =
      batch.model_seed_count * batch.trials_per_condition;
  for (int seed = 0; seed < batch.model_seed_count; ++seed) {
    for (int trial = 0; trial < batch.trials_per_condition; ++trial) {
      const FrozenTrialResult& value =
          batch_trial(batch, seed, condition_index, trial);
      const MsiPopulationBins bins =
          smooth ? smoothed_msi_population_bins(value)
                 : MsiPopulationBins(value.features);
      for (int bin = 0; bin < kEvaluationFeatureBins; ++bin) {
        mean[static_cast<std::size_t>(bin)] +=
            bins[static_cast<std::size_t>(bin)] /
            static_cast<float>(trial_count);
      }
    }
  }
  return mean;
}

NeuralResponseWindow primary_response_window(
    const MsiPopulationBins& mean, float threshold,
    const char* modality) {
  constexpr int first_response_bin =
      100 / kEvaluationBinWidthMs;
  constexpr int primary_response_end_bin =
      (100 + 250 + kEvaluationBinWidthMs - 1) /
      kEvaluationBinWidthMs;
  int peak_bin = first_response_bin;
  for (int bin = first_response_bin + 1;
       bin < primary_response_end_bin; ++bin) {
    if (mean[static_cast<std::size_t>(bin)] >
        mean[static_cast<std::size_t>(peak_bin)]) {
      peak_bin = bin;
    }
  }
  if (!(mean[static_cast<std::size_t>(peak_bin)] > threshold)) {
    throw std::runtime_error(
        std::string("Frozen ") + modality +
        " control has no supra-threshold primary MSI-E response.");
  }
  NeuralResponseWindow window{peak_bin, peak_bin};
  while (window.first_bin > first_response_bin &&
         mean[static_cast<std::size_t>(window.first_bin - 1)] >
             threshold) {
    --window.first_bin;
  }
  while (window.last_bin + 1 < kEvaluationFeatureBins &&
         mean[static_cast<std::size_t>(window.last_bin + 1)] >
             threshold) {
    ++window.last_bin;
  }
  return window;
}

NeuralFusionRule estimate_neural_fusion_rule_impl(
    const FrozenTrialBatch& batch, int auditory_condition,
    int visual_condition) {
  const MsiPopulationBins auditory_raw =
      pooled_control_mean(batch, auditory_condition, false);
  const MsiPopulationBins visual_raw =
      pooled_control_mean(batch, visual_condition, false);
  constexpr int prestimulus_bins =
      100 / kEvaluationBinWidthMs;
  double baseline_mean = 0.0;
  for (int bin = 0; bin < prestimulus_bins; ++bin) {
    baseline_mean +=
        0.5 *
        (auditory_raw[static_cast<std::size_t>(bin)] +
         visual_raw[static_cast<std::size_t>(bin)]) /
        static_cast<double>(prestimulus_bins);
  }
  double baseline_sum_squared = 0.0;
  for (int bin = 0; bin < prestimulus_bins; ++bin) {
    const double pooled_bin =
        0.5 *
        (auditory_raw[static_cast<std::size_t>(bin)] +
         visual_raw[static_cast<std::size_t>(bin)]);
    const double centered = pooled_bin - baseline_mean;
    baseline_sum_squared += centered * centered;
  }
  const double baseline_sd =
      prestimulus_bins > 1
          ? std::sqrt(
                baseline_sum_squared /
                static_cast<double>(prestimulus_bins - 1))
          : 0.0;
  NeuralFusionRule rule{};
  rule.threshold =
      static_cast<float>(baseline_mean + 3.0 * baseline_sd);
  if (!std::isfinite(rule.threshold) || rule.threshold < 0.0f) {
    throw std::runtime_error(
        "Frozen control prestimulus MSI-E threshold is invalid.");
  }
  // Control-window detection uses the already pooled raw 20-ms PSTHs.
  // Smoothing remains confined to per-trial episode classification.
  rule.auditory =
      primary_response_window(
          auditory_raw, rule.threshold, "auditory-only");
  rule.visual =
      primary_response_window(
          visual_raw, rule.threshold, "visual-only");
  return rule;
}

bool neural_fusion_event(
    const FrozenTrialResult& trial,
    const NeuralFusionRule& rule) {
  if (trial.auditory_received_onset_ms < 0 ||
      trial.visual_received_onset_ms < 0) {
    return false;
  }
  struct TimeWindow {
    int begin_ms;
    int end_ms;
  };
  const auto shifted_window =
      [](const NeuralResponseWindow& window, int onset_ms) {
        return TimeWindow{
            onset_ms - 100 +
                window.first_bin * kEvaluationBinWidthMs,
            onset_ms - 100 +
                (window.last_bin + 1) * kEvaluationBinWidthMs};
      };
  const TimeWindow auditory =
      shifted_window(
          rule.auditory, trial.auditory_received_onset_ms);
  const TimeWindow visual =
      shifted_window(
          rule.visual, trial.visual_received_onset_ms);
  const TimeWindow response_span{
      std::min(auditory.begin_ms, visual.begin_ms),
      std::max(auditory.end_ms, visual.end_ms)};
  const auto overlaps =
      [](int begin_ms, int end_ms, const TimeWindow& window) {
        return begin_ms < window.end_ms &&
               end_ms > window.begin_ms;
      };

  const MsiPopulationBins bins =
      smoothed_msi_population_bins(trial);
  constexpr int first_response_bin =
      100 / kEvaluationBinWidthMs;
  int relevant_episode_count = 0;
  bool single_overlaps_auditory = false;
  bool single_overlaps_visual = false;
  for (int bin = first_response_bin;
       bin < kEvaluationFeatureBins;) {
    if (!(bins[static_cast<std::size_t>(bin)] >
          rule.threshold)) {
      ++bin;
      continue;
    }
    const int first_bin = bin;
    while (bin + 1 < kEvaluationFeatureBins &&
           bins[static_cast<std::size_t>(bin + 1)] >
               rule.threshold) {
      ++bin;
    }
    const int last_bin = bin;
    const int begin_ms =
        -100 + first_bin * kEvaluationBinWidthMs;
    const int end_ms =
        -100 + (last_bin + 1) * kEvaluationBinWidthMs;
    const bool overlaps_auditory =
        overlaps(begin_ms, end_ms, auditory);
    const bool overlaps_visual =
        overlaps(begin_ms, end_ms, visual);
    if (overlaps(begin_ms, end_ms, response_span)) {
      ++relevant_episode_count;
      single_overlaps_auditory = overlaps_auditory;
      single_overlaps_visual = overlaps_visual;
    }
    ++bin;
  }
  return relevant_episode_count == 1 &&
         single_overlaps_auditory &&
         single_overlaps_visual;
}

struct ShapeFit {
  double baseline = 0.0;
  double amplitude = 0.0;
  double mse = DBL_MAX;
};

ShapeFit fit_linear_shape(
    const std::vector<float>& values,
    const std::vector<double>& shape, bool probability_bounds) {
  double sum_shape = 0.0;
  double sum_shape_squared = 0.0;
  double sum_value = 0.0;
  double sum_shape_value = 0.0;
  for (std::size_t index = 0; index < values.size(); ++index) {
    sum_shape += shape[index];
    sum_shape_squared += shape[index] * shape[index];
    sum_value += values[index];
    sum_shape_value += shape[index] * values[index];
  }
  const double count = static_cast<double>(values.size());
  const double determinant =
      count * sum_shape_squared - sum_shape * sum_shape;
  ShapeFit fit{};
  if (std::abs(determinant) > 1.0e-14) {
    fit.baseline =
        (sum_value * sum_shape_squared -
         sum_shape * sum_shape_value) /
        determinant;
    fit.amplitude =
        (count * sum_shape_value - sum_shape * sum_value) /
        determinant;
  } else {
    fit.baseline = sum_value / count;
    fit.amplitude = 0.0;
  }
  if (fit.amplitude < 0.0) {
    fit.amplitude = 0.0;
    fit.baseline = sum_value / count;
  }
  if (probability_bounds) {
    fit.baseline = std::min(std::max(fit.baseline, 0.0), 1.0);
    fit.amplitude =
        std::min(
            std::max(fit.amplitude, 0.0),
            1.0 - fit.baseline);
  }
  double error = 0.0;
  for (std::size_t index = 0; index < values.size(); ++index) {
    const double residual =
        values[index] -
        (fit.baseline + fit.amplitude * shape[index]);
    error += residual * residual;
  }
  fit.mse = error / count;
  return fit;
}

AsymmetricGaussianFit fit_asymmetric_gaussian(
    const std::array<float, kTbwPointCount>& locations,
    const std::array<TemporalCurvePoint, kTbwPointCount>& points) {
  std::vector<float> values(kTbwPointCount);
  for (int index = 0; index < kTbwPointCount; ++index) {
    values[static_cast<std::size_t>(index)] =
        points[static_cast<std::size_t>(index)].probability_mean;
  }
  double best_center = 0.0;
  double best_left = 100.0;
  double best_right = 100.0;
  ShapeFit best{};
  const std::array<double, 10> sigmas{
      12.5, 25.0, 37.5, 50.0, 75.0,
      100.0, 150.0, 225.0, 325.0, 450.0};
  std::vector<double> shape(kTbwPointCount);
  for (float center_value : locations) {
    for (double left : sigmas) {
      for (double right : sigmas) {
        for (int index = 0; index < kTbwPointCount; ++index) {
          const double offset =
              locations[static_cast<std::size_t>(index)] -
              center_value;
          const double sigma = offset < 0.0 ? left : right;
          shape[static_cast<std::size_t>(index)] =
              std::exp(-0.5 * offset * offset / (sigma * sigma));
        }
        const ShapeFit candidate =
            fit_linear_shape(values, shape, true);
        if (candidate.mse < best.mse) {
          best = candidate;
          best_center = center_value;
          best_left = left;
          best_right = right;
        }
      }
    }
  }
  double center_step = 25.0;
  double log_sigma_step = std::log(1.5);
  for (int refinement = 0; refinement < 10; ++refinement) {
    double next_center = best_center;
    double next_left = best_left;
    double next_right = best_right;
    ShapeFit next_fit = best;
    for (int center_delta = -1; center_delta <= 1; ++center_delta) {
      for (int left_delta = -1; left_delta <= 1; ++left_delta) {
        for (int right_delta = -1; right_delta <= 1; ++right_delta) {
          const double center =
              std::min(
                  std::max(
                      best_center + center_delta * center_step,
                      -500.0),
                  500.0);
          const double left =
              std::min(
                  std::max(
                      best_left *
                          std::exp(left_delta * log_sigma_step),
                      1.0),
                  1000.0);
          const double right =
              std::min(
                  std::max(
                      best_right *
                          std::exp(right_delta * log_sigma_step),
                      1.0),
                  1000.0);
          for (int index = 0; index < kTbwPointCount; ++index) {
            const double offset =
                locations[static_cast<std::size_t>(index)] - center;
            const double sigma = offset < 0.0 ? left : right;
            shape[static_cast<std::size_t>(index)] =
                std::exp(
                    -0.5 * offset * offset / (sigma * sigma));
          }
          const ShapeFit candidate =
              fit_linear_shape(values, shape, true);
          if (candidate.mse < next_fit.mse) {
            next_fit = candidate;
            next_center = center;
            next_left = left;
            next_right = right;
          }
        }
      }
    }
    best = next_fit;
    best_center = next_center;
    best_left = next_left;
    best_right = next_right;
    center_step *= 0.5;
    log_sigma_step *= 0.5;
  }
  AsymmetricGaussianFit fit{};
  fit.valid =
      std::isfinite(best.mse) && best.amplitude > 1.0e-6 &&
      best_left > 0.0 && best_right > 0.0;
  fit.baseline = static_cast<float>(best.baseline);
  fit.amplitude = static_cast<float>(best.amplitude);
  fit.center_ms = static_cast<float>(best_center);
  fit.sigma_left_ms = static_cast<float>(best_left);
  fit.sigma_right_ms = static_cast<float>(best_right);
  fit.mse = static_cast<float>(best.mse);
  if (fit.valid) {
    const double offset50 = std::sqrt(-2.0 * std::log(0.50));
    const double offset75 = std::sqrt(-2.0 * std::log(0.75));
    fit.left_50_ms =
        static_cast<float>(best_center - best_left * offset50);
    fit.right_50_ms =
        static_cast<float>(best_center + best_right * offset50);
    fit.left_75_ms =
        static_cast<float>(best_center - best_left * offset75);
    fit.right_75_ms =
        static_cast<float>(best_center + best_right * offset75);
    fit.tbw50_ms = fit.right_50_ms - fit.left_50_ms;
    fit.tbw75_ms = fit.right_75_ms - fit.left_75_ms;
    fit.crossings_valid =
        fit.left_50_ms >= locations.front() &&
        fit.right_50_ms <= locations.back() &&
        fit.left_75_ms >= locations.front() &&
        fit.right_75_ms <= locations.back();
  }
  return fit;
}

ComponentResponseMetrics component_response_metrics(
    const FrozenTrialBatch& batch, int auditory_condition,
    int visual_condition, int audiovisual_condition,
    float response_floor_hz, bool matched_group_metrics) {
  std::vector<float> auditory;
  std::vector<float> visual;
  std::vector<float> audiovisual;
  std::vector<float> enhancements;
  const int count =
      batch.model_seed_count * batch.trials_per_condition;
  auditory.reserve(static_cast<std::size_t>(count));
  visual.reserve(static_cast<std::size_t>(count));
  audiovisual.reserve(static_cast<std::size_t>(count));
  enhancements.reserve(static_cast<std::size_t>(count));
  for (int seed = 0; seed < batch.model_seed_count; ++seed) {
    for (int trial = 0; trial < batch.trials_per_condition; ++trial) {
      const float rate_a =
          batch_trial(
              batch, seed, auditory_condition, trial)
              .response_rate_hz;
      const float rate_v =
          batch_trial(
              batch, seed, visual_condition, trial)
              .response_rate_hz;
      const float rate_av =
          batch_trial(
              batch, seed, audiovisual_condition, trial)
              .response_rate_hz;
      auditory.push_back(rate_a);
      visual.push_back(rate_v);
      audiovisual.push_back(rate_av);
      enhancements.push_back(rate_av - std::max(rate_a, rate_v));
    }
  }
  if (matched_group_metrics) {
    return summarize_matched_component_trials(
        auditory, visual, audiovisual, response_floor_hz);
  }
  const MeanSem mean_a = mean_sem(auditory);
  const MeanSem mean_v = mean_sem(visual);
  const MeanSem mean_av = mean_sem(audiovisual);
  const MeanSem mean_g = mean_sem(enhancements);
  ComponentResponseMetrics metrics{};
  metrics.auditory_rate_hz = mean_a.mean;
  metrics.visual_rate_hz = mean_v.mean;
  metrics.audiovisual_rate_hz = mean_av.mean;
  metrics.raw_enhancement_hz = mean_g.mean;
  metrics.raw_enhancement_sem = mean_g.sem;
  const float strongest_component =
      std::max(mean_a.mean, mean_v.mean);
  metrics.multisensory_enhancement_percent =
      100.0f * mean_g.mean /
      std::max(strongest_component, response_floor_hz);
  metrics.additivity_hz =
      mean_av.mean - (mean_a.mean + mean_v.mean);
  metrics.additivity_percent =
      100.0f * metrics.additivity_hz /
      std::max(
          mean_a.mean + mean_v.mean, response_floor_hz);
  return metrics;
}

SymmetricGaussianFit fit_spatial_gaussian(
    const std::array<float, kSbwPointCount>& disparities,
    const std::array<SpatialCurvePoint, kSbwPointCount>& points) {
  const double baseline =
      (points[kSbwPointCount - 3].pooled.raw_enhancement_hz +
       points[kSbwPointCount - 2].pooled.raw_enhancement_hz +
       points[kSbwPointCount - 1].pooled.raw_enhancement_hz) /
      3.0;
  const double amplitude =
      points[0].pooled.raw_enhancement_hz - baseline;
  SymmetricGaussianFit fit{};
  fit.baseline = static_cast<float>(baseline);
  fit.amplitude = static_cast<float>(amplitude);
  if (!(amplitude > 1.0e-6)) {
    return fit;
  }
  double best_sigma = 1.0;
  double best_mse = DBL_MAX;
  for (int candidate = 0; candidate <= 1190; ++candidate) {
    const double sigma = 1.0 + 0.1 * candidate;
    double error = 0.0;
    for (int index = 0; index < kSbwPointCount; ++index) {
      const double disparity =
          disparities[static_cast<std::size_t>(index)];
      const double predicted =
          baseline +
          amplitude *
              std::exp(
                  -0.5 * disparity * disparity /
                  (sigma * sigma));
      const double residual =
          points[static_cast<std::size_t>(index)]
              .pooled.raw_enhancement_hz -
          predicted;
      error += residual * residual;
    }
    const double mse = error / kSbwPointCount;
    if (mse < best_mse) {
      best_mse = mse;
      best_sigma = sigma;
    }
  }
  fit.sigma_deg = static_cast<float>(best_sigma);
  fit.mse = static_cast<float>(best_mse);
  fit.fitted_hwhm_deg = static_cast<float>(
      best_sigma * std::sqrt(2.0 * std::log(2.0)));
  fit.valid =
      best_mse < DBL_MAX &&
      std::isfinite(best_mse) &&
      std::isfinite(fit.mse);
  fit.hwhm_valid =
      fit.valid && std::isfinite(fit.fitted_hwhm_deg);
  return fit;
}

ControlledCondition controlled_condition(
    bool auditory_present, bool visual_present,
    float auditory_location_deg, float visual_location_deg,
    float auditory_rate_hz, float visual_rate_hz,
    float physical_soa_ms, int label, int random_group,
    CausalControl control) {
  ControlledCondition condition{};
  condition.auditory_present = auditory_present;
  condition.visual_present = visual_present;
  condition.auditory_location_deg = auditory_location_deg;
  condition.visual_location_deg = visual_location_deg;
  condition.auditory_rate_hz = auditory_rate_hz;
  condition.visual_rate_hz = visual_rate_hz;
  condition.physical_soa_ms = physical_soa_ms;
  condition.label = label;
  condition.random_group = random_group;
  condition.control = control;
  return condition;
}

ProjectionControlAudit projection_control_audit(
    CausalControl control) {
  ProjectionControlAudit audit{};
  audit.control = control;
  for (int path = 0; path < 6; ++path) {
    audit.ampa_scale[static_cast<std::size_t>(path)] = 1.0f;
    audit.nmda_scale[static_cast<std::size_t>(path)] = 1.0f;
  }
  audit.gabaa_scale[6] = 1.0f;
  if (has_control(control, CausalControl::kNmdaOff)) {
    audit.nmda_scale.fill(0.0f);
    audit.external_nmda_enabled = false;
    audit.background_nmda_enabled = false;
  }
  if (has_control(control, CausalControl::kGabaaOff)) {
    audit.gabaa_scale[6] = 0.0f;
  }
  if (has_control(
          control, CausalControl::kRecurrentExcitationOff)) {
    audit.ampa_scale[2] = 0.0f;
    audit.nmda_scale[2] = 0.0f;
  }
  if (has_control(
          control, CausalControl::kRecruitedInhibitionOff)) {
    for (const int path : {3, 4}) {
      audit.ampa_scale[static_cast<std::size_t>(path)] = 0.0f;
      audit.nmda_scale[static_cast<std::size_t>(path)] = 0.0f;
    }
    audit.gabaa_scale[6] = 0.0f;
  }
  if (has_control(control, CausalControl::kSourceRowShuffle)) {
    for (const int path : {0, 1, 3, 4}) {
      audit.source_rows_shuffled[
          static_cast<std::size_t>(path)] = true;
    }
  }
  return audit;
}

struct WeightedCenterWidth {
  float center = 0.0f;
  float width = 0.0f;
  bool valid = false;
};

template <std::size_t Count>
WeightedCenterWidth weighted_center_width(
    const std::array<float, Count>& values,
    const std::array<float, Count>& coordinates) {
  double total = 0.0;
  double center = 0.0;
  for (std::size_t index = 0; index < Count; ++index) {
    const double value = std::max(values[index], 0.0f);
    total += value;
    center += value * coordinates[index];
  }
  if (!(total > 1.0e-9)) {
    return {};
  }
  center /= total;
  double variance = 0.0;
  for (std::size_t index = 0; index < Count; ++index) {
    const double value = std::max(values[index], 0.0f);
    const double offset = coordinates[index] - center;
    variance += value * offset * offset;
  }
  WeightedCenterWidth result{};
  result.center = static_cast<float>(center);
  result.width =
      static_cast<float>(std::sqrt(std::max(variance / total, 0.0)));
  result.valid =
      std::isfinite(result.center) && std::isfinite(result.width);
  return result;
}

float pearson_correlation(
    const std::vector<float>& first,
    const std::vector<float>& second) {
  if (first.size() != second.size() || first.size() < 2) {
    return 0.0f;
  }
  double mean_first = 0.0;
  double mean_second = 0.0;
  for (std::size_t index = 0; index < first.size(); ++index) {
    mean_first += first[index] / static_cast<double>(first.size());
    mean_second += second[index] / static_cast<double>(second.size());
  }
  double covariance = 0.0;
  double variance_first = 0.0;
  double variance_second = 0.0;
  for (std::size_t index = 0; index < first.size(); ++index) {
    const double centered_first = first[index] - mean_first;
    const double centered_second = second[index] - mean_second;
    covariance += centered_first * centered_second;
    variance_first += centered_first * centered_first;
    variance_second += centered_second * centered_second;
  }
  const double denominator =
      std::sqrt(variance_first * variance_second);
  return denominator > 0.0
             ? static_cast<float>(covariance / denominator)
             : 0.0f;
}

float median_value(std::vector<float> values) {
  if (values.empty()) {
    return 0.0f;
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 1) {
    return values[middle];
  }
  return 0.5f * (values[middle - 1] + values[middle]);
}

RfTopographyAudit compute_rf_topography(
    const FrozenTrialBatch& batch,
    const std::vector<FrozenWeightAudit>& weight_audits,
    int configured_trials, int selected_seed = -1) {
  if (batch.model_seed_count !=
          static_cast<int>(weight_audits.size()) ||
      selected_seed < -1 ||
      selected_seed >= batch.model_seed_count) {
    throw std::invalid_argument(
        "RF topography requires matching batch/weight seeds.");
  }
  const int first_seed = selected_seed < 0 ? 0 : selected_seed;
  const int final_seed =
      selected_seed < 0 ? batch.model_seed_count
                        : selected_seed + 1;
  const double selected_seed_count =
      static_cast<double>(final_seed - first_seed);
  RfTopographyAudit audit{};
  audit.trials_per_location = configured_trials;
  std::array<float, kRfPointCount> locations{};
  for (int point = 0; point < kRfPointCount; ++point) {
    locations[static_cast<std::size_t>(point)] =
        -60.0f + 5.0f * static_cast<float>(point);
  }
  audit.location_grid_deg = locations;

  std::vector<float> target_coordinates;
  std::vector<float> auditory_rf_centers;
  std::vector<float> visual_rf_centers;
  std::vector<float> mismatch;
  int jointly_valid = 0;
  for (int neuron = 0; neuron < kExcitatoryNeurons; ++neuron) {
    std::array<float, kRfPointCount> auditory_response{};
    std::array<float, kRfPointCount> visual_response{};
    for (int point = 0; point < kRfPointCount; ++point) {
      double auditory_sum = 0.0;
      double visual_sum = 0.0;
      int count = 0;
      for (int seed = first_seed; seed < final_seed; ++seed) {
        for (int trial = 0; trial < batch.trials_per_condition; ++trial) {
          const FrozenTrialResult& auditory =
              batch_trial(batch, seed, point * 2, trial);
          const FrozenTrialResult& visual =
              batch_trial(batch, seed, point * 2 + 1, trial);
          const double auditory_rate =
              4.0 *
                  auditory.excitatory_response_spikes_per_neuron[
                      static_cast<std::size_t>(neuron)] -
              10.0 *
                  auditory.excitatory_baseline_spikes_per_neuron[
                      static_cast<std::size_t>(neuron)];
          const double visual_rate =
              4.0 *
                  visual.excitatory_response_spikes_per_neuron[
                      static_cast<std::size_t>(neuron)] -
              10.0 *
                  visual.excitatory_baseline_spikes_per_neuron[
                      static_cast<std::size_t>(neuron)];
          auditory_sum += auditory_rate;
          visual_sum += visual_rate;
          ++count;
        }
      }
      auditory_response[static_cast<std::size_t>(point)] =
          finalize_rf_signed_rate_mean(auditory_sum, count);
      visual_response[static_cast<std::size_t>(point)] =
          finalize_rf_signed_rate_mean(visual_sum, count);
      audit.auditory_response_hz[
          static_cast<std::size_t>(neuron)]
          [static_cast<std::size_t>(point)] =
          auditory_response[static_cast<std::size_t>(point)];
      audit.visual_response_hz[
          static_cast<std::size_t>(neuron)]
          [static_cast<std::size_t>(point)] =
          visual_response[static_cast<std::size_t>(point)];
    }
    const WeightedCenterWidth auditory =
        weighted_center_width(auditory_response, locations);
    const WeightedCenterWidth visual =
        weighted_center_width(visual_response, locations);
    audit.auditory_rf_center_deg[static_cast<std::size_t>(neuron)] =
        auditory.center;
    audit.auditory_rf_width_deg[static_cast<std::size_t>(neuron)] =
        auditory.width;
    audit.visual_rf_center_deg[static_cast<std::size_t>(neuron)] =
        visual.center;
    audit.visual_rf_width_deg[static_cast<std::size_t>(neuron)] =
        visual.width;
    if (auditory.valid && visual.valid) {
      const float target = -89.5f + static_cast<float>(neuron);
      const float difference = std::abs(auditory.center - visual.center);
      audit.rf_center_mismatch_deg[
          static_cast<std::size_t>(neuron)] = difference;
      target_coordinates.push_back(target);
      auditory_rf_centers.push_back(auditory.center);
      visual_rf_centers.push_back(visual.center);
      mismatch.push_back(difference);
      ++jointly_valid;
    }
  }
  audit.auditory_rf_order_correlation =
      pearson_correlation(target_coordinates, auditory_rf_centers);
  audit.visual_rf_order_correlation =
      pearson_correlation(target_coordinates, visual_rf_centers);
  audit.median_rf_center_mismatch_deg = median_value(mismatch);
  audit.finite_rf_coverage =
      static_cast<float>(jointly_valid) /
      static_cast<float>(kExcitatoryNeurons);

  std::array<float, kExcitatoryNeurons> source_coordinates{};
  std::array<float, kExcitatoryNeurons> direct_a{};
  std::array<float, kExcitatoryNeurons> direct_v{};
  std::array<float, kExcitatoryNeurons> effective_a{};
  std::array<float, kExcitatoryNeurons> effective_v{};
  for (int source = 0; source < kExcitatoryNeurons; ++source) {
    source_coordinates[static_cast<std::size_t>(source)] =
        -89.5f + static_cast<float>(source);
  }
  std::vector<float> all_target_coordinates;
  std::vector<float> auditory_map_centers;
  std::vector<float> visual_map_centers;
  std::vector<float> auditory_effective_centers;
  std::vector<float> visual_effective_centers;
  std::vector<float> effective_alignment;
  for (int post = 0; post < kExcitatoryNeurons; ++post) {
    direct_a.fill(0.0f);
    direct_v.fill(0.0f);
    effective_a.fill(0.0f);
    effective_v.fill(0.0f);
    for (int source = 0; source < kExcitatoryNeurons; ++source) {
      double sum_direct_a = 0.0;
      double sum_direct_v = 0.0;
      double sum_effective_a = 0.0;
      double sum_effective_v = 0.0;
      for (int seed_index = first_seed;
           seed_index < final_seed; ++seed_index) {
        const FrozenWeightAudit& seed =
            weight_audits[static_cast<std::size_t>(seed_index)];
        const std::size_t direct_index =
            static_cast<std::size_t>(source) *
                kExcitatoryNeurons +
            post;
        if (seed.paths[0].masks[direct_index]) {
          sum_direct_a += seed.paths[0].weights[direct_index];
        }
        if (seed.paths[1].masks[direct_index]) {
          sum_direct_v += seed.paths[1].weights[direct_index];
        }
        for (int inhibitory = 0;
             inhibitory < kInhibitoryNeurons; ++inhibitory) {
          const std::size_t source_i =
              static_cast<std::size_t>(source) *
                  kInhibitoryNeurons +
              inhibitory;
          const std::size_t i_post =
              static_cast<std::size_t>(inhibitory) *
                  kExcitatoryNeurons +
              post;
          if (seed.paths[3].masks[source_i] &&
              seed.paths[6].masks[i_post]) {
            sum_effective_a +=
                seed.paths[3].weights[source_i] *
                seed.paths[6].weights[i_post];
          }
          if (seed.paths[4].masks[source_i] &&
              seed.paths[6].masks[i_post]) {
            sum_effective_v +=
                seed.paths[4].weights[source_i] *
                seed.paths[6].weights[i_post];
          }
        }
      }
      direct_a[static_cast<std::size_t>(source)] =
          static_cast<float>(sum_direct_a / selected_seed_count);
      direct_v[static_cast<std::size_t>(source)] =
          static_cast<float>(sum_direct_v / selected_seed_count);
      effective_a[static_cast<std::size_t>(source)] =
          static_cast<float>(sum_effective_a / selected_seed_count);
      effective_v[static_cast<std::size_t>(source)] =
          static_cast<float>(sum_effective_v / selected_seed_count);
    }
    const WeightedCenterWidth auditory =
        weighted_center_width(direct_a, source_coordinates);
    const WeightedCenterWidth visual =
        weighted_center_width(direct_v, source_coordinates);
    const WeightedCenterWidth inhibitory_a =
        weighted_center_width(effective_a, source_coordinates);
    const WeightedCenterWidth inhibitory_v =
        weighted_center_width(effective_v, source_coordinates);
    const std::size_t index = static_cast<std::size_t>(post);
    audit.auditory_excitatory_map_center_deg[index] = auditory.center;
    audit.auditory_excitatory_map_width_deg[index] = auditory.width;
    audit.visual_excitatory_map_center_deg[index] = visual.center;
    audit.visual_excitatory_map_width_deg[index] = visual.width;
    audit.auditory_effective_inhibitory_center_deg[index] =
        inhibitory_a.center;
    audit.auditory_effective_inhibitory_width_deg[index] =
        inhibitory_a.width;
    audit.visual_effective_inhibitory_center_deg[index] =
        inhibitory_v.center;
    audit.visual_effective_inhibitory_width_deg[index] =
        inhibitory_v.width;
    if (auditory.valid && visual.valid &&
        inhibitory_a.valid && inhibitory_v.valid) {
      all_target_coordinates.push_back(
          -89.5f + static_cast<float>(post));
      auditory_map_centers.push_back(auditory.center);
      visual_map_centers.push_back(visual.center);
      auditory_effective_centers.push_back(inhibitory_a.center);
      visual_effective_centers.push_back(inhibitory_v.center);
      effective_alignment.push_back(
          std::abs(auditory.center - inhibitory_a.center));
      effective_alignment.push_back(
          std::abs(visual.center - inhibitory_v.center));
    }
  }
  audit.auditory_weight_order_correlation =
      pearson_correlation(
          all_target_coordinates, auditory_map_centers);
  audit.visual_weight_order_correlation =
      pearson_correlation(
          all_target_coordinates, visual_map_centers);
  audit.effective_inhibitory_order_correlation =
      0.5f *
      (pearson_correlation(
           all_target_coordinates, auditory_effective_centers) +
       pearson_correlation(
           all_target_coordinates, visual_effective_centers));
  audit.median_effective_inhibitory_alignment_deg =
      median_value(effective_alignment);

  double count = 0.0;
  double sum_distance = 0.0;
  double sum_weight = 0.0;
  double sum_distance_squared = 0.0;
  double sum_weight_squared = 0.0;
  double sum_product = 0.0;
  for (int seed_index = first_seed;
       seed_index < final_seed; ++seed_index) {
    const FrozenWeightAudit& seed =
        weight_audits[static_cast<std::size_t>(seed_index)];
    for (int pre = 0; pre < kExcitatoryNeurons; ++pre) {
      for (int post = 0; post < kExcitatoryNeurons; ++post) {
        const std::size_t index =
            static_cast<std::size_t>(pre) *
                kExcitatoryNeurons +
            post;
        if (!seed.paths[2].masks[index]) {
          continue;
        }
        const double distance =
            -std::abs(static_cast<double>(pre - post));
        const double weight = seed.paths[2].weights[index];
        count += 1.0;
        sum_distance += distance;
        sum_weight += weight;
        sum_distance_squared += distance * distance;
        sum_weight_squared += weight * weight;
        sum_product += distance * weight;
      }
    }
  }
  const double covariance =
      count * sum_product - sum_distance * sum_weight;
  const double variance_distance =
      count * sum_distance_squared -
      sum_distance * sum_distance;
  const double variance_weight =
      count * sum_weight_squared - sum_weight * sum_weight;
  const double denominator =
      std::sqrt(variance_distance * variance_weight);
  audit.recurrent_weight_distance_correlation =
      denominator > 0.0
          ? static_cast<float>(covariance / denominator)
          : 0.0f;
  audit.map_monotonicity =
      std::min(
          std::min(
              std::abs(audit.auditory_weight_order_correlation),
              std::abs(audit.visual_weight_order_correlation)),
          std::abs(audit.effective_inhibitory_order_correlation));
  return audit;
}

bool checked_correlation_float(
    const std::vector<double>& first,
    const std::vector<double>& second, float* value) {
  double correlation = 0.0;
  if (!checked_pearson_correlation(
          first, second, &correlation) ||
      !std::isfinite(correlation)) {
    return false;
  }
  *value = static_cast<float>(correlation);
  return std::isfinite(*value);
}

bool checked_weighted_center(
    const std::vector<double>& weights,
    const std::vector<double>& coordinates, double* center) {
  if (center == nullptr || weights.size() != coordinates.size() ||
      weights.empty()) {
    return false;
  }
  double total = 0.0;
  double weighted_sum = 0.0;
  for (std::size_t index = 0; index < weights.size(); ++index) {
    if (!std::isfinite(weights[index]) ||
        !std::isfinite(coordinates[index]) ||
        weights[index] < 0.0) {
      return false;
    }
    total += weights[index];
    weighted_sum += weights[index] * coordinates[index];
  }
  if (!(total > 0.0) || !std::isfinite(total) ||
      !std::isfinite(weighted_sum)) {
    return false;
  }
  *center = weighted_sum / total;
  return std::isfinite(*center);
}

bool frozen_masks_exact(
    const FrozenWeightAudit& initial,
    const FrozenWeightAudit& trained) {
  for (int path = 0; path < kPathCount; ++path) {
    const FrozenPathAudit& first =
        initial.paths[static_cast<std::size_t>(path)];
    const FrozenPathAudit& second =
        trained.paths[static_cast<std::size_t>(path)];
    if (first.masks != second.masks ||
        first.weights.size() != second.weights.size() ||
        first.weights.size() != first.masks.size()) {
      return false;
    }
  }
  return true;
}

TopologyStateMetrics compute_weight_topology_state(
    const FrozenWeightAudit& common_initial,
    const FrozenWeightAudit& state_weights) {
  TopologyStateMetrics metrics{};
  const auto projection_correlation =
      [&](int path, bool exclude_self, float* value,
          int* contact_count) {
        const FrozenPathAudit& common =
            common_initial.paths[static_cast<std::size_t>(path)];
        const FrozenPathAudit& state =
            state_weights.paths[static_cast<std::size_t>(path)];
        const PathLayout layout = path_layout(path);
        std::vector<double> proximity;
        std::vector<double> efficacy;
        for (int pre = 0; pre < layout.pre_size; ++pre) {
          for (int post = 0; post < layout.post_size; ++post) {
            if (exclude_self && pre == post) {
              continue;
            }
            const std::size_t index =
                static_cast<std::size_t>(pre) *
                    layout.post_size +
                post;
            if (index >= common.masks.size() ||
                index >= state.weights.size() ||
                common.masks[index] == 0u) {
              continue;
            }
            proximity.push_back(
                -std::abs(static_cast<double>(pre - post)));
            efficacy.push_back(
                static_cast<double>(state.weights[index]));
          }
        }
        *contact_count = static_cast<int>(proximity.size());
        return checked_correlation_float(
            proximity, efficacy, value);
      };

  metrics.auditory_to_excitatory_valid =
      projection_correlation(
          0, false,
          &metrics
               .auditory_to_excitatory_proximity_efficacy_correlation,
          &metrics.auditory_to_excitatory_contact_count);
  metrics.visual_to_excitatory_valid =
      projection_correlation(
          1, false,
          &metrics
               .visual_to_excitatory_proximity_efficacy_correlation,
          &metrics.visual_to_excitatory_contact_count);
  metrics.recurrent_valid =
      projection_correlation(
          2, true,
          &metrics.recurrent_proximity_efficacy_correlation,
          &metrics.recurrent_contact_count);

  const FrozenPathAudit& auditory_to_inhibitory =
      common_initial.paths[3];
  const FrozenPathAudit& visual_to_inhibitory =
      common_initial.paths[4];
  const FrozenPathAudit& common_inhibitory_to_excitatory =
      common_initial.paths[6];
  const FrozenPathAudit& state_inhibitory_to_excitatory =
      state_weights.paths[6];
  std::vector<double> target_coordinates;
  std::vector<double> auditory_centers;
  std::vector<double> visual_centers;
  std::vector<double> sensory_coordinates(kExcitatoryNeurons);
  for (int source = 0; source < kExcitatoryNeurons; ++source) {
    sensory_coordinates[static_cast<std::size_t>(source)] =
        -89.5 + static_cast<double>(source);
  }
  for (int post = 0; post < kExcitatoryNeurons; ++post) {
    std::vector<double> auditory_kernel(kExcitatoryNeurons, 0.0);
    std::vector<double> visual_kernel(kExcitatoryNeurons, 0.0);
    for (int source = 0; source < kExcitatoryNeurons; ++source) {
      for (int inhibitory = 0;
           inhibitory < kInhibitoryNeurons; ++inhibitory) {
        const std::size_t source_i =
            static_cast<std::size_t>(source) *
                kInhibitoryNeurons +
            inhibitory;
        const std::size_t i_post =
            static_cast<std::size_t>(inhibitory) *
                kExcitatoryNeurons +
            post;
        if (i_post >= common_inhibitory_to_excitatory.masks.size() ||
            i_post >= state_inhibitory_to_excitatory.weights.size() ||
            common_inhibitory_to_excitatory.masks[i_post] == 0u) {
          continue;
        }
        const double inhibitory_weight =
            state_inhibitory_to_excitatory.weights[i_post];
        if (source_i < auditory_to_inhibitory.masks.size() &&
            auditory_to_inhibitory.masks[source_i] != 0u) {
          auditory_kernel[static_cast<std::size_t>(source)] +=
              static_cast<double>(
                  auditory_to_inhibitory.weights[source_i]) *
              inhibitory_weight;
        }
        if (source_i < visual_to_inhibitory.masks.size() &&
            visual_to_inhibitory.masks[source_i] != 0u) {
          visual_kernel[static_cast<std::size_t>(source)] +=
              static_cast<double>(
                  visual_to_inhibitory.weights[source_i]) *
              inhibitory_weight;
        }
      }
    }
    double auditory_center = 0.0;
    double visual_center = 0.0;
    if (checked_weighted_center(
            auditory_kernel, sensory_coordinates,
            &auditory_center) &&
        checked_weighted_center(
            visual_kernel, sensory_coordinates,
            &visual_center)) {
      target_coordinates.push_back(
          -89.5 + static_cast<double>(post));
      auditory_centers.push_back(auditory_center);
      visual_centers.push_back(visual_center);
    }
  }
  metrics.effective_inhibitory_common_target_count =
      static_cast<int>(target_coordinates.size());
  float auditory_order = 0.0f;
  float visual_order = 0.0f;
  const bool auditory_valid =
      checked_correlation_float(
          target_coordinates, auditory_centers,
          &auditory_order);
  const bool visual_valid =
      checked_correlation_float(
          target_coordinates, visual_centers,
          &visual_order);
  metrics.effective_inhibitory_valid =
      auditory_valid && visual_valid;
  if (metrics.effective_inhibitory_valid) {
    metrics.effective_inhibitory_order_correlation =
        0.5f * (auditory_order + visual_order);
  }
  return metrics;
}

void populate_paired_rf_state_metrics(
    const RfTopographyAudit& initial_rf,
    const RfTopographyAudit& trained_rf,
    TopologyStateMetrics* initial_state,
    TopologyStateMetrics* trained_state) {
  std::vector<double> target_coordinates;
  std::vector<double> initial_auditory_centers;
  std::vector<double> initial_visual_centers;
  std::vector<double> trained_auditory_centers;
  std::vector<double> trained_visual_centers;
  std::vector<float> initial_mismatch;
  std::vector<float> trained_mismatch;
  for (int neuron = 0; neuron < kExcitatoryNeurons; ++neuron) {
    const std::size_t index = static_cast<std::size_t>(neuron);
    const WeightedCenterWidth initial_auditory =
        weighted_center_width(
            initial_rf.auditory_response_hz[index],
            initial_rf.location_grid_deg);
    const WeightedCenterWidth initial_visual =
        weighted_center_width(
            initial_rf.visual_response_hz[index],
            initial_rf.location_grid_deg);
    const WeightedCenterWidth trained_auditory =
        weighted_center_width(
            trained_rf.auditory_response_hz[index],
            trained_rf.location_grid_deg);
    const WeightedCenterWidth trained_visual =
        weighted_center_width(
            trained_rf.visual_response_hz[index],
            trained_rf.location_grid_deg);
    if (!initial_auditory.valid || !initial_visual.valid ||
        !trained_auditory.valid || !trained_visual.valid) {
      continue;
    }
    target_coordinates.push_back(
        -89.5 + static_cast<double>(neuron));
    initial_auditory_centers.push_back(initial_auditory.center);
    initial_visual_centers.push_back(initial_visual.center);
    trained_auditory_centers.push_back(trained_auditory.center);
    trained_visual_centers.push_back(trained_visual.center);
    initial_mismatch.push_back(
        std::abs(initial_auditory.center - initial_visual.center));
    trained_mismatch.push_back(
        std::abs(trained_auditory.center - trained_visual.center));
  }
  const int common_count =
      static_cast<int>(target_coordinates.size());
  initial_state->rf_common_neuron_count = common_count;
  trained_state->rf_common_neuron_count = common_count;
  initial_state->auditory_rf_order_valid =
      checked_correlation_float(
          target_coordinates, initial_auditory_centers,
          &initial_state->auditory_rf_order_correlation);
  initial_state->visual_rf_order_valid =
      checked_correlation_float(
          target_coordinates, initial_visual_centers,
          &initial_state->visual_rf_order_correlation);
  trained_state->auditory_rf_order_valid =
      checked_correlation_float(
          target_coordinates, trained_auditory_centers,
          &trained_state->auditory_rf_order_correlation);
  trained_state->visual_rf_order_valid =
      checked_correlation_float(
          target_coordinates, trained_visual_centers,
          &trained_state->visual_rf_order_correlation);
  initial_state->median_rf_center_mismatch_deg =
      median_value(initial_mismatch);
  trained_state->median_rf_center_mismatch_deg =
      median_value(trained_mismatch);
  initial_state->rf_mismatch_valid =
      !initial_mismatch.empty() &&
      std::isfinite(
          initial_state->median_rf_center_mismatch_deg);
  trained_state->rf_mismatch_valid =
      !trained_mismatch.empty() &&
      std::isfinite(
          trained_state->median_rf_center_mismatch_deg);
}

void populate_seed_topology_audit(
    const SeedWeightStateAudit& weights,
    const RfTopographyAudit& initial_rf,
    const RfTopographyAudit& trained_rf,
    SeedTopologyAudit* topology) {
  topology->initial = initial_rf;
  topology->trained = trained_rf;
  topology->initial_trained_masks_exact =
      frozen_masks_exact(weights.initial, weights.trained);
  topology->auditory_to_inhibitory_weights_exact =
      weights.initial.paths[3].weights ==
      weights.trained.paths[3].weights;
  topology->visual_to_inhibitory_weights_exact =
      weights.initial.paths[4].weights ==
      weights.trained.paths[4].weights;
  topology->initial_state =
      compute_weight_topology_state(
          weights.initial, weights.initial);
  topology->trained_state =
      compute_weight_topology_state(
          weights.initial, weights.trained);
  populate_paired_rf_state_metrics(
      initial_rf, trained_rf,
      &topology->initial_state,
      &topology->trained_state);
  topology->auditory_to_excitatory_proximity_efficacy_delta =
      topology->trained_state
          .auditory_to_excitatory_proximity_efficacy_correlation -
      topology->initial_state
          .auditory_to_excitatory_proximity_efficacy_correlation;
  topology->visual_to_excitatory_proximity_efficacy_delta =
      topology->trained_state
          .visual_to_excitatory_proximity_efficacy_correlation -
      topology->initial_state
          .visual_to_excitatory_proximity_efficacy_correlation;
  topology->recurrent_proximity_efficacy_delta =
      topology->trained_state
          .recurrent_proximity_efficacy_correlation -
      topology->initial_state
          .recurrent_proximity_efficacy_correlation;
  topology->effective_inhibitory_order_delta_explicit =
      topology->trained_state
          .effective_inhibitory_order_correlation -
      topology->initial_state
          .effective_inhibitory_order_correlation;
  topology->median_rf_center_mismatch_delta_deg =
      topology->trained_state.median_rf_center_mismatch_deg -
      topology->initial_state.median_rf_center_mismatch_deg;
  const TopologyStateMetrics& initial = topology->initial_state;
  const TopologyStateMetrics& trained = topology->trained_state;
  topology->all_required_metrics_valid =
      initial.auditory_to_excitatory_valid &&
      initial.visual_to_excitatory_valid &&
      initial.recurrent_valid &&
      initial.effective_inhibitory_valid &&
      initial.auditory_rf_order_valid &&
      initial.visual_rf_order_valid &&
      initial.rf_mismatch_valid &&
      trained.auditory_to_excitatory_valid &&
      trained.visual_to_excitatory_valid &&
      trained.recurrent_valid &&
      trained.effective_inhibitory_valid &&
      trained.auditory_rf_order_valid &&
      trained.visual_rf_order_valid &&
      trained.rf_mismatch_valid &&
      std::isfinite(
          topology
              ->auditory_to_excitatory_proximity_efficacy_delta) &&
      std::isfinite(
          topology
              ->visual_to_excitatory_proximity_efficacy_delta) &&
      std::isfinite(
          topology->recurrent_proximity_efficacy_delta) &&
      std::isfinite(
          topology->effective_inhibitory_order_delta_explicit) &&
      std::isfinite(
          topology->median_rf_center_mismatch_delta_deg);
}

TopologyRefinementSummary summarize_topology_refinement(
    const std::vector<SeedEvaluationMetrics>& per_seed,
    const Config& config, CausalControl control,
    const std::vector<TrainingDiagnostics>& diagnostics,
    bool pruning_ever_enabled) {
  TopologyRefinementSummary summary{};
  summary.seed_count = static_cast<int>(per_seed.size());
  summary.training_cohort = config.training_cohort;
  summary.control = control;
  summary.pruning_ever_enabled = pruning_ever_enabled;
  summary.minimum_presentations_per_seed =
      diagnostics.empty()
          ? 0
          : std::numeric_limits<int>::max();
  for (const TrainingDiagnostics& seed : diagnostics) {
    summary.minimum_presentations_per_seed =
        std::min(
            summary.minimum_presentations_per_seed,
            seed.presentation_index);
  }
  summary.all_initial_trained_masks_exact = !per_seed.empty();
  summary.all_fixed_sensory_to_inhibitory_weights_exact =
      !per_seed.empty();
  for (const SeedEvaluationMetrics& seed : per_seed) {
    summary.all_initial_trained_masks_exact =
        summary.all_initial_trained_masks_exact &&
        seed.topology.initial_trained_masks_exact;
    summary.all_fixed_sensory_to_inhibitory_weights_exact =
        summary.all_fixed_sensory_to_inhibitory_weights_exact &&
        seed.topology.auditory_to_inhibitory_weights_exact &&
        seed.topology.visual_to_inhibitory_weights_exact;
  }
  if (config.training_cohort != TrainingCohort::kBaseline) {
    summary.reason = "training_cohort_not_baseline";
    return summary;
  }
  if (control != CausalControl::kNone) {
    summary.reason = "evaluation_control_not_none";
    return summary;
  }
  if (per_seed.size() != 5u || diagnostics.size() != 5u) {
    summary.reason = "requires_exactly_five_seeds";
    return summary;
  }
  if (pruning_ever_enabled) {
    summary.reason = "pruning_was_enabled";
    return summary;
  }
  for (const TrainingDiagnostics& seed : diagnostics) {
    if (seed.presentation_index < 2000) {
      summary.reason =
          "requires_at_least_2000_presentations_per_seed";
      return summary;
    }
  }
  for (const SeedEvaluationMetrics& seed : per_seed) {
    if (!seed.topology.initial_trained_masks_exact) {
      summary.reason = "initial_trained_masks_differ";
      return summary;
    }
    if (!seed.topology.auditory_to_inhibitory_weights_exact ||
        !seed.topology.visual_to_inhibitory_weights_exact) {
      summary.reason =
          "fixed_sensory_to_inhibitory_weights_changed";
      return summary;
    }
  }

  summary.evaluated = true;
  summary.all_values_valid = true;
  for (const SeedEvaluationMetrics& seed : per_seed) {
    const SeedTopologyAudit& topology = seed.topology;
    summary.all_values_valid =
        summary.all_values_valid &&
        topology.all_required_metrics_valid;
    summary.mean_auditory_to_excitatory_delta +=
        topology
            .auditory_to_excitatory_proximity_efficacy_delta /
        5.0f;
    summary.mean_visual_to_excitatory_delta +=
        topology
            .visual_to_excitatory_proximity_efficacy_delta /
        5.0f;
    summary.mean_recurrent_delta +=
        topology.recurrent_proximity_efficacy_delta / 5.0f;
    summary.mean_effective_inhibitory_delta +=
        topology.effective_inhibitory_order_delta_explicit / 5.0f;
    summary.mean_rf_mismatch_delta_deg +=
        topology.median_rf_center_mismatch_delta_deg / 5.0f;
    summary.auditory_to_excitatory_positive_seed_count +=
        topology
                    .auditory_to_excitatory_proximity_efficacy_delta >
                0.0f
            ? 1
            : 0;
    summary.visual_to_excitatory_positive_seed_count +=
        topology
                    .visual_to_excitatory_proximity_efficacy_delta >
                0.0f
            ? 1
            : 0;
    summary.recurrent_positive_seed_count +=
        topology.recurrent_proximity_efficacy_delta > 0.0f ? 1 : 0;
    summary.effective_inhibitory_positive_seed_count +=
        topology.effective_inhibitory_order_delta_explicit > 0.0f
            ? 1
            : 0;
    summary.rf_mismatch_negative_seed_count +=
        topology.median_rf_center_mismatch_delta_deg < 0.0f ? 1 : 0;
  }
  summary.all_values_valid =
      summary.all_values_valid &&
      std::isfinite(
          summary.mean_auditory_to_excitatory_delta) &&
      std::isfinite(
          summary.mean_visual_to_excitatory_delta) &&
      std::isfinite(summary.mean_recurrent_delta) &&
      std::isfinite(summary.mean_effective_inhibitory_delta) &&
      std::isfinite(summary.mean_rf_mismatch_delta_deg);
  summary.passed =
      summary.all_values_valid &&
      summary.mean_auditory_to_excitatory_delta > 0.0f &&
      summary.auditory_to_excitatory_positive_seed_count >= 4 &&
      summary.mean_visual_to_excitatory_delta > 0.0f &&
      summary.visual_to_excitatory_positive_seed_count >= 4 &&
      summary.mean_recurrent_delta > 0.0f &&
      summary.recurrent_positive_seed_count >= 4 &&
      summary.mean_effective_inhibitory_delta > 0.0f &&
      summary.effective_inhibitory_positive_seed_count >= 4 &&
      summary.mean_rf_mismatch_delta_deg < 0.0f &&
      summary.rf_mismatch_negative_seed_count >= 4;
  return summary;
}

}  // namespace

float finalize_rf_signed_rate_mean(
    double signed_rate_sum, int sample_count) {
  if (sample_count <= 0 || !std::isfinite(signed_rate_sum)) {
    throw std::invalid_argument(
        "RF signed-rate mean requires finite data and samples.");
  }
  const double signed_mean =
      signed_rate_sum / static_cast<double>(sample_count);
  const float floored_mean =
      static_cast<float>(std::max(signed_mean, 0.0));
  if (!std::isfinite(floored_mean)) {
    throw std::invalid_argument(
        "RF signed-rate mean must be representable and finite.");
  }
  return floored_mean;
}

bool checked_pearson_correlation(
    const std::vector<double>& first,
    const std::vector<double>& second,
    double* correlation) {
  if (correlation == nullptr) {
    return false;
  }
  *correlation = std::numeric_limits<double>::quiet_NaN();
  if (first.size() != second.size() || first.size() < 2) {
    return false;
  }
  double first_mean = 0.0;
  double second_mean = 0.0;
  for (std::size_t index = 0; index < first.size(); ++index) {
    if (!std::isfinite(first[index]) ||
        !std::isfinite(second[index])) {
      return false;
    }
    first_mean += first[index];
    second_mean += second[index];
  }
  first_mean /= static_cast<double>(first.size());
  second_mean /= static_cast<double>(second.size());
  double covariance = 0.0;
  double first_variance = 0.0;
  double second_variance = 0.0;
  for (std::size_t index = 0; index < first.size(); ++index) {
    const double centered_first = first[index] - first_mean;
    const double centered_second = second[index] - second_mean;
    covariance += centered_first * centered_second;
    first_variance += centered_first * centered_first;
    second_variance += centered_second * centered_second;
  }
  if (!(first_variance > 0.0) ||
      !(second_variance > 0.0) ||
      !std::isfinite(covariance) ||
      !std::isfinite(first_variance) ||
      !std::isfinite(second_variance)) {
    return false;
  }
  const double denominator =
      std::sqrt(first_variance * second_variance);
  if (!(denominator > 0.0) || !std::isfinite(denominator)) {
    return false;
  }
  const double value = covariance / denominator;
  if (!std::isfinite(value)) {
    return false;
  }
  *correlation = value;
  return true;
}

NeuralFusionRule estimate_neural_fusion_rule(
    const FrozenTrialBatch& batch, int auditory_condition,
    int visual_condition) {
  return estimate_neural_fusion_rule_impl(
      batch, auditory_condition, visual_condition);
}

EmpiricalTemporalCrossings empirical_temporal_crossings(
    const std::array<float, kTbwPointCount>& locations,
    const std::array<TemporalCurvePoint, kTbwPointCount>& points) {
  EmpiricalTemporalCrossings result{};
  const float unavailable =
      std::numeric_limits<float>::quiet_NaN();
  result.left_50_ms = unavailable;
  result.right_50_ms = unavailable;
  result.left_75_ms = unavailable;
  result.right_75_ms = unavailable;
  result.tbw50_ms = unavailable;
  result.tbw75_ms = unavailable;
  for (int index = 0; index < kTbwPointCount; ++index) {
    const std::size_t position = static_cast<std::size_t>(index);
    if (!std::isfinite(locations[position]) ||
        !std::isfinite(points[position].probability_mean) ||
        points[position].probability_mean < 0.0f ||
        points[position].probability_mean > 1.0f ||
        (index > 0 &&
         !(locations[position] > locations[position - 1u]))) {
      return result;
    }
  }

  constexpr int kTailPointCount = 3;
  for (int index = 0; index < kTailPointCount; ++index) {
    result.left_tail_baseline +=
        points[static_cast<std::size_t>(index)].probability_mean /
        static_cast<float>(kTailPointCount);
    result.right_tail_baseline +=
        points[static_cast<std::size_t>(
                   kTbwPointCount - 1 - index)]
            .probability_mean /
        static_cast<float>(kTailPointCount);
  }
  result.peak_index = 0;
  result.peak_probability = points[0].probability_mean;
  for (int index = 1; index < kTbwPointCount; ++index) {
    const float value =
        points[static_cast<std::size_t>(index)].probability_mean;
    if (value > result.peak_probability) {
      result.peak_probability = value;
      result.peak_index = index;
    }
  }
  constexpr float central_peak_maximum_absolute_soa_ms = 100.0f;
  const bool centered =
      std::fabs(
          locations[
              static_cast<std::size_t>(result.peak_index)]) <=
      central_peak_maximum_absolute_soa_ms;
  if (result.peak_index <= 0 ||
      result.peak_index >= kTbwPointCount - 1 ||
      !centered) {
    return result;
  }

  struct Crossing {
    bool found = false;
    bool contiguous = false;
    float location = 0.0f;
  };
  const auto probability =
      [&points](int index) {
        return points[static_cast<std::size_t>(index)]
            .probability_mean;
      };
  const auto interpolate =
      [&locations, &probability](
          int outer, int inner, float threshold) {
        const float outer_value = probability(outer);
        const float inner_value = probability(inner);
        const float fraction =
            (threshold - outer_value) /
            (inner_value - outer_value);
        return locations[static_cast<std::size_t>(outer)] +
               fraction *
                   (locations[static_cast<std::size_t>(inner)] -
                    locations[static_cast<std::size_t>(outer)]);
      };
  const auto left_crossing =
      [&result, &probability, &interpolate](float threshold) {
        Crossing crossing{};
        for (int inner = result.peak_index; inner > 0; --inner) {
          const int outer = inner - 1;
          if (probability(inner) < threshold) {
            return crossing;
          }
          if (probability(outer) >= threshold) {
            continue;
          }
          crossing.found = true;
          crossing.contiguous = true;
          for (int tail = outer - 1; tail >= 0; --tail) {
            if (probability(tail) >= threshold) {
              crossing.contiguous = false;
              break;
            }
          }
          crossing.location =
              interpolate(outer, inner, threshold);
          return crossing;
        }
        return crossing;
      };
  const auto right_crossing =
      [&result, &probability, &interpolate](float threshold) {
        Crossing crossing{};
        for (int inner = result.peak_index;
             inner < kTbwPointCount - 1; ++inner) {
          const int outer = inner + 1;
          if (probability(inner) < threshold) {
            return crossing;
          }
          if (probability(outer) >= threshold) {
            continue;
          }
          crossing.found = true;
          crossing.contiguous = true;
          for (int tail = outer + 1;
               tail < kTbwPointCount; ++tail) {
            if (probability(tail) >= threshold) {
              crossing.contiguous = false;
              break;
            }
          }
          crossing.location =
              interpolate(outer, inner, threshold);
          return crossing;
        }
        return crossing;
      };

  const Crossing left50 = left_crossing(0.50f);
  const Crossing right50 = right_crossing(0.50f);
  const Crossing left75 = left_crossing(0.75f);
  const Crossing right75 = right_crossing(0.75f);
  const bool crossings50 = left50.found && right50.found;
  const bool crossings75 = left75.found && right75.found;
  result.unimodal =
      centered && crossings50 &&
      left50.contiguous && right50.contiguous;
  if (crossings50) {
    result.left_50_ms = left50.location;
    result.right_50_ms = right50.location;
    result.tbw50_ms =
        result.right_50_ms - result.left_50_ms;
  }
  if (crossings75) {
    result.left_75_ms = left75.location;
    result.right_75_ms = right75.location;
    result.tbw75_ms =
        result.right_75_ms - result.left_75_ms;
  }
  result.valid =
      result.unimodal &&
      result.left_50_ms > locations.front() &&
      result.right_50_ms < locations.back() &&
      result.tbw50_ms > 0.0f;
  return result;
}

bool temporal_gate_passes(
    const EmpiricalTemporalCrossings& empirical,
    float peak_minus_tail) {
  return empirical.valid &&
         empirical.unimodal &&
         std::isfinite(empirical.tbw50_ms) &&
         empirical.tbw50_ms >= 100.0f &&
         empirical.tbw50_ms <= 300.0f &&
         std::isfinite(peak_minus_tail) &&
         peak_minus_tail >= 0.25f;
}

DirectSpatialAudit direct_spatial_audit(
    const std::vector<float>& disparities_deg,
    const std::vector<float>& pooled_enhancement_hz,
    const std::vector<float>& pooled_enhancement_sem) {
  DirectSpatialAudit result{};
  result.input_valid =
      disparities_deg.size() >= 4u &&
      disparities_deg.size() == pooled_enhancement_hz.size() &&
      disparities_deg.size() == pooled_enhancement_sem.size();
  if (!result.input_valid) {
    return result;
  }

  result.finite = true;
  result.strictly_increasing = true;
  for (std::size_t index = 0; index < disparities_deg.size(); ++index) {
    result.finite =
        result.finite &&
        std::isfinite(disparities_deg[index]) &&
        std::isfinite(pooled_enhancement_hz[index]) &&
        std::isfinite(pooled_enhancement_sem[index]);
    if (index > 0u &&
        !(disparities_deg[index] > disparities_deg[index - 1u])) {
      result.strictly_increasing = false;
    }
  }
  if (!result.finite || !result.strictly_increasing) {
    return result;
  }

  result.center_enhancement_hz = pooled_enhancement_hz.front();
  double tail_sum = 0.0;
  for (std::size_t offset = 0; offset < 3u; ++offset) {
    tail_sum +=
        pooled_enhancement_hz[
            pooled_enhancement_hz.size() - 1u - offset];
  }
  result.tail_baseline_hz =
      static_cast<float>(tail_sum / 3.0);
  result.contrast_hz =
      result.center_enhancement_hz -
      result.tail_baseline_hz;
  result.center_positive =
      result.center_enhancement_hz > 0.0f;
  result.contrast_positive = result.contrast_hz > 0.0f;
  result.endpoints_valid =
      result.center_positive && result.contrast_positive;
  result.half_level_hz =
      result.tail_baseline_hz + 0.5f * result.contrast_hz;

  result.peak_index = 0;
  float peak_enhancement = pooled_enhancement_hz.front();
  for (std::size_t index = 1; index < pooled_enhancement_hz.size();
       ++index) {
    if (pooled_enhancement_hz[index] >= peak_enhancement) {
      peak_enhancement = pooled_enhancement_hz[index];
      result.peak_index = static_cast<int>(index);
    }
  }
  result.peak_disparity_deg =
      disparities_deg[
          static_cast<std::size_t>(result.peak_index)];
  result.peak_location_valid =
      result.peak_disparity_deg <= disparities_deg[1];

  bool seen_below_half = false;
  bool recrossed_above_half = false;
  int outward_crossings = 0;
  for (std::size_t index = 0; index < pooled_enhancement_hz.size();
       ++index) {
    const bool at_or_above_half =
        pooled_enhancement_hz[index] >= result.half_level_hz;
    if (at_or_above_half && seen_below_half) {
      recrossed_above_half = true;
    }
    if (!at_or_above_half) {
      seen_below_half = true;
    }
    if (index == 0u) {
      continue;
    }
    const bool previous_at_or_above_half =
        pooled_enhancement_hz[index - 1u] >=
        result.half_level_hz;
    if (previous_at_or_above_half && !at_or_above_half) {
      ++outward_crossings;
      result.crossing_outer_index = static_cast<int>(index);
    }
  }
  result.contiguous_prefix =
      pooled_enhancement_hz.front() >= result.half_level_hz &&
      seen_below_half && !recrossed_above_half;
  result.single_outward_crossing =
      outward_crossings == 1 && !recrossed_above_half;

  const bool shape_valid =
      result.endpoints_valid &&
      result.peak_location_valid &&
      result.contiguous_prefix &&
      result.single_outward_crossing;
  if (shape_valid) {
    const std::size_t outer = static_cast<std::size_t>(
        result.crossing_outer_index);
    const std::size_t inner = outer - 1u;
    const float inner_enhancement =
        pooled_enhancement_hz[inner];
    const float outer_enhancement =
        pooled_enhancement_hz[outer];
    const float denominator =
        inner_enhancement - outer_enhancement;
    const float fraction =
        (inner_enhancement - result.half_level_hz) /
        denominator;
    result.sbw50_deg =
        disparities_deg[inner] +
        fraction *
            (disparities_deg[outer] - disparities_deg[inner]);
    result.width_valid =
        denominator > 0.0f &&
        std::isfinite(result.sbw50_deg) &&
        result.sbw50_deg >= disparities_deg[inner] &&
        result.sbw50_deg <= disparities_deg[outer];
  }
  result.spatial_gate =
      result.input_valid &&
      result.finite &&
      result.strictly_increasing &&
      result.endpoints_valid &&
      result.peak_location_valid &&
      result.contiguous_prefix &&
      result.single_outward_crossing &&
      result.width_valid &&
      result.sbw50_deg >= 15.0f &&
      result.sbw50_deg <= 25.0f;
  return result;
}

ComponentResponseMetrics summarize_matched_component_trials(
    const std::vector<float>& auditory,
    const std::vector<float>& visual,
    const std::vector<float>& audiovisual,
    float response_floor_hz) {
  if (auditory.empty() || auditory.size() != visual.size() ||
      auditory.size() != audiovisual.size() ||
      !(response_floor_hz > 0.0f) ||
      !std::isfinite(response_floor_hz)) {
    throw std::invalid_argument(
        "Matched component trials require equal non-empty vectors "
        "and a positive finite response floor.");
  }
  std::vector<float> enhancements;
  std::vector<float> enhancement_percent;
  std::vector<float> additivity;
  std::vector<float> additivity_percent;
  enhancements.reserve(auditory.size());
  enhancement_percent.reserve(auditory.size());
  additivity.reserve(auditory.size());
  additivity_percent.reserve(auditory.size());
  for (std::size_t index = 0; index < auditory.size(); ++index) {
    const float rate_a = auditory[index];
    const float rate_v = visual[index];
    const float rate_av = audiovisual[index];
    if (!std::isfinite(rate_a) || !std::isfinite(rate_v) ||
        !std::isfinite(rate_av)) {
      throw std::invalid_argument(
          "Matched component trial rates must be finite.");
    }
    const float strongest = std::max(rate_a, rate_v);
    const float gain = rate_av - strongest;
    const float additive = rate_av - (rate_a + rate_v);
    enhancements.push_back(gain);
    enhancement_percent.push_back(
        100.0f * gain /
        std::max(strongest, response_floor_hz));
    additivity.push_back(additive);
    additivity_percent.push_back(
        100.0f * additive /
        std::max(rate_a + rate_v, response_floor_hz));
  }
  ComponentResponseMetrics result{};
  result.auditory_rate_hz = mean_sem(auditory).mean;
  result.visual_rate_hz = mean_sem(visual).mean;
  result.audiovisual_rate_hz = mean_sem(audiovisual).mean;
  const MeanSem gain = mean_sem(enhancements);
  result.raw_enhancement_hz = gain.mean;
  result.raw_enhancement_sem = gain.sem;
  result.multisensory_enhancement_percent =
      mean_sem(enhancement_percent).mean;
  result.additivity_hz = mean_sem(additivity).mean;
  result.additivity_percent = mean_sem(additivity_percent).mean;
  return result;
}

ComponentResponseMetrics pool_spatial_orientations(
    const ComponentResponseMetrics& first,
    const ComponentResponseMetrics& second) {
  ComponentResponseMetrics pooled{};
  pooled.auditory_rate_hz =
      0.5f * (first.auditory_rate_hz + second.auditory_rate_hz);
  pooled.visual_rate_hz =
      0.5f * (first.visual_rate_hz + second.visual_rate_hz);
  pooled.audiovisual_rate_hz =
      0.5f *
      (first.audiovisual_rate_hz + second.audiovisual_rate_hz);
  pooled.raw_enhancement_hz =
      0.5f *
      (first.raw_enhancement_hz + second.raw_enhancement_hz);
  pooled.raw_enhancement_sem =
      0.5f *
      std::sqrt(
          first.raw_enhancement_sem * first.raw_enhancement_sem +
          second.raw_enhancement_sem * second.raw_enhancement_sem);
  pooled.multisensory_enhancement_percent =
      0.5f *
      (first.multisensory_enhancement_percent +
       second.multisensory_enhancement_percent);
  pooled.additivity_hz =
      0.5f * (first.additivity_hz + second.additivity_hz);
  pooled.additivity_percent =
      0.5f *
      (first.additivity_percent + second.additivity_percent);
  return pooled;
}

std::vector<TrainingDiagnostics> NativeModel::diagnostics() const {
  CLEAN_MSI_CUDA(cudaSetDevice(impl_->device));
  const int seed_count = impl_->seed_count;
  const std::size_t seeds = static_cast<std::size_t>(seed_count);
  const DeviceTrainingState& state = impl_->state;
  const std::vector<float> voltage =
      copy_training_array(state.voltage, seeds * kTotalNeurons);
  const std::vector<float> recovery =
      copy_training_array(state.recovery, seeds * kTotalNeurons);
  const std::vector<float> weights =
      copy_training_array(state.weights, seeds * kWeightCount);
  const std::vector<std::uint8_t> masks =
      copy_training_array(state.masks, seeds * kWeightCount);
  const std::vector<std::uint8_t> changed =
      copy_training_array(state.changed, seeds * kWeightCount);
  const std::vector<int> dwell =
      copy_training_array(state.low_weight_dwell,
                          seeds * kWeightCount);
  const std::vector<int> presentations =
      copy_training_array(state.presentation_index, seeds);
  const std::vector<std::uint64_t> accepted =
      copy_training_array(state.accepted_steps, seeds);
  const std::vector<std::uint64_t> silent =
      copy_training_array(state.silent_iti_steps, seeds);
  const std::vector<std::uint64_t> silent_afferent_arrivals =
      copy_training_array(
          state.silent_iti_afferent_arrivals, seeds);
  const std::vector<std::uint64_t> recurrent =
      copy_training_array(state.recurrent_plasticity_steps, seeds);
  const std::vector<std::uint64_t> population_spikes =
      copy_training_array(state.population_spikes,
                          seeds * kPopulationCount);
  const std::vector<std::uint64_t> scheduled =
      copy_training_array(state.scheduled_spikes,
                          seeds * kPathCount);
  const std::vector<std::uint64_t> arrived =
      copy_training_array(state.arrived_spikes,
                          seeds * kPathCount);
  const std::vector<int> pruned =
      copy_training_array(state.pruned_contacts, seeds);

  std::vector<TrainingDiagnostics> result(seeds);
  for (int seed = 0; seed < seed_count; ++seed) {
    TrainingDiagnostics& diagnostic =
        result[static_cast<std::size_t>(seed)];
    diagnostic.presentation_index =
        presentations[static_cast<std::size_t>(seed)];
    diagnostic.accepted_steps =
        accepted[static_cast<std::size_t>(seed)];
    diagnostic.silent_iti_steps =
        silent[static_cast<std::size_t>(seed)];
    diagnostic.silent_iti_afferent_arrivals =
        silent_afferent_arrivals[static_cast<std::size_t>(seed)];
    diagnostic.recurrent_plasticity_steps =
        recurrent[static_cast<std::size_t>(seed)];
    diagnostic.pruned_contacts =
        pruned[static_cast<std::size_t>(seed)];
    const int neuron_base = seed * kTotalNeurons;
    for (int population = 0; population < kPopulationCount;
         ++population) {
      PopulationDiagnostics& values =
          diagnostic.populations[static_cast<std::size_t>(population)];
      values.voltage_min_mv = FLT_MAX;
      values.voltage_max_mv = -FLT_MAX;
      values.recovery_min = FLT_MAX;
      values.recovery_max = -FLT_MAX;
      values.finite = true;
      const int offset = population_offset(population);
      const int size = population_size(population);
      for (int neuron = 0; neuron < size; ++neuron) {
        const float v =
            voltage[static_cast<std::size_t>(
                neuron_base + offset + neuron)];
        const float u =
            recovery[static_cast<std::size_t>(
                neuron_base + offset + neuron)];
        values.finite =
            values.finite && std::isfinite(v) && std::isfinite(u);
        values.voltage_min_mv = std::min(values.voltage_min_mv, v);
        values.voltage_max_mv = std::max(values.voltage_max_mv, v);
        values.recovery_min = std::min(values.recovery_min, u);
        values.recovery_max = std::max(values.recovery_max, u);
      }
      values.cumulative_spikes =
          population_spikes[static_cast<std::size_t>(
              seed * kPopulationCount + population)];
    }
    const int weight_base = seed * kWeightCount;
    for (int path = 0; path < kPathCount; ++path) {
      PathDiagnostics& values =
          diagnostic.paths[static_cast<std::size_t>(path)];
      values.weight_min = FLT_MAX;
      values.weight_max = -FLT_MAX;
      double sum = 0.0;
      const PathLayout layout = path_layout(path);
      const int contacts = layout.pre_size * layout.post_size;
      for (int local = 0; local < contacts; ++local) {
        const int index = weight_base + layout.weight_offset + local;
        if (masks[static_cast<std::size_t>(index)]) {
          const float weight = weights[static_cast<std::size_t>(index)];
          values.weight_min = std::min(values.weight_min, weight);
          values.weight_max = std::max(values.weight_max, weight);
          sum += weight;
          ++values.active_contacts;
          values.changed_contacts +=
              changed[static_cast<std::size_t>(index)] ? 1 : 0;
        }
        values.maximum_low_weight_dwell = std::max(
            values.maximum_low_weight_dwell,
            dwell[static_cast<std::size_t>(index)]);
      }
      if (values.active_contacts > 0) {
        values.weight_mean =
            static_cast<float>(sum / values.active_contacts);
      } else {
        values.weight_min = 0.0f;
        values.weight_max = 0.0f;
      }
      values.paired_mask_mismatches = 0;
      values.scheduled_source_spikes =
          scheduled[static_cast<std::size_t>(
              seed * kPathCount + path)];
      values.arrived_source_spikes =
          arrived[static_cast<std::size_t>(
              seed * kPathCount + path)];
    }
  }
  return result;
}

TrainingMetrics NativeModel::train(const TrainingOptions& options) {
  if (options.presentations <= 0) {
    throw std::invalid_argument("Training presentations must be positive.");
  }
  if (options.seed_count != impl_->seed_count) {
    throw std::invalid_argument(
        "TrainingOptions seed_count must match NativeModel seed_count.");
  }
  if (options.chunk_presentations <= 0 ||
      options.chunk_presentations > kTrainingChunkPresentations) {
    throw std::invalid_argument(
        "chunk_presentations must be in [1, 100].");
  }
  impl_->pruning_ever_enabled =
      impl_->pruning_ever_enabled || options.enable_pruning;
  CLEAN_MSI_CUDA(cudaSetDevice(impl_->device));
  const std::vector<TrainingDiagnostics> before = diagnostics();
  const auto start = std::chrono::steady_clock::now();
  int completed = 0;
  while (completed < options.presentations) {
    const int chunk =
        std::min(options.chunk_presentations,
                 options.presentations - completed);
    bool enable_pruning = options.enable_pruning;
    void* arguments[] = {
        &impl_->state, const_cast<int*>(&chunk), &enable_pruning};
    CLEAN_MSI_CUDA(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(persistent_training_kernel),
        dim3(impl_->seed_count * kBlocksPerTrainingSeed),
        dim3(kTrainingThreads), arguments, 0, nullptr));
    CLEAN_MSI_CUDA(cudaGetLastError());
    CLEAN_MSI_CUDA(cudaDeviceSynchronize());
    completed += chunk;
  }
  const auto finish = std::chrono::steady_clock::now();
  const std::vector<TrainingDiagnostics> after = diagnostics();
  TrainingMetrics result{};
  result.completed_presentations = options.presentations;
  result.elapsed_seconds =
      std::chrono::duration<float>(finish - start).count();
  for (int seed = 0; seed < impl_->seed_count; ++seed) {
    const TrainingDiagnostics& first =
        before[static_cast<std::size_t>(seed)];
    const TrainingDiagnostics& last =
        after[static_cast<std::size_t>(seed)];
    result.total_a_spikes +=
        last.populations[0].cumulative_spikes -
        first.populations[0].cumulative_spikes;
    result.total_v_spikes +=
        last.populations[1].cumulative_spikes -
        first.populations[1].cumulative_spikes;
    result.total_e_spikes +=
        last.populations[2].cumulative_spikes -
        first.populations[2].cumulative_spikes;
    result.total_i_spikes +=
        last.populations[3].cumulative_spikes -
        first.populations[3].cumulative_spikes;
    result.mean_a_to_e += last.paths[0].weight_mean / impl_->seed_count;
    result.mean_v_to_e += last.paths[1].weight_mean / impl_->seed_count;
    result.mean_e_to_e += last.paths[2].weight_mean / impl_->seed_count;
    result.mean_i_to_e += last.paths[6].weight_mean / impl_->seed_count;
    result.pruned_contacts +=
        last.pruned_contacts - first.pruned_contacts;
  }
  std::cout << std::setprecision(9)
            << "{\"kind\":\"frozen_path_endpoints\",\"paths\":[";
  bool first_path = true;
  for (int seed = 0; seed < impl_->seed_count; ++seed) {
    for (const int path : {3, 4}) {
      if (!first_path) {
        std::cout << ',';
      }
      first_path = false;
      const PathDiagnostics& values =
          after[static_cast<std::size_t>(seed)]
              .paths[static_cast<std::size_t>(path)];
      std::cout << "{\"seed\":" << seed
                << ",\"path\":" << path
                << ",\"label\":\""
                << (path == 3 ? "A_to_I" : "V_to_I")
                << "\",\"weight_mean\":" << values.weight_mean
                << ",\"weight_min\":" << values.weight_min
                << ",\"weight_max\":" << values.weight_max
                << ",\"active_contacts\":"
                << values.active_contacts
                << ",\"changed_contacts\":"
                << values.changed_contacts << '}';
    }
  }
  std::cout << "]}\n";
  print_endpoint_homeostasis_quantiles(
      impl_->state, impl_->seed_count);
  return result;
}

EvaluationMetrics NativeModel::evaluate(
    const EvaluationOptions& options) const {
  return evaluate_impl(options, nullptr, false);
}

std::vector<SeedEvaluationMetrics> NativeModel::evaluate_per_seed(
    const EvaluationOptions& options) const {
  return evaluate(options).per_seed;
}

EvaluationMetrics NativeModel::evaluate_impl(
    const EvaluationOptions& options,
    const ObserverMetrics* frozen_observer,
    bool causal_only) const {
  const auto evaluation_start = std::chrono::steady_clock::now();
  if (options.observer_training_trials <= 0 ||
      options.observer_training_trials % 4 != 0 ||
      options.observer_holdout_trials <= 0 ||
      options.observer_holdout_trials % 4 != 0) {
    throw std::invalid_argument(
        "Observer train and holdout counts must be positive multiples of 4.");
  }
  if (options.trials_per_tbw_soa <= 0 ||
      options.trials_per_sbw_condition <= 0 ||
      options.trials_per_rf_location <= 0 ||
      options.trials_per_inverse_condition <= 0 ||
      options.burn_in_ms < 0 ||
      !(options.observer_rate_hz > 0.0f) ||
      !(options.response_floor_hz > 0.0f)) {
    throw std::invalid_argument(
        "Evaluation trial counts, rates, floors, and burn-in are invalid.");
  }

  const float rate_hz = options.observer_rate_hz;
  std::vector<ControlledCondition> observer_conditions;
  observer_conditions.reserve(4);
  observer_conditions.push_back(
      controlled_condition(
          true, true, 0.0f, 0.0f, rate_hz, rate_hz,
          -50.0f, 1, 0, options.control));
  observer_conditions.push_back(
      controlled_condition(
          true, true, 0.0f, 0.0f, rate_hz, rate_hz,
          -50.0f, 1, 1, options.control));
  observer_conditions.push_back(
      controlled_condition(
          true, true, 0.0f, 0.0f, rate_hz, rate_hz,
          -500.0f, 0, 2, options.control));
  observer_conditions.push_back(
      controlled_condition(
          true, true, 0.0f, 0.0f, rate_hz, rate_hz,
          500.0f, 0, 3, options.control));
  const FrozenTrialBatch observer_training =
      run_frozen_trials(
          observer_conditions,
          options.observer_training_trials / 4,
          options.evaluation_seed ^ 0x545241494Eull,
          options.burn_in_ms);
  const FrozenTrialBatch observer_holdout =
      run_frozen_trials(
          observer_conditions,
          options.observer_holdout_trials / 4,
          options.evaluation_seed ^ 0x484F4C444F5554ull,
          options.burn_in_ms);
  const auto require_finite_batch =
      [](const FrozenTrialBatch& batch, const char* label) {
        for (const FrozenTrialResult& trial : batch.trials) {
          if (!trial.finite ||
              !std::isfinite(trial.baseline_rate_hz) ||
              !std::isfinite(trial.response_rate_hz)) {
            throw std::runtime_error(
                std::string(label) +
                " produced a non-finite frozen trial.");
          }
        }
      };
  require_finite_batch(observer_training, "Observer training");
  require_finite_batch(observer_holdout, "Observer holdout");

  EvaluationMetrics metrics{};
  metrics.simulated_trials =
      observer_training.trials.size() +
      observer_holdout.trials.size();
  metrics.frozen_kernel_seconds =
      observer_training.kernel_seconds +
      observer_holdout.kernel_seconds;
  metrics.control_audit =
      projection_control_audit(options.control);
  metrics.observer =
      frozen_observer != nullptr
          ? *frozen_observer
          : fit_observer(observer_training, observer_holdout);

  std::vector<ControlledCondition> temporal_conditions;
  temporal_conditions.reserve(kTbwPointCount + 2);
  temporal_conditions.push_back(
      controlled_condition(
          true, false, 0.0f, 0.0f, rate_hz, rate_hz,
          0.0f, 0, 1000, options.control));
  temporal_conditions.push_back(
      controlled_condition(
          false, true, 0.0f, 0.0f, rate_hz, rate_hz,
          0.0f, 0, 1001, options.control));
  for (int point = 0; point < kTbwPointCount; ++point) {
    const float physical_soa =
        -500.0f + 25.0f * static_cast<float>(point);
    metrics.temporal.physical_soa_grid_ms[
        static_cast<std::size_t>(point)] = physical_soa;
    temporal_conditions.push_back(
        controlled_condition(
            true, true, 0.0f, 0.0f, rate_hz, rate_hz,
            physical_soa, 0, 1100 + point, options.control));
  }
  const FrozenTrialBatch temporal_batch =
      run_frozen_trials(
          temporal_conditions, options.trials_per_tbw_soa,
          options.evaluation_seed ^ 0x544257ull,
          options.burn_in_ms);
  require_finite_batch(temporal_batch, "TBW");
  metrics.simulated_trials += temporal_batch.trials.size();
  metrics.frozen_kernel_seconds += temporal_batch.kernel_seconds;
  metrics.temporal.trials_per_soa = options.trials_per_tbw_soa;
  std::vector<float> auditory_rates;
  std::vector<float> visual_rates;
  auditory_rates.reserve(
      static_cast<std::size_t>(impl_->seed_count) *
      options.trials_per_tbw_soa);
  visual_rates.reserve(auditory_rates.capacity());
  for (int seed = 0; seed < temporal_batch.model_seed_count; ++seed) {
    for (int trial = 0;
         trial < temporal_batch.trials_per_condition; ++trial) {
      auditory_rates.push_back(
          batch_trial(temporal_batch, seed, 0, trial)
              .response_rate_hz);
      visual_rates.push_back(
          batch_trial(temporal_batch, seed, 1, trial)
              .response_rate_hz);
    }
  }
  const float auditory_rate = mean_sem(auditory_rates).mean;
  const float visual_rate = mean_sem(visual_rates).mean;
  const NeuralFusionRule neural_fusion_rule =
      estimate_neural_fusion_rule(temporal_batch, 0, 1);
  float peak_probability = 0.0f;
  for (int point = 0; point < kTbwPointCount; ++point) {
    std::vector<float> audiovisual_rates;
    audiovisual_rates.reserve(
        static_cast<std::size_t>(impl_->seed_count) *
        options.trials_per_tbw_soa);
    int fusion_count = 0;
    int fusion_trial_count = 0;
    for (int seed = 0; seed < temporal_batch.model_seed_count; ++seed) {
      for (int trial = 0;
           trial < temporal_batch.trials_per_condition; ++trial) {
        const FrozenTrialResult& value =
            batch_trial(
                temporal_batch, seed, point + 2, trial);
        fusion_count +=
            neural_fusion_event(value, neural_fusion_rule) ? 1 : 0;
        ++fusion_trial_count;
        audiovisual_rates.push_back(value.response_rate_hz);
      }
    }
    const float fusion_probability =
        static_cast<float>(fusion_count) /
        static_cast<float>(fusion_trial_count);
    const float fusion_sem =
        std::sqrt(
            fusion_probability * (1.0f - fusion_probability) /
            static_cast<float>(fusion_trial_count));
    const MeanSem audiovisual = mean_sem(audiovisual_rates);
    TemporalCurvePoint& output =
        metrics.temporal.points[static_cast<std::size_t>(point)];
    output.physical_soa_ms =
        metrics.temporal.physical_soa_grid_ms[
            static_cast<std::size_t>(point)];
    output.probability_mean = fusion_probability;
    output.probability_sem = fusion_sem;
    output.auditory_rate_hz = auditory_rate;
    output.visual_rate_hz = visual_rate;
    output.audiovisual_rate_hz = audiovisual.mean;
    output.raw_neural_enhancement_hz =
        audiovisual.mean - std::max(auditory_rate, visual_rate);
    peak_probability =
        std::max(peak_probability, fusion_probability);
  }
  metrics.temporal.empirical =
      empirical_temporal_crossings(
          metrics.temporal.physical_soa_grid_ms,
          metrics.temporal.points);
  metrics.temporal.peak_minus_tail =
      peak_probability -
      0.5f *
          (metrics.temporal.empirical.left_tail_baseline +
           metrics.temporal.empirical.right_tail_baseline);
  metrics.temporal.fit =
      fit_asymmetric_gaussian(
          metrics.temporal.physical_soa_grid_ms,
          metrics.temporal.points);
  metrics.tbw50_ms = metrics.temporal.empirical.tbw50_ms;
  metrics.tbw75_ms = metrics.temporal.empirical.tbw75_ms;
  metrics.tbw_peak_minus_tail =
      metrics.temporal.peak_minus_tail;

  std::vector<ControlledCondition> spatial_conditions;
  spatial_conditions.reserve(kSbwPointCount * 2 * 3);
  for (int point = 0; point < kSbwPointCount; ++point) {
    const float disparity = 2.5f * static_cast<float>(point);
    metrics.spatial.disparity_grid_deg[
        static_cast<std::size_t>(point)] = disparity;
    for (int orientation = 0; orientation < 2; ++orientation) {
      const float sign = orientation == 0 ? 1.0f : -1.0f;
      const float auditory_location = -0.5f * sign * disparity;
      const float visual_location = 0.5f * sign * disparity;
      const int random_group = 2000 + point * 2 + orientation;
      spatial_conditions.push_back(
          controlled_condition(
              true, false, auditory_location, visual_location,
              rate_hz, rate_hz, -50.0f, 0, random_group,
              options.control));
      spatial_conditions.push_back(
          controlled_condition(
              false, true, auditory_location, visual_location,
              rate_hz, rate_hz, -50.0f, 0, random_group,
              options.control));
      spatial_conditions.push_back(
          controlled_condition(
              true, true, auditory_location, visual_location,
              rate_hz, rate_hz, -50.0f, 0, random_group,
              options.control));
    }
  }
  const FrozenTrialBatch spatial_batch =
      run_frozen_trials(
          spatial_conditions,
          options.trials_per_sbw_condition,
          options.evaluation_seed ^ 0x534257ull,
          options.burn_in_ms);
  require_finite_batch(spatial_batch, "SBW");
  metrics.simulated_trials += spatial_batch.trials.size();
  metrics.frozen_kernel_seconds += spatial_batch.kernel_seconds;
  metrics.spatial.trials_per_orientation =
      options.trials_per_sbw_condition;
  metrics.spatial.response_floor_hz =
      options.response_floor_hz;
  for (int point = 0; point < kSbwPointCount; ++point) {
    SpatialCurvePoint& output =
        metrics.spatial.points[static_cast<std::size_t>(point)];
    output.disparity_deg =
        metrics.spatial.disparity_grid_deg[
            static_cast<std::size_t>(point)];
    for (int orientation = 0; orientation < 2; ++orientation) {
      const int base = (point * 2 + orientation) * 3;
      output.orientations[static_cast<std::size_t>(orientation)] =
          component_response_metrics(
              spatial_batch, base, base + 1, base + 2,
              options.response_floor_hz, true);
    }
    output.pooled =
        pool_spatial_orientations(
            output.orientations[0], output.orientations[1]);
  }
  metrics.spatial.fit =
      fit_spatial_gaussian(
          metrics.spatial.disparity_grid_deg,
          metrics.spatial.points);
  std::vector<float> pooled_disparities;
  std::vector<float> pooled_enhancement;
  std::vector<float> pooled_enhancement_sem;
  pooled_disparities.reserve(kSbwPointCount);
  pooled_enhancement.reserve(kSbwPointCount);
  pooled_enhancement_sem.reserve(kSbwPointCount);
  for (const SpatialCurvePoint& point : metrics.spatial.points) {
    pooled_disparities.push_back(point.disparity_deg);
    pooled_enhancement.push_back(
        point.pooled.raw_enhancement_hz);
    pooled_enhancement_sem.push_back(
        point.pooled.raw_enhancement_sem);
  }
  metrics.spatial.direct =
      direct_spatial_audit(
          pooled_disparities,
          pooled_enhancement,
          pooled_enhancement_sem);
  const DirectSpatialAudit& spatial_direct =
      metrics.spatial.direct;
  const SymmetricGaussianFit& spatial_fit =
      metrics.spatial.fit;
  const float max_sampled_disparity_deg =
      metrics.spatial.disparity_grid_deg.back();
  const bool raw_spatial_prerequisites =
      spatial_direct.input_valid &&
      spatial_direct.finite &&
      spatial_direct.strictly_increasing &&
      spatial_direct.endpoints_valid &&
      spatial_direct.peak_location_valid &&
      spatial_direct.contiguous_prefix &&
      spatial_direct.single_outward_crossing &&
      spatial_direct.width_valid;
  const bool gaussian_spatial_gate =
      spatial_fit.valid &&
      spatial_fit.hwhm_valid &&
      spatial_fit.amplitude > 1.0e-6f &&
      std::isfinite(spatial_fit.mse) &&
      spatial_fit.fitted_hwhm_deg > 0.0f &&
      spatial_fit.fitted_hwhm_deg <= max_sampled_disparity_deg &&
      spatial_fit.fitted_hwhm_deg >= 15.0f &&
      spatial_fit.fitted_hwhm_deg <= 25.0f;
  metrics.sbw50_deg = spatial_fit.fitted_hwhm_deg;
  metrics.spatial_gate =
      raw_spatial_prerequisites && gaussian_spatial_gate;

  if (causal_only) {
    metrics.elapsed_seconds =
        std::chrono::duration<float>(
            std::chrono::steady_clock::now() - evaluation_start)
            .count();
    return metrics;
  }

  std::vector<ControlledCondition> rf_conditions;
  rf_conditions.reserve(kRfPointCount * 2);
  for (int point = 0; point < kRfPointCount; ++point) {
    const float location =
        -60.0f + 5.0f * static_cast<float>(point);
    rf_conditions.push_back(
        controlled_condition(
            true, false, location, 0.0f, rate_hz, rate_hz,
            0.0f, 0, 3000 + point * 2, options.control));
    rf_conditions.push_back(
        controlled_condition(
            false, true, 0.0f, location, rate_hz, rate_hz,
            0.0f, 0, 3001 + point * 2, options.control));
  }
  std::vector<ControlledCondition> initial_rf_conditions =
      rf_conditions;
  for (ControlledCondition& condition : initial_rf_conditions) {
    condition.use_initial_weights = true;
  }
  const FrozenTrialBatch rf_batch =
      run_frozen_trials(
          rf_conditions, options.trials_per_rf_location,
          options.evaluation_seed ^ 0x5246ull,
          options.burn_in_ms);
  const FrozenTrialBatch initial_rf_batch =
      run_frozen_trials(
          initial_rf_conditions, options.trials_per_rf_location,
          options.evaluation_seed ^ 0x5246ull,
          options.burn_in_ms);
  require_finite_batch(rf_batch, "RF");
  require_finite_batch(initial_rf_batch, "Initial-weight RF");
  metrics.simulated_trials +=
      rf_batch.trials.size() + initial_rf_batch.trials.size();
  metrics.frozen_kernel_seconds +=
      rf_batch.kernel_seconds + initial_rf_batch.kernel_seconds;
  const std::vector<SeedWeightStateAudit> weight_states =
      weight_state_audit();
  std::vector<FrozenWeightAudit> initial_weight_audits;
  std::vector<FrozenWeightAudit> trained_weight_audits;
  initial_weight_audits.reserve(weight_states.size());
  trained_weight_audits.reserve(weight_states.size());
  for (const SeedWeightStateAudit& state : weight_states) {
    initial_weight_audits.push_back(state.initial);
    trained_weight_audits.push_back(state.trained);
  }
  metrics.rf_topography =
      compute_rf_topography(
          rf_batch, trained_weight_audits,
          options.trials_per_rf_location);
  metrics.per_seed.resize(
      static_cast<std::size_t>(impl_->seed_count));
  for (int seed = 0; seed < impl_->seed_count; ++seed) {
    SeedEvaluationMetrics& seed_metrics =
        metrics.per_seed[static_cast<std::size_t>(seed)];
    seed_metrics.seed_index = seed;
    seed_metrics.global_seed =
        weight_states[static_cast<std::size_t>(seed)].global_seed;
    const RfTopographyAudit initial_rf =
        compute_rf_topography(
            initial_rf_batch, initial_weight_audits,
            options.trials_per_rf_location, seed);
    const RfTopographyAudit trained_rf =
        compute_rf_topography(
            rf_batch, trained_weight_audits,
            options.trials_per_rf_location, seed);
    populate_seed_topology_audit(
        weight_states[static_cast<std::size_t>(seed)],
        initial_rf, trained_rf, &seed_metrics.topology);
  }
  metrics.topology_refinement =
      summarize_topology_refinement(
          metrics.per_seed, impl_->config, options.control,
          diagnostics(), impl_->pruning_ever_enabled);
  metrics.auditory_topography_correlation =
      metrics.rf_topography.auditory_weight_order_correlation;
  metrics.visual_topography_correlation =
      metrics.rf_topography.visual_weight_order_correlation;
  metrics.inhibitory_topography_correlation =
      metrics.rf_topography.effective_inhibitory_order_correlation;
  metrics.recurrent_topography_correlation =
      metrics.rf_topography.recurrent_weight_distance_correlation;

  std::vector<ControlledCondition> inverse_conditions;
  inverse_conditions.reserve(kInverseEffectivenessCount * 3);
  for (int salience = 0;
       salience < kInverseEffectivenessCount; ++salience) {
    const float salience_hz =
        metrics.inverse_effectiveness.salience_hz[
            static_cast<std::size_t>(salience)];
    const int random_group = 4000 + salience;
    inverse_conditions.push_back(
        controlled_condition(
            true, false, 0.0f, 0.0f, salience_hz, salience_hz,
            -50.0f, 0, random_group, options.control));
    inverse_conditions.push_back(
        controlled_condition(
            false, true, 0.0f, 0.0f, salience_hz, salience_hz,
            -50.0f, 0, random_group, options.control));
    inverse_conditions.push_back(
        controlled_condition(
            true, true, 0.0f, 0.0f, salience_hz, salience_hz,
            -50.0f, 0, random_group, options.control));
  }
  const FrozenTrialBatch inverse_batch =
      run_frozen_trials(
          inverse_conditions,
          options.trials_per_inverse_condition,
          options.evaluation_seed ^ 0x494E5645525345ull,
          options.burn_in_ms);
  require_finite_batch(inverse_batch, "Inverse effectiveness");
  metrics.simulated_trials += inverse_batch.trials.size();
  metrics.frozen_kernel_seconds += inverse_batch.kernel_seconds;
  metrics.inverse_effectiveness.trials_per_salience =
      options.trials_per_inverse_condition;
  for (int salience = 0;
       salience < kInverseEffectivenessCount; ++salience) {
    const int base = salience * 3;
    metrics.inverse_effectiveness.responses[
        static_cast<std::size_t>(salience)] =
        component_response_metrics(
            inverse_batch, base, base + 1, base + 2,
            options.response_floor_hz, false);
    metrics.inverse_effectiveness_percent[
        static_cast<std::size_t>(salience)] =
        metrics.inverse_effectiveness.responses[
            static_cast<std::size_t>(salience)]
            .multisensory_enhancement_percent;
  }

  metrics.criteria_passed =
      metrics.observer.auroc >= 0.80f &&
      metrics.observer.brier <= 0.20f &&
      temporal_gate_passes(
          metrics.temporal.empirical,
          metrics.tbw_peak_minus_tail) &&
      metrics.spatial_gate &&
      (!metrics.topology_refinement.evaluated ||
       metrics.topology_refinement.passed);
  metrics.elapsed_seconds =
      std::chrono::duration<float>(
          std::chrono::steady_clock::now() - evaluation_start)
          .count();
  return metrics;
}

std::vector<CausalControlEvaluation>
NativeModel::evaluate_all_controls(
    const EvaluationOptions& options,
    const EvaluationMetrics& baseline) const {
  if (options.control != CausalControl::kNone ||
      baseline.control_audit.control != CausalControl::kNone) {
    throw std::invalid_argument(
        "All-controls evaluation requires an uncontrolled baseline.");
  }
  constexpr std::array<CausalControl, 5> kControls{
      CausalControl::kNmdaOff,
      CausalControl::kGabaaOff,
      CausalControl::kRecurrentExcitationOff,
      CausalControl::kRecruitedInhibitionOff,
      CausalControl::kMsiEAdaptationOff,
  };
  std::vector<CausalControlEvaluation> results;
  results.reserve(kControls.size());
  for (const CausalControl control : kControls) {
    EvaluationOptions controlled_options = options;
    controlled_options.control = control;
    CausalControlEvaluation result{};
    result.control = control;
    result.evaluation =
        evaluate_impl(
            controlled_options, &baseline.observer, true);
    result.observer_frozen = true;
    results.push_back(std::move(result));
  }
  return results;
}

namespace {

CausalControl parse_causal_control(const std::string& text) {
  if (text.empty() || text == "none") {
    return CausalControl::kNone;
  }
  CausalControl control = CausalControl::kNone;
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t end = text.find_first_of(",+", begin);
    const std::string token =
        text.substr(
            begin,
            end == std::string::npos
                ? std::string::npos
                : end - begin);
    if (token == "nmda-off") {
      control = control | CausalControl::kNmdaOff;
    } else if (token == "gabaa-off") {
      control = control | CausalControl::kGabaaOff;
    } else if (token == "recurrent-off") {
      control =
          control | CausalControl::kRecurrentExcitationOff;
    } else if (token == "recruited-inhibition-off") {
      control =
          control | CausalControl::kRecruitedInhibitionOff;
    } else if (token == "msi-e-adaptation-off") {
      control =
          control | CausalControl::kMsiEAdaptationOff;
    } else if (token == "source-row-shuffle") {
      control = control | CausalControl::kSourceRowShuffle;
    } else {
      throw std::invalid_argument(
          "Unknown causal control token: " + token);
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return control;
}

void write_json_number(float value) {
  if (std::isfinite(value)) {
    std::cout << value;
  } else {
    std::cout << "null";
  }
}

template <std::size_t Count>
void write_float_array(const std::array<float, Count>& values) {
  std::cout << '[';
  for (std::size_t index = 0; index < Count; ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    write_json_number(values[index]);
  }
  std::cout << ']';
}

template <std::size_t Count>
void write_int_array(const std::array<int, Count>& values) {
  std::cout << '[';
  for (std::size_t index = 0; index < Count; ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << values[index];
  }
  std::cout << ']';
}

template <std::size_t Count>
void write_bool_array(const std::array<bool, Count>& values) {
  std::cout << '[';
  for (std::size_t index = 0; index < Count; ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << (values[index] ? "true" : "false");
  }
  std::cout << ']';
}

void write_float_vector(const std::vector<float>& values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    write_json_number(values[index]);
  }
  std::cout << ']';
}

void write_label_vector(const std::vector<std::uint8_t>& values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << static_cast<int>(values[index]);
  }
  std::cout << ']';
}

template <std::size_t Rows, std::size_t Columns>
void write_float_matrix(
    const std::array<std::array<float, Columns>, Rows>& values) {
  std::cout << '[';
  for (std::size_t row = 0; row < Rows; ++row) {
    if (row != 0) {
      std::cout << ',';
    }
    write_float_array(values[row]);
  }
  std::cout << ']';
}

void print_observer_json(
    int device, const EvaluationOptions& options,
    const EvaluationMetrics& metrics) {
  const ObserverMetrics& observer = metrics.observer;
  std::array<int, kEvaluationFeatureBins + 1> edges{};
  for (int edge = 0; edge <= kEvaluationFeatureBins; ++edge) {
    edges[static_cast<std::size_t>(edge)] =
        -100 + edge * kEvaluationBinWidthMs;
  }
  std::cout << "{\"kind\":\"observer\",\"device\":" << device
            << ",\"physical_soa_feature\":false"
            << ",\"feature_reference\":\"earlier_received_onset\""
            << ",\"physical_soa_definition\":\"tV_minus_tA\""
            << ",\"burn_in_background_active_ms\":"
            << options.burn_in_ms
            << ",\"feature_bin_edges_ms\":";
  write_int_array(edges);
  std::cout << ",\"l2\":";
  write_json_number(observer.l2);
  std::cout << ",\"intercept\":";
  write_json_number(observer.intercept);
  std::cout << ",\"feature_mean\":";
  write_float_array(observer.feature_mean);
  std::cout << ",\"feature_std\":";
  write_float_array(observer.feature_std);
  std::cout << ",\"coefficients\":";
  write_float_array(observer.coefficients);
  std::cout << ",\"iterations\":" << observer.iterations
            << ",\"converged\":"
            << (observer.converged ? "true" : "false")
            << ",\"auroc\":";
  write_json_number(observer.auroc);
  std::cout << ",\"brier\":";
  write_json_number(observer.brier);
  std::cout << ",\"calibration_intercept\":";
  write_json_number(observer.calibration_intercept);
  std::cout << ",\"calibration_slope\":";
  write_json_number(observer.calibration_slope);
  std::cout << ",\"expected_calibration_error\":";
  write_json_number(observer.expected_calibration_error);
  std::cout << ",\"training_labels\":";
  write_label_vector(observer.training_labels);
  std::cout << ",\"training_probabilities\":";
  write_float_vector(observer.training_probabilities);
  std::cout << ",\"holdout_labels\":";
  write_label_vector(observer.holdout_labels);
  std::cout << ",\"holdout_probabilities\":";
  write_float_vector(observer.holdout_probabilities);
  std::cout << "}\n";
}

void print_tbw_json(
    int device, const EvaluationMetrics& metrics) {
  const TemporalAudit& temporal = metrics.temporal;
  const EmpiricalTemporalCrossings& empirical =
      temporal.empirical;
  const AsymmetricGaussianFit& fit = temporal.fit;
  std::cout << "{\"kind\":\"tbw\",\"device\":" << device
            << ",\"physical_soa_definition\":\"tV_minus_tA\""
            << ",\"trials_per_soa\":" << temporal.trials_per_soa
            << ",\"physical_soa_ms\":";
  write_float_array(temporal.physical_soa_grid_ms);
  const auto write_point_field =
      [&temporal](auto selector) {
        std::cout << '[';
        for (int point = 0; point < kTbwPointCount; ++point) {
          if (point != 0) {
            std::cout << ',';
          }
          write_json_number(
              selector(
                  temporal.points[
                      static_cast<std::size_t>(point)]));
        }
        std::cout << ']';
      };
  std::cout << ",\"probability_mean\":";
  write_point_field(
      [](const TemporalCurvePoint& point) {
        return point.probability_mean;
      });
  std::cout << ",\"probability_sem\":";
  write_point_field(
      [](const TemporalCurvePoint& point) {
        return point.probability_sem;
      });
  std::cout << ",\"auditory_rate_hz\":";
  write_point_field(
      [](const TemporalCurvePoint& point) {
        return point.auditory_rate_hz;
      });
  std::cout << ",\"visual_rate_hz\":";
  write_point_field(
      [](const TemporalCurvePoint& point) {
        return point.visual_rate_hz;
      });
  std::cout << ",\"audiovisual_rate_hz\":";
  write_point_field(
      [](const TemporalCurvePoint& point) {
        return point.audiovisual_rate_hz;
      });
  std::cout << ",\"raw_neural_enhancement_hz\":";
  write_point_field(
      [](const TemporalCurvePoint& point) {
        return point.raw_neural_enhancement_hz;
      });
  std::cout << ",\"peak_minus_tail\":";
  write_json_number(temporal.peak_minus_tail);
  std::cout << ",\"empirical\":{\"valid\":"
            << (empirical.valid ? "true" : "false")
            << ",\"unimodal\":"
            << (empirical.unimodal ? "true" : "false")
            << ",\"peak_index\":" << empirical.peak_index
            << ",\"peak_probability\":";
  write_json_number(empirical.peak_probability);
  std::cout << ",\"left_tail_baseline\":";
  write_json_number(empirical.left_tail_baseline);
  std::cout << ",\"right_tail_baseline\":";
  write_json_number(empirical.right_tail_baseline);
  std::cout << ",\"left_50_ms\":";
  write_json_number(empirical.left_50_ms);
  std::cout << ",\"right_50_ms\":";
  write_json_number(empirical.right_50_ms);
  std::cout << ",\"left_75_ms\":";
  write_json_number(empirical.left_75_ms);
  std::cout << ",\"right_75_ms\":";
  write_json_number(empirical.right_75_ms);
  std::cout << ",\"tbw50_ms\":";
  write_json_number(empirical.tbw50_ms);
  std::cout << ",\"tbw75_ms\":";
  write_json_number(empirical.tbw75_ms);
  std::cout << "},\"parametric_fit\":{\"valid\":"
            << (fit.valid ? "true" : "false")
            << ",\"crossings_valid\":"
            << (fit.crossings_valid ? "true" : "false")
            << ",\"baseline\":";
  write_json_number(fit.baseline);
  std::cout << ",\"amplitude\":";
  write_json_number(fit.amplitude);
  std::cout << ",\"center_ms\":";
  write_json_number(fit.center_ms);
  std::cout << ",\"sigma_left_ms\":";
  write_json_number(fit.sigma_left_ms);
  std::cout << ",\"sigma_right_ms\":";
  write_json_number(fit.sigma_right_ms);
  std::cout << ",\"mse\":";
  write_json_number(fit.mse);
  std::cout << ",\"left_50_ms\":";
  write_json_number(fit.left_50_ms);
  std::cout << ",\"right_50_ms\":";
  write_json_number(fit.right_50_ms);
  std::cout << ",\"left_75_ms\":";
  write_json_number(fit.left_75_ms);
  std::cout << ",\"right_75_ms\":";
  write_json_number(fit.right_75_ms);
  std::cout << ",\"tbw50_ms\":";
  write_json_number(fit.tbw50_ms);
  std::cout << ",\"tbw75_ms\":";
  write_json_number(fit.tbw75_ms);
  std::cout << "}}\n";
}

void print_component_json(const ComponentResponseMetrics& values) {
  std::cout << "{\"R_A_hz\":";
  write_json_number(values.auditory_rate_hz);
  std::cout << ",\"R_V_hz\":";
  write_json_number(values.visual_rate_hz);
  std::cout << ",\"R_AV_hz\":";
  write_json_number(values.audiovisual_rate_hz);
  std::cout << ",\"G_hz\":";
  write_json_number(values.raw_enhancement_hz);
  std::cout << ",\"G_sem\":";
  write_json_number(values.raw_enhancement_sem);
  std::cout << ",\"ME_percent\":";
  write_json_number(values.multisensory_enhancement_percent);
  std::cout << ",\"additivity_hz\":";
  write_json_number(values.additivity_hz);
  std::cout << ",\"additivity_percent\":";
  write_json_number(values.additivity_percent);
  std::cout << '}';
}

void print_sbw_json(
    int device, const EvaluationMetrics& metrics) {
  const SpatialAudit& spatial = metrics.spatial;
  const DirectSpatialAudit& direct = spatial.direct;
  const SymmetricGaussianFit& fit = spatial.fit;
  std::cout << "{\"kind\":\"sbw\",\"device\":" << device
            << ",\"trials_per_orientation\":"
            << spatial.trials_per_orientation
            << ",\"response_floor_hz\":";
  write_json_number(spatial.response_floor_hz);
  std::cout << ",\"disparity_deg\":";
  write_float_array(spatial.disparity_grid_deg);
  std::cout << ",\"points\":[";
  for (int point = 0; point < kSbwPointCount; ++point) {
    if (point != 0) {
      std::cout << ',';
    }
    const SpatialCurvePoint& value =
        spatial.points[static_cast<std::size_t>(point)];
    std::cout << "{\"disparity_deg\":";
    write_json_number(value.disparity_deg);
    std::cout << ",\"A_left_V_right\":";
    print_component_json(value.orientations[0]);
    std::cout << ",\"A_right_V_left\":";
    print_component_json(value.orientations[1]);
    std::cout << ",\"pooled\":";
    print_component_json(value.pooled);
    std::cout << '}';
  }
  std::cout << "],\"sbw50_method\":\"gaussian_fitted_hwhm\""
            << ",\"sbw50_deg\":";
  write_json_number(metrics.sbw50_deg);
  std::cout << ",\"direct_raw_sbw50_deg\":";
  write_json_number(direct.sbw50_deg);
  std::cout << ",\"spatial_gate\":"
            << (metrics.spatial_gate ? "true" : "false")
            << ",\"direct\":{\"role\":\"raw_crossing_audit\""
            << ",\"input_valid\":"
            << (direct.input_valid ? "true" : "false")
            << ",\"finite\":"
            << (direct.finite ? "true" : "false")
            << ",\"strictly_increasing\":"
            << (direct.strictly_increasing ? "true" : "false")
            << ",\"center_positive\":"
            << (direct.center_positive ? "true" : "false")
            << ",\"contrast_positive\":"
            << (direct.contrast_positive ? "true" : "false")
            << ",\"endpoints_valid\":"
            << (direct.endpoints_valid ? "true" : "false")
            << ",\"peak_location_valid\":"
            << (direct.peak_location_valid ? "true" : "false")
            << ",\"contiguous_prefix\":"
            << (direct.contiguous_prefix ? "true" : "false")
            << ",\"single_outward_crossing\":"
            << (direct.single_outward_crossing ? "true" : "false")
            << ",\"width_valid\":"
            << (direct.width_valid ? "true" : "false")
            << ",\"spatial_gate\":"
            << (direct.spatial_gate ? "true" : "false")
            << ",\"peak_index\":" << direct.peak_index
            << ",\"crossing_outer_index\":"
            << direct.crossing_outer_index
            << ",\"center_enhancement_hz\":";
  write_json_number(direct.center_enhancement_hz);
  std::cout << ",\"tail_baseline_hz\":";
  write_json_number(direct.tail_baseline_hz);
  std::cout << ",\"contrast_hz\":";
  write_json_number(direct.contrast_hz);
  std::cout << ",\"peak_disparity_deg\":";
  write_json_number(direct.peak_disparity_deg);
  std::cout << ",\"half_level_hz\":";
  write_json_number(direct.half_level_hz);
  std::cout << ",\"sbw50_deg\":";
  write_json_number(direct.sbw50_deg);
  std::cout
      << "},\"fit\":{\"role\":\"acceptance_primary\",\"valid\":"
            << (fit.valid ? "true" : "false")
            << ",\"hwhm_valid\":"
            << (fit.hwhm_valid ? "true" : "false")
            << ",\"baseline\":";
  write_json_number(fit.baseline);
  std::cout << ",\"amplitude\":";
  write_json_number(fit.amplitude);
  std::cout << ",\"sigma_deg\":";
  write_json_number(fit.sigma_deg);
  std::cout << ",\"fitted_hwhm_deg\":";
  write_json_number(fit.fitted_hwhm_deg);
  std::cout << ",\"mse\":";
  write_json_number(fit.mse);
  std::cout << "}}\n";
}

void print_rf_json(
    int device, const EvaluationMetrics& metrics) {
  const RfTopographyAudit& rf = metrics.rf_topography;
  std::cout << "{\"kind\":\"rf_topography\",\"device\":" << device
            << ",\"trials_per_location\":" << rf.trials_per_location
            << ",\"location_deg\":";
  write_float_array(rf.location_grid_deg);
  std::cout << ",\"auditory_response_hz\":";
  write_float_matrix(rf.auditory_response_hz);
  std::cout << ",\"visual_response_hz\":";
  write_float_matrix(rf.visual_response_hz);
  std::cout << ",\"auditory_rf_center_deg\":";
  write_float_array(rf.auditory_rf_center_deg);
  std::cout << ",\"auditory_rf_width_deg\":";
  write_float_array(rf.auditory_rf_width_deg);
  std::cout << ",\"visual_rf_center_deg\":";
  write_float_array(rf.visual_rf_center_deg);
  std::cout << ",\"visual_rf_width_deg\":";
  write_float_array(rf.visual_rf_width_deg);
  std::cout << ",\"rf_center_mismatch_deg\":";
  write_float_array(rf.rf_center_mismatch_deg);
  std::cout << ",\"auditory_excitatory_map_center_deg\":";
  write_float_array(rf.auditory_excitatory_map_center_deg);
  std::cout << ",\"auditory_excitatory_map_width_deg\":";
  write_float_array(rf.auditory_excitatory_map_width_deg);
  std::cout << ",\"visual_excitatory_map_center_deg\":";
  write_float_array(rf.visual_excitatory_map_center_deg);
  std::cout << ",\"visual_excitatory_map_width_deg\":";
  write_float_array(rf.visual_excitatory_map_width_deg);
  std::cout << ",\"auditory_effective_inhibitory_center_deg\":";
  write_float_array(rf.auditory_effective_inhibitory_center_deg);
  std::cout << ",\"auditory_effective_inhibitory_width_deg\":";
  write_float_array(rf.auditory_effective_inhibitory_width_deg);
  std::cout << ",\"visual_effective_inhibitory_center_deg\":";
  write_float_array(rf.visual_effective_inhibitory_center_deg);
  std::cout << ",\"visual_effective_inhibitory_width_deg\":";
  write_float_array(rf.visual_effective_inhibitory_width_deg);
  std::cout << ",\"auditory_rf_order_correlation\":";
  write_json_number(rf.auditory_rf_order_correlation);
  std::cout << ",\"visual_rf_order_correlation\":";
  write_json_number(rf.visual_rf_order_correlation);
  std::cout << ",\"auditory_weight_order_correlation\":";
  write_json_number(rf.auditory_weight_order_correlation);
  std::cout << ",\"visual_weight_order_correlation\":";
  write_json_number(rf.visual_weight_order_correlation);
  std::cout << ",\"effective_inhibitory_order_correlation\":";
  write_json_number(rf.effective_inhibitory_order_correlation);
  std::cout << ",\"recurrent_weight_distance_correlation\":";
  write_json_number(rf.recurrent_weight_distance_correlation);
  std::cout << ",\"median_rf_center_mismatch_deg\":";
  write_json_number(rf.median_rf_center_mismatch_deg);
  std::cout << ",\"median_effective_inhibitory_alignment_deg\":";
  write_json_number(rf.median_effective_inhibitory_alignment_deg);
  std::cout << ",\"finite_rf_coverage\":";
  write_json_number(rf.finite_rf_coverage);
  std::cout << ",\"map_monotonicity\":";
  write_json_number(rf.map_monotonicity);
  std::cout << "}\n";
}

const char* training_cohort_name(TrainingCohort cohort) {
  switch (cohort) {
    case TrainingCohort::kBaseline:
      return "baseline";
    case TrainingCohort::kPlasticityOff:
      return "plasticity-off";
    case TrainingCohort::kSpatialShuffle:
      return "spatial-shuffle";
    case TrainingCohort::kFixedOffset:
      return "fixed-offset";
  }
  return "unknown";
}

void print_topology_correlation_json(
    float correlation, int count, bool valid,
    const char* count_name) {
  std::cout << "{\"correlation\":";
  write_json_number(correlation);
  std::cout << ",\"" << count_name << "\":" << count
            << ",\"valid\":" << (valid ? "true" : "false")
            << '}';
}

void print_topology_state_json(
    const TopologyStateMetrics& state) {
  std::cout << "{\"auditory_to_excitatory\":";
  print_topology_correlation_json(
      state
          .auditory_to_excitatory_proximity_efficacy_correlation,
      state.auditory_to_excitatory_contact_count,
      state.auditory_to_excitatory_valid, "contact_count");
  std::cout << ",\"visual_to_excitatory\":";
  print_topology_correlation_json(
      state
          .visual_to_excitatory_proximity_efficacy_correlation,
      state.visual_to_excitatory_contact_count,
      state.visual_to_excitatory_valid, "contact_count");
  std::cout << ",\"recurrent_excitatory\":";
  print_topology_correlation_json(
      state.recurrent_proximity_efficacy_correlation,
      state.recurrent_contact_count, state.recurrent_valid,
      "contact_count");
  std::cout << ",\"effective_inhibition\":";
  print_topology_correlation_json(
      state.effective_inhibitory_order_correlation,
      state.effective_inhibitory_common_target_count,
      state.effective_inhibitory_valid, "common_target_count");
  std::cout << ",\"auditory_rf_order\":";
  print_topology_correlation_json(
      state.auditory_rf_order_correlation,
      state.rf_common_neuron_count,
      state.auditory_rf_order_valid, "common_neuron_count");
  std::cout << ",\"visual_rf_order\":";
  print_topology_correlation_json(
      state.visual_rf_order_correlation,
      state.rf_common_neuron_count,
      state.visual_rf_order_valid, "common_neuron_count");
  std::cout << ",\"rf_center_mismatch\":{\"median_deg\":";
  write_json_number(state.median_rf_center_mismatch_deg);
  std::cout << ",\"common_neuron_count\":"
            << state.rf_common_neuron_count
            << ",\"valid\":"
            << (state.rf_mismatch_valid ? "true" : "false")
            << "}}";
}

void print_topology_refinement_json(
    int device, const EvaluationOptions& options,
    const EvaluationMetrics& metrics) {
  const TopologyRefinementSummary& summary =
      metrics.topology_refinement;
  std::cout
      << "{\"kind\":\"topology_refinement\",\"device\":"
      << device
      << ",\"definitions\":{"
      << "\"projection_formula\":"
      << "\"Pearson(p=-abs(pre_index-post_index),efficacy)\","
      << "\"projection_paths\":"
      << "\"A_to_E,V_to_E,E_to_E_nonself\","
      << "\"effective_inhibition_formula\":"
      << "\"K_m(s,e)=sum_i G_m(s,i)*H(i,e); "
         "mean_of_A_and_V_target_center_order_correlations\","
      << "\"rf_mismatch_formula\":"
      << "\"median_e(abs(A_center_e-V_center_e))\","
      << "\"delta_sign_convention\":\"trained_minus_initial\","
      << "\"weight_common_mask\":"
      << "\"initial_active_contacts\","
      << "\"rf_common_intersection\":"
      << "\"E_neurons_with_valid_A_and_V_centers_in_both_states\","
      << "\"rf_order_is_reported_not_gated\":true,"
      << "\"seed_exclusions\":\"none\"},"
      << "\"provenance\":{\"training_cohort\":\""
      << training_cohort_name(summary.training_cohort)
      << "\",\"control_flags\":"
      << static_cast<std::uint32_t>(summary.control)
      << ",\"seed_count\":" << summary.seed_count
      << ",\"minimum_presentations_per_seed\":"
      << summary.minimum_presentations_per_seed
      << ",\"pruning_ever_enabled\":"
      << (summary.pruning_ever_enabled ? "true" : "false")
      << ",\"all_initial_trained_masks_exact\":"
      << (summary.all_initial_trained_masks_exact
              ? "true"
              : "false")
      << ",\"all_fixed_A_to_I_and_V_to_I_weights_exact\":"
      << (summary
                  .all_fixed_sensory_to_inhibitory_weights_exact
              ? "true"
              : "false")
      << ",\"rf_pairing\":{\"evaluation_seed\":"
      << (options.evaluation_seed ^ 0x5246ull)
      << ",\"trials_per_location\":"
      << options.trials_per_rf_location
      << ",\"burn_in_ms\":" << options.burn_in_ms
      << ",\"identical_conditions\":true,"
      << "\"identical_random_groups\":true,"
      << "\"identical_evaluation_seed\":true}},"
      << "\"applicability\":{\"scope\":"
      << "\"baseline_uncontrolled_exactly_5_seeds_each_at_least_"
         "2000_presentations_pruning_never_enabled_exact_masks\","
      << "\"evaluated\":"
      << (summary.evaluated ? "true" : "false")
      << ",\"reason\":";
  if (summary.reason.empty()) {
    std::cout << "null";
  } else {
    std::cout << '"' << summary.reason << '"';
  }
  std::cout << ",\"gate\":";
  if (!summary.evaluated) {
    std::cout << "null";
  } else {
    std::cout << (summary.passed ? "true" : "false");
  }
  std::cout << "},\"per_seed\":[";
  for (std::size_t index = 0;
       index < metrics.per_seed.size(); ++index) {
    if (index != 0u) {
      std::cout << ',';
    }
    const SeedEvaluationMetrics& seed = metrics.per_seed[index];
    const SeedTopologyAudit& topology = seed.topology;
    std::cout << "{\"seed_index\":" << seed.seed_index
              << ",\"global_seed\":" << seed.global_seed
              << ",\"validity\":{"
              << "\"all_required_metrics_valid\":"
              << (topology.all_required_metrics_valid
                      ? "true"
                      : "false")
              << ",\"initial_trained_masks_exact\":"
              << (topology.initial_trained_masks_exact
                      ? "true"
                      : "false")
              << ",\"A_to_I_weights_exact\":"
              << (topology.auditory_to_inhibitory_weights_exact
                      ? "true"
                      : "false")
              << ",\"V_to_I_weights_exact\":"
              << (topology.visual_to_inhibitory_weights_exact
                      ? "true"
                      : "false")
              << "},\"initial\":";
    print_topology_state_json(topology.initial_state);
    std::cout << ",\"trained\":";
    print_topology_state_json(topology.trained_state);
    std::cout << ",\"deltas\":{"
              << "\"auditory_to_excitatory\":";
    write_json_number(
        topology
            .auditory_to_excitatory_proximity_efficacy_delta);
    std::cout << ",\"visual_to_excitatory\":";
    write_json_number(
        topology
            .visual_to_excitatory_proximity_efficacy_delta);
    std::cout << ",\"recurrent_excitatory\":";
    write_json_number(
        topology.recurrent_proximity_efficacy_delta);
    std::cout << ",\"effective_inhibition\":";
    write_json_number(
        topology.effective_inhibitory_order_delta_explicit);
    std::cout << ",\"median_rf_center_mismatch_deg\":";
    write_json_number(
        topology.median_rf_center_mismatch_delta_deg);
    std::cout << "}}";
  }
  std::cout << "],\"summary\":{\"all_values_valid\":"
            << (summary.all_values_valid ? "true" : "false")
            << ",\"auditory_to_excitatory\":{\"mean_delta\":";
  write_json_number(
      summary.mean_auditory_to_excitatory_delta);
  std::cout << ",\"positive_seed_count\":"
            << summary
                   .auditory_to_excitatory_positive_seed_count
            << "},\"visual_to_excitatory\":{\"mean_delta\":";
  write_json_number(
      summary.mean_visual_to_excitatory_delta);
  std::cout << ",\"positive_seed_count\":"
            << summary.visual_to_excitatory_positive_seed_count
            << "},\"recurrent_excitatory\":{\"mean_delta\":";
  write_json_number(summary.mean_recurrent_delta);
  std::cout << ",\"positive_seed_count\":"
            << summary.recurrent_positive_seed_count
            << "},\"effective_inhibition\":{\"mean_delta\":";
  write_json_number(
      summary.mean_effective_inhibitory_delta);
  std::cout << ",\"positive_seed_count\":"
            << summary.effective_inhibitory_positive_seed_count
            << "},\"rf_center_mismatch\":{\"mean_delta_deg\":";
  write_json_number(summary.mean_rf_mismatch_delta_deg);
  std::cout << ",\"negative_seed_count\":"
            << summary.rf_mismatch_negative_seed_count
            << "}}}\n";
}

void print_inverse_json(
    int device, const EvaluationMetrics& metrics) {
  const InverseEffectivenessAudit& inverse =
      metrics.inverse_effectiveness;
  std::cout << "{\"kind\":\"inverse_effectiveness\",\"device\":"
            << device << ",\"trials_per_salience\":"
            << inverse.trials_per_salience << ",\"points\":[";
  for (int index = 0; index < kInverseEffectivenessCount; ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << "{\"salience_hz\":";
    write_json_number(
        inverse.salience_hz[static_cast<std::size_t>(index)]);
    std::cout << ",\"responses\":";
    print_component_json(
        inverse.responses[static_cast<std::size_t>(index)]);
    std::cout << '}';
  }
  std::cout << "]}\n";
}

void print_control_json(
    int device, const EvaluationMetrics& metrics) {
  const ProjectionControlAudit& control = metrics.control_audit;
  std::cout << "{\"kind\":\"causal_control\",\"device\":" << device
            << ",\"flags\":"
            << static_cast<std::uint32_t>(control.control)
            << ",\"ampa_scale\":";
  write_float_array(control.ampa_scale);
  std::cout << ",\"nmda_scale\":";
  write_float_array(control.nmda_scale);
  std::cout << ",\"gabaa_scale\":";
  write_float_array(control.gabaa_scale);
  std::cout << ",\"source_rows_shuffled\":";
  write_bool_array(control.source_rows_shuffled);
  std::cout << ",\"external_nmda_enabled\":"
            << (control.external_nmda_enabled ? "true" : "false")
            << ",\"background_nmda_enabled\":"
            << (control.background_nmda_enabled ? "true" : "false")
            << "}\n";
}

const char* causal_control_name(CausalControl control) {
  switch (control) {
    case CausalControl::kNmdaOff:
      return "nmda-off";
    case CausalControl::kGabaaOff:
      return "gabaa-off";
    case CausalControl::kRecurrentExcitationOff:
      return "recurrent-off";
    case CausalControl::kRecruitedInhibitionOff:
      return "recruited-inhibition-off";
    case CausalControl::kSourceRowShuffle:
      return "source-row-shuffle";
    case CausalControl::kMsiEAdaptationOff:
      return "msi-e-adaptation-off";
    case CausalControl::kNone:
      return "none";
  }
  return "unknown";
}

bool observer_parameters_identical(
    const ObserverMetrics& first,
    const ObserverMetrics& second) {
  return first.intercept == second.intercept &&
         first.l2 == second.l2 &&
         first.feature_mean == second.feature_mean &&
         first.feature_std == second.feature_std &&
         first.coefficients == second.coefficients;
}

void print_paired_number(float baseline, float control) {
  std::cout << "{\"baseline\":";
  write_json_number(baseline);
  std::cout << ",\"control\":";
  write_json_number(control);
  std::cout << ",\"delta\":";
  write_json_number(control - baseline);
  std::cout << '}';
}

void print_paired_control_json(
    int device, const EvaluationMetrics& baseline,
    const CausalControlEvaluation& control) {
  const EvaluationMetrics& changed = control.evaluation;
  std::cout << "{\"kind\":\"paired_causal_control\",\"device\":"
            << device << ",\"control\":\""
            << causal_control_name(control.control)
            << "\",\"flags\":"
            << static_cast<std::uint32_t>(control.control)
            << ",\"observer_frozen\":"
            << (control.observer_frozen ? "true" : "false")
            << ",\"observer_parameters_identical\":"
            << (observer_parameters_identical(
                    baseline.observer, changed.observer)
                    ? "true"
                    : "false")
            << ",\"tbw\":{\"tbw50_ms\":";
  print_paired_number(baseline.tbw50_ms, changed.tbw50_ms);
  std::cout << ",\"tbw75_ms\":";
  print_paired_number(baseline.tbw75_ms, changed.tbw75_ms);
  std::cout << ",\"points\":[";
  for (int index = 0; index < kTbwPointCount; ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    const TemporalCurvePoint& reference =
        baseline.temporal.points[static_cast<std::size_t>(index)];
    const TemporalCurvePoint& value =
        changed.temporal.points[static_cast<std::size_t>(index)];
    std::cout << "{\"physical_soa_ms\":";
    write_json_number(reference.physical_soa_ms);
    std::cout << ",\"probability\":";
    print_paired_number(
        reference.probability_mean, value.probability_mean);
    std::cout << ",\"R_A_hz\":";
    print_paired_number(
        reference.auditory_rate_hz, value.auditory_rate_hz);
    std::cout << ",\"R_V_hz\":";
    print_paired_number(
        reference.visual_rate_hz, value.visual_rate_hz);
    std::cout << ",\"R_AV_hz\":";
    print_paired_number(
        reference.audiovisual_rate_hz,
        value.audiovisual_rate_hz);
    std::cout << ",\"G_hz\":";
    print_paired_number(
        reference.raw_neural_enhancement_hz,
        value.raw_neural_enhancement_hz);
    std::cout << '}';
  }
  std::cout
      << "]},\"sbw\":{\"sbw50_method\":\"gaussian_fitted_hwhm\""
      << ",\"spatial_gate\":{\"baseline\":"
            << (baseline.spatial_gate ? "true" : "false")
            << ",\"control\":"
            << (changed.spatial_gate ? "true" : "false")
            << "},\"sbw50_deg\":";
  print_paired_number(baseline.sbw50_deg, changed.sbw50_deg);
  std::cout << ",\"direct_raw_sbw50_deg\":";
  print_paired_number(
      baseline.spatial.direct.sbw50_deg,
      changed.spatial.direct.sbw50_deg);
  std::cout << ",\"points\":[";
  for (int index = 0; index < kSbwPointCount; ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    const SpatialCurvePoint& reference =
        baseline.spatial.points[static_cast<std::size_t>(index)];
    const SpatialCurvePoint& value =
        changed.spatial.points[static_cast<std::size_t>(index)];
    std::cout << "{\"disparity_deg\":";
    write_json_number(reference.disparity_deg);
    std::cout << ",\"R_A_hz\":";
    print_paired_number(
        reference.pooled.auditory_rate_hz,
        value.pooled.auditory_rate_hz);
    std::cout << ",\"R_V_hz\":";
    print_paired_number(
        reference.pooled.visual_rate_hz,
        value.pooled.visual_rate_hz);
    std::cout << ",\"R_AV_hz\":";
    print_paired_number(
        reference.pooled.audiovisual_rate_hz,
        value.pooled.audiovisual_rate_hz);
    std::cout << ",\"G_hz\":";
    print_paired_number(
        reference.pooled.raw_enhancement_hz,
        value.pooled.raw_enhancement_hz);
    std::cout << ",\"ME_percent\":";
    print_paired_number(
        reference.pooled.multisensory_enhancement_percent,
        value.pooled.multisensory_enhancement_percent);
    std::cout << ",\"additivity_hz\":";
    print_paired_number(
        reference.pooled.additivity_hz,
        value.pooled.additivity_hz);
    std::cout << ",\"additivity_percent\":";
    print_paired_number(
        reference.pooled.additivity_percent,
        value.pooled.additivity_percent);
    std::cout << '}';
  }
  std::cout << "]}}\n";
}

void print_evaluation_json(
    int device, const EvaluationOptions& options,
    const EvaluationMetrics& metrics) {
  std::cout << std::setprecision(9)
            << "{\"kind\":\"evaluation_summary\",\"device\":"
            << device << ",\"simulated_trials\":"
            << metrics.simulated_trials
            << ",\"frozen_kernel_seconds\":";
  write_json_number(metrics.frozen_kernel_seconds);
  std::cout << ",\"elapsed_seconds\":";
  write_json_number(metrics.elapsed_seconds);
  std::cout << ",\"burn_in_ms\":" << options.burn_in_ms
            << ",\"response_window_ms\":[0,250]"
            << ",\"feature_window_ms\":[-100,700]"
            << ",\"physical_soa_definition\":\"tV_minus_tA\""
            << ",\"received_reference\":\"earlier_latency_shifted_onset\""
            << ",\"tbw50_ms\":";
  write_json_number(metrics.tbw50_ms);
  std::cout << ",\"tbw75_ms\":";
  write_json_number(metrics.tbw75_ms);
  std::cout
      << ",\"sbw50_method\":\"gaussian_fitted_hwhm\""
      << ",\"sbw50_deg\":";
  write_json_number(metrics.sbw50_deg);
  std::cout << ",\"direct_raw_sbw50_deg\":";
  write_json_number(metrics.spatial.direct.sbw50_deg);
  std::cout << ",\"spatial_gate\":"
            << (metrics.spatial_gate ? "true" : "false");
  std::cout << ",\"topology_refinement_gate\":";
  if (!metrics.topology_refinement.evaluated) {
    std::cout << "null";
  } else {
    std::cout
        << (metrics.topology_refinement.passed
                ? "true"
                : "false");
  }
  std::cout << ",\"criteria_passed\":"
            << (metrics.criteria_passed ? "true" : "false")
            << "}\n";
  print_observer_json(device, options, metrics);
  print_tbw_json(device, metrics);
  print_sbw_json(device, metrics);
  print_rf_json(device, metrics);
  print_topology_refinement_json(device, options, metrics);
  print_inverse_json(device, metrics);
  print_control_json(device, metrics);
}

}  // namespace

int cli_main(int argc, char** argv) {
  std::string command = "calibrate";
  int device = 0;
  std::uint64_t seed = 0;
  int presentations = 10000;
  int seeds = 1;
  int chunk_presentations = kTrainingChunkPresentations;
  bool all_devices = false;
  bool all_controls = false;
  EvaluationOptions evaluation_options{};
  if (argc >= 2) {
    command = argv[1];
  }
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--device" && index + 1 < argc) {
      device = std::stoi(argv[++index]);
    } else if (argument == "--seed" && index + 1 < argc) {
      seed = static_cast<std::uint64_t>(
          std::stoull(argv[++index]));
    } else if (argument == "--presentations" && index + 1 < argc) {
      presentations = std::stoi(argv[++index]);
    } else if (argument == "--seeds" && index + 1 < argc) {
      seeds = std::stoi(argv[++index]);
    } else if (argument == "--chunk" && index + 1 < argc) {
      chunk_presentations = std::stoi(argv[++index]);
    } else if (argument == "--observer-train" &&
               index + 1 < argc) {
      evaluation_options.observer_training_trials =
          std::stoi(argv[++index]);
    } else if (argument == "--observer-holdout" &&
               index + 1 < argc) {
      evaluation_options.observer_holdout_trials =
          std::stoi(argv[++index]);
    } else if (argument == "--tbw-trials" && index + 1 < argc) {
      evaluation_options.trials_per_tbw_soa =
          std::stoi(argv[++index]);
    } else if (argument == "--sbw-trials" && index + 1 < argc) {
      evaluation_options.trials_per_sbw_condition =
          std::stoi(argv[++index]);
    } else if (argument == "--rf-trials" && index + 1 < argc) {
      evaluation_options.trials_per_rf_location =
          std::stoi(argv[++index]);
    } else if (argument == "--inverse-trials" &&
               index + 1 < argc) {
      evaluation_options.trials_per_inverse_condition =
          std::stoi(argv[++index]);
    } else if (argument == "--burn-in" && index + 1 < argc) {
      evaluation_options.burn_in_ms = std::stoi(argv[++index]);
    } else if (argument == "--evaluation-seed" &&
               index + 1 < argc) {
      evaluation_options.evaluation_seed =
          static_cast<std::uint64_t>(
              std::stoull(argv[++index]));
    } else if (argument == "--evaluation-rate" &&
               index + 1 < argc) {
      evaluation_options.observer_rate_hz =
          std::stof(argv[++index]);
    } else if (argument == "--response-floor" &&
               index + 1 < argc) {
      evaluation_options.response_floor_hz =
          std::stof(argv[++index]);
    } else if (argument == "--control" && index + 1 < argc) {
      evaluation_options.control =
          parse_causal_control(argv[++index]);
    } else if (argument == "--all-devices") {
      all_devices = true;
    } else if (argument == "--all-controls") {
      all_controls = true;
    } else {
      throw std::invalid_argument("Unknown or incomplete CLI argument.");
    }
  }
  if (command != "calibrate" && command != "train" &&
      command != "validate" && command != "train-validate") {
    throw std::invalid_argument(
        "Native CLI command must be `calibrate`, `train`, "
        "`validate`, or `train-validate`.");
  }
  if (all_controls &&
      (command != "train-validate" ||
       evaluation_options.control != CausalControl::kNone)) {
    throw std::invalid_argument(
        "--all-controls requires an uncontrolled train-validate run.");
  }
  Config config{};
  config.seed = seed;
  if (command == "calibrate") {
    const CalibrationResult result = calibrate(device, config);
    std::cout << std::setprecision(9)
              << "{\"kind\":\"calibration\",\"device\":" << device
              << ",\"q_external_RS\":" << result.config.q_external_rs
              << ",\"q_background_RS\":" << result.config.q_background_rs
              << ",\"q_background_FS\":" << result.config.q_background_fs
              << ",\"q_ff_E\":" << result.config.q_ff_e
              << ",\"q_ff_I\":" << result.config.q_ff_i
              << ",\"q_GABAA\":" << result.config.q_gabaa
              << ",\"gabaa_unitary_target_basis\":"
                 "\"comparative_provisional_deep_sc\""
              << ",\"gabaa_unitary_target_mV\":"
              << kGabaaUnitaryTargetMv
              << ",\"gabaa_unitary_ipsp_mV\":"
              << result.gabaa_efficacy.unitary.amplitude_mv
              << ",\"gabaa_unitary_signed_nadir_mV\":"
              << result.gabaa_efficacy.unitary.signed_nadir_mv
              << ",\"gabaa_unitary_area_mV_ms\":"
              << result.gabaa_efficacy.unitary.area_mv_ms
              << ",\"gabaa_unitary_outward_charge\":"
              << result.gabaa_efficacy.unitary.outward_charge
              << ",\"gabaa_compound_contacts\":"
              << result.gabaa_efficacy.compound.contact_count
              << ",\"gabaa_compound_ipsp_mV\":"
              << result.gabaa_efficacy.compound.amplitude_mv
              << ",\"gabaa_compound_signed_nadir_mV\":"
              << result.gabaa_efficacy.compound.signed_nadir_mv
              << ",\"gabaa_compound_area_mV_ms\":"
              << result.gabaa_efficacy.compound.area_mv_ms
              << ",\"gabaa_compound_outward_charge\":"
              << result.gabaa_efficacy.compound.outward_charge
              << ",\"gabaa_relay_spikes\":"
              << result.gabaa_relay.relay_spikes
              << ",\"gabaa_arrivals\":"
              << result.gabaa_relay.gabaa_arrivals
              << ",\"gabaa_causal_without_spikes\":"
              << result.gabaa_relay.without_gabaa_spikes
              << ",\"gabaa_causal_with_spikes\":"
              << result.gabaa_relay.with_gabaa_spikes
              << ",\"gabaa_lag_min_ms\":"
              << result.gabaa_relay.minimum_lag_ms
              << ",\"gabaa_lag_max_ms\":"
              << result.gabaa_relay.maximum_lag_ms
              << ",\"gabaa_lag_mean_ms\":"
              << result.gabaa_relay.mean_lag_ms
              << ",\"gabaa_lag_counts\":[";
    for (int lag = 0; lag < kGabaaDurationMs; ++lag) {
      if (lag != 0) {
        std::cout << ',';
      }
      std::cout
          << result.gabaa_relay
                 .direct_to_gabaa_lag_counts[lag];
    }
    std::cout << ']'
              << ",\"eta_clopath_ff\":"
              << result.config.eta_clopath_ff
              << ",\"eta_oja\":" << result.config.eta_oja
              << ",\"eta_istdp\":" << result.config.eta_istdp
              << ",\"eta_istdp_basis\":\"fixed_primary\""
              << ",\"istdp_assay_basis\":"
                 "\"native_81_point_grid_row_16\""
              << ",\"istdp_assay_role\":"
                 "\"diagnostic_nonsaturation\""
              << ",\"istdp_alpha_setpoint_hz\":"
              << kIstdpTargetRateHz
              << ",\"istdp_weak_rate_hz\":"
              << result.learning.istdp_weak_rate_hz
              << ",\"istdp_strong_rate_hz\":"
              << result.learning.istdp_strong_rate_hz
              << ",\"istdp_weak_bound_fraction\":"
              << result.learning.istdp_weak_bound_fraction
              << ",\"istdp_strong_bound_fraction\":"
              << result.learning.istdp_strong_bound_fraction
              << ",\"conductance_seconds\":"
              << result.conductance_seconds
              << ",\"learning_seconds\":" << result.learning_seconds
              << ",\"criteria_passed\":"
              << (result.criteria_passed ? "true" : "false")
              << "}\n";
    return 0;
  }

  if (command == "validate" || command == "train-validate") {
    struct ValidationRun {
      int device = 0;
      int seeds = 0;
      bool train_first = false;
      TrainingMetrics training{};
      std::vector<TrainingDiagnostics> diagnostics;
      EvaluationMetrics evaluation{};
      std::vector<CausalControlEvaluation> controls;
      std::exception_ptr error;
    };
    const bool train_first = command == "train-validate";
    const auto run_validation =
        [presentations, chunk_presentations, evaluation_options,
         all_controls](
            ValidationRun* run, Config run_config) {
          try {
            const CalibrationResult calibration =
                calibrate(run->device, run_config);
            NativeModel model(
                calibration.config, run->device, run->seeds);
            if (run->train_first) {
              TrainingOptions training_options{};
              training_options.presentations = presentations;
              training_options.seed_count = run->seeds;
              training_options.chunk_presentations =
                  chunk_presentations;
              run->training = model.train(training_options);
              run->diagnostics = model.diagnostics();
            }
            run->evaluation = model.evaluate(evaluation_options);
            if (all_controls) {
              run->controls =
                  model.evaluate_all_controls(
                      evaluation_options, run->evaluation);
            }
          } catch (...) {
            run->error = std::current_exception();
          }
        };

    std::vector<ValidationRun> validation_runs;
    if (all_devices) {
      int device_count = 0;
      CLEAN_MSI_CUDA(cudaGetDeviceCount(&device_count));
      if (device_count < 2) {
        throw std::runtime_error(
            "--all-devices requires cuda:0 and cuda:1.");
      }
      validation_runs.resize(2);
      validation_runs[0].device = 0;
      validation_runs[0].seeds = 8;
      validation_runs[0].train_first = train_first;
      validation_runs[1].device = 1;
      validation_runs[1].seeds = 4;
      validation_runs[1].train_first = train_first;
      Config second_config = config;
      second_config.seed += 8;
      std::thread first(
          run_validation, &validation_runs[0], config);
      std::thread second(
          run_validation, &validation_runs[1], second_config);
      first.join();
      second.join();
    } else {
      validation_runs.resize(1);
      validation_runs[0].device = device;
      validation_runs[0].seeds = seeds;
      validation_runs[0].train_first = train_first;
      run_validation(&validation_runs[0], config);
    }
    for (const ValidationRun& run : validation_runs) {
      if (run.error) {
        std::rethrow_exception(run.error);
      }
    }
    for (const ValidationRun& run : validation_runs) {
      if (run.train_first) {
        std::uint64_t accepted_steps = 0;
        int presentation_index = 0;
        for (const TrainingDiagnostics& diagnostic :
             run.diagnostics) {
          accepted_steps += diagnostic.accepted_steps;
          presentation_index =
              std::max(
                  presentation_index,
                  diagnostic.presentation_index);
        }
        std::cout << std::setprecision(9)
                  << "{\"kind\":\"training\",\"device\":"
                  << run.device << ",\"seeds\":" << run.seeds
                  << ",\"presentations\":"
                  << run.training.completed_presentations
                  << ",\"presentation_index\":"
                  << presentation_index
                  << ",\"accepted_steps\":" << accepted_steps
                  << ",\"A_spikes\":"
                  << run.training.total_a_spikes
                  << ",\"V_spikes\":"
                  << run.training.total_v_spikes
                  << ",\"E_spikes\":"
                  << run.training.total_e_spikes
                  << ",\"I_spikes\":"
                  << run.training.total_i_spikes
                  << ",\"mean_AE\":"
                  << run.training.mean_a_to_e
                  << ",\"mean_VE\":"
                  << run.training.mean_v_to_e
                  << ",\"mean_EE\":"
                  << run.training.mean_e_to_e
                  << ",\"mean_IE\":"
                  << run.training.mean_i_to_e
                  << ",\"pruned\":"
                  << run.training.pruned_contacts
                  << ",\"elapsed_seconds\":"
                  << run.training.elapsed_seconds << "}\n";
      }
      print_evaluation_json(
          run.device, evaluation_options, run.evaluation);
      for (const CausalControlEvaluation& control : run.controls) {
        print_paired_control_json(
            run.device, run.evaluation, control);
      }
    }
    return 0;
  }

  struct TrainingRun {
    int device = 0;
    int seeds = 0;
    TrainingMetrics metrics{};
    std::vector<TrainingDiagnostics> diagnostics;
    std::exception_ptr error;
  };
  const auto run_training =
      [presentations, chunk_presentations](
          TrainingRun* run, Config run_config) {
        try {
          const CalibrationResult calibration =
              calibrate(run->device, run_config);
          NativeModel model(
              calibration.config, run->device, run->seeds);
          TrainingOptions options{};
          options.presentations = presentations;
          options.seed_count = run->seeds;
          options.chunk_presentations = chunk_presentations;
          run->metrics = model.train(options);
          run->diagnostics = model.diagnostics();
        } catch (...) {
          run->error = std::current_exception();
        }
      };

  std::vector<TrainingRun> runs;
  if (all_devices) {
    int device_count = 0;
    CLEAN_MSI_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
      throw std::runtime_error(
          "--all-devices requires cuda:0 and cuda:1.");
    }
    runs.resize(2);
    runs[0].device = 0;
    runs[0].seeds = 8;
    runs[1].device = 1;
    runs[1].seeds = 4;
    Config second_config = config;
    second_config.seed += 8;
    std::thread first(run_training, &runs[0], config);
    std::thread second(run_training, &runs[1], second_config);
    first.join();
    second.join();
  } else {
    runs.resize(1);
    runs[0].device = device;
    runs[0].seeds = seeds;
    run_training(&runs[0], config);
  }
  for (const TrainingRun& run : runs) {
    if (run.error) {
      std::rethrow_exception(run.error);
    }
  }
  for (const TrainingRun& run : runs) {
    std::uint64_t accepted_steps = 0;
    int presentation_index = 0;
    for (const TrainingDiagnostics& diagnostic : run.diagnostics) {
      accepted_steps += diagnostic.accepted_steps;
      presentation_index =
          std::max(presentation_index, diagnostic.presentation_index);
    }
    std::cout << std::setprecision(9)
              << "{\"kind\":\"training\",\"device\":" << run.device
              << ",\"seeds\":" << run.seeds
              << ",\"presentations\":"
              << run.metrics.completed_presentations
              << ",\"presentation_index\":" << presentation_index
              << ",\"accepted_steps\":" << accepted_steps
              << ",\"A_spikes\":" << run.metrics.total_a_spikes
              << ",\"V_spikes\":" << run.metrics.total_v_spikes
              << ",\"E_spikes\":" << run.metrics.total_e_spikes
              << ",\"I_spikes\":" << run.metrics.total_i_spikes
              << ",\"mean_AE\":" << run.metrics.mean_a_to_e
              << ",\"mean_VE\":" << run.metrics.mean_v_to_e
              << ",\"mean_EE\":" << run.metrics.mean_e_to_e
              << ",\"mean_IE\":" << run.metrics.mean_i_to_e
              << ",\"pruned\":" << run.metrics.pruned_contacts
              << ",\"elapsed_seconds\":"
              << run.metrics.elapsed_seconds << "}\n";
  }
  return 0;
}

}  // namespace clean_msi

#ifndef CLEAN_MSI_NO_MAIN
int main(int argc, char** argv) {
  try {
    return clean_msi::cli_main(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "clean_msi: " << error.what() << '\n';
    return 1;
  }
}
#endif
