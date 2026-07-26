#include "../clean_msi.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using clean_msi::AssayKind;
using clean_msi::AssayMetrics;
using clean_msi::CalibrationResult;
using clean_msi::CausalControl;
using clean_msi::Config;
using clean_msi::ControlledCondition;
using clean_msi::DevelopmentalSampleAudit;
using clean_msi::EvaluationMetrics;
using clean_msi::EvaluationOptions;
using clean_msi::FrozenTrialBatch;
using clean_msi::FrozenTrialResult;
using clean_msi::FrozenWeightAudit;
using clean_msi::GeneratorProfileAudit;
using clean_msi::LearningAssayMetrics;
using clean_msi::NativeModel;
using clean_msi::PathDiagnostics;
using clean_msi::PlasticPath;
using clean_msi::Population;
using clean_msi::PopulationDiagnostics;
using clean_msi::PresentationKind;
using clean_msi::ReceptorKernel;
using clean_msi::SolverInput;
using clean_msi::SolverOutput;
using clean_msi::TrainingDiagnostics;
using clean_msi::TrainingMetrics;
using clean_msi::TrainingOptions;

class TestFailure final : public std::runtime_error {
 public:
  explicit TestFailure(const std::string& message)
      : std::runtime_error(message) {}
};

[[noreturn]] void fail(const char* file, int line,
                       const std::string& message) {
  std::ostringstream stream;
  stream << file << ':' << line << ": " << message;
  throw TestFailure(stream.str());
}

#define REQUIRE(condition)                                                  \
  do {                                                                      \
    if (!(condition)) {                                                     \
      fail(__FILE__, __LINE__, "requirement failed: " #condition);          \
    }                                                                       \
  } while (false)

#define REQUIRE_MESSAGE(condition, message)                                 \
  do {                                                                      \
    if (!(condition)) {                                                     \
      fail(__FILE__, __LINE__, (message));                                  \
    }                                                                       \
  } while (false)

void require_finite(float value, const std::string& label) {
  REQUIRE_MESSAGE(std::isfinite(value), label + " is not finite");
}

void require_near(float actual, float expected, float absolute_tolerance,
                  float relative_tolerance, const std::string& label) {
  require_finite(actual, label + " actual");
  require_finite(expected, label + " expected");
  const float allowance =
      absolute_tolerance + relative_tolerance * std::fabs(expected);
  if (std::fabs(actual - expected) > allowance) {
    std::ostringstream stream;
    stream << std::setprecision(9) << label << ": actual=" << actual
           << ", expected=" << expected << ", allowance=" << allowance;
    fail(__FILE__, __LINE__, stream.str());
  }
}

void require_exact_metrics(const AssayMetrics& actual,
                           const AssayMetrics& expected,
                           const std::string& label) {
  REQUIRE_MESSAGE(actual.primary == expected.primary,
                  label + ".primary differs");
  REQUIRE_MESSAGE(actual.secondary == expected.secondary,
                  label + ".secondary differs");
  REQUIRE_MESSAGE(actual.tertiary == expected.tertiary,
                  label + ".tertiary differs");
}

void require_learning_near(const LearningAssayMetrics& actual,
                           const LearningAssayMetrics& expected,
                           float tolerance, const std::string& label) {
  require_near(actual.clopath_delta, expected.clopath_delta, tolerance,
               tolerance, label + ".clopath_delta");
  require_near(actual.clopath_shuffled_drift,
               expected.clopath_shuffled_drift, tolerance, tolerance,
               label + ".clopath_shuffled_drift");
  require_near(actual.oja_selection_ratio, expected.oja_selection_ratio,
               tolerance, tolerance, label + ".oja_selection_ratio");
  require_near(actual.oja_bound_fraction, expected.oja_bound_fraction,
               tolerance, tolerance, label + ".oja_bound_fraction");
  require_near(actual.istdp_weak_rate_hz, expected.istdp_weak_rate_hz,
               tolerance, tolerance, label + ".istdp_weak_rate_hz");
  require_near(actual.istdp_strong_rate_hz, expected.istdp_strong_rate_hz,
               tolerance, tolerance, label + ".istdp_strong_rate_hz");
  require_near(actual.istdp_weak_bound_fraction,
               expected.istdp_weak_bound_fraction, tolerance, tolerance,
               label + ".istdp_weak_bound_fraction");
  require_near(actual.istdp_strong_bound_fraction,
               expected.istdp_strong_bound_fraction, tolerance, tolerance,
               label + ".istdp_strong_bound_fraction");
}

std::vector<int> available_devices() {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice ||
      status == cudaErrorInsufficientDriver) {
    cudaGetLastError();
    return {};
  }
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("cudaGetDeviceCount failed: ") +
        cudaGetErrorString(status));
  }
  std::vector<int> devices;
  for (int device = 0; device < std::min(count, 2); ++device) {
    devices.push_back(device);
  }
  return devices;
}

void synchronize_device(int device) {
  const cudaError_t set_status = cudaSetDevice(device);
  if (set_status != cudaSuccess) {
    throw std::runtime_error(
        std::string("cudaSetDevice failed: ") +
        cudaGetErrorString(set_status));
  }
  const cudaError_t sync_status = cudaDeviceSynchronize();
  if (sync_status != cudaSuccess) {
    throw std::runtime_error(
        std::string("cudaDeviceSynchronize failed: ") +
        cudaGetErrorString(sync_status));
  }
}

template <typename Function>
double timed_seconds(int device, Function&& function) {
  synchronize_device(device);
  const auto begin = std::chrono::steady_clock::now();
  function();
  synchronize_device(device);
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now() - begin)
      .count();
}

// ---------------------------------------------------------------------------
// Independent scalar references. None of these functions call production
// helpers; they are deliberately simple equations against which CUDA results
// are checked.

constexpr std::uint32_t kPhiloxM0 = 0xD2511F53u;
constexpr std::uint32_t kPhiloxM1 = 0xCD9E8D57u;
constexpr std::uint32_t kPhiloxW0 = 0x9E3779B9u;
constexpr std::uint32_t kPhiloxW1 = 0xBB67AE85u;

std::pair<std::uint32_t, std::uint32_t> scalar_mul_hi_lo(
    std::uint32_t lhs, std::uint32_t rhs) {
  const std::uint64_t product =
      static_cast<std::uint64_t>(lhs) * static_cast<std::uint64_t>(rhs);
  return {static_cast<std::uint32_t>(product >> 32u),
          static_cast<std::uint32_t>(product)};
}

std::array<std::uint32_t, 4> scalar_philox4x32_10(
    std::array<std::uint32_t, 4> counter,
    std::array<std::uint32_t, 2> key) {
  for (int round = 0; round < 10; ++round) {
    const auto product0 = scalar_mul_hi_lo(kPhiloxM0, counter[0]);
    const auto product1 = scalar_mul_hi_lo(kPhiloxM1, counter[2]);
    counter = {
        product1.first ^ counter[1] ^ key[0],
        product1.second,
        product0.first ^ counter[3] ^ key[1],
        product0.second,
    };
    if (round != 9) {
      key[0] += kPhiloxW0;
      key[1] += kPhiloxW1;
    }
  }
  return counter;
}

float scalar_nmda_block(float voltage_mv) {
  const float exponent =
      std::clamp(-0.062f * voltage_mv, -80.0f, 80.0f);
  return 1.0f / (1.0f + std::exp(exponent) / 3.57f);
}

float scalar_receptor_current(float voltage_mv, float g_ampa, float g_nmda,
                              float g_gabaa, float additive_current,
                              float excitatory_reversal_mv,
                              float gabaa_reversal_mv) {
  return g_ampa * (excitatory_reversal_mv - voltage_mv) +
         g_nmda * scalar_nmda_block(voltage_mv) *
             (excitatory_reversal_mv - voltage_mv) +
         g_gabaa * (gabaa_reversal_mv - voltage_mv) +
         additive_current;
}

ReceptorKernel scalar_receptor_kernel(float rise_ms, float decay_ms,
                                      float dt_ms) {
  REQUIRE(rise_ms > 0.0f);
  REQUIRE(decay_ms > rise_ms);
  REQUIRE(dt_ms > 0.0f);
  const float peak_time =
      rise_ms * decay_ms * std::log(decay_ms / rise_ms) /
      (decay_ms - rise_ms);
  const float unnormalized =
      std::exp(-peak_time / decay_ms) -
      std::exp(-peak_time / rise_ms);
  ReceptorKernel kernel{};
  kernel.rise_ms = rise_ms;
  kernel.decay_ms = decay_ms;
  kernel.rise_decay = std::exp(-dt_ms / rise_ms);
  kernel.decay_decay = std::exp(-dt_ms / decay_ms);
  kernel.normalization = 1.0f / unnormalized;
  return kernel;
}

std::vector<float> scalar_receptor_trace(
    const ReceptorKernel& kernel, const std::vector<float>& arrivals) {
  float rise = 0.0f;
  float decay = 0.0f;
  std::vector<float> result(arrivals.size(), 0.0f);
  for (std::size_t index = 0; index < arrivals.size(); ++index) {
    REQUIRE(arrivals[index] >= 0.0f);
    rise = rise * kernel.rise_decay + arrivals[index];
    decay = decay * kernel.decay_decay + arrivals[index];
    result[index] =
        std::max(kernel.normalization * (decay - rise), 0.0f);
  }
  return result;
}

struct ScalarDerivatives {
  float voltage = 0.0f;
  float recovery = 0.0f;
};

ScalarDerivatives scalar_derivatives(const SolverInput& input,
                                     float voltage_mv, float recovery) {
  const float current =
      scalar_receptor_current(
          voltage_mv, input.g_ampa, input.g_nmda, input.g_gabaa,
          input.additive_current, input.excitatory_reversal_mv,
          input.gabaa_reversal_mv);
  return {
      0.04f * voltage_mv * voltage_mv + 5.0f * voltage_mv + 140.0f -
          recovery + current,
      input.parameters.a *
          (input.parameters.b * voltage_mv - recovery),
  };
}

std::pair<float, float> scalar_safe_heun(const SolverInput& input,
                                         float voltage_mv, float recovery,
                                         float interval_ms) {
  const ScalarDerivatives first =
      scalar_derivatives(input, voltage_mv, recovery);
  const float predicted_voltage =
      voltage_mv + interval_ms * first.voltage;
  const float predicted_recovery =
      recovery + interval_ms * first.recovery;
  const float bounded_predictor =
      std::min(predicted_voltage, input.threshold_mv);
  const ScalarDerivatives second = scalar_derivatives(
      input, bounded_predictor, predicted_recovery);
  return {
      voltage_mv +
          0.5f * interval_ms * (first.voltage + second.voltage),
      recovery +
          0.5f * interval_ms * (first.recovery + second.recovery),
  };
}

bool scalar_crossing_occurs(const SolverInput& input, float voltage_mv,
                            float recovery, float interval_ms) {
  const ScalarDerivatives first =
      scalar_derivatives(input, voltage_mv, recovery);
  const float predicted_voltage =
      voltage_mv + interval_ms * first.voltage;
  const float predicted_recovery =
      recovery + interval_ms * first.recovery;
  const float bounded_predictor =
      std::min(predicted_voltage, input.threshold_mv);
  const ScalarDerivatives second = scalar_derivatives(
      input, bounded_predictor, predicted_recovery);
  const float end_voltage =
      voltage_mv +
      0.5f * interval_ms * (first.voltage + second.voltage);
  return predicted_voltage >= input.threshold_mv ||
         end_voltage >= input.threshold_mv;
}

SolverOutput scalar_solver(const SolverInput& input) {
  REQUIRE(input.dt_ms > 0.0f);
  REQUIRE(input.parameters.c_mv < input.threshold_mv);
  REQUIRE(input.g_ampa >= 0.0f);
  REQUIRE(input.g_nmda >= 0.0f);
  REQUIRE(input.g_gabaa >= 0.0f);

  float voltage = input.state.voltage_mv;
  float recovery = input.state.recovery;
  float reported_voltage = voltage;
  float reported_recovery = recovery;
  int spike_count = 0;
  bool emitted = false;

  const float maximum_conductance =
      input.g_ampa + input.g_nmda + input.g_gabaa;
  const float stable_interval_ms =
      0.5f / std::max(1.0f, maximum_conductance);
  const int segment_count =
      std::max(1, static_cast<int>(
                      std::ceil(input.dt_ms / stable_interval_ms)));
  const float segment_dt_ms =
      input.dt_ms / static_cast<float>(segment_count);
  const float time_tolerance =
      std::max(segment_dt_ms * 1.0e-7f, 1.0e-8f);

  for (int segment = 0; segment < segment_count; ++segment) {
    float remaining = segment_dt_ms;
    bool segment_emitted = false;
    float segment_pre_voltage = voltage;
    float segment_pre_recovery = recovery;
    int event_iteration = 0;
    while (remaining > time_tolerance && event_iteration < 65) {
      ++event_iteration;
      const bool immediate = voltage >= input.threshold_mv;
      const bool crosses =
          immediate ||
          scalar_crossing_occurs(
              input, voltage, recovery, remaining);
      if (!crosses) {
        const auto advanced =
            scalar_safe_heun(input, voltage, recovery, remaining);
        voltage = advanced.first;
        recovery = advanced.second;
        if (!segment_emitted) {
          segment_pre_voltage = voltage;
          segment_pre_recovery = recovery;
        }
        remaining = 0.0f;
        continue;
      }

      float crossing_time = 0.0f;
      float crossing_recovery = recovery;
      if (!immediate) {
        float lower = 0.0f;
        float upper = remaining;
        for (int iteration = 0; iteration < 32; ++iteration) {
          const float midpoint = 0.5f * (lower + upper);
          if (scalar_crossing_occurs(
                  input, voltage, recovery, midpoint)) {
            upper = midpoint;
          } else {
            lower = midpoint;
          }
        }
        crossing_time = upper;
        crossing_recovery =
            scalar_safe_heun(input, voltage, recovery, crossing_time)
                .second;
      }

      segment_emitted = true;
      emitted = true;
      ++spike_count;
      segment_pre_voltage = input.threshold_mv;
      segment_pre_recovery = crossing_recovery;
      voltage = input.parameters.c_mv;
      recovery = crossing_recovery + input.parameters.d;
      remaining = std::max(remaining - crossing_time, 0.0f);
    }
    if (remaining > time_tolerance || spike_count > 64) {
      SolverOutput overflow{};
      overflow.state = {voltage, recovery};
      overflow.pre_reset_voltage_mv = segment_pre_voltage;
      overflow.pre_reset_recovery = segment_pre_recovery;
      overflow.spike_count = spike_count;
      overflow.emitted = emitted;
      overflow.overflow = true;
      return overflow;
    }
    if (segment_emitted) {
      reported_voltage = segment_pre_voltage;
      reported_recovery = segment_pre_recovery;
    }
  }

  if (!emitted) {
    reported_voltage = voltage;
    reported_recovery = recovery;
  }
  SolverOutput result{};
  result.state = {voltage, recovery};
  result.pre_reset_voltage_mv = reported_voltage;
  result.pre_reset_recovery = reported_recovery;
  result.spike_count = spike_count;
  result.emitted = emitted;
  result.overflow = false;
  return result;
}

std::vector<float> scalar_delay_impulse(int delay_steps, int duration_steps) {
  REQUIRE(delay_steps > 0);
  REQUIRE(delay_steps < clean_msi::kDelayRing);
  REQUIRE(duration_steps > delay_steps);
  std::array<float, clean_msi::kDelayRing> ring{};
  std::vector<float> arrivals(
      static_cast<std::size_t>(duration_steps), 0.0f);
  for (int step = 0; step < duration_steps; ++step) {
    const int read_slot = step % clean_msi::kDelayRing;
    arrivals[static_cast<std::size_t>(step)] = ring[read_slot];
    ring[read_slot] = 0.0f;
    if (step == 0) {
      const int arrival_slot =
          (step + delay_steps) % clean_msi::kDelayRing;
      ring[arrival_slot] += 1.0f;
    }
  }
  return arrivals;
}

float lower_median(std::vector<float> values) {
  REQUIRE(!values.empty());
  std::sort(values.begin(), values.end());
  return values[(values.size() - 1u) / 2u];
}

float scalar_oja_delta(float weight, float pre_trace, float post_trace,
                       float eta, float dt_ms) {
  return eta * dt_ms * post_trace *
         (pre_trace - post_trace * weight);
}

// ---------------------------------------------------------------------------
// Fast core tests.

void test_exact_scientific_contract() {
  static_assert(clean_msi::kAuditoryNeurons == 180);
  static_assert(clean_msi::kVisualNeurons == 180);
  static_assert(clean_msi::kExcitatoryNeurons == 180);
  static_assert(clean_msi::kInhibitoryNeurons == 60);
  static_assert(clean_msi::kPopulationCount == 4);
  static_assert(clean_msi::kDelayRing == 4);

  static_assert(clean_msi::kCalibrationCoarseNodes == 37);
  static_assert(clean_msi::kCalibrationFineNodes == 145);
  static_assert(clean_msi::kCalibrationRefinementNodes == 15);
  static_assert(clean_msi::kLearningRateCandidates == 81);

  static_assert(clean_msi::kExternalTrials == 256);
  static_assert(clean_msi::kExternalDurationMs == 150);
  static_assert(clean_msi::kExternalDriveMs == 50);
  static_assert(clean_msi::kFeedforwardLargeContacts == 12);
  static_assert(clean_msi::kFeedforwardSmallContacts == 4);
  static_assert(clean_msi::kFeedforwardDurationMs == 20);
  static_assert(clean_msi::kGabaaInhibitoryInputs == 10);
  static_assert(clean_msi::kGabaaDurationMs == 60);
  static_assert(
      clean_msi::kGabaaMedianInitialContactWeight == 0.035f);
  static_assert(clean_msi::kGabaaUnitaryDelayMs == 1);
  static_assert(clean_msi::kGabaaIpspSettleMs == 100);
  static_assert(clean_msi::kGabaaIpspWindowMs == 100);
  static_assert(clean_msi::kGabaaUnitaryTargetMv == 1.00f);
  static_assert(clean_msi::kGabaaUnitaryMinimumMv == 0.95f);
  static_assert(clean_msi::kGabaaUnitaryMaximumMv == 1.05f);
  static_assert(clean_msi::kBackgroundTrials == 128);
  static_assert(clean_msi::kBackgroundDurationMs == 5000);
  static_assert(clean_msi::kBackgroundDiscardMs == 1000);
  static_assert(clean_msi::kClopathPairings == 400);
  static_assert(
      clean_msi::kClopathHomeostasisTauMs == 100000.0f);
  static_assert(
      clean_msi::kClopathHomeostasisReferenceMv2 ==
      16.3214752406f);
  static_assert(clean_msi::kOjaEvents == 1000);
  static_assert(clean_msi::kIstdpSteps == 60000);
  static_assert(clean_msi::kIstdpMeasurementStart == 40000);
  static_assert(clean_msi::kIstdpInhibitoryInputs == 10);
  static_assert(clean_msi::kMsiEMarkedPhenotypesPerSeed == 9);
  static_assert(
      clean_msi::kMsiERegularRecoveryIncrement == 0.10f);
  static_assert(
      clean_msi::kMsiEMarkedRecoveryIncrement == 1.0f);
  static_assert(clean_msi::kMsiEPhenotypeAuditSeeds == 5);
  static_assert(clean_msi::kMsiEPhenotypeSettleMs == 100);
  static_assert(clean_msi::kMsiEPhenotypeDriveMs == 400);
  static_assert(clean_msi::kMsiEPhenotypeDriveCurrent == 10.0f);

  const Config config{};
  REQUIRE(config.dt_ms == 1.0f);
  REQUIRE(config.threshold_mv == 30.0f);
  REQUIRE(config.excitatory_reversal_mv == 0.0f);
  REQUIRE(config.gabaa_reversal_mv == -75.0f);
  REQUIRE(config.ampa_rise_ms == 0.5f);
  REQUIRE(config.ampa_decay_ms == 5.0f);
  REQUIRE(config.nmda_rise_ms == 2.0f);
  REQUIRE(config.nmda_decay_ms == 80.0f);
  REQUIRE(config.gabaa_rise_ms == 1.0f);
  REQUIRE(config.gabaa_decay_ms == 15.0f);
  REQUIRE(config.eta_istdp == 1.0e-4f);
}

void test_philox_determinism_and_known_answer() {
  const std::array<std::uint32_t, 4> zero_counter{};
  const std::array<std::uint32_t, 2> zero_key{};
  const std::array<std::uint32_t, 4> known_answer{
      0x6627E8D5u, 0xE169C58Du, 0xBC57AC4Cu, 0x9B00DBD8u};
  const auto scalar =
      scalar_philox4x32_10(zero_counter, zero_key);
  REQUIRE(scalar == known_answer);
  const auto production =
      clean_msi::philox4x32_10_host(zero_counter, zero_key);
  REQUIRE(production == known_answer);
  REQUIRE(clean_msi::philox4x32_10_host(zero_counter, zero_key) ==
          production);

  const std::array<std::uint32_t, 4> counter{
      0x01234567u, 0x89ABCDEFu, 0xDEADBEEFu, 0x10203040u};
  const std::array<std::uint32_t, 2> key{
      0xA5A5A5A5u, 0x5A5A5A5Au};
  REQUIRE(clean_msi::philox4x32_10_host(counter, key) ==
          scalar_philox4x32_10(counter, key));
  auto changed_counter = counter;
  ++changed_counter[0];
  REQUIRE(clean_msi::philox4x32_10_host(changed_counter, key) !=
          clean_msi::philox4x32_10_host(counter, key));
}

void test_voltage_coupled_current_equations() {
  const float gaba_below =
      scalar_receptor_current(-80.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                              0.0f, -75.0f);
  const float gaba_at =
      scalar_receptor_current(-75.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                              0.0f, -75.0f);
  const float gaba_above =
      scalar_receptor_current(-65.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                              0.0f, -75.0f);
  REQUIRE(gaba_below > 0.0f);
  REQUIRE(gaba_at == 0.0f);
  REQUIRE(gaba_above < 0.0f);

  const std::array<float, 4> voltages{-90.0f, -65.0f, -30.0f, 0.0f};
  float previous = -1.0f;
  for (const float voltage : voltages) {
    const float expected = scalar_nmda_block(voltage);
    const float actual = clean_msi::nmda_voltage_block_host(voltage);
    require_near(actual, expected, 2.0e-7f, 2.0e-7f,
                 "NMDA voltage block");
    REQUIRE(actual > previous);
    previous = actual;
  }

  const float q_ampa = 0.37f;
  const float q_nmda = clean_msi::nmda_quantum_from_ampa(q_ampa);
  const float ampa_current =
      scalar_receptor_current(-40.0f, q_ampa, 0.0f, 0.0f, 0.0f,
                              0.0f, -75.0f);
  const float nmda_current =
      scalar_receptor_current(-40.0f, 0.0f, q_nmda, 0.0f, 0.0f,
                              0.0f, -75.0f);
  require_near(ampa_current, 2.0f * nmda_current, 3.0e-5f, 3.0e-6f,
               "AMPA:NMDA current ratio at -40 mV");

  const float hyperpolarized_nmda =
      scalar_receptor_current(-80.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                              0.0f, -75.0f);
  const float depolarized_nmda =
      scalar_receptor_current(-20.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                              0.0f, -75.0f);
  REQUIRE(depolarized_nmda > hyperpolarized_nmda);
}

void test_biexponential_scalar_equations() {
  const std::array<std::pair<float, float>, 3> time_constants{{
      {0.5f, 5.0f},
      {2.0f, 80.0f},
      {1.0f, 15.0f},
  }};
  for (const auto [rise, decay] : time_constants) {
    const ReceptorKernel expected =
        scalar_receptor_kernel(rise, decay, 1.0f);
    const ReceptorKernel production =
        clean_msi::make_receptor_kernel(rise, decay, 1.0f);
    require_near(production.rise_decay, expected.rise_decay, 2.0e-7f,
                 2.0e-7f, "receptor rise recurrence");
    require_near(production.decay_decay, expected.decay_decay, 2.0e-7f,
                 2.0e-7f, "receptor decay recurrence");
    require_near(production.normalization, expected.normalization,
                 2.0e-6f, 2.0e-6f, "receptor peak normalization");
  }

  std::vector<float> impulse(128, 0.0f);
  impulse[0] = 1.0f;
  const auto ampa = scalar_receptor_trace(
      scalar_receptor_kernel(0.5f, 5.0f, 1.0f), impulse);
  const auto nmda = scalar_receptor_trace(
      scalar_receptor_kernel(2.0f, 80.0f, 1.0f), impulse);
  const auto gabaa = scalar_receptor_trace(
      scalar_receptor_kernel(1.0f, 15.0f, 1.0f), impulse);
  REQUIRE(*std::max_element(ampa.begin(), ampa.end()) > 0.90f);
  REQUIRE(*std::max_element(nmda.begin(), nmda.end()) > 0.95f);
  REQUIRE(*std::max_element(gabaa.begin(), gabaa.end()) > 0.95f);
  REQUIRE(nmda[19] > ampa[19]);
  REQUIRE(ampa.back() < 1.0e-8f);
  REQUIRE(nmda.back() > ampa.back());
}

void test_delay_impulses() {
  for (const int delay : {1, 2, 3}) {
    const auto arrivals = scalar_delay_impulse(delay, 9);
    for (int step = 0; step < 9; ++step) {
      const float expected = step == delay ? 1.0f : 0.0f;
      REQUIRE(arrivals[static_cast<std::size_t>(step)] == expected);
    }
  }
  const auto external = scalar_delay_impulse(1, 6);
  const auto recurrent = scalar_delay_impulse(2, 6);
  const auto feedforward = scalar_delay_impulse(3, 6);
  REQUIRE(external[1] == 1.0f);
  REQUIRE(recurrent[2] == 1.0f);
  REQUIRE(feedforward[3] == 1.0f);

  const auto path6_receptor_delivery =
      scalar_delay_impulse(1, 4);
  const auto path6_plastic_pre_event =
      scalar_delay_impulse(1, 4);
  REQUIRE(path6_receptor_delivery ==
          path6_plastic_pre_event);
  REQUIRE(path6_plastic_pre_event[0] == 0.0f);
  REQUIRE(path6_plastic_pre_event[1] == 1.0f);
  const auto delayed_weight_trajectory =
      [&path6_receptor_delivery,
       &path6_plastic_pre_event]() {
        constexpr float kEta = 0.10f;
        const float decay =
            std::exp(
                -1.0f / clean_msi::kIstdpTraceTauMs);
        float pre_trace = 0.0f;
        float post_trace = 0.0f;
        float weight = 0.50f;
        std::vector<float> trajectory;
        trajectory.reserve(path6_plastic_pre_event.size());
        for (std::size_t step = 0;
             step < path6_plastic_pre_event.size(); ++step) {
          const bool receptor_event =
              path6_receptor_delivery[step] != 0.0f;
          const bool pre_event =
              path6_plastic_pre_event[step] != 0.0f;
          REQUIRE(pre_event == receptor_event);
          pre_trace =
              pre_trace * decay +
              (pre_event ? 1.0f : 0.0f);
          post_trace *= decay;
          weight =
              clean_msi::ordered_vogels_istdp_update(
                  weight, pre_event, false,
                  pre_trace, post_trace,
                  kEta, clean_msi::kIstdpAlpha,
                  0.0f, 1.0f);
          trajectory.push_back(weight);
        }
        return trajectory;
      };
  const std::vector<float> first =
      delayed_weight_trajectory();
  const std::vector<float> replay =
      delayed_weight_trajectory();
  REQUIRE(first == replay);
  REQUIRE(first[0] == 0.50f);
  require_near(
      first[1] - first[0],
      -clean_msi::kIstdpAlpha * 0.10f,
      1.0e-7f, 1.0e-7f,
      "path6 delayed pre-event plasticity");
}

void test_lower_median() {
  REQUIRE(lower_median({7.0f}) == 7.0f);
  REQUIRE(lower_median({3.0f, 1.0f, 2.0f}) == 2.0f);
  REQUIRE(lower_median({4.0f, 1.0f, 3.0f, 2.0f}) == 2.0f);
  std::vector<float> values(256, 0.0f);
  for (int index = 0; index < 256; ++index) {
    values[static_cast<std::size_t>(index)] =
        static_cast<float>(255 - index);
  }
  REQUIRE(lower_median(values) == 127.0f);
  values.assign(128, 0.0f);
  for (int index = 0; index < 128; ++index) {
    values[static_cast<std::size_t>(index)] =
        static_cast<float>(index);
  }
  REQUIRE(lower_median(values) == 63.0f);
}

void test_local_plasticity_equations_and_bounds() {
  constexpr float eta = 0.01f;
  constexpr float homeostasis_reference =
      clean_msi::kClopathHomeostasisReferenceMv2;
  REQUIRE(
      clean_msi::clopath_homeostasis_multiplier(
          homeostasis_reference) == 1.0f);
  REQUIRE(
      clean_msi::clopath_homeostasis_multiplier(
          2.0f * homeostasis_reference) == 2.0f);
  REQUIRE(
      clean_msi::clopath_homeostasis_multiplier(0.0f) == 0.0f);
  for (const float b : {0.17f, 0.20f, 0.25f}) {
    const float stable_root =
        clean_msi::izhikevich_stable_rest_voltage_mv(b);
    REQUIRE(std::isfinite(stable_root));
    const float linear_coefficient = 5.0f - b;
    const float discriminant =
        linear_coefficient * linear_coefficient -
        4.0f * 0.04f * 140.0f;
    const float expected_lower_root =
        (-linear_coefficient - std::sqrt(discriminant)) / 0.08f;
    REQUIRE(stable_root == expected_lower_root);
    require_near(
        0.04f * stable_root * stable_root +
            linear_coefficient * stable_root + 140.0f,
        0.0f, 2.0e-4f, 0.0f,
        "stable Izhikevich rest equilibrium");
    REQUIRE(0.08f * stable_root + linear_coefficient < 0.0f);
  }
  require_near(
      clean_msi::izhikevich_stable_rest_voltage_mv(0.20f),
      -70.0f, 2.0e-5f, 0.0f,
      "regular-spiking stable rest root");
  REQUIRE(clean_msi::kClopathThetaPlusMv == -45.0f);

  const float calibration_theta_minus =
      clean_msi::izhikevich_stable_rest_voltage_mv(0.20f);
  const float production_e_theta_minus =
      clean_msi::izhikevich_stable_rest_voltage_mv(0.20f);
  const float production_i_theta_minus =
      clean_msi::izhikevich_stable_rest_voltage_mv(0.25f);
  REQUIRE(calibration_theta_minus == production_e_theta_minus);
  REQUIRE(production_i_theta_minus > production_e_theta_minus);
  const float calibration_semantics =
      clean_msi::clopath_pair_delta(
          eta, std::exp(-10.0f / 15.0f),
          -20.0f, -40.0f, -40.0f,
          calibration_theta_minus,
          clean_msi::kClopathThetaPlusMv,
          homeostasis_reference, false);
  const float production_e_semantics =
      clean_msi::clopath_pair_delta(
          eta, std::exp(-10.0f / 15.0f),
          -20.0f, -40.0f, -40.0f,
          production_e_theta_minus,
          clean_msi::kClopathThetaPlusMv,
          homeostasis_reference, false);
  REQUIRE(calibration_semantics == production_e_semantics);
  const float production_i_semantics =
      clean_msi::clopath_pair_delta(
          eta, 0.0f, -65.0f, -65.0f, -65.0f,
          production_i_theta_minus,
          clean_msi::kClopathThetaPlusMv,
          homeostasis_reference, true);
  REQUIRE(production_i_semantics == 0.0f);

  const float causal =
      clean_msi::clopath_pair_delta(
          eta, 0.8f, -35.0f, -40.0f, -50.0f,
          production_e_theta_minus,
          clean_msi::kClopathThetaPlusMv,
          homeostasis_reference, false);
  const float rest_threshold_ltd =
      clean_msi::clopath_pair_delta(
          eta, 0.0f, -65.0f, -65.0f, -65.0f,
          production_e_theta_minus,
          clean_msi::kClopathThetaPlusMv,
          homeostasis_reference, true);
  const float legacy_minus_threshold_ltd =
      clean_msi::clopath_pair_delta(
          eta, 0.0f, -65.0f, -65.0f, -65.0f,
          -60.0f, clean_msi::kClopathThetaPlusMv,
          homeostasis_reference, true);
  REQUIRE(causal > 0.0f);
  REQUIRE(rest_threshold_ltd < 0.0f);
  REQUIRE(legacy_minus_threshold_ltd == 0.0f);
  const float expected_minus_factor =
      std::max(
          (-65.0f - production_e_theta_minus) / 20.0f,
          0.0f);
  require_near(
      rest_threshold_ltd,
      -1.3125f * eta * expected_minus_factor,
      1.0e-8f, 1.0e-6f,
      "Clopath stable-rest theta-minus LTD gate");

  const float uncapped_instant_plus =
      clean_msi::clopath_pair_delta(
          1.0f, 1.0f, -5.0f, -70.0f, -50.0f,
          -70.0f, clean_msi::kClopathThetaPlusMv,
          homeostasis_reference, false);
  const float uncapped_filtered_plus =
      clean_msi::clopath_pair_delta(
          1.0f, 1.0f, -25.0f, -70.0f, -30.0f,
          -70.0f, clean_msi::kClopathThetaPlusMv,
          homeostasis_reference, false);
  const float uncapped_minus =
      clean_msi::clopath_pair_delta(
          1.0f, 0.0f, -65.0f, -30.0f, -65.0f,
          -70.0f, clean_msi::kClopathThetaPlusMv,
          homeostasis_reference, true);
  REQUIRE(uncapped_instant_plus == 2.0f);
  REQUIRE(uncapped_filtered_plus == 2.0f);
  REQUIRE(uncapped_minus == -2.625f);

  require_near(
      clean_msi::clopath_pair_delta(
          2.0f * eta, 0.8f, -35.0f, -40.0f, -50.0f,
          production_e_theta_minus,
          clean_msi::kClopathThetaPlusMv,
          homeostasis_reference, false),
      2.0f * causal, 1.0e-8f, 1.0e-6f, "Clopath eta linearity");
  const float low_additive =
      clean_msi::additive_hard_bound_update(
          0.20f, 0.10f, 0.0f, 1.0f);
  const float high_additive =
      clean_msi::additive_hard_bound_update(
          0.70f, 0.10f, 0.0f, 1.0f);
  require_near(
      low_additive - 0.20f, high_additive - 0.70f,
      1.0e-7f, 1.0e-6f,
      "Clopath additive update weight independence");
  REQUIRE(
      clean_msi::additive_hard_bound_update(
          0.99f, 0.10f, 0.0f, 1.0f) == 1.0f);
  REQUIRE(
      clean_msi::additive_hard_bound_update(
          0.01f, -0.10f, 0.0f, 1.0f) == 0.0f);

  const float oja_positive =
      scalar_oja_delta(0.25f, 0.8f, 0.4f, 0.10f, 1.0f);
  const float oja_equilibrium =
      scalar_oja_delta(2.0f, 0.8f, 0.4f, 0.10f, 1.0f);
  REQUIRE(oja_positive > 0.0f);
  require_near(oja_equilibrium, 0.0f, 1.0e-8f, 0.0f,
               "Oja equilibrium");
  const float oja_low_soft =
      clean_msi::multiplicative_soft_bound_update(
          0.20f, oja_positive, 0.0f, 1.0f);
  const float oja_high_soft =
      clean_msi::multiplicative_soft_bound_update(
          0.80f, oja_positive, 0.0f, 1.0f);
  REQUIRE(
      oja_low_soft - 0.20f >
      oja_high_soft - 0.80f);

  constexpr float kVogelsEta = 0.01f;
  const float lone_pre =
      clean_msi::ordered_vogels_istdp_update(
          0.50f, true, false, 1.0f, 0.0f,
          kVogelsEta, clean_msi::kIstdpAlpha,
          0.0f, 1.0f);
  require_near(
      lone_pre - 0.50f,
      -clean_msi::kIstdpAlpha * kVogelsEta,
      1.0e-8f, 1.0e-6f,
      "Vogels lone delayed pre event");
  const float lone_post =
      clean_msi::ordered_vogels_istdp_update(
          0.50f, false, true, 0.0f, 1.0f,
          kVogelsEta, clean_msi::kIstdpAlpha,
          0.0f, 1.0f);
  REQUIRE(lone_post == 0.50f);
  const float simultaneous =
      clean_msi::ordered_vogels_istdp_update(
          0.50f, true, true, 1.0f, 1.0f,
          kVogelsEta, clean_msi::kIstdpAlpha,
          0.0f, 1.0f);
  require_near(
      simultaneous - 0.50f,
      0.80f * kVogelsEta,
      1.0e-8f, 1.0e-6f,
      "Vogels simultaneous ordered events");
  const float sequential_restoration =
      clean_msi::ordered_vogels_istdp_update(
          0.10f, true, true, 1.0f, 1.0f,
          1.0f, clean_msi::kIstdpAlpha,
          0.0f, 1.0f);
  REQUIRE(sequential_restoration == 1.0f);
  REQUIRE(sequential_restoration != 0.90f);
  REQUIRE(
      clean_msi::ordered_vogels_istdp_update(
          0.10f, true, true, 1.0f, 1.0f,
          1.0f, clean_msi::kIstdpAlpha,
          0.0f, 1.0f) ==
      sequential_restoration);
}

std::vector<SolverInput> solver_reference_cases() {
  std::vector<SolverInput> inputs(5);
  for (SolverInput& input : inputs) {
    input.state = {-65.0f, -13.0f};
    input.parameters = {0.0f, 0.20f, -65.0f, 1.0f};
    input.dt_ms = 1.0f;
  }
  inputs[0].additive_current = 0.0f;
  inputs[1].additive_current = 100.0f;
  inputs[2].additive_current = 200.0f;

  inputs[3].state = {29.0f, -13.0f};
  inputs[3].parameters = {0.02f, 0.20f, -62.0f, 6.0f};
  inputs[3].additive_current = 0.0f;

  inputs[4].state = {-71.3912735f, -12.6741419f};
  inputs[4].parameters =
      {0.02f, 0.20f, -63.2751045f, 7.3100414f};
  inputs[4].g_ampa = 98.0901871f;
  inputs[4].g_nmda = 4.30702925f;
  return inputs;
}

void require_solver_output_near(const SolverOutput& actual,
                                const SolverOutput& expected,
                                const std::string& label) {
  require_near(actual.state.voltage_mv, expected.state.voltage_mv,
               3.0e-5f, 3.0e-5f, label + ".voltage");
  require_near(actual.state.recovery, expected.state.recovery,
               3.0e-5f, 3.0e-5f, label + ".recovery");
  require_near(actual.pre_reset_voltage_mv,
               expected.pre_reset_voltage_mv, 3.0e-5f, 3.0e-5f,
               label + ".pre_reset_voltage");
  require_near(actual.pre_reset_recovery,
               expected.pre_reset_recovery, 3.0e-5f, 3.0e-5f,
               label + ".pre_reset_recovery");
  REQUIRE_MESSAGE(actual.spike_count == expected.spike_count,
                  label + ".spike_count differs");
  REQUIRE_MESSAGE(actual.emitted == expected.emitted,
                  label + ".emitted differs");
  REQUIRE_MESSAGE(actual.overflow == expected.overflow,
                  label + ".overflow differs");
}

void require_solver_output_exact(const SolverOutput& actual,
                                 const SolverOutput& expected,
                                 const std::string& label) {
  REQUIRE_MESSAGE(actual.state.voltage_mv == expected.state.voltage_mv,
                  label + ".voltage differs");
  REQUIRE_MESSAGE(actual.state.recovery == expected.state.recovery,
                  label + ".recovery differs");
  REQUIRE_MESSAGE(
      actual.pre_reset_voltage_mv == expected.pre_reset_voltage_mv,
      label + ".pre_reset_voltage differs");
  REQUIRE_MESSAGE(
      actual.pre_reset_recovery == expected.pre_reset_recovery,
      label + ".pre_reset_recovery differs");
  REQUIRE_MESSAGE(actual.spike_count == expected.spike_count,
                  label + ".spike_count differs");
  REQUIRE_MESSAGE(actual.emitted == expected.emitted,
                  label + ".emitted differs");
  REQUIRE_MESSAGE(actual.overflow == expected.overflow,
                  label + ".overflow differs");
}

void test_solver_scalar_cases() {
  const auto inputs = solver_reference_cases();
  std::vector<SolverOutput> outputs;
  outputs.reserve(inputs.size());
  for (const SolverInput& input : inputs) {
    outputs.push_back(scalar_solver(input));
  }
  REQUIRE(outputs[0].spike_count == 0);
  REQUIRE(outputs[1].spike_count == 1);
  REQUIRE(outputs[2].spike_count == 2);
  REQUIRE(!outputs[0].emitted);
  REQUIRE(outputs[1].emitted);
  REQUIRE(outputs[2].emitted);

  const SolverOutput& remainder = outputs[3];
  REQUIRE(remainder.spike_count == 1);
  REQUIRE(remainder.pre_reset_voltage_mv == 30.0f);
  REQUIRE(remainder.state.voltage_mv < inputs[3].parameters.c_mv - 1.0f);
  REQUIRE(remainder.state.recovery <
          remainder.pre_reset_recovery + inputs[3].parameters.d - 0.05f);

  const SolverOutput& stiff = outputs[4];
  REQUIRE(stiff.spike_count == 0);
  REQUIRE(!stiff.overflow);
  require_finite(stiff.state.voltage_mv,
                 "stiff voltage-coupled scalar voltage");
  require_finite(stiff.state.recovery,
                 "stiff voltage-coupled scalar recovery");
  REQUIRE(std::fabs(stiff.state.voltage_mv - 1.58f) <= 3.0f);
}

void test_solver_device(int device) {
  const auto inputs = solver_reference_cases();
  std::vector<SolverOutput> expected;
  expected.reserve(inputs.size());
  for (const SolverInput& input : inputs) {
    expected.push_back(scalar_solver(input));
  }
  const auto actual = clean_msi::run_solver_batch(device, inputs);
  REQUIRE(actual.size() == expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    require_solver_output_near(
        actual[index], expected[index],
        "cuda" + std::to_string(device) +
            " solver case " + std::to_string(index));
  }
  const auto replay = clean_msi::run_solver_batch(device, inputs);
  REQUIRE(replay.size() == actual.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    require_solver_output_exact(
        replay[index], actual[index],
        "cuda" + std::to_string(device) +
            " solver deterministic replay " + std::to_string(index));
  }
}

void test_receptor_device(int device) {
  std::vector<float> arrivals(192, 0.0f);
  arrivals[0] = 1.0f;
  arrivals[7] = 0.5f;
  arrivals[31] = 2.0f;
  const std::array<std::pair<float, float>, 3> time_constants{{
      {0.5f, 5.0f},
      {2.0f, 80.0f},
      {1.0f, 15.0f},
  }};
  for (const auto [rise, decay] : time_constants) {
    const ReceptorKernel kernel =
        scalar_receptor_kernel(rise, decay, 1.0f);
    const auto expected = scalar_receptor_trace(kernel, arrivals);
    const auto actual =
        clean_msi::run_receptor_trace(device, kernel, arrivals);
    REQUIRE(actual.size() == expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
      REQUIRE(actual[index] >= 0.0f);
      require_near(
          actual[index], expected[index], 2.0e-6f, 2.0e-6f,
          "cuda" + std::to_string(device) + " receptor trace");
    }
    const auto replay =
        clean_msi::run_receptor_trace(device, kernel, arrivals);
    REQUIRE(replay == actual);
  }
}

void test_cuda_core_parity() {
  const std::vector<int> devices = available_devices();
  REQUIRE_MESSAGE(!devices.empty(),
                  "CUDA core parity requires at least one CUDA device");
  for (const int device : devices) {
    const auto info = clean_msi::query_device(device);
    REQUIRE(info.ordinal == device);
    REQUIRE(info.major > 0);
    REQUIRE(info.multiprocessors > 0);
    REQUIRE(!info.name.empty());
    test_solver_device(device);
    test_receptor_device(device);
  }
}

// ---------------------------------------------------------------------------
// Calibration parity and performance tests. These are separate command-line
// entry points because they execute all accepted biological assay sizes.

std::vector<float> representative_candidates(AssayKind kind) {
  switch (kind) {
    case AssayKind::kExternalRs:
      return {0.10f, 0.30f, 0.60f};
    case AssayKind::kFeedforwardE:
      return {0.01f, 0.04f, 0.08f};
    case AssayKind::kFeedforwardI:
      return {0.003f, 0.015f, 0.05f};
    case AssayKind::kGabaa:
      return {0.00349115161f, 4.0f, 10.0f};
    case AssayKind::kBackgroundRs:
    case AssayKind::kBackgroundFs:
      return {1.0f, 3.0f, 10.0f};
  }
  throw std::logic_error("unreachable assay kind");
}

const char* assay_name(AssayKind kind) {
  switch (kind) {
    case AssayKind::kExternalRs:
      return "external_rs";
    case AssayKind::kFeedforwardE:
      return "feedforward_e";
    case AssayKind::kFeedforwardI:
      return "feedforward_i";
    case AssayKind::kGabaa:
      return "gabaa";
    case AssayKind::kBackgroundRs:
      return "background_rs";
    case AssayKind::kBackgroundFs:
      return "background_fs";
  }
  return "unknown";
}

constexpr std::array<AssayKind, 6> kAllAssays{
    AssayKind::kExternalRs,
    AssayKind::kFeedforwardE,
    AssayKind::kFeedforwardI,
    AssayKind::kGabaa,
    AssayKind::kBackgroundRs,
    AssayKind::kBackgroundFs,
};

void test_candidate_independence_device(int device) {
  Config config{};
  config.seed = 7001;
  for (const AssayKind kind : kAllAssays) {
    const auto candidates = representative_candidates(kind);
    const auto fused = clean_msi::evaluate_assay_candidates(
        device, config, kind, candidates);
    REQUIRE(fused.size() == candidates.size());
    const auto replay = clean_msi::evaluate_assay_candidates(
        device, config, kind, candidates);
    REQUIRE(replay.size() == fused.size());
    for (std::size_t index = 0; index < fused.size(); ++index) {
      require_exact_metrics(
          replay[index], fused[index],
          std::string(assay_name(kind)) + " fixed-device replay");
      const auto singleton = clean_msi::evaluate_assay_candidates(
          device, config, kind, {candidates[index]});
      REQUIRE(singleton.size() == 1u);
      require_exact_metrics(
          singleton[0], fused[index],
          std::string(assay_name(kind)) + " fused/singleton");
    }

    const std::vector<float> permuted{
        candidates[2], candidates[0], candidates[1], candidates[0]};
    const auto permuted_metrics =
        clean_msi::evaluate_assay_candidates(
            device, config, kind, permuted);
    REQUIRE(permuted_metrics.size() == permuted.size());
    require_exact_metrics(permuted_metrics[0], fused[2],
                          std::string(assay_name(kind)) + " permutation 0");
    require_exact_metrics(permuted_metrics[1], fused[0],
                          std::string(assay_name(kind)) + " permutation 1");
    require_exact_metrics(permuted_metrics[2], fused[1],
                          std::string(assay_name(kind)) + " permutation 2");
    require_exact_metrics(permuted_metrics[3], fused[0],
                          std::string(assay_name(kind)) + " duplicate");
  }
}

void require_calibration_science(const CalibrationResult& result,
                                 const std::string& label) {
  REQUIRE_MESSAGE(result.config.calibrated,
                  label + " did not mark config calibrated");
  REQUIRE_MESSAGE(result.criteria_passed,
                  label + " failed scientific criteria");
  for (const auto [name, value] :
       std::array<std::pair<const char*, float>, 9>{{
           {"q_external_rs", result.config.q_external_rs},
           {"q_background_rs", result.config.q_background_rs},
           {"q_background_fs", result.config.q_background_fs},
           {"q_ff_e", result.config.q_ff_e},
           {"q_ff_i", result.config.q_ff_i},
           {"q_gabaa", result.config.q_gabaa},
           {"eta_clopath_ff", result.config.eta_clopath_ff},
           {"eta_oja", result.config.eta_oja},
           {"eta_istdp", result.config.eta_istdp},
       }}) {
    require_finite(value, label + "." + name);
    REQUIRE_MESSAGE(value > 0.0f, label + "." + name + " is not positive");
  }

  REQUIRE(result.external.primary == 3.0f);
  REQUIRE(result.external.secondary <= 2.0f);
  REQUIRE(result.external.tertiary <= 8.0f);
  REQUIRE(result.feedforward_e.primary == 1.0f);
  REQUIRE(result.feedforward_e.secondary == 0.0f);
  REQUIRE(result.feedforward_e.tertiary == 2.0f);
  REQUIRE(result.feedforward_i.primary == 1.0f);
  REQUIRE(result.feedforward_i.secondary == 0.0f);
  REQUIRE(
      result.gabaa.primary >=
      clean_msi::kGabaaUnitaryMinimumMv);
  REQUIRE(
      result.gabaa.primary <=
      clean_msi::kGabaaUnitaryMaximumMv);
  REQUIRE(result.gabaa.secondary < 0.0f);
  REQUIRE(result.gabaa.tertiary > 0.0f);
  REQUIRE(
      result.gabaa_efficacy.unitary.contact_count == 1);
  REQUIRE(
      result.gabaa_efficacy.unitary.amplitude_mv ==
      result.gabaa.primary);
  REQUIRE(
      result.gabaa_efficacy.unitary.signed_nadir_mv ==
      result.gabaa.secondary);
  REQUIRE(
      result.gabaa_efficacy.unitary.area_mv_ms ==
      result.gabaa.tertiary);
  REQUIRE(
      result.gabaa_efficacy.unitary.outward_charge > 0.0f);
  REQUIRE(
      result.gabaa_efficacy.compound.contact_count ==
      clean_msi::kInhibitoryScaffoldInDegree);
  REQUIRE(
      result.gabaa_efficacy.compound.amplitude_mv >= 3.0f);
  REQUIRE(
      result.gabaa_efficacy.compound.signed_nadir_mv < 0.0f);
  REQUIRE(
      result.gabaa_efficacy.compound.area_mv_ms >
      result.gabaa_efficacy.unitary.area_mv_ms);
  REQUIRE(
      result.gabaa_efficacy.compound.outward_charge >
      result.gabaa_efficacy.unitary.outward_charge);
  REQUIRE(result.gabaa_relay.relay_spikes > 0);
  REQUIRE(result.gabaa_relay.gabaa_arrivals > 0);
  REQUIRE(result.gabaa_relay.lag_count > 0);
  REQUIRE(
      result.gabaa_relay.without_gabaa_spikes >
      result.gabaa_relay.with_gabaa_spikes);
  REQUIRE(result.background_rs.primary == 20.0f);
  REQUIRE(result.background_fs.primary == 20.0f);

  REQUIRE(std::fabs(result.learning.clopath_delta - 0.25f) <= 0.01f);
  REQUIRE(result.learning.clopath_shuffled_drift <= 0.05f);
  REQUIRE(result.learning.oja_selection_ratio >= 2.0f);
  REQUIRE(result.learning.oja_bound_fraction < 0.05f);
  REQUIRE(result.config.eta_istdp == 1.0e-4f);
  require_finite(
      result.learning.istdp_weak_rate_hz,
      label + ".learning.istdp_weak_rate_hz");
  REQUIRE(result.learning.istdp_weak_rate_hz >= 0.0f);
  require_finite(
      result.learning.istdp_strong_rate_hz,
      label + ".learning.istdp_strong_rate_hz");
  REQUIRE(result.learning.istdp_strong_rate_hz >= 0.0f);
  require_finite(
      result.learning.istdp_weak_bound_fraction,
      label + ".learning.istdp_weak_bound_fraction");
  REQUIRE(result.learning.istdp_weak_bound_fraction >= 0.0f);
  REQUIRE(result.learning.istdp_weak_bound_fraction < 0.10f);
  require_finite(
      result.learning.istdp_strong_bound_fraction,
      label + ".learning.istdp_strong_bound_fraction");
  REQUIRE(result.learning.istdp_strong_bound_fraction >= 0.0f);
  REQUIRE(result.learning.istdp_strong_bound_fraction < 0.10f);
}

void require_returned_calibration_points_were_accepted(
    int device, const CalibrationResult& result,
    const std::string& label) {
  const std::array<std::pair<AssayKind, float>, 6> returned_points{{
      {AssayKind::kExternalRs, result.config.q_external_rs},
      {AssayKind::kFeedforwardE, result.config.q_ff_e},
      {AssayKind::kFeedforwardI, result.config.q_ff_i},
      {AssayKind::kGabaa, result.config.q_gabaa},
      {AssayKind::kBackgroundRs, result.config.q_background_rs},
      {AssayKind::kBackgroundFs, result.config.q_background_fs},
  }};
  const std::array<AssayMetrics, 6> recorded_metrics{{
      result.external,
      result.feedforward_e,
      result.feedforward_i,
      result.gabaa,
      result.background_rs,
      result.background_fs,
  }};
  for (std::size_t index = 0; index < returned_points.size(); ++index) {
    const AssayKind kind = returned_points[index].first;
    const float quantum = returned_points[index].second;
    const auto reevaluated = clean_msi::evaluate_assay_candidates(
        device, result.config, kind, {quantum});
    REQUIRE(reevaluated.size() == 1u);
    require_exact_metrics(
        reevaluated[0], recorded_metrics[index],
        label + "." + assay_name(kind) + ".returned_q_reevaluation");
  }
  // These exact equalities are the narrow-plateau regression: refinement may
  // not return a merely bracketing q whose response jumps past the target.
  REQUIRE(result.background_rs.primary == 20.0f);
  REQUIRE(result.background_fs.primary == 20.0f);
}

void require_calibration_replay_exact(
    const CalibrationResult& actual,
    const CalibrationResult& expected,
    const std::string& label) {
  REQUIRE_MESSAGE(actual.config.seed == expected.config.seed,
                  label + ".seed differs");
  REQUIRE_MESSAGE(actual.config.calibrated == expected.config.calibrated,
                  label + ".calibrated differs");
  for (const auto [field, values] :
       std::array<std::pair<const char*, std::pair<float, float>>, 9>{{
           {"q_external_rs",
            {actual.config.q_external_rs,
             expected.config.q_external_rs}},
           {"q_background_rs",
            {actual.config.q_background_rs,
             expected.config.q_background_rs}},
           {"q_background_fs",
            {actual.config.q_background_fs,
             expected.config.q_background_fs}},
           {"q_ff_e", {actual.config.q_ff_e, expected.config.q_ff_e}},
           {"q_ff_i", {actual.config.q_ff_i, expected.config.q_ff_i}},
           {"q_gabaa",
            {actual.config.q_gabaa, expected.config.q_gabaa}},
           {"eta_clopath_ff",
            {actual.config.eta_clopath_ff,
             expected.config.eta_clopath_ff}},
           {"eta_oja",
            {actual.config.eta_oja, expected.config.eta_oja}},
           {"eta_istdp",
            {actual.config.eta_istdp, expected.config.eta_istdp}},
       }}) {
    REQUIRE_MESSAGE(values.first == values.second,
                    label + "." + field + " differs");
  }
  require_exact_metrics(actual.external, expected.external,
                        label + ".external");
  require_exact_metrics(actual.feedforward_e, expected.feedforward_e,
                        label + ".feedforward_e");
  require_exact_metrics(actual.feedforward_i, expected.feedforward_i,
                        label + ".feedforward_i");
  require_exact_metrics(actual.gabaa, expected.gabaa, label + ".gabaa");
  const auto require_ipsp_exact =
      [&](const clean_msi::GabaaIpspMetrics& observed,
          const clean_msi::GabaaIpspMetrics& reference,
          const std::string& metric_label) {
        REQUIRE_MESSAGE(
            observed.contact_count == reference.contact_count,
            metric_label + ".contact_count differs");
        REQUIRE_MESSAGE(
            observed.amplitude_mv == reference.amplitude_mv,
            metric_label + ".amplitude_mv differs");
        REQUIRE_MESSAGE(
            observed.signed_nadir_mv == reference.signed_nadir_mv,
            metric_label + ".signed_nadir_mv differs");
        REQUIRE_MESSAGE(
            observed.area_mv_ms == reference.area_mv_ms,
            metric_label + ".area_mv_ms differs");
        REQUIRE_MESSAGE(
            observed.outward_charge == reference.outward_charge,
            metric_label + ".outward_charge differs");
        REQUIRE_MESSAGE(
            observed.control_spikes == reference.control_spikes,
            metric_label + ".control_spikes differs");
        REQUIRE_MESSAGE(
            observed.gabaa_spikes == reference.gabaa_spikes,
            metric_label + ".gabaa_spikes differs");
      };
  require_ipsp_exact(
      actual.gabaa_efficacy.unitary,
      expected.gabaa_efficacy.unitary,
      label + ".gabaa_efficacy.unitary");
  require_ipsp_exact(
      actual.gabaa_efficacy.compound,
      expected.gabaa_efficacy.compound,
      label + ".gabaa_efficacy.compound");
  REQUIRE_MESSAGE(
      actual.gabaa_relay.relay_neurons ==
          expected.gabaa_relay.relay_neurons,
      label + ".gabaa_relay.relay_neurons differs");
  REQUIRE_MESSAGE(
      actual.gabaa_relay.excitatory_packets ==
          expected.gabaa_relay.excitatory_packets,
      label + ".gabaa_relay.excitatory_packets differs");
  REQUIRE_MESSAGE(
      actual.gabaa_relay.relay_spikes ==
          expected.gabaa_relay.relay_spikes,
      label + ".gabaa_relay.relay_spikes differs");
  REQUIRE_MESSAGE(
      actual.gabaa_relay.gabaa_arrivals ==
          expected.gabaa_relay.gabaa_arrivals,
      label + ".gabaa_relay.gabaa_arrivals differs");
  REQUIRE_MESSAGE(
      actual.gabaa_relay.lag_count ==
          expected.gabaa_relay.lag_count,
      label + ".gabaa_relay.lag_count differs");
  REQUIRE_MESSAGE(
      actual.gabaa_relay.minimum_lag_ms ==
          expected.gabaa_relay.minimum_lag_ms,
      label + ".gabaa_relay.minimum_lag_ms differs");
  REQUIRE_MESSAGE(
      actual.gabaa_relay.maximum_lag_ms ==
          expected.gabaa_relay.maximum_lag_ms,
      label + ".gabaa_relay.maximum_lag_ms differs");
  REQUIRE_MESSAGE(
      actual.gabaa_relay.mean_lag_ms ==
          expected.gabaa_relay.mean_lag_ms,
      label + ".gabaa_relay.mean_lag_ms differs");
  REQUIRE_MESSAGE(
      actual.gabaa_relay.without_gabaa_spikes ==
          expected.gabaa_relay.without_gabaa_spikes,
      label + ".gabaa_relay.without_gabaa_spikes differs");
  REQUIRE_MESSAGE(
      actual.gabaa_relay.with_gabaa_spikes ==
          expected.gabaa_relay.with_gabaa_spikes,
      label + ".gabaa_relay.with_gabaa_spikes differs");
  for (int lag = 0; lag < clean_msi::kGabaaDurationMs; ++lag) {
    REQUIRE_MESSAGE(
        actual.gabaa_relay.direct_to_gabaa_lag_counts[lag] ==
            expected.gabaa_relay.direct_to_gabaa_lag_counts[lag],
        label + ".gabaa_relay.lag_counts differs");
  }
  require_exact_metrics(actual.background_rs, expected.background_rs,
                        label + ".background_rs");
  require_exact_metrics(actual.background_fs, expected.background_fs,
                        label + ".background_fs");
  require_learning_near(actual.learning, expected.learning, 0.0f,
                        label + ".learning");
  REQUIRE_MESSAGE(actual.criteria_passed == expected.criteria_passed,
                  label + ".criteria_passed differs");
}

struct TimedCalibration {
  CalibrationResult result{};
  double wall_seconds = 0.0;
};

std::map<int, TimedCalibration> calibration_cache;

const TimedCalibration& calibration_for(int device) {
  const auto existing = calibration_cache.find(device);
  if (existing != calibration_cache.end()) {
    return existing->second;
  }
  Config config{};
  config.seed = 0;
  TimedCalibration timed{};
  timed.wall_seconds = timed_seconds(device, [&] {
    timed.result = clean_msi::calibrate(device, config);
  });
  require_calibration_science(
      timed.result, "cuda" + std::to_string(device) + " calibration");
  require_returned_calibration_points_were_accepted(
      device, timed.result,
      "cuda" + std::to_string(device) + " calibration");
  const auto inserted =
      calibration_cache.emplace(device, std::move(timed));
  return inserted.first->second;
}

void require_calibration_cross_device_near(
    const CalibrationResult& actual,
    const CalibrationResult& expected,
    const std::string& label) {
  const auto compare_parameter =
      [&](float lhs, float rhs, const std::string& field) {
        require_near(lhs, rhs, 1.0e-5f, 1.0e-5f,
                     label + "." + field);
      };
  compare_parameter(actual.config.q_external_rs,
                    expected.config.q_external_rs, "q_external_rs");
  compare_parameter(actual.config.q_background_rs,
                    expected.config.q_background_rs, "q_background_rs");
  compare_parameter(actual.config.q_background_fs,
                    expected.config.q_background_fs, "q_background_fs");
  compare_parameter(actual.config.q_ff_e, expected.config.q_ff_e, "q_ff_e");
  compare_parameter(actual.config.q_ff_i, expected.config.q_ff_i, "q_ff_i");
  compare_parameter(actual.config.q_gabaa, expected.config.q_gabaa,
                    "q_gabaa");
  compare_parameter(actual.config.eta_clopath_ff,
                    expected.config.eta_clopath_ff, "eta_clopath_ff");
  compare_parameter(actual.config.eta_oja, expected.config.eta_oja,
                    "eta_oja");
  compare_parameter(actual.config.eta_istdp,
                    expected.config.eta_istdp, "eta_istdp");

  require_exact_metrics(actual.external, expected.external,
                        label + ".external");
  require_exact_metrics(actual.feedforward_e, expected.feedforward_e,
                        label + ".feedforward_e");
  require_exact_metrics(actual.feedforward_i, expected.feedforward_i,
                        label + ".feedforward_i");
  require_exact_metrics(actual.gabaa, expected.gabaa, label + ".gabaa");
  require_exact_metrics(actual.background_rs, expected.background_rs,
                        label + ".background_rs");
  require_exact_metrics(actual.background_fs, expected.background_fs,
                        label + ".background_fs");
  require_learning_near(actual.learning, expected.learning, 1.0e-5f,
                        label + ".learning");
}

void test_calibration_parity() {
  const std::vector<int> devices = available_devices();
  REQUIRE_MESSAGE(!devices.empty(),
                  "calibration parity requires at least one CUDA device");
  for (const int device : devices) {
    test_candidate_independence_device(device);

    const Config config{};
    const auto first_learning =
        clean_msi::evaluate_learning_candidates(
            device, config, 1.0e-3f, 1.0e-3f, 1.0e-2f);
    const auto replay_learning =
        clean_msi::evaluate_learning_candidates(
            device, config, 1.0e-3f, 1.0e-3f, 1.0e-2f);
    require_learning_near(
        replay_learning, first_learning, 0.0f,
        "cuda" + std::to_string(device) + " learning replay");
    const CalibrationResult& first = calibration_for(device).result;
    const auto selected_learning =
        clean_msi::evaluate_learning_candidates(
            device, first.config,
            first.config.eta_clopath_ff,
            first.config.eta_oja,
            first.config.eta_istdp);
    require_learning_near(
        selected_learning, first.learning, 0.0f,
        "cuda" + std::to_string(device) +
            " selected learning-point replay");
    CalibrationResult replay{};
    replay = clean_msi::calibrate(device, Config{});
    require_calibration_science(
        replay,
        "cuda" + std::to_string(device) + " replay calibration");
    require_returned_calibration_points_were_accepted(
        device, replay,
        "cuda" + std::to_string(device) + " replay calibration");
    require_calibration_replay_exact(
        replay, first,
        "cuda" + std::to_string(device) +
            " full-calibration deterministic replay");
  }
  if (devices.size() > 1u) {
    require_calibration_cross_device_near(
        calibration_for(devices[1]).result,
        calibration_for(devices[0]).result,
        "cuda1/cuda0 calibration parity");
  }
}

std::vector<float> logarithmic_candidates(float lower, float upper,
                                          int count) {
  REQUIRE(lower > 0.0f);
  REQUIRE(upper > lower);
  REQUIRE(count > 1);
  std::vector<float> candidates(static_cast<std::size_t>(count));
  const float log_lower = std::log(lower);
  const float log_upper = std::log(upper);
  for (int index = 0; index < count; ++index) {
    const float fraction =
        static_cast<float>(index) / static_cast<float>(count - 1);
    candidates[static_cast<std::size_t>(index)] =
        std::exp(log_lower + fraction * (log_upper - log_lower));
  }
  return candidates;
}

void benchmark_fused_candidates(int device, AssayKind kind,
                                const std::vector<float>& candidates) {
  Config config{};
  config.seed = 8128;
  clean_msi::evaluate_assay_candidates(
      device, config, kind, {candidates.front()});

  std::vector<AssayMetrics> fused;
  const double fused_seconds = timed_seconds(device, [&] {
    fused = clean_msi::evaluate_assay_candidates(
        device, config, kind, candidates);
  });
  std::vector<AssayMetrics> serialized;
  serialized.reserve(candidates.size());
  const double serialized_seconds = timed_seconds(device, [&] {
    for (const float candidate : candidates) {
      const auto one = clean_msi::evaluate_assay_candidates(
          device, config, kind, {candidate});
      REQUIRE(one.size() == 1u);
      serialized.push_back(one[0]);
    }
  });
  REQUIRE(fused.size() == serialized.size());
  for (std::size_t index = 0; index < fused.size(); ++index) {
    require_exact_metrics(serialized[index], fused[index],
                          std::string(assay_name(kind)) +
                              " benchmark parity");
  }
  const double speedup = serialized_seconds / fused_seconds;
  std::cout << "[PERF] cuda" << device << ' ' << assay_name(kind)
            << " candidates=" << candidates.size()
            << " fused_s=" << std::fixed << std::setprecision(6)
            << fused_seconds << " serialized_s=" << serialized_seconds
            << " speedup=" << std::setprecision(2) << speedup << "x\n";
  REQUIRE_MESSAGE(
      fused_seconds < serialized_seconds,
      std::string("fused ") + assay_name(kind) +
          " candidate evaluation was not faster than serialized launches");
}

void test_calibration_performance() {
  const std::vector<int> devices = available_devices();
  REQUIRE_MESSAGE(!devices.empty(),
                  "calibration performance requires at least one CUDA device");
  for (const int device : devices) {
    benchmark_fused_candidates(
        device, AssayKind::kExternalRs,
        logarithmic_candidates(0.05f, 1.0f,
                               clean_msi::kLearningRateCandidates));
    benchmark_fused_candidates(
        device, AssayKind::kBackgroundRs,
        logarithmic_candidates(0.5f, 15.0f,
                               clean_msi::kLearningRateCandidates));
    const TimedCalibration& timed = calibration_for(device);
    std::cout << "[PERF] cuda" << device
              << " full_calibration_wall_s=" << std::fixed
              << std::setprecision(6) << timed.wall_seconds
              << " internal_conductance_s="
              << timed.result.conductance_seconds
              << " internal_learning_s="
              << timed.result.learning_seconds << '\n';
  }
}

// ---------------------------------------------------------------------------
// Independent locked stimulus-environment specification. This reference is
// intentionally test-local until the native generator result contract lands.

enum class ReferenceConditionKind : int {
  kCommonAv = 0,
  kIndependentAv = 1,
  kAuditoryOnly = 2,
  kVisualOnly = 3,
};

struct ReferenceCondition {
  ReferenceConditionKind kind = ReferenceConditionKind::kCommonAv;
  bool auditory_present = false;
  bool visual_present = false;
  double auditory_latent_location_deg =
      std::numeric_limits<double>::quiet_NaN();
  double visual_latent_location_deg =
      std::numeric_limits<double>::quiet_NaN();
  double auditory_location_deg =
      std::numeric_limits<double>::quiet_NaN();
  double visual_location_deg =
      std::numeric_limits<double>::quiet_NaN();
  double auditory_noise_deg =
      std::numeric_limits<double>::quiet_NaN();
  double visual_noise_deg =
      std::numeric_limits<double>::quiet_NaN();
  double auditory_rate_hz =
      std::numeric_limits<double>::quiet_NaN();
  double visual_rate_hz =
      std::numeric_limits<double>::quiet_NaN();
  double physical_soa_ms = 0.0;
  double auditory_latency_ms =
      std::numeric_limits<double>::quiet_NaN();
  double visual_latency_ms =
      std::numeric_limits<double>::quiet_NaN();
};

struct ReferenceTiming {
  double auditory_onset_ms =
      std::numeric_limits<double>::quiet_NaN();
  double visual_onset_ms =
      std::numeric_limits<double>::quiet_NaN();
  double earlier_onset_ms = 100.0;
  double poststimulus_end_ms = 0.0;
  double first_zero_afferent_receptor_step = 0.0;
  double valid_end_ms = 0.0;
};

double reflect_reference_azimuth(double value_deg) {
  constexpr double low = -90.0;
  constexpr double high = 90.0;
  constexpr double width = high - low;
  double phase = std::fmod(value_deg - low, 2.0 * width);
  if (phase < 0.0) {
    phase += 2.0 * width;
  }
  return low + width - std::fabs(phase - width);
}

class ReferenceStimulusGenerator {
 public:
  explicit ReferenceStimulusGenerator(std::uint64_t seed)
      : engine_(seed) {}

  ReferenceCondition sample() {
    const double mixture = uniform(0.0, 1.0);
    ReferenceConditionKind kind = ReferenceConditionKind::kVisualOnly;
    if (mixture < 0.40) {
      kind = ReferenceConditionKind::kCommonAv;
    } else if (mixture < 0.60) {
      kind = ReferenceConditionKind::kIndependentAv;
    } else if (mixture < 0.80) {
      kind = ReferenceConditionKind::kAuditoryOnly;
    }

    ReferenceCondition condition{};
    condition.kind = kind;
    if (kind == ReferenceConditionKind::kCommonAv) {
      condition.auditory_present = true;
      condition.visual_present = true;
      const double latent = uniform(-60.0, 60.0);
      condition.auditory_latent_location_deg = latent;
      condition.visual_latent_location_deg = latent;
      condition.auditory_noise_deg = normal(0.0, 8.0);
      condition.visual_noise_deg = normal(0.0, 2.0);
      condition.auditory_location_deg = reflect_reference_azimuth(
          condition.auditory_latent_location_deg +
          condition.auditory_noise_deg);
      condition.visual_location_deg = reflect_reference_azimuth(
          condition.visual_latent_location_deg +
          condition.visual_noise_deg);
      const double salience = log_uniform_salience();
      condition.auditory_rate_hz = salience;
      condition.visual_rate_hz = salience;
      condition.physical_soa_ms = truncated_common_soa();
      condition.auditory_latency_ms = positive_latency(21.0, 5.0);
      condition.visual_latency_ms = positive_latency(69.0, 10.0);
      return condition;
    }
    if (kind == ReferenceConditionKind::kIndependentAv) {
      condition.auditory_present = true;
      condition.visual_present = true;
      condition.auditory_latent_location_deg =
          uniform(-60.0, 60.0);
      condition.visual_latent_location_deg =
          uniform(-60.0, 60.0);
      condition.auditory_noise_deg = normal(0.0, 8.0);
      condition.visual_noise_deg = normal(0.0, 2.0);
      condition.auditory_location_deg = reflect_reference_azimuth(
          condition.auditory_latent_location_deg +
          condition.auditory_noise_deg);
      condition.visual_location_deg = reflect_reference_azimuth(
          condition.visual_latent_location_deg +
          condition.visual_noise_deg);
      condition.auditory_rate_hz = log_uniform_salience();
      condition.visual_rate_hz = log_uniform_salience();
      condition.physical_soa_ms = uniform(-600.0, 600.0);
      condition.auditory_latency_ms = positive_latency(21.0, 5.0);
      condition.visual_latency_ms = positive_latency(69.0, 10.0);
      return condition;
    }
    if (kind == ReferenceConditionKind::kAuditoryOnly) {
      condition.auditory_present = true;
      condition.auditory_latent_location_deg =
          uniform(-60.0, 60.0);
      condition.auditory_noise_deg = normal(0.0, 8.0);
      condition.auditory_location_deg = reflect_reference_azimuth(
          condition.auditory_latent_location_deg +
          condition.auditory_noise_deg);
      condition.auditory_rate_hz = log_uniform_salience();
      condition.auditory_latency_ms = positive_latency(21.0, 5.0);
      return condition;
    }
    condition.visual_present = true;
    condition.visual_latent_location_deg =
        uniform(-60.0, 60.0);
    condition.visual_noise_deg = normal(0.0, 2.0);
    condition.visual_location_deg = reflect_reference_azimuth(
        condition.visual_latent_location_deg +
        condition.visual_noise_deg);
    condition.visual_rate_hz = log_uniform_salience();
    condition.visual_latency_ms = positive_latency(69.0, 10.0);
    return condition;
  }

 private:
  double uniform(double low, double high) {
    return low + (high - low) *
                     std::generate_canonical<double, 53>(engine_);
  }

  double normal(double mean, double standard_deviation) {
    return mean + standard_deviation * standard_normal_(engine_);
  }

  double log_uniform_salience() {
    return std::exp(
        uniform(std::log(25.0), std::log(100.0)));
  }

  double positive_latency(double mean, double standard_deviation) {
    for (int draw = 0; draw < 10000; ++draw) {
      const double value = normal(mean, standard_deviation);
      if (value > 0.0) {
        return value;
      }
    }
    throw std::runtime_error(
        "reference positive-latency rejection did not terminate");
  }

  double truncated_common_soa() {
    for (int draw = 0; draw < 10000; ++draw) {
      const double value = normal(-50.0, 60.0);
      if (std::fabs(value) <= 250.0) {
        return value;
      }
    }
    throw std::runtime_error(
        "reference common-SOA rejection did not terminate");
  }

  std::mt19937_64 engine_;
  std::normal_distribution<double> standard_normal_{0.0, 1.0};
};

ReferenceTiming reference_timing(const ReferenceCondition& condition) {
  const double raw_a =
      condition.auditory_present
          ? condition.auditory_latency_ms
          : std::numeric_limits<double>::infinity();
  const double raw_v =
      condition.visual_present
          ? condition.physical_soa_ms +
                condition.visual_latency_ms
          : std::numeric_limits<double>::infinity();
  const double earlier = std::min(raw_a, raw_v);
  const double later =
      condition.auditory_present && condition.visual_present
          ? std::max(raw_a, raw_v)
          : earlier;
  const double shift = 100.0 - earlier;
  ReferenceTiming timing{};
  timing.auditory_onset_ms =
      condition.auditory_present
          ? raw_a + shift
          : std::numeric_limits<double>::quiet_NaN();
  timing.visual_onset_ms =
      condition.visual_present
          ? raw_v + shift
          : std::numeric_limits<double>::quiet_NaN();
  timing.earlier_onset_ms = 100.0;
  timing.poststimulus_end_ms =
      later + shift + 50.0 + 250.0;
  timing.first_zero_afferent_receptor_step =
      timing.poststimulus_end_ms + 1.0;
  timing.valid_end_ms =
      timing.first_zero_afferent_receptor_step + 250.0;
  return timing;
}

std::vector<double> reflected_gaussian_reference(
    double center_deg, double sigma_deg) {
  REQUIRE(sigma_deg > 0.0);
  const double center = reflect_reference_azimuth(center_deg);
  constexpr double low = -90.0;
  constexpr double width = 180.0;
  std::vector<double> values(
      clean_msi::kAuditoryNeurons, 0.0);
  for (int index = 0; index < clean_msi::kAuditoryNeurons; ++index) {
    const double coordinate =
        -89.5 + static_cast<double>(index);
    double value = 0.0;
    for (int period = -1; period <= 1; ++period) {
      const double direct =
          center + 2.0 * static_cast<double>(period) * width;
      const double reflected =
          2.0 * low - center +
          2.0 * static_cast<double>(period) * width;
      const double direct_z = (coordinate - direct) / sigma_deg;
      const double reflected_z =
          (coordinate - reflected) / sigma_deg;
      value += std::exp(-0.5 * direct_z * direct_z);
      value += std::exp(-0.5 * reflected_z * reflected_z);
    }
    values[static_cast<std::size_t>(index)] = value;
  }
  const double peak =
      *std::max_element(values.begin(), values.end());
  REQUIRE(peak > 0.0);
  for (double& value : values) {
    value /= peak;
  }
  return values;
}

std::vector<double> exact_gaussian_reference(
    double center_deg, double sigma_deg) {
  REQUIRE(sigma_deg > 0.0);
  std::vector<double> values(
      clean_msi::kAuditoryNeurons, 0.0);
  for (int index = 0; index < clean_msi::kAuditoryNeurons; ++index) {
    const double coordinate =
        -89.5 + static_cast<double>(index);
    const double z = (coordinate - center_deg) / sigma_deg;
    values[static_cast<std::size_t>(index)] =
        std::exp(-0.5 * z * z);
  }
  const double peak =
      *std::max_element(values.begin(), values.end());
  REQUIRE(peak > 0.0);
  for (double& value : values) {
    value /= peak;
  }
  return values;
}

class Moments {
 public:
  void add(double value) {
    REQUIRE(std::isfinite(value));
    ++count_;
    sum_ += value;
    sum_square_ += value * value;
  }

  std::size_t count() const { return count_; }

  double mean() const {
    REQUIRE(count_ > 0u);
    return sum_ / static_cast<double>(count_);
  }

  double standard_deviation() const {
    REQUIRE(count_ > 1u);
    const double n = static_cast<double>(count_);
    return std::sqrt(std::max(
        (sum_square_ - sum_ * sum_ / n) / (n - 1.0), 0.0));
  }

 private:
  std::size_t count_ = 0;
  double sum_ = 0.0;
  double sum_square_ = 0.0;
};

class BivariateMoments {
 public:
  void add(double first, double second) {
    REQUIRE(std::isfinite(first));
    REQUIRE(std::isfinite(second));
    ++count_;
    first_sum_ += first;
    second_sum_ += second;
    first_square_sum_ += first * first;
    second_square_sum_ += second * second;
    product_sum_ += first * second;
  }

  std::size_t count() const { return count_; }

  double covariance() const {
    REQUIRE(count_ > 1u);
    const double count = static_cast<double>(count_);
    return (product_sum_ -
            first_sum_ * second_sum_ / count) /
           (count - 1.0);
  }

  double correlation() const {
    REQUIRE(count_ > 1u);
    const double count = static_cast<double>(count_);
    const double first_variance =
        (first_square_sum_ -
         first_sum_ * first_sum_ / count) /
        (count - 1.0);
    const double second_variance =
        (second_square_sum_ -
         second_sum_ * second_sum_ / count) /
        (count - 1.0);
    return covariance() /
           std::sqrt(first_variance * second_variance);
  }

 private:
  std::size_t count_ = 0;
  double first_sum_ = 0.0;
  double second_sum_ = 0.0;
  double first_square_sum_ = 0.0;
  double second_square_sum_ = 0.0;
  double product_sum_ = 0.0;
};

struct LocationAudit {
  Moments values{};
  int outside_latent_extent = 0;

  void add(double value) {
    values.add(value);
    if (std::fabs(value) > 60.0) {
      ++outside_latent_extent;
    }
  }

  double tail_fraction() const {
    REQUIRE(values.count() > 0u);
    return static_cast<double>(outside_latent_extent) /
           static_cast<double>(values.count());
  }
};

void test_locked_generator_statistics_20k() {
  constexpr int kConditions = 20000;
  ReferenceStimulusGenerator generator(20260724u);
  std::array<int, 4> kind_counts{};
  Moments common_soa;
  Moments independent_soa;
  Moments auditory_latency;
  Moments visual_latency;
  Moments auditory_noise;
  Moments visual_noise;
  Moments common_disparity;
  Moments log_salience;
  Moments independent_a_log_salience;
  Moments independent_v_log_salience;
  double independent_log_product_sum = 0.0;
  std::array<LocationAudit, 4> auditory_locations_by_kind{};
  std::array<LocationAudit, 4> visual_locations_by_kind{};
  BivariateMoments common_locations;
  BivariateMoments independent_locations;
  Moments independent_disparity;

  for (int index = 0; index < kConditions; ++index) {
    const ReferenceCondition condition = generator.sample();
    ++kind_counts[static_cast<std::size_t>(condition.kind)];
    REQUIRE(condition.auditory_present ||
            condition.visual_present);
    if (condition.auditory_present) {
      REQUIRE(std::isfinite(
          condition.auditory_latent_location_deg));
      REQUIRE(condition.auditory_latent_location_deg >= -60.0);
      REQUIRE(condition.auditory_latent_location_deg <= 60.0);
      REQUIRE(std::isfinite(condition.auditory_noise_deg));
      REQUIRE(std::isfinite(condition.auditory_location_deg));
      REQUIRE(condition.auditory_location_deg >= -90.0);
      REQUIRE(condition.auditory_location_deg <= 90.0);
      REQUIRE(condition.auditory_rate_hz >= 25.0);
      REQUIRE(condition.auditory_rate_hz <= 100.0);
      REQUIRE(condition.auditory_latency_ms > 0.0);
      auditory_latency.add(condition.auditory_latency_ms);
      log_salience.add(std::log(condition.auditory_rate_hz));
      auditory_noise.add(condition.auditory_noise_deg);
      auditory_locations_by_kind[
          static_cast<std::size_t>(condition.kind)]
          .add(condition.auditory_location_deg);
      require_near(
          static_cast<float>(condition.auditory_location_deg),
          static_cast<float>(reflect_reference_azimuth(
              condition.auditory_latent_location_deg +
              condition.auditory_noise_deg)),
          1.0e-5f, 0.0f,
          "auditory latent-noise-reflection equation");
    } else {
      REQUIRE(!std::isfinite(
          condition.auditory_latent_location_deg));
      REQUIRE(!std::isfinite(condition.auditory_noise_deg));
      REQUIRE(!std::isfinite(condition.auditory_location_deg));
      REQUIRE(!std::isfinite(condition.auditory_rate_hz));
      REQUIRE(!std::isfinite(condition.auditory_latency_ms));
    }
    if (condition.visual_present) {
      REQUIRE(std::isfinite(
          condition.visual_latent_location_deg));
      REQUIRE(condition.visual_latent_location_deg >= -60.0);
      REQUIRE(condition.visual_latent_location_deg <= 60.0);
      REQUIRE(std::isfinite(condition.visual_noise_deg));
      REQUIRE(std::isfinite(condition.visual_location_deg));
      REQUIRE(condition.visual_location_deg >= -90.0);
      REQUIRE(condition.visual_location_deg <= 90.0);
      REQUIRE(condition.visual_rate_hz >= 25.0);
      REQUIRE(condition.visual_rate_hz <= 100.0);
      REQUIRE(condition.visual_latency_ms > 0.0);
      visual_latency.add(condition.visual_latency_ms);
      log_salience.add(std::log(condition.visual_rate_hz));
      visual_noise.add(condition.visual_noise_deg);
      visual_locations_by_kind[
          static_cast<std::size_t>(condition.kind)]
          .add(condition.visual_location_deg);
      require_near(
          static_cast<float>(condition.visual_location_deg),
          static_cast<float>(reflect_reference_azimuth(
              condition.visual_latent_location_deg +
              condition.visual_noise_deg)),
          1.0e-5f, 0.0f,
          "visual latent-noise-reflection equation");
    } else {
      REQUIRE(!std::isfinite(
          condition.visual_latent_location_deg));
      REQUIRE(!std::isfinite(condition.visual_noise_deg));
      REQUIRE(!std::isfinite(condition.visual_location_deg));
      REQUIRE(!std::isfinite(condition.visual_rate_hz));
      REQUIRE(!std::isfinite(condition.visual_latency_ms));
    }

    if (condition.kind == ReferenceConditionKind::kCommonAv) {
      REQUIRE(condition.auditory_latent_location_deg ==
              condition.visual_latent_location_deg);
      REQUIRE(condition.auditory_rate_hz ==
              condition.visual_rate_hz);
      REQUIRE(std::fabs(condition.physical_soa_ms) <= 250.0);
      common_soa.add(condition.physical_soa_ms);
      common_disparity.add(
          condition.auditory_location_deg -
          condition.visual_location_deg);
      common_locations.add(
          condition.auditory_location_deg,
          condition.visual_location_deg);
    } else if (
        condition.kind == ReferenceConditionKind::kIndependentAv) {
      REQUIRE(condition.auditory_latent_location_deg !=
              condition.visual_latent_location_deg);
      REQUIRE(std::fabs(condition.physical_soa_ms) <= 600.0);
      independent_soa.add(condition.physical_soa_ms);
      const double auditory_log =
          std::log(condition.auditory_rate_hz);
      const double visual_log =
          std::log(condition.visual_rate_hz);
      independent_a_log_salience.add(auditory_log);
      independent_v_log_salience.add(visual_log);
      independent_log_product_sum += auditory_log * visual_log;
      independent_locations.add(
          condition.auditory_location_deg,
          condition.visual_location_deg);
      independent_disparity.add(
          condition.auditory_location_deg -
          condition.visual_location_deg);
    } else {
      REQUIRE(condition.physical_soa_ms == 0.0);
    }

    const ReferenceTiming timing = reference_timing(condition);
    const double observed_earlier =
        condition.auditory_present && condition.visual_present
            ? std::min(timing.auditory_onset_ms,
                       timing.visual_onset_ms)
            : (condition.auditory_present
                   ? timing.auditory_onset_ms
                   : timing.visual_onset_ms);
    const double observed_later =
        condition.auditory_present && condition.visual_present
            ? std::max(timing.auditory_onset_ms,
                       timing.visual_onset_ms)
            : observed_earlier;
    require_near(
        static_cast<float>(observed_earlier), 100.0f,
        1.0e-4f, 0.0f, "reference prestimulus onset");
    require_near(
        static_cast<float>(
            timing.poststimulus_end_ms -
            (observed_later + 50.0)),
        250.0f, 1.0e-4f, 0.0f,
        "reference poststimulus duration");
    require_near(
        static_cast<float>(
            timing.first_zero_afferent_receptor_step -
            timing.poststimulus_end_ms),
        1.0f, 1.0e-4f, 0.0f,
        "reference first zero-afferent receptor step");
    require_near(
        static_cast<float>(
            timing.valid_end_ms -
            timing.first_zero_afferent_receptor_step),
        250.0f, 1.0e-4f, 0.0f,
        "reference valid silent ITI");
    require_near(
        static_cast<float>(
            timing.valid_end_ms -
            timing.poststimulus_end_ms),
        251.0f, 1.0e-4f, 0.0f,
        "reference metadata-to-valid boundary");
    REQUIRE(std::ceil(timing.valid_end_ms) >
            std::ceil(timing.poststimulus_end_ms));
  }

  const std::array<double, 4> expected_mixture{
      0.40, 0.20, 0.20, 0.20};
  for (std::size_t kind = 0; kind < kind_counts.size(); ++kind) {
    const double observed =
        static_cast<double>(kind_counts[kind]) /
        static_cast<double>(kConditions);
    REQUIRE_MESSAGE(
        std::fabs(observed - expected_mixture[kind]) <= 0.015,
        "20k developmental mixture exceeded sampling tolerance");
  }

  const auto require_auditory_marginal =
      [](const LocationAudit& audit, const std::string& label) {
        REQUIRE_MESSAGE(audit.values.count() > 3000u,
                        label + " sample count is too small");
        REQUIRE_MESSAGE(std::fabs(audit.values.mean()) <= 2.5,
                        label + " location mean is biased");
        REQUIRE_MESSAGE(
            audit.values.standard_deviation() >= 34.0 &&
                audit.values.standard_deviation() <= 37.0,
            label + " does not contain U[-60,60]+N(0,8^2)");
        REQUIRE_MESSAGE(
            audit.tail_fraction() >= 0.035 &&
                audit.tail_fraction() <= 0.075,
            label + " auditory localization-noise tail is absent");
      };
  const auto require_visual_marginal =
      [](const LocationAudit& audit, const std::string& label) {
        REQUIRE_MESSAGE(audit.values.count() > 3000u,
                        label + " sample count is too small");
        REQUIRE_MESSAGE(std::fabs(audit.values.mean()) <= 2.5,
                        label + " location mean is biased");
        REQUIRE_MESSAGE(
            audit.values.standard_deviation() >= 33.5 &&
                audit.values.standard_deviation() <= 36.0,
            label + " does not contain U[-60,60]+N(0,2^2)");
        REQUIRE_MESSAGE(
            audit.tail_fraction() >= 0.006 &&
                audit.tail_fraction() <= 0.025,
            label + " visual localization-noise tail is absent");
      };
  for (const std::size_t kind :
       {std::size_t{0}, std::size_t{1}, std::size_t{2}}) {
    require_auditory_marginal(
        auditory_locations_by_kind[kind],
        "reference auditory kind " + std::to_string(kind));
  }
  for (const std::size_t kind :
       {std::size_t{0}, std::size_t{1}, std::size_t{3}}) {
    require_visual_marginal(
        visual_locations_by_kind[kind],
        "reference visual kind " + std::to_string(kind));
  }
  REQUIRE_MESSAGE(
      std::fabs(
          auditory_locations_by_kind[0].values.standard_deviation() -
          auditory_locations_by_kind[1].values.standard_deviation()) <=
          1.5,
      "common and independent auditory marginals differ");
  REQUIRE_MESSAGE(
      std::fabs(
          auditory_locations_by_kind[0].values.standard_deviation() -
          auditory_locations_by_kind[2].values.standard_deviation()) <=
          1.5,
      "common and unimodal auditory marginals differ");
  REQUIRE_MESSAGE(
      std::fabs(
          visual_locations_by_kind[0].values.standard_deviation() -
          visual_locations_by_kind[1].values.standard_deviation()) <=
          1.5,
      "common and independent visual marginals differ");
  REQUIRE_MESSAGE(
      std::fabs(
          visual_locations_by_kind[0].values.standard_deviation() -
          visual_locations_by_kind[3].values.standard_deviation()) <=
          1.5,
      "common and unimodal visual marginals differ");
  REQUIRE_MESSAGE(common_locations.correlation() >= 0.94,
                  "common AV locations do not share a latent cause");
  REQUIRE_MESSAGE(common_locations.covariance() >= 1100.0 &&
                      common_locations.covariance() <= 1300.0,
                  "common AV location covariance is incorrect");
  REQUIRE_MESSAGE(std::fabs(independent_locations.correlation()) <=
                      0.06,
                  "independent AV locations retain common covariance");
  REQUIRE_MESSAGE(std::fabs(independent_locations.covariance()) <=
                      80.0,
                  "independent AV location covariance is nonzero");
  REQUIRE_MESSAGE(
      independent_disparity.standard_deviation() >= 47.0 &&
          independent_disparity.standard_deviation() <= 52.0,
      "independent AV spatial disparity has wrong width");

  REQUIRE(std::fabs(common_soa.mean() - (-50.0)) <= 3.0);
  REQUIRE(common_soa.standard_deviation() >= 57.0);
  REQUIRE(common_soa.standard_deviation() <= 63.0);
  const double independent_soa_mean_standard_error =
      (600.0 / std::sqrt(3.0)) /
      std::sqrt(static_cast<double>(independent_soa.count()));
  REQUIRE(
      std::fabs(independent_soa.mean()) <=
      4.0 * independent_soa_mean_standard_error);
  REQUIRE(independent_soa.standard_deviation() >= 335.0);
  REQUIRE(independent_soa.standard_deviation() <= 358.0);

  REQUIRE(std::fabs(auditory_latency.mean() - 21.0) <= 0.25);
  REQUIRE(std::fabs(auditory_latency.standard_deviation() - 5.0) <=
          0.25);
  REQUIRE(std::fabs(visual_latency.mean() - 69.0) <= 0.50);
  REQUIRE(std::fabs(visual_latency.standard_deviation() - 10.0) <=
          0.50);
  REQUIRE(std::fabs(auditory_noise.mean()) <= 0.25);
  REQUIRE(std::fabs(auditory_noise.standard_deviation() - 8.0) <=
          0.30);
  REQUIRE(std::fabs(visual_noise.mean()) <= 0.10);
  REQUIRE(std::fabs(visual_noise.standard_deviation() - 2.0) <=
          0.10);
  REQUIRE(std::fabs(common_disparity.mean()) <= 0.30);
  REQUIRE(common_disparity.standard_deviation() >= 7.8);
  REQUIRE(common_disparity.standard_deviation() <= 8.7);

  const double expected_log_mean =
      0.5 * (std::log(25.0) + std::log(100.0));
  const double expected_log_sd =
      (std::log(100.0) - std::log(25.0)) / std::sqrt(12.0);
  REQUIRE(std::fabs(log_salience.mean() - expected_log_mean) <= 0.015);
  REQUIRE(std::fabs(log_salience.standard_deviation() -
                    expected_log_sd) <= 0.015);
  const double independent_count =
      static_cast<double>(independent_a_log_salience.count());
  const double covariance =
      independent_log_product_sum / independent_count -
      independent_a_log_salience.mean() *
          independent_v_log_salience.mean();
  const double correlation =
      covariance /
      (independent_a_log_salience.standard_deviation() *
       independent_v_log_salience.standard_deviation());
  REQUIRE(std::fabs(correlation) <= 0.05);

  require_near(
      static_cast<float>(reflect_reference_azimuth(100.0)),
      80.0f, 1.0e-6f, 0.0f, "right azimuth reflection");
  require_near(
      static_cast<float>(reflect_reference_azimuth(-100.0)),
      -80.0f, 1.0e-6f, 0.0f, "left azimuth reflection");
  require_near(
      static_cast<float>(reflect_reference_azimuth(460.0)),
      80.0f, 1.0e-6f, 0.0f, "multi-period reflection");

  const auto auditory_profile =
      reflected_gaussian_reference(0.0, 8.0);
  const auto visual_profile =
      reflected_gaussian_reference(0.0, 2.0);
  const auto reflected_profile =
      reflected_gaussian_reference(100.0, 8.0);
  const auto equivalent_profile =
      reflected_gaussian_reference(80.0, 8.0);
  REQUIRE(auditory_profile.size() == 180u);
  REQUIRE(visual_profile.size() == 180u);
  REQUIRE(reflected_profile == equivalent_profile);
  require_near(
      static_cast<float>(*std::max_element(
          auditory_profile.begin(), auditory_profile.end())),
      1.0f, 1.0e-7f, 0.0f, "auditory profile peak");
  require_near(
      static_cast<float>(*std::max_element(
          visual_profile.begin(), visual_profile.end())),
      1.0f, 1.0e-7f, 0.0f, "visual profile peak");
  const auto auditory_half =
      std::count_if(
          auditory_profile.begin(), auditory_profile.end(),
          [](double value) { return value >= 0.5; });
  const auto visual_half =
      std::count_if(
          visual_profile.begin(), visual_profile.end(),
          [](double value) { return value >= 0.5; });
  REQUIRE(auditory_half > visual_half);

  const auto require_equal_population_mass =
      [](double center_deg, bool reflect_edges) {
        const auto auditory =
            reflect_edges
                ? reflected_gaussian_reference(center_deg, 8.0)
                : exact_gaussian_reference(center_deg, 8.0);
        auto visual =
            reflect_edges
                ? reflected_gaussian_reference(center_deg, 2.0)
                : exact_gaussian_reference(center_deg, 2.0);
        const double auditory_mass =
            std::accumulate(auditory.begin(), auditory.end(), 0.0);
        const double visual_mass =
            std::accumulate(visual.begin(), visual.end(), 0.0);
        const double visual_scale = auditory_mass / visual_mass;
        for (double& value : visual) {
          value *= visual_scale;
        }
        const double scaled_visual_mass =
            std::accumulate(visual.begin(), visual.end(), 0.0);
        REQUIRE_MESSAGE(
            std::fabs(auditory_mass - scaled_visual_mass) <= 1.0e-12,
            "equal-salience A/V packet masses differ");
      };
  for (const double center_deg :
       {-89.5, -80.0, -37.25, 0.0, 37.25, 80.0, 89.5}) {
    require_equal_population_mass(center_deg, true);
    require_equal_population_mass(center_deg, false);
  }
}

void require_native_sample_exact(
    const DevelopmentalSampleAudit& actual,
    const DevelopmentalSampleAudit& expected,
    std::size_t index) {
  const std::string label =
      "cross-device generator row " + std::to_string(index);
  REQUIRE_MESSAGE(actual.kind == expected.kind,
                  label + ".kind differs");
  REQUIRE_MESSAGE(
      actual.auditory_present == expected.auditory_present,
      label + ".auditory_present differs");
  REQUIRE_MESSAGE(actual.visual_present == expected.visual_present,
                  label + ".visual_present differs");
#define REQUIRE_SAMPLE_FIELD(field)                                         \
  REQUIRE_MESSAGE(actual.field == expected.field,                           \
                  label + "." #field " differs")
  REQUIRE_SAMPLE_FIELD(auditory_latent_deg);
  REQUIRE_SAMPLE_FIELD(visual_latent_deg);
  REQUIRE_SAMPLE_FIELD(auditory_noise_deg);
  REQUIRE_SAMPLE_FIELD(visual_noise_deg);
  REQUIRE_SAMPLE_FIELD(auditory_location_deg);
  REQUIRE_SAMPLE_FIELD(visual_location_deg);
  REQUIRE_SAMPLE_FIELD(auditory_rate_hz);
  REQUIRE_SAMPLE_FIELD(visual_rate_hz);
  REQUIRE_SAMPLE_FIELD(physical_soa_ms);
  REQUIRE_SAMPLE_FIELD(auditory_latency_ms);
  REQUIRE_SAMPLE_FIELD(visual_latency_ms);
  REQUIRE_SAMPLE_FIELD(auditory_onset_ms);
  REQUIRE_SAMPLE_FIELD(visual_onset_ms);
  REQUIRE_SAMPLE_FIELD(poststimulus_end_ms);
  REQUIRE_SAMPLE_FIELD(first_zero_afferent_receptor_step);
  REQUIRE_SAMPLE_FIELD(valid_end_ms);
#undef REQUIRE_SAMPLE_FIELD
}

void test_native_generator_audit_20k() {
  constexpr int kSamples = 20000;
  constexpr std::uint64_t kSeed = 20260724u;
  constexpr float kProfileCenterDeg = 100.0f;
  constexpr float kProfileSigmaDeg = 8.0f;
  const std::vector<int> devices = available_devices();
  REQUIRE_MESSAGE(!devices.empty(),
                  "native generator audit requires CUDA");
  std::vector<GeneratorProfileAudit> audits;
  audits.reserve(devices.size());
  for (const int device : devices) {
    audits.push_back(clean_msi::audit_developmental_generator(
        device, kSeed, kSamples, kProfileCenterDeg,
        kProfileSigmaDeg));
    REQUIRE(audits.back().samples.size() ==
            static_cast<std::size_t>(kSamples));
  }
  if (audits.size() > 1u) {
    REQUIRE(audits[1].samples.size() == audits[0].samples.size());
    for (std::size_t index = 0; index < audits[0].samples.size();
         ++index) {
      require_native_sample_exact(
          audits[1].samples[index], audits[0].samples[index], index);
    }
    REQUIRE(audits[1].developmental_reflected_profile ==
            audits[0].developmental_reflected_profile);
    REQUIRE(audits[1].controlled_exact_profile ==
            audits[0].controlled_exact_profile);
  }

  const GeneratorProfileAudit& audit = audits.front();
  std::array<int, 4> kind_counts{};
  std::array<LocationAudit, 4> auditory_locations{};
  std::array<LocationAudit, 4> visual_locations{};
  Moments auditory_noise;
  Moments visual_noise;
  Moments auditory_latency;
  Moments visual_latency;
  Moments all_log_salience;
  Moments common_soa;
  Moments independent_soa;
  Moments common_disparity;
  Moments independent_disparity;
  BivariateMoments common_locations;
  BivariateMoments independent_locations;
  Moments independent_a_log_salience;
  Moments independent_v_log_salience;
  double independent_salience_product_sum = 0.0;

  for (std::size_t index = 0; index < audit.samples.size(); ++index) {
    const DevelopmentalSampleAudit& sample = audit.samples[index];
    const int raw_kind = static_cast<int>(sample.kind);
    REQUIRE(raw_kind >= 0 && raw_kind < 4);
    ++kind_counts[static_cast<std::size_t>(raw_kind)];
    const bool expected_auditory =
        sample.kind == PresentationKind::kCommonAv ||
        sample.kind == PresentationKind::kIndependentAv ||
        sample.kind == PresentationKind::kAuditoryOnly;
    const bool expected_visual =
        sample.kind == PresentationKind::kCommonAv ||
        sample.kind == PresentationKind::kIndependentAv ||
        sample.kind == PresentationKind::kVisualOnly;
    REQUIRE(sample.auditory_present == expected_auditory);
    REQUIRE(sample.visual_present == expected_visual);

    if (sample.auditory_present) {
      REQUIRE(sample.auditory_latent_deg >= -60.0f);
      REQUIRE(sample.auditory_latent_deg <= 60.0f);
      REQUIRE(std::isfinite(sample.auditory_noise_deg));
      REQUIRE(sample.auditory_location_deg >= -90.0f);
      REQUIRE(sample.auditory_location_deg <= 90.0f);
      REQUIRE(sample.auditory_rate_hz >= 25.0f);
      REQUIRE(sample.auditory_rate_hz <= 100.0f);
      REQUIRE(sample.auditory_latency_ms > 0.0f);
      REQUIRE(sample.auditory_onset_ms >= 100);
      require_near(
          sample.auditory_location_deg,
          static_cast<float>(reflect_reference_azimuth(
              static_cast<double>(sample.auditory_latent_deg) +
              static_cast<double>(sample.auditory_noise_deg))),
          2.0e-5f, 0.0f,
          "native auditory latent-noise-reflection");
      auditory_locations[static_cast<std::size_t>(raw_kind)].add(
          sample.auditory_location_deg);
      auditory_noise.add(sample.auditory_noise_deg);
      auditory_latency.add(sample.auditory_latency_ms);
      all_log_salience.add(std::log(sample.auditory_rate_hz));
    } else {
      REQUIRE(sample.auditory_latent_deg == 0.0f);
      REQUIRE(sample.auditory_noise_deg == 0.0f);
      REQUIRE(sample.auditory_location_deg == 0.0f);
      REQUIRE(sample.auditory_rate_hz == 0.0f);
      REQUIRE(sample.auditory_latency_ms == 0.0f);
      REQUIRE(sample.auditory_onset_ms == -1);
    }
    if (sample.visual_present) {
      REQUIRE(sample.visual_latent_deg >= -60.0f);
      REQUIRE(sample.visual_latent_deg <= 60.0f);
      REQUIRE(std::isfinite(sample.visual_noise_deg));
      REQUIRE(sample.visual_location_deg >= -90.0f);
      REQUIRE(sample.visual_location_deg <= 90.0f);
      REQUIRE(sample.visual_rate_hz >= 25.0f);
      REQUIRE(sample.visual_rate_hz <= 100.0f);
      REQUIRE(sample.visual_latency_ms > 0.0f);
      REQUIRE(sample.visual_onset_ms >= 100);
      require_near(
          sample.visual_location_deg,
          static_cast<float>(reflect_reference_azimuth(
              static_cast<double>(sample.visual_latent_deg) +
              static_cast<double>(sample.visual_noise_deg))),
          2.0e-5f, 0.0f,
          "native visual latent-noise-reflection");
      visual_locations[static_cast<std::size_t>(raw_kind)].add(
          sample.visual_location_deg);
      visual_noise.add(sample.visual_noise_deg);
      visual_latency.add(sample.visual_latency_ms);
      all_log_salience.add(std::log(sample.visual_rate_hz));
    } else {
      REQUIRE(sample.visual_latent_deg == 0.0f);
      REQUIRE(sample.visual_noise_deg == 0.0f);
      REQUIRE(sample.visual_location_deg == 0.0f);
      REQUIRE(sample.visual_rate_hz == 0.0f);
      REQUIRE(sample.visual_latency_ms == 0.0f);
      REQUIRE(sample.visual_onset_ms == -1);
    }

    const int earlier_onset =
        sample.auditory_present && sample.visual_present
            ? std::min(sample.auditory_onset_ms,
                       sample.visual_onset_ms)
            : (sample.auditory_present
                   ? sample.auditory_onset_ms
                   : sample.visual_onset_ms);
    const int later_onset =
        sample.auditory_present && sample.visual_present
            ? std::max(sample.auditory_onset_ms,
                       sample.visual_onset_ms)
            : earlier_onset;
    REQUIRE(earlier_onset == 100);
    REQUIRE(sample.poststimulus_end_ms ==
            later_onset + 50 + 250);
    REQUIRE(sample.first_zero_afferent_receptor_step ==
            sample.poststimulus_end_ms + 1);
    REQUIRE(sample.valid_end_ms ==
            sample.first_zero_afferent_receptor_step + 250);
    REQUIRE(sample.valid_end_ms ==
            sample.poststimulus_end_ms + 251);

    if (sample.kind == PresentationKind::kCommonAv) {
      REQUIRE(sample.auditory_latent_deg ==
              sample.visual_latent_deg);
      REQUIRE(sample.auditory_rate_hz ==
              sample.visual_rate_hz);
      REQUIRE(std::fabs(sample.physical_soa_ms) <= 250.0f);
      common_soa.add(sample.physical_soa_ms);
      common_disparity.add(
          sample.auditory_location_deg -
          sample.visual_location_deg);
      common_locations.add(
          sample.auditory_location_deg,
          sample.visual_location_deg);
    } else if (sample.kind == PresentationKind::kIndependentAv) {
      REQUIRE(sample.auditory_latent_deg !=
              sample.visual_latent_deg);
      REQUIRE(std::fabs(sample.physical_soa_ms) <= 600.0f);
      independent_soa.add(sample.physical_soa_ms);
      independent_disparity.add(
          sample.auditory_location_deg -
          sample.visual_location_deg);
      independent_locations.add(
          sample.auditory_location_deg,
          sample.visual_location_deg);
      const double auditory_log =
          std::log(sample.auditory_rate_hz);
      const double visual_log =
          std::log(sample.visual_rate_hz);
      independent_a_log_salience.add(auditory_log);
      independent_v_log_salience.add(visual_log);
      independent_salience_product_sum +=
          auditory_log * visual_log;
    } else {
      REQUIRE(sample.physical_soa_ms == 0.0f);
    }
  }

  const std::array<double, 4> expected_mixture{
      0.40, 0.20, 0.20, 0.20};
  for (std::size_t kind = 0; kind < kind_counts.size(); ++kind) {
    const double fraction =
        static_cast<double>(kind_counts[kind]) /
        static_cast<double>(kSamples);
    REQUIRE_MESSAGE(
        std::fabs(fraction - expected_mixture[kind]) <= 0.015,
        "native 20k mixture exceeds sampling tolerance");
  }
  const auto require_auditory_marginal =
      [](const LocationAudit& marginal, const std::string& label) {
        REQUIRE(marginal.values.count() > 3000u);
        REQUIRE_MESSAGE(std::fabs(marginal.values.mean()) <= 2.5,
                        label + " mean is biased");
        REQUIRE_MESSAGE(
            marginal.values.standard_deviation() >= 34.0 &&
                marginal.values.standard_deviation() <= 37.0,
            label + " width omits auditory noise");
        REQUIRE_MESSAGE(
            marginal.tail_fraction() >= 0.035 &&
                marginal.tail_fraction() <= 0.075,
            label + " tail omits auditory noise");
      };
  const auto require_visual_marginal =
      [](const LocationAudit& marginal, const std::string& label) {
        REQUIRE(marginal.values.count() > 3000u);
        REQUIRE_MESSAGE(std::fabs(marginal.values.mean()) <= 2.5,
                        label + " mean is biased");
        REQUIRE_MESSAGE(
            marginal.values.standard_deviation() >= 33.5 &&
                marginal.values.standard_deviation() <= 36.0,
            label + " width omits visual noise");
        REQUIRE_MESSAGE(
            marginal.tail_fraction() >= 0.006 &&
                marginal.tail_fraction() <= 0.025,
            label + " tail omits visual noise");
      };
  for (const std::size_t kind :
       {std::size_t{0}, std::size_t{1}, std::size_t{2}}) {
    require_auditory_marginal(
        auditory_locations[kind],
        "native auditory kind " + std::to_string(kind));
  }
  for (const std::size_t kind :
       {std::size_t{0}, std::size_t{1}, std::size_t{3}}) {
    require_visual_marginal(
        visual_locations[kind],
        "native visual kind " + std::to_string(kind));
  }
  REQUIRE(std::fabs(
              auditory_locations[0].values.standard_deviation() -
              auditory_locations[1].values.standard_deviation()) <=
          1.5);
  REQUIRE(std::fabs(
              auditory_locations[0].values.standard_deviation() -
              auditory_locations[2].values.standard_deviation()) <=
          1.5);
  REQUIRE(std::fabs(
              visual_locations[0].values.standard_deviation() -
              visual_locations[1].values.standard_deviation()) <=
          1.5);
  REQUIRE(std::fabs(
              visual_locations[0].values.standard_deviation() -
              visual_locations[3].values.standard_deviation()) <=
          1.5);
  REQUIRE(common_locations.correlation() >= 0.94);
  REQUIRE(common_locations.covariance() >= 1100.0);
  REQUIRE(common_locations.covariance() <= 1300.0);
  REQUIRE(std::fabs(independent_locations.correlation()) <= 0.06);
  REQUIRE(std::fabs(independent_locations.covariance()) <= 80.0);
  REQUIRE(common_disparity.standard_deviation() >= 7.8);
  REQUIRE(common_disparity.standard_deviation() <= 8.7);
  REQUIRE(independent_disparity.standard_deviation() >= 47.0);
  REQUIRE(independent_disparity.standard_deviation() <= 52.0);
  REQUIRE(std::fabs(auditory_noise.mean()) <= 0.25);
  REQUIRE(std::fabs(auditory_noise.standard_deviation() - 8.0) <=
          0.30);
  REQUIRE(std::fabs(visual_noise.mean()) <= 0.10);
  REQUIRE(std::fabs(visual_noise.standard_deviation() - 2.0) <=
          0.10);
  REQUIRE(std::fabs(auditory_latency.mean() - 21.0) <= 0.25);
  REQUIRE(std::fabs(auditory_latency.standard_deviation() - 5.0) <=
          0.25);
  REQUIRE(std::fabs(visual_latency.mean() - 69.0) <= 0.50);
  REQUIRE(std::fabs(visual_latency.standard_deviation() - 10.0) <=
          0.50);
  REQUIRE(std::fabs(common_soa.mean() - (-50.0)) <= 3.0);
  REQUIRE(common_soa.standard_deviation() >= 57.0);
  REQUIRE(common_soa.standard_deviation() <= 63.0);
  const double independent_soa_standard_error =
      (600.0 / std::sqrt(3.0)) /
      std::sqrt(static_cast<double>(independent_soa.count()));
  REQUIRE(std::fabs(independent_soa.mean()) <=
          4.0 * independent_soa_standard_error);
  REQUIRE(independent_soa.standard_deviation() >= 335.0);
  REQUIRE(independent_soa.standard_deviation() <= 358.0);
  const double expected_log_mean =
      0.5 * (std::log(25.0) + std::log(100.0));
  const double expected_log_sd =
      (std::log(100.0) - std::log(25.0)) / std::sqrt(12.0);
  REQUIRE(std::fabs(all_log_salience.mean() -
                    expected_log_mean) <= 0.015);
  REQUIRE(std::fabs(all_log_salience.standard_deviation() -
                    expected_log_sd) <= 0.015);
  const double independent_count =
      static_cast<double>(independent_a_log_salience.count());
  const double salience_covariance =
      independent_salience_product_sum / independent_count -
      independent_a_log_salience.mean() *
          independent_v_log_salience.mean();
  const double salience_correlation =
      salience_covariance /
      (independent_a_log_salience.standard_deviation() *
       independent_v_log_salience.standard_deviation());
  REQUIRE(std::fabs(salience_correlation) <= 0.05);

  const auto expected_reflected =
      reflected_gaussian_reference(
          kProfileCenterDeg, kProfileSigmaDeg);
  const auto expected_controlled =
      exact_gaussian_reference(
          kProfileCenterDeg, kProfileSigmaDeg);
  float reflected_peak = 0.0f;
  float controlled_peak = 0.0f;
  for (std::size_t index = 0;
       index < audit.developmental_reflected_profile.size(); ++index) {
    const float reflected =
        audit.developmental_reflected_profile[index];
    const float controlled =
        audit.controlled_exact_profile[index];
    REQUIRE(reflected >= 0.0f);
    REQUIRE(controlled >= 0.0f);
    reflected_peak = std::max(reflected_peak, reflected);
    controlled_peak = std::max(controlled_peak, controlled);
    require_near(
        reflected,
        static_cast<float>(expected_reflected[index]),
        2.0e-5f, 2.0e-5f,
        "native reflected profile bin");
    require_near(
        controlled,
        static_cast<float>(expected_controlled[index]),
        2.0e-5f, 2.0e-5f,
        "native controlled profile bin");
  }
  require_near(reflected_peak, 1.0f, 1.0e-6f, 0.0f,
               "native reflected profile peak");
  require_near(controlled_peak, 1.0f, 1.0e-6f, 0.0f,
               "native controlled profile peak");
  const auto reflected_peak_iterator = std::max_element(
      audit.developmental_reflected_profile.begin(),
      audit.developmental_reflected_profile.end());
  const auto controlled_peak_iterator = std::max_element(
      audit.controlled_exact_profile.begin(),
      audit.controlled_exact_profile.end());
  const int reflected_peak_index = static_cast<int>(
      std::distance(
          audit.developmental_reflected_profile.begin(),
          reflected_peak_iterator));
  const int controlled_peak_index = static_cast<int>(
      std::distance(
          audit.controlled_exact_profile.begin(),
          controlled_peak_iterator));
  const int expected_reflected_peak_index = static_cast<int>(
      std::distance(
          expected_reflected.begin(),
          std::max_element(
              expected_reflected.begin(), expected_reflected.end())));
  REQUIRE(reflected_peak_index == expected_reflected_peak_index);
  REQUIRE(controlled_peak_index ==
          clean_msi::kAuditoryNeurons - 1);
  REQUIRE(audit.developmental_reflected_profile !=
          audit.controlled_exact_profile);

  std::cout
      << "[GENERATOR] samples=" << kSamples
      << " devices=" << devices.size()
      << " mixture="
      << static_cast<double>(kind_counts[0]) / kSamples << ','
      << static_cast<double>(kind_counts[1]) / kSamples << ','
      << static_cast<double>(kind_counts[2]) / kSamples << ','
      << static_cast<double>(kind_counts[3]) / kSamples << '\n';
  for (const std::size_t kind :
       {std::size_t{0}, std::size_t{1}, std::size_t{2}}) {
    std::cout
        << "[GENERATOR] A_kind=" << kind
        << " mean_deg=" << auditory_locations[kind].values.mean()
        << " sd_deg="
        << auditory_locations[kind].values.standard_deviation()
        << " tail_abs_gt60="
        << auditory_locations[kind].tail_fraction() << '\n';
  }
  for (const std::size_t kind :
       {std::size_t{0}, std::size_t{1}, std::size_t{3}}) {
    std::cout
        << "[GENERATOR] V_kind=" << kind
        << " mean_deg=" << visual_locations[kind].values.mean()
        << " sd_deg="
        << visual_locations[kind].values.standard_deviation()
        << " tail_abs_gt60="
        << visual_locations[kind].tail_fraction() << '\n';
  }
  std::cout
      << "[GENERATOR] common_cov=" << common_locations.covariance()
      << " common_corr=" << common_locations.correlation()
      << " common_disparity_sd="
      << common_disparity.standard_deviation()
      << " independent_cov=" << independent_locations.covariance()
      << " independent_corr=" << independent_locations.correlation()
      << " independent_disparity_sd="
      << independent_disparity.standard_deviation() << '\n'
      << "[GENERATOR] A_latency_mean_sd="
      << auditory_latency.mean() << ','
      << auditory_latency.standard_deviation()
      << " V_latency_mean_sd=" << visual_latency.mean() << ','
      << visual_latency.standard_deviation()
      << " common_soa_mean_sd=" << common_soa.mean() << ','
      << common_soa.standard_deviation()
      << " independent_soa_mean_sd=" << independent_soa.mean() << ','
      << independent_soa.standard_deviation() << '\n'
      << "[GENERATOR] log_salience_mean_sd="
      << all_log_salience.mean() << ','
      << all_log_salience.standard_deviation()
      << " independent_salience_corr=" << salience_correlation
      << " reflected_peak_index=" << reflected_peak_index
      << " controlled_peak_index=" << controlled_peak_index << '\n';
}

// ---------------------------------------------------------------------------
// Independent Stage-3 evaluation references.

constexpr int kObserverFeatureBins = 40;
constexpr int kObserverFeatureMilliseconds = 800;

std::array<int, kObserverFeatureBins + 1> observer_bin_edges_ms() {
  std::array<int, kObserverFeatureBins + 1> edges{};
  for (int index = 0; index <= kObserverFeatureBins; ++index) {
    edges[static_cast<std::size_t>(index)] = -100 + 20 * index;
  }
  return edges;
}

std::array<double, kObserverFeatureBins> scalar_observer_features(
    const std::array<unsigned int, kObserverFeatureMilliseconds>&
        excitatory_spikes_per_ms) {
  std::array<double, kObserverFeatureBins> features{};
  for (int bin = 0; bin < kObserverFeatureBins; ++bin) {
    double count = 0.0;
    for (int offset = 0; offset < 20; ++offset) {
      count += excitatory_spikes_per_ms[
          static_cast<std::size_t>(20 * bin + offset)];
    }
    features[static_cast<std::size_t>(bin)] = count;
  }
  return features;
}

double stable_sigmoid(double value) {
  if (value >= 0.0) {
    const double negative = std::exp(-value);
    return 1.0 / (1.0 + negative);
  }
  const double positive = std::exp(value);
  return positive / (1.0 + positive);
}

struct ScalarLogisticModel {
  std::vector<double> mean;
  std::vector<double> scale;
  std::vector<double> coefficients;
  double intercept = 0.0;
  double l2 = 1.0e-3;
};

ScalarLogisticModel fit_scalar_logistic(
    const std::vector<std::vector<double>>& features,
    const std::vector<int>& labels, double l2 = 1.0e-3) {
  REQUIRE(!features.empty());
  REQUIRE(features.size() == labels.size());
  const std::size_t sample_count = features.size();
  const std::size_t feature_count = features.front().size();
  REQUIRE(feature_count > 0u);
  for (const auto& row : features) {
    REQUIRE(row.size() == feature_count);
  }
  ScalarLogisticModel model{};
  model.mean.assign(feature_count, 0.0);
  model.scale.assign(feature_count, 0.0);
  model.coefficients.assign(feature_count, 0.0);
  model.l2 = l2;
  for (const auto& row : features) {
    for (std::size_t column = 0; column < feature_count; ++column) {
      model.mean[column] += row[column] /
                            static_cast<double>(sample_count);
    }
  }
  for (const auto& row : features) {
    for (std::size_t column = 0; column < feature_count; ++column) {
      const double centered = row[column] - model.mean[column];
      model.scale[column] += centered * centered;
    }
  }
  for (double& scale : model.scale) {
    scale = std::sqrt(
        scale / static_cast<double>(sample_count));
    if (scale < 1.0e-8) {
      scale = 1.0;
    }
  }

  constexpr int kIterations = 3000;
  constexpr double kLearningRate = 0.15;
  std::vector<double> gradient(feature_count, 0.0);
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    std::fill(gradient.begin(), gradient.end(), 0.0);
    double intercept_gradient = 0.0;
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
      double logit = model.intercept;
      for (std::size_t column = 0; column < feature_count; ++column) {
        const double standardized =
            (features[sample][column] - model.mean[column]) /
            model.scale[column];
        logit += model.coefficients[column] * standardized;
      }
      const double residual =
          stable_sigmoid(logit) -
          static_cast<double>(labels[sample]);
      intercept_gradient += residual;
      for (std::size_t column = 0; column < feature_count; ++column) {
        const double standardized =
            (features[sample][column] - model.mean[column]) /
            model.scale[column];
        gradient[column] += residual * standardized;
      }
    }
    const double inverse_count =
        1.0 / static_cast<double>(sample_count);
    model.intercept -=
        kLearningRate * intercept_gradient * inverse_count;
    for (std::size_t column = 0; column < feature_count; ++column) {
      const double regularized =
          gradient[column] * inverse_count +
          l2 * model.coefficients[column];
      model.coefficients[column] -= kLearningRate * regularized;
    }
  }
  return model;
}

std::vector<double> scalar_logistic_probabilities(
    const ScalarLogisticModel& model,
    const std::vector<std::vector<double>>& features) {
  std::vector<double> probabilities;
  probabilities.reserve(features.size());
  for (const auto& row : features) {
    REQUIRE(row.size() == model.coefficients.size());
    double logit = model.intercept;
    for (std::size_t column = 0; column < row.size(); ++column) {
      logit += model.coefficients[column] *
               ((row[column] - model.mean[column]) /
                model.scale[column]);
    }
    probabilities.push_back(stable_sigmoid(logit));
  }
  return probabilities;
}

double scalar_brier(const std::vector<double>& probabilities,
                    const std::vector<int>& labels) {
  REQUIRE(probabilities.size() == labels.size());
  double sum = 0.0;
  for (std::size_t index = 0; index < probabilities.size(); ++index) {
    const double error =
        probabilities[index] - static_cast<double>(labels[index]);
    sum += error * error;
  }
  return sum / static_cast<double>(probabilities.size());
}

double scalar_auroc(const std::vector<double>& probabilities,
                    const std::vector<int>& labels) {
  REQUIRE(probabilities.size() == labels.size());
  double wins = 0.0;
  std::size_t pairs = 0u;
  for (std::size_t positive = 0; positive < labels.size(); ++positive) {
    if (labels[positive] != 1) {
      continue;
    }
    for (std::size_t negative = 0; negative < labels.size(); ++negative) {
      if (labels[negative] != 0) {
        continue;
      }
      ++pairs;
      if (probabilities[positive] > probabilities[negative]) {
        wins += 1.0;
      } else if (probabilities[positive] ==
                 probabilities[negative]) {
        wins += 0.5;
      }
    }
  }
  REQUIRE(pairs > 0u);
  return wins / static_cast<double>(pairs);
}

struct ScalarAsymmetricFit {
  bool valid = false;
  bool crossings_valid = false;
  double baseline = 0.0;
  double amplitude = 0.0;
  double center = 0.0;
  double sigma_left = 0.0;
  double sigma_right = 0.0;
  double tbw50 = 0.0;
  double tbw75 = 0.0;
};

double median_double(std::vector<double> values) {
  REQUIRE(!values.empty());
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2u;
  if (values.size() % 2u == 1u) {
    return values[middle];
  }
  return 0.5 * (values[middle - 1u] + values[middle]);
}

ScalarAsymmetricFit scalar_asymmetric_fit(
    const std::vector<double>& x,
    const std::vector<double>& y) {
  REQUIRE(x.size() == y.size());
  REQUIRE(x.size() >= 5u);
  ScalarAsymmetricFit fit{};
  fit.baseline = 0.5 * (y.front() + y.back());
  const auto peak_iterator =
      std::max_element(y.begin(), y.end());
  const std::size_t peak_index = static_cast<std::size_t>(
      std::distance(y.begin(), peak_iterator));
  fit.center = x[peak_index];
  fit.amplitude = *peak_iterator - fit.baseline;
  if (!(fit.amplitude > 1.0e-6)) {
    return fit;
  }
  std::vector<double> left_sigma;
  std::vector<double> right_sigma;
  for (std::size_t index = 0; index < x.size(); ++index) {
    if (index == peak_index) {
      continue;
    }
    const double fraction =
        (y[index] - fit.baseline) / fit.amplitude;
    if (!(fraction > 0.05 && fraction < 0.95)) {
      continue;
    }
    const double denominator =
        std::sqrt(-2.0 * std::log(fraction));
    const double sigma =
        std::fabs(x[index] - fit.center) / denominator;
    if (x[index] < fit.center) {
      left_sigma.push_back(sigma);
    } else {
      right_sigma.push_back(sigma);
    }
  }
  if (left_sigma.empty() || right_sigma.empty()) {
    return fit;
  }
  fit.sigma_left = median_double(std::move(left_sigma));
  fit.sigma_right = median_double(std::move(right_sigma));
  fit.valid = std::isfinite(fit.sigma_left) &&
              std::isfinite(fit.sigma_right) &&
              fit.sigma_left > 0.0 && fit.sigma_right > 0.0;
  if (!fit.valid) {
    return fit;
  }
  const double factor50 = std::sqrt(2.0 * std::log(2.0));
  const double factor75 = std::sqrt(-2.0 * std::log(0.75));
  fit.tbw50 = factor50 * (fit.sigma_left + fit.sigma_right);
  fit.tbw75 = factor75 * (fit.sigma_left + fit.sigma_right);
  const double left_crossing =
      fit.center - factor75 * fit.sigma_left;
  const double right_crossing =
      fit.center + factor75 * fit.sigma_right;
  fit.crossings_valid =
      left_crossing >= x.front() && right_crossing <= x.back();
  return fit;
}

struct ScalarSymmetricFit {
  bool valid = false;
  bool crossing_valid = false;
  double baseline = 0.0;
  double amplitude = 0.0;
  double sigma = 0.0;
  double half_width = 0.0;
};

ScalarSymmetricFit scalar_symmetric_fit(
    const std::vector<double>& disparity,
    const std::vector<double>& enhancement) {
  REQUIRE(disparity.size() == enhancement.size());
  REQUIRE(disparity.size() >= 4u);
  REQUIRE(disparity.front() == 0.0);
  ScalarSymmetricFit fit{};
  fit.baseline = enhancement.back();
  fit.amplitude = enhancement.front() - fit.baseline;
  if (!(fit.amplitude > 1.0e-6)) {
    return fit;
  }
  std::vector<double> sigma_estimates;
  for (std::size_t index = 1; index < disparity.size(); ++index) {
    const double fraction =
        (enhancement[index] - fit.baseline) / fit.amplitude;
    if (!(fraction > 0.05 && fraction < 0.95)) {
      continue;
    }
    sigma_estimates.push_back(
        disparity[index] /
        std::sqrt(-2.0 * std::log(fraction)));
  }
  if (sigma_estimates.empty()) {
    return fit;
  }
  fit.sigma = median_double(std::move(sigma_estimates));
  fit.valid = std::isfinite(fit.sigma) && fit.sigma > 0.0;
  fit.half_width = std::sqrt(2.0 * std::log(2.0)) * fit.sigma;
  fit.crossing_valid =
      fit.valid && fit.half_width <= disparity.back();
  return fit;
}

std::vector<float> direct_sbw_grid_deg() {
  std::vector<float> grid;
  grid.reserve(clean_msi::kSbwPointCount);
  for (int index = 0; index < clean_msi::kSbwPointCount; ++index) {
    grid.push_back(2.5f * static_cast<float>(index));
  }
  return grid;
}

std::vector<float> direct_piecewise_sbw_curve(
    const std::vector<float>& disparity_deg,
    float sbw50_deg,
    float center_enhancement_hz = 11.0f,
    float tail_baseline_hz = 1.0f) {
  REQUIRE(sbw50_deg > 0.0f);
  REQUIRE(center_enhancement_hz > tail_baseline_hz);
  std::vector<float> enhancement;
  enhancement.reserve(disparity_deg.size());
  for (const float disparity : disparity_deg) {
    const float normalized =
        std::max(0.0f, 1.0f - disparity / (2.0f * sbw50_deg));
    enhancement.push_back(
        tail_baseline_hz +
        (center_enhancement_hz - tail_baseline_hz) *
            normalized);
  }
  return enhancement;
}

float legacy_first_raw_half_crossing(
    const std::vector<float>& disparity_deg,
    const std::vector<float>& pooled_enhancement_hz) {
  REQUIRE(disparity_deg.size() == pooled_enhancement_hz.size());
  REQUIRE(disparity_deg.size() >= 4u);
  const std::size_t count = pooled_enhancement_hz.size();
  const float tail_baseline =
      (pooled_enhancement_hz[count - 3u] +
       pooled_enhancement_hz[count - 2u] +
       pooled_enhancement_hz[count - 1u]) /
      3.0f;
  const float half_level =
      tail_baseline +
      0.5f *
          (pooled_enhancement_hz.front() - tail_baseline);
  for (std::size_t index = 1; index < count; ++index) {
    const float previous = pooled_enhancement_hz[index - 1u];
    const float current = pooled_enhancement_hz[index];
    if (previous >= half_level && current <= half_level) {
      const float denominator = previous - current;
      const float fraction =
          denominator > 0.0f
              ? (previous - half_level) / denominator
              : 0.0f;
      return disparity_deg[index - 1u] +
             fraction *
                 (disparity_deg[index] -
                  disparity_deg[index - 1u]);
    }
  }
  return std::numeric_limits<float>::quiet_NaN();
}

double pearson_double(const std::vector<double>& first,
                      const std::vector<double>& second) {
  REQUIRE(first.size() == second.size());
  REQUIRE(first.size() >= 2u);
  const double first_mean =
      std::accumulate(first.begin(), first.end(), 0.0) /
      static_cast<double>(first.size());
  const double second_mean =
      std::accumulate(second.begin(), second.end(), 0.0) /
      static_cast<double>(second.size());
  double numerator = 0.0;
  double first_square = 0.0;
  double second_square = 0.0;
  for (std::size_t index = 0; index < first.size(); ++index) {
    const double first_centered = first[index] - first_mean;
    const double second_centered = second[index] - second_mean;
    numerator += first_centered * second_centered;
    first_square += first_centered * first_centered;
    second_square += second_centered * second_centered;
  }
  return numerator / std::sqrt(first_square * second_square);
}

std::pair<double, double> weighted_center_width(
    const std::vector<double>& coordinates,
    const std::vector<double>& weights) {
  REQUIRE(coordinates.size() == weights.size());
  double total = 0.0;
  double weighted = 0.0;
  for (std::size_t index = 0; index < weights.size(); ++index) {
    REQUIRE(weights[index] >= 0.0);
    total += weights[index];
    weighted += coordinates[index] * weights[index];
  }
  REQUIRE(total > 0.0);
  const double center = weighted / total;
  double variance = 0.0;
  for (std::size_t index = 0; index < weights.size(); ++index) {
    const double offset = coordinates[index] - center;
    variance += weights[index] * offset * offset;
  }
  return {center, std::sqrt(variance / total)};
}

void test_stage3_feature_bins_and_logistic() {
  const auto edges = observer_bin_edges_ms();
  REQUIRE(edges.front() == -100);
  REQUIRE(edges.back() == 700);
  for (std::size_t index = 1; index < edges.size(); ++index) {
    REQUIRE(edges[index] - edges[index - 1u] == 20);
  }
  std::array<unsigned int, kObserverFeatureMilliseconds> per_ms{};
  for (std::size_t index = 0; index < per_ms.size(); ++index) {
    per_ms[index] =
        static_cast<unsigned int>((index * 7u + 3u) % 5u);
  }
  const auto binned = scalar_observer_features(per_ms);
  double total_binned = 0.0;
  for (int bin = 0; bin < kObserverFeatureBins; ++bin) {
    double expected = 0.0;
    for (int offset = 0; offset < 20; ++offset) {
      expected += per_ms[static_cast<std::size_t>(20 * bin + offset)];
    }
    REQUIRE(binned[static_cast<std::size_t>(bin)] == expected);
    total_binned += binned[static_cast<std::size_t>(bin)];
  }
  const double total_ms =
      std::accumulate(per_ms.begin(), per_ms.end(), 0.0);
  REQUIRE(total_binned == total_ms);

  // Physical SOA is trial metadata, not an observer feature.  Re-labelling the
  // same 800 ms spike train with different physical SOAs must therefore leave
  // the exact 40-dimensional feature vector unchanged.
  const std::array<double, 3> physical_soa_ms{-500.0, 0.0, 500.0};
  for (const double ignored_physical_soa_ms : physical_soa_ms) {
    static_cast<void>(ignored_physical_soa_ms);
    REQUIRE(scalar_observer_features(per_ms) == binned);
  }

  constexpr int kSamples = 800;
  std::vector<std::vector<double>> all_features(
      kSamples, std::vector<double>(kObserverFeatureBins, 0.0));
  std::vector<int> labels(kSamples, 0);
  for (int sample = 0; sample < kSamples; ++sample) {
    const int label = sample % 2;
    labels[static_cast<std::size_t>(sample)] = label;
    const double sign = label == 1 ? 1.0 : -1.0;
    for (int column = 0; column < kObserverFeatureBins; ++column) {
      const double nuisance =
          0.20 * std::sin(0.17 * sample + 0.31 * column);
      all_features[static_cast<std::size_t>(sample)]
                  [static_cast<std::size_t>(column)] =
          nuisance +
          (column < 4 ? sign * (2.0 - 0.2 * column) : 0.0);
    }
  }
  std::vector<std::vector<double>> train_features(
      all_features.begin(), all_features.begin() + 600);
  std::vector<int> train_labels(labels.begin(), labels.begin() + 600);
  std::vector<std::vector<double>> holdout_features(
      all_features.begin() + 600, all_features.end());
  std::vector<int> holdout_labels(labels.begin() + 600, labels.end());
  const ScalarLogisticModel first =
      fit_scalar_logistic(train_features, train_labels, 1.0e-3);
  const ScalarLogisticModel replay =
      fit_scalar_logistic(train_features, train_labels, 1.0e-3);
  REQUIRE(first.mean == replay.mean);
  REQUIRE(first.scale == replay.scale);
  REQUIRE(first.coefficients == replay.coefficients);
  REQUIRE(first.intercept == replay.intercept);
  REQUIRE(first.coefficients.size() ==
          static_cast<std::size_t>(kObserverFeatureBins));
  const auto probabilities =
      scalar_logistic_probabilities(first, holdout_features);
  std::vector<std::vector<double>> identical_spike_features(
      physical_soa_ms.size(),
      std::vector<double>(binned.begin(), binned.end()));
  const auto metadata_invariant_probabilities =
      scalar_logistic_probabilities(first, identical_spike_features);
  REQUIRE(metadata_invariant_probabilities.size() ==
          physical_soa_ms.size());
  for (std::size_t index = 1;
       index < metadata_invariant_probabilities.size(); ++index) {
    REQUIRE(metadata_invariant_probabilities[index] ==
            metadata_invariant_probabilities[0]);
  }
  const double auroc = scalar_auroc(probabilities, holdout_labels);
  const double brier = scalar_brier(probabilities, holdout_labels);
  REQUIRE(auroc >= 0.99);
  REQUIRE(brier <= 0.02);
}

void test_stage3_raw_control_response_windows() {
  FrozenTrialBatch batch{};
  batch.model_seed_count = 1;
  batch.condition_count = 2;
  batch.trials_per_condition = 1;
  batch.trials.resize(2);

  constexpr int prestimulus_bins =
      100 / clean_msi::kEvaluationBinWidthMs;
  for (FrozenTrialResult& trial : batch.trials) {
    for (int bin = 0; bin < prestimulus_bins; ++bin) {
      trial.features[static_cast<std::size_t>(bin)] = 1.0f;
    }
  }

  // A one-bin onset is genuinely supra-threshold in the raw PSTH.  The
  // former centered smoother mapped it to exactly threshold:
  // 0.25 * 1.0 + 0.50 * 1.5 + 0.25 * 0.0 = 1.0.
  batch.trials[0].features[5] = 1.5f;
  batch.trials[1].features[7] = 2.0f;
  batch.trials[1].features[8] = 2.0f;
  batch.trials[1].features[9] = 2.0f;

  const clean_msi::NeuralFusionRule rule =
      clean_msi::estimate_neural_fusion_rule(batch, 0, 1);
  require_near(
      rule.threshold, 1.0f, 1.0e-7f, 0.0f,
      "raw control response threshold");
  const float formerly_smoothed_onset =
      0.25f * batch.trials[0].features[4] +
      0.50f * batch.trials[0].features[5] +
      0.25f * batch.trials[0].features[6];
  REQUIRE(formerly_smoothed_onset == rule.threshold);
  REQUIRE(!(formerly_smoothed_onset > rule.threshold));
  REQUIRE(rule.auditory.first_bin == 5);
  REQUIRE(rule.auditory.last_bin == 5);
  REQUIRE(
      -100 +
          rule.auditory.first_bin *
              clean_msi::kEvaluationBinWidthMs ==
      0);
  REQUIRE(
      -100 +
          (rule.auditory.last_bin + 1) *
              clean_msi::kEvaluationBinWidthMs ==
      20);
  REQUIRE(rule.visual.first_bin == 7);
  REQUIRE(rule.visual.last_bin == 9);

  const clean_msi::NeuralFusionRule swapped =
      clean_msi::estimate_neural_fusion_rule(batch, 1, 0);
  REQUIRE(swapped.threshold == rule.threshold);
  REQUIRE(
      swapped.auditory.first_bin == rule.visual.first_bin);
  REQUIRE(
      swapped.auditory.last_bin == rule.visual.last_bin);
  REQUIRE(
      swapped.visual.first_bin == rule.auditory.first_bin);
  REQUIRE(
      swapped.visual.last_bin == rule.auditory.last_bin);

  FrozenTrialBatch silent = batch;
  for (FrozenTrialResult& trial : silent.trials) {
    for (int bin = prestimulus_bins;
         bin < clean_msi::kEvaluationFeatureBins; ++bin) {
      trial.features[static_cast<std::size_t>(bin)] = 1.0f;
    }
  }
  bool rejected = false;
  try {
    static_cast<void>(
        clean_msi::estimate_neural_fusion_rule(silent, 0, 1));
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  REQUIRE_MESSAGE(
      rejected,
      "raw controls without a strict supra-threshold response were accepted");
}

void test_stage3_exact_grids_and_synthetic_fits() {
  std::vector<double> tbw_grid;
  for (int soa = -500; soa <= 500; soa += 25) {
    tbw_grid.push_back(static_cast<double>(soa));
  }
  REQUIRE(tbw_grid.size() == 41u);
  REQUIRE(tbw_grid.front() == -500.0);
  REQUIRE(tbw_grid.back() == 500.0);
  constexpr double baseline = 0.10;
  constexpr double amplitude = 0.80;
  constexpr double center = -25.0;
  constexpr double sigma_left = 70.0;
  constexpr double sigma_right = 110.0;
  std::vector<double> fusion;
  for (const double soa : tbw_grid) {
    const double sigma = soa < center ? sigma_left : sigma_right;
    fusion.push_back(
        baseline + amplitude *
                       std::exp(-0.5 * std::pow((soa - center) / sigma, 2)));
  }
  const ScalarAsymmetricFit tbw =
      scalar_asymmetric_fit(tbw_grid, fusion);
  REQUIRE(tbw.valid);
  REQUIRE(tbw.crossings_valid);
  REQUIRE(std::fabs(tbw.center - center) <= 1.0e-9);
  REQUIRE(std::fabs(tbw.sigma_left - sigma_left) <= 1.0);
  REQUIRE(std::fabs(tbw.sigma_right - sigma_right) <= 1.0);
  REQUIRE(std::isfinite(tbw.tbw75));
  REQUIRE(tbw.tbw75 > 0.0);
  REQUIRE(tbw.amplitude >= 0.25);
  const ScalarAsymmetricFit flat_tbw =
      scalar_asymmetric_fit(
          tbw_grid, std::vector<double>(tbw_grid.size(), 0.4));
  REQUIRE(!flat_tbw.valid);
  REQUIRE(!flat_tbw.crossings_valid);
  std::vector<double> broad_tbw(tbw_grid.size(), 0.1 + 0.8 * 0.94);
  broad_tbw.front() = 0.1;
  broad_tbw.back() = 0.1;
  broad_tbw[tbw_grid.size() / 2u] = 0.9;
  const ScalarAsymmetricFit no_crossing_tbw =
      scalar_asymmetric_fit(tbw_grid, broad_tbw);
  REQUIRE(no_crossing_tbw.valid);
  REQUIRE(!no_crossing_tbw.crossings_valid);

  std::vector<double> sbw_grid;
  for (int index = 0; index <= 24; ++index) {
    sbw_grid.push_back(2.5 * static_cast<double>(index));
  }
  REQUIRE(sbw_grid.size() == 25u);
  REQUIRE(sbw_grid.front() == 0.0);
  REQUIRE(sbw_grid.back() == 60.0);
  constexpr double target_half_width = 20.0;
  const double sigma =
      target_half_width / std::sqrt(2.0 * std::log(2.0));
  std::vector<double> enhancement;
  for (const double disparity : sbw_grid) {
    enhancement.push_back(
        1.0 + 10.0 *
                  std::exp(-0.5 * std::pow(disparity / sigma, 2)));
  }
  const ScalarSymmetricFit sbw =
      scalar_symmetric_fit(sbw_grid, enhancement);
  REQUIRE(sbw.valid);
  REQUIRE(sbw.crossing_valid);
  REQUIRE(std::fabs(sbw.half_width - target_half_width) <= 0.75);
  REQUIRE(sbw.half_width >= 15.0 && sbw.half_width <= 25.0);
  const ScalarSymmetricFit flat_sbw =
      scalar_symmetric_fit(
          sbw_grid, std::vector<double>(sbw_grid.size(), 2.0));
  REQUIRE(!flat_sbw.valid);
  REQUIRE(!flat_sbw.crossing_valid);
  std::vector<double> broad_sbw(sbw_grid.size(), 1.0 + 10.0 * 0.94);
  broad_sbw.front() = 11.0;
  broad_sbw.back() = 1.0;
  const ScalarSymmetricFit no_crossing_sbw =
      scalar_symmetric_fit(sbw_grid, broad_sbw);
  REQUIRE(no_crossing_sbw.valid);
  REQUIRE(!no_crossing_sbw.crossing_valid);

  const std::vector<float> direct_grid = direct_sbw_grid_deg();
  const std::vector<float> sem(
      direct_grid.size(), 0.125f);
  const std::vector<float> centered_known_20 =
      direct_piecewise_sbw_curve(direct_grid, 20.0f);
  const clean_msi::DirectSpatialAudit centered =
      clean_msi::direct_spatial_audit(
          direct_grid, centered_known_20, sem);
  REQUIRE(centered.input_valid);
  REQUIRE(centered.finite);
  REQUIRE(centered.strictly_increasing);
  REQUIRE(centered.center_positive);
  REQUIRE(centered.contrast_positive);
  REQUIRE(centered.endpoints_valid);
  REQUIRE(centered.peak_location_valid);
  REQUIRE(centered.contiguous_prefix);
  REQUIRE(centered.single_outward_crossing);
  REQUIRE(centered.width_valid);
  REQUIRE(centered.spatial_gate);
  REQUIRE(centered.peak_index == 0);
  require_near(
      centered.center_enhancement_hz, 11.0f,
      1.0e-6f, 1.0e-6f, "direct SBW center");
  require_near(
      centered.tail_baseline_hz, 1.0f,
      1.0e-6f, 1.0e-6f, "direct SBW tail");
  require_near(
      centered.contrast_hz, 10.0f,
      1.0e-6f, 1.0e-6f, "direct SBW contrast");
  require_near(
      centered.sbw50_deg, 20.0f,
      1.0e-6f, 1.0e-6f, "direct known-20 SBW");

  const std::vector<float> p1000_u_curve{
      0.311555564f, 0.344000f, 0.352556f, 0.396667f,
      0.391944f, 0.358722f, 0.309944451f, 0.293888867f,
      0.212889f, 0.157278f, 0.192056f, 0.162500f,
      0.204333f, 0.270056f, 0.280944f, 0.407389f,
      0.439222f, 0.444000f, 0.471666634f, 0.430611f,
      0.437722f, 0.364611f, 0.306166679f, 0.287388891f,
      0.308722228f};
  REQUIRE(p1000_u_curve.size() == direct_grid.size());
  const float p1000_legacy_crossing =
      legacy_first_raw_half_crossing(
          direct_grid, p1000_u_curve);
  require_near(
      p1000_legacy_crossing, 15.58967584f,
      2.0e-5f, 2.0e-6f,
      "p1000 legacy first-dip loophole");
  const clean_msi::DirectSpatialAudit p1000 =
      clean_msi::direct_spatial_audit(
          direct_grid, p1000_u_curve, sem);
  REQUIRE(p1000.input_valid);
  REQUIRE(p1000.finite);
  REQUIRE(p1000.strictly_increasing);
  REQUIRE(p1000.endpoints_valid);
  require_near(
      p1000.tail_baseline_hz, 0.300759266f,
      1.0e-7f, 1.0e-7f, "p1000 tail baseline");
  require_near(
      p1000.half_level_hz, 0.306157415f,
      2.0e-7f, 2.0e-7f, "p1000 half level");
  REQUIRE(p1000.peak_index == 18);
  require_near(
      p1000.peak_disparity_deg, 45.0f,
      1.0e-6f, 1.0e-6f, "p1000 global peak");
  REQUIRE(!p1000.peak_location_valid);
  REQUIRE(!p1000.contiguous_prefix);
  REQUIRE(!p1000.single_outward_crossing);
  REQUIRE(!p1000.width_valid);
  REQUIRE(!p1000.spatial_gate);

  std::vector<float> late_rebound = centered_known_20;
  late_rebound[20] = 7.0f;
  const clean_msi::DirectSpatialAudit rebound =
      clean_msi::direct_spatial_audit(
          direct_grid, late_rebound, sem);
  REQUIRE(rebound.endpoints_valid);
  REQUIRE(rebound.peak_location_valid);
  REQUIRE(!rebound.contiguous_prefix);
  REQUIRE(!rebound.single_outward_crossing);
  REQUIRE(!rebound.width_valid);
  REQUIRE(!rebound.spatial_gate);

  const clean_msi::DirectSpatialAudit negative_center =
      clean_msi::direct_spatial_audit(
          direct_grid,
          direct_piecewise_sbw_curve(
              direct_grid, 20.0f, -1.0f, -11.0f),
          sem);
  REQUIRE(!negative_center.center_positive);
  REQUIRE(negative_center.contrast_positive);
  REQUIRE(!negative_center.endpoints_valid);
  REQUIRE(!negative_center.width_valid);
  REQUIRE(!negative_center.spatial_gate);

  const clean_msi::DirectSpatialAudit flat =
      clean_msi::direct_spatial_audit(
          direct_grid,
          std::vector<float>(direct_grid.size(), 2.0f),
          sem);
  REQUIRE(flat.center_positive);
  REQUIRE(!flat.contrast_positive);
  REQUIRE(!flat.endpoints_valid);
  REQUIRE(!flat.single_outward_crossing);
  REQUIRE(!flat.spatial_gate);

  std::vector<float> short_sem = sem;
  short_sem.pop_back();
  const clean_msi::DirectSpatialAudit mismatched =
      clean_msi::direct_spatial_audit(
          direct_grid, centered_known_20, short_sem);
  REQUIRE(!mismatched.input_valid);
  REQUIRE(!mismatched.spatial_gate);

  std::vector<float> repeated_grid = direct_grid;
  repeated_grid[2] = repeated_grid[1];
  const clean_msi::DirectSpatialAudit non_increasing =
      clean_msi::direct_spatial_audit(
          repeated_grid, centered_known_20, sem);
  REQUIRE(non_increasing.input_valid);
  REQUIRE(non_increasing.finite);
  REQUIRE(!non_increasing.strictly_increasing);
  REQUIRE(!non_increasing.spatial_gate);

  std::vector<float> nonfinite_curve = centered_known_20;
  nonfinite_curve[4] =
      std::numeric_limits<float>::quiet_NaN();
  const clean_msi::DirectSpatialAudit nonfinite =
      clean_msi::direct_spatial_audit(
          direct_grid, nonfinite_curve, sem);
  REQUIRE(nonfinite.input_valid);
  REQUIRE(!nonfinite.finite);
  REQUIRE(!nonfinite.spatial_gate);

  std::vector<float> nonfinite_sem = sem;
  nonfinite_sem[3] =
      std::numeric_limits<float>::infinity();
  const clean_msi::DirectSpatialAudit nonfinite_uncertainty =
      clean_msi::direct_spatial_audit(
          direct_grid, centered_known_20, nonfinite_sem);
  REQUIRE(nonfinite_uncertainty.input_valid);
  REQUIRE(!nonfinite_uncertainty.finite);
  REQUIRE(!nonfinite_uncertainty.spatial_gate);

  std::vector<float> peak_at_first_step = centered_known_20;
  peak_at_first_step[1] = 11.5f;
  const clean_msi::DirectSpatialAudit peak_2_5 =
      clean_msi::direct_spatial_audit(
          direct_grid, peak_at_first_step, sem);
  REQUIRE(peak_2_5.peak_index == 1);
  REQUIRE(peak_2_5.peak_location_valid);
  REQUIRE(peak_2_5.spatial_gate);

  std::vector<float> peak_at_second_step = centered_known_20;
  peak_at_second_step[2] = 11.5f;
  const clean_msi::DirectSpatialAudit peak_5 =
      clean_msi::direct_spatial_audit(
          direct_grid, peak_at_second_step, sem);
  REQUIRE(peak_5.peak_index == 2);
  REQUIRE(!peak_5.peak_location_valid);
  REQUIRE(!peak_5.width_valid);
  REQUIRE(!peak_5.spatial_gate);

  const clean_msi::DirectSpatialAudit lower_boundary =
      clean_msi::direct_spatial_audit(
          direct_grid,
          direct_piecewise_sbw_curve(direct_grid, 15.0f),
          sem);
  const clean_msi::DirectSpatialAudit upper_boundary =
      clean_msi::direct_spatial_audit(
          direct_grid,
          direct_piecewise_sbw_curve(direct_grid, 25.0f),
          sem);
  require_near(
      lower_boundary.sbw50_deg, 15.0f,
      1.0e-6f, 1.0e-6f, "inclusive lower SBW boundary");
  require_near(
      upper_boundary.sbw50_deg, 25.0f,
      1.0e-6f, 1.0e-6f, "inclusive upper SBW boundary");
  REQUIRE(lower_boundary.spatial_gate);
  REQUIRE(upper_boundary.spatial_gate);

  EvaluationMetrics criteria_wiring{};
  criteria_wiring.spatial.direct = centered;
  criteria_wiring.spatial.fit.valid = false;
  criteria_wiring.spatial.fit.hwhm_valid = false;
  criteria_wiring.sbw50_deg =
      criteria_wiring.spatial.direct.sbw50_deg;
  criteria_wiring.spatial_gate =
      criteria_wiring.spatial.direct.spatial_gate;
  REQUIRE(criteria_wiring.sbw50_deg == centered.sbw50_deg);
  REQUIRE(criteria_wiring.spatial_gate);
  criteria_wiring.spatial.direct = p1000;
  criteria_wiring.spatial.fit.valid = true;
  criteria_wiring.spatial.fit.hwhm_valid = true;
  criteria_wiring.spatial.fit.fitted_hwhm_deg =
      p1000_legacy_crossing;
  criteria_wiring.sbw50_deg =
      criteria_wiring.spatial.direct.sbw50_deg;
  criteria_wiring.spatial_gate =
      criteria_wiring.spatial.direct.spatial_gate;
  REQUIRE(criteria_wiring.sbw50_deg == 0.0f);
  REQUIRE(!criteria_wiring.spatial_gate);
}

void test_stage3_empirical_tbw_crossings_and_rejections() {
  std::array<float, clean_msi::kTbwPointCount> locations{};
  std::array<clean_msi::TemporalCurvePoint,
             clean_msi::kTbwPointCount>
      points{};
  for (int index = 0; index < clean_msi::kTbwPointCount; ++index) {
    const std::size_t position = static_cast<std::size_t>(index);
    const float soa = -500.0f + 25.0f * index;
    locations[position] = soa;
    points[position].physical_soa_ms = soa;
    const float sigma = soa < -25.0f ? 70.0f : 110.0f;
    points[position].probability_mean =
        0.10f +
        0.80f *
            std::exp(
                -0.5f *
                std::pow((soa + 25.0f) / sigma, 2.0f));
  }
  const clean_msi::EmpiricalTemporalCrossings valid =
      clean_msi::empirical_temporal_crossings(locations, points);
  REQUIRE(valid.valid);
  REQUIRE(valid.unimodal);
  REQUIRE(valid.peak_index > 0);
  REQUIRE(valid.peak_index < clean_msi::kTbwPointCount - 1);
  REQUIRE(valid.left_50_ms > locations.front());
  REQUIRE(valid.right_50_ms < locations.back());
  REQUIRE(valid.left_75_ms > locations.front());
  REQUIRE(valid.right_75_ms < locations.back());
  REQUIRE(valid.tbw50_ms > valid.tbw75_ms);

  clean_msi::EmpiricalTemporalCrossings gate_empirical{};
  gate_empirical.valid = true;
  gate_empirical.unimodal = true;
  gate_empirical.tbw50_ms = 150.0f;
  const std::array<float, 4> diagnostic_tbw75{
      std::numeric_limits<float>::quiet_NaN(),
      50.0f, 150.0f, 500.0f};
  for (const float tbw75 : diagnostic_tbw75) {
    gate_empirical.tbw75_ms = tbw75;
    REQUIRE(clean_msi::temporal_gate_passes(
        gate_empirical, 0.25f));
  }

  auto broad_gate = gate_empirical;
  broad_gate.tbw50_ms = 350.0f;
  broad_gate.tbw75_ms = 150.0f;
  REQUIRE(!clean_msi::temporal_gate_passes(
      broad_gate, 0.25f));

  auto lower_boundary = gate_empirical;
  lower_boundary.tbw50_ms = 100.0f;
  auto upper_boundary = gate_empirical;
  upper_boundary.tbw50_ms = 300.0f;
  REQUIRE(clean_msi::temporal_gate_passes(
      lower_boundary, 0.25f));
  REQUIRE(clean_msi::temporal_gate_passes(
      upper_boundary, 0.25f));

  auto invalid_gate = gate_empirical;
  invalid_gate.valid = false;
  REQUIRE(!clean_msi::temporal_gate_passes(
      invalid_gate, 0.25f));
  auto nonunimodal_gate = gate_empirical;
  nonunimodal_gate.unimodal = false;
  REQUIRE(!clean_msi::temporal_gate_passes(
      nonunimodal_gate, 0.25f));
  auto nonfinite_width = gate_empirical;
  nonfinite_width.tbw50_ms =
      std::numeric_limits<float>::quiet_NaN();
  REQUIRE(!clean_msi::temporal_gate_passes(
      nonfinite_width, 0.25f));
  REQUIRE(!clean_msi::temporal_gate_passes(
      gate_empirical,
      std::numeric_limits<float>::infinity()));
  REQUIRE(!clean_msi::temporal_gate_passes(
      gate_empirical, 0.249f));

  std::array<float, clean_msi::kTbwPointCount> raw_locations{};
  std::array<clean_msi::TemporalCurvePoint,
             clean_msi::kTbwPointCount>
      raw_points{};
  for (int index = 0; index < clean_msi::kTbwPointCount; ++index) {
    const std::size_t position = static_cast<std::size_t>(index);
    const float soa = -500.0f + 25.0f * index;
    const float absolute_soa = std::fabs(soa);
    raw_locations[position] = soa;
    raw_points[position].physical_soa_ms = soa;
    if (absolute_soa == 0.0f) {
      raw_points[position].probability_mean = 0.70f;
    } else if (absolute_soa == 25.0f) {
      raw_points[position].probability_mean = 0.633333333f;
    } else if (absolute_soa == 50.0f) {
      raw_points[position].probability_mean = 0.566666667f;
    } else if (absolute_soa == 75.0f) {
      raw_points[position].probability_mean = 0.50f;
    } else if (absolute_soa == 100.0f) {
      raw_points[position].probability_mean = 0.40f;
    } else {
      raw_points[position].probability_mean = 0.10f;
    }
  }
  const clean_msi::EmpiricalTemporalCrossings raw_p50_only =
      clean_msi::empirical_temporal_crossings(
          raw_locations, raw_points);
  REQUIRE(raw_p50_only.valid);
  REQUIRE(raw_p50_only.unimodal);
  require_near(
      raw_p50_only.peak_probability, 0.70f,
      1.0e-7f, 1.0e-7f, "raw P50-only peak");
  require_near(
      raw_p50_only.left_50_ms, -75.0f,
      1.0e-6f, 0.0f, "raw P50-only left crossing");
  require_near(
      raw_p50_only.right_50_ms, 75.0f,
      1.0e-6f, 0.0f, "raw P50-only right crossing");
  require_near(
      raw_p50_only.tbw50_ms, 150.0f,
      1.0e-6f, 0.0f, "raw P50-only width");
  REQUIRE(std::isnan(raw_p50_only.left_75_ms));
  REQUIRE(std::isnan(raw_p50_only.right_75_ms));
  REQUIRE(std::isnan(raw_p50_only.tbw75_ms));
  const float raw_peak_minus_tail =
      raw_p50_only.peak_probability -
      0.5f *
          (raw_p50_only.left_tail_baseline +
           raw_p50_only.right_tail_baseline);
  require_near(
      raw_peak_minus_tail, 0.60f,
      1.0e-7f, 1.0e-7f,
      "raw P50-only peak-minus-tail");
  REQUIRE(clean_msi::temporal_gate_passes(
      raw_p50_only, raw_peak_minus_tail));

  auto flat = points;
  for (auto& point : flat) {
    point.probability_mean = 0.40f;
  }
  const clean_msi::EmpiricalTemporalCrossings no_crossing =
      clean_msi::empirical_temporal_crossings(locations, flat);
  REQUIRE(!no_crossing.valid);
  REQUIRE(!no_crossing.unimodal);

  auto disconnected = points;
  disconnected[8].probability_mean = 0.80f;
  const clean_msi::EmpiricalTemporalCrossings non_unimodal =
      clean_msi::empirical_temporal_crossings(
          locations, disconnected);
  REQUIRE(!non_unimodal.valid);
  REQUIRE(!non_unimodal.unimodal);
}

void test_stage3_sbw_and_inverse_effectiveness_algebra() {
  const std::array<double, 2> orientation_offsets{-0.5, 0.5};
  for (int disparity_index = 0; disparity_index <= 24;
       ++disparity_index) {
    const double disparity = 2.5 * disparity_index;
    const double gain =
        8.0 * std::exp(-0.5 * std::pow(disparity / 17.0, 2));
    std::array<double, 2> orientation_gain{};
    for (std::size_t orientation = 0;
         orientation < orientation_offsets.size(); ++orientation) {
      const double auditory = 10.0 + orientation_offsets[orientation];
      const double visual = 8.0 - orientation_offsets[orientation];
      const double av = std::max(auditory, visual) + gain;
      const double measured_gain =
          av - std::max(auditory, visual);
      const double enhancement_percent =
          100.0 * measured_gain / std::max(auditory, visual);
      const double additivity = av - (auditory + visual);
      const double additivity_percent =
          100.0 * additivity / (auditory + visual);
      require_near(
          static_cast<float>(measured_gain),
          static_cast<float>(gain), 1.0e-6f, 1.0e-6f,
          "SBW raw gain algebra");
      require_near(
          static_cast<float>(enhancement_percent),
          static_cast<float>(
              100.0 * gain / std::max(auditory, visual)),
          1.0e-6f, 1.0e-6f, "SBW enhancement algebra");
      require_near(
          static_cast<float>(additivity_percent),
          static_cast<float>(
              100.0 * (av - auditory - visual) /
              (auditory + visual)),
          1.0e-6f, 1.0e-6f, "SBW additivity algebra");
      orientation_gain[orientation] = measured_gain;
    }
    require_near(
        static_cast<float>(orientation_gain[0]),
        static_cast<float>(orientation_gain[1]),
        1.0e-6f, 1.0e-6f, "SBW orientation pooling");
  }

  const std::array<double, 3> salience{25.0, 50.0, 100.0};
  std::array<double, 3> multisensory_enhancement{};
  for (std::size_t index = 0; index < salience.size(); ++index) {
    const double auditory = 0.30 * salience[index];
    const double visual = 0.25 * salience[index];
    const double gain = 9.0 * std::sqrt(25.0 / salience[index]);
    const double av = std::max(auditory, visual) + gain;
    const double additivity = av - auditory - visual;
    const double expected_enhancement =
        100.0 * gain / std::max(auditory, visual);
    multisensory_enhancement[index] =
        100.0 * (av - std::max(auditory, visual)) /
        std::max(auditory, visual);
    require_near(
        static_cast<float>(av - std::max(auditory, visual)),
        static_cast<float>(gain), 1.0e-6f, 1.0e-6f,
        "inverse-effectiveness raw gain algebra");
    require_near(
        static_cast<float>(multisensory_enhancement[index]),
        static_cast<float>(expected_enhancement),
        1.0e-6f, 1.0e-6f,
        "inverse-effectiveness enhancement algebra");
    require_near(
        static_cast<float>(additivity),
        static_cast<float>(av - (auditory + visual)),
        1.0e-6f, 1.0e-6f,
        "inverse-effectiveness additivity algebra");
  }
  REQUIRE(multisensory_enhancement[0] >
          multisensory_enhancement[1]);
  REQUIRE(multisensory_enhancement[1] >
          multisensory_enhancement[2]);
}

void test_stage3_sbw_matched_trial_orientation_averaging() {
  const auto first =
      clean_msi::summarize_matched_component_trials(
          {10.0f, 2.0f}, {2.0f, 10.0f},
          {11.0f, 11.0f}, 1.0f);
  const auto second =
      clean_msi::summarize_matched_component_trials(
          {20.0f, 4.0f}, {4.0f, 20.0f},
          {21.0f, 21.0f}, 1.0f);
  require_near(
      first.raw_enhancement_hz, 1.0f,
      1.0e-6f, 1.0e-6f, "matched first gain");
  require_near(
      first.multisensory_enhancement_percent, 10.0f,
      1.0e-6f, 1.0e-6f, "matched first ME");
  require_near(
      second.raw_enhancement_hz, 1.0f,
      1.0e-6f, 1.0e-6f, "matched second gain");
  require_near(
      second.multisensory_enhancement_percent, 5.0f,
      1.0e-6f, 1.0e-6f, "matched second ME");

  const auto pooled =
      clean_msi::pool_spatial_orientations(first, second);
  require_near(
      pooled.raw_enhancement_hz, 1.0f,
      1.0e-6f, 1.0e-6f, "pooled matched gain");
  require_near(
      pooled.multisensory_enhancement_percent, 7.5f,
      1.0e-6f, 1.0e-6f, "pooled matched ME");
  require_near(
      pooled.additivity_hz, -2.0f,
      1.0e-6f, 1.0e-6f, "pooled matched additivity");
  REQUIRE_MESSAGE(
      pooled.raw_enhancement_hz !=
          pooled.audiovisual_rate_hz -
              std::max(
                  pooled.auditory_rate_hz,
                  pooled.visual_rate_hz),
      "pooled gain was recomputed after pooling component rates");

  const auto swapped =
      clean_msi::pool_spatial_orientations(second, first);
  REQUIRE(swapped.raw_enhancement_hz ==
          pooled.raw_enhancement_hz);
  REQUIRE(swapped.raw_enhancement_sem ==
          pooled.raw_enhancement_sem);

  const std::vector<float> direct_grid = direct_sbw_grid_deg();
  const std::vector<float> base_curve =
      direct_piecewise_sbw_curve(direct_grid, 20.0f);
  std::vector<float> forward_gain;
  std::vector<float> reverse_gain;
  std::vector<float> forward_sem;
  std::vector<float> reverse_sem;
  forward_gain.reserve(direct_grid.size());
  reverse_gain.reserve(direct_grid.size());
  forward_sem.reserve(direct_grid.size());
  reverse_sem.reserve(direct_grid.size());
  for (std::size_t index = 0; index < direct_grid.size(); ++index) {
    clean_msi::ComponentResponseMetrics left_right{};
    clean_msi::ComponentResponseMetrics right_left{};
    left_right.raw_enhancement_hz =
        base_curve[index] + 0.4f;
    right_left.raw_enhancement_hz =
        base_curve[index] - 0.4f;
    left_right.raw_enhancement_sem =
        0.2f + 0.001f * static_cast<float>(index);
    right_left.raw_enhancement_sem =
        0.5f + 0.001f * static_cast<float>(index);
    const auto forward =
        clean_msi::pool_spatial_orientations(
            left_right, right_left);
    const auto reverse =
        clean_msi::pool_spatial_orientations(
            right_left, left_right);
    REQUIRE(forward.raw_enhancement_hz ==
            reverse.raw_enhancement_hz);
    REQUIRE(forward.raw_enhancement_sem ==
            reverse.raw_enhancement_sem);
    forward_gain.push_back(forward.raw_enhancement_hz);
    reverse_gain.push_back(reverse.raw_enhancement_hz);
    forward_sem.push_back(forward.raw_enhancement_sem);
    reverse_sem.push_back(reverse.raw_enhancement_sem);
  }
  const clean_msi::DirectSpatialAudit forward_audit =
      clean_msi::direct_spatial_audit(
          direct_grid, forward_gain, forward_sem);
  const clean_msi::DirectSpatialAudit reverse_audit =
      clean_msi::direct_spatial_audit(
          direct_grid, reverse_gain, reverse_sem);
  REQUIRE(forward_audit.spatial_gate);
  REQUIRE(reverse_audit.spatial_gate);
  REQUIRE(forward_audit.sbw50_deg ==
          reverse_audit.sbw50_deg);
}

void test_stage3_synthetic_rf_and_topography() {
  std::vector<double> rf_locations;
  for (int location = -60; location <= 60; location += 5) {
    rf_locations.push_back(static_cast<double>(location));
  }
  REQUIRE(rf_locations.size() == 25u);
  std::vector<double> target_centers;
  for (int center = -50; center <= 50; center += 5) {
    target_centers.push_back(static_cast<double>(center));
  }
  std::vector<double> recovered_a;
  std::vector<double> recovered_v;
  for (const double center : target_centers) {
    std::vector<double> auditory_response;
    std::vector<double> visual_response;
    for (const double location : rf_locations) {
      auditory_response.push_back(
          std::exp(-0.5 * std::pow((location - center) / 8.0, 2)));
      visual_response.push_back(
          std::exp(-0.5 * std::pow((location - center) / 4.0, 2)));
    }
    const auto auditory =
        weighted_center_width(rf_locations, auditory_response);
    const auto visual =
        weighted_center_width(rf_locations, visual_response);
    recovered_a.push_back(auditory.first);
    recovered_v.push_back(visual.first);
    REQUIRE(std::fabs(auditory.first - center) <= 1.0);
    REQUIRE(std::fabs(visual.first - center) <= 1.0);
    REQUIRE(auditory.second > visual.second);
  }
  REQUIRE(pearson_double(target_centers, recovered_a) >= 0.99);
  REQUIRE(pearson_double(target_centers, recovered_v) >= 0.99);

  std::vector<double> source_coordinates;
  for (int index = 0; index < 180; ++index) {
    source_coordinates.push_back(-89.5 + index);
  }
  std::vector<double> excitation_a_centers;
  std::vector<double> excitation_v_centers;
  std::vector<double> inhibition_a_centers;
  std::vector<double> inhibition_v_centers;
  std::vector<double> recurrent_centers;
  std::vector<double> inhibitory_coordinates;
  for (int index = 0; index < 60; ++index) {
    inhibitory_coordinates.push_back(-88.5 + 3.0 * index);
  }
  for (const double center : target_centers) {
    std::vector<double> a_excitation;
    std::vector<double> v_excitation;
    std::vector<double> a_effective_inhibition;
    std::vector<double> v_effective_inhibition;
    std::vector<double> recurrent_excitation;
    for (const double source : source_coordinates) {
      a_excitation.push_back(
          std::exp(-0.5 * std::pow((source - center) / 8.0, 2)));
      v_excitation.push_back(
          std::exp(-0.5 * std::pow((source - center) / 3.0, 2)));
      recurrent_excitation.push_back(
          std::exp(-0.5 * std::pow((source - center) / 5.0, 2)));
      double a_via_inhibitory_population = 0.0;
      double v_via_inhibitory_population = 0.0;
      for (const double inhibitory : inhibitory_coordinates) {
        const double inhibitory_to_excitatory =
            std::exp(-0.5 *
                     std::pow((inhibitory - center) / 5.0, 2));
        const double auditory_to_inhibitory =
            std::exp(-0.5 *
                     std::pow((source - inhibitory) / 8.0, 2));
        const double visual_to_inhibitory =
            std::exp(-0.5 *
                     std::pow((source - inhibitory) / 3.0, 2));
        a_via_inhibitory_population +=
            auditory_to_inhibitory * inhibitory_to_excitatory;
        v_via_inhibitory_population +=
            visual_to_inhibitory * inhibitory_to_excitatory;
      }
      a_effective_inhibition.push_back(
          a_via_inhibitory_population);
      v_effective_inhibition.push_back(
          v_via_inhibitory_population);
    }
    excitation_a_centers.push_back(
        weighted_center_width(source_coordinates, a_excitation).first);
    excitation_v_centers.push_back(
        weighted_center_width(source_coordinates, v_excitation).first);
    inhibition_a_centers.push_back(
        weighted_center_width(
            source_coordinates, a_effective_inhibition).first);
    inhibition_v_centers.push_back(
        weighted_center_width(
            source_coordinates, v_effective_inhibition).first);
    recurrent_centers.push_back(
        weighted_center_width(
            source_coordinates, recurrent_excitation).first);
  }
  REQUIRE(pearson_double(target_centers, excitation_a_centers) >= 0.999);
  REQUIRE(pearson_double(target_centers, excitation_v_centers) >= 0.999);
  REQUIRE(pearson_double(target_centers, inhibition_a_centers) >= 0.999);
  REQUIRE(pearson_double(target_centers, inhibition_v_centers) >= 0.999);
  REQUIRE(pearson_double(target_centers, recurrent_centers) >= 0.999);
  std::vector<double> av_mismatch;
  for (std::size_t index = 0; index < target_centers.size(); ++index) {
    av_mismatch.push_back(std::fabs(
        excitation_a_centers[index] - excitation_v_centers[index]));
  }
  REQUIRE(median_double(std::move(av_mismatch)) <= 0.5);
}

enum class SyntheticProjection : int {
  kAeAm = 0,
  kAeNm = 1,
  kVeAm = 2,
  kVeNm = 3,
  kEeAm = 4,
  kEeNm = 5,
  kAiAm = 6,
  kAiNm = 7,
  kViAm = 8,
  kViNm = 9,
  kEiAm = 10,
  kEiNm = 11,
  kIeGaba = 12,
};

struct SyntheticControlState {
  std::array<float, 13> scale{};
  std::array<bool, 13> row_shuffled{};
};

SyntheticControlState synthetic_control_state(CausalControl control) {
  SyntheticControlState state{};
  state.scale.fill(1.0f);
  state.row_shuffled.fill(false);
  const auto disable = [&](SyntheticProjection projection) {
    state.scale[static_cast<std::size_t>(projection)] = 0.0f;
  };
  if (clean_msi::has_control(control, CausalControl::kNmdaOff)) {
    for (const SyntheticProjection projection :
         {SyntheticProjection::kAeNm, SyntheticProjection::kVeNm,
          SyntheticProjection::kEeNm, SyntheticProjection::kAiNm,
          SyntheticProjection::kViNm, SyntheticProjection::kEiNm}) {
      disable(projection);
    }
  }
  if (clean_msi::has_control(control, CausalControl::kGabaaOff)) {
    disable(SyntheticProjection::kIeGaba);
  }
  if (clean_msi::has_control(
          control, CausalControl::kRecurrentExcitationOff)) {
    disable(SyntheticProjection::kEeAm);
    disable(SyntheticProjection::kEeNm);
  }
  if (clean_msi::has_control(
          control, CausalControl::kRecruitedInhibitionOff)) {
    for (const SyntheticProjection projection :
         {SyntheticProjection::kAiAm, SyntheticProjection::kAiNm,
          SyntheticProjection::kViAm, SyntheticProjection::kViNm,
          SyntheticProjection::kEiAm, SyntheticProjection::kEiNm,
          SyntheticProjection::kIeGaba}) {
      disable(projection);
    }
  }
  if (clean_msi::has_control(
          control, CausalControl::kSourceRowShuffle)) {
    for (const SyntheticProjection projection :
         {SyntheticProjection::kAeAm, SyntheticProjection::kAeNm,
          SyntheticProjection::kVeAm, SyntheticProjection::kVeNm,
          SyntheticProjection::kAiAm, SyntheticProjection::kAiNm,
          SyntheticProjection::kViAm, SyntheticProjection::kViNm}) {
      state.row_shuffled[static_cast<std::size_t>(projection)] = true;
    }
  }
  return state;
}

void test_stage3_causal_control_mapping() {
  const SyntheticControlState baseline =
      synthetic_control_state(CausalControl::kNone);
  for (const float scale : baseline.scale) {
    REQUIRE(scale == 1.0f);
  }
  for (const bool shuffled : baseline.row_shuffled) {
    REQUIRE(!shuffled);
  }
  const std::array<CausalControl, 5> controls{
      CausalControl::kNmdaOff,
      CausalControl::kGabaaOff,
      CausalControl::kRecurrentExcitationOff,
      CausalControl::kRecruitedInhibitionOff,
      CausalControl::kSourceRowShuffle,
  };
  const std::array<std::array<bool, 13>, 5> expected_disabled{{
      {{false, true, false, true, false, true, false, true, false,
        true, false, true, false}},
      {{false, false, false, false, false, false, false, false, false,
        false, false, false, true}},
      {{false, false, false, false, true, true, false, false, false,
        false, false, false, false}},
      {{false, false, false, false, false, false, true, true, true,
        true, true, true, true}},
      {{false, false, false, false, false, false, false, false, false,
        false, false, false, false}},
  }};
  const std::array<std::array<bool, 13>, 5> expected_shuffled{{
      {{false, false, false, false, false, false, false, false, false,
        false, false, false, false}},
      {{false, false, false, false, false, false, false, false, false,
        false, false, false, false}},
      {{false, false, false, false, false, false, false, false, false,
        false, false, false, false}},
      {{false, false, false, false, false, false, false, false, false,
        false, false, false, false}},
      {{true, true, true, true, false, false, true, true, true, true,
        false, false, false}},
  }};
  const std::array<float, 13> immutable_weights{
      0.5f,  1.5f,  2.5f,  3.5f,  4.5f,  5.5f,  6.5f,
      7.5f,  8.5f,  9.5f,  10.5f, 11.5f, 12.5f};
  for (std::size_t control_index = 0;
       control_index < controls.size(); ++control_index) {
    const SyntheticControlState active =
        synthetic_control_state(controls[control_index]);
    for (std::size_t projection = 0;
         projection < active.scale.size(); ++projection) {
      REQUIRE((active.scale[projection] == 0.0f) ==
              expected_disabled[control_index][projection]);
      REQUIRE(active.row_shuffled[projection] ==
              expected_shuffled[control_index][projection]);
    }
    REQUIRE(immutable_weights ==
            (std::array<float, 13>{
                0.5f,  1.5f,  2.5f,  3.5f,  4.5f,  5.5f,  6.5f,
                7.5f,  8.5f,  9.5f,  10.5f, 11.5f, 12.5f}));
    REQUIRE(active.scale != baseline.scale ||
            active.row_shuffled != baseline.row_shuffled);
    const SyntheticControlState restored =
        synthetic_control_state(CausalControl::kNone);
    REQUIRE(restored.scale == baseline.scale);
    REQUIRE(restored.row_shuffled == baseline.row_shuffled);
  }
  const SyntheticControlState combined = synthetic_control_state(
      CausalControl::kNmdaOff | CausalControl::kGabaaOff);
  REQUIRE(combined.scale[
              static_cast<std::size_t>(SyntheticProjection::kAeNm)] ==
          0.0f);
  REQUIRE(combined.scale[
              static_cast<std::size_t>(SyntheticProjection::kIeGaba)] ==
          0.0f);
  REQUIRE(combined.scale[
              static_cast<std::size_t>(SyntheticProjection::kAeAm)] ==
          1.0f);
}

struct SyntheticSeedSummary {
  int seed_count = 0;
  int positive_count = 0;
  int negative_count = 0;
  int zero_count = 0;
  double mean = 0.0;
  double ci95_low = 0.0;
  double ci95_high = 0.0;
};

SyntheticSeedSummary summarize_twelve_seed_effects(
    const std::vector<double>& per_seed_effects) {
  REQUIRE(per_seed_effects.size() == 12u);
  SyntheticSeedSummary summary{};
  summary.seed_count = static_cast<int>(per_seed_effects.size());
  for (const double effect : per_seed_effects) {
    REQUIRE(std::isfinite(effect));
    summary.mean += effect /
                    static_cast<double>(per_seed_effects.size());
    if (effect > 0.0) {
      ++summary.positive_count;
    } else if (effect < 0.0) {
      ++summary.negative_count;
    } else {
      ++summary.zero_count;
    }
  }
  double sum_squared = 0.0;
  for (const double effect : per_seed_effects) {
    const double centered = effect - summary.mean;
    sum_squared += centered * centered;
  }
  const double sample_standard_deviation =
      std::sqrt(sum_squared /
                static_cast<double>(per_seed_effects.size() - 1u));
  const double standard_error =
      sample_standard_deviation /
      std::sqrt(static_cast<double>(per_seed_effects.size()));
  constexpr double kStudentT95Df11 = 2.2009851601;
  summary.ci95_low =
      summary.mean - kStudentT95Df11 * standard_error;
  summary.ci95_high =
      summary.mean + kStudentT95Df11 * standard_error;
  return summary;
}

void test_stage3_synthetic_cohort_control_and_seed_directions() {
  constexpr int kSyntheticSamples = 400;
  std::vector<double> auditory_common;
  std::vector<double> visual_common;
  std::vector<double> auditory_shuffle;
  std::vector<double> visual_shuffle;
  auditory_common.reserve(kSyntheticSamples);
  visual_common.reserve(kSyntheticSamples);
  auditory_shuffle.reserve(kSyntheticSamples);
  visual_shuffle.reserve(kSyntheticSamples);
  for (int sample = 0; sample < kSyntheticSamples; ++sample) {
    const double angle =
        2.0 * std::acos(-1.0) * sample / kSyntheticSamples;
    const double latent = std::sin(angle);
    const double orthogonal = std::cos(angle);
    auditory_common.push_back(latent);
    visual_common.push_back(latent);
    auditory_shuffle.push_back(latent);
    visual_shuffle.push_back(orthogonal);
  }
  REQUIRE(pearson_double(auditory_common, visual_common) > 0.999999);
  REQUIRE(std::fabs(
              pearson_double(auditory_shuffle, visual_shuffle)) <
          1.0e-12);
  Moments auditory_common_marginal;
  Moments visual_common_marginal;
  Moments auditory_shuffle_marginal;
  Moments visual_shuffle_marginal;
  for (int sample = 0; sample < kSyntheticSamples; ++sample) {
    auditory_common_marginal.add(
        auditory_common[static_cast<std::size_t>(sample)]);
    visual_common_marginal.add(
        visual_common[static_cast<std::size_t>(sample)]);
    auditory_shuffle_marginal.add(
        auditory_shuffle[static_cast<std::size_t>(sample)]);
    visual_shuffle_marginal.add(
        visual_shuffle[static_cast<std::size_t>(sample)]);
  }
  REQUIRE(std::fabs(auditory_common_marginal.mean() -
                    auditory_shuffle_marginal.mean()) <
          1.0e-12);
  REQUIRE(std::fabs(visual_common_marginal.mean() -
                    visual_shuffle_marginal.mean()) <
          1.0e-12);
  REQUIRE(std::fabs(
              auditory_common_marginal.standard_deviation() -
              auditory_shuffle_marginal.standard_deviation()) <
          1.0e-12);
  REQUIRE(std::fabs(
              visual_common_marginal.standard_deviation() -
              visual_shuffle_marginal.standard_deviation()) <
          1.0e-12);

  for (const double latent : auditory_common) {
    const double positive_a = latent + 10.0;
    const double positive_v = latent - 10.0;
    const double negative_a = latent - 10.0;
    const double negative_v = latent + 10.0;
    REQUIRE(std::fabs((positive_a - positive_v) - 20.0) <
            1.0e-12);
    REQUIRE(std::fabs((negative_a - negative_v) + 20.0) <
            1.0e-12);
  }

  const std::array<std::array<double, 2>, 4> jitter_scales{{
      {{8.0, 2.0}},
      {{0.0, 0.0}},
      {{4.0, 1.0}},
      {{16.0, 4.0}},
  }};
  constexpr std::array<double, 2> kAfferentProfileWidths{8.0, 2.0};
  for (const auto& jitter : jitter_scales) {
    REQUIRE(kAfferentProfileWidths[0] == 8.0);
    REQUIRE(kAfferentProfileWidths[1] == 2.0);
    REQUIRE(jitter[0] == 4.0 * jitter[1]);
  }

  std::vector<double> signed_grid;
  std::vector<double> positive_curve;
  std::vector<double> negative_curve;
  for (int index = 0; index <= 48; ++index) {
    const double disparity = -60.0 + 2.5 * index;
    signed_grid.push_back(disparity);
    positive_curve.push_back(
        std::exp(-0.5 * std::pow((disparity - 20.0) / 7.0, 2)));
    negative_curve.push_back(
        std::exp(-0.5 * std::pow((disparity + 20.0) / 7.0, 2)));
  }
  const std::size_t positive_peak = static_cast<std::size_t>(
      std::distance(
          positive_curve.begin(),
          std::max_element(
              positive_curve.begin(), positive_curve.end())));
  const std::size_t negative_peak = static_cast<std::size_t>(
      std::distance(
          negative_curve.begin(),
          std::max_element(
              negative_curve.begin(), negative_curve.end())));
  REQUIRE(signed_grid[positive_peak] == 20.0);
  REQUIRE(signed_grid[negative_peak] == -20.0);

  struct DirectionalMetrics {
    double temporal_peak_minus_tail;
    double temporal_tail;
    double spatial_center_minus_large;
    double spatial_large_disparity;
    double additivity_nonlinearity;
    double persistence_contrast;
    double functional_order;
  };
  const DirectionalMetrics baseline{
      0.70, 0.10, 0.80, 0.10, 0.60, 0.55, 0.95};
  const DirectionalMetrics nmda_off{
      0.40, 0.10, 0.65, 0.10, 0.20, 0.30, 0.90};
  const DirectionalMetrics gabaa_off{
      0.30, 0.45, 0.35, 0.50, 0.55, 0.50, 0.90};
  const DirectionalMetrics recruited_inhibition_off{
      0.35, 0.40, 0.40, 0.45, 0.55, 0.50, 0.90};
  const DirectionalMetrics recurrence_off{
      0.42, 0.12, 0.55, 0.12, 0.45, 0.20, 0.90};
  const DirectionalMetrics source_row_shuffle{
      0.65, 0.12, 0.25, 0.12, 0.50, 0.48, 0.10};
  REQUIRE(nmda_off.temporal_peak_minus_tail <
          baseline.temporal_peak_minus_tail);
  REQUIRE(nmda_off.additivity_nonlinearity <
          baseline.additivity_nonlinearity);
  for (const DirectionalMetrics* inhibition_off :
       {&gabaa_off, &recruited_inhibition_off}) {
    REQUIRE(inhibition_off->temporal_tail >
            baseline.temporal_tail);
    REQUIRE(inhibition_off->spatial_large_disparity >
            baseline.spatial_large_disparity);
    REQUIRE(inhibition_off->temporal_peak_minus_tail <
            baseline.temporal_peak_minus_tail);
  }
  REQUIRE(recurrence_off.temporal_peak_minus_tail <
          baseline.temporal_peak_minus_tail);
  REQUIRE(recurrence_off.persistence_contrast <
          baseline.persistence_contrast);
  REQUIRE(source_row_shuffle.spatial_center_minus_large <
          baseline.spatial_center_minus_large);
  REQUIRE(source_row_shuffle.functional_order <
          baseline.functional_order);

  std::vector<double> seed_effects;
  std::vector<double> flattened_trials;
  seed_effects.reserve(12);
  seed_effects.push_back(1.0);
  flattened_trials.insert(flattened_trials.end(), 100u, 1.0);
  for (int seed = 1; seed < 12; ++seed) {
    seed_effects.push_back(-1.0);
    flattened_trials.push_back(-1.0);
  }
  const SyntheticSeedSummary inference =
      summarize_twelve_seed_effects(seed_effects);
  const double pooled_trial_mean =
      std::accumulate(
          flattened_trials.begin(), flattened_trials.end(), 0.0) /
      static_cast<double>(flattened_trials.size());
  REQUIRE(inference.seed_count == 12);
  REQUIRE(inference.positive_count == 1);
  REQUIRE(inference.negative_count == 11);
  REQUIRE(inference.zero_count == 0);
  REQUIRE(inference.mean < 0.0);
  REQUIRE(pooled_trial_mean > 0.0);
  REQUIRE(inference.ci95_high < 0.0);
  std::array<int, 12> merged_seed_ids{};
  for (int local_seed = 0; local_seed < 8; ++local_seed) {
    merged_seed_ids[static_cast<std::size_t>(local_seed)] =
        local_seed;
  }
  for (int local_seed = 0; local_seed < 4; ++local_seed) {
    merged_seed_ids[static_cast<std::size_t>(8 + local_seed)] =
        8 + local_seed;
  }
  for (int seed = 0; seed < 12; ++seed) {
    REQUIRE(merged_seed_ids[static_cast<std::size_t>(seed)] == seed);
  }
}

void require_exact_frozen_weight_audit(
    const std::vector<FrozenWeightAudit>& actual,
    const std::vector<FrozenWeightAudit>& expected,
    const std::string& label) {
  REQUIRE_MESSAGE(actual.size() == expected.size(),
                  label + ".seed count differs");
  for (std::size_t seed = 0; seed < actual.size(); ++seed) {
    REQUIRE_MESSAGE(actual[seed].seed_index == expected[seed].seed_index,
                    label + ".seed index differs");
    REQUIRE_MESSAGE(actual[seed].seed_index ==
                        static_cast<int>(seed),
                    label + ".seed index is not sequential");
    REQUIRE_MESSAGE(actual[seed].global_seed ==
                        expected[seed].global_seed,
                    label + ".global seed differs");
    for (std::size_t path = 0; path < actual[seed].paths.size(); ++path) {
      const auto& actual_path = actual[seed].paths[path];
      const auto& expected_path = expected[seed].paths[path];
      REQUIRE_MESSAGE(!actual_path.weights.empty(),
                      label + ".path has no weights");
      REQUIRE_MESSAGE(actual_path.weights.size() ==
                          actual_path.masks.size(),
                      label + ".weight/mask shape differs");
      REQUIRE_MESSAGE(actual_path.weights == expected_path.weights,
                      label + ".weights changed");
      REQUIRE_MESSAGE(actual_path.masks == expected_path.masks,
                      label + ".masks changed");
      for (std::size_t contact = 0;
           contact < actual_path.weights.size(); ++contact) {
        REQUIRE_MESSAGE(std::isfinite(actual_path.weights[contact]),
                        label + ".weight is non-finite");
        REQUIRE_MESSAGE(actual_path.masks[contact] <= 1u,
                        label + ".mask is not binary");
      }
    }
  }
}

void require_exact_frozen_trial(const FrozenTrialResult& actual,
                                const FrozenTrialResult& expected,
                                const std::string& label) {
  REQUIRE_MESSAGE(actual.seed_index == expected.seed_index,
                  label + ".seed_index differs");
  REQUIRE_MESSAGE(actual.condition_index == expected.condition_index,
                  label + ".condition_index differs");
  REQUIRE_MESSAGE(actual.trial_index == expected.trial_index,
                  label + ".trial_index differs");
  REQUIRE_MESSAGE(actual.label == expected.label,
                  label + ".label differs");
  REQUIRE_MESSAGE(actual.physical_soa_ms == expected.physical_soa_ms,
                  label + ".physical_soa_ms differs");
  REQUIRE_MESSAGE(actual.auditory_latency_ms ==
                      expected.auditory_latency_ms,
                  label + ".auditory_latency_ms differs");
  REQUIRE_MESSAGE(actual.visual_latency_ms ==
                      expected.visual_latency_ms,
                  label + ".visual_latency_ms differs");
  REQUIRE_MESSAGE(actual.auditory_received_onset_ms ==
                      expected.auditory_received_onset_ms,
                  label + ".auditory_received_onset_ms differs");
  REQUIRE_MESSAGE(actual.visual_received_onset_ms ==
                      expected.visual_received_onset_ms,
                  label + ".visual_received_onset_ms differs");
  REQUIRE_MESSAGE(actual.earlier_received_onset_step ==
                      expected.earlier_received_onset_step,
                  label + ".earlier_received_onset_step differs");
  REQUIRE_MESSAGE(actual.burn_in_steps == expected.burn_in_steps,
                  label + ".burn_in_steps differs");
  REQUIRE_MESSAGE(
      actual.background_afferent_arrivals_during_burn_in ==
          expected.background_afferent_arrivals_during_burn_in,
      label + ".burn-in background arrivals differ");
  REQUIRE_MESSAGE(actual.excitatory_spikes_per_relative_ms ==
                      expected.excitatory_spikes_per_relative_ms,
                  label + ".per-ms population spikes differ");
  REQUIRE_MESSAGE(actual.excitatory_baseline_spikes_per_neuron ==
                      expected.excitatory_baseline_spikes_per_neuron,
                  label + ".per-neuron baseline spikes differ");
  REQUIRE_MESSAGE(actual.excitatory_response_spikes_per_neuron ==
                      expected.excitatory_response_spikes_per_neuron,
                  label + ".per-neuron response spikes differ");
  REQUIRE_MESSAGE(actual.features == expected.features,
                  label + ".features differ");
  REQUIRE_MESSAGE(actual.baseline_rate_hz == expected.baseline_rate_hz,
                  label + ".baseline_rate_hz differs");
  REQUIRE_MESSAGE(actual.response_rate_hz == expected.response_rate_hz,
                  label + ".response_rate_hz differs");
  REQUIRE_MESSAGE(actual.finite == expected.finite,
                  label + ".finite differs");
}

void require_frozen_boundary_science(
    const FrozenTrialBatch& batch, const ControlledCondition& condition,
    const std::string& label) {
  REQUIRE_MESSAGE(batch.model_seed_count == 1,
                  label + ".model_seed_count");
  REQUIRE_MESSAGE(batch.condition_count == 1,
                  label + ".condition_count");
  REQUIRE_MESSAGE(batch.trials_per_condition == 1,
                  label + ".trials_per_condition");
  REQUIRE_MESSAGE(batch.burn_in_steps == 300,
                  label + ".batch burn-in is not 300 ms");
  REQUIRE_MESSAGE(batch.trials.size() == 1u,
                  label + ".trial count");
  REQUIRE_MESSAGE(std::isfinite(batch.kernel_seconds) &&
                      batch.kernel_seconds > 0.0f,
                  label + ".kernel timing is invalid");
  const auto expected_edges = observer_bin_edges_ms();
  REQUIRE_MESSAGE(batch.relative_bin_edges_ms == expected_edges,
                  label + ".relative feature-bin edges differ");

  const FrozenTrialResult& trial = batch.trials.front();
  REQUIRE_MESSAGE(trial.seed_index == 0, label + ".seed index");
  REQUIRE_MESSAGE(trial.condition_index == 0,
                  label + ".condition index");
  REQUIRE_MESSAGE(trial.trial_index == 0, label + ".trial index");
  REQUIRE_MESSAGE(trial.label == condition.label, label + ".label");
  REQUIRE_MESSAGE(trial.physical_soa_ms == condition.physical_soa_ms,
                  label + ".physical SOA");
  REQUIRE_MESSAGE(trial.burn_in_steps == 300,
                  label + ".trial burn-in is not 300 ms");
  REQUIRE_MESSAGE(trial.earlier_received_onset_step == 400,
                  label + ".relative window is not after burn-in");
  REQUIRE_MESSAGE(
      trial.background_afferent_arrivals_during_burn_in > 0,
      label + ".background afferents were inactive during burn-in");
  REQUIRE_MESSAGE(trial.finite, label + ".trial is non-finite");
  REQUIRE_MESSAGE(std::isfinite(trial.auditory_latency_ms) &&
                      trial.auditory_latency_ms > 0.0f,
                  label + ".auditory latency");
  REQUIRE_MESSAGE(std::isfinite(trial.visual_latency_ms) &&
                      trial.visual_latency_ms > 0.0f,
                  label + ".visual latency");
  REQUIRE_MESSAGE(trial.auditory_received_onset_ms >= 0,
                  label + ".auditory received onset");
  REQUIRE_MESSAGE(trial.visual_received_onset_ms >= 0,
                  label + ".visual received onset");
  REQUIRE_MESSAGE(
      std::min(trial.auditory_received_onset_ms,
               trial.visual_received_onset_ms) == 0,
      label + ".received onsets are not relative to the earlier input");

  unsigned int total_per_ms = 0u;
  for (const std::uint16_t count :
       trial.excitatory_spikes_per_relative_ms) {
    REQUIRE_MESSAGE(count <= clean_msi::kExcitatoryNeurons,
                    label + ".population spike count exceeds neurons");
    total_per_ms += count;
  }
  unsigned int total_features = 0u;
  for (int feature = 0; feature < clean_msi::kEvaluationFeatureBins;
       ++feature) {
    unsigned int expected = 0u;
    for (int offset = 0; offset < clean_msi::kEvaluationBinWidthMs;
         ++offset) {
      expected += trial.excitatory_spikes_per_relative_ms[
          static_cast<std::size_t>(
              feature * clean_msi::kEvaluationBinWidthMs + offset)];
    }
    REQUIRE_MESSAGE(
        trial.features[static_cast<std::size_t>(feature)] ==
            static_cast<float>(expected),
        label + ".20 ms feature does not equal its per-ms sum");
    total_features += expected;
  }
  REQUIRE_MESSAGE(total_features == total_per_ms,
                  label + ".feature bins do not conserve spike count");

  const unsigned int baseline_population = std::accumulate(
      trial.excitatory_spikes_per_relative_ms.begin(),
      trial.excitatory_spikes_per_relative_ms.begin() + 100, 0u);
  const unsigned int response_population = std::accumulate(
      trial.excitatory_spikes_per_relative_ms.begin() + 100,
      trial.excitatory_spikes_per_relative_ms.begin() + 350, 0u);
  const unsigned int baseline_neurons = std::accumulate(
      trial.excitatory_baseline_spikes_per_neuron.begin(),
      trial.excitatory_baseline_spikes_per_neuron.end(), 0u);
  const unsigned int response_neurons = std::accumulate(
      trial.excitatory_response_spikes_per_neuron.begin(),
      trial.excitatory_response_spikes_per_neuron.end(), 0u);
  REQUIRE_MESSAGE(baseline_neurons == baseline_population,
                  label + ".baseline population/neuron counts differ");
  REQUIRE_MESSAGE(response_neurons == response_population,
                  label + ".response population/neuron counts differ");
  const float expected_baseline_hz =
      1000.0f * static_cast<float>(baseline_population) /
      (100.0f * clean_msi::kExcitatoryNeurons);
  const float expected_response_hz =
      1000.0f * static_cast<float>(response_population) /
      (250.0f * clean_msi::kExcitatoryNeurons);
  require_near(trial.baseline_rate_hz, expected_baseline_hz,
               1.0e-6f, 1.0e-6f, label + ".baseline rate");
  require_near(trial.response_rate_hz, expected_response_hz,
               1.0e-6f, 1.0e-6f, label + ".response rate");
}

void test_stage3_frozen_boundary_cuda_replay_and_neutrality() {
  const std::vector<int> devices = available_devices();
  REQUIRE_MESSAGE(devices.size() == 2u,
                  "frozen boundary requires cuda0 and cuda1");
  ControlledCondition condition{};
  condition.auditory_present = true;
  condition.visual_present = true;
  condition.auditory_location_deg = -12.0f;
  condition.visual_location_deg = 14.0f;
  condition.auditory_rate_hz = 50.0f;
  condition.visual_rate_hz = 50.0f;
  condition.physical_soa_ms = -50.0f;
  condition.control = CausalControl::kNone;
  condition.label = 1;
  condition.random_group = 23;
  constexpr std::uint64_t kEvaluationSeed = 0x51A6E3ull;

  for (const int device : devices) {
    const Config config = calibration_for(device).result.config;
    NativeModel model(config, device, 1);
    const auto before = model.frozen_weight_audit();
    require_exact_frozen_weight_audit(
        before, before, "cuda" + std::to_string(device) + ".before");
    const FrozenTrialBatch first = model.run_frozen_trials(
        {condition}, 1, kEvaluationSeed, 300);
    require_frozen_boundary_science(
        first, condition, "cuda" + std::to_string(device) + ".first");
    const auto after_first = model.frozen_weight_audit();
    require_exact_frozen_weight_audit(
        after_first, before,
        "cuda" + std::to_string(device) + ".after_first");
    const FrozenTrialBatch replay = model.run_frozen_trials(
        {condition}, 1, kEvaluationSeed, 300);
    require_frozen_boundary_science(
        replay, condition,
        "cuda" + std::to_string(device) + ".replay");
    require_exact_frozen_trial(
        replay.trials.front(), first.trials.front(),
        "cuda" + std::to_string(device) + ".replay");
    const auto after_replay = model.frozen_weight_audit();
    require_exact_frozen_weight_audit(
        after_replay, before,
        "cuda" + std::to_string(device) + ".after_replay");
    std::cout << "[EVALUATION] cuda" << device
              << " one_trial_kernel_s=" << std::fixed
              << std::setprecision(6) << first.kernel_seconds
              << " replay_kernel_s=" << replay.kernel_seconds << '\n';
  }
}

void test_stage3_background_active_during_frozen_burnin() {
  const std::vector<int> devices = available_devices();
  REQUIRE_MESSAGE(!devices.empty(),
                  "burn-in background audit requires CUDA");
  ControlledCondition condition{};
  condition.auditory_present = true;
  condition.visual_present = false;
  condition.auditory_location_deg = 0.0f;
  condition.auditory_rate_hz = 50.0f;
  condition.physical_soa_ms = 0.0f;
  condition.random_group = 9123;
  const Config config = calibration_for(0).result.config;
  NativeModel model(config, 0, 1);
  const FrozenTrialBatch batch =
      model.run_frozen_trials(
          {condition}, 1, 0x4255524E494Eull, 300);
  REQUIRE(batch.trials.size() == 1u);
  const FrozenTrialResult& trial = batch.trials.front();
  REQUIRE_MESSAGE(
      trial.background_afferent_arrivals_during_burn_in > 0,
      "frozen burn-in contained no background afferent arrivals");
  REQUIRE_MESSAGE(
      trial.earlier_received_onset_step == 400,
      "background burn-in correction changed sensory timing");
  REQUIRE_MESSAGE(
      trial.auditory_received_onset_ms == 0,
      "background burn-in correction changed the received onset");
}

void require_component_response_algebra(
    const clean_msi::ComponentResponseMetrics& response,
    float response_floor_hz, const std::string& label) {
  require_finite(response.auditory_rate_hz, label + ".auditory_rate_hz");
  require_finite(response.visual_rate_hz, label + ".visual_rate_hz");
  require_finite(
      response.audiovisual_rate_hz, label + ".audiovisual_rate_hz");
  require_finite(
      response.raw_enhancement_hz, label + ".raw_enhancement_hz");
  require_finite(
      response.raw_enhancement_sem, label + ".raw_enhancement_sem");
  require_finite(
      response.multisensory_enhancement_percent,
      label + ".multisensory_enhancement_percent");
  require_finite(response.additivity_hz, label + ".additivity_hz");
  require_finite(
      response.additivity_percent, label + ".additivity_percent");
  REQUIRE_MESSAGE(response.auditory_rate_hz >= 0.0f,
                  label + ".auditory rate is negative");
  REQUIRE_MESSAGE(response.visual_rate_hz >= 0.0f,
                  label + ".visual rate is negative");
  REQUIRE_MESSAGE(response.audiovisual_rate_hz >= 0.0f,
                  label + ".audiovisual rate is negative");
  REQUIRE_MESSAGE(response.raw_enhancement_sem >= 0.0f,
                  label + ".enhancement SEM is negative");
  const float expected_gain =
      response.audiovisual_rate_hz -
      std::max(response.auditory_rate_hz, response.visual_rate_hz);
  const float expected_me =
      100.0f * expected_gain /
      std::max(
          std::max(response.auditory_rate_hz,
                   response.visual_rate_hz),
          response_floor_hz);
  const float expected_additivity =
      response.audiovisual_rate_hz -
      (response.auditory_rate_hz + response.visual_rate_hz);
  const float expected_additivity_percent =
      100.0f * expected_additivity /
      std::max(response.auditory_rate_hz + response.visual_rate_hz,
               response_floor_hz);
  require_near(response.raw_enhancement_hz, expected_gain,
               1.0e-6f, 1.0e-6f, label + ".raw gain algebra");
  require_near(response.multisensory_enhancement_percent, expected_me,
               1.0e-5f, 1.0e-6f, label + ".ME algebra");
  require_near(response.additivity_hz, expected_additivity,
               1.0e-6f, 1.0e-6f, label + ".additivity algebra");
  require_near(response.additivity_percent,
               expected_additivity_percent, 1.0e-5f, 1.0e-6f,
               label + ".additivity-percent algebra");
}

void test_stage3_small_production_evaluation_mechanics() {
  const std::vector<int> devices = available_devices();
  REQUIRE_MESSAGE(!devices.empty(),
                  "evaluation mechanics requires CUDA");
  constexpr int kDevice = 0;
  EvaluationOptions options{};
  options.observer_training_trials = 4;
  options.observer_holdout_trials = 4;
  options.trials_per_tbw_soa = 1;
  options.trials_per_sbw_condition = 1;
  options.trials_per_rf_location = 1;
  options.trials_per_inverse_condition = 1;
  options.burn_in_ms = 300;
  options.evaluation_seed = 0x4D454348414E4943ull;
  options.observer_rate_hz = 50.0f;
  options.response_floor_hz = 1.0f;
  options.control = CausalControl::kNone;

  const Config config = calibration_for(kDevice).result.config;
  NativeModel model(config, kDevice, 1);
  const auto before = model.frozen_weight_audit();
  EvaluationMetrics metrics{};
  const double wall_seconds = timed_seconds(kDevice, [&] {
    metrics = model.evaluate(options);
  });
  const auto after = model.frozen_weight_audit();
  require_exact_frozen_weight_audit(
      after, before, "cuda0.small_evaluation.neutrality");

  const auto& observer = metrics.observer;
  REQUIRE_MESSAGE(observer.feature_mean.size() ==
                      clean_msi::kEvaluationFeatureBins,
                  "observer feature mean contains non-spike metadata");
  REQUIRE_MESSAGE(observer.feature_std.size() ==
                      clean_msi::kEvaluationFeatureBins,
                  "observer feature scale contains non-spike metadata");
  REQUIRE_MESSAGE(observer.coefficients.size() ==
                      clean_msi::kEvaluationFeatureBins,
                  "observer coefficients contain non-spike metadata");
  REQUIRE_MESSAGE(observer.training_labels.size() == 4u,
                  "observer training label count");
  REQUIRE_MESSAGE(observer.training_probabilities.size() == 4u,
                  "observer training probability count");
  REQUIRE_MESSAGE(observer.holdout_labels.size() == 4u,
                  "observer holdout label count");
  REQUIRE_MESSAGE(observer.holdout_probabilities.size() == 4u,
                  "observer holdout probability count");
  REQUIRE_MESSAGE(
      std::count(observer.training_labels.begin(),
                 observer.training_labels.end(), 0u) == 2 &&
          std::count(observer.training_labels.begin(),
                     observer.training_labels.end(), 1u) == 2,
      "observer training labels are not balanced");
  REQUIRE_MESSAGE(
      std::count(observer.holdout_labels.begin(),
                 observer.holdout_labels.end(), 0u) == 2 &&
          std::count(observer.holdout_labels.begin(),
                     observer.holdout_labels.end(), 1u) == 2,
      "observer holdout labels are not balanced");
  for (std::size_t feature = 0;
       feature < observer.coefficients.size(); ++feature) {
    require_finite(
        observer.feature_mean[feature],
        "observer.feature_mean[" + std::to_string(feature) + "]");
    require_finite(
        observer.feature_std[feature],
        "observer.feature_std[" + std::to_string(feature) + "]");
    require_finite(
        observer.coefficients[feature],
        "observer.coefficients[" + std::to_string(feature) + "]");
    REQUIRE_MESSAGE(observer.feature_std[feature] > 0.0f,
                    "observer feature scale is not positive");
  }
  for (const auto* probabilities :
       {&observer.training_probabilities,
        &observer.holdout_probabilities}) {
    for (const float probability : *probabilities) {
      REQUIRE_MESSAGE(std::isfinite(probability) &&
                          probability >= 0.0f &&
                          probability <= 1.0f,
                      "observer probability is outside [0,1]");
    }
  }
  require_finite(observer.intercept, "observer.intercept");
  require_finite(observer.l2, "observer.l2");
  require_finite(observer.auroc, "observer.auroc");
  require_finite(observer.brier, "observer.brier");
  require_finite(
      observer.calibration_intercept,
      "observer.calibration_intercept");
  require_finite(
      observer.calibration_slope, "observer.calibration_slope");
  require_finite(
      observer.expected_calibration_error,
      "observer.expected_calibration_error");
  REQUIRE(observer.l2 > 0.0f);
  REQUIRE(observer.auroc >= 0.0f && observer.auroc <= 1.0f);
  REQUIRE(observer.brier >= 0.0f && observer.brier <= 1.0f);
  REQUIRE(observer.expected_calibration_error >= 0.0f &&
          observer.expected_calibration_error <= 1.0f);

  REQUIRE(metrics.temporal.trials_per_soa == 1);
  float peak_probability = 0.0f;
  for (int point = 0; point < clean_msi::kTbwPointCount; ++point) {
    const float expected_soa = -500.0f + 25.0f * point;
    const std::size_t index = static_cast<std::size_t>(point);
    REQUIRE(metrics.temporal.physical_soa_grid_ms[index] ==
            expected_soa);
    const auto& value = metrics.temporal.points[index];
    REQUIRE(value.physical_soa_ms == expected_soa);
    require_finite(
        value.probability_mean,
        "temporal.probability_mean[" + std::to_string(point) + "]");
    require_finite(
        value.probability_sem,
        "temporal.probability_sem[" + std::to_string(point) + "]");
    require_finite(
        value.auditory_rate_hz,
        "temporal.auditory_rate_hz[" + std::to_string(point) + "]");
    require_finite(
        value.visual_rate_hz,
        "temporal.visual_rate_hz[" + std::to_string(point) + "]");
    require_finite(
        value.audiovisual_rate_hz,
        "temporal.audiovisual_rate_hz[" +
            std::to_string(point) + "]");
    require_finite(
        value.raw_neural_enhancement_hz,
        "temporal.raw_enhancement[" + std::to_string(point) + "]");
    REQUIRE(value.probability_mean >= 0.0f &&
            value.probability_mean <= 1.0f);
    REQUIRE(value.probability_sem >= 0.0f);
    REQUIRE(value.auditory_rate_hz >= 0.0f);
    REQUIRE(value.visual_rate_hz >= 0.0f);
    REQUIRE(value.audiovisual_rate_hz >= 0.0f);
    require_near(
        value.raw_neural_enhancement_hz,
        value.audiovisual_rate_hz -
            std::max(value.auditory_rate_hz, value.visual_rate_hz),
        1.0e-6f, 1.0e-6f,
        "temporal raw enhancement algebra");
    peak_probability = std::max(peak_probability,
                                value.probability_mean);
  }
  const float left_tail_baseline =
      (metrics.temporal.points[0].probability_mean +
       metrics.temporal.points[1].probability_mean +
       metrics.temporal.points[2].probability_mean) /
      3.0f;
  const float right_tail_baseline =
      (metrics.temporal.points[clean_msi::kTbwPointCount - 1]
           .probability_mean +
       metrics.temporal.points[clean_msi::kTbwPointCount - 2]
           .probability_mean +
       metrics.temporal.points[clean_msi::kTbwPointCount - 3]
           .probability_mean) /
      3.0f;
  const float expected_peak_minus_tail =
      peak_probability -
      0.5f * (left_tail_baseline + right_tail_baseline);
  require_near(
      metrics.temporal.peak_minus_tail, expected_peak_minus_tail,
      1.0e-6f, 1.0e-6f, "temporal peak-minus-tail algebra");
  REQUIRE(metrics.tbw50_ms == metrics.temporal.empirical.tbw50_ms);
  REQUIRE(metrics.tbw75_ms == metrics.temporal.empirical.tbw75_ms);

  REQUIRE(metrics.spatial.trials_per_orientation == 1);
  REQUIRE(metrics.spatial.response_floor_hz ==
          options.response_floor_hz);
  for (int point = 0; point < clean_msi::kSbwPointCount; ++point) {
    const float expected_disparity = 2.5f * point;
    const std::size_t index = static_cast<std::size_t>(point);
    REQUIRE(metrics.spatial.disparity_grid_deg[index] ==
            expected_disparity);
    const auto& value = metrics.spatial.points[index];
    REQUIRE(value.disparity_deg == expected_disparity);
    for (std::size_t orientation = 0;
         orientation < value.orientations.size(); ++orientation) {
      require_component_response_algebra(
          value.orientations[orientation], options.response_floor_hz,
          "spatial[" + std::to_string(point) + "].orientation[" +
              std::to_string(orientation) + "]");
    }
    require_near(
        value.pooled.auditory_rate_hz,
        0.5f * (value.orientations[0].auditory_rate_hz +
                value.orientations[1].auditory_rate_hz),
        1.0e-6f, 1.0e-6f, "pooled auditory rate");
    require_near(
        value.pooled.visual_rate_hz,
        0.5f * (value.orientations[0].visual_rate_hz +
                value.orientations[1].visual_rate_hz),
        1.0e-6f, 1.0e-6f, "pooled visual rate");
    require_near(
        value.pooled.audiovisual_rate_hz,
        0.5f * (value.orientations[0].audiovisual_rate_hz +
                value.orientations[1].audiovisual_rate_hz),
        1.0e-6f, 1.0e-6f, "pooled audiovisual rate");
    require_near(
        value.pooled.raw_enhancement_hz,
        0.5f *
            (value.orientations[0].raw_enhancement_hz +
             value.orientations[1].raw_enhancement_hz),
        1.0e-6f, 1.0e-6f, "pooled matched gain");
    require_near(
        value.pooled.multisensory_enhancement_percent,
        0.5f *
            (value.orientations[0]
                 .multisensory_enhancement_percent +
             value.orientations[1]
                 .multisensory_enhancement_percent),
        1.0e-6f, 1.0e-6f, "pooled matched ME");
    require_near(
        value.pooled.additivity_hz,
        0.5f *
            (value.orientations[0].additivity_hz +
             value.orientations[1].additivity_hz),
        1.0e-6f, 1.0e-6f, "pooled matched additivity");
    require_near(
        value.pooled.additivity_percent,
        0.5f *
            (value.orientations[0].additivity_percent +
             value.orientations[1].additivity_percent),
        1.0e-6f, 1.0e-6f,
        "pooled matched additivity percent");
  }

  const auto& rf = metrics.rf_topography;
  REQUIRE(rf.trials_per_location == 1);
  for (int point = 0; point < clean_msi::kRfPointCount; ++point) {
    const float expected_location = -60.0f + 5.0f * point;
    REQUIRE(rf.location_grid_deg[static_cast<std::size_t>(point)] ==
            expected_location);
  }
  for (int neuron = 0; neuron < clean_msi::kExcitatoryNeurons;
       ++neuron) {
    const std::size_t index = static_cast<std::size_t>(neuron);
    for (int point = 0; point < clean_msi::kRfPointCount; ++point) {
      const std::size_t location = static_cast<std::size_t>(point);
      require_finite(
          rf.auditory_response_hz[index][location],
          "rf.auditory_response_hz");
      require_finite(
          rf.visual_response_hz[index][location],
          "rf.visual_response_hz");
      REQUIRE(rf.auditory_response_hz[index][location] >= 0.0f);
      REQUIRE(rf.visual_response_hz[index][location] >= 0.0f);
    }
    for (const float value :
         {rf.auditory_rf_center_deg[index],
          rf.auditory_rf_width_deg[index],
          rf.visual_rf_center_deg[index],
          rf.visual_rf_width_deg[index],
          rf.rf_center_mismatch_deg[index],
          rf.auditory_excitatory_map_center_deg[index],
          rf.auditory_excitatory_map_width_deg[index],
          rf.visual_excitatory_map_center_deg[index],
          rf.visual_excitatory_map_width_deg[index],
          rf.auditory_effective_inhibitory_center_deg[index],
          rf.auditory_effective_inhibitory_width_deg[index],
          rf.visual_effective_inhibitory_center_deg[index],
          rf.visual_effective_inhibitory_width_deg[index]}) {
      require_finite(value, "rf per-neuron geometry");
    }
    REQUIRE(rf.auditory_rf_width_deg[index] >= 0.0f);
    REQUIRE(rf.visual_rf_width_deg[index] >= 0.0f);
    REQUIRE(rf.rf_center_mismatch_deg[index] >= 0.0f);
    REQUIRE(rf.auditory_excitatory_map_width_deg[index] >= 0.0f);
    REQUIRE(rf.visual_excitatory_map_width_deg[index] >= 0.0f);
    REQUIRE(
        rf.auditory_effective_inhibitory_width_deg[index] >= 0.0f);
    REQUIRE(
        rf.visual_effective_inhibitory_width_deg[index] >= 0.0f);
  }
  for (const float correlation :
       {rf.auditory_rf_order_correlation,
        rf.visual_rf_order_correlation,
        rf.auditory_weight_order_correlation,
        rf.visual_weight_order_correlation,
        rf.effective_inhibitory_order_correlation,
        rf.recurrent_weight_distance_correlation,
        rf.map_monotonicity}) {
    REQUIRE(std::isfinite(correlation));
    REQUIRE(correlation >= -1.0f && correlation <= 1.0f);
  }
  REQUIRE(std::isfinite(rf.median_rf_center_mismatch_deg) &&
          rf.median_rf_center_mismatch_deg >= 0.0f);
  REQUIRE(
      std::isfinite(rf.median_effective_inhibitory_alignment_deg) &&
      rf.median_effective_inhibitory_alignment_deg >= 0.0f);
  REQUIRE(std::isfinite(rf.finite_rf_coverage) &&
          rf.finite_rf_coverage >= 0.0f &&
          rf.finite_rf_coverage <= 1.0f);
  REQUIRE(metrics.auditory_topography_correlation ==
          rf.auditory_weight_order_correlation);
  REQUIRE(metrics.visual_topography_correlation ==
          rf.visual_weight_order_correlation);
  REQUIRE(metrics.inhibitory_topography_correlation ==
          rf.effective_inhibitory_order_correlation);
  REQUIRE(metrics.recurrent_topography_correlation ==
          rf.recurrent_weight_distance_correlation);

  const auto& inverse = metrics.inverse_effectiveness;
  REQUIRE(inverse.trials_per_salience == 1);
  const std::array<float, clean_msi::kInverseEffectivenessCount>
      expected_salience{25.0f, 50.0f, 100.0f};
  REQUIRE(inverse.salience_hz == expected_salience);
  for (std::size_t salience = 0;
       salience < inverse.responses.size(); ++salience) {
    require_component_response_algebra(
        inverse.responses[salience], options.response_floor_hz,
        "inverse_effectiveness[" + std::to_string(salience) + "]");
    REQUIRE(metrics.inverse_effectiveness_percent[salience] ==
            inverse.responses[salience]
                .multisensory_enhancement_percent);
  }
  REQUIRE_MESSAGE(std::isfinite(wall_seconds) && wall_seconds > 0.0,
                  "evaluation mechanics wall timing");
  std::cout << "[EVALUATION] cuda0 small_mechanics_wall_s="
            << std::fixed << std::setprecision(6) << wall_seconds
            << " observer_iterations=" << observer.iterations
            << " temporal_fit_valid=" << metrics.temporal.fit.valid
            << " spatial_fit_valid=" << metrics.spatial.fit.valid
            << '\n';
}

void require_projection_control_mapping(
    const clean_msi::ProjectionControlAudit& audit,
    CausalControl control, const std::string& label) {
  std::array<float, 7> expected_ampa{};
  std::array<float, 7> expected_nmda{};
  std::array<float, 7> expected_gabaa{};
  std::array<bool, 7> expected_shuffle{};
  for (int path = 0; path < 6; ++path) {
    expected_ampa[static_cast<std::size_t>(path)] = 1.0f;
    expected_nmda[static_cast<std::size_t>(path)] = 1.0f;
  }
  expected_gabaa[6] = 1.0f;
  bool expected_external_nmda = true;
  bool expected_background_nmda = true;
  if (clean_msi::has_control(control, CausalControl::kNmdaOff)) {
    expected_nmda.fill(0.0f);
    expected_external_nmda = false;
    expected_background_nmda = false;
  }
  if (clean_msi::has_control(control, CausalControl::kGabaaOff)) {
    expected_gabaa[6] = 0.0f;
  }
  if (clean_msi::has_control(
          control, CausalControl::kRecurrentExcitationOff)) {
    expected_ampa[2] = 0.0f;
    expected_nmda[2] = 0.0f;
  }
  if (clean_msi::has_control(
          control, CausalControl::kRecruitedInhibitionOff)) {
    for (const std::size_t path : {3u, 4u}) {
      expected_ampa[path] = 0.0f;
      expected_nmda[path] = 0.0f;
    }
    expected_gabaa[6] = 0.0f;
  }
  if (clean_msi::has_control(
          control, CausalControl::kSourceRowShuffle)) {
    for (const std::size_t path : {0u, 1u, 3u, 4u}) {
      expected_shuffle[path] = true;
    }
  }
  REQUIRE_MESSAGE(audit.control == control, label + ".control");
  REQUIRE_MESSAGE(audit.ampa_scale == expected_ampa,
                  label + ".AMPA pathway scope");
  REQUIRE_MESSAGE(audit.nmda_scale == expected_nmda,
                  label + ".NMDA pathway scope");
  REQUIRE_MESSAGE(audit.gabaa_scale == expected_gabaa,
                  label + ".GABAA pathway scope");
  REQUIRE_MESSAGE(audit.source_rows_shuffled == expected_shuffle,
                  label + ".shuffle pathway scope");
  REQUIRE_MESSAGE(
      audit.external_nmda_enabled == expected_external_nmda,
      label + ".external NMDA scope");
  REQUIRE_MESSAGE(
      audit.background_nmda_enabled == expected_background_nmda,
      label + ".background NMDA scope");
}

void require_exact_observer(
    const clean_msi::ObserverMetrics& actual,
    const clean_msi::ObserverMetrics& expected,
    const std::string& label) {
  REQUIRE_MESSAGE(actual.auroc == expected.auroc,
                  label + ".auroc differs");
  REQUIRE_MESSAGE(actual.brier == expected.brier,
                  label + ".brier differs");
  REQUIRE_MESSAGE(
      actual.calibration_intercept ==
          expected.calibration_intercept,
      label + ".calibration intercept differs");
  REQUIRE_MESSAGE(
      actual.calibration_slope == expected.calibration_slope,
      label + ".calibration slope differs");
  REQUIRE_MESSAGE(
      actual.expected_calibration_error ==
          expected.expected_calibration_error,
      label + ".calibration error differs");
  REQUIRE_MESSAGE(actual.intercept == expected.intercept,
                  label + ".intercept differs");
  REQUIRE_MESSAGE(actual.l2 == expected.l2,
                  label + ".l2 differs");
  REQUIRE_MESSAGE(actual.feature_mean == expected.feature_mean,
                  label + ".feature mean differs");
  REQUIRE_MESSAGE(actual.feature_std == expected.feature_std,
                  label + ".feature scaler differs");
  REQUIRE_MESSAGE(actual.coefficients == expected.coefficients,
                  label + ".coefficients differ");
  REQUIRE_MESSAGE(
      actual.training_labels == expected.training_labels,
      label + ".training labels differ");
  REQUIRE_MESSAGE(
      actual.training_probabilities ==
          expected.training_probabilities,
      label + ".training probabilities differ");
  REQUIRE_MESSAGE(actual.holdout_labels == expected.holdout_labels,
                  label + ".holdout labels differ");
  REQUIRE_MESSAGE(
      actual.holdout_probabilities ==
          expected.holdout_probabilities,
      label + ".holdout probabilities differ");
  REQUIRE_MESSAGE(actual.iterations == expected.iterations,
                  label + ".iterations differ");
  REQUIRE_MESSAGE(actual.converged == expected.converged,
                  label + ".convergence differs");
}

void test_stage3_production_causal_controls_and_restoration() {
  const std::vector<int> devices = available_devices();
  REQUIRE_MESSAGE(!devices.empty(),
                  "causal mechanics requires CUDA");
  EvaluationOptions options{};
  options.observer_training_trials = 4;
  options.observer_holdout_trials = 4;
  options.trials_per_tbw_soa = 1;
  options.trials_per_sbw_condition = 1;
  options.trials_per_rf_location = 1;
  options.trials_per_inverse_condition = 1;
  options.burn_in_ms = 300;
  options.evaluation_seed = 0x43415553414Cull;
  options.observer_rate_hz = 50.0f;
  options.response_floor_hz = 1.0f;
  const Config config = calibration_for(0).result.config;
  NativeModel model(config, 0, 1);
  const auto original = model.frozen_weight_audit();

  options.control = CausalControl::kNone;
  const EvaluationMetrics baseline = model.evaluate(options);
  const auto controls =
      model.evaluate_all_controls(options, baseline);
  const std::array<CausalControl, 5> expected_controls{
      CausalControl::kNmdaOff,
      CausalControl::kGabaaOff,
      CausalControl::kRecurrentExcitationOff,
      CausalControl::kRecruitedInhibitionOff,
      CausalControl::kMsiEAdaptationOff,
  };
  REQUIRE(controls.size() == expected_controls.size());
  for (std::size_t index = 0; index < controls.size(); ++index) {
    const auto& result = controls[index];
    REQUIRE(result.control == expected_controls[index]);
    REQUIRE(result.observer_frozen);
    require_exact_observer(
        result.evaluation.observer, baseline.observer,
        "all_controls[" + std::to_string(index) +
            "].frozen_observer");
    REQUIRE(
        result.evaluation.temporal.physical_soa_grid_ms ==
        baseline.temporal.physical_soa_grid_ms);
    REQUIRE(
        result.evaluation.spatial.disparity_grid_deg ==
        baseline.spatial.disparity_grid_deg);
    require_projection_control_mapping(
        result.evaluation.control_audit,
        expected_controls[index],
        "all_controls[" + std::to_string(index) + "]");
    const auto after = model.frozen_weight_audit();
    require_exact_frozen_weight_audit(
        after, original,
        "all_controls[" + std::to_string(index) +
            "].neutrality");
  }

  options.control = CausalControl::kSourceRowShuffle;
  const EvaluationMetrics shuffled = model.evaluate(options);
  require_projection_control_mapping(
      shuffled.control_audit, CausalControl::kSourceRowShuffle,
      "causal_control.source_row_shuffle");
  require_exact_frozen_weight_audit(
      model.frozen_weight_audit(), original,
      "causal_control.source_row_shuffle.neutrality");

  options.control = CausalControl::kNone;
  options.evaluation_seed = 0x524553544F5245ull;
  const EvaluationMetrics restored = model.evaluate(options);
  require_projection_control_mapping(
      restored.control_audit, CausalControl::kNone,
      "causal_control.restored");
  require_exact_frozen_weight_audit(
      model.frozen_weight_audit(), original,
      "causal_control.restored.neutrality");
}

void require_cpu_scalar_frozen_trial(
    const FrozenTrialResult& trial, const std::string& label) {
  std::array<unsigned int, kObserverFeatureMilliseconds> per_ms{};
  for (std::size_t index = 0; index < per_ms.size(); ++index) {
    per_ms[index] =
        trial.excitatory_spikes_per_relative_ms[index];
  }
  const auto scalar_features = scalar_observer_features(per_ms);
  for (std::size_t feature = 0;
       feature < scalar_features.size(); ++feature) {
    REQUIRE_MESSAGE(
        trial.features[feature] ==
            static_cast<float>(scalar_features[feature]),
        label + ".GPU feature differs from CPU scalar binning");
  }
  const unsigned int baseline_population = std::accumulate(
      per_ms.begin(), per_ms.begin() + 100, 0u);
  const unsigned int response_population = std::accumulate(
      per_ms.begin() + 100, per_ms.begin() + 350, 0u);
  const unsigned int baseline_neurons = std::accumulate(
      trial.excitatory_baseline_spikes_per_neuron.begin(),
      trial.excitatory_baseline_spikes_per_neuron.end(), 0u);
  const unsigned int response_neurons = std::accumulate(
      trial.excitatory_response_spikes_per_neuron.begin(),
      trial.excitatory_response_spikes_per_neuron.end(), 0u);
  REQUIRE_MESSAGE(
      baseline_population == baseline_neurons,
      label + ".GPU baseline differs from CPU population sum");
  REQUIRE_MESSAGE(
      response_population == response_neurons,
      label + ".GPU response differs from CPU population sum");
  const float scalar_baseline_hz =
      1000.0f * static_cast<float>(baseline_population) /
      (100.0f * clean_msi::kExcitatoryNeurons);
  const float scalar_response_hz =
      1000.0f * static_cast<float>(response_population) /
      (250.0f * clean_msi::kExcitatoryNeurons);
  require_near(
      trial.baseline_rate_hz, scalar_baseline_hz,
      1.0e-6f, 1.0e-6f, label + ".CPU/GPU baseline rate");
  require_near(
      trial.response_rate_hz, scalar_response_hz,
      1.0e-6f, 1.0e-6f, label + ".CPU/GPU response rate");
}

void test_stage3_evaluation_batching_performance_and_parity() {
  const std::vector<int> devices = available_devices();
  REQUIRE_MESSAGE(devices.size() == 2u,
                  "evaluation performance requires cuda0 and cuda1");
  std::vector<ControlledCondition> conditions(4);
  conditions[0].auditory_present = true;
  conditions[0].visual_present = false;
  conditions[0].auditory_location_deg = -40.0f;
  conditions[0].auditory_rate_hz = 25.0f;
  conditions[0].physical_soa_ms = 0.0f;
  conditions[0].random_group = 5000;
  conditions[1].auditory_present = false;
  conditions[1].visual_present = true;
  conditions[1].visual_location_deg = 40.0f;
  conditions[1].visual_rate_hz = 100.0f;
  conditions[1].physical_soa_ms = 0.0f;
  conditions[1].random_group = 5001;
  conditions[2].auditory_present = true;
  conditions[2].visual_present = true;
  conditions[2].auditory_location_deg = 0.0f;
  conditions[2].visual_location_deg = 0.0f;
  conditions[2].auditory_rate_hz = 50.0f;
  conditions[2].visual_rate_hz = 50.0f;
  conditions[2].physical_soa_ms = -50.0f;
  conditions[2].random_group = 5002;
  conditions[3].auditory_present = true;
  conditions[3].visual_present = true;
  conditions[3].auditory_location_deg = -20.0f;
  conditions[3].visual_location_deg = 20.0f;
  conditions[3].auditory_rate_hz = 75.0f;
  conditions[3].visual_rate_hz = 25.0f;
  conditions[3].physical_soa_ms = 100.0f;
  conditions[3].random_group = 5003;
  for (std::size_t index = 0; index < conditions.size(); ++index) {
    conditions[index].label = static_cast<int>(index % 2u);
    conditions[index].control = CausalControl::kNone;
  }
  constexpr int kTrialsPerCondition = 4;
  constexpr int kRepeats = 3;
  constexpr std::uint64_t kEvaluationSeed = 0x4241544348494E47ull;
  const Config common_config = calibration_for(0).result.config;
  FrozenTrialBatch cuda0_reference{};
  std::vector<FrozenWeightAudit> cuda0_weights;

  for (const int device : devices) {
    NativeModel model(common_config, device, 1);
    const auto before = model.frozen_weight_audit();
    if (device == 0) {
      cuda0_weights = before;
    } else {
      require_exact_frozen_weight_audit(
          before, cuda0_weights, "cuda1/cuda0 initial weights");
    }
    static_cast<void>(model.run_frozen_trials(
        {conditions.front()}, 1, kEvaluationSeed ^ 0x5741524Dull,
        300));

    std::vector<double> batched_seconds;
    std::vector<double> serialized_seconds;
    FrozenTrialBatch batched{};
    std::vector<FrozenTrialBatch> serialized(conditions.size());
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
      batched_seconds.push_back(timed_seconds(device, [&] {
        batched = model.run_frozen_trials(
            conditions, kTrialsPerCondition, kEvaluationSeed, 300);
      }));
      serialized_seconds.push_back(timed_seconds(device, [&] {
        for (std::size_t condition = 0;
             condition < conditions.size(); ++condition) {
          serialized[condition] = model.run_frozen_trials(
              {conditions[condition]}, kTrialsPerCondition,
              kEvaluationSeed, 300);
        }
      }));
    }
    REQUIRE(batched.model_seed_count == 1);
    REQUIRE(batched.condition_count ==
            static_cast<int>(conditions.size()));
    REQUIRE(batched.trials_per_condition == kTrialsPerCondition);
    REQUIRE(
        batched.trials.size() ==
        conditions.size() *
            static_cast<std::size_t>(kTrialsPerCondition));
    for (std::size_t condition = 0;
         condition < conditions.size(); ++condition) {
      REQUIRE(serialized[condition].trials.size() ==
              static_cast<std::size_t>(kTrialsPerCondition));
      for (int trial = 0; trial < kTrialsPerCondition; ++trial) {
        const std::size_t batch_index =
            condition * kTrialsPerCondition +
            static_cast<std::size_t>(trial);
        FrozenTrialResult normalized =
            serialized[condition]
                .trials[static_cast<std::size_t>(trial)];
        normalized.condition_index = static_cast<int>(condition);
        require_exact_frozen_trial(
            batched.trials[batch_index], normalized,
            "cuda" + std::to_string(device) + ".batch_serial[" +
                std::to_string(condition) + "][" +
                std::to_string(trial) + "]");
        require_cpu_scalar_frozen_trial(
            batched.trials[batch_index],
            "cuda" + std::to_string(device) + ".CPU_scalar[" +
                std::to_string(condition) + "][" +
                std::to_string(trial) + "]");
      }
    }
    if (device == 0) {
      cuda0_reference = batched;
    } else {
      REQUIRE(batched.trials.size() == cuda0_reference.trials.size());
      for (std::size_t trial = 0;
           trial < batched.trials.size(); ++trial) {
        require_exact_frozen_trial(
            batched.trials[trial], cuda0_reference.trials[trial],
            "cuda1/cuda0 frozen trial[" + std::to_string(trial) +
                "]");
      }
    }
    require_exact_frozen_weight_audit(
        model.frozen_weight_audit(), before,
        "cuda" + std::to_string(device) +
            ".batching weight neutrality");
    const double batched_median =
        median_double(batched_seconds);
    const double serialized_median =
        median_double(serialized_seconds);
    REQUIRE_MESSAGE(
        batched_median < serialized_median,
        "batched evaluation was not faster than condition-serialized "
        "evaluation");
    const double total_trials =
        static_cast<double>(
            conditions.size() * kTrialsPerCondition);
    std::cout << "[PERF] cuda" << device
              << " evaluation_trials=" << total_trials
              << " batched_wall_s=" << std::fixed
              << std::setprecision(6) << batched_median
              << " serialized_wall_s=" << serialized_median
              << " speedup=" << serialized_median / batched_median
              << " batched_trials_per_s="
              << total_trials / batched_median << '\n';
  }
}

// ---------------------------------------------------------------------------
// Persistent native training tests.

constexpr std::size_t population_index(Population population) {
  return static_cast<std::size_t>(population);
}

constexpr std::size_t path_index(PlasticPath path) {
  return static_cast<std::size_t>(path);
}

const PopulationDiagnostics& population_diagnostics(
    const TrainingDiagnostics& diagnostics, Population population) {
  return diagnostics.populations[population_index(population)];
}

const PathDiagnostics& path_diagnostics(
    const TrainingDiagnostics& diagnostics, PlasticPath path) {
  return diagnostics.paths[path_index(path)];
}

constexpr std::array<Population, clean_msi::kPopulationCount>
    kAllPopulations{
        Population::kAuditory,
        Population::kVisual,
        Population::kExcitatory,
        Population::kInhibitory,
    };

constexpr std::array<PlasticPath, 7> kAllPlasticPaths{
    PlasticPath::kAuditoryToExcitatory,
    PlasticPath::kVisualToExcitatory,
    PlasticPath::kExcitatoryToExcitatory,
    PlasticPath::kAuditoryToInhibitory,
    PlasticPath::kVisualToInhibitory,
    PlasticPath::kExcitatoryToInhibitory,
    PlasticPath::kInhibitoryToExcitatory,
};

const char* population_name(Population population) {
  switch (population) {
    case Population::kAuditory:
      return "A";
    case Population::kVisual:
      return "V";
    case Population::kExcitatory:
      return "E";
    case Population::kInhibitory:
      return "I";
  }
  return "?";
}

const char* path_name(PlasticPath path) {
  switch (path) {
    case PlasticPath::kAuditoryToExcitatory:
      return "A->E";
    case PlasticPath::kVisualToExcitatory:
      return "V->E";
    case PlasticPath::kExcitatoryToExcitatory:
      return "E->E";
    case PlasticPath::kAuditoryToInhibitory:
      return "A->I";
    case PlasticPath::kVisualToInhibitory:
      return "V->I";
    case PlasticPath::kExcitatoryToInhibitory:
      return "E->I";
    case PlasticPath::kInhibitoryToExcitatory:
      return "I->E";
  }
  return "?";
}

int maximum_contacts(PlasticPath path) {
  switch (path) {
    case PlasticPath::kAuditoryToExcitatory:
    case PlasticPath::kVisualToExcitatory:
      return clean_msi::kAuditoryNeurons *
             clean_msi::kExcitatoryNeurons;
    case PlasticPath::kExcitatoryToExcitatory:
      return clean_msi::kExcitatoryNeurons *
             (clean_msi::kExcitatoryNeurons - 1);
    case PlasticPath::kAuditoryToInhibitory:
    case PlasticPath::kVisualToInhibitory:
    case PlasticPath::kExcitatoryToInhibitory:
      return clean_msi::kExcitatoryNeurons *
             clean_msi::kInhibitoryNeurons;
    case PlasticPath::kInhibitoryToExcitatory:
      return clean_msi::kInhibitoryNeurons *
             clean_msi::kExcitatoryNeurons;
  }
  return 0;
}

struct ScaffoldPathContract {
  std::size_t path = 0;
  int pre_size = 0;
  int post_size = 0;
  int in_degree = 0;
  bool exclude_self = false;
  float minimum_weight = 0.0f;
  float maximum_weight = 0.0f;
  const char* label = "";
};

double scaffold_coordinate(int index, int size) {
  return size == clean_msi::kInhibitoryNeurons
             ? 179.0 * static_cast<double>(index) / 59.0
             : static_cast<double>(index);
}

void
test_science_coarse_gaussian_feedforward_and_position_blind_scaffolds() {
  const std::vector<int> devices = available_devices();
  REQUIRE_MESSAGE(!devices.empty(),
                  "science scaffold test requires CUDA");
  constexpr int kDevice = 0;
  static_assert(clean_msi::kAuditoryScaffoldFwhmDeg == 105.0);
  static_assert(clean_msi::kVisualScaffoldFwhmDeg == 45.0);
  static_assert(clean_msi::kFwhmToSigmaDivisor == 2.35482);
  constexpr std::array<ScaffoldPathContract, 6> kContracts{{
      {0u, 180, 180, clean_msi::kFeedforwardScaffoldInDegree,
       false, 0.05f, 0.15f, "A->E"},
      {1u, 180, 180, clean_msi::kFeedforwardScaffoldInDegree,
       false, 0.05f, 0.15f, "V->E"},
      {2u, 180, 180,
       clean_msi::kRecurrentExcitatoryScaffoldInDegree,
       true, 0.01f, 0.03f, "E->E"},
      {3u, 180, 60, clean_msi::kFeedforwardScaffoldInDegree,
       false, 0.05f, 0.15f, "A->I"},
      {4u, 180, 60, clean_msi::kFeedforwardScaffoldInDegree,
       false, 0.05f, 0.15f, "V->I"},
      {6u, 60, 180, clean_msi::kInhibitoryScaffoldInDegree,
       false, 0.02f, 0.05f, "I->E"},
  }};

  std::array<std::vector<double>, kContracts.size()> distances{};
  std::array<std::vector<double>, kContracts.size()> masks{};
  std::array<std::vector<double>, kContracts.size()>
      active_distances{};
  std::array<std::vector<double>, kContracts.size()>
      active_weights{};
  std::array<std::vector<std::uint8_t>, kContracts.size()>
      first_seed_masks{};
  std::array<std::vector<std::uint8_t>, kContracts.size()>
      second_seed_masks{};
  for (int batch = 0; batch < 2; ++batch) {
    Config config{};
    config.calibrated = true;
    config.seed =
        0x53434146464F4C44ull +
        static_cast<std::uint64_t>(batch * 8);
    NativeModel model(config, kDevice, 8);
    const std::vector<FrozenWeightAudit> audits =
        model.frozen_weight_audit();
    REQUIRE(audits.size() == 8u);
    for (std::size_t seed = 0; seed < audits.size(); ++seed) {
      for (std::size_t contract_index = 0;
           contract_index < kContracts.size();
           ++contract_index) {
        const ScaffoldPathContract& contract =
            kContracts[contract_index];
        const auto& path = audits[seed].paths[contract.path];
        REQUIRE(
            path.masks.size() ==
            static_cast<std::size_t>(
                contract.pre_size * contract.post_size));
        REQUIRE(path.weights.size() == path.masks.size());
        for (int post = 0; post < contract.post_size; ++post) {
          int incoming = 0;
          for (int pre = 0; pre < contract.pre_size; ++pre) {
            const std::size_t index =
                static_cast<std::size_t>(pre) *
                    contract.post_size +
                post;
            const bool active = path.masks[index] != 0;
            incoming += active ? 1 : 0;
            if (contract.exclude_self && pre == post) {
              REQUIRE_MESSAGE(
                  !active, std::string(contract.label) +
                               " contains a self-connection");
            }
            const double distance =
                std::abs(
                    scaffold_coordinate(pre, contract.pre_size) -
                    scaffold_coordinate(post, contract.post_size));
            distances[contract_index].push_back(distance);
            masks[contract_index].push_back(active ? 1.0 : 0.0);
            if (active) {
              REQUIRE_MESSAGE(
                  path.weights[index] >= contract.minimum_weight &&
                      path.weights[index] < contract.maximum_weight,
                  std::string(contract.label) +
                      " active weight left its preregistered weak "
                      "distribution");
              active_distances[contract_index].push_back(distance);
              active_weights[contract_index].push_back(
                  path.weights[index]);
            } else {
              REQUIRE_MESSAGE(
                  path.weights[index] == 0.0f,
                  std::string(contract.label) +
                      " inactive contact has nonzero efficacy");
            }
          }
          REQUIRE_MESSAGE(
              incoming == contract.in_degree,
              std::string(contract.label) +
                  " does not have exact fixed in-degree");
        }
        if (batch == 0 && seed == 0) {
          first_seed_masks[contract_index] = path.masks;
        }
        if (batch == 0 && seed == 1) {
          second_seed_masks[contract_index] = path.masks;
        }
      }
    }
  }
  const auto mean =
      [](const std::vector<double>& values) {
        REQUIRE(!values.empty());
        return std::accumulate(
                   values.begin(), values.end(), 0.0) /
               static_cast<double>(values.size());
      };
  std::array<double, kContracts.size()> mean_candidate_distance{};
  std::array<double, kContracts.size()> mean_active_distance{};
  for (std::size_t index = 0; index < kContracts.size(); ++index) {
    REQUIRE_MESSAGE(
        first_seed_masks[index] != second_seed_masks[index],
        std::string(kContracts[index].label) +
            " deterministic scaffold did not vary across seeds");
    mean_candidate_distance[index] = mean(distances[index]);
    mean_active_distance[index] = mean(active_distances[index]);
    if (kContracts[index].path == 2u ||
        kContracts[index].path == 6u) {
      const double mask_distance_correlation =
          pearson_double(distances[index], masks[index]);
      REQUIRE_MESSAGE(
          std::abs(mask_distance_correlation) < 0.03,
          std::string(kContracts[index].label) +
              " position-blind mask retains a spatial-distance trend");
    }
    const double weight_distance_correlation =
        pearson_double(
            active_distances[index], active_weights[index]);
    REQUIRE_MESSAGE(
        std::abs(weight_distance_correlation) < 0.03,
        std::string(kContracts[index].label) +
            " initial weights retain a spatial-distance trend");
  }
  for (const std::size_t index : {0u, 1u, 3u, 4u}) {
    REQUIRE_MESSAGE(
        mean_active_distance[index] <
            mean_candidate_distance[index],
        std::string(kContracts[index].label) +
            " Gaussian mask does not favor nearby candidates");
  }
  REQUIRE(
      mean_active_distance[1] < mean_active_distance[0]);
  REQUIRE(
      mean_active_distance[4] < mean_active_distance[3]);
  constexpr double kInhibitoryTargetStepDeg =
      179.0 /
      static_cast<double>(clean_msi::kInhibitoryNeurons - 1);
  REQUIRE(
      std::abs(
          mean_active_distance[0] -
          mean_active_distance[3]) <=
      kInhibitoryTargetStepDeg);
  REQUIRE(
      std::abs(
          mean_active_distance[1] -
          mean_active_distance[4]) <=
      kInhibitoryTargetStepDeg);
}

void test_science_clopath_additive_hard_bounds_and_oja_soft_bounds() {
  constexpr float kEta = 0.01f;
  const float theta_minus_mv =
      clean_msi::izhikevich_stable_rest_voltage_mv(0.20f);
  const float ltp =
      clean_msi::clopath_pair_delta(
          kEta, 0.8f, -35.0f, -40.0f, -50.0f,
          theta_minus_mv, clean_msi::kClopathThetaPlusMv,
          clean_msi::kClopathHomeostasisReferenceMv2,
          false);
  const float ltd =
      clean_msi::clopath_pair_delta(
          kEta, 0.0f, -65.0f, -65.0f, -65.0f,
          theta_minus_mv, clean_msi::kClopathThetaPlusMv,
          clean_msi::kClopathHomeostasisReferenceMv2,
          true);
  REQUIRE(ltp > 0.0f);
  REQUIRE(ltd < 0.0f);
  const float low_ltp =
      clean_msi::additive_hard_bound_update(
          0.20f, ltp, 0.0f, 1.0f);
  const float high_ltp =
      clean_msi::additive_hard_bound_update(
          0.80f, ltp, 0.0f, 1.0f);
  require_near(
      low_ltp - 0.20f, high_ltp - 0.80f,
      1.0e-7f, 1.0e-6f,
      "Clopath additive LTP has no weight-dependent factor");
  const float low_ltd =
      clean_msi::additive_hard_bound_update(
          0.20f, ltd, 0.0f, 1.0f);
  const float high_ltd =
      clean_msi::additive_hard_bound_update(
          0.80f, ltd, 0.0f, 1.0f);
  require_near(
      0.20f - low_ltd, 0.80f - high_ltd,
      1.0e-7f, 1.0e-6f,
      "Clopath additive LTD has no weight-dependent factor");
  REQUIRE(
      clean_msi::additive_hard_bound_update(
          0.99f, 100.0f * ltp, 0.0f, 1.0f) == 1.0f);
  REQUIRE(
      clean_msi::additive_hard_bound_update(
          0.01f, 100.0f * ltd, 0.0f, 1.0f) == 0.0f);

  const float oja_delta =
      scalar_oja_delta(0.25f, 0.8f, 0.4f, 0.10f, 1.0f);
  const float low_oja =
      clean_msi::multiplicative_soft_bound_update(
          0.20f, oja_delta, 0.0f, 1.0f);
  const float high_oja =
      clean_msi::multiplicative_soft_bound_update(
          0.80f, oja_delta, 0.0f, 1.0f);
  REQUIRE(low_oja - 0.20f > high_oja - 0.80f);
}

void test_science_istdp_target_and_hard_clipping() {
  REQUIRE(clean_msi::kIstdpTraceTauMs == 20.0f);
  REQUIRE(clean_msi::kIstdpTargetRateHz == 5.0f);
  require_near(
      clean_msi::kIstdpAlpha, 0.20f,
      1.0e-7f, 1.0e-7f, "Vogels alpha");
  constexpr float kEta = 0.05f;
  const float depressed =
      clean_msi::ordered_vogels_istdp_update(
          0.50f, true, false, 1.0f, 0.0f,
          kEta, clean_msi::kIstdpAlpha,
          0.0f, 1.0f);
  const float potentiated =
      clean_msi::ordered_vogels_istdp_update(
          0.50f, false, true, 0.50f, 1.0f,
          kEta, clean_msi::kIstdpAlpha,
          0.0f, 1.0f);
  REQUIRE(
      depressed < 0.50f);
  REQUIRE(
      potentiated > 0.50f);
  REQUIRE(
      clean_msi::ordered_vogels_istdp_update(
          0.0f, true, false, 1.0f, 0.0f,
          kEta, clean_msi::kIstdpAlpha,
          0.0f, 1.0f) == 0.0f);
  REQUIRE(
      clean_msi::ordered_vogels_istdp_update(
          1.0f, false, true, 0.50f, 1.0f,
          kEta, clean_msi::kIstdpAlpha,
          0.0f, 1.0f) == 1.0f);
}

void test_science_nmda_shares_contact_efficacy() {
  constexpr float kRatio = 0.75f;
  const float weak =
      clean_msi::shared_nmda_contact_efficacy(
          0.05f, kRatio, true);
  const float strong =
      clean_msi::shared_nmda_contact_efficacy(
          0.20f, kRatio, true);
  require_near(
      strong, 4.0f * weak,
      1.0e-7f, 1.0e-7f,
      "NMDA learned-weight proportionality");
  REQUIRE(
      clean_msi::shared_nmda_contact_efficacy(
          0.20f, kRatio, false) == 0.0f);
  REQUIRE(
      clean_msi::shared_nmda_contact_efficacy(
          0.0f, kRatio, true) == 0.0f);
}

void test_science_calibration_runtime_receptor_ratio_identity() {
  constexpr float kRuntimeLearnedWeight = 0.37f;
  const float calibration_ampa =
      clean_msi::kCalibrationExcitatoryContactWeight;
  const float calibration_nmda =
      clean_msi::calibration_nmda_contact_efficacy(
          calibration_ampa, true);
  const float runtime_nmda =
      clean_msi::shared_nmda_contact_efficacy(
          kRuntimeLearnedWeight, 1.0f, true);
  const float calibration_ratio =
      calibration_nmda / calibration_ampa;
  const float runtime_ratio =
      runtime_nmda / kRuntimeLearnedWeight;
  REQUIRE(calibration_ratio == runtime_ratio);
  REQUIRE(calibration_ratio == 1.0f);
  REQUIRE(calibration_ratio != 0.20f);
  REQUIRE(
      clean_msi::calibration_nmda_contact_efficacy(
          calibration_ampa, false) == 0.0f);
}

void test_science_gabaa_unitary_ipsp_and_runtime_relay() {
  constexpr int kDevice = 0;
  constexpr int kAfferentToRelayDelayMs = 3;
  constexpr float kOldSpikePredicateQuantum = 0.00349115161f;
  const CalibrationResult& result =
      calibration_for(kDevice).result;
  const auto& audit = result.gabaa_relay;
  const auto& efficacy = result.gabaa_efficacy;

  REQUIRE(
      audit.relay_neurons ==
      clean_msi::kInhibitoryScaffoldInDegree);
  REQUIRE(audit.excitatory_packets == 2);
  REQUIRE(audit.relay_spikes > 0);
  REQUIRE(audit.gabaa_arrivals > 0);
  REQUIRE(audit.lag_count > 0);
  REQUIRE(audit.lag_count <= audit.gabaa_arrivals);
  REQUIRE(
      audit.minimum_lag_ms >=
      kAfferentToRelayDelayMs +
          clean_msi::kGabaaUnitaryDelayMs);
  REQUIRE(audit.maximum_lag_ms >= audit.minimum_lag_ms);
  REQUIRE(audit.mean_lag_ms >=
          static_cast<float>(audit.minimum_lag_ms));
  REQUIRE(audit.mean_lag_ms <=
          static_cast<float>(audit.maximum_lag_ms));
  int measured_lag_count = 0;
  for (int lag = 0; lag < clean_msi::kGabaaDurationMs; ++lag) {
    measured_lag_count +=
        audit.direct_to_gabaa_lag_counts[lag];
  }
  REQUIRE(measured_lag_count == audit.lag_count);
  for (int lag = 0;
       lag <
       kAfferentToRelayDelayMs +
           clean_msi::kGabaaUnitaryDelayMs;
       ++lag) {
    REQUIRE(
        audit.direct_to_gabaa_lag_counts[lag] == 0);
  }

  REQUIRE(audit.without_gabaa_spikes == 2);
  REQUIRE(
      audit.with_gabaa_spikes <
      audit.without_gabaa_spikes);

  REQUIRE(efficacy.unitary.contact_count == 1);
  REQUIRE(
      efficacy.unitary.amplitude_mv >=
      clean_msi::kGabaaUnitaryMinimumMv);
  REQUIRE(
      efficacy.unitary.amplitude_mv <=
      clean_msi::kGabaaUnitaryMaximumMv);
  REQUIRE(efficacy.unitary.signed_nadir_mv < 0.0f);
  require_near(
      -efficacy.unitary.signed_nadir_mv,
      efficacy.unitary.amplitude_mv, 1.0e-6f, 1.0e-6f,
      "unitary GABAA amplitude/sign");
  REQUIRE(efficacy.unitary.area_mv_ms > 0.0f);
  REQUIRE(efficacy.unitary.outward_charge > 0.0f);
  REQUIRE(efficacy.unitary.control_spikes == 0);
  REQUIRE(efficacy.unitary.gabaa_spikes == 0);
  REQUIRE(
      efficacy.unitary.amplitude_mv == result.gabaa.primary);
  REQUIRE(
      efficacy.unitary.signed_nadir_mv ==
      result.gabaa.secondary);
  REQUIRE(
      efficacy.unitary.area_mv_ms == result.gabaa.tertiary);

  REQUIRE(
      efficacy.compound.contact_count ==
      clean_msi::kInhibitoryScaffoldInDegree);
  REQUIRE(efficacy.compound.amplitude_mv >= 3.0f);
  REQUIRE(
      efficacy.compound.amplitude_mv >
      efficacy.unitary.amplitude_mv);
  REQUIRE(efficacy.compound.signed_nadir_mv < 0.0f);
  REQUIRE(
      efficacy.compound.area_mv_ms >
      efficacy.unitary.area_mv_ms);
  REQUIRE(
      efficacy.compound.outward_charge >
      efficacy.unitary.outward_charge);
  REQUIRE(efficacy.compound.control_spikes == 0);
  REQUIRE(efficacy.compound.gabaa_spikes == 0);

  const auto gabaa_off = clean_msi::audit_gabaa_efficacy(
      kDevice, result.config, 0.0f);
  REQUIRE(gabaa_off.unitary.amplitude_mv == 0.0f);
  REQUIRE(gabaa_off.unitary.signed_nadir_mv == 0.0f);
  REQUIRE(gabaa_off.unitary.area_mv_ms == 0.0f);
  REQUIRE(gabaa_off.unitary.outward_charge == 0.0f);
  REQUIRE(gabaa_off.compound.amplitude_mv == 0.0f);
  REQUIRE(gabaa_off.compound.signed_nadir_mv == 0.0f);
  REQUIRE(gabaa_off.compound.area_mv_ms == 0.0f);
  REQUIRE(gabaa_off.compound.outward_charge == 0.0f);

  const auto old_submillivolt =
      clean_msi::evaluate_assay_candidates(
          kDevice, result.config, AssayKind::kGabaa,
          {kOldSpikePredicateQuantum});
  REQUIRE(old_submillivolt.size() == 1u);
  REQUIRE(old_submillivolt[0].primary > 0.0f);
  REQUIRE(
      old_submillivolt[0].primary <
      clean_msi::kGabaaUnitaryMinimumMv);
  REQUIRE(old_submillivolt[0].primary < 0.01f);

  const auto threshold_check =
      clean_msi::evaluate_assay_candidates(
          kDevice, result.config, AssayKind::kGabaa,
          {0.99f * result.config.q_gabaa,
           result.config.q_gabaa});
  REQUIRE(threshold_check.size() == 2u);
  REQUIRE(
      threshold_check[0].primary <
      clean_msi::kGabaaUnitaryMinimumMv);
  REQUIRE(
      threshold_check[1].primary >=
      clean_msi::kGabaaUnitaryMinimumMv);
  REQUIRE(
      threshold_check[1].primary <=
      clean_msi::kGabaaUnitaryMaximumMv);
  REQUIRE(result.config.q_gabaa > 1.0f);
  REQUIRE(result.config.q_gabaa < 10.0f);

  for (const AssayKind kind :
       std::array<AssayKind, 5>{
           AssayKind::kExternalRs,
           AssayKind::kFeedforwardE,
           AssayKind::kFeedforwardI,
           AssayKind::kBackgroundRs,
           AssayKind::kBackgroundFs}) {
    REQUIRE(
        clean_msi::calibration_interval_requires_closure(kind));
  }

  std::cout
      << "[SCIENCE] gabaa_unitary q="
      << std::setprecision(9) << result.config.q_gabaa
      << " old_q_ipsp_mV="
      << old_submillivolt[0].primary
      << " unitary_mV=" << efficacy.unitary.amplitude_mv
      << " unitary_area_mV_ms="
      << efficacy.unitary.area_mv_ms
      << " unitary_charge="
      << efficacy.unitary.outward_charge
      << " compound_mV="
      << efficacy.compound.amplitude_mv
      << " compound_area_mV_ms="
      << efficacy.compound.area_mv_ms
      << " compound_charge="
      << efficacy.compound.outward_charge
      << " relay_spikes=" << audit.relay_spikes
      << " relay_lag_min_ms=" << audit.minimum_lag_ms
      << " relay_lag_max_ms=" << audit.maximum_lag_ms
      << " causal_spikes=" << audit.without_gabaa_spikes
      << "->" << audit.with_gabaa_spikes << '\n';
}

void test_science_msi_e_intrinsic_phenotype_mix() {
  constexpr int kDevice = 0;
  constexpr std::uint64_t kBaseSeed = 0;
  const auto audit =
      clean_msi::audit_msi_e_intrinsic_phenotypes(
          kDevice, kBaseSeed);

  std::cout
      << "[SCIENCE] msi_e_intrinsic assigned_marked="
      << audit.assigned_marked_total
      << " greater_than_two_prevalence="
      << std::fixed << std::setprecision(6)
      << audit.greater_than_two_prevalence
      << " regular_first_to_mean="
      << audit.regular_first_to_mean_isi
      << " marked_first_to_mean="
      << audit.marked_first_to_mean_isi
      << " regular_greater_than_two="
      << audit.regular_greater_than_two_fraction
      << " marked_greater_than_two="
      << audit.marked_greater_than_two_fraction
      << " index_correlation="
      << audit.position_index_correlation << '\n';

  REQUIRE(
      audit.seed_count ==
      clean_msi::kMsiEPhenotypeAuditSeeds);
  REQUIRE(
      audit.neuron_count ==
      clean_msi::kMsiEPhenotypeAuditSeeds *
          clean_msi::kExcitatoryNeurons);
  REQUIRE(
      audit.assigned_marked_total ==
      clean_msi::kMsiEPhenotypeAuditSeeds *
          clean_msi::kMsiEMarkedPhenotypesPerSeed);
  for (int seed = 0;
       seed < clean_msi::kMsiEPhenotypeAuditSeeds; ++seed) {
    REQUIRE(
        audit.assigned_marked_per_seed[
            static_cast<std::size_t>(seed)] ==
        clean_msi::kMsiEMarkedPhenotypesPerSeed);
  }
  REQUIRE(audit.greater_than_two_prevalence >= 0.03f);
  REQUIRE(audit.greater_than_two_prevalence <= 0.07f);
  REQUIRE(audit.regular_first_to_mean_isi >= 0.50f);
  REQUIRE(audit.regular_first_to_mean_isi <= 0.70f);
  REQUIRE(audit.marked_first_to_mean_isi >= 0.20f);
  REQUIRE(audit.marked_first_to_mean_isi <= 0.35f);
  REQUIRE(audit.regular_greater_than_two_fraction <= 0.05f);
  REQUIRE(audit.marked_greater_than_two_fraction >= 0.80f);
  REQUIRE(std::fabs(audit.position_index_correlation) <= 0.10f);
}

void test_science_msi_e_spike_reset_adaptation_causality() {
  constexpr int kDevice = 0;
  constexpr std::uint64_t kSeed = 0;
  constexpr int kLocalNeuron = 0;
  constexpr int kGlobalNeuron =
      clean_msi::kAuditoryNeurons +
      clean_msi::kVisualNeurons + kLocalNeuron;
  constexpr std::uint32_t kParameterStream = 810u;
  constexpr std::uint32_t kInitialStateStream = 811u;
  constexpr std::uint32_t kStreamKeyMultiplier = 0xA511E9B3u;
  constexpr float kTwoPowMinus32 =
      2.3283064365386962890625e-10f;

  const auto production_uniform =
      [](std::uint32_t stream, std::uint64_t entity) {
        const auto words = clean_msi::philox4x32_10_host(
            {static_cast<std::uint32_t>(entity),
             static_cast<std::uint32_t>(entity >> 32), 0u, 0u},
            {static_cast<std::uint32_t>(kSeed) ^ stream,
             static_cast<std::uint32_t>(kSeed >> 32) +
                 kStreamKeyMultiplier * stream});
        return (static_cast<float>(words[0]) + 0.5f) *
               kTwoPowMinus32;
      };

  const float heterogeneity =
      production_uniform(kParameterStream, kGlobalNeuron);
  clean_msi::NeuronParameters production_parameters{
      0.02f,
      0.20f,
      -65.0f + 15.0f * heterogeneity * heterogeneity,
      clean_msi::kMsiEMarkedRecoveryIncrement};
  clean_msi::NeuronParameters adaptation_off_parameters =
      production_parameters;
  adaptation_off_parameters.d = 0.0f;

  const float initial_draw =
      production_uniform(kInitialStateStream, kGlobalNeuron);
  const float initial_lower =
      std::min(production_parameters.c_mv, -55.0f);
  const float initial_upper =
      std::max(production_parameters.c_mv, -55.0f);
  clean_msi::NeuronState settled_state{
      initial_lower + initial_draw * (initial_upper - initial_lower),
      0.0f};
  settled_state.recovery =
      production_parameters.b * settled_state.voltage_mv;

  const Config config{};
  SolverInput settle_input{};
  settle_input.parameters = production_parameters;
  settle_input.dt_ms = config.dt_ms;
  settle_input.threshold_mv = config.threshold_mv;
  settle_input.excitatory_reversal_mv =
      config.excitatory_reversal_mv;
  settle_input.gabaa_reversal_mv = config.gabaa_reversal_mv;
  for (int time = 0;
       time < clean_msi::kMsiEPhenotypeSettleMs; ++time) {
    settle_input.state = settled_state;
    const auto output =
        clean_msi::run_solver_batch(kDevice, {settle_input});
    REQUIRE(output.size() == 1);
    REQUIRE(!output[0].overflow);
    REQUIRE(!output[0].emitted);
    settled_state = output[0].state;
  }

  std::array<clean_msi::NeuronState, 2> states{
      settled_state, settled_state};
  std::array<std::vector<int>, 2> spike_steps{};
  for (int time = 0;
       time < clean_msi::kMsiEPhenotypeDriveMs; ++time) {
    std::vector<SolverInput> inputs(2);
    for (int condition = 0; condition < 2; ++condition) {
      inputs[static_cast<std::size_t>(condition)].state =
          states[static_cast<std::size_t>(condition)];
      inputs[static_cast<std::size_t>(condition)].parameters =
          condition == 0 ? production_parameters
                         : adaptation_off_parameters;
      inputs[static_cast<std::size_t>(condition)].additive_current =
          clean_msi::kMsiEPhenotypeDriveCurrent;
      inputs[static_cast<std::size_t>(condition)].dt_ms =
          config.dt_ms;
      inputs[static_cast<std::size_t>(condition)].threshold_mv =
          config.threshold_mv;
      inputs[static_cast<std::size_t>(condition)]
          .excitatory_reversal_mv =
          config.excitatory_reversal_mv;
      inputs[static_cast<std::size_t>(condition)]
          .gabaa_reversal_mv = config.gabaa_reversal_mv;
    }

    const auto outputs =
        clean_msi::run_solver_batch(kDevice, inputs);
    REQUIRE(outputs.size() == 2);
    for (int condition = 0; condition < 2; ++condition) {
      const SolverOutput& output =
          outputs[static_cast<std::size_t>(condition)];
      REQUIRE(!output.overflow);
      states[static_cast<std::size_t>(condition)] = output.state;
      for (int spike = 0; spike < output.spike_count; ++spike) {
        spike_steps[static_cast<std::size_t>(condition)]
            .push_back(time);
      }
    }
  }

  const std::vector<int>& production_spikes = spike_steps[0];
  const std::vector<int>& adaptation_off_spikes = spike_steps[1];
  REQUIRE(production_spikes.size() >= 2);
  REQUIRE(adaptation_off_spikes.size() >= 2);
  REQUIRE(production_spikes[0] == adaptation_off_spikes[0]);
  REQUIRE(production_spikes[1] >= adaptation_off_spikes[1]);
  const std::size_t production_post_first =
      production_spikes.size() - 1;
  const std::size_t adaptation_off_post_first =
      adaptation_off_spikes.size() - 1;
  REQUIRE(production_post_first <= adaptation_off_post_first);
  REQUIRE(production_spikes[1] > adaptation_off_spikes[1] ||
          production_post_first < adaptation_off_post_first);

  std::cout
      << "[SCIENCE] msi_e_adaptation first_ms="
      << production_spikes[0]
      << " next_ms_adapted=" << production_spikes[1]
      << " next_ms_off=" << adaptation_off_spikes[1]
      << " post_first_adapted=" << production_post_first
      << " post_first_off=" << adaptation_off_post_first << '\n';
}

void require_path_bounds_and_pairing(
    const PathDiagnostics& path, PlasticPath kind,
    const std::string& label) {
  const std::string path_label =
      label + "." + path_name(kind);
  require_finite(path.weight_min, path_label + ".weight_min");
  require_finite(path.weight_max, path_label + ".weight_max");
  require_finite(path.weight_mean, path_label + ".weight_mean");
  REQUIRE_MESSAGE(path.weight_min >= 0.0f,
                  path_label + ".weight_min is negative");
  const float maximum_weight =
      kind == PlasticPath::kExcitatoryToExcitatory ||
              kind == PlasticPath::kExcitatoryToInhibitory
          ? 0.5f
          : 1.0f;
  REQUIRE_MESSAGE(path.weight_max <= maximum_weight,
                  path_label + ".weight_max exceeds pathway bound");
  REQUIRE_MESSAGE(path.weight_min <= path.weight_mean,
                  path_label + ".mean below minimum");
  REQUIRE_MESSAGE(path.weight_mean <= path.weight_max,
                  path_label + ".mean above maximum");
  REQUIRE_MESSAGE(path.active_contacts > 0,
                  path_label + ".active_contacts is zero");
  REQUIRE_MESSAGE(path.active_contacts <= maximum_contacts(kind),
                  path_label + ".active_contacts exceeds anatomy");
  REQUIRE_MESSAGE(path.changed_contacts >= 0,
                  path_label + ".changed_contacts is negative");
  REQUIRE_MESSAGE(path.changed_contacts <= path.active_contacts,
                  path_label + ".changed_contacts exceeds active contacts");
  REQUIRE_MESSAGE(path.paired_mask_mismatches == 0,
                  path_label + " AMPA/NMDA masks diverged");
  REQUIRE_MESSAGE(path.maximum_low_weight_dwell >= 0,
                  path_label + ".maximum_low_weight_dwell is negative");
  REQUIRE_MESSAGE(
      path.arrived_source_spikes <= path.scheduled_source_spikes,
      path_label + " has more arrivals than scheduled source spikes");
}

void require_population_state_finite(
    const PopulationDiagnostics& population, Population kind,
    const std::string& label) {
  const std::string population_label =
      label + "." + population_name(kind);
  REQUIRE_MESSAGE(population.finite,
                  population_label + " reports nonfinite state");
  require_finite(population.voltage_min_mv,
                 population_label + ".voltage_min_mv");
  require_finite(population.voltage_max_mv,
                 population_label + ".voltage_max_mv");
  require_finite(population.recovery_min,
                 population_label + ".recovery_min");
  require_finite(population.recovery_max,
                 population_label + ".recovery_max");
  REQUIRE_MESSAGE(
      population.voltage_min_mv <= population.voltage_max_mv,
      population_label + " voltage extrema are reversed");
  REQUIRE_MESSAGE(
      population.recovery_min <= population.recovery_max,
      population_label + " recovery extrema are reversed");
  REQUIRE_MESSAGE(population.voltage_max_mv < 30.0f,
                  population_label + " retained suprathreshold state");
}

void require_diagnostics_exact(
    const TrainingDiagnostics& actual,
    const TrainingDiagnostics& expected,
    const std::string& label) {
  REQUIRE_MESSAGE(actual.presentation_index == expected.presentation_index,
                  label + ".presentation_index differs");
  REQUIRE_MESSAGE(actual.accepted_steps == expected.accepted_steps,
                  label + ".accepted_steps differs");
  REQUIRE_MESSAGE(actual.silent_iti_steps == expected.silent_iti_steps,
                  label + ".silent_iti_steps differs");
  REQUIRE_MESSAGE(
      actual.silent_iti_afferent_arrivals ==
          expected.silent_iti_afferent_arrivals,
      label + ".silent_iti_afferent_arrivals differs");
  REQUIRE_MESSAGE(
      actual.recurrent_plasticity_steps ==
          expected.recurrent_plasticity_steps,
      label + ".recurrent_plasticity_steps differs");
  REQUIRE_MESSAGE(actual.pruned_contacts == expected.pruned_contacts,
                  label + ".pruned_contacts differs");
  for (const Population population : kAllPopulations) {
    const auto& lhs =
        population_diagnostics(actual, population);
    const auto& rhs =
        population_diagnostics(expected, population);
    const std::string item =
        label + "." + population_name(population);
    REQUIRE_MESSAGE(lhs.voltage_min_mv == rhs.voltage_min_mv,
                    item + ".voltage_min_mv differs");
    REQUIRE_MESSAGE(lhs.voltage_max_mv == rhs.voltage_max_mv,
                    item + ".voltage_max_mv differs");
    REQUIRE_MESSAGE(lhs.recovery_min == rhs.recovery_min,
                    item + ".recovery_min differs");
    REQUIRE_MESSAGE(lhs.recovery_max == rhs.recovery_max,
                    item + ".recovery_max differs");
    REQUIRE_MESSAGE(lhs.cumulative_spikes == rhs.cumulative_spikes,
                    item + ".cumulative_spikes differs");
    REQUIRE_MESSAGE(lhs.finite == rhs.finite,
                    item + ".finite differs");
  }
  for (const PlasticPath path : kAllPlasticPaths) {
    const auto& lhs = path_diagnostics(actual, path);
    const auto& rhs = path_diagnostics(expected, path);
    const std::string item = label + "." + path_name(path);
    REQUIRE_MESSAGE(lhs.weight_min == rhs.weight_min,
                    item + ".weight_min differs");
    REQUIRE_MESSAGE(lhs.weight_max == rhs.weight_max,
                    item + ".weight_max differs");
    REQUIRE_MESSAGE(lhs.weight_mean == rhs.weight_mean,
                    item + ".weight_mean differs");
    REQUIRE_MESSAGE(lhs.active_contacts == rhs.active_contacts,
                    item + ".active_contacts differs");
    REQUIRE_MESSAGE(lhs.changed_contacts == rhs.changed_contacts,
                    item + ".changed_contacts differs");
    REQUIRE_MESSAGE(
        lhs.paired_mask_mismatches == rhs.paired_mask_mismatches,
        item + ".paired_mask_mismatches differs");
    REQUIRE_MESSAGE(
        lhs.maximum_low_weight_dwell ==
            rhs.maximum_low_weight_dwell,
        item + ".maximum_low_weight_dwell differs");
    REQUIRE_MESSAGE(
        lhs.scheduled_source_spikes == rhs.scheduled_source_spikes,
        item + ".scheduled_source_spikes differs");
    REQUIRE_MESSAGE(
        lhs.arrived_source_spikes == rhs.arrived_source_spikes,
        item + ".arrived_source_spikes differs");
  }
}

void require_training_metrics_science(
    const TrainingMetrics& metrics, int expected_presentations,
    const std::string& label) {
  REQUIRE_MESSAGE(
      metrics.completed_presentations == expected_presentations,
      label + ".completed_presentations differs");
  REQUIRE_MESSAGE(metrics.total_a_spikes > 0,
                  label + ".total_a_spikes is zero");
  REQUIRE_MESSAGE(metrics.total_v_spikes > 0,
                  label + ".total_v_spikes is zero");
  REQUIRE_MESSAGE(metrics.total_e_spikes > 0,
                  label + ".total_e_spikes is zero");
  REQUIRE_MESSAGE(metrics.total_i_spikes > 0,
                  label + ".total_i_spikes is zero");
  for (const auto [field, value] :
       std::array<std::pair<const char*, float>, 4>{{
           {"mean_a_to_e", metrics.mean_a_to_e},
           {"mean_v_to_e", metrics.mean_v_to_e},
           {"mean_e_to_e", metrics.mean_e_to_e},
           {"mean_i_to_e", metrics.mean_i_to_e},
       }}) {
    require_finite(value, label + "." + field);
    REQUIRE_MESSAGE(value >= 0.0f,
                    label + "." + field + " is negative");
    REQUIRE_MESSAGE(value <= 1.0f,
                    label + "." + field + " exceeds bound");
  }
  REQUIRE_MESSAGE(metrics.mean_a_to_e > 0.0f,
                  label + ".mean_a_to_e collapsed");
  REQUIRE_MESSAGE(metrics.mean_v_to_e > 0.0f,
                  label + ".mean_v_to_e collapsed");
  const float sensory_ratio =
      metrics.mean_a_to_e / metrics.mean_v_to_e;
  REQUIRE_MESSAGE(
      sensory_ratio >= 0.10f && sensory_ratio <= 10.0f,
      label + " A/V pathways show gross collapse under training mixture");
  REQUIRE_MESSAGE(metrics.pruned_contacts == 0,
                  label + " pruned contacts before pruning eligibility");
  require_finite(metrics.elapsed_seconds, label + ".elapsed_seconds");
  REQUIRE_MESSAGE(metrics.elapsed_seconds > 0.0f,
                  label + ".elapsed_seconds is not positive");
}

void require_training_metrics_exact(
    const TrainingMetrics& actual, const TrainingMetrics& expected,
    const std::string& label) {
  REQUIRE_MESSAGE(
      actual.completed_presentations == expected.completed_presentations,
      label + ".completed_presentations differs");
  REQUIRE_MESSAGE(actual.total_a_spikes == expected.total_a_spikes,
                  label + ".total_a_spikes differs");
  REQUIRE_MESSAGE(actual.total_v_spikes == expected.total_v_spikes,
                  label + ".total_v_spikes differs");
  REQUIRE_MESSAGE(actual.total_e_spikes == expected.total_e_spikes,
                  label + ".total_e_spikes differs");
  REQUIRE_MESSAGE(actual.total_i_spikes == expected.total_i_spikes,
                  label + ".total_i_spikes differs");
  REQUIRE_MESSAGE(actual.mean_a_to_e == expected.mean_a_to_e,
                  label + ".mean_a_to_e differs");
  REQUIRE_MESSAGE(actual.mean_v_to_e == expected.mean_v_to_e,
                  label + ".mean_v_to_e differs");
  REQUIRE_MESSAGE(actual.mean_e_to_e == expected.mean_e_to_e,
                  label + ".mean_e_to_e differs");
  REQUIRE_MESSAGE(actual.mean_i_to_e == expected.mean_i_to_e,
                  label + ".mean_i_to_e differs");
  REQUIRE_MESSAGE(actual.pruned_contacts == expected.pruned_contacts,
                  label + ".pruned_contacts differs");
}

void require_count_parity(std::uint64_t actual, std::uint64_t expected,
                          const std::string& label) {
  const std::uint64_t maximum = std::max(actual, expected);
  const std::uint64_t minimum = std::min(actual, expected);
  const std::uint64_t allowance =
      std::max<std::uint64_t>(1u, maximum / 100u);
  REQUIRE_MESSAGE(maximum - minimum <= allowance,
                  label + " differs by more than 1%");
}

TrainingOptions training_options(int presentations,
                                 bool enable_pruning = true) {
  TrainingOptions options{};
  options.presentations = presentations;
  options.seed_count = 1;
  options.chunk_presentations =
      clean_msi::kTrainingChunkPresentations;
  options.enable_pruning = enable_pruning;
  return options;
}

void test_science_checked_topology_pearson() {
  double correlation = 0.0;
  REQUIRE(clean_msi::checked_pearson_correlation(
      {1.0, 2.0, 3.0, 4.0},
      {3.0, 5.0, 7.0, 9.0}, &correlation));
  REQUIRE(std::abs(correlation - 1.0) < 1.0e-12);
  REQUIRE(clean_msi::checked_pearson_correlation(
      {1.0, 2.0, 3.0, 4.0},
      {9.0, 7.0, 5.0, 3.0}, &correlation));
  REQUIRE(std::abs(correlation + 1.0) < 1.0e-12);

  REQUIRE(!clean_msi::checked_pearson_correlation(
      {1.0}, {2.0}, &correlation));
  REQUIRE(std::isnan(correlation));
  REQUIRE(!clean_msi::checked_pearson_correlation(
      {1.0, 2.0}, {1.0}, &correlation));
  REQUIRE(std::isnan(correlation));
  REQUIRE(!clean_msi::checked_pearson_correlation(
      {1.0, 1.0, 1.0}, {1.0, 2.0, 3.0}, &correlation));
  REQUIRE(std::isnan(correlation));
  REQUIRE(!clean_msi::checked_pearson_correlation(
      {1.0, 2.0, 3.0}, {4.0, 4.0, 4.0}, &correlation));
  REQUIRE(std::isnan(correlation));
  REQUIRE(!clean_msi::checked_pearson_correlation(
      {1.0, std::numeric_limits<double>::quiet_NaN(), 3.0},
      {1.0, 2.0, 3.0}, &correlation));
  REQUIRE(std::isnan(correlation));
  REQUIRE(!clean_msi::checked_pearson_correlation(
      {1.0, 2.0}, {2.0, 3.0}, nullptr));
}

void test_science_rf_signed_mean_zero_floor() {
  const std::array<double, 2> signed_samples{-10.0, 8.0};
  const double signed_sum =
      std::accumulate(
          signed_samples.begin(), signed_samples.end(), 0.0);
  const double per_trial_floored_mean =
      0.5 *
      (std::max(signed_samples[0], 0.0) +
       std::max(signed_samples[1], 0.0));
  const float production_rate =
      clean_msi::finalize_rf_signed_rate_mean(
          signed_sum,
          static_cast<int>(signed_samples.size()));
  REQUIRE(production_rate == 0.0f);
  REQUIRE(per_trial_floored_mean == 4.0);
  REQUIRE(
      static_cast<double>(production_rate) !=
      per_trial_floored_mean);

  const std::array<double, 2> positive_mean_samples{-10.0, 12.0};
  const double positive_signed_sum =
      std::accumulate(
          positive_mean_samples.begin(),
          positive_mean_samples.end(), 0.0);
  const double positive_per_trial_floored_mean =
      0.5 *
      (std::max(positive_mean_samples[0], 0.0) +
       std::max(positive_mean_samples[1], 0.0));
  const float positive_production_rate =
      clean_msi::finalize_rf_signed_rate_mean(
          positive_signed_sum,
          static_cast<int>(positive_mean_samples.size()));
  REQUIRE(positive_production_rate == 1.0f);
  REQUIRE(positive_per_trial_floored_mean == 6.0);
  REQUIRE(
      static_cast<double>(positive_production_rate) !=
      positive_per_trial_floored_mean);
}

void test_science_weight_snapshot_and_initial_weight_routing() {
  const std::vector<int> devices = available_devices();
  REQUIRE_MESSAGE(!devices.empty(),
                  "weight snapshot audit requires CUDA");
  constexpr int kDevice = 0;
  constexpr int kSeedCount = 5;
  constexpr int kPresentations = 2000;
  constexpr std::array<std::size_t, 7> kPathContactCounts{
      32400u, 32400u, 32400u, 10800u, 10800u, 10800u, 10800u};
  constexpr std::array<int, 7> kKnownActiveContacts{
      8100, 8100, 6480, 2700, 2700, -1, 2160};

  Config config = calibration_for(kDevice).result.config;
  config.seed = 0;
  NativeModel model(config, kDevice, kSeedCount);
  const auto initial_state = model.weight_state_audit();
  REQUIRE(initial_state.size() == kSeedCount);

  std::vector<FrozenWeightAudit> initial_snapshots;
  std::vector<FrozenWeightAudit> initial_current;
  initial_snapshots.reserve(kSeedCount);
  initial_current.reserve(kSeedCount);
  for (int seed = 0; seed < kSeedCount; ++seed) {
    const auto& state =
        initial_state[static_cast<std::size_t>(seed)];
    REQUIRE(state.seed_index == seed);
    REQUIRE(state.global_seed ==
            config.seed + static_cast<std::uint64_t>(seed));
    REQUIRE(state.initial.seed_index == seed);
    REQUIRE(state.initial.global_seed == state.global_seed);
    REQUIRE(state.trained.seed_index == seed);
    REQUIRE(state.trained.global_seed == state.global_seed);
    initial_snapshots.push_back(state.initial);
    initial_current.push_back(state.trained);

    for (int path = 0; path < 7; ++path) {
      const auto& values =
          state.initial.paths[static_cast<std::size_t>(path)];
      REQUIRE(values.weights.size() ==
              kPathContactCounts[static_cast<std::size_t>(path)]);
      REQUIRE(values.masks.size() == values.weights.size());
      int active_contacts = 0;
      for (std::size_t contact = 0;
           contact < values.weights.size(); ++contact) {
        const float weight = values.weights[contact];
        const std::uint8_t expected_mask =
            weight > 0.0f ? 1u : 0u;
        REQUIRE(values.masks[contact] == expected_mask);
        if (expected_mask == 0u) {
          REQUIRE(weight == 0.0f);
          continue;
        }
        ++active_contacts;
        require_finite(weight, "initial snapshot weight");
        if (path == 0 || path == 1 || path == 3 || path == 4) {
          REQUIRE(weight >= 0.05f);
          REQUIRE(weight < 0.15f);
        } else if (path == 2) {
          REQUIRE(weight >= 0.01f);
          REQUIRE(weight < 0.03f);
        } else {
          REQUIRE(weight >= 0.02f);
          REQUIRE(weight < 0.05f);
        }
      }
      const int expected_active =
          kKnownActiveContacts[static_cast<std::size_t>(path)];
      if (expected_active >= 0) {
        REQUIRE(active_contacts == expected_active);
      } else {
        REQUIRE(active_contacts > 0);
      }
    }
  }
  require_exact_frozen_weight_audit(
      initial_snapshots, initial_current,
      "weight_state.initial_matches_initializer");
  require_exact_frozen_weight_audit(
      model.frozen_weight_audit(), initial_current,
      "weight_state.current_audit_at_initialization");

  TrainingOptions train_options{};
  train_options.presentations = kPresentations;
  train_options.seed_count = kSeedCount;
  train_options.chunk_presentations =
      clean_msi::kTrainingChunkPresentations;
  train_options.enable_pruning = false;
  const TrainingMetrics training = model.train(train_options);
  REQUIRE(training.completed_presentations == kPresentations);
  REQUIRE(training.pruned_contacts == 0);

  const auto trained_state = model.weight_state_audit();
  REQUIRE(trained_state.size() == kSeedCount);
  std::vector<FrozenWeightAudit> retained_initial;
  retained_initial.reserve(kSeedCount);
  bool auditory_plastic_weight_changed = false;
  bool visual_plastic_weight_changed = false;
  for (int seed = 0; seed < kSeedCount; ++seed) {
    const auto& before =
        initial_state[static_cast<std::size_t>(seed)];
    const auto& after =
        trained_state[static_cast<std::size_t>(seed)];
    REQUIRE(after.seed_index == before.seed_index);
    REQUIRE(after.global_seed == before.global_seed);
    retained_initial.push_back(after.initial);
    for (int path = 0; path < 7; ++path) {
      const auto& initial_path =
          after.initial.paths[static_cast<std::size_t>(path)];
      const auto& trained_path =
          after.trained.paths[static_cast<std::size_t>(path)];
      REQUIRE(initial_path.masks == trained_path.masks);
      if (path == 3 || path == 4) {
        REQUIRE(initial_path.weights == trained_path.weights);
      }
      if (path == 0 &&
          initial_path.weights != trained_path.weights) {
        auditory_plastic_weight_changed = true;
      }
      if (path == 1 &&
          initial_path.weights != trained_path.weights) {
        visual_plastic_weight_changed = true;
      }
    }
  }
  require_exact_frozen_weight_audit(
      retained_initial, initial_snapshots,
      "weight_state.initial_snapshot_immutable");
  REQUIRE(auditory_plastic_weight_changed);
  REQUIRE(visual_plastic_weight_changed);

  const auto state_before_frozen = model.weight_state_audit();
  const auto diagnostics_before_frozen = model.diagnostics();
  ControlledCondition trained_condition{};
  trained_condition.auditory_present = true;
  trained_condition.visual_present = true;
  trained_condition.auditory_location_deg = 0.0f;
  trained_condition.visual_location_deg = 0.0f;
  trained_condition.auditory_rate_hz = 50.0f;
  trained_condition.visual_rate_hz = 50.0f;
  trained_condition.physical_soa_ms = -50.0f;
  trained_condition.control = CausalControl::kNone;
  trained_condition.label = 1;
  trained_condition.random_group = 7000;
  trained_condition.use_initial_weights = false;
  ControlledCondition initial_condition = trained_condition;
  initial_condition.use_initial_weights = true;
  constexpr int kTrialsPerCondition = 16;
  const FrozenTrialBatch paired = model.run_frozen_trials(
      {trained_condition, initial_condition},
      kTrialsPerCondition, 0x494E495449414Cull, 300);
  REQUIRE(paired.model_seed_count == kSeedCount);
  REQUIRE(paired.condition_count == 2);
  REQUIRE(paired.trials_per_condition == kTrialsPerCondition);

  int differing_trials = 0;
  for (int seed = 0; seed < kSeedCount; ++seed) {
    for (int trial = 0; trial < kTrialsPerCondition; ++trial) {
      const std::size_t trained_index =
          static_cast<std::size_t>(
              (seed * 2) * kTrialsPerCondition + trial);
      const std::size_t initial_index =
          static_cast<std::size_t>(
              (seed * 2 + 1) * kTrialsPerCondition + trial);
      const FrozenTrialResult& trained_trial =
          paired.trials[trained_index];
      const FrozenTrialResult& initial_trial =
          paired.trials[initial_index];
      REQUIRE(trained_trial.seed_index == initial_trial.seed_index);
      REQUIRE(trained_trial.trial_index == initial_trial.trial_index);
      REQUIRE(trained_trial.label == initial_trial.label);
      REQUIRE(trained_trial.physical_soa_ms ==
              initial_trial.physical_soa_ms);
      REQUIRE(trained_trial.auditory_latency_ms ==
              initial_trial.auditory_latency_ms);
      REQUIRE(trained_trial.visual_latency_ms ==
              initial_trial.visual_latency_ms);
      REQUIRE(trained_trial.auditory_received_onset_ms ==
              initial_trial.auditory_received_onset_ms);
      REQUIRE(trained_trial.visual_received_onset_ms ==
              initial_trial.visual_received_onset_ms);
      REQUIRE(trained_trial.earlier_received_onset_step ==
              initial_trial.earlier_received_onset_step);
      REQUIRE(
          trained_trial.background_afferent_arrivals_during_burn_in ==
          initial_trial.background_afferent_arrivals_during_burn_in);
      REQUIRE(trained_trial.finite);
      REQUIRE(initial_trial.finite);
      if (trained_trial.excitatory_spikes_per_relative_ms !=
              initial_trial.excitatory_spikes_per_relative_ms ||
          trained_trial.excitatory_response_spikes_per_neuron !=
              initial_trial.excitatory_response_spikes_per_neuron) {
        ++differing_trials;
      }
    }
  }
  REQUIRE(differing_trials > 0);

  const auto state_after_frozen = model.weight_state_audit();
  REQUIRE(state_after_frozen.size() == state_before_frozen.size());
  std::vector<FrozenWeightAudit> initial_before_frozen;
  std::vector<FrozenWeightAudit> trained_before_frozen;
  std::vector<FrozenWeightAudit> initial_after_frozen;
  std::vector<FrozenWeightAudit> trained_after_frozen;
  for (std::size_t seed = 0;
       seed < state_before_frozen.size(); ++seed) {
    initial_before_frozen.push_back(
        state_before_frozen[seed].initial);
    trained_before_frozen.push_back(
        state_before_frozen[seed].trained);
    initial_after_frozen.push_back(
        state_after_frozen[seed].initial);
    trained_after_frozen.push_back(
        state_after_frozen[seed].trained);
  }
  require_exact_frozen_weight_audit(
      initial_after_frozen, initial_before_frozen,
      "weight_state.initial_frozen_neutrality");
  require_exact_frozen_weight_audit(
      trained_after_frozen, trained_before_frozen,
      "weight_state.trained_frozen_neutrality");
  const auto diagnostics_after_frozen = model.diagnostics();
  REQUIRE(diagnostics_after_frozen.size() ==
          diagnostics_before_frozen.size());
  for (std::size_t seed = 0;
       seed < diagnostics_before_frozen.size(); ++seed) {
    require_diagnostics_exact(
        diagnostics_after_frozen[seed],
        diagnostics_before_frozen[seed],
        "weight_state.frozen_diagnostics.seed" +
            std::to_string(seed));
  }

  EvaluationOptions topology_options{};
  topology_options.observer_training_trials = 4;
  topology_options.observer_holdout_trials = 4;
  topology_options.trials_per_tbw_soa = 1;
  topology_options.trials_per_sbw_condition = 1;
  topology_options.trials_per_rf_location = 100;
  topology_options.trials_per_inverse_condition = 1;
  topology_options.burn_in_ms = 300;
  topology_options.evaluation_seed = 0x544F504F4C4F4759ull;
  topology_options.control = CausalControl::kNone;
  const EvaluationMetrics baseline_evaluation =
      model.evaluate(topology_options);
  const auto& topology_summary =
      baseline_evaluation.topology_refinement;
  std::cout << std::setprecision(9)
            << "[SCIENCE] topology_refinement"
            << " evaluated=" << topology_summary.evaluated
            << " passed=" << topology_summary.passed
            << " valid=" << topology_summary.all_values_valid
            << " dAE="
            << topology_summary
                   .mean_auditory_to_excitatory_delta
            << " nAE="
            << topology_summary
                   .auditory_to_excitatory_positive_seed_count
            << " dVE="
            << topology_summary
                   .mean_visual_to_excitatory_delta
            << " nVE="
            << topology_summary
                   .visual_to_excitatory_positive_seed_count
            << " dEE=" << topology_summary.mean_recurrent_delta
            << " nEE="
            << topology_summary.recurrent_positive_seed_count
            << " dI="
            << topology_summary.mean_effective_inhibitory_delta
            << " nI="
            << topology_summary
                   .effective_inhibitory_positive_seed_count
            << " dRF="
            << topology_summary.mean_rf_mismatch_delta_deg
            << " nRF="
            << topology_summary.rf_mismatch_negative_seed_count
            << " reason=" << topology_summary.reason << '\n';
  for (const auto& seed : baseline_evaluation.per_seed) {
    const auto& topology = seed.topology;
    std::cout << std::setprecision(9)
              << "[SCIENCE] topology_seed=" << seed.seed_index
              << " dAE="
              << topology
                     .auditory_to_excitatory_proximity_efficacy_delta
              << " dVE="
              << topology
                     .visual_to_excitatory_proximity_efficacy_delta
              << " dEE="
              << topology.recurrent_proximity_efficacy_delta
              << " dI="
              << topology
                     .effective_inhibitory_order_delta_explicit
              << " dRF="
              << topology.median_rf_center_mismatch_delta_deg
              << " rf_order_initial_A="
              << topology.initial_state
                     .auditory_rf_order_correlation
              << " rf_order_initial_V="
              << topology.initial_state
                     .visual_rf_order_correlation
              << " rf_order_trained_A="
              << topology.trained_state
                     .auditory_rf_order_correlation
              << " rf_order_trained_V="
              << topology.trained_state
                     .visual_rf_order_correlation
              << '\n';
  }
  REQUIRE(baseline_evaluation.per_seed.size() == kSeedCount);
  REQUIRE(topology_summary.seed_count == kSeedCount);
  REQUIRE(topology_summary.evaluated);
  REQUIRE(topology_summary.passed);
  REQUIRE(topology_summary.all_values_valid);
  REQUIRE(topology_summary.reason.empty());
  REQUIRE(
      topology_summary.mean_auditory_to_excitatory_delta > 0.0f);
  REQUIRE(
      topology_summary
          .auditory_to_excitatory_positive_seed_count >= 4);
  REQUIRE(
      topology_summary.mean_visual_to_excitatory_delta > 0.0f);
  REQUIRE(
      topology_summary
          .visual_to_excitatory_positive_seed_count >= 4);
  REQUIRE(topology_summary.mean_recurrent_delta > 0.0f);
  REQUIRE(topology_summary.recurrent_positive_seed_count >= 4);
  REQUIRE(
      topology_summary.mean_effective_inhibitory_delta > 0.0f);
  REQUIRE(
      topology_summary
          .effective_inhibitory_positive_seed_count >= 4);
  REQUIRE(topology_summary.mean_rf_mismatch_delta_deg < 0.0f);
  REQUIRE(topology_summary.rf_mismatch_negative_seed_count >= 4);
  for (const auto& seed : baseline_evaluation.per_seed) {
    REQUIRE(seed.topology.initial_trained_masks_exact);
    REQUIRE(seed.topology.auditory_to_inhibitory_weights_exact);
    REQUIRE(seed.topology.visual_to_inhibitory_weights_exact);
    REQUIRE(seed.topology.all_required_metrics_valid);
  }

  Config plasticity_off_config = config;
  plasticity_off_config.training_cohort =
      clean_msi::TrainingCohort::kPlasticityOff;
  NativeModel plasticity_off_model(
      plasticity_off_config, kDevice, kSeedCount);
  const auto plasticity_off_initial_state =
      plasticity_off_model.weight_state_audit();
  std::vector<FrozenWeightAudit> plasticity_off_initial;
  std::vector<FrozenWeightAudit> plasticity_off_current;
  for (const auto& state : plasticity_off_initial_state) {
    plasticity_off_initial.push_back(state.initial);
    plasticity_off_current.push_back(state.trained);
  }
  require_exact_frozen_weight_audit(
      plasticity_off_initial, initial_snapshots,
      "plasticity_off.initial_matches_baseline");
  require_exact_frozen_weight_audit(
      plasticity_off_current, plasticity_off_initial,
      "plasticity_off.current_matches_initial_before_training");
  const TrainingMetrics plasticity_off_training =
      plasticity_off_model.train(train_options);
  REQUIRE(
      plasticity_off_training.completed_presentations ==
      kPresentations);
  REQUIRE(plasticity_off_training.pruned_contacts == 0);
  const auto plasticity_off_trained_state =
      plasticity_off_model.weight_state_audit();
  std::vector<FrozenWeightAudit> plasticity_off_retained_initial;
  std::vector<FrozenWeightAudit> plasticity_off_trained;
  for (const auto& state : plasticity_off_trained_state) {
    plasticity_off_retained_initial.push_back(state.initial);
    plasticity_off_trained.push_back(state.trained);
  }
  require_exact_frozen_weight_audit(
      plasticity_off_retained_initial, plasticity_off_initial,
      "plasticity_off.initial_snapshot_immutable");
  require_exact_frozen_weight_audit(
      plasticity_off_trained, plasticity_off_initial,
      "plasticity_off.trained_weights_unchanged");

  const EvaluationMetrics plasticity_off_evaluation =
      plasticity_off_model.evaluate(topology_options);
  const auto& plasticity_off_summary =
      plasticity_off_evaluation.topology_refinement;
  std::cout << std::setprecision(9)
            << "[SCIENCE] topology_plasticity_off"
            << " evaluated=" << plasticity_off_summary.evaluated
            << " passed=" << plasticity_off_summary.passed
            << " valid="
            << plasticity_off_summary.all_values_valid
            << " reason=" << plasticity_off_summary.reason << '\n';
  REQUIRE(
      plasticity_off_evaluation.per_seed.size() ==
      kSeedCount);
  REQUIRE(!plasticity_off_summary.evaluated);
  REQUIRE(!plasticity_off_summary.passed);
  REQUIRE(
      plasticity_off_summary.reason ==
      "training_cohort_not_baseline");
  int exactly_paired_rf_seeds = 0;
  for (const auto& seed : plasticity_off_evaluation.per_seed) {
    const auto& topology = seed.topology;
    REQUIRE(topology.initial_trained_masks_exact);
    REQUIRE(topology.auditory_to_inhibitory_weights_exact);
    REQUIRE(topology.visual_to_inhibitory_weights_exact);
    REQUIRE(topology.all_required_metrics_valid);
    REQUIRE(
        topology
            .auditory_to_excitatory_proximity_efficacy_delta ==
        0.0f);
    REQUIRE(
        topology
            .visual_to_excitatory_proximity_efficacy_delta ==
        0.0f);
    REQUIRE(
        topology.recurrent_proximity_efficacy_delta == 0.0f);
    REQUIRE(
        topology.effective_inhibitory_order_delta_explicit ==
        0.0f);
    REQUIRE(
        topology.median_rf_center_mismatch_delta_deg == 0.0f);
    REQUIRE(
        topology.initial.auditory_response_hz ==
        topology.trained.auditory_response_hz);
    REQUIRE(
        topology.initial.visual_response_hz ==
        topology.trained.visual_response_hz);
    REQUIRE(
        topology.initial.auditory_rf_center_deg ==
        topology.trained.auditory_rf_center_deg);
    REQUIRE(
        topology.initial.visual_rf_center_deg ==
        topology.trained.visual_rf_center_deg);
    REQUIRE(
        topology.initial.rf_center_mismatch_deg ==
        topology.trained.rf_center_mismatch_deg);
    REQUIRE(
        topology.initial_state
            .auditory_to_excitatory_proximity_efficacy_correlation ==
        topology.trained_state
            .auditory_to_excitatory_proximity_efficacy_correlation);
    REQUIRE(
        topology.initial_state
            .visual_to_excitatory_proximity_efficacy_correlation ==
        topology.trained_state
            .visual_to_excitatory_proximity_efficacy_correlation);
    REQUIRE(
        topology.initial_state
            .recurrent_proximity_efficacy_correlation ==
        topology.trained_state
            .recurrent_proximity_efficacy_correlation);
    REQUIRE(
        topology.initial_state
            .effective_inhibitory_order_correlation ==
        topology.trained_state
            .effective_inhibitory_order_correlation);
    REQUIRE(
        topology.initial_state.auditory_rf_order_correlation ==
        topology.trained_state.auditory_rf_order_correlation);
    REQUIRE(
        topology.initial_state.visual_rf_order_correlation ==
        topology.trained_state.visual_rf_order_correlation);
    ++exactly_paired_rf_seeds;
  }
  REQUIRE(exactly_paired_rf_seeds == kSeedCount);

  std::cout
      << "[SCIENCE] weight_snapshot seeds=" << kSeedCount
      << " presentations=" << training.completed_presentations
      << " pruned=" << training.pruned_contacts
      << " paired_trials="
      << kSeedCount * kTrialsPerCondition
      << " differing_trials=" << differing_trials
      << " exact_plasticity_off_rf_seeds="
      << exactly_paired_rf_seeds << '\n';
}

void test_training_initial_scaffold_and_masks() {
  const std::vector<int> devices = available_devices();
  REQUIRE_MESSAGE(!devices.empty(),
                  "initial scaffold test requires CUDA");
  for (const int device : devices) {
    const Config config = calibration_for(device).result.config;
    NativeModel model(config, device, 1);
    const auto diagnostics = model.diagnostics();
    REQUIRE(diagnostics.size() == 1u);
    const TrainingDiagnostics& initial = diagnostics[0];
    const std::string label =
        "cuda" + std::to_string(device) + " initial";
    REQUIRE(initial.presentation_index == 0);
    REQUIRE(initial.accepted_steps == 0u);
    REQUIRE(initial.silent_iti_steps == 0u);
    REQUIRE(initial.silent_iti_afferent_arrivals == 0u);
    REQUIRE(initial.recurrent_plasticity_steps == 0u);
    REQUIRE(initial.pruned_contacts == 0);
    for (const Population population : kAllPopulations) {
      const auto& state =
          population_diagnostics(initial, population);
      require_population_state_finite(state, population, label);
      REQUIRE_MESSAGE(
          state.cumulative_spikes == 0u,
          label + "." + population_name(population) +
              " initial spikes are nonzero");
    }
    for (const PlasticPath path : kAllPlasticPaths) {
      const auto& state = path_diagnostics(initial, path);
      require_path_bounds_and_pairing(state, path, label);
      REQUIRE_MESSAGE(
          state.weight_mean > 0.0f,
          label + "." + path_name(path) +
              " has no weak anatomical scaffold");
      REQUIRE_MESSAGE(
          state.changed_contacts == 0,
          label + "." + path_name(path) +
              " changed before training");
      REQUIRE_MESSAGE(
          state.maximum_low_weight_dwell == 0,
          label + "." + path_name(path) +
              " accumulated pruning dwell before training");
      REQUIRE_MESSAGE(
          state.scheduled_source_spikes == 0u &&
              state.arrived_source_spikes == 0u,
          label + "." + path_name(path) +
              " has delay traffic before training");
    }
    REQUIRE(
        path_diagnostics(
            initial, PlasticPath::kAuditoryToExcitatory)
            .active_contacts ==
        clean_msi::kFeedforwardScaffoldInDegree *
            clean_msi::kExcitatoryNeurons);
    REQUIRE(
        path_diagnostics(
            initial, PlasticPath::kVisualToExcitatory)
            .active_contacts ==
        clean_msi::kFeedforwardScaffoldInDegree *
            clean_msi::kExcitatoryNeurons);
    REQUIRE(
        path_diagnostics(
            initial, PlasticPath::kExcitatoryToExcitatory)
            .active_contacts ==
        clean_msi::kRecurrentExcitatoryScaffoldInDegree *
            clean_msi::kExcitatoryNeurons);
    REQUIRE(
        path_diagnostics(
            initial, PlasticPath::kAuditoryToInhibitory)
            .active_contacts ==
        clean_msi::kFeedforwardScaffoldInDegree *
            clean_msi::kInhibitoryNeurons);
    REQUIRE(
        path_diagnostics(
            initial, PlasticPath::kVisualToInhibitory)
            .active_contacts ==
        clean_msi::kFeedforwardScaffoldInDegree *
            clean_msi::kInhibitoryNeurons);
    REQUIRE(
        path_diagnostics(
            initial, PlasticPath::kInhibitoryToExcitatory)
            .active_contacts ==
        clean_msi::kInhibitoryScaffoldInDegree *
            clean_msi::kExcitatoryNeurons);
  }
}

void test_training_requires_calibration() {
  const std::vector<int> devices = available_devices();
  REQUIRE_MESSAGE(!devices.empty(),
                  "training constructor test requires CUDA");
  for (const int device : devices) {
    bool rejected = false;
    try {
      NativeModel model(Config{}, device, 1);
    } catch (const std::invalid_argument&) {
      rejected = true;
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    REQUIRE_MESSAGE(
        rejected,
        "cuda" + std::to_string(device) +
            " accepted an uncalibrated training config");
  }
}

void test_one_presentation_timing_and_afferent_silence() {
  const std::vector<int> devices = available_devices();
  REQUIRE_MESSAGE(!devices.empty(),
                  "one-presentation timing test requires CUDA");
  for (const int device : devices) {
    const Config config = calibration_for(device).result.config;
    NativeModel model(config, device, 1);
    const TrainingMetrics metrics =
        model.train(training_options(1, false));
    REQUIRE(metrics.completed_presentations == 1);
    const auto diagnostics = model.diagnostics();
    REQUIRE(diagnostics.size() == 1u);
    const TrainingDiagnostics& state = diagnostics[0];
    const std::string label =
        "cuda" + std::to_string(device) +
        " one-presentation timing";
    REQUIRE(state.presentation_index == 1);
    REQUIRE(state.accepted_steps > 250u);
    REQUIRE(state.silent_iti_steps == 250u);
    REQUIRE_MESSAGE(
        state.silent_iti_afferent_arrivals == 0u,
        label +
            " received sensory/background afferents during silent ITI");
    const std::array<std::uint64_t, clean_msi::kPopulationCount>
        metric_spikes{
            metrics.total_a_spikes,
            metrics.total_v_spikes,
            metrics.total_e_spikes,
            metrics.total_i_spikes,
        };
    for (const Population population : kAllPopulations) {
      const PopulationDiagnostics& population_state =
          population_diagnostics(state, population);
      require_population_state_finite(
          population_state, population, label);
      REQUIRE(
          population_state.cumulative_spikes ==
          metric_spikes[population_index(population)]);
      const int neuron_count =
          population == Population::kInhibitory
              ? clean_msi::kInhibitoryNeurons
              : clean_msi::kExcitatoryNeurons;
      const double firing_rate_hz =
          1000.0 *
          static_cast<double>(
              population_state.cumulative_spikes) /
          (static_cast<double>(state.accepted_steps) *
           static_cast<double>(neuron_count));
      REQUIRE_MESSAGE(
          firing_rate_hz > 0.0 && firing_rate_hz < 200.0,
          label + "." + population_name(population) +
              " firing rate outside stable biological range");
    }
    for (const PlasticPath path : kAllPlasticPaths) {
      const PathDiagnostics& projection =
          path_diagnostics(state, path);
      require_path_bounds_and_pairing(
          projection, path, label);
      REQUIRE_MESSAGE(
          projection.scheduled_source_spikes ==
              projection.arrived_source_spikes,
          label + "." + path_name(path) +
              " delay ring did not flush");
    }
  }
}

void test_training_replay_activity_and_cuda_parity() {
  const std::vector<int> devices = available_devices();
  REQUIRE_MESSAGE(!devices.empty(),
                  "training replay test requires CUDA");
  std::vector<TrainingMetrics> device_metrics;
  std::vector<TrainingDiagnostics> device_diagnostics;
  device_metrics.reserve(devices.size());
  device_diagnostics.reserve(devices.size());
  for (const int device : devices) {
    const Config config = calibration_for(device).result.config;
    NativeModel first(config, device, 1);
    NativeModel second(config, device, 1);
    REQUIRE(first.device() == device);
    REQUIRE(second.device() == device);
    REQUIRE(first.seed_count() == 1);
    REQUIRE(second.seed_count() == 1);
    REQUIRE(first.config().calibrated);
    REQUIRE(second.config().calibrated);

    const TrainingOptions options = training_options(100, true);
    const TrainingMetrics first_metrics = first.train(options);
    const TrainingMetrics replay_metrics = second.train(options);
    require_training_metrics_science(
        first_metrics, 100,
        "cuda" + std::to_string(device) + " 100-presentation train");
    require_training_metrics_science(
        replay_metrics, 100,
        "cuda" + std::to_string(device) +
            " 100-presentation replay");
    require_training_metrics_exact(
        replay_metrics, first_metrics,
        "cuda" + std::to_string(device) +
            " deterministic training replay");
    const auto first_diagnostics = first.diagnostics();
    const auto replay_diagnostics = second.diagnostics();
    REQUIRE(first_diagnostics.size() == 1u);
    REQUIRE(replay_diagnostics.size() == 1u);
    const TrainingDiagnostics& trained = first_diagnostics[0];
    const std::string trained_label =
        "cuda" + std::to_string(device) +
        " 100-presentation diagnostics";
    require_diagnostics_exact(
        replay_diagnostics[0], trained,
        "cuda" + std::to_string(device) +
            " deterministic diagnostics replay");
    REQUIRE(trained.presentation_index == 100);
    REQUIRE(trained.accepted_steps > 0u);
    REQUIRE(trained.silent_iti_steps == 250u * 100u);
    REQUIRE_MESSAGE(
        trained.silent_iti_afferent_arrivals == 0u,
        trained_label +
            " received sensory/background afferents during silent ITI");
    REQUIRE(trained.silent_iti_steps < trained.accepted_steps);
    REQUIRE(trained.recurrent_plasticity_steps == 0u);
    REQUIRE(trained.pruned_contacts == 0);
    const std::array<std::uint64_t, clean_msi::kPopulationCount>
        metric_spikes{
            first_metrics.total_a_spikes,
            first_metrics.total_v_spikes,
            first_metrics.total_e_spikes,
            first_metrics.total_i_spikes,
        };
    for (const Population population : kAllPopulations) {
      const auto& state =
          population_diagnostics(trained, population);
      require_population_state_finite(
          state, population, trained_label);
      REQUIRE_MESSAGE(
          state.cumulative_spikes ==
              metric_spikes[population_index(population)],
          trained_label + "." + population_name(population) +
              " spike accounting differs");
      const int neuron_count =
          population == Population::kInhibitory
              ? clean_msi::kInhibitoryNeurons
              : clean_msi::kExcitatoryNeurons;
      const double firing_rate_hz =
          1000.0 * static_cast<double>(state.cumulative_spikes) /
          (static_cast<double>(trained.accepted_steps) *
           static_cast<double>(neuron_count));
      REQUIRE_MESSAGE(
          firing_rate_hz > 0.0 && firing_rate_hz < 200.0,
          trained_label + "." + population_name(population) +
              " firing rate outside stable biological range");
    }
    for (const PlasticPath path : kAllPlasticPaths) {
      const auto& state = path_diagnostics(trained, path);
      require_path_bounds_and_pairing(state, path, trained_label);
      REQUIRE_MESSAGE(
          state.scheduled_source_spikes > 0u,
          trained_label + "." + path_name(path) +
              " scheduled no source spikes");
      REQUIRE_MESSAGE(
          state.arrived_source_spikes ==
              state.scheduled_source_spikes,
          trained_label + "." + path_name(path) +
              " delay ring did not flush exactly");
      if (path == PlasticPath::kAuditoryToInhibitory ||
          path == PlasticPath::kVisualToInhibitory) {
        REQUIRE_MESSAGE(
            state.changed_contacts == 0,
            trained_label + "." + path_name(path) +
                " permanently frozen sensory-to-inhibitory "
                "pathway changed");
      } else if (
          path == PlasticPath::kExcitatoryToExcitatory ||
          path == PlasticPath::kExcitatoryToInhibitory) {
        REQUIRE_MESSAGE(
            state.changed_contacts == 0,
            trained_label + "." + path_name(path) +
                " recurrent pathway changed before maturation");
      } else {
        REQUIRE_MESSAGE(
            state.changed_contacts > 0,
            trained_label + "." + path_name(path) +
                " eligible local plasticity made no change");
      }
    }
    device_metrics.push_back(first_metrics);
    device_diagnostics.push_back(trained);
  }

  if (device_metrics.size() > 1u) {
    const TrainingMetrics& cuda0 = device_metrics[0];
    const TrainingMetrics& cuda1 = device_metrics[1];
    require_count_parity(cuda1.total_a_spikes, cuda0.total_a_spikes,
                         "cuda1/cuda0 auditory spikes");
    require_count_parity(cuda1.total_v_spikes, cuda0.total_v_spikes,
                         "cuda1/cuda0 visual spikes");
    require_count_parity(cuda1.total_e_spikes, cuda0.total_e_spikes,
                         "cuda1/cuda0 excitatory spikes");
    require_count_parity(cuda1.total_i_spikes, cuda0.total_i_spikes,
                         "cuda1/cuda0 inhibitory spikes");
    require_near(cuda1.mean_a_to_e, cuda0.mean_a_to_e,
                 1.0e-4f, 1.0e-2f,
                 "cuda1/cuda0 mean A->E");
    require_near(cuda1.mean_v_to_e, cuda0.mean_v_to_e,
                 1.0e-4f, 1.0e-2f,
                 "cuda1/cuda0 mean V->E");
    require_near(cuda1.mean_e_to_e, cuda0.mean_e_to_e,
                 1.0e-4f, 1.0e-2f,
                 "cuda1/cuda0 mean E->E");
    require_near(cuda1.mean_i_to_e, cuda0.mean_i_to_e,
                 1.0e-4f, 1.0e-2f,
                 "cuda1/cuda0 mean I->E");

    const TrainingDiagnostics& diagnostics0 =
        device_diagnostics[0];
    const TrainingDiagnostics& diagnostics1 =
        device_diagnostics[1];
    REQUIRE(diagnostics1.presentation_index ==
            diagnostics0.presentation_index);
    REQUIRE(diagnostics1.accepted_steps ==
            diagnostics0.accepted_steps);
    REQUIRE(diagnostics1.silent_iti_steps ==
            diagnostics0.silent_iti_steps);
    REQUIRE(diagnostics1.silent_iti_afferent_arrivals ==
            diagnostics0.silent_iti_afferent_arrivals);
    REQUIRE(diagnostics1.recurrent_plasticity_steps ==
            diagnostics0.recurrent_plasticity_steps);
    REQUIRE(diagnostics1.pruned_contacts ==
            diagnostics0.pruned_contacts);
    for (const Population population : kAllPopulations) {
      const auto& first =
          population_diagnostics(diagnostics0, population);
      const auto& second =
          population_diagnostics(diagnostics1, population);
      const std::string label =
          std::string("cuda1/cuda0 ") +
          population_name(population);
      require_count_parity(second.cumulative_spikes,
                           first.cumulative_spikes,
                           label + " cumulative spikes");
      require_near(second.voltage_min_mv, first.voltage_min_mv,
                   1.0e-2f, 1.0e-4f,
                   label + " voltage minimum");
      require_near(second.voltage_max_mv, first.voltage_max_mv,
                   1.0e-2f, 1.0e-4f,
                   label + " voltage maximum");
      require_near(second.recovery_min, first.recovery_min,
                   1.0e-2f, 1.0e-4f,
                   label + " recovery minimum");
      require_near(second.recovery_max, first.recovery_max,
                   1.0e-2f, 1.0e-4f,
                   label + " recovery maximum");
    }
    for (const PlasticPath path : kAllPlasticPaths) {
      const auto& first =
          path_diagnostics(diagnostics0, path);
      const auto& second =
          path_diagnostics(diagnostics1, path);
      const std::string label =
          std::string("cuda1/cuda0 ") + path_name(path);
      REQUIRE(second.active_contacts == first.active_contacts);
      REQUIRE(second.paired_mask_mismatches ==
              first.paired_mask_mismatches);
      require_count_parity(
          static_cast<std::uint64_t>(second.changed_contacts),
          static_cast<std::uint64_t>(first.changed_contacts),
          label + " changed contacts");
      require_count_parity(second.scheduled_source_spikes,
                           first.scheduled_source_spikes,
                           label + " scheduled spikes");
      require_count_parity(second.arrived_source_spikes,
                           first.arrived_source_spikes,
                           label + " arrived spikes");
      require_near(second.weight_mean, first.weight_mean,
                   1.0e-4f, 1.0e-2f,
                   label + " weight mean");
    }
  }
}

void test_training_maturation_and_sequential_index() {
  const std::vector<int> devices = available_devices();
  REQUIRE_MESSAGE(!devices.empty(),
                  "training maturation test requires CUDA");
  for (const int device : devices) {
    const Config config = calibration_for(device).result.config;
    NativeModel model(config, device, 1);

    const TrainingMetrics first =
        model.train(training_options(900, false));
    REQUIRE(first.completed_presentations == 900);
    auto diagnostics = model.diagnostics();
    REQUIRE(diagnostics.size() == 1u);
    REQUIRE(diagnostics[0].presentation_index == 900);
    REQUIRE(diagnostics[0].silent_iti_steps == 250u * 900u);
    REQUIRE(diagnostics[0].silent_iti_afferent_arrivals == 0u);
    REQUIRE(diagnostics[0].recurrent_plasticity_steps == 0u);
    REQUIRE(
        path_diagnostics(
            diagnostics[0], PlasticPath::kExcitatoryToExcitatory)
            .changed_contacts == 0);
    REQUIRE(
        path_diagnostics(
            diagnostics[0], PlasticPath::kExcitatoryToInhibitory)
            .changed_contacts == 0);
    const std::uint64_t accepted_at_900 =
        diagnostics[0].accepted_steps;
    const std::uint64_t silent_at_900 =
        diagnostics[0].silent_iti_steps;

    const TrainingMetrics boundary =
        model.train(training_options(100, false));
    REQUIRE(boundary.completed_presentations == 100);
    diagnostics = model.diagnostics();
    REQUIRE(diagnostics.size() == 1u);
    REQUIRE(diagnostics[0].presentation_index == 1000);
    REQUIRE(diagnostics[0].silent_iti_steps == 250u * 1000u);
    REQUIRE(diagnostics[0].silent_iti_afferent_arrivals == 0u);
    REQUIRE(diagnostics[0].accepted_steps > accepted_at_900);
    REQUIRE(diagnostics[0].silent_iti_steps > silent_at_900);
    REQUIRE(diagnostics[0].recurrent_plasticity_steps == 0u);
    REQUIRE(
        path_diagnostics(
            diagnostics[0], PlasticPath::kExcitatoryToExcitatory)
            .changed_contacts == 0);
    REQUIRE(
        path_diagnostics(
            diagnostics[0], PlasticPath::kExcitatoryToInhibitory)
            .changed_contacts == 0);
    const std::uint64_t accepted_at_1000 =
        diagnostics[0].accepted_steps;
    const std::uint64_t silent_at_1000 =
        diagnostics[0].silent_iti_steps;

    const TrainingMetrics after_boundary =
        model.train(training_options(1, false));
    REQUIRE(after_boundary.completed_presentations == 1);
    diagnostics = model.diagnostics();
    REQUIRE(diagnostics.size() == 1u);
    const TrainingDiagnostics& mature = diagnostics[0];
    const std::string label =
        "cuda" + std::to_string(device) + " maturation";
    REQUIRE(mature.presentation_index == 1001);
    REQUIRE(mature.silent_iti_steps == 250u * 1001u);
    REQUIRE_MESSAGE(
        mature.silent_iti_afferent_arrivals == 0u,
        label +
            " received sensory/background afferents during silent ITI");
    REQUIRE(mature.accepted_steps > accepted_at_1000);
    REQUIRE(mature.silent_iti_steps > silent_at_1000);
    REQUIRE_MESSAGE(
        mature.recurrent_plasticity_steps > 0u,
        label + " did not enable recurrence after presentation 1000");
    REQUIRE_MESSAGE(
        path_diagnostics(
            mature, PlasticPath::kExcitatoryToExcitatory)
                .changed_contacts > 0,
        label + " E->E contacts did not learn after maturation");
    REQUIRE_MESSAGE(
        path_diagnostics(
            mature, PlasticPath::kExcitatoryToInhibitory)
                .changed_contacts > 0,
        label + " E->I contacts did not learn after maturation");
    for (const Population population : kAllPopulations) {
      require_population_state_finite(
          population_diagnostics(mature, population),
          population, label);
    }
    for (const PlasticPath path : kAllPlasticPaths) {
      const auto& state = path_diagnostics(mature, path);
      require_path_bounds_and_pairing(state, path, label);
      REQUIRE_MESSAGE(
          state.scheduled_source_spikes ==
              state.arrived_source_spikes,
          label + "." + path_name(path) +
              " retained a delayed impulse after ITI");
    }
  }
}

void test_training_pruning_lifecycle_and_no_modality_collapse() {
  const std::vector<int> devices = available_devices();
  REQUIRE_MESSAGE(!devices.empty(),
                  "training pruning test requires CUDA");
  const int device = devices.front();
  const Config config = calibration_for(device).result.config;
  NativeModel model(config, device, 1);
  const auto initial_vector = model.diagnostics();
  REQUIRE(initial_vector.size() == 1u);
  const TrainingDiagnostics initial = initial_vector[0];

  const TrainingMetrics pre_eligibility =
      model.train(training_options(5000, true));
  REQUIRE(pre_eligibility.completed_presentations == 5000);
  auto diagnostics = model.diagnostics();
  REQUIRE(diagnostics.size() == 1u);
  REQUIRE(diagnostics[0].presentation_index == 5000);
  REQUIRE(diagnostics[0].silent_iti_steps == 250u * 5000u);
  REQUIRE(diagnostics[0].silent_iti_afferent_arrivals == 0u);
  REQUIRE_MESSAGE(
      diagnostics[0].pruned_contacts == 0,
      "contacts pruned at or before presentation 5000");
  for (const PlasticPath path : kAllPlasticPaths) {
    REQUIRE_MESSAGE(
        path_diagnostics(diagnostics[0], path)
                .paired_mask_mismatches == 0,
        std::string(path_name(path)) +
            " receptor masks diverged before pruning");
  }

  const TrainingMetrics before_dwell =
      model.train(training_options(1998, true));
  REQUIRE(before_dwell.completed_presentations == 1998);
  diagnostics = model.diagnostics();
  REQUIRE(diagnostics.size() == 1u);
  REQUIRE(diagnostics[0].presentation_index == 6998);
  REQUIRE(diagnostics[0].silent_iti_steps == 250u * 6998u);
  REQUIRE(diagnostics[0].silent_iti_afferent_arrivals == 0u);
  REQUIRE_MESSAGE(
      diagnostics[0].pruned_contacts == 0,
      "contacts pruned before 2000-presentation dwell elapsed");
  int maximum_observed_dwell = 0;
  for (const PlasticPath path : kAllPlasticPaths) {
    maximum_observed_dwell = std::max(
        maximum_observed_dwell,
        path_diagnostics(diagnostics[0], path)
            .maximum_low_weight_dwell);
  }
  REQUIRE_MESSAGE(
      maximum_observed_dwell == 1999,
      "no contact accumulated the exact pre-pruning dwell");

  const TrainingMetrics pruning_step =
      model.train(training_options(1, true));
  REQUIRE(pruning_step.completed_presentations == 1);
  diagnostics = model.diagnostics();
  REQUIRE(diagnostics.size() == 1u);
  const TrainingDiagnostics& pruned = diagnostics[0];
  REQUIRE(pruned.presentation_index == 6999);
  REQUIRE(pruned.silent_iti_steps == 250u * 6999u);
  REQUIRE_MESSAGE(
      pruned.silent_iti_afferent_arrivals == 0u,
      "sensory/background afferents leaked into silent ITI");
  REQUIRE_MESSAGE(pruned.pruned_contacts > 0,
                  "no contact pruned after exact 2000-presentation dwell");
  int lost_contacts = 0;
  for (const PlasticPath path : kAllPlasticPaths) {
    const auto& before = path_diagnostics(initial, path);
    const auto& after = path_diagnostics(pruned, path);
    require_path_bounds_and_pairing(
        after, path, "cuda0 post-pruning");
    REQUIRE_MESSAGE(
        after.active_contacts <= before.active_contacts,
        std::string(path_name(path)) +
            " gained contacts during pruning");
    lost_contacts +=
        before.active_contacts - after.active_contacts;
    REQUIRE_MESSAGE(
        after.scheduled_source_spikes ==
            after.arrived_source_spikes,
        std::string(path_name(path)) +
            " delay ring retained traffic after pruning run");
  }
  REQUIRE_MESSAGE(
      lost_contacts > 0,
      "pruning counter increased without deleting an active contact");
  REQUIRE_MESSAGE(
      lost_contacts == pruned.pruned_contacts,
      "pruned contact counter disagrees with active-mask deletion");

  const auto& auditory =
      path_diagnostics(
          pruned, PlasticPath::kAuditoryToExcitatory);
  const auto& visual =
      path_diagnostics(
          pruned, PlasticPath::kVisualToExcitatory);
  REQUIRE(auditory.active_contacts > 0);
  REQUIRE(visual.active_contacts > 0);
  REQUIRE(auditory.weight_mean > 0.0f);
  REQUIRE(visual.weight_mean > 0.0f);
  const float ratio = auditory.weight_mean / visual.weight_mean;
  REQUIRE_MESSAGE(
      ratio >= 0.10f && ratio <= 10.0f,
      "A/V excitatory pathways collapsed under realistic imbalance");
  for (const Population population : kAllPopulations) {
    const auto& state =
        population_diagnostics(pruned, population);
    require_population_state_finite(
        state, population, "cuda0 post-pruning");
    REQUIRE_MESSAGE(
        state.cumulative_spikes > 0u,
        std::string(population_name(population)) +
            " population fell silent during training");
    const int neurons =
        population == Population::kInhibitory
            ? clean_msi::kInhibitoryNeurons
            : clean_msi::kExcitatoryNeurons;
    const double rate_hz =
        1000.0 * static_cast<double>(state.cumulative_spikes) /
        (static_cast<double>(pruned.accepted_steps) *
         static_cast<double>(neurons));
    REQUIRE_MESSAGE(
        rate_hz > 0.0 && rate_hz < 200.0,
        std::string(population_name(population)) +
            " long-run firing rate is unstable");
  }
}

void test_training_performance() {
  const std::vector<int> devices = available_devices();
  REQUIRE_MESSAGE(!devices.empty(),
                  "training performance requires CUDA");
  constexpr int kPerformancePresentations = 1000;
  for (const int device : devices) {
    const Config config = calibration_for(device).result.config;
    NativeModel model(config, device, 1);
    TrainingMetrics metrics{};
    const double wall_seconds = timed_seconds(device, [&] {
      metrics = model.train(
          training_options(kPerformancePresentations, false));
    });
    require_training_metrics_science(
        metrics, kPerformancePresentations,
        "cuda" + std::to_string(device) + " throughput training");
    const double presentations_per_second =
        static_cast<double>(kPerformancePresentations) / wall_seconds;
    std::cout << "[PERF] cuda" << device
              << " training_presentations="
              << kPerformancePresentations
              << " wall_s=" << std::fixed << std::setprecision(6)
              << wall_seconds << " presentations_per_s="
              << std::setprecision(2) << presentations_per_second
              << " internal_s=" << std::setprecision(6)
              << metrics.elapsed_seconds << '\n';
  }
}

enum class Suite {
  kCore,
  kCalibrationParity,
  kCalibrationPerformance,
  kGenerator,
  kTiming,
  kEvaluationSynthetic,
  kEvaluationBoundary,
  kEvaluationMechanics,
  kEvaluationCausal,
  kEvaluationPerformance,
  kScienceRedesign,
  kTraining,
  kTrainingPerformance,
};

struct TestCase {
  const char* name;
  Suite suite;
  void (*function)();
};

constexpr std::array<TestCase, 45> kTests{{
    {"exact_scientific_contract", Suite::kCore,
     test_exact_scientific_contract},
    {"philox_determinism_and_known_answer", Suite::kCore,
     test_philox_determinism_and_known_answer},
    {"voltage_coupled_current_equations", Suite::kCore,
     test_voltage_coupled_current_equations},
    {"biexponential_scalar_equations", Suite::kCore,
     test_biexponential_scalar_equations},
    {"delay_impulses", Suite::kCore, test_delay_impulses},
    {"lower_median", Suite::kCore, test_lower_median},
    {"local_plasticity_equations_and_bounds", Suite::kCore,
     test_local_plasticity_equations_and_bounds},
    {"solver_scalar_cases", Suite::kCore, test_solver_scalar_cases},
    {"cuda_core_parity", Suite::kCore, test_cuda_core_parity},
    {"calibration_parity", Suite::kCalibrationParity,
     test_calibration_parity},
    {"calibration_performance", Suite::kCalibrationPerformance,
     test_calibration_performance},
    {"locked_generator_statistics_20k", Suite::kGenerator,
     test_locked_generator_statistics_20k},
    {"native_generator_audit_20k", Suite::kGenerator,
     test_native_generator_audit_20k},
    {"one_presentation_timing_and_afferent_silence", Suite::kTiming,
     test_one_presentation_timing_and_afferent_silence},
    {"stage3_feature_bins_and_logistic", Suite::kEvaluationSynthetic,
     test_stage3_feature_bins_and_logistic},
    {"stage3_raw_control_response_windows",
     Suite::kEvaluationSynthetic,
     test_stage3_raw_control_response_windows},
    {"stage3_exact_grids_and_synthetic_fits",
     Suite::kEvaluationSynthetic,
     test_stage3_exact_grids_and_synthetic_fits},
    {"stage3_empirical_tbw_crossings_and_rejections",
     Suite::kEvaluationSynthetic,
     test_stage3_empirical_tbw_crossings_and_rejections},
    {"stage3_sbw_and_inverse_effectiveness_algebra",
     Suite::kEvaluationSynthetic,
     test_stage3_sbw_and_inverse_effectiveness_algebra},
    {"stage3_sbw_matched_trial_orientation_averaging",
     Suite::kEvaluationSynthetic,
     test_stage3_sbw_matched_trial_orientation_averaging},
    {"stage3_synthetic_rf_and_topography",
     Suite::kEvaluationSynthetic,
     test_stage3_synthetic_rf_and_topography},
    {"stage3_causal_control_mapping", Suite::kEvaluationSynthetic,
     test_stage3_causal_control_mapping},
    {"stage3_synthetic_cohort_control_and_seed_directions",
     Suite::kEvaluationSynthetic,
     test_stage3_synthetic_cohort_control_and_seed_directions},
    {"stage3_frozen_boundary_cuda_replay_and_neutrality",
     Suite::kEvaluationBoundary,
     test_stage3_frozen_boundary_cuda_replay_and_neutrality},
    {"stage3_background_active_during_frozen_burnin",
     Suite::kEvaluationMechanics,
     test_stage3_background_active_during_frozen_burnin},
    {"stage3_small_production_evaluation_mechanics",
     Suite::kEvaluationMechanics,
     test_stage3_small_production_evaluation_mechanics},
    {"stage3_all_controls_frozen_observer_and_restoration",
     Suite::kEvaluationCausal,
     test_stage3_production_causal_controls_and_restoration},
    {"stage3_evaluation_batching_performance_and_parity",
     Suite::kEvaluationPerformance,
     test_stage3_evaluation_batching_performance_and_parity},
    {"science_coarse_gaussian_feedforward_and_position_blind_scaffolds",
     Suite::kScienceRedesign,
     test_science_coarse_gaussian_feedforward_and_position_blind_scaffolds},
    {"science_clopath_additive_hard_bounds_and_oja_soft_bounds",
     Suite::kScienceRedesign,
     test_science_clopath_additive_hard_bounds_and_oja_soft_bounds},
    {"science_istdp_target_and_hard_clipping",
     Suite::kScienceRedesign,
     test_science_istdp_target_and_hard_clipping},
    {"science_nmda_shares_contact_efficacy",
     Suite::kScienceRedesign,
     test_science_nmda_shares_contact_efficacy},
    {"science_calibration_runtime_receptor_ratio_identity",
     Suite::kScienceRedesign,
     test_science_calibration_runtime_receptor_ratio_identity},
    {"science_gabaa_unitary_ipsp_and_runtime_relay",
     Suite::kScienceRedesign,
     test_science_gabaa_unitary_ipsp_and_runtime_relay},
    {"science_msi_e_intrinsic_phenotype_mix",
     Suite::kScienceRedesign,
     test_science_msi_e_intrinsic_phenotype_mix},
    {"science_msi_e_spike_reset_adaptation_causality",
     Suite::kScienceRedesign,
     test_science_msi_e_spike_reset_adaptation_causality},
    {"science_checked_topology_pearson",
     Suite::kScienceRedesign,
     test_science_checked_topology_pearson},
    {"science_rf_signed_mean_zero_floor",
     Suite::kScienceRedesign,
     test_science_rf_signed_mean_zero_floor},
    {"science_weight_snapshot_and_initial_weight_routing",
     Suite::kScienceRedesign,
     test_science_weight_snapshot_and_initial_weight_routing},
    {"training_initial_scaffold_and_masks", Suite::kTraining,
     test_training_initial_scaffold_and_masks},
    {"training_requires_calibration", Suite::kTraining,
     test_training_requires_calibration},
    {"training_replay_activity_and_cuda_parity", Suite::kTraining,
     test_training_replay_activity_and_cuda_parity},
    {"training_maturation_and_sequential_index", Suite::kTraining,
     test_training_maturation_and_sequential_index},
    {"training_pruning_lifecycle_and_no_modality_collapse",
     Suite::kTraining,
     test_training_pruning_lifecycle_and_no_modality_collapse},
    {"training_performance", Suite::kTrainingPerformance,
     test_training_performance},
}};

bool suite_selected(Suite suite, bool core, bool calibration_parity,
                    bool calibration_performance, bool generator,
                    bool timing,
                    bool evaluation_synthetic,
                    bool evaluation_boundary,
                    bool evaluation_mechanics,
                    bool evaluation_causal,
                    bool evaluation_performance,
                    bool science_redesign, bool training,
                    bool training_performance) {
  switch (suite) {
    case Suite::kCore:
      return core;
    case Suite::kCalibrationParity:
      return calibration_parity;
    case Suite::kCalibrationPerformance:
      return calibration_performance;
    case Suite::kGenerator:
      return generator;
    case Suite::kTiming:
      return timing;
    case Suite::kEvaluationSynthetic:
      return evaluation_synthetic;
    case Suite::kEvaluationBoundary:
      return evaluation_boundary;
    case Suite::kEvaluationMechanics:
      return evaluation_mechanics;
    case Suite::kEvaluationCausal:
      return evaluation_causal;
    case Suite::kEvaluationPerformance:
      return evaluation_performance;
    case Suite::kScienceRedesign:
      return science_redesign;
    case Suite::kTraining:
      return training;
    case Suite::kTrainingPerformance:
      return training_performance;
  }
  return false;
}

void print_usage(const char* executable) {
  std::cout
      << "Usage: " << executable
      << " [--core | --calibration-parity | "
         "--calibration-performance | --generator | --timing | "
         "--evaluation-synthetic | "
         "--evaluation-boundary | "
         "--evaluation-mechanics | "
         "--evaluation-causal | "
         "--evaluation-performance | --evaluation-all | "
         "--science-redesign | "
         "--training | "
         "--training-performance | --all] [--list]\n"
      << "Default: --core\n";
}

}  // namespace

int main(int argc, char** argv) {
  bool core = false;
  bool calibration_parity = false;
  bool calibration_performance = false;
  bool generator = false;
  bool timing = false;
  bool evaluation_synthetic = false;
  bool evaluation_boundary = false;
  bool evaluation_mechanics = false;
  bool evaluation_causal = false;
  bool evaluation_performance = false;
  bool science_redesign = false;
  bool training = false;
  bool training_performance = false;
  bool explicit_suite = false;
  bool list = false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--core") {
      core = true;
      explicit_suite = true;
    } else if (argument == "--calibration-parity") {
      calibration_parity = true;
      explicit_suite = true;
    } else if (argument == "--calibration-performance") {
      calibration_performance = true;
      explicit_suite = true;
    } else if (argument == "--generator") {
      generator = true;
      explicit_suite = true;
    } else if (argument == "--timing") {
      timing = true;
      explicit_suite = true;
    } else if (argument == "--evaluation-synthetic") {
      evaluation_synthetic = true;
      explicit_suite = true;
    } else if (argument == "--evaluation-boundary") {
      evaluation_boundary = true;
      explicit_suite = true;
    } else if (argument == "--evaluation-mechanics") {
      evaluation_mechanics = true;
      explicit_suite = true;
    } else if (argument == "--evaluation-causal") {
      evaluation_causal = true;
      explicit_suite = true;
    } else if (argument == "--evaluation-performance") {
      evaluation_performance = true;
      explicit_suite = true;
    } else if (argument == "--evaluation-all") {
      evaluation_synthetic = true;
      evaluation_boundary = true;
      evaluation_mechanics = true;
      evaluation_causal = true;
      evaluation_performance = true;
      explicit_suite = true;
    } else if (argument == "--science-redesign") {
      science_redesign = true;
      explicit_suite = true;
    } else if (argument == "--training") {
      training = true;
      explicit_suite = true;
    } else if (argument == "--training-performance") {
      training_performance = true;
      explicit_suite = true;
    } else if (argument == "--all") {
      core = true;
      calibration_parity = true;
      calibration_performance = true;
      generator = true;
      timing = true;
      evaluation_synthetic = true;
      evaluation_boundary = true;
      evaluation_mechanics = true;
      evaluation_causal = true;
      evaluation_performance = true;
      science_redesign = true;
      training = true;
      training_performance = true;
      explicit_suite = true;
    } else if (argument == "--list") {
      list = true;
    } else if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << argument << '\n';
      print_usage(argv[0]);
      return 2;
    }
  }
  if (!explicit_suite) {
    core = true;
  }
  if (list) {
    for (const TestCase& test : kTests) {
      if (suite_selected(test.suite, core, calibration_parity,
                         calibration_performance, generator, timing,
                         evaluation_synthetic, evaluation_boundary,
                         evaluation_mechanics, evaluation_causal,
                         evaluation_performance,
                         science_redesign,
                         training,
                         training_performance)) {
        std::cout << test.name << '\n';
      }
    }
    return 0;
  }

  int passed = 0;
  int selected = 0;
  const auto suite_begin = std::chrono::steady_clock::now();
  for (const TestCase& test : kTests) {
    if (!suite_selected(test.suite, core, calibration_parity,
                        calibration_performance, generator, timing,
                        evaluation_synthetic, evaluation_boundary,
                        evaluation_mechanics, evaluation_causal,
                        evaluation_performance,
                        science_redesign,
                        training,
                        training_performance)) {
      continue;
    }
    ++selected;
    const auto begin = std::chrono::steady_clock::now();
    try {
      test.function();
      const double seconds =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - begin)
              .count();
      std::cout << "[PASS] " << test.name << " ("
                << std::fixed << std::setprecision(3) << seconds
                << " s)\n";
      ++passed;
    } catch (const std::exception& error) {
      const double seconds =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - begin)
              .count();
      std::cerr << "[FAIL] " << test.name << " ("
                << std::fixed << std::setprecision(3) << seconds
                << " s): " << error.what() << '\n';
      std::cerr << "[SUMMARY] passed=" << passed
                << " failed=1 selected=" << selected << '\n';
      return 1;
    }
  }
  const double suite_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - suite_begin)
          .count();
  std::cout << "[SUMMARY] passed=" << passed << " failed=0 selected="
            << selected << " elapsed_s=" << std::fixed
            << std::setprecision(3) << suite_seconds << '\n';
  return 0;
}
