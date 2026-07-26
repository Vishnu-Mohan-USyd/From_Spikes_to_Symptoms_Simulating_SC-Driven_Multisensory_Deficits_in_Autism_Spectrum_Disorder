#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace clean_msi {

constexpr int kAuditoryNeurons = 180;
constexpr int kVisualNeurons = 180;
constexpr int kExcitatoryNeurons = 180;
constexpr int kInhibitoryNeurons = 60;
constexpr int kPopulationCount = 4;
constexpr int kDelayRing = 4;
constexpr int kCalibrationCoarseNodes = 37;
constexpr int kCalibrationFineNodes = 145;
constexpr int kCalibrationRefinementNodes = 15;
constexpr int kLearningRateCandidates = 81;
constexpr int kTrainingChunkPresentations = 100;
constexpr int kExternalTrials = 256;
constexpr int kExternalDurationMs = 150;
constexpr int kExternalDriveMs = 50;
constexpr int kFeedforwardLargeContacts = 12;
constexpr int kFeedforwardSmallContacts = 4;
constexpr int kFeedforwardDurationMs = 20;
constexpr float kCalibrationExcitatoryContactWeight = 0.5f;
constexpr int kGabaaInhibitoryInputs = 10;
constexpr int kGabaaDurationMs = 60;
constexpr float kGabaaMedianInitialContactWeight = 0.035f;
constexpr int kGabaaUnitaryDelayMs = 1;
constexpr int kGabaaIpspSettleMs = 100;
constexpr int kGabaaIpspWindowMs = 100;
constexpr float kGabaaUnitaryTargetMv = 1.00f;
constexpr float kGabaaUnitaryMinimumMv = 0.95f;
constexpr float kGabaaUnitaryMaximumMv = 1.05f;
constexpr int kBackgroundTrials = 128;
constexpr int kBackgroundDurationMs = 5000;
constexpr int kBackgroundDiscardMs = 1000;
constexpr int kClopathPairings = 400;
constexpr float kClopathHomeostasisTauMs = 100000.0f;
// Frozen equal-salience, 100,000-step, seed-0 E+I normative assay.
constexpr float kClopathHomeostasisReferenceMv2 = 16.3214752406f;
constexpr float kClopathThetaPlusMv = -45.0f;
constexpr int kOjaEvents = 1000;
constexpr int kIstdpSteps = 60000;
constexpr int kIstdpMeasurementStart = 40000;
constexpr int kIstdpInhibitoryInputs = 10;
constexpr float kIstdpTraceTauMs = 20.0f;
constexpr float kIstdpTargetRateHz = 5.0f;
constexpr float kIstdpAlpha =
    2.0f * (kIstdpTargetRateHz / 1000.0f) *
    kIstdpTraceTauMs;
constexpr int kFeedforwardScaffoldInDegree = 45;
constexpr double kAuditoryScaffoldFwhmDeg = 105.0;
constexpr double kVisualScaffoldFwhmDeg = 45.0;
constexpr double kFwhmToSigmaDivisor = 2.35482;
constexpr int kRecurrentExcitatoryScaffoldInDegree = 36;
constexpr int kInhibitoryScaffoldInDegree = 12;
constexpr int kMsiEMarkedPhenotypesPerSeed = 9;
constexpr float kMsiERegularRecoveryIncrement = 0.10f;
constexpr float kMsiEMarkedRecoveryIncrement = 1.0f;
constexpr int kMsiEPhenotypeAuditSeeds = 5;
constexpr int kMsiEPhenotypeSettleMs = 100;
constexpr int kMsiEPhenotypeDriveMs = 400;
constexpr float kMsiEPhenotypeDriveCurrent = 10.0f;
constexpr int kEvaluationBinWidthMs = 20;
constexpr int kEvaluationFeatureBins = 40;
constexpr int kEvaluationRelativeMs = 800;
constexpr int kTbwPointCount = 41;
constexpr int kSbwPointCount = 25;
constexpr int kSignedSbwPointCount = 49;
constexpr int kRfPointCount = 25;
constexpr int kInverseEffectivenessCount = 3;

// Exact stable lower equilibrium root for u=bv and zero input.
__host__ __device__ inline float
izhikevich_stable_rest_voltage_mv(float b) {
  assert(isfinite(b));
  const float linear_coefficient = 5.0f - b;
  const float discriminant =
      linear_coefficient * linear_coefficient -
      4.0f * 0.04f * 140.0f;
  assert(isfinite(discriminant));
  assert(discriminant > 0.0f);
  const float stable_root =
      (-linear_coefficient - sqrtf(discriminant)) / 0.08f;
  const float stability_slope =
      0.08f * stable_root + linear_coefficient;
  assert(isfinite(stable_root));
  assert(isfinite(stability_slope));
  assert(stability_slope < 0.0f);
  return stable_root;
}

__host__ __device__ constexpr float clopath_homeostasis_multiplier(
    float homeostasis_mv2) {
  return homeostasis_mv2 / kClopathHomeostasisReferenceMv2;
}

enum class Population : int {
  kAuditory = 0,
  kVisual = 1,
  kExcitatory = 2,
  kInhibitory = 3,
};

enum class AssayKind : int {
  kExternalRs = 0,
  kFeedforwardE = 1,
  kFeedforwardI = 2,
  kGabaa = 3,
  kBackgroundRs = 4,
  kBackgroundFs = 5,
};

__host__ __device__ constexpr bool
calibration_interval_requires_closure(AssayKind kind) {
  return kind != AssayKind::kGabaa;
}

enum class CausalControl : std::uint32_t {
  kNone = 0,
  kNmdaOff = 1u << 0,
  kGabaaOff = 1u << 1,
  kRecurrentExcitationOff = 1u << 2,
  kRecruitedInhibitionOff = 1u << 3,
  kSourceRowShuffle = 1u << 4,
  kMsiEAdaptationOff = 1u << 5,
};

enum class TrainingCohort : int {
  kBaseline = 0,
  kPlasticityOff = 1,
  kSpatialShuffle = 2,
  kFixedOffset = 3,
};

enum class JitterScale : int {
  kBaseline = 0,
  kZero = 1,
  kHalf = 2,
  kDouble = 3,
};

__host__ __device__ constexpr CausalControl operator|(
    CausalControl lhs, CausalControl rhs) {
  return static_cast<CausalControl>(
      static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
}

__host__ __device__ constexpr bool has_control(
    CausalControl value, CausalControl flag) {
  return (static_cast<std::uint32_t>(value) &
          static_cast<std::uint32_t>(flag)) != 0u;
}

struct Config {
  std::uint64_t seed = 0;
  float dt_ms = 1.0f;
  float threshold_mv = 30.0f;
  float excitatory_reversal_mv = 0.0f;
  float gabaa_reversal_mv = -75.0f;

  float ampa_rise_ms = 0.5f;
  float ampa_decay_ms = 5.0f;
  float nmda_rise_ms = 2.0f;
  float nmda_decay_ms = 80.0f;
  float gabaa_rise_ms = 1.0f;
  float gabaa_decay_ms = 15.0f;

  float external_ampa_weight = 1.0f;
  float external_nmda_weight = 0.10f;
  float background_ampa_weight = 0.01f;
  float background_nmda_weight = 0.01f;

  float q_external_rs = 1.0f;
  float q_background_rs = 1.0f;
  float q_background_fs = 1.0f;
  float q_ff_e = 1.0f;
  float q_ff_i = 1.0f;
  float q_gabaa = 1.0f;

  float eta_clopath_ff = 1.0e-4f;
  float eta_oja = 1.0e-4f;
  float eta_istdp = 1.0e-4f;
  TrainingCohort training_cohort = TrainingCohort::kBaseline;
  JitterScale jitter_scale = JitterScale::kBaseline;
  float fixed_offset_deg = 20.0f;
  bool calibrated = false;
};

struct ReceptorKernel {
  float rise_ms = 0.5f;
  float decay_ms = 5.0f;
  float rise_decay = 0.0f;
  float decay_decay = 0.0f;
  float normalization = 0.0f;
};

struct ReceptorState {
  float rise = 0.0f;
  float decay = 0.0f;
};

struct NeuronParameters {
  float a = 0.02f;
  float b = 0.20f;
  float c_mv = -65.0f;
  float d = 8.0f;
};

struct NeuronState {
  float voltage_mv = -65.0f;
  float recovery = -13.0f;
};

struct SolverInput {
  NeuronState state{};
  NeuronParameters parameters{};
  float g_ampa = 0.0f;
  float g_nmda = 0.0f;
  float g_gabaa = 0.0f;
  float additive_current = 0.0f;
  float dt_ms = 1.0f;
  float threshold_mv = 30.0f;
  float excitatory_reversal_mv = 0.0f;
  float gabaa_reversal_mv = -75.0f;
};

struct SolverOutput {
  NeuronState state{};
  float pre_reset_voltage_mv = -65.0f;
  float pre_reset_recovery = -13.0f;
  int spike_count = 0;
  bool emitted = false;
  bool overflow = false;
};

struct AssayMetrics {
  float primary = 0.0f;
  float secondary = 0.0f;
  float tertiary = 0.0f;
};

struct GabaaRelayAudit {
  int relay_neurons = 0;
  int excitatory_packets = 0;
  int relay_spikes = 0;
  int gabaa_arrivals = 0;
  int lag_count = 0;
  int minimum_lag_ms = 0;
  int maximum_lag_ms = 0;
  float mean_lag_ms = 0.0f;
  int without_gabaa_spikes = 0;
  int with_gabaa_spikes = 0;
  int direct_to_gabaa_lag_counts[kGabaaDurationMs]{};
};

struct GabaaIpspMetrics {
  int contact_count = 0;
  float amplitude_mv = 0.0f;
  float signed_nadir_mv = 0.0f;
  float area_mv_ms = 0.0f;
  float outward_charge = 0.0f;
  int control_spikes = 0;
  int gabaa_spikes = 0;
};

struct GabaaEfficacyAudit {
  GabaaIpspMetrics unitary{};
  GabaaIpspMetrics compound{};
};

struct LearningAssayMetrics {
  float clopath_delta = 0.0f;
  float clopath_shuffled_drift = 0.0f;
  float oja_selection_ratio = 0.0f;
  float oja_bound_fraction = 0.0f;
  float istdp_weak_rate_hz = 0.0f;
  float istdp_strong_rate_hz = 0.0f;
  float istdp_weak_bound_fraction = 0.0f;
  float istdp_strong_bound_fraction = 0.0f;
};

struct CalibrationResult {
  Config config{};
  AssayMetrics external{};
  AssayMetrics feedforward_e{};
  AssayMetrics feedforward_i{};
  AssayMetrics gabaa{};
  GabaaEfficacyAudit gabaa_efficacy{};
  GabaaRelayAudit gabaa_relay{};
  AssayMetrics background_rs{};
  AssayMetrics background_fs{};
  LearningAssayMetrics learning{};
  float conductance_seconds = 0.0f;
  float learning_seconds = 0.0f;
  bool criteria_passed = false;
};

enum class PresentationKind : int {
  kCommonAv = 0,
  kIndependentAv = 1,
  kAuditoryOnly = 2,
  kVisualOnly = 3,
};

enum class PlasticPath : int {
  kAuditoryToExcitatory = 0,
  kVisualToExcitatory = 1,
  kExcitatoryToExcitatory = 2,
  kAuditoryToInhibitory = 3,
  kVisualToInhibitory = 4,
  kExcitatoryToInhibitory = 5,
  kInhibitoryToExcitatory = 6,
};

struct TrainingOptions {
  int presentations = 10000;
  int seed_count = 1;
  int chunk_presentations = kTrainingChunkPresentations;
  bool enable_pruning = false;
};

struct TrainingMetrics {
  int completed_presentations = 0;
  std::uint64_t total_a_spikes = 0;
  std::uint64_t total_v_spikes = 0;
  std::uint64_t total_e_spikes = 0;
  std::uint64_t total_i_spikes = 0;
  float mean_a_to_e = 0.0f;
  float mean_v_to_e = 0.0f;
  float mean_e_to_e = 0.0f;
  float mean_i_to_e = 0.0f;
  int pruned_contacts = 0;
  float elapsed_seconds = 0.0f;
};

struct PopulationDiagnostics {
  float voltage_min_mv = 0.0f;
  float voltage_max_mv = 0.0f;
  float recovery_min = 0.0f;
  float recovery_max = 0.0f;
  std::uint64_t cumulative_spikes = 0;
  bool finite = false;
};

struct PathDiagnostics {
  float weight_min = 0.0f;
  float weight_max = 0.0f;
  float weight_mean = 0.0f;
  int active_contacts = 0;
  int changed_contacts = 0;
  int paired_mask_mismatches = 0;
  int maximum_low_weight_dwell = 0;
  std::uint64_t scheduled_source_spikes = 0;
  std::uint64_t arrived_source_spikes = 0;
};

struct TrainingDiagnostics {
  int presentation_index = 0;
  std::uint64_t accepted_steps = 0;
  std::uint64_t silent_iti_steps = 0;
  std::uint64_t silent_iti_afferent_arrivals = 0;
  std::uint64_t recurrent_plasticity_steps = 0;
  int pruned_contacts = 0;
  std::array<PopulationDiagnostics, kPopulationCount> populations{};
  std::array<PathDiagnostics, 7> paths{};
};

struct MsiEIntrinsicPhenotypeAudit {
  int seed_count = 0;
  int neuron_count = 0;
  int assigned_marked_total = 0;
  int observed_greater_than_two_total = 0;
  std::array<int, kMsiEPhenotypeAuditSeeds>
      assigned_marked_per_seed{};
  std::array<int, kMsiEPhenotypeAuditSeeds>
      observed_greater_than_two_per_seed{};
  float greater_than_two_prevalence = 0.0f;
  float regular_first_to_mean_isi = 0.0f;
  float marked_first_to_mean_isi = 0.0f;
  float regular_greater_than_two_fraction = 0.0f;
  float marked_greater_than_two_fraction = 0.0f;
  float position_index_correlation = 0.0f;
};

struct DevelopmentalSampleAudit {
  PresentationKind kind = PresentationKind::kCommonAv;
  bool auditory_present = false;
  bool visual_present = false;
  float auditory_latent_deg = 0.0f;
  float visual_latent_deg = 0.0f;
  float auditory_noise_deg = 0.0f;
  float visual_noise_deg = 0.0f;
  float auditory_location_deg = 0.0f;
  float visual_location_deg = 0.0f;
  float auditory_rate_hz = 0.0f;
  float visual_rate_hz = 0.0f;
  float physical_soa_ms = 0.0f;
  float auditory_latency_ms = 0.0f;
  float visual_latency_ms = 0.0f;
  int auditory_onset_ms = -1;
  int visual_onset_ms = -1;
  int poststimulus_end_ms = 0;
  int first_zero_afferent_receptor_step = 0;
  int valid_end_ms = 0;
};

struct GeneratorProfileAudit {
  std::vector<DevelopmentalSampleAudit> samples;
  std::array<float, kAuditoryNeurons> developmental_reflected_profile{};
  std::array<float, kAuditoryNeurons> controlled_exact_profile{};
};

struct ControlledCondition {
  bool auditory_present = true;
  bool visual_present = true;
  float auditory_location_deg = 0.0f;
  float visual_location_deg = 0.0f;
  float auditory_rate_hz = 50.0f;
  float visual_rate_hz = 50.0f;
  float physical_soa_ms = -50.0f;
  CausalControl control = CausalControl::kNone;
  int label = 0;
  int random_group = -1;
  bool use_initial_weights = false;
};

struct FrozenTrialResult {
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
  std::uint64_t background_afferent_arrivals_during_burn_in = 0;
  std::array<std::uint16_t, kEvaluationRelativeMs>
      excitatory_spikes_per_relative_ms{};
  std::array<std::uint16_t, kExcitatoryNeurons>
      excitatory_baseline_spikes_per_neuron{};
  std::array<std::uint16_t, kExcitatoryNeurons>
      excitatory_response_spikes_per_neuron{};
  std::array<float, kEvaluationFeatureBins> features{};
  float baseline_rate_hz = 0.0f;
  float response_rate_hz = 0.0f;
  bool finite = false;
};

struct FrozenTrialBatch {
  int model_seed_count = 0;
  int condition_count = 0;
  int trials_per_condition = 0;
  int burn_in_steps = 0;
  std::array<int, kEvaluationFeatureBins + 1>
      relative_bin_edges_ms{};
  std::vector<FrozenTrialResult> trials;
  float kernel_seconds = 0.0f;
};

struct NeuralResponseWindow {
  int first_bin = -1;
  int last_bin = -1;
};

struct NeuralFusionRule {
  float threshold = 0.0f;
  NeuralResponseWindow auditory{};
  NeuralResponseWindow visual{};
};

struct FrozenPathAudit {
  std::vector<float> weights;
  std::vector<std::uint8_t> masks;
};

struct FrozenWeightAudit {
  int seed_index = 0;
  std::uint64_t global_seed = 0;
  std::array<FrozenPathAudit, 7> paths;
};

struct SeedWeightStateAudit {
  int seed_index = 0;
  std::uint64_t global_seed = 0;
  FrozenWeightAudit initial{};
  FrozenWeightAudit trained{};
};

struct ObserverMetrics {
  float auroc = 0.0f;
  float brier = 1.0f;
  float calibration_intercept = 0.0f;
  float calibration_slope = 0.0f;
  float expected_calibration_error = 1.0f;
  float intercept = 0.0f;
  float l2 = 1.0e-3f;
  std::array<float, kEvaluationFeatureBins> feature_mean{};
  std::array<float, kEvaluationFeatureBins> feature_std{};
  std::array<float, kEvaluationFeatureBins> coefficients{};
  std::vector<std::uint8_t> training_labels;
  std::vector<float> training_probabilities;
  std::vector<std::uint8_t> holdout_labels;
  std::vector<float> holdout_probabilities;
  int iterations = 0;
  bool converged = false;
};

struct AsymmetricGaussianFit {
  bool valid = false;
  bool crossings_valid = false;
  float baseline = 0.0f;
  float amplitude = 0.0f;
  float center_ms = 0.0f;
  float sigma_left_ms = 0.0f;
  float sigma_right_ms = 0.0f;
  float mse = 0.0f;
  float left_50_ms = 0.0f;
  float right_50_ms = 0.0f;
  float left_75_ms = 0.0f;
  float right_75_ms = 0.0f;
  float tbw50_ms = 0.0f;
  float tbw75_ms = 0.0f;
};

struct EmpiricalTemporalCrossings {
  bool valid = false;
  bool unimodal = false;
  int peak_index = -1;
  float peak_probability = 0.0f;
  float left_tail_baseline = 0.0f;
  float right_tail_baseline = 0.0f;
  float left_50_ms = 0.0f;
  float right_50_ms = 0.0f;
  float left_75_ms = 0.0f;
  float right_75_ms = 0.0f;
  float tbw50_ms = 0.0f;
  float tbw75_ms = 0.0f;
};

struct TemporalCurvePoint {
  float physical_soa_ms = 0.0f;
  float probability_mean = 0.0f;
  float probability_sem = 0.0f;
  float auditory_rate_hz = 0.0f;
  float visual_rate_hz = 0.0f;
  float audiovisual_rate_hz = 0.0f;
  float raw_neural_enhancement_hz = 0.0f;
};

struct TemporalAudit {
  int trials_per_soa = 0;
  std::array<float, kTbwPointCount> physical_soa_grid_ms{};
  std::array<TemporalCurvePoint, kTbwPointCount> points{};
  EmpiricalTemporalCrossings empirical{};
  AsymmetricGaussianFit fit{};
  float peak_minus_tail = 0.0f;
};

struct ComponentResponseMetrics {
  float auditory_rate_hz = 0.0f;
  float visual_rate_hz = 0.0f;
  float audiovisual_rate_hz = 0.0f;
  float raw_enhancement_hz = 0.0f;
  float raw_enhancement_sem = 0.0f;
  float multisensory_enhancement_percent = 0.0f;
  float additivity_hz = 0.0f;
  float additivity_percent = 0.0f;
};

struct SpatialCurvePoint {
  float disparity_deg = 0.0f;
  std::array<ComponentResponseMetrics, 2> orientations{};
  ComponentResponseMetrics pooled{};
};

struct DirectSpatialAudit {
  bool input_valid = false;
  bool finite = false;
  bool strictly_increasing = false;
  bool center_positive = false;
  bool contrast_positive = false;
  bool endpoints_valid = false;
  bool peak_location_valid = false;
  bool contiguous_prefix = false;
  bool single_outward_crossing = false;
  bool width_valid = false;
  bool spatial_gate = false;
  int peak_index = -1;
  int crossing_outer_index = -1;
  float center_enhancement_hz = 0.0f;
  float tail_baseline_hz = 0.0f;
  float contrast_hz = 0.0f;
  float peak_disparity_deg = 0.0f;
  float half_level_hz = 0.0f;
  float sbw50_deg = 0.0f;
};

struct SymmetricGaussianFit {
  bool valid = false;
  bool hwhm_valid = false;
  float baseline = 0.0f;
  float amplitude = 0.0f;
  float sigma_deg = 0.0f;
  float fitted_hwhm_deg = 0.0f;
  float mse = 0.0f;
};

struct SpatialAudit {
  int trials_per_orientation = 0;
  float response_floor_hz = 1.0f;
  std::array<float, kSbwPointCount> disparity_grid_deg{};
  std::array<SpatialCurvePoint, kSbwPointCount> points{};
  DirectSpatialAudit direct{};
  SymmetricGaussianFit fit{};
};

struct SignedSpatialAudit {
  int trials_per_disparity = 0;
  std::array<float, kSignedSbwPointCount> disparity_grid_deg{};
  std::array<ComponentResponseMetrics, kSignedSbwPointCount> points{};
  float positive_peak_deg = 0.0f;
  float positive_peak_enhancement_hz = 0.0f;
  bool positive_peak_valid = false;
};

struct RfTopographyAudit {
  int trials_per_location = 0;
  std::array<float, kRfPointCount> location_grid_deg{};
  std::array<
      std::array<float, kRfPointCount>, kExcitatoryNeurons>
      auditory_response_hz{};
  std::array<
      std::array<float, kRfPointCount>, kExcitatoryNeurons>
      visual_response_hz{};
  std::array<float, kExcitatoryNeurons> auditory_rf_center_deg{};
  std::array<float, kExcitatoryNeurons> auditory_rf_width_deg{};
  std::array<float, kExcitatoryNeurons> visual_rf_center_deg{};
  std::array<float, kExcitatoryNeurons> visual_rf_width_deg{};
  std::array<float, kExcitatoryNeurons> rf_center_mismatch_deg{};
  std::array<float, kExcitatoryNeurons>
      auditory_excitatory_map_center_deg{};
  std::array<float, kExcitatoryNeurons>
      auditory_excitatory_map_width_deg{};
  std::array<float, kExcitatoryNeurons>
      visual_excitatory_map_center_deg{};
  std::array<float, kExcitatoryNeurons>
      visual_excitatory_map_width_deg{};
  std::array<float, kExcitatoryNeurons>
      auditory_effective_inhibitory_center_deg{};
  std::array<float, kExcitatoryNeurons>
      auditory_effective_inhibitory_width_deg{};
  std::array<float, kExcitatoryNeurons>
      visual_effective_inhibitory_center_deg{};
  std::array<float, kExcitatoryNeurons>
      visual_effective_inhibitory_width_deg{};
  float auditory_rf_order_correlation = 0.0f;
  float visual_rf_order_correlation = 0.0f;
  float auditory_weight_order_correlation = 0.0f;
  float visual_weight_order_correlation = 0.0f;
  float effective_inhibitory_order_correlation = 0.0f;
  float recurrent_weight_distance_correlation = 0.0f;
  float median_rf_center_mismatch_deg = 0.0f;
  float median_effective_inhibitory_alignment_deg = 0.0f;
  float finite_rf_coverage = 0.0f;
  float map_monotonicity = 0.0f;
};

struct InverseEffectivenessAudit {
  int trials_per_salience = 0;
  std::array<float, kInverseEffectivenessCount> salience_hz{
      25.0f, 50.0f, 100.0f};
  std::array<ComponentResponseMetrics, kInverseEffectivenessCount>
      responses{};
};

struct ProjectionControlAudit {
  CausalControl control = CausalControl::kNone;
  std::array<float, 7> ampa_scale{};
  std::array<float, 7> nmda_scale{};
  std::array<float, 7> gabaa_scale{};
  std::array<bool, 7> source_rows_shuffled{};
  bool external_nmda_enabled = true;
  bool background_nmda_enabled = true;
};

struct TopologyStateMetrics {
  float auditory_to_excitatory_proximity_efficacy_correlation = 0.0f;
  int auditory_to_excitatory_contact_count = 0;
  bool auditory_to_excitatory_valid = false;
  float visual_to_excitatory_proximity_efficacy_correlation = 0.0f;
  int visual_to_excitatory_contact_count = 0;
  bool visual_to_excitatory_valid = false;
  float recurrent_proximity_efficacy_correlation = 0.0f;
  int recurrent_contact_count = 0;
  bool recurrent_valid = false;
  float effective_inhibitory_order_correlation = 0.0f;
  int effective_inhibitory_common_target_count = 0;
  bool effective_inhibitory_valid = false;
  float auditory_rf_order_correlation = 0.0f;
  bool auditory_rf_order_valid = false;
  float visual_rf_order_correlation = 0.0f;
  bool visual_rf_order_valid = false;
  float median_rf_center_mismatch_deg = 0.0f;
  int rf_common_neuron_count = 0;
  bool rf_mismatch_valid = false;
};

struct SeedTopologyAudit {
  RfTopographyAudit initial{};
  RfTopographyAudit trained{};
  float auditory_weight_order_delta = 0.0f;
  float visual_weight_order_delta = 0.0f;
  float effective_inhibitory_order_delta = 0.0f;
  float recurrent_distance_delta = 0.0f;
  float rf_monotonicity_delta = 0.0f;
  float signed_rf_mismatch_initial_deg = 0.0f;
  float signed_rf_mismatch_trained_deg = 0.0f;
  float signed_rf_mismatch_delta_deg = 0.0f;
  TopologyStateMetrics initial_state{};
  TopologyStateMetrics trained_state{};
  float auditory_to_excitatory_proximity_efficacy_delta = 0.0f;
  float visual_to_excitatory_proximity_efficacy_delta = 0.0f;
  float recurrent_proximity_efficacy_delta = 0.0f;
  float effective_inhibitory_order_delta_explicit = 0.0f;
  float median_rf_center_mismatch_delta_deg = 0.0f;
  bool initial_trained_masks_exact = false;
  bool auditory_to_inhibitory_weights_exact = false;
  bool visual_to_inhibitory_weights_exact = false;
  bool all_required_metrics_valid = false;
};

struct TopologyRefinementSummary {
  int seed_count = 0;
  TrainingCohort training_cohort = TrainingCohort::kBaseline;
  CausalControl control = CausalControl::kNone;
  int minimum_presentations_per_seed = 0;
  bool pruning_ever_enabled = false;
  bool all_initial_trained_masks_exact = false;
  bool all_fixed_sensory_to_inhibitory_weights_exact = false;
  bool evaluated = false;
  bool passed = false;
  bool all_values_valid = false;
  std::string reason;
  float mean_auditory_to_excitatory_delta = 0.0f;
  int auditory_to_excitatory_positive_seed_count = 0;
  float mean_visual_to_excitatory_delta = 0.0f;
  int visual_to_excitatory_positive_seed_count = 0;
  float mean_recurrent_delta = 0.0f;
  int recurrent_positive_seed_count = 0;
  float mean_effective_inhibitory_delta = 0.0f;
  int effective_inhibitory_positive_seed_count = 0;
  float mean_rf_mismatch_delta_deg = 0.0f;
  int rf_mismatch_negative_seed_count = 0;
};

struct SeedEvaluationMetrics {
  int seed_index = 0;
  std::uint64_t global_seed = 0;
  ObserverMetrics observer{};
  TemporalAudit temporal{};
  SpatialAudit spatial{};
  SignedSpatialAudit signed_spatial{};
  SeedTopologyAudit topology{};
  InverseEffectivenessAudit inverse_effectiveness{};
  ProjectionControlAudit control_audit{};
  float tbw50_ms = 0.0f;
  float tbw75_ms = 0.0f;
  float tbw_peak_minus_tail = 0.0f;
  float sbw50_deg = 0.0f;
  bool observer_gate = false;
  bool temporal_gate = false;
  bool spatial_gate = false;
  bool inverse_effectiveness_direction = false;
  bool topology_direction = false;
  bool criteria_passed = false;
};

struct EvaluationOptions {
  int observer_training_trials = 800;
  int observer_holdout_trials = 400;
  int trials_per_tbw_soa = 200;
  int trials_per_sbw_condition = 200;
  int trials_per_rf_location = 100;
  int trials_per_inverse_condition = 200;
  int burn_in_ms = 300;
  std::uint64_t evaluation_seed = 0x4556414C55415445ull;
  float observer_rate_hz = 50.0f;
  float response_floor_hz = 1.0f;
  CausalControl control = CausalControl::kNone;
};

struct EvaluationMetrics {
  ObserverMetrics observer{};
  TemporalAudit temporal{};
  SpatialAudit spatial{};
  RfTopographyAudit rf_topography{};
  InverseEffectivenessAudit inverse_effectiveness{};
  ProjectionControlAudit control_audit{};
  TopologyRefinementSummary topology_refinement{};
  std::vector<SeedEvaluationMetrics> per_seed;
  float tbw50_ms = 0.0f;
  float tbw75_ms = 0.0f;
  float tbw_peak_minus_tail = 0.0f;
  float sbw50_deg = 0.0f;
  std::array<float, 3> inverse_effectiveness_percent{};
  float auditory_topography_correlation = 0.0f;
  float visual_topography_correlation = 0.0f;
  float inhibitory_topography_correlation = 0.0f;
  float recurrent_topography_correlation = 0.0f;
  std::uint64_t simulated_trials = 0;
  float frozen_kernel_seconds = 0.0f;
  float elapsed_seconds = 0.0f;
  bool spatial_gate = false;
  bool criteria_passed = false;
};

struct CausalControlEvaluation {
  CausalControl control = CausalControl::kNone;
  EvaluationMetrics evaluation{};
  bool observer_frozen = false;
};

struct DeviceInfo {
  int ordinal = -1;
  int major = 0;
  int minor = 0;
  int multiprocessors = 0;
  int max_cooperative_blocks = 0;
  bool cooperative_launch = false;
  std::string name;
};

class NativeModel {
 public:
  NativeModel(const Config& config, int device, int seed_count);
  ~NativeModel();
  NativeModel(NativeModel&&) noexcept;
  NativeModel& operator=(NativeModel&&) noexcept;
  NativeModel(const NativeModel&) = delete;
  NativeModel& operator=(const NativeModel&) = delete;

  TrainingMetrics train(const TrainingOptions& options);
  std::vector<TrainingDiagnostics> diagnostics() const;
  FrozenTrialBatch run_frozen_trials(
      const std::vector<ControlledCondition>& conditions,
      int trials_per_condition, std::uint64_t evaluation_seed,
      int burn_in_steps = 300) const;
  std::vector<FrozenWeightAudit> frozen_weight_audit() const;
  std::vector<SeedWeightStateAudit> weight_state_audit() const;
  std::vector<SeedEvaluationMetrics> evaluate_per_seed(
      const EvaluationOptions& options) const;
  EvaluationMetrics evaluate(const EvaluationOptions& options) const;
  std::vector<CausalControlEvaluation> evaluate_all_controls(
      const EvaluationOptions& options,
      const EvaluationMetrics& baseline) const;
  Config config() const;
  int device() const;
  int seed_count() const;

 private:
  EvaluationMetrics evaluate_impl(
      const EvaluationOptions& options,
      const ObserverMetrics* frozen_observer,
      bool causal_only) const;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

DeviceInfo query_device(int device);
__host__ __device__ ReceptorKernel make_receptor_kernel(
    float rise_ms, float decay_ms, float dt_ms);
float nmda_voltage_block_host(float voltage_mv);
bool checked_pearson_correlation(
    const std::vector<double>& first,
    const std::vector<double>& second,
    double* correlation);
float finalize_rf_signed_rate_mean(
    double signed_rate_sum, int sample_count);
__host__ __device__ float nmda_quantum_from_ampa(float ampa_quantum);
__host__ __device__ float clopath_pair_delta(
    float eta, float pre_trace, float post_voltage_mv,
    float post_minus_mv, float post_plus_mv,
    float theta_minus_mv, float theta_plus_mv,
    float post_homeostasis_mv2, bool pre_spike);
__host__ __device__ float additive_hard_bound_update(
    float weight, float raw_delta, float lower_bound,
    float upper_bound);
__host__ __device__ float multiplicative_soft_bound_update(
    float weight, float raw_delta, float lower_bound,
    float upper_bound);
// Both traces are the values after this time bin's events have been added.
// Coincident events are applied presynaptic-first, with clipping after each.
__host__ __device__ float ordered_vogels_istdp_update(
    float weight, bool pre_event, bool post_event,
    float pre_trace_after, float post_trace_after,
    float eta, float alpha, float lower_bound,
    float upper_bound);
__host__ __device__ float shared_nmda_contact_efficacy(
    float learned_weight, float fixed_receptor_ratio,
    bool contact_active);
__host__ __device__ float calibration_nmda_contact_efficacy(
    float ampa_contact_efficacy, bool contact_active);

std::array<std::uint32_t, 4> philox4x32_10_host(
    std::array<std::uint32_t, 4> counter,
    std::array<std::uint32_t, 2> key);

std::vector<SolverOutput> run_solver_batch(
    int device, const std::vector<SolverInput>& inputs);
std::vector<float> run_receptor_trace(
    int device, const ReceptorKernel& kernel,
    const std::vector<float>& arrivals);
std::vector<AssayMetrics> evaluate_assay_candidates(
    int device, const Config& config, AssayKind kind,
    const std::vector<float>& candidates);
GabaaRelayAudit audit_gabaa_relay_timing(
    int device, const Config& config, float q_gabaa);
GabaaEfficacyAudit audit_gabaa_efficacy(
    int device, const Config& config, float q_gabaa);
LearningAssayMetrics evaluate_learning_candidates(
    int device, const Config& config, float eta_clopath, float eta_oja,
    float eta_istdp);
CalibrationResult calibrate(int device, const Config& base_config);
MsiEIntrinsicPhenotypeAudit audit_msi_e_intrinsic_phenotypes(
    int device, std::uint64_t base_seed);
GeneratorProfileAudit audit_developmental_generator(
    int device, std::uint64_t seed, int sample_count,
    float profile_center_deg, float profile_sigma_deg);
GeneratorProfileAudit audit_developmental_generator(
    int device, const Config& config, std::uint64_t seed,
    int sample_count, float profile_center_deg,
    float profile_sigma_deg);
NeuralFusionRule estimate_neural_fusion_rule(
    const FrozenTrialBatch& batch, int auditory_condition,
    int visual_condition);
EmpiricalTemporalCrossings empirical_temporal_crossings(
    const std::array<float, kTbwPointCount>& locations,
    const std::array<TemporalCurvePoint, kTbwPointCount>& points);
// Primary empirical P=0.50 validation gate; TBW75 is diagnostic only.
bool temporal_gate_passes(
    const EmpiricalTemporalCrossings& empirical,
    float peak_minus_tail);
// Pure audit of the raw orientation-pooled matched-trial enhancement curve.
// The SEM is validated as measurement metadata but does not weight, smooth,
// or otherwise alter the direct half-contrast interpolation.
DirectSpatialAudit direct_spatial_audit(
    const std::vector<float>& disparities_deg,
    const std::vector<float>& pooled_enhancement_hz,
    const std::vector<float>& pooled_enhancement_sem);
ComponentResponseMetrics summarize_matched_component_trials(
    const std::vector<float>& auditory,
    const std::vector<float>& visual,
    const std::vector<float>& audiovisual,
    float response_floor_hz);
ComponentResponseMetrics pool_spatial_orientations(
    const ComponentResponseMetrics& first,
    const ComponentResponseMetrics& second);

int cli_main(int argc, char** argv);

}  // namespace clean_msi
