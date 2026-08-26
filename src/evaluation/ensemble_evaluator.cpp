#include "evaluation/ensemble_evaluator.hpp"

#include <stdexcept>

namespace adversarial_2048 {

EnsembleEvaluator::EnsembleEvaluator(
    std::vector<std::unique_ptr<learning::NTupleNetwork>> networks)
    : networks_(std::move(networks)) {
    if (networks_.empty()) {
        throw std::invalid_argument("ensemble needs at least one network");
    }
    for (const auto& network : networks_) {
        if (network == nullptr) {
            throw std::invalid_argument("ensemble member is null");
        }
    }
    inverse_count_ = 1.0 / static_cast<double>(networks_.size());
}

double EnsembleEvaluator::evaluate(Board board) const {
    // Plain mean. Deliberately unweighted: weighting by each member's measured
    // strength would fit the weights to the benchmark seeds, and the whole
    // point is that the members are individually comparable but fail on
    // different positions.
    double total = 0.0;
    for (const auto& network : networks_) {
        total += network->value(board);
    }
    return total * inverse_count_;
}

}  // namespace adversarial_2048
