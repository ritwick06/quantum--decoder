#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {
constexpr unsigned fanout = 48;
constexpr unsigned tag_out_bits = 14;
constexpr unsigned tag_out_mask = (1u << tag_out_bits) - 1;
constexpr unsigned channels = 5;
constexpr double mass_floor = 2e-15;

struct Edge {
    std::uint32_t outgoing;
    float mass[channels];
};
static_assert(sizeof(Edge) == 24);

void walsh_hadamard(double* values, std::size_t size) {
    for (std::size_t step = 1; step < size; step *= 2) {
        const auto blocks = size / (2 * step);
        const auto segment = std::min(step, std::size_t(65536));
        const auto segments = step / segment;
        #pragma omp parallel for if(size >= (1u << 20)) num_threads(4) schedule(static)
        for (std::size_t task = 0; task < blocks * segments; ++task) {
            const auto start = (task / segments) * 2 * step + (task % segments) * segment;
            double* low = values + start;
            double* high = low + step;
            for (std::size_t i = 0; i < segment; ++i) {
                const double a = low[i];
                const double b = high[i];
                low[i] = a + b;
                high[i] = a - b;
            }
        }
    }
}

void distribution(std::vector<double>& work, const std::uint32_t* tags, const double* p,
                  int count, unsigned out_bits, unsigned channel) {
    const auto size = work.size();
    #pragma omp parallel for if(size >= (1u << 20)) num_threads(4) schedule(static)
    for (std::size_t k = 0; k < size; ++k) work[k] = 0;
    for (int i = 0; i < count; ++i) {
        const auto tag = tags[i];
        const auto incoming = tag >> (tag_out_bits + 4);
        const auto outgoing = (tag >> 4) & tag_out_mask;
        const auto mask = (incoming << out_bits) | outgoing;
        const double coefficient = 0.5 * std::log1p(-2 * p[i]);
        const bool flip = channel && (tag & (1u << (channel - 1)));
        work[0] += coefficient;
        work[mask] += flip ? coefficient : -coefficient;
    }
    walsh_hadamard(work.data(), size);
    #pragma omp parallel for if(size >= (1u << 20)) num_threads(4) schedule(static)
    for (std::size_t k = 0; k < size; ++k) work[k] = std::exp(work[k]) / double(size);
    walsh_hadamard(work.data(), size);
}

struct Ranked {
    unsigned outgoing;
    double mass;
};

bool earlier(const Ranked& a, const Ranked& b) {
    if (a.mass != b.mass) return a.mass > b.mass;
    return a.outgoing < b.outgoing;
}
}

extern "C" int temporal_build_spectral(const std::uint32_t* tags, const double* p, int count,
                                       unsigned in_bits, unsigned out_bits, Edge* output) {
    if (count < 0 || in_bits > 14 || out_bits > 14) return 2;
    const unsigned rows = 1u << in_bits;
    const unsigned columns = 1u << out_bits;
    for (int i = 0; i < count; ++i) {
        if (!(p[i] >= 0 && p[i] < 0.5)
            || (tags[i] >> (tag_out_bits + 4)) >= rows
            || ((tags[i] >> 4) & tag_out_mask) >= columns) return 2;
    }
    try {
        std::vector<double> work(std::size_t(rows) * columns);
        distribution(work, tags, p, count, out_bits, 0);
        #pragma omp parallel for if(rows >= 1024) num_threads(4) schedule(static)
        for (unsigned row = 0; row < rows; ++row) {
            Ranked ranked[fanout];
            unsigned used = 0;
            const double* local = work.data() + std::size_t(row) * columns;
            for (unsigned col = 0; col < columns; ++col) {
                Ranked candidate{col, local[col]};
                if (candidate.mass <= mass_floor) continue;
                if (used < fanout) {
                    ranked[used++] = candidate;
                    std::push_heap(ranked, ranked + used, earlier);
                } else if (earlier(candidate, ranked[0])) {
                    std::pop_heap(ranked, ranked + used, earlier);
                    ranked[used - 1] = candidate;
                    std::push_heap(ranked, ranked + used, earlier);
                }
            }
            std::sort_heap(ranked, ranked + used, earlier);
            for (unsigned j = 0; j < used; ++j) {
                auto& edge = output[std::size_t(row) * fanout + j];
                edge = {ranked[j].outgoing, {float(ranked[j].mass), 0, 0, 0, 0}};
            }
            for (unsigned j = used; j < fanout; ++j) output[std::size_t(row) * fanout + j] = {};
        }
        for (unsigned channel = 1; channel < channels; ++channel) {
            distribution(work, tags, p, count, out_bits, channel);
            #pragma omp parallel for if(rows >= 1024) num_threads(4) schedule(static)
            for (unsigned row = 0; row < rows; ++row) {
                const double* local = work.data() + std::size_t(row) * columns;
                auto* edges = output + std::size_t(row) * fanout;
                for (unsigned j = 0; j < fanout && edges[j].mass[0] > 0; ++j) {
                    auto& edge = edges[j];
                    const double mass = edge.mass[0];
                    edge.mass[channel] = float(std::clamp(local[edge.outgoing], -mass, mass));
                }
            }
        }
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" int temporal_build_suffix(const std::uint32_t* tags, const double* p, int count,
                                     float* output) {
    constexpr unsigned in_bits = 14;
    constexpr unsigned out_bits = 7;
    constexpr std::size_t size = 1u << (in_bits + out_bits);
    for (int i = 0; i < count; ++i) {
        if (!(p[i] >= 0 && p[i] < 0.5) || ((tags[i] >> 4) & tag_out_mask) >= (1u << out_bits)) return 2;
    }
    try {
        std::vector<double> work(size);
        for (unsigned channel = 0; channel < channels; ++channel) {
            distribution(work, tags, p, count, out_bits, channel);
            #pragma omp parallel for num_threads(4) schedule(static)
            for (std::size_t k = 0; k < size; ++k) {
                auto* moments = output + channels * k;
                if (channel == 0) {
                    moments[0] = work[k] > mass_floor ? float(work[k]) : 0.0f;
                } else {
                    const double mass = moments[0];
                    moments[channel] = float(std::clamp(work[k], -mass, mass));
                }
            }
        }
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" int temporal_build_lookahead(const std::uint32_t* tags, const double* p, int count,
                                       const std::uint16_t* parities, unsigned selected, float* output) {
    constexpr unsigned width = 1u << 14;
    for (unsigned row = 0; row < selected; ++row) {
        if (parities[row] >= width) return 2;
    }
    for (int i = 0; i < count; ++i) {
        if (!(p[i] >= 0 && p[i] < 0.5)) return 2;
    }
    try {
        std::vector<double> work(std::size_t(width) * width);
        distribution(work, tags, p, count, 14, 0);
        for (unsigned row = 0; row < selected; ++row) {
            const double* values = work.data() + std::size_t(parities[row]) * width;
            for (unsigned i = 0; i < width; ++i) {
                output[std::size_t(row) * width + i] = values[i] > mass_floor ? float(values[i]) : 0.0f;
            }
        }
        return 0;
    } catch (...) {
        return 1;
    }
}
