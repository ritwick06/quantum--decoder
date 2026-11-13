#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
constexpr int moments = 5;
constexpr int fanout = 48;
constexpr int beam_width = 32;
constexpr int join_width = 48;
constexpr int wide_width = 128;
constexpr unsigned out_bits = 14;
constexpr unsigned out_mask = (1u << out_bits) - 1;
using Mass = std::array<double, moments>;

struct Subset {
    std::uint32_t tag;
    double odds;
};

struct Edge {
    std::uint32_t outgoing;
    float mass[moments];
};
static_assert(sizeof(Edge) == 24);

struct Link {
    unsigned boundary;
    Mass mass;
};

struct RankedLink {
    unsigned boundary;
    double score;
};

struct BackEdge {
    std::uint32_t incoming;
    float mass;
};

struct BackwardIndex {
    const std::uint32_t* offsets;
    const BackEdge* edges;
    const std::uint16_t* residuals;
};

bool earlier(const RankedLink& left, const RankedLink& right) {
    if (left.score != right.score) return left.score > right.score;
    return left.boundary < right.boundary;
}

void collect(const std::vector<Subset>& row, std::vector<Link>& edges) {
    edges.clear();
    for (const auto& subset : row) {
        const unsigned boundary = subset.tag >> 4;
        if (edges.empty() || edges.back().boundary != boundary) edges.push_back({boundary, {}});
        auto& mass = edges.back().mass;
        mass[0] += subset.odds;
        for (int j = 0; j < 4; ++j) {
            mass[j + 1] += (subset.tag & (1u << j)) ? -subset.odds : subset.odds;
        }
    }
}

void propagate(const Edge& edge, const Mass& source, Mass& target) {
    for (int j = 0; j < moments; ++j) target[j] += source[j] * edge.mass[j];
}
}

// Each tag places the input parity above its outgoing parity and logical bits.
extern "C" int temporal_build_width(const std::uint32_t* tags, const double* odds, int count,
                                    int in_size, Edge* output, unsigned edge_width) {
    try {
        std::vector<std::vector<Subset>> rows(in_size);
        const auto append = [&](std::uint32_t tag, double weight) {
            rows[tag >> (out_bits + 4)].push_back({tag & ((out_mask << 4) | 15u), weight});
        };
        append(0, 1.0);
        for (int i = 0; i < count; ++i) {
            append(tags[i], odds[i]);
            for (int j = i + 1; j < count; ++j) {
                const auto pair = tags[i] ^ tags[j];
                const auto pair_odds = odds[i] * odds[j];
                append(pair, pair_odds);
                for (int k = j + 1; k < count; ++k) append(pair ^ tags[k], pair_odds * odds[k]);
            }
        }
        std::vector<Link> links;
        for (int i = 0; i < in_size; ++i) {
            auto& row = rows[i];
            std::sort(row.begin(), row.end(), [](const Subset& a, const Subset& b) {
                if (a.tag != b.tag) return a.tag < b.tag;
                return a.odds < b.odds;
            });
            collect(row, links);
            const auto size = std::min(links.size(), std::size_t(edge_width));
            std::partial_sort(links.begin(), links.begin() + size, links.end(),
                              [](const Link& a, const Link& b) {
                return earlier({a.boundary, a.mass[0]}, {b.boundary, b.mass[0]});
            });
            for (std::size_t j = 0; j < size; ++j) {
                Edge& edge = output[std::size_t(i) * edge_width + j];
                edge.outgoing = links[j].boundary;
                for (int k = 0; k < moments; ++k) edge.mass[k] = static_cast<float>(links[j].mass[k]);
            }
            std::vector<Subset>().swap(row);
        }
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" int temporal_build(const std::uint32_t* tags, const double* odds, int count,
                              int in_size, Edge* output) {
    return temporal_build_width(tags, odds, count, in_size, output, fanout);
}

namespace {
struct Beam {
    std::array<Link, wide_width> links;
    int count = 1;

    Beam() { links[0] = {0, {1.0, 1.0, 1.0, 1.0, 1.0}}; }
};

struct Scratch {
    std::vector<Mass> sums = std::vector<Mass>(1u << out_bits);
    std::vector<RankedLink> ranked;

    Scratch() { ranked.reserve(beam_width * fanout); }

    template <typename Likelihood>
    void advance_ranked(Beam& beam, const Edge* table, unsigned syndrome,
                        int width, Likelihood likelihood, unsigned edge_width = fanout) {
        ranked.clear();
        for (int i = 0; i < beam.count; ++i) {
            const auto& state = beam.links[i];
            const auto row = state.boundary ^ syndrome;
            const Edge* edges = table + std::size_t(row) * edge_width;
            for (unsigned j = 0; j < edge_width && edges[j].mass[0] > 0; ++j) {
                const auto& edge = edges[j];
                auto& mass = sums[edge.outgoing];
                if (mass[0] == 0) ranked.push_back({edge.outgoing, 0});
                propagate(edge, state.mass, mass);
            }
        }
        for (auto& item : ranked) {
            item.score = sums[item.boundary][0] * likelihood(item.boundary);
        }
        beam.count = std::min(int(ranked.size()), width);
        std::partial_sort(ranked.begin(), ranked.begin() + beam.count, ranked.end(), earlier);
        for (int i = 0; i < beam.count; ++i) {
            beam.links[i] = {ranked[i].boundary, sums[ranked[i].boundary]};
        }
        for (const auto& item : ranked) sums[item.boundary] = {};
    }

    void advance(Beam& beam, const Edge* table, const double* marginal,
                 unsigned syndrome, unsigned next_syndrome, int width,
                 unsigned likelihood_shift = 0, unsigned terminal_syndrome = 0,
                 unsigned edge_width = fanout) {
        advance_ranked(beam, table, syndrome, width, [=](unsigned boundary) {
            return marginal[((boundary ^ next_syndrome) << likelihood_shift) | terminal_syndrome];
        }, edge_width);
    }
};

template <typename Finish>
int decode_rows(const std::uint16_t* observed, std::size_t shots,
                const Edge* const* tables, const double* const* marginals,
                const double* terminal, int width, Finish finish) {
    try {
        Scratch scratch;
        for (std::size_t shot = 0; shot < shots; ++shot) {
            const auto* syndrome = observed + 6 * shot;
            Beam beam;
            for (int time = 0; time < 5 && beam.count; ++time) {
                scratch.advance(beam, tables[time], marginals[time], syndrome[time],
                                syndrome[time + 1], width);
            }
            Mass posterior{};
            for (int i = 0; i < beam.count; ++i) {
                const auto& state = beam.links[i];
                const auto* local = terminal + (state.boundary ^ syndrome[5]) * moments;
                for (int j = 0; j < moments; ++j) posterior[j] += state.mass[j] * local[j];
            }
            finish(shot, posterior);
        }
        return 0;
    } catch (...) {
        return 1;
    }
}
}

extern "C" int temporal_decode(const std::uint16_t* observed, std::size_t shots,
                               const Edge* const* tables, const double* const* marginals,
                               const double* terminal, std::int8_t* output) {
    return decode_rows(observed, shots, tables, marginals, terminal, beam_width, [output](std::size_t shot, const Mass& posterior) {
        for (int bit = 0; bit < 4; ++bit) {
            const auto signal = posterior[bit + 1];
            output[4 * shot + bit] = posterior[0] <= 0 || std::abs(signal) <= 0.025 * posterior[0]
                ? -1 : std::int8_t(signal < 0);
        }
    });
}

extern "C" int temporal_moments(const std::uint16_t* observed, std::size_t shots,
                                const Edge* const* tables, const double* const* marginals,
                                const double* terminal, double* output) {
    return decode_rows(observed, shots, tables, marginals, terminal, beam_width, [output](std::size_t shot, const Mass& posterior) {
        std::copy(posterior.begin(), posterior.end(), output + moments * shot);
    });
}

extern "C" int temporal_wide_moments(const std::uint16_t* observed, std::size_t shots,
                                     const Edge* const* tables, const double* const* marginals,
                                     const double* terminal, double* output) {
    return decode_rows(observed, shots, tables, marginals, terminal, wide_width, [output](std::size_t shot, const Mass& posterior) {
        std::copy(posterior.begin(), posterior.end(), output + moments * shot);
    });
}

namespace {
int decode_suffix(const std::uint16_t* observed, std::size_t shots,
                  const Edge* const* tables, const double* const* marginals,
                  const float* suffix, const float* const* joints,
                  const std::int16_t* const* indexes, double* output, unsigned edge_width = fanout,
                  const BackwardIndex* backward = nullptr) {
    try {
        Scratch scratch;
        std::vector<double> tail(backward ? 1u << out_bits : 0u);
        for (std::size_t shot = 0; shot < shots; ++shot) {
            const auto* syndrome = observed + 6 * shot;
            Beam beam;
            for (int t = 0; t < 3 && beam.count; ++t) {
                const auto next = syndrome[t + 1];
                const auto row = indexes ? indexes[t][syndrome[t + 2]] : -1;
                if (backward && t == 2) {
                    constexpr unsigned kept = 64;
                    const auto last = syndrome[5];
                    const auto* parities = backward->residuals + last * (kept + 1);
                    const double floor = marginals[3][(parities[kept] << 7) | last];
                    std::fill(tail.begin(), tail.end(), 0.0);
                    for (unsigned i = 0; i < kept; ++i) {
                        const auto residual = parities[i];
                        const auto outgoing = residual ^ syndrome[4];
                        const double weight = marginals[3][(residual << 7) | last];
                        for (auto j = backward->offsets[outgoing]; j < backward->offsets[outgoing + 1]; ++j) {
                            const auto edge = backward->edges[j];
                            tail[edge.incoming] += edge.mass * weight;
                        }
                    }
                    scratch.advance_ranked(beam, tables[t], syndrome[t], beam_width, [&](unsigned boundary) {
                        const auto incoming = boundary ^ next;
                        return tail[incoming] + floor * marginals[t][incoming];
                    }, edge_width);
                } else if (row >= 0) {
                    const auto* joint = joints[t] + (std::size_t(row) << out_bits);
                    scratch.advance_ranked(beam, tables[t], syndrome[t], beam_width,
                                           [=](unsigned boundary) { return joint[boundary ^ next]; });
                } else {
                    scratch.advance(beam, tables[t], marginals[t], syndrome[t], next, beam_width, 0, 0, edge_width);
                }
            }
            scratch.advance(beam, tables[3], marginals[3], syndrome[3], syndrome[4],
                            beam_width, 7, syndrome[5], edge_width);
            Mass posterior{};
            for (int i = 0; i < beam.count; ++i) {
                const auto& state = beam.links[i];
                const auto parity = ((state.boundary ^ syndrome[4]) << 7) | syndrome[5];
                const auto* local = suffix + std::size_t(parity) * moments;
                for (int j = 0; j < moments; ++j) posterior[j] += state.mass[j] * local[j];
            }
            std::copy(posterior.begin(), posterior.end(), output + moments * shot);
        }
        return 0;
    } catch (...) {
        return 1;
    }
}
}

extern "C" int temporal_suffix_moments(const std::uint16_t* observed, std::size_t shots,
                                       const Edge* const* tables, const double* const* marginals,
                                       const float* suffix, double* output) {
    return decode_suffix(observed, shots, tables, marginals, suffix, nullptr, nullptr, output);
}

extern "C" int temporal_expanded_moments(const std::uint16_t* observed, std::size_t shots,
                                         const Edge* const* tables, const double* const* marginals,
                                         const float* suffix, double* output) {
    return decode_suffix(observed, shots, tables, marginals, suffix, nullptr, nullptr, output, 96);
}

extern "C" int temporal_backward_moments(const std::uint16_t* observed, std::size_t shots,
                                         const Edge* const* tables, const double* const* marginals,
                                         const float* suffix, const std::uint32_t* offsets,
                                         const BackEdge* edges, const std::uint16_t* residuals,
                                         double* output) {
    const BackwardIndex backward{offsets, edges, residuals};
    return decode_suffix(observed, shots, tables, marginals, suffix, nullptr, nullptr, output, 96, &backward);
}

extern "C" int temporal_lookahead_moments(const std::uint16_t* observed, std::size_t shots,
                                          const Edge* const* tables, const double* const* marginals,
                                          const float* suffix, const float* const* joints,
                                          const std::int16_t* const* indexes, double* output) {
    return decode_suffix(observed, shots, tables, marginals, suffix, joints, indexes, output);
}

// Tables are ordered left to right for two slices, then right to left for three.
extern "C" int temporal_midpoint(const std::uint16_t* observed, std::size_t shots,
                                 const Edge* const* tables, const double* const* marginals,
                                 const double* middle, double* output) {
    try {
        Scratch scratch;
        for (std::size_t shot = 0; shot < shots; ++shot) {
            const auto* syndrome = observed + 6 * shot;
            Beam left, right;
            scratch.advance(left, tables[0], marginals[0], syndrome[0], syndrome[1], beam_width);
            scratch.advance(left, tables[1], marginals[1], syndrome[1], syndrome[2], join_width);
            scratch.advance(right, tables[2], marginals[2], syndrome[5], syndrome[4], beam_width);
            scratch.advance(right, tables[3], marginals[3], syndrome[4], syndrome[3], beam_width);
            scratch.advance(right, tables[4], marginals[4], syndrome[3], syndrome[2], join_width);

            Mass posterior{};
            for (int i = 0; i < left.count; ++i) {
                const auto& a = left.links[i];
                const auto residual = syndrome[2] ^ a.boundary;
                Mass contracted{};
                for (int j = 0; j < right.count; ++j) {
                    const auto& b = right.links[j];
                    const auto* local = middle + (residual ^ b.boundary) * moments;
                    for (int k = 0; k < moments; ++k) contracted[k] += b.mass[k] * local[k];
                }
                for (int k = 0; k < moments; ++k) posterior[k] += a.mass[k] * contracted[k];
            }
            std::copy(posterior.begin(), posterior.end(), output + moments * shot);
        }
        return 0;
    } catch (...) {
        return 1;
    }
}
