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
extern "C" int temporal_build(const std::uint32_t* tags, const double* odds, int count,
                              int in_size, Edge* output) {
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
            const auto size = std::min(links.size(), std::size_t(fanout));
            std::partial_sort(links.begin(), links.begin() + size, links.end(),
                              [](const Link& a, const Link& b) {
                return earlier({a.boundary, a.mass[0]}, {b.boundary, b.mass[0]});
            });
            for (std::size_t j = 0; j < size; ++j) {
                Edge& edge = output[std::size_t(i) * fanout + j];
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

namespace {
struct Beam {
    std::array<Link, join_width> links;
    int count = 1;

    Beam() { links[0] = {0, {1.0, 1.0, 1.0, 1.0, 1.0}}; }
};

struct Scratch {
    std::vector<Mass> sums = std::vector<Mass>(1u << out_bits);
    std::vector<RankedLink> ranked;

    Scratch() { ranked.reserve(beam_width * fanout); }

    void advance(Beam& beam, const Edge* table, const double* marginal,
                 unsigned syndrome, unsigned next_syndrome, int width) {
        ranked.clear();
        for (int i = 0; i < beam.count; ++i) {
            const auto& state = beam.links[i];
            const auto row = state.boundary ^ syndrome;
            const Edge* edges = table + std::size_t(row) * fanout;
            for (int j = 0; j < fanout && edges[j].mass[0] > 0; ++j) {
                const auto& edge = edges[j];
                auto& mass = sums[edge.outgoing];
                if (mass[0] == 0) ranked.push_back({edge.outgoing, 0});
                propagate(edge, state.mass, mass);
            }
        }
        for (auto& item : ranked) {
            item.score = sums[item.boundary][0] * marginal[item.boundary ^ next_syndrome];
        }
        beam.count = std::min(int(ranked.size()), width);
        std::partial_sort(ranked.begin(), ranked.begin() + beam.count, ranked.end(), earlier);
        for (int i = 0; i < beam.count; ++i) {
            beam.links[i] = {ranked[i].boundary, sums[ranked[i].boundary]};
        }
        for (const auto& item : ranked) sums[item.boundary] = {};
    }
};

template <typename Finish>
int decode_rows(const std::uint16_t* observed, std::size_t shots,
                const Edge* const* tables, const double* const* marginals,
                const double* terminal, Finish finish) {
    try {
        Scratch scratch;
        for (std::size_t shot = 0; shot < shots; ++shot) {
            const auto* syndrome = observed + 6 * shot;
            Beam beam;
            for (int time = 0; time < 5 && beam.count; ++time) {
                scratch.advance(beam, tables[time], marginals[time], syndrome[time],
                                syndrome[time + 1], beam_width);
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
    return decode_rows(observed, shots, tables, marginals, terminal, [output](std::size_t shot, const Mass& posterior) {
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
    return decode_rows(observed, shots, tables, marginals, terminal, [output](std::size_t shot, const Mass& posterior) {
        std::copy(posterior.begin(), posterior.end(), output + moments * shot);
    });
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
