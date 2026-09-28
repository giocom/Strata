// src/core/expert_cache.cpp - R4's slot storage and residency table.  Read the header first.
#include "strata/core/expert_cache.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <utility>
#include <cstring>

namespace strata::core {

bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "read_expert_profile: cannot open " + path;
        return false;
    }
    char magic[4] = {0, 0, 0, 0};
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    if (std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 5, f) != 5) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " is too short to hold a header";
        return false;
    }
    if (std::memcmp(magic, "STRP", 4) != 0) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " does not start with STRP";
        return false;
    }
    const uint32_t version = hdr[0], nl = hdr[1], ne = hdr[2], want = hdr[3], n_ranked = hdr[4];
    if ((int64_t) nl != n_layers || (int64_t) ne != n_expert) {
        std::fclose(f);
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "read_expert_profile: %s is %ux%u but this model is %lldx%lld - it is a profile for a "
                      "different artifact", path.c_str(), nl, ne, (long long) n_layers, (long long) n_expert);
        err = buf;
        return false;
    }
    if (n_ranked > want) {
        std::fclose(f);
        err = "read_expert_profile: the header claims more ranked pairs than slots";
        return false;
    }
    ranked.assign(n_ranked, {0, 0});
    std::vector<uint16_t> raw((size_t) n_ranked * 2);
    if (n_ranked > 0 && std::fread(raw.data(), 2, (size_t) n_ranked * 2, f) != (size_t) n_ranked * 2) {
        std::fclose(f);
        err = "read_expert_profile: the ranked list is truncated";
        return false;
    }
    std::fclose(f);
    for (uint32_t i = 0; i < n_ranked; ++i) {
        const int32_t l = (int32_t) raw[(size_t) i * 2], e = (int32_t) raw[(size_t) i * 2 + 1];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "read_expert_profile: pair %u is (layer %d, expert %d), out of range",
                          i, l, e);
            err = buf;
            return false;
        }
        ranked[(size_t) i] = {l, e};
    }
    slots = (int64_t) want;
    (void) version;   // a future format bumps it; the layout check above is what protects this reader today
    return true;
}

ExpertCache::~ExpertCache() { close(); }

bool ExpertCache::open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
                       std::string& err) {
    close();
    if (n_slots <= 0) {
        err = "ExpertCache: n_slots must be positive";
        return false;
    }
    if (n_layers <= 0 || n_expert <= 0 || blob_bytes <= 0) {
        err = "ExpertCache: n_layers, n_expert and blob_bytes must all be positive";
        return false;
    }

    int dev_count = 1;
    cudaGetDeviceCount(&dev_count);
    int orig_device = 0;
    cudaGetDevice(&orig_device);

    const uint64_t total_want = (uint64_t) n_slots * (uint64_t) blob_bytes;

    std::vector<size_t> dev_free(dev_count, 0);
    uint64_t total_free_b = 0;
    for (int d = 0; d < dev_count; ++d) {
        cudaSetDevice(d);
        size_t fb = 0, tb = 0;
        cudaMemGetInfo(&fb, &tb);
        const size_t reserve = 512ull << 20;
        dev_free[d] = fb > reserve ? fb - reserve : 0;
        total_free_b += dev_free[d];
    }
    cudaSetDevice(orig_device);

    if (total_free_b < total_want) {
        char buf[320];
        std::snprintf(buf, sizeof buf,
                      "ExpertCache: %lld slots x %lld B = %.2f GiB, but only %.2f GiB of VRAM is free across %d GPUs.",
                      (long long) n_slots, (long long) blob_bytes, (double) total_want / 1073741824.0,
                      (double) total_free_b / 1073741824.0, dev_count);
        err = buf;
        return false;
    }

    int64_t current_slot = 0;
    for (int d = 0; d < dev_count && current_slot < n_slots; ++d) {
        const int64_t start_s = current_slot;
        const int64_t max_d_slots = (int64_t) (dev_free[d] / (size_t) blob_bytes);
        int64_t s_count = n_slots - current_slot;
        if (d + 1 < dev_count && s_count > max_d_slots) {
            s_count = max_d_slots;
        }
        if (s_count <= 0) continue;
        const uint64_t seg_bytes = (uint64_t) s_count * (uint64_t) blob_bytes;

        cudaSetDevice(d);
        uint8_t* p = nullptr;
        if (cudaMalloc((void**) &p, (size_t) seg_bytes) != cudaSuccess) {
            cudaSetDevice(orig_device);
            close();
            char buf[256];
            std::snprintf(buf, sizeof buf, "ExpertCache: cudaMalloc(%.2f GiB) on GPU %d failed: %s",
                          (double) seg_bytes / 1073741824.0, d, cudaGetErrorString(cudaGetLastError()));
            err = buf;
            return false;
        }
        if (cudaMemset(p, 0, (size_t) seg_bytes) != cudaSuccess) {
            cudaSetDevice(orig_device);
            close();
            err = "ExpertCache: cudaMemset failed on GPU " + std::to_string(d);
            return false;
        }

        CacheSegment seg;
        seg.base = p;
        seg.ordinal = d;
        seg.start_slot = start_s;
        seg.n_slots = s_count;
        seg.bytes = seg_bytes;
        segments_.push_back(seg);
        current_slot += s_count;
    }
    for (int i = 0; i < dev_count; ++i) {
        cudaSetDevice(i);
        for (int j = 0; j < dev_count; ++j) {
            if (i != j) {
                int can_access = 0;
                cudaDeviceCanAccessPeer(&can_access, i, j);
                if (can_access) {
                    cudaDeviceEnablePeerAccess(j, 0);
                    cudaGetLastError();
                }
            }
        }
    }
    cudaSetDevice(orig_device);

    if (segments_.empty()) {
        err = "ExpertCache: no segments allocated";
        return false;
    }
    base_ = segments_[0].base;

    residency_.assign((size_t) (n_layers * n_expert), kNotResident);
    slots_ = n_slots;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    blob_ = blob_bytes;
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    layer_next_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    for (int64_t l = 0; l < n_layers; ++l) {
        int64_t lo = 0, hi = 0;
        layer_slot_range(l, lo, hi);
        layer_next_[(size_t) l] = (int32_t) lo;
    }
    return true;
}

bool ExpertCache::open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_layers, int64_t n_expert,
                             std::string& err) {
    close();
    if (slot_bytes.empty()) { err = "ExpertCache: no slots"; return false; }
    if (n_layers <= 0 || n_expert <= 0) {
        err = "ExpertCache: n_layers and n_expert must be positive";
        return false;
    }

    int dev_count = 1;
    cudaGetDeviceCount(&dev_count);
    int orig_device = 0;
    cudaGetDevice(&orig_device);

    int64_t mx = 0;
    std::vector<uint64_t> off(slot_bytes.size() + 1, 0);
    for (size_t i = 0; i < slot_bytes.size(); ++i) {
        off[i + 1] = off[i] + ((uint64_t) slot_bytes[i] + 255) / 256 * 256;
        mx = slot_bytes[i] > mx ? slot_bytes[i] : mx;
    }
    const uint64_t total_want = off.back();

    std::vector<size_t> dev_free(dev_count, 0);
    uint64_t total_free_b = 0;
    for (int d = 0; d < dev_count; ++d) {
        cudaSetDevice(d);
        size_t fb = 0, tb = 0;
        cudaMemGetInfo(&fb, &tb);
        const size_t reserve = 512ull << 20;
        dev_free[d] = fb > reserve ? fb - reserve : 0;
        total_free_b += dev_free[d];
    }
    cudaSetDevice(orig_device);

    if (total_free_b < total_want) {
        char buf[320];
        std::snprintf(buf, sizeof buf,
                      "ExpertCache: %zu sized slots = %.2f GiB, but only %.2f GiB of VRAM is free across %d GPUs.",
                      slot_bytes.size(), (double) total_want / 1073741824.0, (double) total_free_b / 1073741824.0, dev_count);
        err = buf;
        return false;
    }

    size_t current_slot = 0;
    for (int d = 0; d < dev_count && current_slot < slot_bytes.size(); ++d) {
        const size_t start_s = current_slot;
        uint64_t seg_bytes = 0;
        while (current_slot < slot_bytes.size()) {
            const uint64_t b = ((uint64_t) slot_bytes[current_slot] + 255) / 256 * 256;
            if (d + 1 < dev_count && seg_bytes + b > dev_free[d] && current_slot > start_s) {
                break;
            }
            seg_bytes += b;
            ++current_slot;
        }
        if (seg_bytes == 0) continue;

        cudaSetDevice(d);
        uint8_t* p = nullptr;
        if (cudaMalloc((void**) &p, (size_t) seg_bytes) != cudaSuccess) {
            cudaSetDevice(orig_device);
            close();
            char buf[256];
            std::snprintf(buf, sizeof buf, "ExpertCache: cudaMalloc(%.2f GiB) on GPU %d failed: %s",
                          (double) seg_bytes / 1073741824.0, d, cudaGetErrorString(cudaGetLastError()));
            err = buf;
            return false;
        }
        if (cudaMemset(p, 0, (size_t) seg_bytes) != cudaSuccess) {
            cudaSetDevice(orig_device);
            close();
            err = "ExpertCache: cudaMemset failed on GPU " + std::to_string(d);
            return false;
        }

        CacheSegment seg;
        seg.base = p;
        seg.ordinal = d;
        seg.start_slot = (int64_t) start_s;
        seg.n_slots = (int64_t) (current_slot - start_s);
        seg.bytes = seg_bytes;
        segments_.push_back(seg);
    }
    for (int i = 0; i < dev_count; ++i) {
        cudaSetDevice(i);
        for (int j = 0; j < dev_count; ++j) {
            if (i != j) {
                int can_access = 0;
                cudaDeviceCanAccessPeer(&can_access, i, j);
                if (can_access) {
                    cudaDeviceEnablePeerAccess(j, 0);
                    cudaGetLastError();
                }
            }
        }
    }
    cudaSetDevice(orig_device);

    if (segments_.empty()) {
        err = "ExpertCache: no segments allocated";
        return false;
    }
    base_ = segments_[0].base;

    residency_.assign((size_t) (n_layers * n_expert), kNotResident);
    slots_ = (int64_t) slot_bytes.size();
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    blob_ = mx;
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    off_ = std::move(off);
    layer_next_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    for (int64_t l = 0; l < n_layers; ++l) {
        int64_t lo = 0, hi = 0;
        layer_slot_range(l, lo, hi);
        layer_next_[(size_t) l] = (int32_t) lo;
    }
    return true;
}

void ExpertCache::zero_all() {
    int orig_device = 0;
    cudaGetDevice(&orig_device);
    for (const auto& seg : segments_) {
        if (seg.base != nullptr && seg.bytes > 0) {
            cudaSetDevice(seg.ordinal);
            cudaMemset(seg.base, 0, (size_t) seg.bytes);
        }
    }
    cudaSetDevice(orig_device);
}

void ExpertCache::close() {
    off_.clear();
    int orig_device = 0;
    cudaGetDevice(&orig_device);
    for (auto& seg : segments_) {
        if (seg.base != nullptr) {
            cudaSetDevice(seg.ordinal);
            cudaFree(seg.base);
            seg.base = nullptr;
        }
    }
    cudaSetDevice(orig_device);
    segments_.clear();
    base_ = nullptr;
    residency_.clear();
    slots_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    blob_ = 0;
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    layer_next_.clear();
}

/// R4.2g.  Layer `l` owns `[l*q, (l+1)*q)` with `q = slots_ / n_layers_`; the LAST layer takes whatever is
/// left over, so the ranges always cover `0..slots_` exactly and no slot is orphaned by the division.
void ExpertCache::layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const {
    lo = 0;
    hi = 0;
    if (n_layers_ <= 0 || slots_ <= 0 || layer < 0 || layer >= n_layers_) return;
    const int64_t q = slots_ / n_layers_;
    lo = layer * q;
    hi = (layer == n_layers_ - 1) ? slots_ : (layer + 1) * q;
}

int32_t ExpertCache::slot_of(int64_t layer, int64_t expert) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    return residency_[(size_t) (layer * n_expert_ + expert)];
}

int32_t ExpertCache::admit(int64_t layer, int64_t expert) {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    const size_t at = (size_t) (layer * n_expert_ + expert);
    if (residency_[at] != kNotResident) return residency_[at];
    if (per_layer_) {
        if (layer_next_.empty()) return kNotResident;
        int64_t lo = 0, hi = 0;
        layer_slot_range(layer, lo, hi);
        if ((int64_t) layer_next_[(size_t) layer] >= hi) return kNotResident;
        residency_[at] = layer_next_[(size_t) layer]++;
        ++admitted_;
        return residency_[at];
    }
    if (next_free_ >= slots_) return kNotResident;
    residency_[at] = (int32_t) next_free_;
    return (int32_t) next_free_++;
}

uint8_t* ExpertCache::device_slot(int32_t slot) {
    if (slot < 0 || slot >= slots_) return nullptr;
    if (!segments_.empty()) {
        for (const auto& seg : segments_) {
            if (slot >= seg.start_slot && slot < seg.start_slot + seg.n_slots) {
                if (!off_.empty()) {
                    uint64_t local_off = off_[(size_t) slot] - off_[(size_t) seg.start_slot];
                    return seg.base + local_off;
                }
                return seg.base + (size_t) (slot - seg.start_slot) * (size_t) blob_;
            }
        }
    }
    if (!off_.empty()) return base_ + off_[(size_t) slot];
    return base_ + (size_t) slot * (size_t) blob_;
}

const uint8_t* ExpertCache::device_slot(int32_t slot) const {
    if (slot < 0 || slot >= slots_) return nullptr;
    if (!segments_.empty()) {
        for (const auto& seg : segments_) {
            if (slot >= seg.start_slot && slot < seg.start_slot + seg.n_slots) {
                if (!off_.empty()) {
                    uint64_t local_off = off_[(size_t) slot] - off_[(size_t) seg.start_slot];
                    return seg.base + local_off;
                }
                return seg.base + (size_t) (slot - seg.start_slot) * (size_t) blob_;
            }
        }
    }
    if (!off_.empty()) return base_ + off_[(size_t) slot];
    return base_ + (size_t) slot * (size_t) blob_;
}

int ExpertCache::slot_device(int32_t slot) const {
    if (!segments_.empty()) {
        for (const auto& seg : segments_) {
            if (slot >= seg.start_slot && slot < seg.start_slot + seg.n_slots) {
                return seg.ordinal;
            }
        }
    }
    return 0;
}

bool ExpertCache::fill_slot(int32_t slot, const uint8_t* host_blob, void* stream, std::string& err, int64_t bytes) {
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "ExpertCache::fill_slot: slot " + std::to_string(slot) + " is outside 0.." +
              std::to_string(slots_ - 1);
        return false;
    }
    if (host_blob == nullptr) {
        err = "ExpertCache::fill_slot: the host blob is null";
        return false;
    }
    int orig_device = 0;
    cudaGetDevice(&orig_device);
    int target_device = slot_device(slot);
    if (target_device != orig_device) cudaSetDevice(target_device);

    const cudaError_t e = cudaMemcpyAsync(dst, host_blob, n, cudaMemcpyDefault,
                                          (cudaStream_t) stream);
    if (target_device != orig_device) cudaSetDevice(orig_device);

    if (e != cudaSuccess) {
        err = std::string("ExpertCache::fill_slot: ") + cudaGetErrorString(e);
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::fill_slot_blocking(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "ExpertCache::fill_slot_blocking: slot outside the arena";
        return false;
    }
    if (host_blob == nullptr) {
        err = "ExpertCache::fill_slot_blocking: the host blob is null";
        return false;
    }
    int orig_device = 0;
    cudaGetDevice(&orig_device);
    int target_device = slot_device(slot);
    if (target_device != orig_device) cudaSetDevice(target_device);

    const cudaError_t e = cudaMemcpy(dst, host_blob, n, cudaMemcpyDefault);
    if (target_device != orig_device) cudaSetDevice(orig_device);

    if (e != cudaSuccess) {
        err = std::string("ExpertCache::fill_slot_blocking: ") + cudaGetErrorString(e);
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::verify_slot(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    const int64_t nb = bytes > 0 && bytes <= blob_ ? bytes : blob_;
    const uint8_t* src = device_slot(slot);
    if (src == nullptr) {
        err = "ExpertCache::verify_slot: slot outside the arena";
        return false;
    }
    std::vector<uint8_t> got((size_t) nb);
    int orig_device = 0;
    cudaGetDevice(&orig_device);
    int target_device = slot_device(slot);
    if (target_device != orig_device) cudaSetDevice(target_device);

    const cudaError_t e = cudaMemcpy(got.data(), src, (size_t) nb, cudaMemcpyDefault);
    if (target_device != orig_device) cudaSetDevice(orig_device);

    if (e != cudaSuccess) {
        err = std::string("ExpertCache::verify_slot: ") + cudaGetErrorString(e);
        return false;
    }
    if (std::memcmp(got.data(), host_blob, (size_t) nb) != 0) {
        size_t first = 0;
        while (first < (size_t) nb && got[first] == host_blob[first]) ++first;
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "ExpertCache::verify_slot: slot %d differs from the arena at byte %llu (of %lld)",
                      (int) slot, (unsigned long long) first, (long long) blob_);
        err = buf;
        return false;
    }
    return true;
}

}  // namespace strata::core
