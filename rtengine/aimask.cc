#include "aimask.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <glib/gstdio.h>

namespace rtengine
{

namespace
{

const char MAGIC[8] = { 'R', 'T', 'A', 'I', 'M', 'S', 'K', '1' };

struct CacheEntry {
    gint64 mtime;
    gint64 size;
    std::shared_ptr<const AiMask> mask;
};

std::mutex cacheMutex;
std::map<std::string, CacheEntry> cache;

void put32(unsigned char* p, unsigned v)
{
    p[0] = v & 0xff;
    p[1] = (v >> 8) & 0xff;
    p[2] = (v >> 16) & 0xff;
    p[3] = (v >> 24) & 0xff;
}

unsigned get32(const unsigned char* p)
{
    return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<unsigned>(p[3]) << 24);
}

std::shared_ptr<const AiMask> readFile(const Glib::ustring& path)
{
    FILE* f = g_fopen(path.c_str(), "rb");
    if (!f) {
        return {};
    }
    std::shared_ptr<AiMask> m;
    unsigned char head[16];
    if (fread(head, 1, 16, f) == 16 && !memcmp(head, MAGIC, 8)) {
        const unsigned w = get32(head + 8);
        const unsigned h = get32(head + 12);
        if (w >= 2 && h >= 2 && w <= 16384 && h <= 16384) {
            m = std::make_shared<AiMask>();
            m->w = w;
            m->h = h;
            m->data.resize(static_cast<size_t>(w) * h);
            // run-length encoded: (count 1..255, value) pairs
            size_t pos = 0;
            const size_t total = m->data.size();
            unsigned char buf[8192];
            size_t n;
            bool ok = true;
            int pendingCount = -1;
            while (ok && (n = fread(buf, 1, sizeof(buf), f)) > 0) {
                for (size_t i = 0; i < n; ++i) {
                    if (pendingCount < 0) {
                        pendingCount = buf[i];
                    } else {
                        if (pendingCount == 0 || pos + pendingCount > total) {
                            ok = false;
                            break;
                        }
                        memset(m->data.data() + pos, buf[i], pendingCount);
                        pos += pendingCount;
                        pendingCount = -1;
                    }
                }
            }
            if (!ok || pos != total) {
                m.reset();
            }
        }
    }
    fclose(f);
    return m;
}

}

std::shared_ptr<const AiMask> loadAiMask(const Glib::ustring& path)
{
    if (path.empty()) {
        return {};
    }
    GStatBuf st;
    if (g_stat(path.c_str(), &st) != 0) {
        return {};
    }
    const std::string key = path.raw();
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        const auto it = cache.find(key);
        if (it != cache.end() && it->second.mtime == static_cast<gint64>(st.st_mtime) && it->second.size == static_cast<gint64>(st.st_size)) {
            return it->second.mask;
        }
    }
    std::shared_ptr<const AiMask> m = readFile(path);
    if (m) {
        std::lock_guard<std::mutex> lock(cacheMutex);
        if (cache.size() > 12) {
            cache.clear();   // masks in use stay alive through their shared_ptr
        }
        cache[key] = CacheEntry{ static_cast<gint64>(st.st_mtime), static_cast<gint64>(st.st_size), m };
    }
    return m;
}

bool saveAiMask(const Glib::ustring& path, int w, int h, const unsigned char* data)
{
    if (w < 2 || h < 2 || !data) {
        return false;
    }
    FILE* f = g_fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    unsigned char head[16];
    memcpy(head, MAGIC, 8);
    put32(head + 8, w);
    put32(head + 12, h);
    bool ok = fwrite(head, 1, 16, f) == 16;

    std::vector<unsigned char> out;
    out.reserve(1 << 16);
    const size_t total = static_cast<size_t>(w) * h;
    size_t i = 0;
    while (ok && i < total) {
        const unsigned char v = data[i];
        size_t run = 1;
        while (run < 255 && i + run < total && data[i + run] == v) {
            ++run;
        }
        out.push_back(static_cast<unsigned char>(run));
        out.push_back(v);
        i += run;
        if (out.size() >= (1 << 16) - 2) {
            ok = fwrite(out.data(), 1, out.size(), f) == out.size();
            out.clear();
        }
    }
    if (ok && !out.empty()) {
        ok = fwrite(out.data(), 1, out.size(), f) == out.size();
    }
    if (fclose(f) != 0) {
        ok = false;
    }
    if (!ok) {
        g_remove(path.c_str());
    }
    return ok;
}

}
