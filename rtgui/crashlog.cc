/*
 *  This file is part of RawTherapee.
 *
 *  RawTherapee is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  RawTherapee is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with RawTherapee.  If not, see <https://www.gnu.org/licenses/>.
 */
#include "crashlog.h"

#ifdef _WIN32

#define PSAPI_VERSION 2
#include <windows.h>
#include <psapi.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <exception>
#include <typeinfo>

#include <process.h>

#include <cxxabi.h>
#include <glib.h>
#include <glibmm/exception.h>

namespace
{

wchar_t logPath[1024] = L"";
char buildInfo[256] = "";
char lastMessages[4][400];
unsigned int messageCount = 0;
volatile LONG reported = 0;
volatile LONG normalExit = 0;
DWORD mainThreadId = 0;
LPTOP_LEVEL_EXCEPTION_FILTER previousFilter = nullptr;
EXCEPTION_POINTERS* pendingException = nullptr;

HANDLE openLog()
{
    return CreateFileW(logPath, FILE_APPEND_DATA, FILE_SHARE_READ, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
}

void put(HANDLE h, const char* text)
{
    DWORD written = 0;
    WriteFile(h, text, static_cast<DWORD>(std::strlen(text)), &written, nullptr);
}

void putAddress(HANDLE h, const void* address)
{
    char line[600];
    char path[MAX_PATH] = "";
    HMODULE module = nullptr;

    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(address), &module) && module
            && GetModuleFileNameA(module, path, sizeof(path))) {
        const char* name = std::strrchr(path, '\\');
        name = name ? name + 1 : path;
        std::snprintf(line, sizeof(line), "  %s + 0x%llx\r\n", name,
                      static_cast<unsigned long long>(static_cast<const char*>(address) - reinterpret_cast<const char*>(module)));
    } else {
        std::snprintf(line, sizeof(line), "  (no module) %p\r\n", address);
    }

    put(h, line);
}

void putHeader(HANDLE h, const char* what)
{
    char line[700];
    SYSTEMTIME t;
    GetLocalTime(&t);
    std::snprintf(line, sizeof(line), "\r\n==== CRASH %04d-%02d-%02d %02d:%02d:%02d ====\r\n%s\r\nBuild: %s\r\nThread: %lu%s\r\n",
                  t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond, what, buildInfo,
                  static_cast<unsigned long>(GetCurrentThreadId()),
                  GetCurrentThreadId() == mainThreadId ? " (main)" : " (worker)");
    put(h, line);

    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);

    if (GlobalMemoryStatusEx(&mem)) {
        std::snprintf(line, sizeof(line),
                      "System memory: %lu%% used, RAM free %llu of %llu MB, commit free %llu of %llu MB\r\n",
                      static_cast<unsigned long>(mem.dwMemoryLoad),
                      static_cast<unsigned long long>(mem.ullAvailPhys >> 20),
                      static_cast<unsigned long long>(mem.ullTotalPhys >> 20),
                      static_cast<unsigned long long>(mem.ullAvailPageFile >> 20),
                      static_cast<unsigned long long>(mem.ullTotalPageFile >> 20));
        put(h, line);
    }

    PROCESS_MEMORY_COUNTERS counters;

    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        std::snprintf(line, sizeof(line), "RawTherapee memory: %llu MB in RAM (peak %llu), %llu MB committed (peak %llu)\r\n",
                      static_cast<unsigned long long>(counters.WorkingSetSize >> 20),
                      static_cast<unsigned long long>(counters.PeakWorkingSetSize >> 20),
                      static_cast<unsigned long long>(counters.PagefileUsage >> 20),
                      static_cast<unsigned long long>(counters.PeakPagefileUsage >> 20));
        put(h, line);
    }

    const unsigned int shown = messageCount < 4 ? messageCount : 4;

    if (shown) {
        put(h, "Last GLib/GTK messages:\r\n");

        for (unsigned int i = messageCount - shown; i < messageCount; ++i) {
            put(h, "  ");
            put(h, lastMessages[i % 4]);
            put(h, "\r\n");
        }
    }
}

void putStack(HANDLE h, CONTEXT context)
{
#if defined(_WIN64) && defined(__x86_64__)
    put(h, "Stack:\r\n");

    for (int frame = 0; frame < 60 && context.Rip; ++frame) {
        putAddress(h, reinterpret_cast<const void*>(context.Rip));

        DWORD64 imageBase = 0;
        PRUNTIME_FUNCTION function = RtlLookupFunctionEntry(context.Rip, &imageBase, nullptr);

        if (function) {
            PVOID handlerData = nullptr;
            DWORD64 establisherFrame = 0;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, context.Rip, function, &context, &handlerData, &establisherFrame, nullptr);
        } else {
            // Leaf function: the return address is on top of the stack.
            MEMORY_BASIC_INFORMATION info;

            if (!VirtualQuery(reinterpret_cast<LPCVOID>(context.Rsp), &info, sizeof(info)) || info.State != MEM_COMMIT) {
                break;
            }

            context.Rip = *reinterpret_cast<const DWORD64*>(context.Rsp);
            context.Rsp += 8;
        }
    }

#else
    (void)context;
    put(h, "Stack: not available on this architecture\r\n");
#endif
}

void putModules(HANDLE h)
{
    static HMODULE modules[2048];
    DWORD bytes = 0;

    if (!EnumProcessModules(GetCurrentProcess(), modules, sizeof(modules), &bytes)) {
        return;
    }

    char own[MAX_PATH] = "";
    GetModuleFileNameA(nullptr, own, sizeof(own));

    if (char* slash = std::strrchr(own, '\\')) {
        *slash = '\0';
    }

    const std::size_t ownLength = std::strlen(own);
    const DWORD count = bytes / sizeof(HMODULE) < 2048 ? bytes / sizeof(HMODULE) : 2048;

    put(h, "Loaded modules not shipped with RawTherapee:\r\n");

    for (DWORD i = 0; i < count; ++i) {
        char path[MAX_PATH] = "";

        if (!GetModuleFileNameA(modules[i], path, sizeof(path)) || !_strnicmp(path, own, ownLength)) {
            continue;
        }

        put(h, "  ");
        put(h, path);
        put(h, "\r\n");
    }
}

DWORD WINAPI writeExceptionReport(LPVOID)
{
    const EXCEPTION_POINTERS* const ep = pendingException;
    const HANDLE h = openLog();

    if (h == INVALID_HANDLE_VALUE || !ep) {
        return 0;
    }

    const EXCEPTION_RECORD* const record = ep->ExceptionRecord;
    char what[300];
    const char* name = "";

    switch (record->ExceptionCode) {
        case EXCEPTION_ACCESS_VIOLATION:
            name = " (access violation)";
            break;

        case EXCEPTION_STACK_OVERFLOW:
            name = " (stack overflow)";
            break;

        case EXCEPTION_ILLEGAL_INSTRUCTION:
            name = " (illegal instruction)";
            break;

        case EXCEPTION_IN_PAGE_ERROR:
            name = " (in-page error)";
            break;

        case EXCEPTION_INT_DIVIDE_BY_ZERO:
            name = " (integer divide by zero)";
            break;

        case EXCEPTION_BREAKPOINT:
            name = " (breakpoint, usually a fatal GLib error)";
            break;

        case 0x20474343:
            name = " (uncaught C++ exception)";
            break;
    }

    if (record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && record->NumberParameters >= 2) {
        std::snprintf(what, sizeof(what), "Exception 0x%08lx%s: %s address 0x%llx", static_cast<unsigned long>(record->ExceptionCode), name,
                      record->ExceptionInformation[0] == 1 ? "writing" : record->ExceptionInformation[0] == 8 ? "executing" : "reading",
                      static_cast<unsigned long long>(record->ExceptionInformation[1]));
    } else {
        std::snprintf(what, sizeof(what), "Exception 0x%08lx%s", static_cast<unsigned long>(record->ExceptionCode), name);
    }

    putHeader(h, what);
    put(h, "Fault location:\r\n");
    putAddress(h, record->ExceptionAddress);
    putStack(h, *ep->ContextRecord);
    putModules(h);
    CloseHandle(h);
    return 0;
}

// The report is written from a fresh thread so that it also works when the
// crashing thread has run out of stack.
void reportException(EXCEPTION_POINTERS* ep)
{
    if (InterlockedExchange(&reported, 1)) {
        return;
    }

    pendingException = ep;
    const HANDLE thread = CreateThread(nullptr, 0, writeExceptionReport, nullptr, 0, nullptr);

    if (thread) {
        WaitForSingleObject(thread, 10000);
        CloseHandle(thread);
    }
}

LONG WINAPI unhandledFilter(EXCEPTION_POINTERS* ep)
{
    reportException(ep);
    return previousFilter ? previousFilter(ep) : EXCEPTION_CONTINUE_SEARCH;
}

LONG WINAPI vectoredHandler(EXCEPTION_POINTERS* ep)
{
    // First-chance handler: only for stack overflows, which may never reach
    // the unhandled exception filter.
    if (ep->ExceptionRecord->ExceptionCode == EXCEPTION_STACK_OVERFLOW) {
        reportException(ep);
    }

    return EXCEPTION_CONTINUE_SEARCH;
}

void reportHere(const char* what)
{
    if (InterlockedExchange(&reported, 1)) {
        return;
    }

    const HANDLE h = openLog();

    if (h == INVALID_HANDLE_VALUE) {
        return;
    }

    CONTEXT context;
    RtlCaptureContext(&context);
    putHeader(h, what);
    putStack(h, context);
    putModules(h);
    CloseHandle(h);
}

void onTerminate()
{
    char what[700] = "std::terminate called without an active exception";

    if (const std::exception_ptr current = std::current_exception()) {
        try {
            std::rethrow_exception(current);
        } catch (const std::exception& e) {
            int status = 0;
            char* const demangled = abi::__cxa_demangle(typeid(e).name(), nullptr, nullptr, &status);
            std::snprintf(what, sizeof(what), "Uncaught C++ exception %s: %s", demangled && !status ? demangled : typeid(e).name(), e.what());
            std::free(demangled);
        } catch (const Glib::Exception& e) {
            std::snprintf(what, sizeof(what), "Uncaught Glib exception: %s", e.what().c_str());
        } catch (...) {
            std::snprintf(what, sizeof(what), "Uncaught C++ exception of a type not derived from std::exception");
        }
    }

    reportHere(what);
    std::abort();
}

void onAbort(int)
{
    reportHere("abort() was called");
    _exit(3);
}

void onExit()
{
    if (!normalExit && !reported) {
        reportHere("exit() was called while RawTherapee was still running");
    }
}

void logHandler(const gchar* domain, GLogLevelFlags level, const gchar* message, gpointer data)
{
    if (level & (G_LOG_LEVEL_ERROR | G_LOG_LEVEL_CRITICAL | G_LOG_LEVEL_WARNING)) {
        std::snprintf(lastMessages[messageCount % 4], sizeof(lastMessages[0]), "%s%s: %s",
                      domain ? domain : "", level & G_LOG_LEVEL_ERROR ? " ERROR" : level & G_LOG_LEVEL_CRITICAL ? " CRITICAL" : " WARNING",
                      message ? message : "");
        ++messageCount;
    }

    g_log_default_handler(domain, level, message, data);
}

void putLine(const char* text)
{
    const HANDLE h = openLog();

    if (h != INVALID_HANDLE_VALUE) {
        char line[500];
        SYSTEMTIME t;
        GetLocalTime(&t);
        std::snprintf(line, sizeof(line), "%04d-%02d-%02d %02d:%02d:%02d  %s\r\n", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond, text);
        put(h, line);
        CloseHandle(h);
    }
}

}

void CrashLog::install(const char* version)
{
    const gchar* directory = g_get_user_special_dir(G_USER_DIRECTORY_DOWNLOAD);

    if (!directory) {
        directory = g_get_tmp_dir();
    }

    gchar* const path = g_build_filename(directory, "RawTherapee-crash-log.txt", nullptr);
    wchar_t* const wide = reinterpret_cast<wchar_t*>(g_utf8_to_utf16(path, -1, nullptr, nullptr, nullptr));

    if (wide) {
        wcsncpy(logPath, wide, sizeof(logPath) / sizeof(logPath[0]) - 1);
        g_free(wide);
    }

    g_free(path);

    std::snprintf(buildInfo, sizeof(buildInfo), "%s", version ? version : "");
    mainThreadId = GetCurrentThreadId();

    ULONG stackGuarantee = 64 * 1024;
    SetThreadStackGuarantee(&stackGuarantee);

    previousFilter = SetUnhandledExceptionFilter(unhandledFilter);
    AddVectoredExceptionHandler(1, vectoredHandler);
    std::set_terminate(onTerminate);
    std::signal(SIGABRT, onAbort);
    std::atexit(onExit);
    g_log_set_default_handler(logHandler, nullptr);

    char line[300];
    std::snprintf(line, sizeof(line), "started (%s)", buildInfo);
    putLine(line);
}

void CrashLog::markNormalExit()
{
    if (!InterlockedExchange(&normalExit, 1)) {
        putLine("closed normally");
    }
}

#else

void CrashLog::install(const char*)
{
}

void CrashLog::markNormalExit()
{
}

#endif
