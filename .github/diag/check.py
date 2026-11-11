import os, sys, pefile
app = sys.argv[1]
sysdirs = [r"C:\Windows\System32", r"C:\Windows"]
local = {f.lower(): os.path.join(app, f) for f in os.listdir(app) if f.lower().endswith(('.dll', '.exe'))}
exports = {}
def exp(path):
    if path not in exports:
        pe = pefile.PE(path, fast_load=True)
        pe.parse_data_directories([pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']])
        exports[path] = {e.name for e in getattr(pe, 'DIRECTORY_ENTRY_EXPORT', None).symbols if e.name} if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT') else set()
    return exports[path]
def resolve(name):
    n = name.lower()
    if n in local: return local[n]
    for d in sysdirs:
        p = os.path.join(d, name)
        if os.path.exists(p): return p
    return None
seen, todo, problems = set(), [os.path.join(app, x) for x in sys.argv[2:]], 0
while todo:
    path = todo.pop()
    if path.lower() in seen: continue
    seen.add(path.lower())
    pe = pefile.PE(path, fast_load=True)
    pe.parse_data_directories([pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']])
    for imp in getattr(pe, 'DIRECTORY_ENTRY_IMPORT', []):
        dll = imp.dll.decode()
        if dll.lower().startswith(('api-ms-', 'ext-ms-')): continue
        r = resolve(dll)
        if not r:
            print("MISSING DLL:", os.path.basename(path), "needs", dll); problems += 1; continue
        if not r.lower().startswith(app.lower()):
            if not dll.lower() in ('kernel32.dll','user32.dll','gdi32.dll','advapi32.dll','shell32.dll','ole32.dll','msvcrt.dll','ws2_32.dll','winmm.dll'):
                print("from Windows:", os.path.basename(path), "->", r)
            continue
        ex = exp(r)
        miss = [s.name.decode() for s in imp.imports if s.name and s.name not in ex]
        if miss:
            print("MISSING SYMBOLS:", os.path.basename(path), "imports from", dll, ":", miss[:8], len(miss)); problems += 1
        todo.append(r)
print("checked", len(seen), "modules, problems:", problems)
