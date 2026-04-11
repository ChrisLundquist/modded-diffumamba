// Stub out rocprofiler-register to prevent WSL2 crash
int rocprofiler_set_api_table(void* a, void* b, unsigned long c, void* d) { return 0; }
int rocprofiler_register_library_api_table(void* a, void* b, unsigned long c, void* d) { return 0; }
