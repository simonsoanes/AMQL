// repro_f32.cu — probe amql_cuda_gemm_transposed_b_f32 exactly as
// MergeGpu does: upload the map with amql_cuda_host_to_device_f32 (a sync
// NULL-stream copy), then run the blocked Sgemm loop. Checks c = b @ map^T
// against a host computation.
#include <cuda_runtime.h>
#include <windows.h>
#include <cstdio>
#include <cstdint>

typedef int (*fn_available)(void);
typedef int (*fn_init)(void);
typedef int (*fn_h2d_f32)(void*, const float*, size_t);
typedef int (*fn_gemm)(const float*, void*, float*, int, int, int, int, int);

int main()
{
    HMODULE h = LoadLibraryA("D:\\Dev\\AMQL\\native\\amql_cuda\\bin\\amql_cuda.dll");
    if (!h) { printf("LoadLibrary FAILED err=%u\n", (unsigned)GetLastError()); return 1; }
    auto available = (fn_available)GetProcAddress(h, "amql_cuda_available");
    auto init = (fn_init)GetProcAddress(h, "amql_cuda_init");
    auto h2d = (fn_h2d_f32)GetProcAddress(h, "amql_cuda_host_to_device_f32");
    auto gemm = (fn_gemm)GetProcAddress(h, "amql_cuda_gemm_transposed_b_f32");
    printf("available=%p init=%p h2d_f32=%p gemm_f32=%p\n", (void*)available, (void*)init, (void*)h2d, (void*)gemm);
    if (!available || !init || !h2d || !gemm) return 2;
    printf("available -> %d, init -> %d\n", available(), init());

    const int n = 2048, d = 256;
    auto* b = new float[(size_t)n * d];
    auto* m = new float[(size_t)d * d];
    auto* hostC = new float[(size_t)n * d];
    for (int i = 0; i < n * d; i++) b[i] = ((float)(i % 100)) / 100.f;
    for (int i = 0; i < d * d; i++) m[i] = ((float)(i % 61)) / 61.f;
    // host reference: c[i][j] = sum_k b[i][k] * m[j][k]
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
        {
            double s = 0;
            for (int k = 0; k < d; k++) s += (double)b[i * d + k] * m[j * d + k];
            hostC[i * d + j] = (float)s;
        }

    void* mDev = nullptr;
    // mirror MergeGpu.UploadMap: AllocDevice + CopyHostToDevice(sync)
    typedef int (*fn_malloc)(void**, size_t);
    auto mall = (fn_malloc)GetProcAddress(h, "amql_cuda_malloc");
    if (!mall || mall(&mDev, (size_t)d * d * 4) != 0) { printf("alloc map FAILED\n"); return 3; }
    int up = h2d(mDev, m, (size_t)d * d * 4);
    printf("map upload code=%d\n", up);

    auto* c = new float[(size_t)n * d];
    int code = gemm(b, mDev, c, n, d, d, 8192, 0);
    printf("gemm_f32 code=%d\n", code);
    double maxDiff = 0; int worst = -1;
    for (int i = 0; i < n * d; i++)
    {
        double diff = (double)c[i] - hostC[i];
        if (diff < 0) diff = -diff;
        if (diff > maxDiff) { maxDiff = diff; worst = i; }
    }
    printf("max |c - b@M^T| = %g at index %d (c=%g host=%g)  first32: ", maxDiff, worst,
        worst >= 0 ? c[worst] : 0.0, worst >= 0 ? hostC[worst] : 0.0);
    for (int i = 0; i < 4; i++) printf("%g ", c[i]);
    printf("\n");

    typedef int (*fn_free)(void*);
    auto fre = (fn_free)GetProcAddress(h, "amql_cuda_free");
    fre(mDev);
    delete[] b; delete[] m; delete[] c; delete[] hostC;
    return 0;
}