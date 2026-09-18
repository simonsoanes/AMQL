// repro_lt.cu — first inference GEMM in isolation: (3,5120,10240) FP16.
#include <cuda_runtime.h>
#include <windows.h>
#include <cstdio>
#include <cstdint>

typedef int (*fn_available)(void);
typedef int (*fn_init)(void);
typedef int (*fn_gemm)(const float*, void*, float*, int, int, int, int);

int main()
{
    HMODULE h = LoadLibraryA("D:\\Dev\\AMQL\\native\\amql_cuda\\bin\\amql_cuda.dll");
    if (!h) { printf("LoadLibrary FAILED err=%u\n", (unsigned)GetLastError()); return 1; }
    auto available = (fn_available)GetProcAddress(h, "amql_cuda_available");
    auto init = (fn_init)GetProcAddress(h, "amql_cuda_init");
    auto gemm = (fn_gemm)GetProcAddress(h, "amql_cuda_gemm_transposed_b");
    printf("available=%p init=%p gemm=%p\n", (void*)available, (void*)init, (void*)gemm);
    if (!available || !init || !gemm) return 2;
    printf("available -> %d, init -> %d\n", available(), init());

    const int m = 3, k = 5120, n = 10240;
    auto* a = new float[(size_t)m * k];
    auto* c = new float[(size_t)m * n];
    for (int i = 0; i < m * k; i++) a[i] = 0.5f;
    void* wD = nullptr;
    cudaMalloc(&wD, (size_t)n * k * 2);
    cudaMemset(wD, 0, (size_t)n * k * 2);

    int code = gemm(a, wD, c, m, k, n, 0);
    cudaError_t le = cudaGetLastError();
    printf("gemm(3,5120,10240,FP16) ret=%d lastErr=%d\n", code, (int)le);

    // second shape (decode step m=1)
    int code2 = gemm(a, wD, c, 1, k, n, 0);
    cudaError_t le2 = cudaGetLastError();
    printf("gemm(1,5120,10240,FP16) ret=%d lastErr=%d\n", code2, (int)le2);

    // a smaller known-good-ish shape from the tests (m=8,k=1024,n=1024)
    int code3 = gemm(a, wD, c, 8, 1024, 1024, 0);
    cudaError_t le3 = cudaGetLastError();
    printf("gemm(8,1024,1024,FP16) ret=%d lastErr=%d\n", code3, (int)le3);

    cudaFree(wD);
    delete[] a; delete[] c;
    return 0;
}