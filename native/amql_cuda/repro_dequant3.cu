// repro_dequant3.cu — pin the faulting call: sync + lastError after each op.
#include <cuda_runtime.h>
#include <windows.h>
#include <cstdio>
#include <cstdint>

typedef int (*fn_available)(void);
typedef int (*fn_init)(void);
typedef int (*fn_dequant)(const unsigned char*, const unsigned char*, void*, int, int, int);
typedef int (*fn_malloc)(void**, size_t);

static void probe_fn(fn_dequant dequant, fn_malloc mall, int rows, int cols, int tag)
{
    long total = (long)rows * cols;
    auto* packed = new unsigned char[(size_t)(total / 2)];
    auto* scales = new unsigned char[(size_t)(rows * ((cols + 31) / 32))];
    for (size_t i = 0; i < (size_t)(total / 2); i++) packed[i] = (unsigned char)(i & 0xFF);
    for (size_t i = 0; i < (size_t)(rows * ((cols + 31) / 32)); i++) scales[i] = (unsigned char)(127 + (i % 32));
    void* outD = nullptr;
    printf("[t%d] malloc... ", tag);
    fflush(stdout);
    int m = mall(&outD, (size_t)total * 2);
    cudaError_t me = cudaGetLastError();
    printf("ret=%d last=%d; ", m, (int)me);
    fflush(stdout);
    // streamOrdinal=1 -> the kernel launches on the DEFAULT stream, so
    // cudaStreamSynchronize(0) below genuinely waits for it.
    printf("dequant1... ");
    fflush(stdout);
    int c1 = dequant(packed, scales, outD, rows, cols, 1);
    cudaError_t s1 = cudaStreamSynchronize(0);
    cudaError_t le1 = cudaGetLastError();
    printf("ret=%d syncStream=%d last=%d; ", c1, (int)s1, (int)le1);
    fflush(stdout);
    printf("dequant2... ");
    fflush(stdout);
    int c2 = dequant(packed, scales, outD, rows, cols, 1);
    cudaError_t s2 = cudaStreamSynchronize(0);
    cudaError_t le2 = cudaGetLastError();
    printf("ret=%d syncStream=%d last=%d\n", c2, (int)s2, (int)le2);
    fflush(stdout);
    cudaFree(outD);
    delete[] packed; delete[] scales;
}

int main()
{
    HMODULE h = LoadLibraryA("D:\\Dev\\AMQL\\native\\amql_cuda\\bin\\amql_cuda.dll");
    if (!h) { printf("LoadLibrary FAILED err=%u\n", (unsigned)GetLastError()); return 1; }
    auto available = (fn_available)GetProcAddress(h, "amql_cuda_available");
    auto init = (fn_init)GetProcAddress(h, "amql_cuda_init");
    auto dequant = (fn_dequant)GetProcAddress(h, "amql_cuda_dequant_to_f16");
    auto mall = (fn_malloc)GetProcAddress(h, "amql_cuda_malloc");
    printf("available -> %d, init -> %d\n", available(), init());

    probe_fn(dequant, mall, 12288, 5120, 1);   // know-good smaller shape
    probe_fn(dequant, mall, 17408, 5120, 2);   // the failing shape
    probe_fn(dequant, mall, 1024, 5120, 3);    // tiny after
    return 0;
}