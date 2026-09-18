// repro_gram.cu — direct DLL repro for amql_cuda_gram_cross.
// Loads the built amql_cuda.dll, calls init + gram_cross on a
// deterministic 4096x512 slab, prints the native return code and a few
// output entries (gram = B^T B, cross = B^T A). Also confirms whether
// amql_cuda_host_to_device_f32 is exported (NULL => missing).
#include <cuda_runtime.h>
#include <windows.h>
#include <cstdio>
#include <cstdint>

typedef int (*fn_available)(void);
typedef int (*fn_init)(void);
typedef int (*fn_cross)(const float*, const float*, double*, double*, int, int, int, int);

int main()
{
    HMODULE h = LoadLibraryA("D:\\Dev\\AMQL\\native\\amql_cuda\\bin\\amql_cuda.dll");
    if (!h) { printf("LoadLibrary FAILED err=%u\n", (unsigned)GetLastError()); return 1; }
    auto available = (fn_available)GetProcAddress(h, "amql_cuda_available");
    auto init = (fn_init)GetProcAddress(h, "amql_cuda_init");
    auto cross = (fn_cross)GetProcAddress(h, "amql_cuda_gram_cross");
    void* f32exp = (void*)GetProcAddress(h, "amql_cuda_host_to_device_f32");
    printf("available=%p init=%p gram_cross=%p host_to_device_f32=%s\n",
        (void*)available, (void*)init, (void*)cross,
        f32exp ? "EXPORTED" : "MISSING");
    if (!available || !init || !cross) return 2;

    printf("amql_cuda_available -> %d\n", available());
    printf("amql_cuda_init -> %d\n", init());

    const int n = 4096, d = 512, blockRows = 8192;
    auto* b = new float[(size_t)n * d];
    auto* a = new float[(size_t)n * d];
    for (size_t i = 0; i < (size_t)n * d; i++)
    {
        b[i] = (float)((int)(i % 100)) / 100.f;
        a[i] = (float)((int)(i % 77)) / 77.f;
    }
    auto* gram = new double[(size_t)d * d];
    auto* cors = new double[(size_t)d * d];

    int code = cross(a, b, gram, cors, n, d, blockRows, 0);
    printf("gram_cross code = %d\n", code);
    printf("gram[0]=%.6f gram[1]=%.6f gram[d+1]=%.6f\n", gram[0], gram[1], gram[d + 1]);
    printf("cross[0]=%.6f cross[1]=%.6f\n", cors[0], cors[1]);
    // expected gram[0] = sum_{i=0}^{4095} ((i%100)/100)^2 = 40*32.835 + 29.032 = 1342.432
    double expected = 0;
    for (int i = 0; i < n; i++) { double v = (double)((int)(((size_t)i * d) % 100)) / 100.0; expected += v * v; }
    printf("expected gram[0] = %.6f\n", expected);
    // transpose-sensitive: cross[r,d] = sum_i b[i*d+r] * a[i*d+c]
    double expectedC01 = 0, expectedC10 = 0;
    for (int i = 0; i < n; i++)
    {
        double br = (double)((int)(((size_t)i * d + 0) % 100)) / 100.0;
        double ac = (double)((int)(((size_t)i * d + 1) % 77)) / 77.0;
        double ar = (double)((int)(((size_t)i * d + 0) % 77)) / 77.0;
        double bc = (double)((int)(((size_t)i * d + 1) % 100)) / 100.0;
        expectedC01 += br * ac;
        expectedC10 += bc * ar;
    }
    printf("expected cross[0*d+1] (B^T A)[0,1] = %.6f   cross[1*d+0] (B^T A)[1,0] = %.6f\n", expectedC01, expectedC10);

    // second call (reuse path) to check idempotence
    int code2 = cross(a, b, gram, cors, n, d, blockRows, 0);
    printf("gram_cross code2 = %d gram[0]=%.6f\n", code2, gram[0]);

    delete[] a; delete[] b; delete[] gram; delete[] cors;
    return 0;
}