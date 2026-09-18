// repro2.cu — raw-cublas isolation mirroring the DLL context EXACTLY:
// Lt handle + workspace, one cublas handle on a non-blocking stream,
// 4096 real rows zero-padded to an 8192-row slab, two back-to-back
// cublasDgemm per "invocation", two invocations total.
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cstdio>

static void run_pair(cublasHandle_t h, cudaStream_t s,
    double* bD, double* aD, double* gramD, double* crossD,
    const double* bT, const double* aT, int rows, int d, int blockRows, int tag)
{
    // zero tail on device, upload full slab (mirror of gram_cross)
    size_t tailBytes = (size_t)(blockRows - rows) * d * 8;
    cudaMemsetAsync((char*)bD + (size_t)rows * d * 8, 0, tailBytes, s);
    cudaMemsetAsync((char*)aD + (size_t)rows * d * 8, 0, tailBytes, s);
    cudaMemcpyAsync(bD, bT, (size_t)blockRows * d * 8, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(aD, aT, (size_t)blockRows * d * 8, cudaMemcpyHostToDevice, s);

    double alpha = 1.0, beta = 0.0;
    cublasStatus_t st1 = cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, d, d, blockRows,
        &alpha, bD, d, bD, d, &beta, gramD, d);
    cublasStatus_t st2 = cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, d, d, blockRows,
        &alpha, bD, d, aD, d, &beta, crossD, d);
    cudaStreamSynchronize(s);
    double g[2] = {0, 0}, c[2] = {0, 0};
    cudaMemcpy(g, gramD, sizeof(g), cudaMemcpyDeviceToHost);
    cudaMemcpy(c, crossD, sizeof(c), cudaMemcpyDeviceToHost);
    printf("inv%d status1=%d status2=%d gram[0..1]=%g %g cross[0..1]=%g %g\n",
        tag, (int)st1, (int)st2, g[0], g[1], c[0], c[1]);
}

int main()
{
    cublasLtHandle_t lt = nullptr;
    cublasHandle_t h = nullptr;
    void* ws = nullptr;
    if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) { printf("Lt create FAILED\n"); return 1; }
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { printf("cublasCreate FAILED\n"); return 1; }
    cudaMalloc(&ws, 64u << 20);
    cudaStream_t s = nullptr;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) { printf("stream FAILED\n"); return 1; }
    cublasSetStream(h, s);

    const int d = 512, rows = 4096, blockRows = 8192;
    double* bD = nullptr; double* aD = nullptr;
    double* gramD = nullptr; double* crossD = nullptr;
    cudaMalloc(&bD, (size_t)blockRows * d * 8);
    cudaMalloc(&aD, (size_t)blockRows * d * 8);
    cudaMalloc(&gramD, (size_t)d * d * 8);
    cudaMalloc(&crossD, (size_t)d * d * 8);

    auto* bT = new double[(size_t)blockRows * d];
    auto* aT = new double[(size_t)blockRows * d];
    for (int i = 0; i < rows * d; i++)
    {
        bT[i] = ((double)(int)(i % 100)) / 100.0;
        aT[i] = ((double)(int)(i % 77)) / 77.0;
    }
    for (size_t i = (size_t)rows * d; i < (size_t)blockRows * d; i++) { bT[i] = 0.0; aT[i] = 0.0; }

    run_pair(h, s, bD, aD, gramD, crossD, bT, aT, rows, d, blockRows, 1);
    run_pair(h, s, bD, aD, gramD, crossD, bT, aT, rows, d, blockRows, 2);

    printf("expected gram[0] = %g (4096 rows of ((i%%100)/100)^2 at stride 512)\n", 1283.728);

    cudaFree(bD); cudaFree(aD); cudaFree(gramD); cudaFree(crossD);
    cudaFree(ws);
    delete[] bT; delete[] aT;
    cublasDestroy(h);
    cublasLtDestroy(lt);
    cudaStreamDestroy(s);
    return 0;
}