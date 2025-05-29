import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    PID = tl.program_id(axis=0)

    block_start = PID*BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=None)
    y = tl.load(y_ptr + offsets, mask=mask, other=None)

    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)

    return

def add(x, y):
    output = torch.empty_like(x)

    assert x.device == DEVICE and y.device == DEVICE

    n_elements = torch.numel(output)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    
    add_kernel[grid](
        x, 
        y, 
        output, 
        n_elements,
        BLOCK_SIZE=1024
    )

    return output

def test_add_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE)
    y = torch.randn(size, device=DEVICE)

    z_tri = add(x,y)
    z_ref = x+y

    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("passed")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals = [2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['torch', 'triton', 'third'],
        line_names=['Torch', 'Triton', 'Third'],
        styles=[("blue", "-"), ("green", "-"), ("red", "-.")],
        ylabel='GB/s',
        plot_name="vector-add-performance",
        args={}
    )
)
def benchmark(size, provider):
    x = torch.randn(size, device=DEVICE, dtype=torch.float32)
    y = torch.randn(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.05, 0.95]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x+y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    if provider == 'third':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y+x, quantiles=quantiles)

    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms*1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    test_add_kernel(4096)
    test_add_kernel(4097)
    test_add_kernel(98432)

    import sys 
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run( save_path='.', print_data=True)