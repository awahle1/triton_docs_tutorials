import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
TOTAL_SRAM_PER_SM = properties["max_shared_mem"]
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
WARP_SIZE = properties["warpSize"]

@triton.jit
def _softmax_kernel(
    input_ptr, output_ptr, 
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
    ):
    PID = tl.program_id(0)

    row_step = tl.num_programs(0)

    for row_idx in tl.range(PID, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx*input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets

        mask = col_offsets < n_cols

        row = tl.load(input_ptrs, mask=mask, other=float('-inf'))

        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator)
        row_softmax = numerator/denominator

        output_row_start_ptr = output_ptr + row_idx*output_row_stride
        tl.store(output_row_start_ptr + col_offsets, row_softmax, mask=mask)


def softmax(x):
    assert x.ndim == 2
    assert x.is_contiguous

    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    num_stages = 4 if TOTAL_SRAM_PER_SM >= 200_000 else 2

    y = torch.empty_like(x)

    kernel = _softmax_kernel.warmup(
        x, y,
        x.stride(0), y.stride(0), 
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,)
    )

    kernel._init_handles()
    n_regs_per_program = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared 

    reg_occupancy = NUM_REGS//(WARP_SIZE*n_regs_per_program*num_warps)

    sram_occupancy = TOTAL_SRAM_PER_SM//sram_needed_per_program

    programs_per_sm = min(reg_occupancy, sram_occupancy)

    num_programs = min(programs_per_sm*NUM_SM, n_rows)

    grid = (num_programs, 1, 1)

    kernel[grid](
        x, y, 
        x.stride(0), y.stride(0),
        n_rows, n_cols
    )

    return y

def test_softmax_kernel(size:tuple, atol=1e-3, rtol=1e-3):
    torch.manual_seed(0)
    assert len(size) == 2
    x = torch.randn(size[0], size[1], device=DEVICE)

    z_tri = softmax(x)
    z_torch = torch.softmax(x, dim=1)

    triton.testing.assert_close(z_tri, z_torch)
    print("passed")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names = ['N'],
        x_vals = [128*i for i in range(2, 100)],
        line_arg = 'provider',
        line_vals = ['torch', 'triton'],
        line_names = ['Torch', 'Triton'],
        styles=[("blue", "-"), ("green", "-")],
        ylabel='GB/s',
        plot_name="softmax-performance",
        args={'M': 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M,N, device=DEVICE, dtype=torch.float32) 

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms*1e-3)

    return gbps(ms)


if __name__ == "__main__":
    test_softmax_kernel((1823, 781))

    import sys 
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run( save_path='.', print_data=True)