import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

def naive_softmax(x):
    x_max = x.max(dim=1)[0]

    z = x - x_max[:, None]

    

    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)

    out = numerator/denominator[:, None]

    return out

def test_softmax_kernel(size:tuple, atol=1e-3, rtol=1e-3):
    torch.manual_seed(0)
    assert len(size) == 2
    x = torch.randn(size[0], size[1], device=DEVICE)

    naive_out = naive_softmax(x)

    z_tri = softmax(x)
    z_torch = torch.softmax(x, dim=1)

    triton.testing.assert_close(naive_out, torch.softmax(x, dim=1))
    print("passed")


if __name__ == "__main__":
    test_softmax_kernel((100, 100))