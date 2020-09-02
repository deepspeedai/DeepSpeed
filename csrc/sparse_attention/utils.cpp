// DeepSpeed note, code taken & adapted from commit 9aa94789f13ada713af36cfd8cca2fc9a7f6b79a
// https://github.com/ptillet/torch-blocksparse/blob/master/csrc/utils.cpp

#include <torch/extension.h>
#include <string>
#include <tuple>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef std::vector<std::tuple<int, torch::Tensor>> ret_t;

void segment_blocks(torch::Tensor layout,
                    torch::Tensor idx,
                    torch::Tensor scratch,
                    int max_width,
                    ret_t& ret)
{
    size_t H = layout.size(0);
    size_t M = layout.size(1);
    size_t N = layout.size(2);
    torch::Tensor tmp = torch::zeros_like(layout);

    auto _tmp = tmp.accessor<int, 3>();
    auto _layout = layout.accessor<int, 3>();
    auto _idx = idx.accessor<int, 3>();
    auto _scratch = scratch.accessor<int, 3>();
    std::vector<int> current(H, 0);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t h = 0; h < H; h++) {
        // surrounding indices
        std::vector<int> ii_left(max_width, -1);
        std::vector<std::vector<int>> ii_top(max_width, std::vector<int>(N, -1));

        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                int v = _layout[h][m][n];
                if (v == 0) continue;
                int n_left = ii_left[max_width - 1];
                int m_top = ii_top[max_width - 1][n];
                int top = (m_top >= 0) ? _tmp[h][m_top][n] : 0;
                int left = (n_left >= 0) ? _tmp[h][m][n_left] : 0;
                int topleft = (m_top >= 0 && n_left >= 0) ? _tmp[h][m_top][n_left] : 0;
                int width = std::min(left, std::min(top, topleft)) + 1;

                // reset width if blocks cannot be
                // packed together (i.e., there's a 1 "in the middle")
                for (int nn = n_left + 1; nn < n; nn++)
                    if (ii_top[max_width - 1][nn] > ii_top[max_width - 1][n]) width = 1;
                _tmp[h][m][n] = width;

                // update n_left ring buffer
                for (int k = 0; k < max_width - 1; k++) ii_left[k] = ii_left[k + 1];
                ii_left[max_width - 1] = n;

                // update ii_top ring buffer
                for (int k = 0; k < max_width - 1; k++) ii_top[k][n] = ii_top[k + 1][n];
                ii_top[max_width - 1][n] = m;

                // block is too small -- skip
                if (width != max_width) continue;

                // retained blocks are set to zeros
                for (size_t km = 0; km < max_width; km++)
                    for (size_t kn = 0; kn < max_width; kn++) {
                        int mm = ii_top[km][n];
                        int nn = ii_left[kn];
                        if (mm < 0 || nn < 0) continue;
                        _layout[h][mm][nn] = 0;
                        _tmp[h][mm][nn] = 0;
                        _scratch[h][current[h]][0] = (int)h;
                        _scratch[h][current[h]][1] = (int)mm;
                        _scratch[h][current[h]][2] = (int)nn;
                        _scratch[h][current[h]][3] = _idx[h][mm][nn];
                        current[h]++;
                    }
            }
        }
    }
    std::vector<torch::Tensor> to_cat;
    for (size_t h = 0; h < H; h++)
        if (current[h] > 0) to_cat.push_back(scratch[h].slice(0, 0, current[h]));
    if (!to_cat.empty()) ret.push_back({max_width, torch::cat(to_cat)});
}

ret_t sdd_segment(torch::Tensor layout, int start_width)
{
    ret_t ret;

    // block index
    torch::Tensor idx = torch::zeros_like(layout);
    int current = 0;
    size_t H = layout.size(0);
    size_t M = layout.size(1);
    size_t N = layout.size(2);
    auto _layout = layout.accessor<int, 3>();
    auto _idx = idx.accessor<int, 3>();
    for (size_t h = 0; h < H; h++)
        for (size_t m = 0; m < M; m++)
            for (size_t n = 0; n < N; n++) {
                if (_layout[h][m][n] == 0) continue;
                _idx[h][m][n] = current++;
            }

    // scratch memory
    torch::Tensor scratch = torch::empty({H, layout.sum().item<int>(), 4}, layout.dtype());

    for (int max_width = start_width; max_width > 0; max_width /= 2)
        segment_blocks(layout, idx, scratch, max_width, ret);
    return ret;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sdd_segment", &sdd_segment, "SDD segmentation handler");
}
