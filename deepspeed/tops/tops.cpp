#include "moe_gating.h"
#include "rope.h"
#include "swiglu.h"

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("swiglu_fwd", &swiglu_fwd, "swiglu_fwd function (fwd)");
    m.def("swiglu_bwd", &swiglu_bwd, "swiglu_bwd function (bwd)");
    m.def("rope_fwd", &rope_fwd, "rope_fwd function (fwd)");
    m.def("rope_bwd", &rope_bwd, "rope_bwd function (bwd)");
    m.def("moe_gating_fwd", &gate_fwd, "MoE gating function (fwd)");
    m.def("moe_gating_scatter", &gate_scatter, "MoE gating scatter function (fwd)");
    m.def("moe_gating_bwd", &gate_bwd, "MoE gating function (bwd)");
    m.def("moe_gather_fwd", &gather_fwd, "MoE gather function (fwd)");
    m.def("moe_gather_bwd", &gather_bwd, "MoE gather function (bwd)");
}