<!-- markdownlint-disable MD033 MD041 MD009 MD012 -->
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="description"
    content="SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips (GH200/GB200/MI300A).">
  <meta name="keywords" content="SuperOffload, Offload Training, GH200, GB200, MI300A, Large Language Models, ZeRO-3, DeepSpeed, Hugging Face">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips</title>

  <link href="https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro"
        rel="stylesheet">

  <link rel="stylesheet" href="./static/css/bulma.min.css">
  <link rel="stylesheet" href="./static/css/bulma-carousel.min.css">
  <link rel="stylesheet" href="./static/css/bulma-slider.min.css">
  <link rel="stylesheet" href="./static/css/fontawesome.all.min.css">
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1/css/academicons.min.css">
  <link rel="stylesheet" href="./static/css/index.css">
  <link rel="icon" href="./static/images/favicon.svg">

  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script defer src="./static/js/fontawesome.all.min.js"></script>
  <script src="./static/js/bulma-carousel.min.js"></script>
  <script src="./static/js/bulma-slider.min.js"></script>
  <script src="./static/js/index.js"></script>
  <style>
    /* Custom style to increase the max width of is-max-desktop */
    .is-max-desktop {
      max-width: 1020px; /* Adjust this value as needed */
    }
  /* Highlight bullet list styling */
  .highlight-list { margin:10px 0 60px 0; padding-left:1.4em; list-style:disc; }
  .highlight-list li { font-size:19px; line-height:1.45; margin:0 0 12px; }
  .highlight-list li:last-child { margin-bottom:0; }
  .highlight-list li strong { font-weight:600; }
  /* Indent code blocks inside lists */
  .pre-indent { margin-left: 5px; }
  </style>
</head>
<body>

  <nav class="navbar" role="navigation" aria-label="main navigation">
    <div class="navbar-brand">
      <a role="button" class="navbar-burger" aria-label="menu" aria-expanded="false">
        <span aria-hidden="true"></span>
        <span aria-hidden="true"></span>
        <span aria-hidden="true"></span>
      </a>
    </div>
    <div class="navbar-menu">
      <div class="navbar-start" style="flex-grow: 1; justify-content: center;">
        <a class="navbar-item" href="https://minjiazhang.github.io">
        <span class="icon">
            <i class="fas fa-home"></i>
        </span>
        </a>
  
        <div class="navbar-item has-dropdown is-hoverable">
          <a class="navbar-link">
            More Research
          </a>
          <div class="navbar-dropdown">
            <a class="navbar-item" href="https://minjiazhang.github.io/">
              Publications
            </a>
          </div>
        </div>
      </div>
  
    </div>
  </nav>

<section class="hero">
  <div class="hero-body">
    <div class="container is-max-desktop">
      <div class="columns is-centered">
        <div class="column has-text-centered">
          <h1 class="title is-1 publication-title" style="font-size: 48px;">SuperOffload: Unleashing the Power of<br>Large-Scale LLM Training on Superchips</h1>
          <p class="subtitle is-5" style="margin-top:20px; font-size:25px;">Efficient full-parameter fine-tuning of GPT-OSS-20B & Qwen3-14B models on a single GPU and Llama3-70B on four GPUs, achieving up to 600 TFLOPS</p>
 
          <div class="is-size-5 publication-authors" style="margin-top:20px;">
            <span class="author-block"><a href="https://xinyulian.tech/">Xinyu Lian</a><sup>1</sup>,</span>
            <span class="author-block">
              <a href="https://tohtana.github.io/">Masahiro Tanaka</a><sup>2</sup>,
            </span>
            <span class="author-block">
              <a href="https://www.snowflake.com/en/blog/authors/olatunji--tunji--ruwase/">Olatunji Ruwase</a><sup>3</sup>,
            </span>
            <span class="author-block">
              <a href="https://minjiazhang.github.io/">Minjia Zhang</a><sup>1</sup>
            </span>
          </div>
            <div class="is-size-5 publication-authors" style="margin-top:6px;">
            <span class="author-block" style="margin-right: 10px;"><sup>1</sup>University of Illinois Urbana-Champaign</span>
            <span class="author-block" style="margin-right: 10px;"><sup>2</sup>Anyscale</span>
            <span class="author-block"><sup>3</sup>Snowflake</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>


<section class="section" style="margin-top:-30px;">
  <div class="container is-max-desktop">
    <div class="columns is-centered">
      <div class="column">
        <div class="content">
          <p>
            Recent models, especially MoE, at the scale of tens to hundreds of billions of parameters, making fine-tuning on limited GPUs difficult. Offloading to CPU memory helps reduce GPU demand but typically assumes GPU-CPU connections over PCIe, which is bandwidth-limited (e.g., 32 GB/s on PCIe-Gen4). Thus, prior work mainly optimizes data transfers to avoid PCIe becoming a major performance bottleneck. However, hardware vendors are introducing a new class of tightly coupled architectures—such as NVIDIA GH200, GB200, and AMD MI300A—that challenge these long-standing assumptions.
          </p>
          <p>
            The open-source release of <strong>SuperOffload</strong> addresses this gap by providing a set of modular techniques for efficient large-model training. With SuperOffload, models such as <strong>GPT-OSS-20B</strong>, <strong>Qwen3-14B</strong>, and <strong>Phi-4</strong> can be fully fine-tuned on a single GH200, achieving <strong>600 TFLOPS</strong> under modest settings (sequence length 4k, batch size 4). This delivers up to <strong>4x</strong> higher throughput compared to ZeRO-Offload.
          </p>
          <p>
            Built on top of ZeRO Stage 3, SuperOffload enables scaling to even larger models, including Qwen3-30B-A3B, Seed-OSS-36B on two GH200s and Llama-70B on four GH200s. All of this is supported natively through Hugging Face Transformers and DeepSpeed, with no need for custom modeling code.
          </p>
          <div style="margin-top:25px;">
            <figure class="image" style="display:inline-block; max-width:1024px;">
              <img src="static/images/comparision.jpg" alt="SuperOffload system overview" style="width:100%; border:1px solid #ddd; border-radius:6px;">
              <figcaption style="font-size:14px; color:#555; margin-top:6px;">Figure 1: SuperOffload delivers up to 4x higher throughput than ZeRO-Offload for large-model fine-tuning across varying sequence lengths and batch sizes, achieving a peak throughput of 600 TFLOPS.</figcaption>
            </figure>
          </div>

        </div>
      </div>
    </div>
  </div>
</section>

<!-- <section class="hero teaser">
  <div class="container is-max-desktop">
    <div class="hero-body">
      <img src="./static/images/ucp.jpg"
      class="interpolation-image"
      alt="UCP main results image."/>
      <h2 class="subtitle">
        UCP boosts large-scale training efficiency: 
        <ul>
          <li>🚀 Flexible change of parallelism (PP, SP, TP, ZeRO-DP) or GPU count mid-stream </li>
          <li>🚀 Improve resilience by scaling down to healthy nodes</li>
          <li>🚀 Increase throughput by scaling up to elastic nodes</li>
        </ul>
      </h2>
    </div>
  </div>
</section> -->

<section class="section" style="margin-top: -40px;">
  <div class="container is-max-desktop">
    <!-- Abstract. -->
    <div class="columns is-centered">
      <div class="column">
        <h2 class="title is-3" style="font-weight: bold; font-family: Arial;">SuperOffoad Highlight</h2>
                <ul class="highlight-list">
                  <li><strong>Single GH200:</strong> Full fine-tuning of GPT-OSS-20B, Qwen3-14B, achieving ~600 TFLOPS (seq len 4K, batch size 4).</li>
                  <li><strong>Scales Further:</strong> Qwen3-30B-A3B &amp; Seed-OSS-36B on 2x GH200; Llama-70B on 4x GH200.</li>
                  <li><strong>Throughput Gains:</strong> Up to 4x vs ZeRO-Offload under modest settings.</li>
                  <li><strong>Built On:</strong> DeepSpeed ZeRO Stage 3 + native Hugging Face Transformers integration.</li>
                  <li><strong>No Custom Modeling Code:</strong> Drop-in configuration driven.</li>
                </ul>

        <h2 class="title is-3" style="font-weight: bold; font-family: Arial; font-size: 2.5rem;">How SuperOffload works</h2>
        <section class="section" style="margin-top:-40px;">
          <div class="container is-max-desktop">
            <div class="columns is-centered">
              <div class="column">
                <h2 class="title is-3" style="font-weight: bold; font-family: Arial;">1. Speculation-then-Validation (STV): Overlap CPU-Adam with backward propagation on the GPU</h2>
                <div class="content has-text-justified" style="margin-top:-8pt;">
                  <p>
                    As shown in the Figure 2 (ZeroOffload), the clipping of the gradient norm requires calculating the global gradient norm, and mixed precision training requires a global check of NAN and INF values. Which requires the CPU to wait until all gradients have been received before the optimizer step and weight updates. As illustrated by the idle block in Figure 2 (ZeroOffload), this dependency exposes the optimizer step to the critical path, preventing it from overlapping with the backward pass.
                  </p>
                  <p>
                    To address this limitation, we propose a <strong>speculation-then-validation</strong> schedule, which largely bypasses these synchronizations while preserving the exact convergence property of the training. Our mechanism is based on a key observation: most of the time the global states have no effects. For example, gradient clipping is rarely triggered, especially after the initial warm-up phase when gradient variance significantly reduces. As shown in Figure 3, in BLOOM (176B) training, after iteration 1000, when training becomes more stable, gradient clipping rarely happens - occurring only 93 times between steps 1000 and 80000, which represents 0.12% of the total iterations. Similarly, mixed precision training rarely encounters NAN and INF, as a healthy training run should not have numerical instability issues. The situation improves further with BF16 training and during fine-tuning, where the process is considerably more stable compared to FP16 and large-scale pre-training.
                  </p>
                  <p>
                    Therefore, instead of waiting for all gradients to arrive, the CPU initiates the optimizer step speculatively using the gradients available at that moment. Once the update is complete, the new parameters are copied back to the GPU and replace the old ones. During the validation phase (1) if NaNs and INFs are detected, the iteration is skipped; (2) if gradients exceed clipping thresholds (e.g., after finishing computing the global gradient norm across all parameter gradients), SuperOffload reverts the previous optimizer update and re-executes it using the clipped gradients. We implemented the in-place rollback as one function of the CPU-Adam.
                  </p>
                </div>
                <div class="has-text-centered" style="margin-top:25px;">
                  <figure class="image" style="display:inline-block; max-width:800px; margin:0 auto;">
                    <img src="static/images/schedule.jpg" alt="SuperOffload system overview" style="width:100%; border:1px solid #ddd; border-radius:6px;">
                    <figcaption style="font-size:14px; color:#555; margin-top:6px;">Figure 2: Previous offloading approach suffers from global gradient norm and global check of NAN and INF values, which expose the optimizer step to the critical path and prevent overlapping opportunities. In SuperOffload, we introduce a speculation-then-validation schedule to address this issue.</figcaption>
                  </figure>
                  <figure class="image" style="display:inline-block; max-width:800px; margin:40px auto 0;">
                    <img src="static/images/rollback.jpg" alt="SuperOffload system overview" style="width:100%; border:1px solid #ddd; border-radius:6px;">
                    <figcaption style="font-size:14px; color:#555; margin-top:6px;">Figure 3: Red points indicate instances in the figure showing the gradient clipping triggered during training. In BLOOM pre-training, gradient clipping is rarely triggered after the initial warm-up phase, demonstrating the practical benefits of our speculation-then-validation approach.</figcaption>
                  </figure>
                </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section class="section" id="stv" style="margin-top:-40px;">
          <div class="container is-max-desktop">
            <div class="columns is-centered">
              <div class="column">
                <h2 class="title is-3" style="font-weight: bold; font-family: Arial;">2. Partial Offloading with Fine-Grained Bucketization</h2>
                <div class="content has-text-justified" style="margin-top:-8pt;">
                  <p>
                    In contrast to traditional ZeRO-Offload, where the forward pass of the next iteration waits for all updated parameters to return from CPU, SuperOffload reduces this synchronization bubble. It does so by keeping the optimizer states and gradients of the <em>last few buckets</em> directly in GPU memory (when Hopper HBM allows). At the same time, it ensures that the final offloaded bucket finishes its optimization step early enough so the next iteration can begin without stalling. The number of these “tail buckets” kept on the GPU is denoted as <code>n'</code>. Adjusting <code>n'</code> lets us trade a small amount of extra memory use for an throughput improvement in overlap and less idle time at the end of each iteration.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section class="section" style="margin-top:-40px;">
          <div class="container is-max-desktop">
            <div class="columns is-centered">
              <div class="column">
                <h2 class="title is-3" style="font-weight: bold; font-family: Arial;">3. Superchip-Aware Casting</h2>
                <div class="content has-text-justified" style="margin-top:-8pt;">
                  <p>
                    In DL training frameworks like PyTorch and DeepSpeed, the mixed precision training is implemented through a graph rewriting process. The default precision of all ops is float32 (FP32). Mixed precision training casts certain model states (e.g., weights, gradients) from FP32 to float16 (FP16,BF16), or vice versa. For example, the gradients in the backward pass are produced in FP16/BF16/FP8, and the optimizer computes the updates using FP32 gradients. when considering offloading strategies, the cost is from transfer tensors between the GPU and CPU but also involve converting tensor data types.
                  </p>
                  <p>
                    Existing offloading-based solutions often adopt a minimum edge cut algorithm to computation graph for minimal edge cut on a computation graph assuming casting + transfer costs are dominated by bandwidth. On Superchips, the high-bandwidth CPU↔GPU link shifts the cost balance and <em>casting</em> becomes non-negligible. As illustrated in Figure 4, SuperOffload improves efficiency by performing casting on the GPU and transferring high-precision tensors to the CPU.
                  </p>
                </div>
                  <div class="has-text-centered" style="margin-top:25px;">
                  <figure class="image" style="display:inline-block; max-width:700px; margin:0 auto;">
                    <img src="static/images/cast_transfer.jpg" alt="SuperOffload system overview" style="width:100%; border:1px solid #ddd; border-radius:6px;">
                    <figcaption style="font-size:14px; color:#555; margin-top:6px;">Figure 4: Compared to the previous approach—transferring low-precision tensors and casting them to higher precision on the CPU—casting to higher precision first and then transferring high-precision tensors is more efficient on Superchips.</figcaption>
                  </figure>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section class="section" style="margin-top:-40px;">
          <div class="container is-max-desktop">
            <div class="columns is-centered">
              <div class="column">
                <h2 class="title is-3" style="font-weight: bold; font-family: Arial;"># Experience and Insights</h2>
                <div class="content has-text-justified" style="margin-top:-8pt;">
                  <ul>
                    <li><strong>NUMA Binding:</strong> NUMA Binding is required for efficient training on Nvidia GH200. Each GPU is paired with a CPU to ensure that the training process is launched on the CPU directly associated with that GPU. This pairing improves affinity, delivering higher CPU-GPU bandwidth and greater throughput. In DeepSpeed, we provide a simple interface to enable NUMA binding: simply add the `--bind_cores_to_rank` flag when launching the DeepSpeed engine.</li>
                    <li><strong>MPAM:</strong> Memory System Resource Partitioning and Monitoring (MPAM) is essential for achieving optimal throughput performance. In SuperOffload, GPU execution is overlapped with CPU-based Adam execution. MPAM helps reduce interference between these two processes, leading to smoother execution and better performance.</li>
                    <li><strong>How to enable MPAM on Nvidia Superchips:</strong>
                      <ol style="margin-top:6px; margin-bottom:0;">
                        <li>Install the kernel from NVIDIA <a href="https://github.com/NVIDIA/NV-Kernels/tree/24.04_linux-nvidia-adv-6.11">NV-Kernels</a>.</li>
                        <li>Check that MPAM is supported and enabled on the system:
                          <pre class="pre-indent"><code class="language-bash">grep MPAM /boot/config-$(uname -r)</code></pre>
                          <p class="pre-indent">Expected output:</p>
                          <pre class="pre-indent"><code class="language-text">CONFIG_ARM64_MPAM=y
CONFIG_ACPI_MPAM=y
CONFIG_ARM64_MPAM_DRIVER=y
CONFIG_ARM64_MPAM_RESCTRL_FS=y</code></pre>
                          <p class="pre-indent"><em>Optional:</em> Verify resctrl filesystem:</p>
                          <pre class="pre-indent"><code class="language-bash">ls -ld /sys/fs/resctrl</code></pre>
                        </li>
                        <li>Mount resctrl</li>
                        <pre class="pre-indent"><code class="language-bash">mount -t resctrl resctrl /sys/fs/resctrl</code></pre>
                        <li>Create partition p1, p2</li>
                        <pre class="pre-indent"><code class="language-bash">mkdir /sys/fs/resctrl/p1 /sys/fs/resctrl/p2</code></pre>
                        <li>Set cpu cores list and cache partition and memory partition for p1 and p2</li>
                        <p class="pre-indent">Recommended config based on our experiments: </p>
                        <pre class="pre-indent"><code class="language-bash">/sys/fs/resctrl/p1/cpus_list:
0-6
/sys/fs/resctrl/p2/cpus_list:
7-71
/sys/fs/resctrl/p1/schemata:
MB:1=100
L3:1=ff0
/sys/fs/resctrl/p2/schemata:
MB:1=20
L3:1=f</code></pre>
                      </ol>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section class="section" style="margin-top:-40px;">
          <div class="container is-max-desktop">
            <div class="columns is-centered">
              <div class="column">
                <h2 class="title is-3" style="font-weight: bold; font-family: Arial;">Status &amp; Availability</h2>
                <div class="content has-text-justified" style="margin-top:-8pt;">
                  <p>
                    SuperOffload is released as modular extensions atop ZeRO Stage 3 inside DeepSpeed with native configuration hooks exposed to Hugging Face Transformers (no model code changes). Community feedback &amp; contributions are welcome.
                  </p>
                  <p>
                    To enable SuperOffload, simply add the following one line in while box to your DeepSpeed configuration file:
                  </p>
                </div>
                  <div class="has-text-centered" style="margin-top:25px;">
                  <figure class="image" style="display:inline-block; max-width:500px;">
                    <img src="static/images/enable_superoffload.jpg" alt="SuperOffload system overview" style="width:100%; border:1px solid #ddd; border-radius:6px;">
                    <figcaption style="font-size:14px; color:#555; margin-top:6px;">Figure 5: Enable SuperOffload with single one line on DeepSpeed config.</figcaption>
                  </figure>
                </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section class="section" id="BibTeX" style="margin-top:-40px;">
          <div class="container is-max-desktop content">
            <h2 class="title">Acknowledgements</h2>
            <p>This work is the result of a close collaboration between University of Illinois Urbana-Champaign (UIUC) and DeepSpeed team.</p>
            <p>We also gratefully acknowledge William Gropp, Brett Bode, and Gregory H. Bauer from the National Center for Supercomputing Applications (NCSA), as well as Dan Ernst, Ian Karlin, Giridhar Chukkapalli, Kurt Rago, and others from NVIDIA for their valuable discussions and guidance on MPAM support on Grace CPU.</p>
        </section>
        <section class="section" id="BibTeX" style="margin-top:-40px;">
          <div class="container is-max-desktop content">
            <h2 class="title">BibTeX</h2>
            <pre><code>@inproceedings{superoffload,
    author = {Xinyu Lian and Masahiro Tanaka and Olatunji Ruwase and Minjia Zhang},
    title = "{SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips}",
    year = {2026},
    booktitle = {Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating System (ASPLOS'26)}
}</code></pre>
          </div>
        </section>
      <div class="column is-8">
        <div class="content">
          <p>
            We thank the authors of <a href="https://nerfies.github.io/">Nerfies</a> that kindly open sourced the template of this website. It is licensed under a <a href="https://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
          </p>
        </div>
      </div>
    </div>
  </div>
</footer>

</body>
</html>


