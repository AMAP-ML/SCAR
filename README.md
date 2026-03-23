<div align="center">

<h1>Semantic Context Matters: Improving Conditioning for Autoregressive Models</h1>

<div>
    <a href="https://scholar.google.com.hk/citations?user=1xA5KxAAAAAJ&hl=en&oi=ao" target="_blank">Dongyang Jin</a><sup>1,*</sup>,
    <a href="https://scholar.google.com.hk/citations?hl=en&user=MDrO_twAAAAJ" target="_blank">Ryan Xu</a><sup>1,*</sup>,
    <a href="https://zengjianhao.github.io/" target="_blank">Jianhao Zeng</a><sup>1</sup>,
    <a href="https://blue1rui.github.io/" target="_blank">Rui Lan</a><sup>1</sup>,
</div>
<div>
    <a href="https://scholar.google.com.hk/citations?user=Ilx8WNkAAAAJ&hl=en&oi=ao" target="_blank">Yancheng Bai</a><sup>1,#</sup>,
    <a href="https://allylei.github.io/" target="_blank">Lei Sun</a><sup>1</sup>,
  <a href="https://cxxgtxy.github.io/" target="_blank">Xiangxiang Chu</a><sup>1</sup>
</div>


<div>
    <sup>1</sup>Amap, Alibaba Group
</div>
<br>
</div>



[![Arxiv](https://img.shields.io/badge/arXiv-2507.19946-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.14063)


## Abstract

>Recently, autoregressive (AR) models have shown strong potential in image generation, offering better scalability and easier integration with unified multi-modal systems compared to diffusion-based methods. However, extending AR models to general image editing remains challenging due to weak and inefficient conditioning, often leading to poor instruction adherence and visual artifacts. To address this, we propose SCAR, a Semantic-Context-driven method for AutoregRessive models. SCAR introduces two key components: Compressed Semantic Prefilling, which encodes high-level semantics into a compact and efficient prefix, and Semantic Alignment Guidance, which aligns the last visual hidden states with target semantics during autoregressive decoding to enhance instruction fidelity. Unlike decoding-stage injection methods, SCAR builds upon the flexibility and generality of vector-quantized-based prefilling while overcoming its semantic limitations and high cost. It generalizes across both next-token and next-set AR paradigms with minimal architectural changes. SCAR achieves superior visual fidelity and semantic alignment on both instruction editing and controllable generation benchmarks, outperforming prior AR-based methods while maintaining controllability.

