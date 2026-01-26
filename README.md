# BiKA: Kolmogorov-Arnold-Network-inspired Ultra Lightweight Neural Network Hardware Accelerator # 

**Yuhao Liu<sup>1,2,3</sup>, Salim Ullah<sup>1</sup>, Akash Kumar<sup>1</sup>**

<sup>1</sup>Ruhr University Bochum, Germany; <sup>2</sup>Dresden University of Technology, Germany; <sip>3</sup>Center for Scalable Data Analytics and Artificial Intelligence (ScaDS.AI Dresden/Leipzig), Germany

Email: yuhao.liu@rub.de, salim.ullah@rub.de, akash.kumar@rub.de

## Abstract
Lightweight neural network accelerators are essential for edge devices with limited resources and power constraints. While quantization and binarization can efficiently reduce hardware cost, they still rely on the conventional *Artificial Neural Network* (ANN) computation pattern. The recently proposed
*Kolmogorov-Arnold Network* (KAN) presents a novel network paradigm built on learnable nonlinear functions. However, it is computationally expensive for hardware deployment. Inspired by KAN, we propose **BiKA**, a multiply-free architecture that replaces nonlinear functions with binary, learnable thresholds, introducing an extremely lightweight computational pattern that requires only comparators and accumulators. Our FPGA prototype on *Ultra96-V2* shows that BiKA reduces hardware resource usage by **27.73%** and **51.54%** compared with binarized and quantized neural network systolic array accelerators, while maintaining competitive accuracy. BiKA provides a promising direction for hardware-friendly neural network design on edge devices.

## Experiment Environment
| Library | Version |
| :--- | :---: |
| ***Python*** | 3.9.18 |
| ***PyTorch*** | 2.1.0 |
| ***pykan*** | 0.2.8 |
| ***Brevitas*** | 0.11.0 |
| ***JupyterLab*** | 4.3.5 |
| ***bika***| 0.1.3 |

Please install our CUDA-based library ***bika*** before testing examples.

