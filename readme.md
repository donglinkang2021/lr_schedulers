<div align="center">

# LR_Schedulers

<img src="https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=FFFFFF" alt="Streamlit">
<img src="https://img.shields.io/badge/-PyTorch-FF4B4B?style=flat-square&logo=pytorch&logoColor=FFFFFF" alt="PyTorch">
<img src="https://img.shields.io/badge/-Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=FFFFFF" alt="Plotly">

</div>

## 介绍

LR_Schedulers 是一个用于比较不同学习率调度器（LR Scheduler）的UI界面的项目。它可以帮助用户直观地看到不同调度器在训练过程中学习率的变化情况，从而选择最适合的调度器。

## 功能

1. **选择调度器类型**：用户可以选择多种常见的学习率调度器类型，如 StepLR、MultiStepLR、ExponentialLR 等。
2. **设置参数**：用户可以为每个调度器设置相应的参数，如步长、衰减因子等。
3. **生成学习率曲线**：根据用户选择的调度器类型和参数，生成学习率曲线，并绘制在图表上。
4. **保存和删除曲线**：用户可以保存当前的曲线，并删除已保存的曲线。
