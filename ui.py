import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objs as go

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 学习率调度器函数
def get_scheduler(name, optimizer, num_epochs, params):
    if name == 'LambdaLR':
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=params.get('lr_lambda', lambda epoch: 0.95 ** epoch))
    elif name == 'LambdaLR2':
        def rate(step: int, model_size: int, factor: int, warmup: int):
            if step == 0:
                step = 1
            return factor * (
                model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
            )
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: rate(
            step, 
            params.get('model_size', 10), 
            params.get('factor', 1), 
            params.get('warmup', 5)
        ))
    elif name == 'MultiplicativeLR':
        return optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=params.get('lr_lambda', lambda epoch: 0.95))
    elif name == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=params.get('step_size', 10), gamma=params.get('gamma', 0.1))
    elif name == 'MultiStepLR':
        milestones = params.get('milestones', [20, 30])
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=params.get('gamma', 0.1))
    elif name == 'ConstantLR':
        return optim.lr_scheduler.ConstantLR(optimizer, factor=params.get('factor', 1.0))
    elif name == 'LinearLR':
        return optim.lr_scheduler.LinearLR(optimizer, start_factor=params.get('start_factor', 1.0), 
                                          end_factor=params.get('end_factor', 0.1), total_iters=params.get('total_iters', 40))
    elif name == 'ExponentialLR':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.get('gamma', 0.9))
    elif name == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.get('T_max', 20))
    elif name == 'CyclicLR':
        return optim.lr_scheduler.CyclicLR(optimizer, base_lr=params.get('base_lr', 0.001), 
                                          max_lr=params.get('max_lr', 0.1), 
                                          step_size_up=params.get('step_size_up', 10), 
                                          mode=params.get('mode', 'triangular'))
    elif name == 'CosineAnnealingWarmRestarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=params.get('T_0', 10), 
                                                               T_mult=params.get('T_mult', 2))
    elif name == 'OneCycleLR':
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=params.get('max_lr', 0.1), 
                                            total_steps=params.get('total_steps', num_epochs))
    elif name == 'PolynomialLR':
        return optim.lr_scheduler.PolynomialLR(optimizer, total_iters=params.get('total_iters', num_epochs), 
                                              power=params.get('power', 2))
    elif name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=params.get('factor', 0.1), 
                                                   patience=params.get('patience', 10))
    else:
        return None

# Streamlit 应用
def main():
    st.set_page_config(page_title="学习率调度器可视化工具", layout="wide")
    st.title("📈 学习率调度器交互式可视化工具")

    st.sidebar.header("🔧 设置参数")

    num_epochs = st.sidebar.slider("训练轮数", min_value=1, max_value=100, value=40, step=1)
    initial_lr = st.sidebar.slider("初始学习率", min_value=0.0001, max_value=1.0, value=0.1, step=0.001)

    scheduler_name = st.sidebar.selectbox(
        "选择学习率调度器",
        ('LambdaLR', 'LambdaLR2', 'MultiplicativeLR', 'StepLR', 'MultiStepLR',
         'ConstantLR', 'LinearLR', 'ExponentialLR', 'CosineAnnealingLR',
         'CyclicLR', 'CosineAnnealingWarmRestarts', 'OneCycleLR',
         'PolynomialLR', 'ReduceLROnPlateau')
    )

    # 根据选择的调度器显示相应的参数设置
    params = {}
    with st.sidebar.expander("⚙️ 调整调度器参数"):
        if scheduler_name == 'LambdaLR':
            st.subheader("LambdaLR 参数")
            decay_rate = st.slider("衰减率 (每个epoch)", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
            params['lr_lambda'] = lambda epoch: decay_rate ** epoch
        elif scheduler_name == 'LambdaLR2':
            st.subheader("LambdaLR2 参数")
            model_size = st.slider("模型大小", min_value=1, max_value=100, value=10, step=1)
            factor = st.slider("因子", min_value=1, max_value=10, value=1, step=1)
            warmup = st.slider("预热步数", min_value=1, max_value=20, value=5, step=1)
            params['model_size'] = model_size
            params['factor'] = factor
            params['warmup'] = warmup
        elif scheduler_name == 'MultiplicativeLR':
            st.subheader("MultiplicativeLR 参数")
            gamma = st.slider("乘数因子", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
            params['lr_lambda'] = lambda epoch: gamma
        elif scheduler_name == 'StepLR':
            st.subheader("StepLR 参数")
            step_size = st.slider("步长 (多少轮后衰减)", min_value=1, max_value=50, value=10, step=1)
            gamma = st.slider("衰减因子", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
            params['step_size'] = step_size
            params['gamma'] = gamma
        elif scheduler_name == 'MultiStepLR':
            st.subheader("MultiStepLR 参数")
            milestones = st.text_input("里程碑 (以逗号分隔)", value="20,30")
            try:
                milestones = [int(x.strip()) for x in milestones.split(',')]
                params['milestones'] = milestones
            except:
                st.error("请输入有效的里程碑，以逗号分隔的整数")
            gamma = st.slider("衰减因子", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
            params['gamma'] = gamma
        elif scheduler_name == 'ConstantLR':
            st.subheader("ConstantLR 参数")
            factor = st.slider("因子", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
            params['factor'] = factor
        elif scheduler_name == 'LinearLR':
            st.subheader("LinearLR 参数")
            start_factor = st.slider("起始因子", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
            end_factor = st.slider("结束因子", min_value=0.0, max_value=2.0, value=0.1, step=0.1)
            total_iters = st.slider("总迭代次数", min_value=1, max_value=100, value=40, step=1)
            params['start_factor'] = start_factor
            params['end_factor'] = end_factor
            params['total_iters'] = total_iters
        elif scheduler_name == 'ExponentialLR':
            st.subheader("ExponentialLR 参数")
            gamma = st.slider("衰减因子", min_value=0.0, max_value=1.0, value=0.9, step=0.01)
            params['gamma'] = gamma
        elif scheduler_name == 'CosineAnnealingLR':
            st.subheader("CosineAnnealingLR 参数")
            T_max = st.slider("T_max", min_value=1, max_value=50, value=20, step=1)
            params['T_max'] = T_max
        elif scheduler_name == 'CyclicLR':
            st.subheader("CyclicLR 参数")
            base_lr = st.slider("基础学习率", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001)
            max_lr = st.slider("最大学习率", min_value=0.0001, max_value=1.0, value=0.1, step=0.0001)
            step_size_up = st.slider("上升步数", min_value=1, max_value=50, value=10, step=1)
            mode = st.selectbox("模式", ('triangular', 'triangular2', 'exp_range'))
            params['base_lr'] = base_lr
            params['max_lr'] = max_lr
            params['step_size_up'] = step_size_up
            params['mode'] = mode
        elif scheduler_name == 'CosineAnnealingWarmRestarts':
            st.subheader("CosineAnnealingWarmRestarts 参数")
            T_0 = st.slider("初始重启周期", min_value=1, max_value=50, value=10, step=1)
            T_mult = st.slider("重启周期倍增因子", min_value=1, max_value=5, value=2, step=1)
            params['T_0'] = T_0
            params['T_mult'] = T_mult
        elif scheduler_name == 'OneCycleLR':
            st.subheader("OneCycleLR 参数")
            max_lr = st.slider("最大学习率", min_value=0.0001, max_value=1.0, value=0.1, step=0.0001)
            total_steps = st.slider("总步数", min_value=1, max_value=100, value=num_epochs, step=1)
            params['max_lr'] = max_lr
            params['total_steps'] = total_steps
        elif scheduler_name == 'PolynomialLR':
            st.subheader("PolynomialLR 参数")
            total_iters = st.slider("总迭代次数", min_value=1, max_value=100, value=num_epochs, step=1)
            power = st.slider("多项式次数", min_value=1, max_value=5, value=2, step=1)
            params['total_iters'] = total_iters
            params['power'] = power
        elif scheduler_name == 'ReduceLROnPlateau':
            st.subheader("ReduceLROnPlateau 参数")
            factor = st.slider("衰减因子", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
            patience = st.slider("耐心值", min_value=1, max_value=20, value=10, step=1)
            params['factor'] = factor
            params['patience'] = patience

    # 初始化或获取保存的曲线
    if 'saved_curves' not in st.session_state:
        st.session_state.saved_curves = []

    with st.spinner("训练中..."):
        model = SimpleNet()
        optimizer = optim.SGD(model.parameters(), lr=initial_lr)
        scheduler = get_scheduler(scheduler_name, optimizer, num_epochs, params)

        if scheduler is None:
            st.error("未选择有效的调度器")
            return

        lr_curves = []
        for epoch in range(num_epochs):
            optimizer.step()  # 进行一次优化步骤
            lr = optimizer.param_groups[0]['lr']
            lr_curves.append(lr)  # 记录当前学习率

            # 处理 ReduceLROnPlateau 需要的指标
            if scheduler_name == 'ReduceLROnPlateau':
                metrics = torch.tensor(1.0)  # 模拟一个指标
                scheduler.step(metrics)  # 更新学习率
            else:
                scheduler.step()  # 更新学习率

        # 使用 Plotly 进行可视化
        fig = go.Figure()
        
        # 添加保存的曲线
        for i, curve in enumerate(st.session_state.saved_curves):
            fig.add_trace(go.Scatter(
                x=list(range(1, num_epochs + 1)),
                y=curve['lr_curves'],
                mode='lines+markers',
                name=f"{curve['name']} (保存的)",
                line=dict(dash='dash'),
                marker=dict(size=4)
            ))

        # 添加当前曲线
        fig.add_trace(go.Scatter(
            x=list(range(1, num_epochs + 1)),
            y=lr_curves,
            mode='lines+markers',
            name=f"{scheduler_name} (当前)",
            line=dict(color='royalblue', width=2),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title=f"学习率曲线比较",
            xaxis_title="轮数",
            yaxis_title="学习率",
            template="plotly_dark",
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)
        st.success("🎉 训练完成！")

        # 添加保存和删除曲线的功能
        col1, col2 = st.columns(2)
        with col1:
            if st.button("保存当前曲线"):
                st.session_state.saved_curves.append({
                    'name': scheduler_name,
                    'lr_curves': lr_curves
                })
                st.success(f"已保存 {scheduler_name} 曲线")

        with col2:
            if st.session_state.saved_curves:
                curve_to_delete = st.selectbox("选择要删除的曲线", 
                                               [f"{curve['name']} (保存的)" for curve in st.session_state.saved_curves])
                if st.button("删除选中的曲线"):
                    index = [f"{curve['name']} (保存的)" for curve in st.session_state.saved_curves].index(curve_to_delete)
                    del st.session_state.saved_curves[index]
                    st.success(f"已删除 {curve_to_delete}")

if __name__ == "__main__":
    main()