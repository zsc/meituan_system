# 第5章：规划引擎（网络/站点/运力结构规划）

在美团外卖的庞大系统中，如果说调度引擎是"战术执行官"，那么规划引擎就是"战略参谋长"。它不关心某个具体订单该派给哪个骑手，而是思考更宏观的问题：站点应该建在哪里？每个区域需要多少运力？高峰期如何提前布局？这些看似简单的问题，直接决定了整个系统的效率上限。一个糟糕的站点布局，即使有再智能的调度算法也无法弥补；一个不合理的运力结构，会让高峰期永远处于崩溃边缘。本章将深入探讨规划引擎如何通过数学优化、仿真评估和数据驱动，为千万级订单的履约奠定坚实的结构性基础。

## 5.1 规划层级与时间尺度

规划引擎采用经典的三层规划体系，不同层级关注不同的时间尺度和决策粒度：

### 5.1.1 战略规划（月度/季度）

战略规划关注长期结构性问题，其决策影响深远且调整成本高昂：

```
┌─────────────────────────────────────────────────────────────┐
│                      战略规划决策空间                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入维度：                     输出决策：                   │
│  ┌──────────────┐             ┌──────────────┐            │
│  │ • 城市扩张   │             │ • 站点布局   │            │
│  │ • 市场预测   │  ────────►  │ • 运力规模   │            │
│  │ • 竞争格局   │             │ • 服务边界   │            │
│  │ • 政策变化   │             │ • 投资计划   │            │
│  └──────────────┘             └──────────────┘            │
│                                                             │
│  关键指标：                                                 │
│  • 覆盖率：可服务区域占比 > 95%                            │
│  • 密度：平均配送半径 < 3km                                │
│  • 成本：单均履约成本下降 10%                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

战略规划的核心是设施选址问题（Facility Location Problem），其数学模型可表示为：

```
minimize: Σᵢ fᵢ·yᵢ + ΣᵢΣⱼ cᵢⱼ·xᵢⱼ

subject to:
  Σᵢ xᵢⱼ = 1,        ∀j ∈ Customers    (每个客户必须被服务)
  xᵢⱼ ≤ yᵢ,          ∀i,j              (只能从开放的站点配送)
  Σⱼ dⱼ·xᵢⱼ ≤ Cᵢ·yᵢ, ∀i               (站点容量约束)
  yᵢ ∈ {0,1},        ∀i                (站点开关决策)
  xᵢⱼ ≥ 0,           ∀i,j              (配送分配比例)

其中：
  fᵢ: 站点i的固定成本
  cᵢⱼ: 从站点i到客户j的配送成本
  dⱼ: 客户j的需求量
  Cᵢ: 站点i的最大容量
```

### 5.1.2 战术规划（周度/日度）

战术规划在战略框架内做中期优化，重点解决供需平衡和资源配置：

```
┌─────────────────────────────────────────────────────────────┐
│                      战术规划优化流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  历史数据         预测模型         优化决策                │
│  ┌─────┐        ┌─────┐        ┌─────┐                  │
│  │订单 │───────►│需求 │───────►│运力 │                  │
│  │轨迹 │        │预测 │        │调配 │                  │
│  │时长 │        └─────┘        └─────┘                  │
│  └─────┘            │              │                       │
│                     ▼              ▼                       │
│              ┌──────────┐   ┌──────────┐                 │
│              │时段分解  │   │区域分配  │                 │
│              │(峰/谷/平)│   │(跨区支援)│                 │
│              └──────────┘   └──────────┘                 │
│                                                             │
│  核心算法：Multi-Period Vehicle Routing Problem (MPVRP)    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

战术规划的典型问题是班次优化（Shift Scheduling），需要在满足服务水平的前提下最小化人力成本：

```
minimize: Σₜ Σₛ cₛ·xₜₛ

subject to:
  Σₛ aₜₛ·xₜₛ ≥ Dₜ,    ∀t ∈ TimePeriods  (满足各时段需求)
  xₜₛ ≤ Aₛ,           ∀t,s              (可用人力约束)
  Σₜ xₜₛ·hₛ ≤ Hmax,   ∀s                (工时上限约束)
  xₜₛ ≥ 0,            ∀t,s              (非负约束)

其中：
  cₛ: 班次s的单位成本
  aₜₛ: 班次s在时段t的覆盖系数
  Dₜ: 时段t的运力需求
  Aₛ: 班次s的可用人数
  hₛ: 班次s的工作时长
```

### 5.1.3 操作规划（小时级/分钟级）

操作规划负责实时微调，确保系统平稳运行：

```
┌─────────────────────────────────────────────────────────────┐
│                    操作规划实时决策引擎                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  实时监控                    动态调整                       │
│  ┌──────────┐              ┌──────────┐                  │
│  │ 订单积压 │              │ 运力调度 │                  │
│  │ 骑手分布 │  ─────────►  │ 区域联动 │                  │
│  │ 路况变化 │              │ 激励发放 │                  │
│  └──────────┘              └──────────┘                  │
│       │                          │                         │
│       ▼                          ▼                         │
│  ┌────────────────────────────────────┐                  │
│  │        控制策略库                   │                  │
│  ├────────────────────────────────────┤                  │
│  │ IF 积压 > 阈值 THEN 发放加价券     │                  │
│  │ IF 运力缺口 > 20% THEN 跨区调配    │                  │
│  │ IF 雨天 THEN 延长配送时限         │                  │
│  └────────────────────────────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 5.2 站点选址与容量规划

站点是外卖履约网络的基础节点，其位置和容量直接影响配送效率和成本结构。

### 5.2.1 多目标选址优化

现实中的站点选址需要平衡多个相互冲突的目标：

```
┌─────────────────────────────────────────────────────────────┐
│                    站点选址多目标优化                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  目标函数组合：                                             │
│                                                             │
│  1. 成本最小化：                                           │
│     minimize: Σ(租金ᵢ + 人力ᵢ + 运营ᵢ)                    │
│                                                             │
│  2. 覆盖最大化：                                           │
│     maximize: Σ Population(coverage_areaᵢ)                 │
│                                                             │
│  3. 时效最优化：                                           │
│     minimize: Σ avg_delivery_timeᵢ × orderᵢ                │
│                                                             │
│  4. 鲁棒性提升：                                           │
│     maximize: min(backup_capacityᵢ)                        │
│                                                             │
│  Pareto最优前沿：                                          │
│                                                             │
│     成本 ↑                                                  │
│         │ ○ 低成本                                         │
│         │   ＼ 高积压                                      │
│         │     ○─────○ 平衡点                              │
│         │           ＼                                     │
│         │             ○ 高服务                             │
│         │               低积压                             │
│         └────────────────► 服务质量                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2.2 层级化站点体系

美团采用多层级站点体系，不同层级承担不同功能：

```
┌─────────────────────────────────────────────────────────────┐
│                      站点层级架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  一级枢纽站（Hub）                                          │
│  ┌─────────────────────────────────┐                      │
│  │ • 覆盖范围：5-10km              │                      │
│  │ • 运力规模：200-500人           │                      │
│  │ • 功能定位：区域调度中心       │                      │
│  └────────────┬────────────────────┘                      │
│               │                                             │
│      ┌────────┴────────┬────────────┐                     │
│      ▼                 ▼            ▼                     │
│  二级配送站        二级配送站    二级配送站                │
│  ┌─────────┐      ┌─────────┐  ┌─────────┐              │
│  │ 3-5km   │      │ 3-5km   │  │ 3-5km   │              │
│  │ 50-100人│      │ 50-100人│  │ 50-100人│              │
│  └────┬────┘      └────┬────┘  └────┬────┘              │
│       │                │            │                      │
│   ┌───┴───┐       ┌───┴───┐    ┌───┴───┐                │
│   ▼       ▼       ▼       ▼    ▼       ▼                │
│  前置点  前置点  前置点  前置点 前置点  前置点            │
│  (1-2km) (1-2km) (1-2km) (1-2km)(1-2km) (1-2km)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2.3 动态容量调整

站点容量需要根据时段和季节动态调整：

```python
# 容量规划算法伪代码
def calculate_station_capacity(station_id, time_period):
    # 基础容量
    base_capacity = get_base_capacity(station_id)
    
    # 时段调整系数
    time_factor = {
        'morning_peak': 1.3,    # 早高峰
        'lunch_peak': 1.8,      # 午高峰
        'dinner_peak': 1.6,     # 晚高峰
        'normal': 1.0,          # 平峰
        'valley': 0.6           # 低谷
    }[time_period]
    
    # 季节调整系数
    season_factor = get_season_factor(current_date)
    
    # 天气调整系数
    weather_factor = get_weather_factor(weather_condition)
    
    # 历史溢出率修正
    overflow_correction = 1 + historical_overflow_rate * 0.2
    
    # 最终容量
    final_capacity = base_capacity * time_factor * season_factor * \
                    weather_factor * overflow_correction
    
    return min(final_capacity, physical_limit)
```

## 5.3 运力结构设计

运力结构设计是规划引擎的核心任务之一，需要在成本、效率、稳定性之间找到最优平衡点。

### 5.3.1 多元化运力体系

美团采用"专送+快送+众包"的三层运力结构，各有特点和适用场景：

```
┌─────────────────────────────────────────────────────────────┐
│                    运力结构金字塔                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    ╱─────────╲                             │
│                   ╱  众包骑手  ╲                           │
│                  ╱ • 弹性运力  ╲                          │
│                 ╱  • 低固定成本  ╲                         │
│                ╱   • 峰值补充    ╲                        │
│               ╱─────────────────────╲                      │
│              ╱      快送骑手         ╲                     │
│             ╱   • 兼职/灵活排班      ╲                    │
│            ╱    • 中等稳定性         ╲                   │
│           ╱     • 区域化管理          ╲                  │
│          ╱───────────────────────────────╲                 │
│         ╱         专送骑手              ╲                │
│        ╱     • 全职/固定排班             ╲               │
│       ╱      • 高稳定性/高服务质量        ╲              │
│      ╱       • 核心运力保障               ╲             │
│     ╱─────────────────────────────────────────╲            │
│                                                             │
│  运力配比优化模型：                                         │
│  专送 : 快送 : 众包 = 40% : 35% : 25% (典型配比)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3.2 运力成本结构分析

不同类型运力的成本结构差异显著，需要精细化管理：

```
┌─────────────────────────────────────────────────────────────┐
│                    运力成本结构对比                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  成本项目         专送        快送        众包             │
│  ─────────────────────────────────────────────────         │
│  基本工资         60%         40%         0%               │
│  绩效提成         25%         35%         70%              │
│  社保福利         10%         5%          0%               │
│  管理成本         3%          10%         15%              │
│  装备补贴         2%          10%         15%              │
│                                                             │
│  单均成本(元)     6.5         5.8         5.2              │
│  服务稳定性       ★★★★★       ★★★☆☆       ★★☆☆☆            │
│  峰值弹性         ★★☆☆☆       ★★★★☆       ★★★★★            │
│                                                             │
│  最优使用场景：                                             │
│  • 专送：核心商圈、品牌商家、高价值订单                    │
│  • 快送：次核心区域、常规订单、日常运力                    │
│  • 众包：峰值补充、偏远订单、临时需求                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3.3 动态运力调配算法

运力调配需要考虑时空分布不均的问题：

```python
# 跨区域运力调配优化
def optimize_cross_region_allocation(regions, time_slot):
    """
    目标：最小化全局运力缺口
    约束：调配成本、骑手接受度、服务质量
    """
    
    # 构建运力供需矩阵
    supply_demand_matrix = build_matrix(regions, time_slot)
    
    # 识别盈余区域和缺口区域
    surplus_regions = []
    deficit_regions = []
    
    for region in regions:
        net_capacity = supply_demand_matrix[region]['supply'] - \
                      supply_demand_matrix[region]['demand']
        if net_capacity > 0:
            surplus_regions.append((region, net_capacity))
        else:
            deficit_regions.append((region, -net_capacity))
    
    # 构建调配方案（最小成本流问题）
    allocation_plan = solve_min_cost_flow(
        surplus_regions,
        deficit_regions,
        distance_matrix,
        incentive_cost
    )
    
    # 考虑骑手意愿和历史表现
    adjusted_plan = adjust_for_rider_preference(
        allocation_plan,
        rider_historical_data
    )
    
    return adjusted_plan

# 运力结构优化模型
def optimize_workforce_mix(demand_forecast, cost_params):
    """
    决策变量：
    x1: 专送骑手数量
    x2: 快送骑手数量
    x3: 众包骑手数量
    """
    
    model = OptimizationModel()
    
    # 目标函数：最小化总成本
    total_cost = (cost_params['zhuansong'] * x1 +
                 cost_params['kuaisong'] * x2 +
                 cost_params['zhongbao'] * x3)
    
    model.set_objective(minimize(total_cost))
    
    # 约束条件
    # 1. 满足各时段需求
    for t in time_periods:
        capacity = (efficiency['zhuansong'][t] * x1 +
                   efficiency['kuaisong'][t] * x2 +
                   efficiency['zhongbao'][t] * x3)
        model.add_constraint(capacity >= demand_forecast[t])
    
    # 2. 服务质量约束（专送比例不低于下限）
    model.add_constraint(x1 >= 0.3 * (x1 + x2 + x3))
    
    # 3. 管理跨度约束
    model.add_constraint(x1 <= max_fulltime_riders)
    model.add_constraint(x3 <= 0.4 * (x1 + x2 + x3))  # 众包不超40%
    
    # 4. 预算约束
    model.add_constraint(total_cost <= budget_limit)
    
    return model.solve()
```

## 5.4 区域划分与边界优化

科学的区域划分是提升配送效率的基础，需要综合考虑地理、需求、运力等多维因素。

### 5.4.1 多维度区域划分

区域划分不是简单的地理切分，而是多维优化问题：

```
┌─────────────────────────────────────────────────────────────┐
│                    区域划分决策框架                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入维度：                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   地理因素   │  │   需求因素   │  │   运力因素   │    │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤    │
│  │ • 道路网络   │  │ • 订单密度   │  │ • 骑手分布   │    │
│  │ • 自然边界   │  │ • 时段特征   │  │ • 运力类型   │    │
│  │ • 商圈分布   │  │ • 客单价     │  │ • 效率差异   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│           │                 │                 │             │
│           └─────────────────┼─────────────────┘             │
│                             ▼                               │
│                    ┌──────────────┐                        │
│                    │  聚类算法    │                        │
│                    │  (K-means+)  │                        │
│                    └──────────────┘                        │
│                             │                               │
│                             ▼                               │
│  输出结果：                                                 │
│  ┌────────────────────────────────────────────┐           │
│  │         优化后的配送区域                   │           │
│  │  ┌────┬────┬────┬────┐                   │           │
│  │  │ A1 │ A2 │ A3 │ A4 │  均衡度指标：     │           │
│  │  ├────┼────┼────┼────┤  • 订单量方差↓   │           │
│  │  │ B1 │ B2 │ B3 │ B4 │  • 配送距离↓     │           │
│  │  ├────┼────┼────┼────┤  • 运力利用率↑   │           │
│  │  │ C1 │ C2 │ C3 │ C4 │  • 跨区订单↓     │           │
│  │  └────┴────┴────┴────┘                   │           │
│  └────────────────────────────────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.4.2 动态边界调整机制

区域边界需要根据实时情况动态调整：

```python
# 动态边界优化算法
class DynamicBoundaryOptimizer:
    def __init__(self):
        self.boundary_history = []
        self.performance_metrics = {}
    
    def optimize_boundaries(self, current_state):
        """
        基于当前状态动态调整区域边界
        """
        # 1. 识别问题区域
        problem_zones = self.identify_problem_zones(current_state)
        
        # 2. 生成调整方案
        adjustment_proposals = []
        for zone in problem_zones:
            if zone['issue_type'] == 'overload':
                # 过载区域：缩小边界或分割
                proposal = self.generate_split_proposal(zone)
            elif zone['issue_type'] == 'underutilized':
                # 利用不足：扩大边界或合并
                proposal = self.generate_merge_proposal(zone)
            else:
                # 不均衡：重新划分
                proposal = self.generate_rebalance_proposal(zone)
            
            adjustment_proposals.append(proposal)
        
        # 3. 评估调整影响
        best_proposal = self.evaluate_proposals(adjustment_proposals)
        
        # 4. 执行渐进式调整
        return self.gradual_adjustment(best_proposal)
    
    def identify_problem_zones(self, state):
        """识别需要调整的问题区域"""
        problems = []
        
        for zone_id, metrics in state.items():
            # 计算关键指标
            utilization = metrics['assigned_orders'] / metrics['capacity']
            avg_distance = metrics['total_distance'] / metrics['order_count']
            delay_rate = metrics['delayed_orders'] / metrics['total_orders']
            
            # 判定问题类型
            if utilization > 0.95:
                problems.append({
                    'zone_id': zone_id,
                    'issue_type': 'overload',
                    'severity': utilization - 0.95
                })
            elif utilization < 0.6:
                problems.append({
                    'zone_id': zone_id,
                    'issue_type': 'underutilized',
                    'severity': 0.6 - utilization
                })
            elif delay_rate > 0.1:
                problems.append({
                    'zone_id': zone_id,
                    'issue_type': 'inefficient',
                    'severity': delay_rate
                })
        
        return sorted(problems, key=lambda x: x['severity'], reverse=True)
```

### 5.4.3 基于图论的区域优化

利用图论方法进行区域划分，确保连通性和紧凑性：

```python
# 基于最小割的区域划分
def graph_based_zone_partition(city_graph, num_zones):
    """
    使用谱聚类方法进行区域划分
    目标：最小化跨区流量，最大化区内连通性
    """
    
    # 构建邻接矩阵（基于订单流量）
    adjacency_matrix = build_flow_adjacency_matrix(city_graph)
    
    # 计算拉普拉斯矩阵
    degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
    laplacian = degree_matrix - adjacency_matrix
    
    # 归一化拉普拉斯矩阵
    normalized_laplacian = normalize_laplacian(laplacian)
    
    # 特征分解
    eigenvalues, eigenvectors = np.linalg.eig(normalized_laplacian)
    
    # 选择前k个最小特征值对应的特征向量
    k_smallest_eigenvectors = eigenvectors[:, :num_zones]
    
    # K-means聚类
    kmeans = KMeans(n_clusters=num_zones)
    zone_labels = kmeans.fit_predict(k_smallest_eigenvectors)
    
    # 后处理：确保连通性
    zones = ensure_connectivity(zone_labels, adjacency_matrix)
    
    # 边界平滑
    smoothed_zones = smooth_boundaries(zones, city_graph)
    
    return smoothed_zones
```

## 5.5 仿真系统与方案评估

仿真系统是规划引擎的"试验场"，通过模拟真实场景来评估不同规划方案的效果。

### 5.5.1 离散事件仿真架构

```
┌─────────────────────────────────────────────────────────────┐
│                    配送网络仿真系统架构                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  仿真引擎核心                                               │
│  ┌──────────────────────────────────────────┐             │
│  │           事件调度器 (Event Scheduler)     │             │
│  │  ┌────────────────────────────────────┐  │             │
│  │  │ 事件队列 (优先级队列)              │  │             │
│  │  │ • 订单生成事件                     │  │             │
│  │  │ • 骑手状态变更                     │  │             │
│  │  │ • 配送完成事件                     │  │             │
│  │  └────────────────────────────────────┘  │             │
│  └──────────────────────────────────────────┘             │
│                         │                                   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────┐             │
│  │           仿真实体 (Simulation Entities)  │             │
│  ├──────────────────────────────────────────┤             │
│  │  骑手Agent    订单Agent    商家Agent     │             │
│  │     │            │            │          │             │
│  │     └────────────┴────────────┘          │             │
│  │                  │                        │             │
│  │           状态转换模型                    │             │
│  └──────────────────────────────────────────┘             │
│                         │                                   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────┐             │
│  │         性能指标收集器 (Metrics Collector) │             │
│  │  • 平均配送时长  • 准时率               │             │
│  │  • 运力利用率    • 成本结构             │             │
│  │  • 服务覆盖率    • 异常率               │             │
│  └──────────────────────────────────────────┘             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.5.2 What-if 场景分析

仿真系统支持多种假设场景的对比分析：

```python
class ScenarioSimulator:
    def __init__(self, base_config):
        self.base_config = base_config
        self.scenarios = []
        
    def create_scenario(self, name, modifications):
        """创建假设场景"""
        scenario = deepcopy(self.base_config)
        
        # 应用场景修改
        for key, value in modifications.items():
            if key == 'demand_surge':
                # 需求激增场景
                scenario['order_rate'] *= value
            elif key == 'rider_shortage':
                # 运力短缺场景
                scenario['rider_count'] *= value
            elif key == 'bad_weather':
                # 恶劣天气场景
                scenario['delivery_speed'] *= value
                scenario['order_rate'] *= 1.2  # 需求增加
            elif key == 'new_station':
                # 新增站点场景
                scenario['stations'].append(value)
            
        self.scenarios.append({
            'name': name,
            'config': scenario
        })
        
    def run_comparison(self, duration=3600):
        """运行多场景对比"""
        results = {}
        
        for scenario in self.scenarios:
            # 初始化仿真环境
            env = SimulationEnvironment(scenario['config'])
            
            # 运行仿真
            env.run(duration)
            
            # 收集结果
            results[scenario['name']] = {
                'avg_delivery_time': env.get_avg_delivery_time(),
                'on_time_rate': env.get_on_time_rate(),
                'utilization': env.get_utilization_rate(),
                'cost_per_order': env.get_unit_cost(),
                'unserved_orders': env.get_unserved_count()
            }
        
        return self.generate_comparison_report(results)
    
    def sensitivity_analysis(self, parameter, value_range):
        """参数敏感性分析"""
        sensitivity_results = []
        
        for value in value_range:
            scenario = deepcopy(self.base_config)
            scenario[parameter] = value
            
            env = SimulationEnvironment(scenario)
            env.run(3600)
            
            sensitivity_results.append({
                parameter: value,
                'performance': env.get_key_metrics()
            })
        
        return self.plot_sensitivity_curve(sensitivity_results)
```

### 5.5.3 蒙特卡洛仿真优化

通过大量随机仿真寻找最优规划方案：

```python
def monte_carlo_optimization(objective_function, constraints, iterations=10000):
    """
    蒙特卡洛方法搜索最优配置
    """
    best_solution = None
    best_score = float('-inf')
    
    for i in range(iterations):
        # 随机生成候选方案
        candidate = generate_random_configuration()
        
        # 检查约束条件
        if not satisfy_constraints(candidate, constraints):
            continue
        
        # 运行多次仿真取平均（降低随机性影响）
        scores = []
        for _ in range(10):
            env = SimulationEnvironment(candidate)
            env.run(3600)
            score = objective_function(env.get_metrics())
            scores.append(score)
        
        avg_score = np.mean(scores)
        
        # 更新最优解
        if avg_score > best_score:
            best_score = avg_score
            best_solution = candidate
            
        # 自适应调整搜索范围
        if i % 1000 == 0:
            adjust_search_space(best_solution)
    
    return best_solution, best_score
```

## 5.6 规划与调度的协同

规划引擎与调度引擎需要紧密协同，形成"战略-战术"的联动机制。

### 5.6.1 分层决策框架

```
┌─────────────────────────────────────────────────────────────┐
│                 规划-调度协同决策框架                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  规划层（Planning Layer）                                   │
│  ┌────────────────────────────────────┐                   │
│  │  时间尺度：天/周/月                │                   │
│  │  决策内容：                        │                   │
│  │  • 站点布局                       │                   │
│  │  • 运力配置                       │                   │
│  │  • 服务边界                       │                   │
│  └─────────────┬──────────────────────┘                   │
│                │                                            │
│                │ 约束与指导                                │
│                ▼                                            │
│  ┌────────────────────────────────────┐                   │
│  │         协同接口层                 │                   │
│  │  • 容量上限  • 成本预算           │                   │
│  │  • 服务标准  • 覆盖要求           │                   │
│  └─────────────┬──────────────────────┘                   │
│                │                                            │
│                ▼                                            │
│  调度层（Scheduling Layer）                                │
│  ┌────────────────────────────────────┐                   │
│  │  时间尺度：秒/分钟                 │                   │
│  │  决策内容：                        │                   │
│  │  • 订单分配                       │                   │
│  │  • 路径规划                       │                   │
│  │  • 实时调整                       │                   │
│  └─────────────┬──────────────────────┘                   │
│                │                                            │
│                │ 反馈与学习                                │
│                ▼                                            │
│  ┌────────────────────────────────────┐                   │
│  │         执行效果评估                │                   │
│  │  • 实际vs计划  • 异常分析         │                   │
│  │  • 改进建议    • 参数调优         │                   │
│  └────────────────────────────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.6.2 双向反馈机制

```python
class PlanningSchedulingCoordinator:
    def __init__(self):
        self.planning_engine = PlanningEngine()
        self.scheduling_engine = SchedulingEngine()
        self.feedback_buffer = []
        
    def coordinate(self):
        """规划与调度的协同流程"""
        
        # 1. 规划层生成指导方案
        planning_decision = self.planning_engine.generate_plan()
        
        # 2. 转换为调度约束
        scheduling_constraints = self.translate_to_constraints(planning_decision)
        
        # 3. 调度层在约束下优化
        scheduling_result = self.scheduling_engine.optimize(
            constraints=scheduling_constraints
        )
        
        # 4. 收集执行反馈
        execution_feedback = self.collect_feedback(scheduling_result)
        
        # 5. 反馈给规划层
        self.planning_engine.update_model(execution_feedback)
        
        return scheduling_result
    
    def translate_to_constraints(self, planning_decision):
        """将规划决策转换为调度约束"""
        constraints = {
            'max_delivery_distance': planning_decision['zone_radius'],
            'min_rider_per_zone': planning_decision['min_capacity'],
            'service_level_target': planning_decision['sla_target'],
            'cost_budget': planning_decision['unit_cost_limit']
        }
        
        # 动态调整约束松紧度
        if self.is_peak_hour():
            constraints['service_level_target'] *= 0.95  # 高峰期适当放松
            
        return constraints
    
    def adaptive_learning(self):
        """基于历史反馈的自适应学习"""
        if len(self.feedback_buffer) >= 100:
            # 分析规划与实际的偏差
            planning_accuracy = self.analyze_planning_accuracy()
            
            # 调整规划参数
            if planning_accuracy < 0.8:
                self.planning_engine.adjust_parameters({
                    'demand_forecast_weight': 1.2,
                    'safety_margin': 1.15,
                    'response_speed': 0.9
                })
            
            # 清理旧数据
            self.feedback_buffer = self.feedback_buffer[-50:]
```

### 5.6.3 联合优化模型

规划与调度的联合优化可以获得全局最优解：

```python
def joint_optimization_model():
    """
    规划-调度联合优化模型
    同时考虑长期规划和短期调度
    """
    
    model = MixedIntegerProgram()
    
    # 决策变量
    # 规划层变量
    y_station = {}  # 站点开设决策
    z_zone = {}     # 区域划分决策
    w_rider = {}    # 运力配置决策
    
    # 调度层变量
    x_assign = {}   # 订单分配决策
    r_route = {}    # 路径选择决策
    
    # 目标函数：最小化总成本
    total_cost = (
        sum(station_cost[i] * y_station[i] for i in stations) +  # 站点成本
        sum(rider_cost[j] * w_rider[j] for j in rider_types) +   # 运力成本
        sum(delivery_cost[k] * x_assign[k] for k in orders)      # 配送成本
    )
    
    model.set_objective(minimize(total_cost))
    
    # 约束条件
    # 1. 规划层约束
    for zone in zones:
        # 每个区域至少一个站点
        model.add_constraint(sum(y_station[s] for s in zone.stations) >= 1)
        
        # 运力容量约束
        model.add_constraint(
            sum(w_rider[r] * capacity[r] for r in zone.riders) >= 
            zone.expected_demand * 1.2  # 20%缓冲
        )
    
    # 2. 调度层约束
    for order in orders:
        # 每个订单必须被分配
        model.add_constraint(sum(x_assign[order, rider] for rider in riders) == 1)
        
        # 时间窗约束
        model.add_constraint(
            delivery_time[order] <= order.promised_time
        )
    
    # 3. 耦合约束（规划影响调度）
    for order, rider in order_rider_pairs:
        # 只能从开放的站点派送
        station = get_rider_station(rider)
        model.add_constraint(x_assign[order, rider] <= y_station[station])
        
        # 配送距离受区域划分影响
        model.add_constraint(
            distance[order, rider] <= max_zone_radius * z_zone[order.zone]
        )
    
    return model.solve()
```

## 本章小结

规划引擎是美团外卖系统的"大脑"，通过战略性的结构优化为整个系统奠定效率基础。本章我们学习了：

**核心概念**：
- **三层规划体系**：战略规划（月度）、战术规划（周度）、操作规划（小时级），不同层级解决不同时间尺度的问题
- **设施选址问题**：经典的NP-hard问题，需要在成本和服务质量间平衡
- **运力结构设计**：专送、快送、众包的组合优化，实现成本与稳定性的平衡
- **区域划分优化**：基于图论和聚类的方法，最小化跨区流量，最大化区内效率
- **仿真评估系统**：通过离散事件仿真和蒙特卡洛方法评估规划方案

**关键公式**：

1. **设施选址目标函数**：
   ```
   min Σᵢ fᵢ·yᵢ + ΣᵢΣⱼ cᵢⱼ·xᵢⱼ
   ```
   其中fᵢ是固定成本，cᵢⱼ是配送成本

2. **运力配比优化**：
   ```
   专送:快送:众包 = 40%:35%:25% (典型配比)
   ```

3. **区域划分评价指标**：
   ```
   均衡度 = 1 - σ(zone_loads) / μ(zone_loads)
   ```

4. **容量规划公式**：
   ```
   容量 = 基础容量 × 时段系数 × 季节系数 × 天气系数 × 溢出修正
   ```

**实践要点**：
- 规划决策影响深远，需要充分的数据支撑和仿真验证
- 多目标优化常常没有唯一最优解，需要权衡取舍
- 动态调整机制很重要，要能根据实际运行情况优化规划
- 规划与调度的协同是关键，需要建立双向反馈机制

## 常见陷阱与错误（Gotchas）

### 1. 过度优化陷阱
**问题**：追求数学上的最优解，忽视实际可执行性
```python
# ❌ 错误做法
optimal_zones = solve_perfect_partition(city_graph)  # 完美但不实际

# ✅ 正确做法
practical_zones = solve_with_constraints(
    city_graph,
    min_zone_size=1000,  # 最小管理规模
    max_shape_irregularity=0.3,  # 形状规整性
    respect_natural_boundaries=True  # 尊重自然边界
)
```

### 2. 静态规划误区
**问题**：一次性规划后不再调整，无法适应变化
```python
# ❌ 错误做法
annual_plan = create_yearly_plan()
execute_blindly(annual_plan)  # 全年不变

# ✅ 正确做法
adaptive_plan = create_adaptive_plan()
for month in year:
    adjusted_plan = adaptive_plan.adjust(
        current_performance,
        market_changes,
        seasonal_factors
    )
    execute_with_monitoring(adjusted_plan)
```

### 3. 孤立优化问题
**问题**：各模块独立优化，忽视相互影响
```python
# ❌ 错误做法
station_plan = optimize_stations()
zone_plan = optimize_zones()
rider_plan = optimize_riders()
# 三个计划可能相互冲突

# ✅ 正确做法
integrated_plan = joint_optimization(
    station_variables,
    zone_variables,
    rider_variables,
    coupling_constraints
)
```

### 4. 仿真偏差风险
**问题**：仿真模型过度简化，结果与实际偏差大
```python
# ❌ 错误做法
simple_sim = BasicSimulation()
decision = make_decision(simple_sim.run())  # 过度信任简化模型

# ✅ 正确做法
validated_sim = CalibratedSimulation()
validated_sim.calibrate_with_historical_data()
results = validated_sim.run_multiple_scenarios()
decision = make_robust_decision(
    results,
    confidence_interval=0.95,
    worst_case_analysis=True
)
```

### 5. 成本估算偏差
**问题**：低估隐性成本，导致规划失真
```python
# ❌ 错误做法
cost = rent + salary  # 只考虑显性成本

# ✅ 正确做法
total_cost = (
    rent +
    salary +
    training_cost +
    turnover_cost +
    management_overhead +
    equipment_depreciation +
    opportunity_cost
)
```

### 6. 边界硬切割问题
**问题**：区域边界过于僵硬，导致边界订单处理困难
```python
# ❌ 错误做法
if order.location in zone_a:
    assign_to_zone_a_rider()  # 即使zone_b骑手更近

# ✅ 正确做法
nearby_zones = get_adjacent_zones(order.location)
best_rider = find_best_rider_across_zones(
    nearby_zones,
    allow_cross_zone=True,
    max_extra_distance=500  # 允许500米的跨区
)
```

## 练习题

### 基础题

#### 练习 5.1：站点选址的贪心算法
设计一个贪心算法来解决简化版的站点选址问题。给定n个候选站点位置和m个客户位置，每个站点有固定成本fᵢ和容量Cᵢ，每个客户有需求dⱼ。请选择k个站点使得总成本（固定成本+配送成本）最小。

**Hint**：考虑每个站点的"性价比"，即覆盖需求量与成本的比值。

<details>
<summary>参考答案</summary>

贪心算法步骤：
1. 计算每个候选站点的覆盖效率：efficiency[i] = Σⱼ(可覆盖的需求dⱼ) / (fᵢ + 平均配送成本)
2. 按效率降序排序候选站点
3. 依次选择效率最高的站点，直到：
   - 已选择k个站点，或
   - 所有需求已被满足
4. 对于每个客户，分配给最近的已选站点（考虑容量约束）

时间复杂度：O(n×m + nlogn + k×m)
空间复杂度：O(n+m)

注意：贪心算法不保证全局最优，但在实践中能快速得到较好的解。
</details>

#### 练习 5.2：运力配比计算
某配送站覆盖区域日均订单量1000单，订单时段分布为：早高峰(10-12点)占20%，午高峰(12-14点)占40%，晚高峰(18-20点)占30%，其他时段占10%。已知专送骑手每小时可送4单，快送3单，众包2.5单。专送成本6元/单，快送5元/单，众包4.5元/单。请设计最优运力配比。

**Hint**：建立线性规划模型，考虑峰值需求和成本约束。

<details>
<summary>参考答案</summary>

解题步骤：
1. 计算峰值需求：
   - 午高峰：400单/2小时 = 200单/小时
   - 需要运力：200/效率 ≈ 50-80人（根据运力类型）

2. 建立优化模型：
   - 决策变量：x₁(专送), x₂(快送), x₃(众包)
   - 目标：min 6×1000×p₁ + 5×1000×p₂ + 4.5×1000×p₃
   - 约束：
     * 4x₁ + 3x₂ + 2.5x₃ ≥ 200 (峰值容量)
     * x₁ ≥ 0.3(x₁+x₂+x₃) (服务质量)
     * x₃ ≤ 0.3(x₁+x₂+x₃) (管理约束)

3. 求解结果：
   - 专送：25人 (31%)
   - 快送：35人 (44%)
   - 众包：20人 (25%)
   - 日成本：约5250元
</details>

#### 练习 5.3：区域边界调整
某区域当前划分为4个配送区，各区订单量和运力如下：
- A区：订单200单/小时，运力40人
- B区：订单150单/小时，运力35人
- C区：订单180单/小时，运力30人
- D区：订单120单/小时，运力30人

假设每个骑手每小时可配送4单，请分析哪些区域需要调整边界，并给出调整建议。

**Hint**：计算各区域的运力利用率，识别失衡区域。

<details>
<summary>参考答案</summary>

分析过程：
1. 计算运力利用率：
   - A区：200/(40×4) = 125% (严重过载)
   - B区：150/(35×4) = 107% (轻微过载)
   - C区：180/(30×4) = 150% (严重过载)
   - D区：120/(30×4) = 100% (满载)

2. 识别问题：
   - C区最严重，需要立即调整
   - A区次之，需要支援
   - D区刚好平衡，可作为缓冲

3. 调整建议：
   - 将C区靠近D区的部分订单（约30单/小时）划给D区
   - 将A区靠近B区的部分订单（约20单/小时）划给B区
   - 考虑增加C区和A区的运力配置
   - 设置弹性边界，允许高峰期跨区支援
</details>

### 挑战题

#### 练习 5.4：多目标站点优化
设计一个算法同时优化三个目标：(1)最小化总成本，(2)最大化服务覆盖率，(3)最小化平均配送距离。使用Pareto最优的概念，给出求解思路。

**Hint**：考虑使用NSGA-II等多目标优化算法。

<details>
<summary>参考答案</summary>

算法设计：
1. 问题建模：
   - 决策变量：站点位置向量X = [x₁, x₂, ..., xₖ]
   - 目标函数：
     * f₁(X) = Σ(固定成本 + 运营成本)
     * f₂(X) = -覆盖客户数/总客户数
     * f₃(X) = Σ(配送距离×订单量)/总订单量

2. NSGA-II算法流程：
   - 初始化：随机生成N个解
   - 迭代优化：
     * 非支配排序：找出Pareto前沿
     * 拥挤度计算：保持解的多样性
     * 选择、交叉、变异：生成新解
     * 精英保留：保留最优解

3. 决策支持：
   - 生成Pareto前沿的3D可视化
   - 提供交互式权重调整
   - 基于业务优先级推荐折中方案

4. 实践考虑：
   - 加入地理约束（不可建站区域）
   - 考虑竞争对手站点位置
   - 预留未来扩展空间
</details>

#### 练习 5.5：仿真系统设计
设计一个配送网络仿真系统的核心架构，要求支持：(1)离散事件仿真，(2)多场景对比，(3)实时可视化。给出主要类的设计和事件处理流程。

**Hint**：使用事件驱动架构，参考SimPy等仿真框架。

<details>
<summary>参考答案</summary>

系统架构设计：

1. 核心类设计：
```python
class SimulationEngine:
    - event_queue: PriorityQueue
    - entities: Dict[Entity]
    - clock: SimulationClock
    - metrics_collector: MetricsCollector
    
class Entity(ABC):
    - id: str
    - state: EntityState
    - handle_event(event): void
    
class Order(Entity):
    - create_time: float
    - pickup_location: Location
    - delivery_location: Location
    - status: OrderStatus
    
class Rider(Entity):
    - capacity: int
    - speed: float
    - current_orders: List[Order]
    - location: Location
    
class Station(Entity):
    - location: Location
    - riders: List[Rider]
    - coverage_area: Polygon
```

2. 事件处理流程：
   - 事件类型：OrderCreated, RiderAssigned, PickupCompleted, DeliveryCompleted
   - 处理循环：
     * 从优先队列取出最早事件
     * 更新仿真时钟
     * 调用相关实体的handle_event
     * 生成后续事件
     * 收集性能指标

3. 多场景支持：
   - ScenarioManager管理不同配置
   - 并行运行多个仿真实例
   - 结果聚合和对比分析

4. 可视化设计：
   - 实时地图展示（WebSocket推送）
   - 关键指标仪表板
   - 事件日志和回放功能
</details>

#### 练习 5.6：规划-调度协同优化
设计一个机制实现规划层和调度层的双向信息交互和协同优化。要求：(1)规划层能指导调度，(2)调度层反馈能改进规划，(3)支持在线学习。

**Hint**：考虑使用强化学习或贝叶斯优化方法。

<details>
<summary>参考答案</summary>

协同机制设计：

1. 信息流设计：
   - 下行（规划→调度）：
     * 容量约束、成本预算
     * 服务水平目标
     * 区域划分方案
   - 上行（调度→规划）：
     * 实际执行指标
     * 异常事件统计
     * 资源利用率

2. 协同优化算法：
```python
class AdaptiveCoordinator:
    def __init__(self):
        self.planning_model = BayesianOptimizer()
        self.scheduling_model = RLAgent()
        self.feedback_buffer = CircularBuffer(1000)
    
    def coordinate_decision(self):
        # 规划层决策
        plan_params = self.planning_model.suggest()
        
        # 转换为调度约束
        constraints = self.translate_constraints(plan_params)
        
        # 调度层执行
        schedule = self.scheduling_model.act(constraints)
        
        # 收集反馈
        performance = self.evaluate_performance(schedule)
        
        # 更新模型
        self.planning_model.update(plan_params, performance)
        self.scheduling_model.learn(performance)
        
        return schedule
```

3. 在线学习机制：
   - 使用Thompson采样平衡探索与利用
   - 维护不确定性估计
   - 渐进式参数调整
   - 异常检测和快速适应

4. 实施要点：
   - 设置安全边界防止激进调整
   - 保留人工干预接口
   - A/B测试验证改进效果
   - 定期离线重训练
</details>

#### 练习 5.7：动态容量规划
某站点需要应对季节性和突发性需求变化。设计一个动态容量规划算法，要求能够：(1)预测未来需求，(2)优化人员调度，(3)处理突发事件。

**Hint**：结合时间序列预测和鲁棒优化。

<details>
<summary>参考答案</summary>

算法设计：

1. 需求预测模块：
   - 使用SARIMA处理季节性
   - LSTM捕捉复杂模式
   - 集成多个模型降低风险
   - 置信区间估计

2. 鲁棒优化模型：
```python
def robust_capacity_planning(demand_forecast, uncertainty_set):
    model = RobustOptimizationModel()
    
    # 决策变量
    regular_staff = model.add_var('regular', lower=0)
    flexible_staff = model.add_var('flexible', lower=0)
    contingency = model.add_var('contingency', lower=0)
    
    # 鲁棒约束（最坏情况下仍满足需求）
    for scenario in uncertainty_set:
        capacity = regular_staff * 4 + flexible_staff * 3
        model.add_constraint(
            capacity + contingency >= 
            demand_forecast * (1 + scenario['demand_surge'])
        )
    
    # 目标：最小化期望成本
    expected_cost = (
        regular_staff * 1000 +
        flexible_staff * 800 +
        contingency * 500 * prob_emergency
    )
    model.minimize(expected_cost)
    
    return model.solve()
```

3. 突发事件处理：
   - 实时监控触发条件
   - 分级响应机制
   - 跨区域资源调配
   - 动态激励发放

4. 实施策略：
   - 滚动规划（每周更新）
   - 场景库维护
   - 历史案例学习
   - 弹性资源池建设
</details>

#### 练习 5.8：全局优化框架
设计一个能够同时优化站点布局、区域划分、运力配置的全局优化框架。要求考虑各决策变量之间的相互影响，并给出求解策略。

**Hint**：可以使用分解协调方法或元启发式算法。

<details>
<summary>参考答案</summary>

全局优化框架设计：

1. 问题分解（Benders分解）：
   - 主问题：站点选址和区域划分
   - 子问题：运力配置和路径优化
   - 迭代协调：通过割平面传递信息

2. 分层优化算法：
```python
class HierarchicalOptimizer:
    def optimize_global(self):
        # Level 1: 战略决策（站点布局）
        station_layout = self.optimize_stations(
            demand_forecast,
            available_locations,
            budget_constraint
        )
        
        # Level 2: 战术决策（区域划分）
        zone_partition = self.optimize_zones(
            station_layout,
            flow_matrix,
            balance_requirement
        )
        
        # Level 3: 操作决策（运力配置）
        workforce_plan = self.optimize_workforce(
            zone_partition,
            time_varying_demand,
            service_level
        )
        
        # 反馈优化
        if not self.is_feasible(workforce_plan):
            # 添加可行性割
            self.add_feasibility_cut()
            return self.optimize_global()  # 递归
        
        return station_layout, zone_partition, workforce_plan
```

3. 元启发式方法（遗传算法）：
   - 染色体编码：[站点基因|区域基因|运力基因]
   - 适应度函数：weighted_sum(成本, 服务质量, 效率)
   - 特殊算子：
     * 修复算子（保证可行性）
     * 局部搜索（改进质量）
     * 自适应变异（避免早熟）

4. 并行计算策略：
   - 种群并行评估
   - 多目标分解并行
   - GPU加速距离计算
   - 分布式仿真验证

5. 实践建议：
   - 使用热启动（历史最优解）
   - 设置时间限制
   - 记录中间解供人工选择
   - 增量式优化（逐步改进）
</details>