# 第7章：LBS系统（地图/地址库/路径规划）

## 章节大纲

### 7.1 系统架构概览
- LBS在超脑系统中的定位
- 核心能力矩阵
- 性能指标要求
- 与其他模块的接口设计

### 7.2 五级地址库体系
- 地址结构化设计
- 地址解析与标准化
- 楼宇/楼层级精度实现
- 地址纠错与补全

### 7.3 地理围栏与POI管理
- 商家地理围栏设计
- 配送范围动态调整
- POI数据采集与更新
- 围栏计算优化

### 7.4 路径规划与导航
- 多模式路径规划（步行/骑行/电动车）
- 实时路况融合
- 室内导航方案
- 路径偏离检测与重规划

### 7.5 高并发架构设计
- 400万QPS的架构挑战
- 多级缓存策略
- 服务降级与熔断
- 地理索引优化

### 7.6 轨迹处理与分析
- 实时轨迹清洗
- 轨迹压缩与存储
- 异常轨迹识别
- 历史轨迹挖掘

### 7.7 与调度/ETA的协同
- 距离矩阵预计算
- 路网约束建模
- 实时路况反馈
- 精度与性能平衡

### 本章小结
### 练习题
### 常见陷阱与错误

---

## 学习目标

本章结束后，你将能够：

1. **理解LBS系统架构**：掌握支撑百万级骑手的地理信息系统设计
2. **构建地址体系**：设计精确到楼层的五级地址库
3. **实现路径规划**：理解多约束条件下的实时路径计算
4. **优化系统性能**：处理400万QPS的地理查询请求
5. **设计围栏系统**：实现高效的地理围栏判定算法

---

## 引言

想象一下，在晚高峰的北京国贸，数千名骑手同时在配送，每秒都有上万次路径查询请求。一个骑手站在写字楼下，需要知道去18层取餐要多久；另一个骑手正在规划去3公里外最快的路线；调度系统需要实时计算10000个骑手到5000个商家的距离矩阵...这就是美团LBS系统每天面对的挑战。

LBS（Location Based Service）系统是超脑的"眼睛"和"地图"，它将真实世界的地理空间映射为可计算的数字世界。没有精准的地理信息，再智能的调度算法也无法落地；没有高效的路径规划，再优秀的时间预估也会失准。本章将深入剖析这个日调用量达760亿次的地理信息引擎。

## 7.1 系统架构概览

### 7.1.1 LBS在超脑系统中的定位

LBS系统在整个超脑架构中扮演着基础设施的角色，它是连接物理世界与数字世界的桥梁：

```
┌─────────────────────────────────────────────────────────┐
│                    业务决策层                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │调度引擎 │  │ETA系统  │  │定价系统 │  │规划引擎 │  │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  │
│       │            │            │            │         │
│       └────────────┴────────────┴────────────┘         │
│                         │                              │
│                         ▼                              │
│               ┌──────────────────┐                     │
│               │   LBS服务层      │                     │
│               └──────────────────┘                     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    LBS核心能力                          │
├─────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │地址服务  │  │路径规划  │  │地理围栏  │            │
│  ├──────────┤  ├──────────┤  ├──────────┤            │
│  │·地址解析 │  │·多模式   │  │·围栏判定 │            │
│  │·地址纠错 │  │·实时路况 │  │·范围查询 │            │
│  │·POI管理  │  │·室内导航 │  │·区域管理 │            │
│  └──────────┘  └──────────┘  └──────────┘            │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │距离计算  │  │轨迹服务  │  │地图渲染  │            │
│  ├──────────┤  ├──────────┤  ├──────────┤            │
│  │·直线距离 │  │·轨迹存储 │  │·热力图   │            │
│  │·路网距离 │  │·轨迹纠偏 │  │·覆盖图   │            │
│  │·时间距离 │  │·轨迹分析 │  │·实时位置 │            │
│  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────┘
```

### 7.1.2 核心能力矩阵

LBS系统需要提供的核心能力可以用一个能力矩阵来描述：

| 能力维度 | 基础能力 | 进阶能力 | 智能化能力 |
|---------|---------|---------|-----------|
| **地址** | 地址解析、标准化 | 模糊匹配、纠错 | 语义理解、自动补全 |
| **路径** | 最短路径 | 多约束路径、实时路况 | 个性化路径、预测规划 |
| **围栏** | 点面判定 | 动态围栏、复杂多边形 | 智能扩缩、自动优化 |
| **距离** | 直线距离 | 路网距离、步行距离 | 时间距离、成本距离 |
| **轨迹** | 轨迹存储 | 轨迹压缩、纠偏 | 异常检测、模式挖掘 |

### 7.1.3 性能指标要求

根据公开数据，美团LBS系统需要支撑的性能指标令人震撼：

```
性能指标金字塔
     ╱╲
    ╱  ╲     峰值QPS: 400万
   ╱    ╲    
  ╱──────╲   日调用量: 760亿
 ╱        ╲  
╱──────────╲ 每小时路径规划: 29亿

关键延迟要求：
- 地址解析: < 5ms (P99)
- 路径规划: < 10ms (P99) 
- 围栏判定: < 1ms (P99)
- 距离计算: < 1ms (P99)
```

### 7.1.4 与其他模块的接口设计

LBS系统作为基础服务，需要为上层业务提供统一、稳定、高效的接口：

```
接口设计原则：
1. 统一协议：所有接口采用统一的RPC协议（如Thrift/gRPC）
2. 批量优化：支持批量查询，减少网络开销
3. 缓存友好：接口设计考虑缓存命中率
4. 降级方案：每个接口都有降级策略
5. 监控完备：接口级别的监控和报警

核心接口示例：

// 地址解析接口
service AddressService {
    // 批量地址解析
    AddressResult batchParse(List<string> addresses);
    
    // 地址标准化
    StandardAddress standardize(RawAddress addr);
    
    // POI查询
    List<POI> searchPOI(Location center, int radius);
}

// 路径规划接口  
service RouteService {
    // 单点到单点路径
    Route planRoute(Location from, Location to, RouteOptions options);
    
    // 多点路径规划（TSP问题）
    MultiRoute planMultiRoute(List<Location> points, Constraints constraints);
    
    // 距离矩阵计算
    DistanceMatrix calcDistanceMatrix(List<Location> origins, List<Location> destinations);
}

// 围栏服务接口
service GeofenceService {
    // 判定点是否在围栏内
    bool contains(Location point, GeofenceId fence);
    
    // 查询点所在的所有围栏
    List<GeofenceId> queryFences(Location point);
    
    // 范围查询
    List<POI> rangeQuery(Location center, Shape shape, Filter filter);
}

## 7.2 五级地址库体系

### 7.2.1 地址结构化设计

美团的五级地址库是精确配送的基石。传统地址系统只能定位到小区或大楼，而外卖配送需要精确到楼层甚至房间。五级地址体系的设计理念：

```
五级地址层次结构
┌─────────────────────────────────────────────────────┐
│                    Level 1: 城市                     │
│                   (City: 北京市)                     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                    Level 2: 区域                     │
│                 (District: 朝阳区)                   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                    Level 3: 道路/小区                │
│            (Road/Community: 建国路88号)              │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                    Level 4: 楼宇                     │
│              (Building: SOHO现代城A座)               │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                    Level 5: 楼层/门牌                │
│               (Floor/Room: 18层1801室)               │
└─────────────────────────────────────────────────────┘

数据结构设计：
{
    "addressId": "110105001234567890",  // 18位唯一ID
    "level1": {
        "code": "110000",
        "name": "北京市",
        "center": [116.405285, 39.904989],
        "boundary": {...}  // GeoJSON格式边界
    },
    "level2": {
        "code": "110105", 
        "name": "朝阳区",
        "parentCode": "110000"
    },
    "level3": {
        "code": "110105001",
        "name": "建国路",
        "number": "88号",
        "type": "ROAD"  // ROAD/COMMUNITY/MALL
    },
    "level4": {
        "code": "110105001001",
        "name": "SOHO现代城A座",
        "entrances": [  // 多入口信息
            {"type": "MAIN", "location": [116.4753, 39.9042]},
            {"type": "SIDE", "location": [116.4755, 39.9041]}
        ],
        "elevators": 6,  // 电梯数量
        "floors": 28     // 总楼层
    },
    "level5": {
        "code": "110105001001018001",
        "floor": 18,
        "room": "1801",
        "walkTime": 120  // 从主入口步行时间(秒)
    }
}
```

### 7.2.2 地址解析与标准化

用户输入的地址往往是非结构化的自然语言，需要强大的解析能力：

```
地址解析流程：

原始输入: "朝阳建国路soho现代城a座18楼美团外卖"
                    │
                    ▼
        ┌──────────────────────┐
        │    1. 预处理清洗      │
        │  去除特殊字符、统一大小写│
        └───────────┬──────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │    2. 分词与标注      │
        │  使用NER识别地址要素   │
        └───────────┬──────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │    3. 地址要素抽取    │
        │  区域/道路/楼宇/楼层   │
        └───────────┬──────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │    4. 候选地址匹配    │
        │  模糊匹配+相似度计算   │
        └───────────┬──────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │    5. 歧义消解       │
        │  上下文+历史订单辅助  │
        └───────────┬──────────┘
                    │
                    ▼
标准化输出: {
    "formatted": "北京市朝阳区建国路88号SOHO现代城A座18层",
    "addressId": "110105001001018000",
    "confidence": 0.95,
    "alternates": [...]  // 备选地址
}

关键技术点：
1. 分词技术：CRF/LSTM-CRF模型
2. 地址要素识别：BiLSTM+Attention
3. 相似度计算：编辑距离+语义相似度
4. 缩写展开：soho→SOHO现代城
5. 别名映射：国贸→中国国际贸易中心
```

### 7.2.3 楼宇/楼层级精度实现

实现楼层级精度需要解决室内定位和时间估算问题：

```
楼层时间模型：

楼层配送时间 = 电梯等待时间 + 电梯运行时间 + 步行时间

1. 电梯等待时间建模：
   - 基于时段的泊松分布
   - 考虑楼层热度（如1楼、餐饮楼层）
   - 高峰期动态调整

2. 电梯运行时间：
   - 楼层差 × 单层时间
   - 考虑停靠概率
   - 区分直达和站站停

3. 步行时间网格：
   楼层内步行时间热力图
   ┌─────────────────────┐
   │ 30  35  40  45  50  │  秒
   │ 25  20  25  30  35  │
   │ 20  15  10  15  20  │  
   │ 25  20  15  20  25  │
   │ 30  25  20  25  30  │
   └─────────────────────┘
   基于历史轨迹学习每个区域的步行时间

楼宇画像系统：
{
    "buildingId": "B000A6GQ61",
    "profile": {
        "type": "OFFICE",  // OFFICE/RESIDENTIAL/MALL
        "floors": 28,
        "elevators": {
            "count": 6,
            "speed": 2.5,  // m/s
            "avgWaitTime": {
                "morning": 45,   // 秒
                "noon": 120,
                "evening": 60
            }
        },
        "security": {
            "needRegister": true,
            "avgTime": 30  // 登记时间
        },
        "peakHours": ["11:30-13:00", "18:00-19:30"],
        "specialNotes": "负一层美食广场，18-22层为美团办公区"
    }
}
```

### 7.2.4 地址纠错与补全

基于海量历史数据的地址智能纠错：

```
纠错策略层次：

1. 字符级纠错（拼写错误）：
   "建国陆" → "建国路"
   使用编辑距离 + 拼音相似度

2. 语义级纠错（理解错误）：
   "soho三期" → "SOHO现代城"
   基于知识图谱的实体链接

3. 上下文纠错（关联推理）：
   用户历史: ["朝阳SOHO", "建外SOHO", "？？SOHO"]
   商家位置: 建国路
   推理结果: "现代城SOHO"

4. 协同过滤纠错：
   相似用户的地址模式
   相同商圈的地址分布

地址补全决策树：
                 输入片段
                    │
          ┌─────────┴─────────┐
          │                   │
      有历史订单           无历史订单
          │                   │
    ┌─────┴─────┐       ┌─────┴─────┐
    │           │       │           │
  同商家     不同商家   热门地址   冷门地址
    │           │       │           │
 优先推荐    参考推荐  通用推荐   模糊匹配
```

地址纠错的核心挑战在于平衡准确性和用户体验。过度纠正可能导致错误的地址替换，而纠正不足又会增加配送难度。美团的解决方案是构建多层次的置信度体系：

**置信度分层策略**：
- **高置信度（>0.95）**：直接自动纠正，如常见拼写错误
- **中置信度（0.7-0.95）**：提示用户确认，展示纠正建议
- **低置信度（<0.7）**：保留原始输入，通过骑手端反馈学习

**纠错知识库构建**：系统维护了一个包含千万级别名、简称、错别字的映射库。这个知识库通过以下方式持续更新：
1. 骑手反馈的地址修正
2. 用户主动的地址纠正
3. OCR识别的门牌照片
4. 商家上传的位置信息

**实时学习机制**：当某个地址纠错被多次确认后，系统会自动提升该纠错规则的权重，实现自我进化。

## 7.3 地理围栏与POI管理

### 7.3.1 商家地理围栏设计

地理围栏（Geofence）是LBS系统的核心概念，它定义了商家的配送范围、骑手的工作区域、以及各种地理相关的业务规则：

```
围栏类型层次结构：

┌─────────────────────────────────────────────────────┐
│                    围栏类型分类                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  业务围栏                   物理围栏                 │
│  ├─ 配送围栏               ├─ 行政区划            │
│  ├─ 营销围栏               ├─ 商圈边界            │
│  ├─ 价格围栏               ├─ 道路网格            │
│  └─ 服务围栏               └─ 建筑轮廓            │
│                                                      │
│  动态围栏                   静态围栏                 │
│  ├─ 天气围栏               ├─ 基础配送范围        │
│  ├─ 时段围栏               ├─ 禁行区域            │
│  └─ 运力围栏               └─ 特殊区域            │
└─────────────────────────────────────────────────────┘

商家配送围栏数据结构：
{
    "merchantId": "M1234567",
    "fences": [
        {
            "fenceId": "F001",
            "type": "DELIVERY",
            "priority": 1,
            "shape": {
                "type": "POLYGON",
                "coordinates": [
                    [[116.4741, 39.9041], [116.4751, 39.9041], ...]
                ]
            },
            "properties": {
                "minOrder": 20,      // 起送价
                "deliveryFee": 5,    // 配送费
                "maxDistance": 3000,  // 最大配送距离(米)
                "estimatedTime": 30   // 预估配送时间(分)
            },
            "timeRules": [
                {
                    "dayOfWeek": [1,2,3,4,5],  // 周一到周五
                    "timeRange": "11:00-14:00",
                    "adjustment": {
                        "deliveryFee": 3,  // 高峰期加价
                        "estimatedTime": 45
                    }
                }
            ],
            "excludeAreas": [  // 排除区域（如湖泊、禁区）
                {
                    "type": "CIRCLE",
                    "center": [116.4745, 39.9045],
                    "radius": 100
                }
            ]
        }
    ]
}
```

**围栏计算优化技术**：

美团每天需要处理数亿次的围栏判定请求，传统的点与多边形判定算法（如射线法）在如此规模下会成为性能瓶颈。优化方案包括：

1. **空间索引**：使用R-Tree或Geohash进行空间索引，快速过滤候选围栏
2. **围栏简化**：使用Douglas-Peucker算法简化多边形，减少判定计算量
3. **分层判定**：先用外接矩形快速过滤，再做精确的多边形判定
4. **缓存策略**：对热点区域的判定结果进行缓存
5. **并行计算**：利用SIMD指令集并行处理多个点的判定

### 7.3.2 配送范围动态调整

配送范围不是一成不变的，需要根据实时运力、天气、交通等因素动态调整：

```
动态调整决策流程：

                 实时信号输入
    ┌────────────────┼────────────────┐
    │                │                │
运力状态          天气状况         交通状况
    │                │                │
    ▼                ▼                ▼
运力热力图      恶劣天气区域      拥堵路段
    │                │                │
    └────────────────┼────────────────┘
                     │
              ┌──────▼──────┐
              │ 决策引擎    │
              └──────┬──────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
缩小范围          维持现状         扩大范围
(运力不足)        (供需平衡)       (运力充足)
```

**动态调整策略**：

1. **运力驱动调整**：
   - 当区域内可用骑手数 < 阈值时，缩小配送范围
   - 优先保证核心区域的配送质量
   - 通过价格杠杆引导运力流动

2. **天气响应机制**：
   - 雨雪天气自动缩小15-30%配送范围
   - 极端天气启动应急围栏方案
   - 为骑手安全设置禁行区域

3. **需求预测调整**：
   - 基于历史数据预测未来30分钟订单分布
   - 提前调整围栏以应对需求高峰
   - 动态平衡不同商圈的运力分配

4. **实时反馈优化**：
   - 监控超时率、拒单率等指标
   - 当指标异常时自动触发围栏调整
   - A/B测试不同调整策略的效果

### 7.3.3 POI管理

POI（Point of Interest）是地理信息系统的基础数据，美团维护着数千万个POI：

```
POI分类体系：

一级分类          二级分类            三级分类
├─ 餐饮          ├─ 中餐            ├─ 川菜
│                │                  ├─ 粤菜
│                │                  └─ ...
│                ├─ 西餐            ├─ 法国菜
│                │                  ├─ 意大利菜
│                │                  └─ ...
│                └─ 快餐            ├─ 汉堡
│                                   └─ ...
├─ 购物          ├─ 超市            ├─ 大型超市
│                ├─ 便利店          ├─ 24小时便利店
│                └─ ...             └─ ...
├─ 生活服务      ├─ 美容            ├─ 美发
│                ├─ 洗衣            ├─ 干洗
│                └─ ...             └─ ...
└─ ...           └─ ...             └─ ...

POI数据模型：
{
    "poiId": "P1234567890",
    "name": "海底捞(国贸店)",
    "category": {
        "level1": "餐饮",
        "level2": "中餐",
        "level3": "火锅"
    },
    "location": {
        "lng": 116.4752,
        "lat": 39.9042,
        "address": "北京市朝阳区建国路88号SOHO现代城",
        "addressId": "110105001001000000"
    },
    "attributes": {
        "businessHours": "10:00-22:00",
        "avgPrice": 120,
        "rating": 4.8,
        "phone": "010-12345678",
        "images": [...],
        "tags": ["网红店", "需排队", "有包间"]
    },
    "spatial": {
        "entrance": [[116.4752, 39.9042]],
        "outline": {...},  // 建筑轮廓
        "floor": 3,
        "area": 500  // 平方米
    },
    "metadata": {
        "createTime": "2020-01-01T00:00:00Z",
        "updateTime": "2024-01-01T00:00:00Z",
        "source": "MERCHANT_UPLOAD",  // 数据来源
        "verified": true,
        "confidence": 0.99
    }
}
```

**POI数据采集与更新**：

POI数据的准确性直接影响用户体验，美团通过多种方式保持POI数据的新鲜度：

1. **众包采集**：骑手在配送过程中上传门店照片、位置纠偏
2. **商家维护**：商家自主更新营业时间、菜单等信息
3. **用户反馈**：用户报告错误信息、提供更新建议
4. **自动挖掘**：从用户评论、订单地址中提取POI信息
5. **第三方同步**：与地图服务商、工商数据同步

**POI去重与融合**：

同一个实体可能有多个数据源，需要智能去重和融合：

```
POI融合决策矩阵：

数据源优先级：
商家上传 > 官方验证 > 骑手采集 > 用户反馈 > 自动挖掘

冲突解决策略：
┌─────────────┬──────────────┬──────────────┬──────────────┐
│ 属性类型     │ 位置信息     │ 营业时间      │ 联系方式     │
├─────────────┼──────────────┼──────────────┼──────────────┤
│ 解决方案     │ 加权平均     │ 最新优先      │ 商家优先     │
│ 置信度权重   │ 基于来源     │ 基于时间      │ 基于验证     │
└─────────────┴──────────────┴──────────────┴──────────────┘
```

### 7.3.4 围栏计算优化

围栏计算是LBS系统的性能瓶颈之一，需要精心的算法设计和工程优化：

```
围栏判定优化层次：

第一层：粗筛（毫秒级）
├─ Geohash网格索引
├─ 外接矩形过滤
└─ 空间R-Tree索引

第二层：精判（微秒级）
├─ 射线法判定
├─ 向量叉积法
└─ GPU并行计算

第三层：缓存（纳秒级）
├─ 热点区域缓存
├─ 用户历史缓存
└─ 会话级缓存
```

**高性能围栏引擎架构**：

为了支撑每秒百万级的围栏查询，美团设计了分布式围栏引擎：

1. **数据分片**：按地理区域将围栏数据分片存储
2. **多级缓存**：L1(进程内) → L2(Redis) → L3(数据库)
3. **预计算**：离线预计算热点区域的围栏关系
4. **增量更新**：围栏变更通过消息队列实时同步
5. **降级方案**：当精确判定超时时，返回粗略结果

**围栏索引数据结构**：

```cpp
// 空间索引节点
struct SpatialNode {
    Rectangle mbr;           // 最小外接矩形
    vector<FenceId> fences; // 围栏ID列表
    vector<SpatialNode*> children; // 子节点
    
    bool contains(Point p) {
        // 快速判定点是否在MBR内
        return p.x >= mbr.minX && p.x <= mbr.maxX &&
               p.y >= mbr.minY && p.y <= mbr.maxY;
    }
};

// 围栏判定优化
class FenceEngine {
    SpatialIndex index;      // 空间索引
    Cache<Point, FenceList> cache; // 判定缓存
    
    FenceList query(Point p) {
        // 1. 查缓存
        if (cache.contains(p)) {
            return cache.get(p);
        }
        
        // 2. 空间索引粗筛
        auto candidates = index.query(p);
        
        // 3. 精确判定
        FenceList results;
        for (auto fence : candidates) {
            if (fence.contains(p)) {
                results.add(fence);
            }
        }
        
        // 4. 更新缓存
        cache.put(p, results);
        return results;
    }
};
```

## 7.4 路径规划与导航

路径规划是LBS系统的核心算法，直接影响配送效率和用户体验。美团的路径规划需要处理每小时29亿次的规划请求，同时保证10ms以内的响应时间。

### 7.4.1 多模式路径规划（步行/骑行/电动车）

不同的交通模式有着截然不同的路径特征和约束条件：

```
多模式路径特征对比：

┌──────────┬────────────┬────────────┬────────────┬────────────┐
│ 模式     │ 步行       │ 自行车     │ 电动车     │ 汽车       │
├──────────┼────────────┼────────────┼────────────┼────────────┤
│ 速度     │ 5km/h      │ 15km/h     │ 25km/h     │ 30km/h     │
│ 可达性   │ 最高       │ 高         │ 中         │ 低         │
│ 楼梯     │ ✓          │ ✗          │ ✗          │ ✗          │
│ 逆行     │ ✓          │ 部分       │ ✗          │ ✗          │
│ 人行道   │ ✓          │ ✓          │ 部分       │ ✗          │
│ 停车     │ 无需       │ 方便       │ 需要       │ 困难       │
│ 天气影响 │ 高         │ 高         │ 中         │ 低         │
└──────────┴────────────┴────────────┴────────────┴────────────┘

路网建模：
Graph = {
    nodes: [
        {
            id: "N001",
            location: [116.4752, 39.9042],
            type: "INTERSECTION",  // 交叉口
            elevation: 0,           // 海拔
            connections: [...]
        }
    ],
    edges: [
        {
            id: "E001",
            from: "N001",
            to: "N002",
            distance: 150,          // 米
            modes: ["WALK", "BIKE", "EBIKE"],
            properties: {
                width: 3.5,         // 道路宽度
                slope: 0.02,        // 坡度
                surface: "ASPHALT", // 路面材质
                traffic: "LIGHT",   // 交通状况
                restricted: false   // 是否限行
            },
            costs: {
                WALK: 120,          // 步行耗时(秒)
                BIKE: 36,           // 自行车耗时
                EBIKE: 22           // 电动车耗时
            }
        }
    ]
}
```

**多模式路径规划算法**：

美团采用了改进的A*算法，针对不同交通模式定制化优化：

```python
# 多模式路径规划核心算法
def multiModalRoute(start, end, mode, constraints):
    # 1. 初始化
    openSet = PriorityQueue()
    openSet.add(start, 0)
    cameFrom = {}
    gScore = {start: 0}
    fScore = {start: heuristic(start, end, mode)}
    
    # 2. 搜索主循环
    while not openSet.empty():
        current = openSet.pop()
        
        if current == end:
            return reconstructPath(cameFrom, current)
        
        # 3. 遍历邻居节点
        for neighbor in getNeighbors(current, mode):
            # 计算通过当前节点到邻居的成本
            tentativeGScore = gScore[current] + \
                            getCost(current, neighbor, mode, constraints)
            
            # 4. 剪枝优化
            if tentativeGScore < gScore.get(neighbor, INF):
                # 记录路径
                cameFrom[neighbor] = current
                gScore[neighbor] = tentativeGScore
                fScore[neighbor] = tentativeGScore + \
                                 heuristic(neighbor, end, mode)
                
                # 加入开放集合
                if neighbor not in openSet:
                    openSet.add(neighbor, fScore[neighbor])
    
    return None  # 无可达路径

# 成本计算函数（考虑多种因素）
def getCost(from, to, mode, constraints):
    baseCost = edge.costs[mode]
    
    # 实时路况调整
    trafficFactor = getTrafficFactor(edge, currentTime)
    
    # 天气影响
    weatherFactor = getWeatherFactor(mode, currentWeather)
    
    # 用户偏好（如避开主路）
    preferenceFactor = getPreferenceFactor(edge, constraints)
    
    # 安全系数（夜间、事故路段等）
    safetyFactor = getSafetyFactor(edge, currentTime)
    
    return baseCost * trafficFactor * weatherFactor * \
           preferenceFactor * safetyFactor
```

### 7.4.2 实时路况融合

实时路况是影响路径规划质量的关键因素。美团通过多源数据融合构建实时路况图：

```
路况数据源架构：

┌─────────────────────────────────────────────────────────┐
│                    实时路况融合系统                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  数据源层                处理层              输出层      │
│  ┌──────────┐         ┌──────────┐      ┌──────────┐  │
│  │骑手轨迹  │────────▶│          │      │          │  │
│  ├──────────┤         │  数据    │      │  路况    │  │
│  │用户反馈  │────────▶│  清洗    │─────▶│  融合    │  │
│  ├──────────┤         │  标准化  │      │  模型    │  │
│  │第三方数据│────────▶│          │      │          │  │
│  ├──────────┤         └──────────┘      └──────────┘  │
│  │IoT传感器 │────────▶ 实时流处理          │           │
│  └──────────┘                              ▼           │
│                                    ┌──────────────┐     │
│                                    │ 路况图输出   │     │
│                                    │ (5秒更新)    │     │
│                                    └──────────────┘     │
└─────────────────────────────────────────────────────────┘

路况等级定义：
Level 1: 畅通    (绿色)  - 实际速度 > 期望速度 * 0.9
Level 2: 基本畅通 (黄色)  - 实际速度 > 期望速度 * 0.7
Level 3: 轻度拥堵 (橙色)  - 实际速度 > 期望速度 * 0.5
Level 4: 中度拥堵 (红色)  - 实际速度 > 期望速度 * 0.3
Level 5: 严重拥堵 (深红)  - 实际速度 < 期望速度 * 0.3
```

**路况预测模型**：

除了实时路况，美团还构建了路况预测模型，预测未来30分钟的路况变化：

```python
# 路况预测模型架构
class TrafficPredictor:
    def __init__(self):
        self.historical_patterns = {}  # 历史模式
        self.event_calendar = {}        # 事件日历
        self.weather_forecast = {}      # 天气预报
        
    def predict(self, road_segment, future_time):
        # 1. 历史同期路况
        historical = self.getHistoricalPattern(
            road_segment, 
            future_time.weekday, 
            future_time.hour
        )
        
        # 2. 趋势分析
        trend = self.analyzeTrend(road_segment, last_30_minutes)
        
        # 3. 事件影响（演唱会、球赛等）
        event_impact = self.getEventImpact(road_segment, future_time)
        
        # 4. 天气影响
        weather_impact = self.getWeatherImpact(
            self.weather_forecast[future_time]
        )
        
        # 5. 融合预测
        prediction = self.fusePrediction(
            historical * 0.4 +
            trend * 0.3 +
            event_impact * 0.2 +
            weather_impact * 0.1
        )
        
        return prediction
```

### 7.4.3 室内导航方案

外卖配送的"最后100米"往往是最复杂的，特别是在大型商场、写字楼等室内场景：

```
室内导航技术栈：

1. 室内地图构建
   ├─ 楼层平面图数字化
   ├─ POI标注（电梯、楼梯、商铺）
   ├─ 通行区域识别
   └─ 障碍物标记

2. 室内定位技术
   ├─ WiFi指纹定位
   ├─ 蓝牙Beacon
   ├─ 地磁定位
   └─ 惯性导航（IMU）

3. 路径规划算法
   ├─ 多楼层路径搜索
   ├─ 电梯等待时间估算
   ├─ 楼梯vs电梯决策
   └─ 拥挤度避让

室内导航数据结构：
{
    "buildingId": "B001",
    "floors": [
        {
            "level": 1,
            "map": {
                "image": "floor1.png",
                "scale": 0.1,  // 米/像素
                "rotation": 0
            },
            "walkableArea": [
                // 多边形定义的可行走区域
                [[100, 100], [200, 100], [200, 200], [100, 200]]
            ],
            "pois": [
                {
                    "id": "E01",
                    "type": "ELEVATOR",
                    "location": [150, 150],
                    "connections": [2, 3, 4]  // 连接楼层
                },
                {
                    "id": "S01",
                    "type": "STAIR",
                    "location": [180, 150],
                    "connections": [0, 2]
                }
            ],
            "shops": [
                {
                    "id": "SHOP001",
                    "name": "星巴克",
                    "entrance": [120, 130],
                    "category": "咖啡店"
                }
            ]
        }
    ]
}
```

**室内外无缝切换**：

```python
# 室内外导航切换逻辑
class HybridNavigator:
    def navigate(self, start, end):
        # 1. 判断起终点位置
        startIndoor = self.isIndoor(start)
        endIndoor = self.isIndoor(end)
        
        if not startIndoor and not endIndoor:
            # 纯室外导航
            return self.outdoorRoute(start, end)
            
        elif startIndoor and endIndoor and \
             self.sameBuilding(start, end):
            # 同建筑物内导航
            return self.indoorRoute(start, end)
            
        else:
            # 室内外混合导航
            segments = []
            
            if startIndoor:
                # 室内到建筑物出口
                exit = self.findNearestExit(start)
                segments.append(self.indoorRoute(start, exit))
                start = exit
                
            # 室外路径
            if endIndoor:
                entrance = self.findBestEntrance(end)
                segments.append(self.outdoorRoute(start, entrance))
                # 入口到室内终点
                segments.append(self.indoorRoute(entrance, end))
            else:
                segments.append(self.outdoorRoute(start, end))
                
            return self.mergeSegments(segments)
```

### 7.4.4 路径偏离检测与重规划

骑手在配送过程中可能因为各种原因偏离规划路径，系统需要实时检测并做出响应：

```
偏离检测状态机：

         ┌─────────┐
         │  正常   │
         └────┬────┘
              │
              ▼
    ┌──────────────────┐
    │ 检测到位置偏移   │
    └─────────┬────────┘
              │
         判定偏离程度
              │
    ┌─────────┼─────────┐
    │         │         │
    ▼         ▼         ▼
轻微偏离  中度偏离  严重偏离
    │         │         │
    ▼         ▼         ▼
继续监控  建议返回  强制重规划

偏离检测算法：
def detectDeviation(currentLocation, plannedPath, threshold):
    # 1. 计算到规划路径的垂直距离
    distance = calculatePerpendicularDistance(
        currentLocation, 
        plannedPath
    )
    
    # 2. 计算偏离角度
    if len(recentLocations) >= 3:
        movingDirection = calculateDirection(recentLocations)
        plannedDirection = getPlannedDirection(plannedPath)
        angle = angleDifference(movingDirection, plannedDirection)
    
    # 3. 综合判定
    if distance > threshold.distance:
        return "DISTANCE_DEVIATION"
    elif angle > threshold.angle:
        return "DIRECTION_DEVIATION"
    elif self.isMovingAway(currentLocation, destination):
        return "WRONG_DIRECTION"
    else:
        return "ON_TRACK"

# 重规划策略
def replanStrategy(deviation, currentLocation, destination):
    if deviation == "MINOR":
        # 软提醒，不强制
        return {
            "action": "SUGGEST",
            "message": "建议返回规划路线"
        }
    elif deviation == "MODERATE":
        # 提供返回路径
        returnPath = planPath(currentLocation, nearestPointOnPath)
        return {
            "action": "RETURN_PATH",
            "path": returnPath
        }
    elif deviation == "SEVERE":
        # 完全重新规划
        newPath = planPath(currentLocation, destination)
        return {
            "action": "REPLAN",
            "path": newPath,
            "reason": "严重偏离原路径"
        }
```

**智能重规划决策**：

重规划不是简单地计算新路径，还需要考虑多种业务因素：

```python
class SmartReplanner:
    def shouldReplan(self, context):
        # 评估重规划的收益和成本
        factors = {
            "time_saved": self.estimateTimeSaved(context),
            "distance_saved": self.estimateDistanceSaved(context),
            "delivery_risk": self.assessDeliveryRisk(context),
            "user_experience": self.predictUserImpact(context),
            "rider_effort": self.estimateRiderEffort(context)
        }
        
        # 加权决策
        score = (
            factors["time_saved"] * 0.3 +
            factors["distance_saved"] * 0.2 +
            factors["delivery_risk"] * 0.3 +
            factors["user_experience"] * 0.15 +
            factors["rider_effort"] * 0.05
        )
        
        return score > REPLAN_THRESHOLD
    
    def replan(self, context):
        # 生成多个候选路径
        candidates = []
        
        # 最短路径
        candidates.append(self.shortestPath(
            context.current, 
            context.destination
        ))
        
        # 最快路径（考虑实时路况）
        candidates.append(self.fastestPath(
            context.current, 
            context.destination,
            context.traffic
        ))
        
        # 最安全路径（避开事故多发区）
        candidates.append(self.safestPath(
            context.current,
            context.destination
        ))
        
        # 选择最优路径
        return self.selectBest(candidates, context)
```
