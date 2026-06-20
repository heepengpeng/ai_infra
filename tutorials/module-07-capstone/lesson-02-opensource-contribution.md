# M7 · Lesson 2:开源贡献指南——两周内提交你的第一个 PR

> Capstone 证明你"能独立做事";一个被合入的开源 PR,证明你"能在世界级团队的代码标准下做事"。对推理 Infra 岗位来说,后者是**最强的背书之一**——因为 vLLM/SGLang/TensorRT-LLM 正是这个行业的"军火库",你的名字出现在它们的 contributor 列表里,等于行业给你盖了章。
> 预计用时:阅读 2 小时;行动(到合入第一个 PR)1–3 周。
> 前置:Git 基础;M4(引擎机制),读 vLLM 源码会用上;Capstone 做过更佳。

## 学习目标

学完本课你应该能:
1. 说清为什么开源贡献对求职是"高杠杆"动作,以及它**到底在向面试官证明什么**。
2. 在 vLLM / SGLang / TensorRT-LLM 之间,**按自己的情况选对**一个主攻项目。
3. 找到适合新手的切入点(good first issue / 文档 / 复现 bug / 小 bugfix → feature)。
4. 走完一条规范的 PR 流程(fork → 分支 → 改 → 测 → commit → PR → 过 review)。
5. 掌握**读大型项目源码**的方法,不被几十万行代码吓退。
6. 拿到一份"两周内提交第一个 PR"的可执行行动清单。

---

## 1. 为什么开源是最强背书

招聘的本质是**降低不确定性**。面试官最大的焦虑是"这人简历写得好,真上手行不行?"。开源 PR 一次性消除了好几个不确定性:

| 面试官的疑虑 | 一个合入的 PR 如何回答 |
|---|---|
| 他能读懂复杂代码吗? | 能。我在 X 万行的 vLLM 里定位并改对了问题 |
| 他的代码质量行吗? | 行。我的代码通过了核心维护者的 review 和 CI |
| 他懂这个领域吗? | 懂。我改的就是 PagedAttention/调度器/量化这块 |
| 他能协作吗? | 能。我和 maintainer 来回讨论、改了 3 版才合入 |
| 他是真热爱还是来混的? | 真热爱。业余时间给行业顶级项目做贡献 |

> **一句话:开源 PR 把你简历上的形容词,变成了可点击验证的事实。** 面试官能直接打开你的 PR 看代码、看讨论、看 review——这比任何自述都可信。

而且对你的处境特别合适:你想从应用/大数据转推理 Infra,**缺的不是能力,是"推理 Infra 方向的可信证据"**。开源 PR 正是最快补上这块的方式之一。

---

## 2. 选哪个项目:vLLM / SGLang / TensorRT-LLM 对比

不要三个都碰,**选一个深耕**。三个里挑一个做主攻,其余了解即可。

| 项目 | 语言/技术栈 | 社区/上手 | 适合你的理由 | 不适合的点 |
|---|---|---|---|---|
| **vLLM** | Python 为主 + 部分 CUDA/C++ | ⭐ 最大最活跃,issue 多、good first issue 多 | **首选**:Python 友好(你强项),你 M4 精读过 nano-vLLM,概念迁移直接;岗位需求量最大 | 太热门,简单 issue 会被抢 |
| **SGLang** | Python + Triton + CUDA | 增长极快,RadixAttention 等亮点 | 相对 vLLM 竞争小、容易出彩;Triton kernel 多(你 M2 学过 Triton) | 体量略小,文档不如 vLLM |
| **TensorRT-LLM** | C++ 为主 + Python 接口 | NVIDIA 官方,工业级 | 含金量高、企业认可度高;你有 C++ 基础(M0)+ 系统结构背景 | C++/CUDA 占比高,上手陡,审核慢 |

**给你的建议**:
- **主攻 vLLM**:Python 是你的强项,你在 M4 精读过 nano-vLLM(它就是 vLLM 的极简版),源码概念对得上,且招聘市场上"vLLM 经验"出现频率最高。
- **想差异化**:辅以 **SGLang**,竞争小、有 Triton(你会),容易拿到第一个 merge。
- **TensorRT-LLM** 作为长期目标:等你 C++/CUDA 更熟、想冲 NVIDIA 系或大厂底层岗时再上。

---

## 3. 贡献的阶梯:从最低风险到高价值

新手最大的错误是**一上来就想提 feature**,结果半途而废。正确路径是**爬阶梯**,每一阶都积累对项目的熟悉度和 maintainer 的信任:

```
第 5 阶  Feature / 性能优化   ← 终极目标(新算子、新调度策略、kernel 优化)
   ↑
第 4 阶  实质性 bugfix         ← 定位并修复一个真实 bug
   ↑
第 3 阶  复现/分类 issue       ← 帮忙复现别人报的 bug、补最小复现脚本
   ↑
第 2 阶  good first issue      ← 维护者贴了"新手友好"标签的小任务
   ↑
第 1 阶  文档 / typo / 注释    ← 零风险,熟悉 PR 流程的练手(别停在这)
```

> 策略:**第 1 阶只用来跑通 PR 流程(1 个就够),真正的目标是尽快爬到第 4–5 阶。** 纯文档 PR 面试加分有限,"修了一个真实 bug"或"加了一个小 feature"才有故事。

### 各阶怎么找

- **good first issue**:在 GitHub 仓库 Issues 里筛标签 `good first issue`、`good-first-issue`、`help wanted`。**手快有手慢无**,看到合适的立刻在评论区说"I'd like to work on this"占坑。
- **复现 issue**:翻 `bug` 标签里"无人复现"的,自己跑一遍,贴上最小复现脚本和环境信息——maintainer 会很感激,这也是建立信任的好方式。
- **从你的 Capstone 找**:做 Lesson 1 时你大概率会撞到真实 bug 或文档错误(某参数行为不符、某模型报错、README 命令过时)。**这是你最自然的第一个 PR 来源**——你有完整的复现环境和上下文。

---

## 4. 如何读大型项目源码(不被吓退)

vLLM 几十万行,从头读到尾是灾难。**正确方法是"带着问题、自顶向下、动态追踪"。**

### 4.1 三条原则

1. **不要通读,要追一条线**。选一个具体问题(如"一个请求从进来到吐出第一个 token 经过了哪些函数"),只读这条调用链上的代码。
2. **自顶向下**。从入口(API server / `LLM.generate`)开始,顺着调用往下钻,而不是从底层 kernel 往上猜。
3. **动态验证胜过静态猜测**。打断点 / 加 `print` 日志 / 用调试器实际跑一遍,看数据怎么流动——比盯着代码空想快 10 倍。

### 4.2 读 vLLM 的一条推荐主线(你 M4 有基础)

追踪"一次推理请求的生命周期":

```
LLM.generate / api_server 入口
   → LLMEngine.add_request        (请求如何入队)
   → Scheduler.schedule           (★ 连续批处理:这一步决定本次迭代跑哪些请求)
   → ModelRunner.execute_model    (前向计算)
   → Attention / PagedAttention   (★ KV Cache 怎么按 block 取)
   → Sampler                      (采样出 token)
   → 输出回传,更新 KV Cache,循环
```

> 带 ★ 的两处(Scheduler 和 PagedAttention)正是 vLLM 的灵魂,也是你 M4 学过原理的地方。**把这两块读透,你就有了和 maintainer 对话的资本**,也最容易发现可改进点。

### 4.3 实用技巧

- 用 IDE 的 **"跳转到定义 / 查找引用"**,顺着符号跳,别手动翻文件。
- 看 **测试代码** 来理解一个模块怎么用——测试是最好的"用法文档"。
- 看一个相关功能**最近合入的 PR**(尤其 maintainer 自己的),学它怎么改、怎么加测试、怎么写 commit。这是模仿项目"惯例"的最快方式。
- 看 `CONTRIBUTING.md` 和 `docs/`,每个项目都有自己的"家规"(代码风格、测试要求、PR 模板)。

---

## 5. 标准 PR 流程与礼仪

### 5.1 流程(以 vLLM 为例)

```bash
# 1. Fork 仓库到自己账号,然后 clone 你的 fork
git clone https://github.com/<你的用户名>/vllm.git
cd vllm
git remote add upstream https://github.com/vllm-project/vllm.git

# 2. 同步最新主分支,基于它开新分支(分支名见名知意)
git fetch upstream
git checkout -b fix/scheduler-typo upstream/main

# 3. 按 CONTRIBUTING.md 装好开发环境
pip install -e .
pre-commit install        # 很多项目用 pre-commit 自动格式化/lint

# 4. 改代码 + 加/改测试

# 5. 本地跑 lint 和相关测试(别把 CI 当你的测试机)
pre-commit run --all-files
pytest tests/路径/相关测试.py

# 6. 提交(commit message 规范见下),推到自己的 fork
git add -p
git commit -s -m "fix: correct off-by-one in scheduler block accounting"
git push origin fix/scheduler-typo

# 7. 在 GitHub 上对 upstream/main 发起 Pull Request,按模板填写
```

> `git commit -s`(signoff,DCO)很多项目强制要求,漏了 CI 会红。先看 `CONTRIBUTING.md` 确认。

### 5.2 PR 描述怎么写(maintainer 一眼就懂)

```
标题:[Bugfix] Fix off-by-one in scheduler block accounting

## What
一句话说清这个 PR 做了什么。

## Why / Problem
关联 issue(Fixes #1234)。描述 bug 现象 + 根因。

## How
你的修复思路,关键改动点。

## Test
你怎么验证的:加了什么测试、本地跑的结果(贴输出/截图)。
```

### 5.3 开源礼仪(决定你被不被欢迎)

- **先沟通再动手**:对非 trivial 的改动,先在 issue 里说明思路,得到 maintainer 认可("go ahead")再写。避免吭哧吭哧写完发现方向被否。
- **小而专注**:一个 PR 只做一件事。混杂格式化 + 功能 + 重构的大 PR 没人想 review。
- **接受 review 不玻璃心**:被要求改 3 版很正常。把每条 comment 当学习机会,改完逐条回复"Done"或解释。(这部分心态参见 `receiving-code-review` 的思路:有理有据地讨论,而不是盲从或对抗。)
- **耐心**:maintainer 是义务劳动,几天没回是常态。礼貌 ping 一次即可,别催。
- **英文沟通**:简洁、礼貌、就事论事。语法不完美没关系,清楚最重要。

> 一条铁律:**别"认领了 issue 然后人间蒸发"。** 如果做不下去,留言说一声让别人接手。社区记仇也记好。

---

## 6. 动手任务:两周行动清单

这是一个**可直接执行**的两周计划。目标:**两周末提交(不一定合入)你的第一个 PR。**

### 第 1 周:扎根 + 第一个练手 PR

- [ ] **D1**:选定主攻项目(建议 vLLM)。Star、clone、读完 `CONTRIBUTING.md`。
- [ ] **D2**:按文档**从源码装好开发环境**,能本地跑起来(这一步常卡人,提前踩坑)。
- [ ] **D3–D4**:按第 4.2 节**追一遍"请求生命周期"主线**,边读边画调用图,搞懂 Scheduler 和 PagedAttention 入口。
- [ ] **D5**:在 Issues 里筛 `good first issue` / `documentation`,或从你 Capstone 中找到的真实文档/小问题,**锁定一个**。评论占坑。
- [ ] **D6–D7**:完成这个**练手 PR**(文档修正/typo/小注释也行),走通完整 fork→PR 流程,体验 CI 和 review。**目标是跑通流程,不求难度。**

### 第 2 周:一个有故事的 PR

- [ ] **D8–D9**:在 `bug` / `good first issue` 里找一个**有点实质的小任务**(复现一个 bug、修一个小逻辑错、补一个边界处理),或从 Capstone 实战中你真实遇到的 bug 出发。
- [ ] **D10**:在 issue 里**先讲思路**,等 maintainer 点头。
- [ ] **D11–D12**:写代码 + **写测试**(没测试的功能/修复很难被合)。本地跑通 lint + 测试。
- [ ] **D13**:提交 PR,按模板写清 What/Why/How/Test。
- [ ] **D14**:**响应 review**,该改改、该解释解释。把这个过程本身写进你的"经历素材库"(Lesson 3 要用)。

> 现实预期:**第一个 PR 不一定在两周内合入(review 周期不可控),但"提交了、在被讨论"本身就已经是面试可讲的素材。** 别因为没 merge 就不敢写进简历——写"正在贡献 vLLM,提交了关于 XX 的 PR(#1234)"完全 OK。

---

## 练习题

1. 你看中了一个 `good first issue`,但发现已经有人评论"I'll take this"两周了却没动静。你怎么做?
2. 你想给 vLLM 加一个"新采样策略"的 feature,该先做什么?直接写完代码发 PR 行不行?
3. maintainer 在你 PR 下留言:"Can you add a unit test for this edge case?" 但你觉得这个 case 不可能发生。你怎么回应?

<details>
<summary>参考答案</summary>

1. 礼貌地在 issue 下问一句:"Hi @原认领者, are you still working on this? If not, I'd be happy to take it over." 等 1–2 天没回应,再 @ maintainer 说明情况、请求接手。**不要直接抢着提 PR**,那样不礼貌也容易撞车。

2. **先开 issue / 在已有 issue 里提案,讲清动机、设计、API 影响,等 maintainer 认可方向再写**。直接写完发大 PR 风险极高:可能方向被否、可能和 roadmap 冲突、可能 maintainer 根本不想要这个 feature,你的功夫全白费。Feature 类贡献"沟通先于编码"。

3. **不要直接对抗,也不要盲目照做**。先理解他的顾虑,然后有理有据地回应:要么"你说得对,我没考虑到 X 场景,这就加测试";要么"我分析了一下,这个 case 在 Y 约束下确实不会发生,因为……,你看是否还需要测试?"。把它当技术讨论,用证据说话。这种专业的来回讨论,本身就是面试官最想看到的协作能力。(详见 `receiving-code-review` 的原则:验证而非表演式同意。)

</details>

---

## 小结

- 开源 PR 是**高杠杆求职动作**:把简历的形容词变成可验证的事实,一次性消除面试官对你"读码/质量/领域/协作"的疑虑。
- **选一个深耕**:首选 vLLM(Python 友好、市场需求大、你有 nano-vLLM 基础),想差异化加 SGLang,长期看 TensorRT-LLM。
- 走**贡献阶梯**:文档练手 1 个 → 尽快爬到"实质 bugfix / 小 feature",那才有故事。
- 读源码靠**追一条线、自顶向下、动态验证**,vLLM 重点追 Scheduler + PagedAttention。
- PR 礼仪核心:**先沟通再动手、小而专注、有测试、耐心、不玻璃心、不人间蒸发**。
- 现实预期:**两周内"提交并在讨论中"就是成功**,不必等 merge 才敢写进简历。

## 自测验收
- [ ] 能说清开源 PR 在向面试官证明哪 5 件事。
- [ ] 选定了主攻项目并说得出选它的理由。
- [ ] 本地从源码装好了开发环境并能跑起来。
- [ ] 追完了一遍"请求生命周期"调用链,能讲出 Scheduler 和 PagedAttention 的入口。
- [ ] 提交了至少 1 个 PR(哪怕是文档/小修),完整体验了 fork→PR→CI→review。
- [ ] 锁定了第 2 个"有故事"的 PR 目标。

---

下一课:**Lesson 3 — 简历重写 + 面试八股精讲**。你已经有了 Capstone(独立做事的证据)和开源 PR(团队标准下做事的证据)。最后一步,是把你**全部经历**——SDXL 10×、Tachyons、九问、加上这两件新作品——翻译成"推理 Infra 工程师"的语言,并系统过一遍面试必考的八股。这是临门一脚。
