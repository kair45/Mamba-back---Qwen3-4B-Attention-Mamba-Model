#!/usr/bin/env python3
"""
Step 0 (本地): 用 API 预生成蒸馏训练数据。

本脚本 **不需要 GPU**，可以在任何有网络的机器上运行。
生成的数据会缓存到磁盘，之后拷贝到 GPU 服务器上直接训练。

工作流:
    ┌─────────────────────────────┐     拷贝 cache 目录     ┌───────────────────┐
    │  本地电脑 (无需 GPU)          │  ─────────────────→   │  GPU 服务器         │
    │  python generate_api_data.py │                        │  python train_     │
    │  → 输出 ./cache/api_distill/ │                        │  distill.py        │
    └─────────────────────────────┘                        │  --cached_data_dir │
                                                           │  ./cache/api_distill│
                                                           └───────────────────┘

Usage:
    # 基本用法 (100 条, 快速验证)
    python scripts/generate_api_data.py \
        --num_samples 100 \
        --api_model qwen-max \
        --output_dir ./cache/api_distill

    # 正式训练 (5000 条, 推荐)
    python scripts/generate_api_data.py \
        --num_samples 5000 \
        --api_model qwen-max \
        --output_dir ./cache/api_distill

    # 用 qwen-plus 省钱
    python scripts/generate_api_data.py \
        --num_samples 5000 \
        --api_model qwen-plus \
        --output_dir ./cache/api_distill

    # 自定义 prompt 文件 (每行一条 prompt)
    python scripts/generate_api_data.py \
        --num_samples 5000 \
        --prompt_file my_prompts.txt \
        --output_dir ./cache/api_distill

环境变量:
    DASHSCOPE_API_KEY: 阿里云 DashScope API 密钥
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 默认种子 prompt 集（覆盖多领域知识，中英文混合，共 500+ 条）
# ---------------------------------------------------------------------------
DEFAULT_SEED_PROMPTS = [
    # ── AI / 深度学习 ──
    "请详细解释量子计算的基本原理，包括量子比特、叠加态和量子纠缠的概念。",
    "Explain the transformer architecture in detail, including self-attention, multi-head attention, and positional encoding.",
    "Describe the mathematical foundations of neural networks, including backpropagation and gradient descent.",
    "Describe the mathematical foundations of state space models and their relationship to recurrent neural networks.",
    "Describe the key innovations in the Mamba architecture and how selective state spaces differ from attention.",
    "Explain mixture of experts (MoE) models, their routing mechanisms, and load balancing strategies.",
    "Describe the evolution of computer vision from CNNs to Vision Transformers to modern foundation models.",
    "Explain how large language models are trained, covering pretraining, fine-tuning, and RLHF.",
    "Explain the concept of attention mechanisms in natural language processing, from Bahdanau attention to modern variants.",
    "Explain the differences between supervised, unsupervised, and reinforcement learning with practical examples.",
    "详细介绍机器学习中的正则化方法，包括L1、L2正则化、Dropout和数据增强的原理与区别。",
    "Explain the concept of transfer learning in deep learning and its applications in NLP and computer vision.",
    "Describe the architecture and training methodology of diffusion models for image generation.",
    "Describe how graph neural networks work and their applications in molecular property prediction and social network analysis.",
    "请分析人工智能在医疗诊断领域的应用现状、技术瓶颈和未来前景。",
    "Explain the mathematical concepts behind dimensionality reduction techniques like PCA, t-SNE, and UMAP.",
    "Explain the principles behind recommendation systems, including collaborative filtering, content-based filtering, and hybrid approaches.",
    "Describe the architecture of BERT and GPT models and explain their key differences in pretraining objectives.",
    "Explain how reinforcement learning from human feedback (RLHF) works and its role in aligning language models.",
    "Describe the concept of neural architecture search (NAS) and its main approaches.",
    "Explain how contrastive learning works and describe its applications in self-supervised representation learning.",
    "Describe the Retrieval-Augmented Generation (RAG) framework and its advantages over pure parametric models.",
    "Explain the concept of knowledge distillation: how it works, its loss functions, and practical applications.",
    "Describe the differences between encoder-only, decoder-only, and encoder-decoder transformer architectures.",
    "Explain how tokenization works in large language models, covering BPE, WordPiece, and SentencePiece.",
    "Describe the concept of scaling laws in deep learning and their implications for model design.",
    "Explain the intuition and mathematics behind variational autoencoders (VAEs).",
    "Describe the architecture and training process of CLIP and its impact on multimodal AI.",
    "Explain how sparse attention mechanisms reduce the quadratic complexity of standard self-attention.",
    "Describe the key ideas behind low-rank adaptation (LoRA) and parameter-efficient fine-tuning methods.",
    "Explain how model quantization works and the trade-offs between INT8, INT4, and lower-bit quantization.",
    "Describe the concept of chain-of-thought prompting and its effect on reasoning in language models.",
    "Explain the architecture of AlphaFold2 and how it achieves accurate protein structure prediction.",
    "Describe the differences between autoregressive and masked language model pretraining objectives.",
    "Explain how speculative decoding accelerates inference in large language models.",
    "Describe the concept of mixture of depths and other dynamic computation approaches in transformers.",
    "Explain the training instabilities in large language models and techniques to address them.",
    "Describe how constitutional AI and RLAIF differ from standard RLHF.",
    "Explain the intuition behind Flash Attention and how it optimizes memory access patterns.",
    "Describe the key components of a modern vector database and how approximate nearest neighbor search works.",
    # ── 系统 / 软件工程 ──
    "Explain the CAP theorem in distributed systems and how different databases make trade-offs.",
    "请详细解释操作系统的内存管理机制，包括虚拟内存、页面置换算法和内存映射。",
    "Explain the principles of compiler design, including lexical analysis, parsing, and code generation.",
    "Describe the evolution of programming languages from assembly to modern high-level languages and their design philosophies.",
    "Explain how garbage collection works in modern programming languages, covering mark-and-sweep, reference counting, and generational GC.",
    "Describe the SOLID principles in object-oriented design with concrete examples.",
    "Explain the differences between microservices and monolithic architectures and when to use each.",
    "Describe how the Linux kernel schedules processes and manages CPU time.",
    "Explain the concept of lock-free and wait-free data structures and their use in concurrent programming.",
    "Describe the TCP/IP protocol stack and explain how data flows from application to physical layer.",
    "Explain how DNS works, including recursive resolution, caching, and DNSSEC.",
    "Describe the design principles of the MapReduce programming model and its limitations.",
    "Explain how relational database query optimizers work, covering cost-based optimization and join algorithms.",
    "Describe the internals of a B-tree index and why it is preferred in database systems.",
    "Explain the concept of eventual consistency and how distributed systems like DynamoDB implement it.",
    "Describe the design of the Raft consensus algorithm and how it differs from Paxos.",
    "Explain how Docker and container runtimes work at the Linux kernel level (namespaces, cgroups).",
    "Describe the Kubernetes architecture and explain how pods are scheduled and managed.",
    "Explain the concept of reactive programming and its advantages for building scalable systems.",
    "Describe how LLVM IR works as a mid-level representation and its role in modern compiler toolchains.",
    "Explain the differences between OLTP and OLAP systems and how columnar storage enables analytics.",
    "Describe the design of a distributed cache like Redis and the challenges of cache invalidation.",
    "Explain how WebAssembly works and its potential as a universal runtime.",
    "Describe the architecture of a modern GPU and how it differs from a CPU for parallel workloads.",
    "Explain how version control systems like Git implement branching, merging, and history storage internally.",
    # ── 数学 / 统计 ──
    "请详细解释博弈论的基本概念，包括纳什均衡、囚徒困境和机制设计。",
    "Explain the mathematical intuition behind Bayesian inference and how it differs from frequentist statistics.",
    "Describe the central limit theorem and its practical implications in statistics.",
    "Explain the mathematical foundations of Fourier transforms and their applications in signal processing.",
    "Describe the concept of information entropy and its role in machine learning and data compression.",
    "Explain convex optimization: key concepts, gradient descent variants, and convergence guarantees.",
    "Describe the mathematical framework of Markov decision processes and their relationship to reinforcement learning.",
    "Explain how Monte Carlo methods work and describe the Metropolis-Hastings algorithm.",
    "Describe the intuition and mathematics behind support vector machines and kernel methods.",
    "Explain the concept of singular value decomposition (SVD) and its applications in data analysis.",
    "Describe the mathematical foundations of graph theory and its applications in computer science.",
    "Explain how hypothesis testing works, covering p-values, confidence intervals, and multiple testing correction.",
    "Describe the mathematics of Gaussian processes and their use in Bayesian optimization.",
    "Explain the concept of mutual information and its use in feature selection and causal discovery.",
    "Describe the mathematical foundations of topology and its surprising applications in data analysis (TDA).",
    # ── 自然科学 ──
    "详细解释相对论的基本概念，包括狭义相对论和广义相对论的区别。",
    "请详细分析太阳系各行星的特征，并讨论寻找外星生命的科学方法。",
    "请介绍CRISPR基因编辑技术的原理、应用前景和伦理挑战。",
    "Describe the process of protein folding and explain why AlphaFold was a breakthrough in computational biology.",
    "Describe the physics behind nuclear fusion and the current progress toward commercial fusion energy.",
    "请详细解释免疫系统的工作原理，包括先天免疫和适应性免疫的区别。",
    "Explain how mRNA vaccines work and why they represent a new paradigm in vaccine development.",
    "Describe the standard model of particle physics and explain the role of the Higgs boson.",
    "Explain the concept of entropy in thermodynamics and its relationship to the arrow of time.",
    "Describe the life cycle of stars from nebula formation to white dwarf, neutron star, or black hole.",
    "Explain how CRISPR base editing and prime editing improve upon first-generation CRISPR-Cas9.",
    "Describe the mechanisms of antibiotic resistance and strategies to combat superbugs.",
    "Explain the concept of epigenetics and how gene expression is regulated without changing DNA sequence.",
    "Describe the physics and chemistry of climate change, focusing on greenhouse gas radiative forcing.",
    "Explain how neuroplasticity works and its implications for learning and recovery from brain injury.",
    "请分析量子力学的哥本哈根诠释、多世界诠释和德布罗意-玻姆诠释的异同。",
    "Describe the mechanisms of photosynthesis, including the light-dependent and light-independent reactions.",
    "Explain how the human gut microbiome influences health, immunity, and mental well-being.",
    "Describe the process of plate tectonics and its role in shaping Earth's geography and causing earthquakes.",
    "Explain the physics of superconductivity and the challenges in achieving room-temperature superconductors.",
    # ── 经济 / 金融 ──
    "详细解释经济学中的供需关系理论，并举例说明价格机制的作用。",
    "详细解释金融衍生品的种类和定价原理，包括期权、期货和互换合约。",
    "写一篇关于区块链技术原理和去中心化金融（DeFi）的技术分析。",
    "请分析全球半导体产业链的现状、技术瓶颈和未来发展趋势。",
    "Explain the Black-Scholes model for option pricing and its key assumptions and limitations.",
    "Describe the concept of modern portfolio theory and the efficient frontier.",
    "Explain how central banks use monetary policy tools to control inflation and stimulate growth.",
    "Describe the causes and consequences of the 2008 global financial crisis.",
    "Explain the concept of behavioral economics and how cognitive biases affect financial decision-making.",
    "Describe the mechanics of high-frequency trading and its impact on market microstructure.",
    "Explain how credit default swaps work and their role in systemic financial risk.",
    "Describe the economic theory behind comparative advantage and international trade.",
    "请分析数字人民币的技术架构和对现有金融体系的潜在影响。",
    "Explain the concept of game theory in economics, covering auction theory and mechanism design.",
    "Describe the relationship between inflation, unemployment, and the Phillips curve in modern macroeconomics.",
    "解释现代货币理论（MMT）的核心主张及其争议。",
    "Describe the economics of platform markets and network effects, using examples like Uber and Airbnb.",
    "Explain how ESG investing works and the challenges in measuring environmental and social impact.",
    # ── 社会 / 人文 / 历史 ──
    "写一篇关于人工智能伦理的深度分析文章，讨论AI决策的公平性问题。",
    "详细介绍中国古代四大发明的历史背景和对世界文明的影响。",
    "请分析全球气候变化的主要原因、当前影响和未来预测。",
    "请分析第二次世界大战对国际格局的深远影响，包括联合国的建立和冷战的起源。",
    "请分析城市化进程对社会结构、环境和经济发展的多维度影响。",
    "请详细介绍中医理论的基本框架，包括阴阳五行、经络学说和辨证论治。",
    "分析文艺复兴运动对欧洲文化、科学和艺术的深远影响。",
    "请详细分析儒家思想的核心理念及其对东亚社会的历史影响。",
    "Explain the philosophical foundations of liberal democracy and the challenges it faces today.",
    "Describe the history and impact of the Industrial Revolution on society, labor, and the environment.",
    "Analyze the causes and long-term consequences of colonialism in Africa and Asia.",
    "Explain the key ideas of Immanuel Kant's Critique of Pure Reason and their influence on Western philosophy.",
    "Describe the development of human rights law from the Universal Declaration to contemporary debates.",
    "请分析社交媒体对现代民主政治和公共舆论的影响。",
    "Describe the linguistic Sapir-Whorf hypothesis and the evidence for and against linguistic relativity.",
    "Explain the sociology of religion: how religious institutions shape social order and cultural identity.",
    "Analyze the role of propaganda in 20th-century totalitarian regimes.",
    "请比较东西方哲学对自我、意识和自由意志的不同理解。",
    "Describe the history of the internet from ARPANET to the modern web and its societal implications.",
    "Explain the psychological theory of cognitive dissonance and its applications in everyday life.",
    # ── 能源 / 环境 ──
    "请分析可再生能源（太阳能、风能、氢能）的技术现状和未来发展路线。",
    "Describe the technology behind solid-state batteries and their potential to revolutionize electric vehicles.",
    "Explain how carbon capture and storage (CCS) works and evaluate its feasibility at scale.",
    "Describe the concept of a circular economy and how it differs from the traditional linear model.",
    "Explain the science behind ocean acidification and its impact on marine ecosystems.",
    "Describe the mechanisms of soil degradation and sustainable agriculture practices to restore soil health.",
    "Explain how smart grids work and the role of AI in optimizing energy distribution.",
    "Describe the environmental trade-offs of lithium mining for EV batteries.",
    "Explain the physics of solar photovoltaic cells and the efficiency limits of different technologies.",
    "Describe the concept of planetary boundaries and which ones humanity has already crossed.",
    # ── 医学 / 生命科学 ──
    "Explain how mRNA technology could be applied beyond COVID vaccines to treat cancer and genetic diseases.",
    "Describe the mechanism of action of GLP-1 receptor agonists and their use in treating obesity and diabetes.",
    "Explain the concept of synthetic biology and its potential to engineer microorganisms for industrial use.",
    "Describe the technology behind single-cell RNA sequencing and what it has revealed about cell biology.",
    "Explain how organ-on-a-chip technology works and its applications in drug testing.",
    "Describe the neuroscience of addiction: how drugs hijack the brain's reward system.",
    "Explain the mechanisms of aging at the cellular level, including telomere shortening and senescence.",
    "Describe how the blood-brain barrier works and the challenges it poses for drug delivery to the brain.",
    "Explain the concept of precision medicine and how genomics is enabling personalized cancer treatment.",
    "Describe the role of the lymphatic system in immunity and how it interacts with the circulatory system.",
    # ── 工程 / 硬件 ──
    "Describe the architecture of a modern CPU, including pipelines, caches, branch prediction, and out-of-order execution.",
    "Explain how DRAM works at the hardware level and the challenges of scaling to smaller process nodes.",
    "Describe the design of a modern GPU and how CUDA enables general-purpose GPU computing.",
    "Explain the principles of RISC-V and why it is gaining traction as an open ISA.",
    "Describe how photolithography works in semiconductor manufacturing and the role of EUV.",
    "Explain the concept of hardware security modules (HSMs) and trusted execution environments (TEEs).",
    "Describe the design principles of FPGA architectures and their advantages over ASICs for prototyping.",
    "Explain how 5G networks work, covering massive MIMO, millimeter waves, and network slicing.",
    "Describe the engineering challenges of building a hyperloop transportation system.",
    "Explain how LiDAR works and compare it to radar and camera-based perception for autonomous vehicles.",
    # ── 数据科学 / 统计应用 ──
    "Explain the concept of A/B testing and the statistical pitfalls that lead to incorrect conclusions.",
    "Describe the difference between causality and correlation and how causal inference methods address this.",
    "Explain how survival analysis works and its applications in medicine and customer churn prediction.",
    "Describe the challenges of working with imbalanced datasets and techniques to address them.",
    "Explain the concept of time series forecasting and compare ARIMA, Prophet, and deep learning approaches.",
    "Describe the ethical considerations in data collection, privacy, and algorithmic bias.",
    "Explain how federated learning works and its advantages for privacy-preserving machine learning.",
    "Describe the concept of data versioning and the principles of MLOps for production machine learning.",
    "Explain how anomaly detection works and describe both statistical and deep learning approaches.",
    "Describe the principles of experiment design, including randomization, blocking, and factorial designs.",
    # ── 编程语言 / 范式 ──
    "Explain the concept of functional programming and how it differs from imperative and OOP paradigms.",
    "Describe how Python's GIL works and its implications for multithreaded programs.",
    "Explain the ownership and borrowing system in Rust and how it achieves memory safety without a GC.",
    "Describe the actor model of concurrency and how Erlang/Elixir implements it.",
    "Explain how type inference works in statically typed languages like Haskell and OCaml.",
    "Describe the principles of metaprogramming and macros in languages like Lisp and Rust.",
    "Explain how async/await works under the hood in Python and JavaScript.",
    "Describe the design of the Go programming language and how goroutines differ from OS threads.",
    "Explain the concept of monads in functional programming with practical examples.",
    "Describe the differences between dynamic dispatch and static dispatch in object-oriented languages.",
    # ── 网络安全 ──
    "Describe the principles of cryptography, including symmetric encryption, asymmetric encryption, and hash functions.",
    "Explain how zero-knowledge proofs work and their applications in privacy-preserving systems.",
    "Describe the most common web application vulnerabilities (OWASP Top 10) and how to mitigate them.",
    "Explain how SSL/TLS works and describe the TLS 1.3 handshake process.",
    "Describe the concept of a supply chain attack and analyze recent high-profile examples.",
    "Explain how fuzzing works as a software testing technique for finding security vulnerabilities.",
    "Describe the architecture of a modern SIEM system and how it detects threats.",
    "Explain the concept of homomorphic encryption and its potential applications.",
    "Describe how ransomware attacks work and best practices for defense and recovery.",
    "Explain the principles of secure multiparty computation (MPC) and its real-world use cases.",
    # ── 哲学 / 认知科学 ──
    "Explain the hard problem of consciousness and the main philosophical positions on the mind-body problem.",
    "Describe the philosophical implications of the simulation hypothesis.",
    "Explain Gödel's incompleteness theorems and their implications for mathematics and logic.",
    "Describe the main theories of truth in philosophy: correspondence, coherence, and pragmatic theories.",
    "Explain the trolley problem and its relevance to AI ethics and autonomous vehicle decision-making.",
    "Describe the philosophy of science: what distinguishes scientific knowledge from pseudoscience?",
    "Explain the concept of emergence and how complex systems exhibit properties not present in their parts.",
    "Describe the main theories of consciousness: global workspace theory, integrated information theory, and predictive processing.",
    "Explain the paradox of tolerance and its relevance to free speech debates in modern democracies.",
    "Describe the philosophical debate between determinism and free will in light of modern neuroscience.",
    # ── 教育 / 认知心理学 ──
    "Explain the science of learning: spaced repetition, interleaving, retrieval practice, and their effectiveness.",
    "Describe the differences between growth mindset and fixed mindset and the research supporting each.",
    "Explain how working memory works and its implications for education and cognitive load theory.",
    "Describe the main theories of intelligence, including general intelligence (g factor) and multiple intelligences.",
    "Explain the psychological concept of flow and how to design environments that facilitate it.",
    "Describe how sleep affects memory consolidation and cognitive performance.",
    "Explain the concept of metacognition and its role in expert performance.",
    "Describe the neuroscience of decision-making and how emotions interact with rational thought.",
    "Explain the principles of behavioral economics applied to public policy and nudge theory.",
    "Describe the psychology of creativity: what research tells us about how creative insights happen.",
    # ── 艺术 / 语言 / 文学 ──
    "Analyze the narrative techniques used in magical realism, using works by Márquez and Borges as examples.",
    "Describe the evolution of jazz music and its influence on 20th-century Western culture.",
    "Explain the principles of color theory and how they are applied in design and visual art.",
    "Describe the history and cultural significance of the Silk Road in connecting civilizations.",
    "Explain the linguistic concept of pragmatics and how context shapes meaning in communication.",
    "Describe the development of the novel as a literary form from Cervantes to contemporary fiction.",
    "Explain how architectural styles reflect the social and technological context of their era.",
    "Describe the role of mythology in human culture and the common patterns identified by Joseph Campbell.",
    "Explain the concept of intertextuality and how it operates in postmodern literature.",
    "Describe the impact of the printing press on European society, religion, and the spread of knowledge.",
    # ── 中文专项（高质量长文本）──
    "请深入分析唐诗宋词的艺术特征与美学意境，并举例说明。",
    "请详细介绍《红楼梦》的主要人物、主题思想和艺术成就。",
    "请分析改革开放以来中国经济腾飞的主要驱动因素和历史意义。",
    "请介绍丝绸之路的历史地理、贸易商品和文明交流内容。",
    "请分析中国传统哲学中儒、道、佛三家思想的异同与融合。",
    "请深入分析《孙子兵法》的核心战略思想及其现代应用。",
    "请介绍中国古代天文历法的主要成就和对农业文明的影响。",
    "请分析汉字演变史：从甲骨文到楷书的字形演化规律。",
    "请探讨现代中国城乡差距的成因、现状与政策应对。",
    "请介绍中国传统节日的文化内涵、历史起源和民俗习惯。",
    "请分析人工智能时代对就业市场的冲击与机遇。",
    "请深入介绍中国高铁技术的发展历程、核心技术和全球影响。",
    "请分析新冠疫情对全球供应链和产业格局的长远影响。",
    "请介绍敦煌莫高窟的历史背景、艺术价值和保护工作。",
    "请分析全球人口老龄化趋势对经济、医疗和社会保障的影响。",
    "请介绍中国古代科举制度的历史演变、选拔机制和历史评价。",
    "请分析元宇宙概念的技术基础、商业前景和潜在风险。",
    "请深入介绍中国的航天工程成就，包括载人航天、月球探测和空间站建设。",
    "请分析全球化背景下文化多样性的保护与文化冲突问题。",
    "请介绍中国传统医学中针灸的理论基础、临床应用和现代研究进展。",
    # ── 跨学科 / 前沿议题 ──
    "Explain the concept of synthetic data generation and its role in training AI systems responsibly.",
    "Describe the technology and ethics of brain-computer interfaces like Neuralink.",
    "Explain how CRISPR could be used for gene drives to eliminate disease-carrying mosquitoes and the ethical concerns.",
    "Describe the concept of digital twins and their applications in manufacturing and urban planning.",
    "Explain the potential societal impacts of artificial general intelligence (AGI) if achieved.",
    "Describe the concept of longevity research and the main scientific approaches to extending human lifespan.",
    "Explain how quantum cryptography and quantum key distribution work.",
    "Describe the technology behind advanced robotics and the challenges of dexterous manipulation.",
    "Explain the concept of open-source AI and the tension between openness and safety in model release.",
    "Describe the main approaches to AI alignment and why it is considered an important unsolved problem.",
    "Explain the concept of neuromorphic computing and how it mimics the brain's architecture.",
    "Describe the potential of spatial computing and mixed reality for education and remote collaboration.",
    "Explain how autonomous vehicles handle edge cases and the current limitations preventing full deployment.",
    "Describe the technology behind large-scale language model serving infrastructure and optimization techniques.",
    "Explain the concept of AI agents and tool use: how LLMs can plan and execute multi-step tasks.",
    "Describe the impact of generative AI on creative industries including art, music, and writing.",
    "Explain the concept of biosafety levels and the governance of dangerous pathogen research.",
    "Describe the economic and geopolitical implications of the global AI race between the US and China.",
    "Explain how materials science is enabling new battery chemistries beyond lithium-ion.",
    "Describe the concept of smart cities and the role of IoT and data analytics in urban management.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="预生成 API 蒸馏训练数据（本地运行，不需要 GPU）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速验证 (100 条)
  python scripts/generate_api_data.py --num_samples 100

  # 正式训练 (5000 条, qwen-plus 省钱)
  python scripts/generate_api_data.py --num_samples 5000 --api_model qwen-plus

  # 自定义 prompt
  python scripts/generate_api_data.py --num_samples 5000 --prompt_file prompts.txt
        """,
    )
    parser.add_argument("--num_samples", type=int, default=100,
                        help="要生成的样本总数 (默认 100)")
    parser.add_argument("--api_model", type=str, default="qwen-max",
                        help="API 模型名称 (默认 qwen-max), 支持任意 DashScope 兼容模型")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API 密钥 (或设置 DASHSCOPE_API_KEY 环境变量)")
    parser.add_argument("--api_base_url", type=str,
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help="API Base URL")
    parser.add_argument("--output_dir", type=str, default="./cache/api_distill",
                        help="输出缓存目录 (默认 ./cache/api_distill)")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="自定义 prompt 文件路径 (每行一条 prompt)")
    parser.add_argument("--use_fineweb", action="store_true", default=False,
                        help="使用 HuggingFace fineweb-edu 数据集作为 prompt 来源")
    parser.add_argument("--fineweb_parquet", type=str, default=None,
                        help="本地 parquet 文件路径（手动下载后使用，优先于在线下载）")
    parser.add_argument("--fineweb_config", type=str, default="sample-10BT",
                        help="fineweb-edu 配置名 (默认 sample-10BT)")
    parser.add_argument("--fineweb_split", type=str, default="train",
                        help="fineweb-edu 数据集 split (默认 train)")
    parser.add_argument("--prefix_tokens", type=int, default=64,
                        help="截取文本前 N 个词作为续写 prompt (默认 64)")
    parser.add_argument("--min_text_length", type=int, default=200,
                        help="fineweb 文本最短字符数过滤阈值 (默认 200)")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="每条生成的最大 token 数 (默认 2048)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="生成温度 (默认 0.7)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p 采样 (默认 0.9)")
    parser.add_argument("--use_logprobs", action="store_true", default=True,
                        help="是否请求 logprobs (默认开启)")
    parser.add_argument("--no_logprobs", action="store_true",
                        help="关闭 logprobs 请求")
    parser.add_argument("--logprob_top_k", type=int, default=5,
                        help="请求的 top-K logprobs 数量 (默认 5, 部分模型最大支持 5)")
    parser.add_argument("--request_delay", type=float, default=0.3,
                        help="请求间隔秒数, 避免限流 (默认 0.3)")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="单条请求最大重试次数 (默认 5)")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="单条请求超时秒数 (默认 120)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="自动跳过已缓存的 prompt (默认开启)")
    parser.add_argument("--workers", type=int, default=10,
                        help="并发请求线程数 (默认 10, 上限 32)")
    parser.add_argument("--max_workers", type=int, default=32,
                        help="并发线程数上限 (默认 32)")

    return parser.parse_args()


def load_prompts_from_fineweb(
    num_samples: int,
    config: str = "sample-10BT",
    split: str = "train",
    prefix_tokens: int = 64,
    min_text_length: int = 200,
    local_parquet: str = None,
) -> list:
    """
    从 fineweb-edu 数据集提取文本前缀作为续写 prompt。
    优先使用本地 parquet 文件（--fineweb_parquet），避免网络问题。
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("请先安装 pandas: pip install pandas pyarrow")

    parquet_paths = []

    # 优先使用本地文件
    if local_parquet:
        if os.path.isfile(local_parquet):
            parquet_paths = [local_parquet]
            logger.info(f"  使用本地 parquet 文件: {local_parquet}")
        elif os.path.isdir(local_parquet):
            parquet_paths = sorted([
                os.path.join(local_parquet, f)
                for f in os.listdir(local_parquet)
                if f.endswith(".parquet")
            ])
            logger.info(f"  从目录加载 {len(parquet_paths)} 个 parquet 文件: {local_parquet}")
        else:
            raise FileNotFoundError(f"找不到文件或目录: {local_parquet}")
    else:
        # 在线下载（用 wget/curl 更可靠，这里仅作 fallback）
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        try:
            from huggingface_hub import hf_hub_download
            # sample-10BT 的实际路径
            shard_name = f"sample/10BT/train-00000-of-00099.parquet"
            logger.info(f"  在线下载: {shard_name}")
            local_path = hf_hub_download(
                repo_id="HuggingFaceFW/fineweb-edu",
                filename=shard_name,
                repo_type="dataset",
            )
            parquet_paths = [local_path]
        except Exception as e:
            raise RuntimeError(
                f"在线下载失败: {e}\n\n"
                f"请手动下载后使用 --fineweb_parquet 参数:\n"
                f"  wget 'https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu"
                f"/resolve/main/sample/10BT/train-00000-of-00099.parquet' -O /root/fw_shard0.parquet\n"
                f"  python scripts/generate_api_data.py --use_fineweb "
                f"--fineweb_parquet /root/fw_shard0.parquet ..."
            )

    prompts = []
    seen = set()

    for parquet_path in parquet_paths:
        if len(prompts) >= num_samples:
            break
        try:
            df = pd.read_parquet(parquet_path, columns=["text"])
            logger.info(f"  读取 {parquet_path}: {len(df)} 条记录")
            for text in df["text"]:
                if len(prompts) >= num_samples:
                    break
                text = str(text).strip()
                if len(text) < min_text_length:
                    continue
                words = text.split()
                prefix = " ".join(words[:prefix_tokens])
                if prefix in seen:
                    continue
                seen.add(prefix)
                prompts.append(prefix)
            logger.info(f"  累计提取 prompt: {len(prompts)}/{num_samples}")
        except Exception as e:
            logger.warning(f"  读取失败 ({parquet_path}): {e}")

    if not prompts:
        raise RuntimeError("无法提取 prompt，请检查 parquet 文件是否有效")

    logger.info(f"  共提取 {len(prompts)} 条唯一 prompt")
    if len(prompts) < num_samples:
        logger.warning(f"  少于目标 {num_samples}，循环补齐")
        repeats = (num_samples // max(len(prompts), 1)) + 1
        prompts = (prompts * repeats)[:num_samples]

    return prompts

    # 确保使用镜像
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    logger.info(f"正在从 HuggingFace 下载 fineweb-edu parquet 文件 ({config})...")
    logger.info(f"  HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")

    # 先列出仓库文件，找到实际 parquet 路径
    try:
        from huggingface_hub import list_repo_files
        all_files = list(list_repo_files(
            "HuggingFaceFW/fineweb-edu",
            repo_type="dataset",
        ))
    except Exception as e:
        raise RuntimeError(f"无法列出仓库文件: {e}")

    # 根据 config 名筛选对应 parquet 文件
    config_key = config.replace("-", "/").lower()  # sample-10BT → sample/10bt
    parquet_files = [
        f for f in all_files
        if f.endswith(".parquet") and config_key in f.lower()
    ]
    if not parquet_files:
        # fallback: 列出所有 parquet 供调试
        all_parquet = [f for f in all_files if f.endswith(".parquet")]
        sample = all_parquet[:5]
        raise RuntimeError(
            f"找不到 config={config} 对应的 parquet 文件。\n"
            f"仓库中 parquet 示例 (前5): {sample}\n"
            f"请用 --fineweb_config 指定正确的 config 名称。"
        )

    parquet_files = sorted(parquet_files)
    logger.info(f"  找到 {len(parquet_files)} 个 parquet 分片，开始下载...")

    prompts = []
    seen = set()

    for filename in parquet_files:
        if len(prompts) >= num_samples:
            break
        local_path = None
        try:
            logger.info(f"  下载分片: {filename}")
            local_path = hf_hub_download(
                repo_id="HuggingFaceFW/fineweb-edu",
                filename=filename,
                repo_type="dataset",
                revision="main",
            )
            logger.info(f"  已下载: {local_path}")
        except Exception as e:
            logger.warning(f"  下载失败 ({filename}): {e}")
            continue

        try:
            df = pd.read_parquet(local_path, columns=["text"])
            logger.info(f"  读取到 {len(df)} 条记录")

            for text in df["text"]:
                if len(prompts) >= num_samples:
                    break
                text = str(text).strip()
                if len(text) < min_text_length:
                    continue
                words = text.split()
                prefix = " ".join(words[:prefix_tokens])
                if prefix in seen:
                    continue
                seen.add(prefix)
                prompts.append(prefix)

            logger.info(f"  累计提取 prompt: {len(prompts)}/{num_samples}")
        except Exception as e:
            logger.warning(f"  读取 parquet 失败: {e}")
            continue

        if len(prompts) >= num_samples:
            break

    if not prompts:
        raise RuntimeError("无法从 fineweb-edu 获取数据，请检查网络或手动设置 HF_ENDPOINT")

    logger.info(f"  从 fineweb-edu 提取了 {len(prompts)} 条唯一 prompt")

    # 不足时循环补齐
    if len(prompts) < num_samples:
        logger.warning(f"  实际获取 {len(prompts)} 条，少于目标 {num_samples}，循环补齐")
        repeats = (num_samples // max(len(prompts), 1)) + 1
        prompts = (prompts * repeats)[:num_samples]

    return prompts


def load_prompts(args) -> list:
    """
    加载 prompt 列表，支持三种来源：
    1. fineweb-edu 数据集（--use_fineweb）
    2. 自定义文件（--prompt_file）
    3. 内置种子 prompt（默认）
    """
    num_samples = args.num_samples

    if args.use_fineweb:
        logger.info("数据来源: HuggingFace fineweb-edu")
        return load_prompts_from_fineweb(
            num_samples=num_samples,
            config=args.fineweb_config,
            split=args.fineweb_split,
            prefix_tokens=args.prefix_tokens,
            min_text_length=args.min_text_length,
            local_parquet=args.fineweb_parquet,
        )

    if args.prompt_file and os.path.exists(args.prompt_file):
        logger.info(f"数据来源: 自定义文件 {args.prompt_file}")
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        logger.info(f"  加载了 {len(prompts)} 条唯一 prompt")
        if len(prompts) < num_samples:
            repeats = (num_samples // len(prompts)) + 1
            prompts = (prompts * repeats)[:num_samples]
        return prompts

    logger.info(f"数据来源: 内置种子 prompt ({len(DEFAULT_SEED_PROMPTS)} 条，循环复用)")
    repeats = (num_samples // len(DEFAULT_SEED_PROMPTS)) + 1
    return (DEFAULT_SEED_PROMPTS * repeats)[:num_samples]


def generate_single(client, prompt: str, args, cache_dir: Path, sample_idx: int = 0) -> dict:
    """
    生成单条数据并缓存到磁盘。

    如果缓存已存在则跳过（支持断点续传）。
    返回 dict: {"text": ..., "cached": True/False, "tokens": ...}
    """
    # 加入 sample_idx，使相同 prompt 的不同次调用拥有独立缓存
    cache_key = hashlib.md5(
        f"{sample_idx}:{prompt}:{args.api_model}:{args.temperature}".encode()
    ).hexdigest()
    cache_file = cache_dir / f"{cache_key}.json"

    # 检查缓存
    if cache_file.exists():
        return {"text": "", "cached": True, "cache_file": str(cache_file)}

    use_logprobs = args.use_logprobs and not args.no_logprobs

    # 调用 API
    for attempt in range(args.max_retries):
        try:
            kwargs = {
                "model": args.api_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
            if use_logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = min(args.logprob_top_k, 5)  # 部分模型最大支持 5

            response = client.chat.completions.create(**kwargs)

            result = {
                "text": response.choices[0].message.content,
                "finish_reason": response.choices[0].finish_reason,
                "prompt": prompt,
                "model": args.api_model,
                "temperature": args.temperature,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }

            # 提取 logprobs
            if use_logprobs and response.choices[0].logprobs:
                token_logprobs = []
                for token_info in response.choices[0].logprobs.content:
                    entry = {
                        "token": token_info.token,
                        "logprob": token_info.logprob,
                        "top_logprobs": [
                            {"token": lp.token, "logprob": lp.logprob}
                            for lp in (token_info.top_logprobs or [])
                        ],
                    }
                    token_logprobs.append(entry)
                result["logprobs"] = token_logprobs

            # 写入缓存
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            return {
                "text": result["text"],
                "cached": False,
                "tokens": result["usage"]["total_tokens"],
                "cache_file": str(cache_file),
            }

        except Exception as e:
            wait = args.request_delay * (2 ** attempt)
            logger.warning(f"  请求失败 (尝试 {attempt+1}/{args.max_retries}): {e}")
            if attempt < args.max_retries - 1:
                logger.info(f"  等待 {wait:.1f}s 后重试...")
                time.sleep(wait)
            else:
                logger.error(f"  放弃: {prompt[:50]}...")
                return {"text": "", "cached": False, "error": str(e)}


def main():
    args = parse_args()

    # API key
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        logger.error(
            "未设置 API 密钥！请通过以下方式之一设置:\n"
            "  1. 环境变量: export DASHSCOPE_API_KEY='sk-xxxxx'\n"
            "  2. 命令行参数: --api_key sk-xxxxx"
        )
        sys.exit(1)

    # 初始化 OpenAI 兼容客户端
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("需要安装 openai 包: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=args.api_base_url, timeout=args.timeout)

    # 准备输出目录
    cache_dir = Path(args.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 加载 prompt
    prompts = load_prompts(args)

    logger.info("=" * 60)
    logger.info("  API 数据生成 (本地运行, 无需 GPU)")
    logger.info("=" * 60)
    logger.info(f"  API 模型:     {args.api_model}")
    logger.info(f"  生成数量:     {len(prompts)} 条")
    logger.info(f"  max_tokens:   {args.max_tokens}")
    logger.info(f"  temperature:  {args.temperature}")
    logger.info(f"  logprobs:     {args.use_logprobs and not args.no_logprobs}")
    logger.info(f"  输出目录:     {args.output_dir}")
    logger.info(f"  请求间隔:     {args.request_delay}s")
    logger.info("=" * 60)

    # 统计（线程安全）
    total_api_tokens = 0
    generated_count = 0
    cached_count = 0
    failed_count = 0
    done_count = 0
    start_time = time.time()
    stats_lock = threading.Lock()

    num_workers = min(args.workers, args.max_workers)
    logger.info(f"  并发线程数:   {num_workers}")
    logger.info("=" * 60)

    def _worker(item):
        i, prompt = item
        return i, generate_single(client, prompt, args, cache_dir, sample_idx=i)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_worker, (i, p)): i for i, p in enumerate(prompts)}
        for future in as_completed(futures):
            i, result = future.result()
            with stats_lock:
                if result.get("cached"):
                    cached_count += 1
                elif result.get("error"):
                    failed_count += 1
                else:
                    generated_count += 1
                    total_api_tokens += result.get("tokens", 0)
                done_count += 1
                done = done_count
                _gen = generated_count
                _cached = cached_count
                _failed = failed_count
                _tokens = total_api_tokens

            if done % 10 == 0 or done == len(prompts):
                elapsed = time.time() - start_time
                rate = done / max(elapsed, 0.1)
                remaining = (len(prompts) - done) / max(rate, 0.01)
                logger.info(
                    f"  进度: {done}/{len(prompts)} | "
                    f"新生成: {_gen} | 已缓存: {_cached} | 失败: {_failed} | "
                    f"API token: {_tokens:,} | "
                    f"剩余: ~{remaining/60:.1f}min"
                )

    # 最终统计
    elapsed = time.time() - start_time
    total_cached_files = len(list(cache_dir.glob("*.json")))

    logger.info("")
    logger.info("=" * 60)
    logger.info("  数据生成完成!")
    logger.info("=" * 60)
    logger.info(f"  新生成:         {generated_count} 条")
    logger.info(f"  已缓存 (跳过):  {cached_count} 条")
    logger.info(f"  失败:           {failed_count} 条")
    logger.info(f"  缓存目录文件数: {total_cached_files} 个")
    logger.info(f"  总 API token:   {total_api_tokens:,}")
    logger.info(f"  耗时:           {elapsed/60:.1f} 分钟")
    logger.info(f"  缓存目录:       {args.output_dir}")
    logger.info("")
    logger.info("  下一步: 将缓存目录拷贝到 GPU 服务器, 然后运行训练:")
    logger.info(f"    scp -r {args.output_dir} user@server:/path/to/mamba/cache/api_distill")
    logger.info(f"    python scripts/train_distill.py --linear_type mamba --use_api \\")
    logger.info(f"        --cached_data_dir {args.output_dir}")
    logger.info("=" * 60)

    # 保存生成统计信息
    stats = {
        "timestamp": datetime.now().isoformat(),
        "api_model": args.api_model,
        "num_requested": len(prompts),
        "num_generated": generated_count,
        "num_cached": cached_count,
        "num_failed": failed_count,
        "total_api_tokens": total_api_tokens,
        "total_cached_files": total_cached_files,
        "elapsed_seconds": round(elapsed, 1),
        "settings": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "use_logprobs": args.use_logprobs and not args.no_logprobs,
        },
    }
    stats_file = cache_dir / "_generation_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"  统计信息已保存到: {stats_file}")


if __name__ == "__main__":
    main()
