from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InterviewQuestion:
    """One interview question with a toy-level trap answer and the pro keywords.

    The articles frame every question as "菜鸟怎么答 vs 资深怎么答".  Here that
    becomes runnable data: `pro_keywords` are the工业级 talking points a strong
    answer must hit, and `grade()` scores a candidate answer by coverage.
    """

    topic: str
    question: str
    toy_answer: str
    pro_keywords: tuple[str, ...]
    source_chapter: int

    def grade(self, candidate: str) -> dict[str, Any]:
        hit = [kw for kw in self.pro_keywords if kw in candidate]
        missed = [kw for kw in self.pro_keywords if kw not in candidate]
        coverage = len(hit) / len(self.pro_keywords) if self.pro_keywords else 0.0
        level = "工业级" if coverage >= 0.6 else ("及格" if coverage >= 0.3 else "玩具级")
        return {
            "question": self.question,
            "coverage": round(coverage, 2),
            "level": level,
            "hit_keywords": hit,
            "missed_keywords": missed,
        }


# 上篇（第 49 篇）：原理与进阶高频题
PRINCIPLE_QUESTIONS: tuple[InterviewQuestion, ...] = (
    InterviewQuestion(
        "LLM", "大模型逐字生成文本的本质是什么？",
        "模型根据问题生成回答。",
        ("自回归", "Token", "概率分布", "采样"), 2,
    ),
    InterviewQuestion(
        "LLM", "长上下文模型能取代 RAG 吗？",
        "窗口够大就能取代 RAG。",
        ("Lost in the Middle", "成本", "时效性", "互补"), 5,
    ),
    InterviewQuestion(
        "RAG", "RAG 召回率上不去怎么排查优化？",
        "换更好的 embedding 或调大 top-k。",
        ("召回", "精排", "Hybrid", "rerank", "query 改写"), 11,
    ),
    InterviewQuestion(
        "RAG", "Cross-Encoder 和 Bi-Encoder 的区别？",
        "都是算相似度的模型。",
        ("双塔", "初筛", "精排", "拼在一起"), 14,
    ),
    InterviewQuestion(
        "Memory", "记忆检索和 RAG 是一回事吗？",
        "都是向量检索，一样的。",
        ("写回", "演化", "按用户隔离", "动态"), 16,
    ),
    InterviewQuestion(
        "Tool", "Function Calling 时模型真的执行了函数吗？",
        "模型调用了我的函数。",
        ("不执行", "JSON", "意图", "应用层"), 6,
    ),
    InterviewQuestion(
        "Tool", "MCP 和 Function Calling 是什么关系？",
        "MCP 替代了 Function Calling。",
        ("标准", "互补", "发现", "N×M"), 6,
    ),
    InterviewQuestion(
        "AgentLoop", "Agent 循环怎么安全停下来？",
        "设个最大轮数就行。",
        ("最大迭代", "token 预算", "重复动作", "状态机"), 7,
    ),
    InterviewQuestion(
        "MultiAgent", "什么时候该上多 Agent？",
        "任务复杂就上多 Agent。",
        ("瓶颈", "上下文隔离", "级联", "单 Agent"), 8,
    ),
    InterviewQuestion(
        "Framework", "LangGraph 的 Checkpoint 和 HITL 怎么实现？",
        "就是存个状态。",
        ("快照", "thread_id", "interrupt", "持久化", "恢复"), 21,
    ),
)

# 下篇（第 50 篇）：工程化高频题
ENGINEERING_QUESTIONS: tuple[InterviewQuestion, ...] = (
    InterviewQuestion(
        "Gateway", "LLM 网关限流为什么不能只限请求数？",
        "用 Redis 限 QPS 就行。",
        ("TPM", "Token", "预扣", "校正"), 34,
    ),
    InterviewQuestion(
        "Reliability", "工具/模型超时怎么设计重试熔断降级？",
        "失败就重试几次。",
        ("指数退避", "可重试", "熔断", "降级", "幂等"), 44,
    ),
    InterviewQuestion(
        "Streaming", "AI 流式输出为什么首选 SSE 而非 WebSocket？",
        "WebSocket 更强大所以用它。",
        ("单向", "轻量", "自动重连", "HTTP"), 45,
    ),
    InterviewQuestion(
        "Security", "Prompt Injection 和 SQL 注入本质一样吗？",
        "完全不一样。",
        ("数据被当指令", "同一通道", "无法隔离", "不能根除"), 40,
    ),
    InterviewQuestion(
        "Security", "Agent 越权的根因？RBAC 怎么落地？",
        "加个权限判断。",
        ("高权限身份", "透传", "最小权限", "tenant_id", "审计"), 41,
    ),
    InterviewQuestion(
        "Cost", "语义缓存和精确缓存的区别？有什么坑？",
        "缓存命中就返回。",
        ("embedding", "相似度", "阈值", "错误命中"), 44,
    ),
    InterviewQuestion(
        "Observability", "解释 Trace 和 Span，为什么 LLM 更难观测？",
        "看看日志就行。",
        ("Trace", "Span", "非确定性", "链路长"), 38,
    ),
    InterviewQuestion(
        "Eval", "RAGAS 三个核心指标分别评什么？",
        "就是评准确率。",
        ("Faithfulness", "Context Precision", "Answer Relevance", "幻觉"), 36,
    ),
)


# 系统设计大题的标准答题框架
SYSTEM_DESIGN_FRAMEWORK: tuple[str, ...] = (
    "① 澄清需求（对内/对外、是否执行操作、并发量、更新频率、合规）",
    "② 画整体架构（套 7 层：交互/编排/RAG/Memory/Tool/网关/安全/可观测）",
    "③ 逐层展开关键设计",
    "④ 主动暴露难点与权衡（这里拿大分）",
    "⑤ 谈评测、安全、成本",
    "⑥ 给演进路径（先单 Agent，量大再拆多 Agent）",
)

SYSTEM_DESIGN_CASES: dict[str, dict[str, Any]] = {
    "智能客服 Agent": {
        "clarify": ["对内还是对外", "是否执行退款等操作", "并发量", "知识库更新频率"],
        "hard_points": ["置信度不足转人工(escalation)", "上下文超长做摘要", "防注入+RBAC", "语义缓存+token预算熔断"],
        "eval": "RAGAS + BadCase 闭环",
    },
    "Text-to-SQL": {
        "clarify": ["库有多大", "只读还是可写", "用户懂不懂 SQL", "错误容忍度"],
        "hard_points": ["Schema Linking 检索相关表", "只读账号+SQL Guard+白名单+LIMIT", "ReAct 错误自纠", "schema registry 统一真源"],
        "eval": "execution accuracy（执行结果是否正确）",
    },
}


@dataclass
class InterviewProject:
    """Continuous project for chapters 49-50 (面试番外篇).

    Turns the two interview articles into a runnable self-check: a question
    bank with toy-vs-pro answers, a keyword-coverage grader, and the system
    design answer framework.
    """

    run_log: list[str] = field(default_factory=list)

    def run_stage(self, chapter: int) -> dict[str, Any]:
        if chapter == 49:
            return self.stage_49_principles()
        if chapter == 50:
            return self.stage_50_engineering()
        raise ValueError(f"chapter {chapter} is outside the supported range 49-50")

    def _record(self, stage: int, name: str, summary: str, details: dict[str, Any]) -> dict[str, Any]:
        self.run_log.append(name)
        return {
            "project": "interview_prep",
            "chapter": stage,
            "stage_name": name,
            "summary": summary,
            "details": details,
            "run_log_size": len(self.run_log),
        }

    def stage_49_principles(self) -> dict[str, Any]:
        # 演示评分器：一个工业级答案 vs 一个玩具级答案
        rag_question = PRINCIPLE_QUESTIONS[2]
        pro_answer = "先分召回和精排两个阶段：召回用 query 改写 + Hybrid，精排用 rerank；rerank 救不了没召回的内容。"
        toy_answer = "换个更好的模型。"
        return self._record(
            49,
            "interview_principles",
            "原理与进阶面试题：每题给菜鸟答法与资深答法，并用关键词覆盖率打分。",
            {
                "topics": sorted({q.topic for q in PRINCIPLE_QUESTIONS}),
                "question_count": len(PRINCIPLE_QUESTIONS),
                "sample_questions": [
                    {"topic": q.topic, "question": q.question, "toy_answer": q.toy_answer, "chapter": q.source_chapter}
                    for q in PRINCIPLE_QUESTIONS[:3]
                ],
                "grader_demo": {
                    "pro_answer_graded": rag_question.grade(pro_answer),
                    "toy_answer_graded": rag_question.grade(toy_answer),
                },
            },
        )

    def stage_50_engineering(self) -> dict[str, Any]:
        gateway_q = ENGINEERING_QUESTIONS[0]
        pro_answer = "不能只限请求数，要做 TPM 多维限流；难点是只知道 input Token，所以预扣 + 流式计量 + 结束后校正。"
        graded_all = [q.grade(pro_answer) if q is gateway_q else None for q in ENGINEERING_QUESTIONS]
        return self._record(
            50,
            "interview_engineering_system_design",
            "工程化与系统设计面试题 + 系统设计大题的标准答题框架。",
            {
                "topics": sorted({q.topic for q in ENGINEERING_QUESTIONS}),
                "question_count": len(ENGINEERING_QUESTIONS),
                "gateway_grader_demo": graded_all[0],
                "system_design_framework": list(SYSTEM_DESIGN_FRAMEWORK),
                "system_design_cases": SYSTEM_DESIGN_CASES,
                "scoring_rule": "60% 分在'先澄清需求'和'主动讲权衡'，40% 在架构本身",
            },
        )


def run_interview_stage(chapter: int) -> dict[str, Any]:
    return InterviewProject().run_stage(chapter)
