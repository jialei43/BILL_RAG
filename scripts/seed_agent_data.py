# scripts/seed_agent_data.py
# 多智能体系统种子数据初始化脚本
# 功能：
#   1. 创建 14 张新增表（如已存在则跳过）
#   2. 向每张表插入约 100 条测试数据，覆盖各种业务场景
#   3. 数据分布符合真实业务比例（如 80% 合规、20% 违规）
# 运行方式：python scripts/seed_agent_data.py
# 前置条件：PostgreSQL 已启动，tenants/users/documents/bill_records 等基础表已存在数据

import asyncio       # 异步执行主函数
import uuid          # 生成随机 UUID
import random        # 生成随机数据，模拟真实业务分布
from datetime import datetime, timedelta  # 日期时间处理
from typing import List                   # 类型注解

from loguru import logger  # 日志库

# 将项目根目录加入 Python 路径，确保可以导入 app 模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 插入项目根目录

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, func  # select 用于查询，func 用于聚合

from config.settings import settings       # 读取 DATABASE_URL 等配置
from app.models.db_models import Base, Tenant, Document, BillRecord  # 基础表 ORM
from app.models.agent_models import (      # 14 张新增表的 ORM 类
    AuditTask, AuditTaskStatus, AuditTaskType,
    BillElement,
    ComplianceCheck, ViolationLevel,
    EndorsementChain,
    ContractReview,
    RiskAssessment, RiskLevel,
    AuditReport,
    FlowTrackingTask, FlowTrackingStatus,
    FlowMessage, FlowMessageStatus,
    FraudDetection,
    BatchTask, BatchTaskStatus,
    BatchTaskItem,
    ErrorCodeMapping,
    BlacklistEntity, EntityType,
)


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def gen_id() -> str:
    """生成 UUID4 格式的唯一标识符"""
    return str(uuid.uuid4())


def rand_date(days_back: int = 365) -> datetime:
    """生成过去 days_back 天内的随机日期"""
    return datetime.now() - timedelta(days=random.randint(0, days_back))


def rand_date_str(days_future: int = 365) -> str:
    """生成未来 days_future 天内的随机日期字符串（YYYY-MM-DD）"""
    return (datetime.now() + timedelta(days=random.randint(30, days_future))).strftime("%Y-%m-%d")


def rand_amount() -> float:
    """生成随机票面金额（10万~5000万之间的整万数）"""
    return round(random.choice([10, 20, 50, 100, 200, 500, 1000, 2000, 5000]) * 10000 * random.uniform(0.5, 2.0), 2)


def rand_company() -> str:
    """生成随机公司名称（用于出票人/背书人等字段）"""
    prefixes = ["中国", "华夏", "光大", "民生", "交通", "建设", "工商", "农业", "招商", "浦发"]
    types = ["银行", "贸易", "科技", "实业", "商业", "工贸", "进出口", "供应链", "物流", "金融"]
    suffixes = ["有限公司", "股份有限公司", "有限责任公司", "集团有限公司"]
    return f"{random.choice(prefixes)}{random.choice(types)}{random.choice(suffixes)}"


def rand_bank() -> str:
    """生成随机银行名称"""
    banks = [
        "中国工商银行股份有限公司上海支行", "中国建设银行股份有限公司北京分行",
        "中国农业银行股份有限公司深圳支行", "招商银行股份有限公司广州分行",
        "中国银行股份有限公司杭州支行", "交通银行股份有限公司南京分行",
        "浦发银行股份有限公司成都分行", "光大银行股份有限公司武汉分行",
        "民生银行股份有限公司重庆分行", "华夏银行股份有限公司天津分行",
    ]
    return random.choice(banks)


def rand_ticket_number() -> str:
    """生成随机票据号码（模拟真实格式）"""
    prefix = random.choice(["SHBH", "BJCH", "GZBH", "SZBH", "HZBH"])
    return f"{prefix}{datetime.now().strftime('%Y%m%d')}{random.randint(100000, 999999)}"


# ──────────────────────────────────────────────────────────────────────────────
# 基础数据准备：获取现有租户和文档 ID（种子数据需要合法的外键）
# ──────────────────────────────────────────────────────────────────────────────

async def get_base_ids(db: AsyncSession) -> tuple:
    """
    查询现有基础数据 ID，用于新表记录的外键引用
    如果基础表为空，自动创建最小化测试数据

    Returns:
        (tenant_ids, document_ids, bill_record_ids) 三个列表
    """
    # 查询现有租户 ID
    tenant_result = await db.execute(select(Tenant.id).limit(10))
    tenant_ids = [row[0] for row in tenant_result.fetchall()]

    # 如果租户表为空，创建测试租户
    if not tenant_ids:
        logger.info("[seed] 租户表为空，创建测试租户...")
        from app.models.db_models import TenantStatus
        test_tenants = []
        for i in range(3):  # 创建 3 个测试租户
            t = Tenant(
                id=gen_id(),
                name=f"测试银行{i+1}分行",
                code=f"TEST_BANK_{i+1:02d}",
                license_no=f"{random.randint(10**17, 10**18-1)}",  # 18位统一信用代码
                status=TenantStatus.ACTIVE,
                doc_quota=10000,
                qps_limit=20,
                milvus_partition=f"tenant_TEST_BANK_{i+1:02d}",
            )
            test_tenants.append(t)
            db.add(t)
        await db.flush()  # 刷新到数据库（不提交），获取自动生成的 ID
        tenant_ids = [t.id for t in test_tenants]
        logger.info(f"[seed] 已创建 {len(tenant_ids)} 个测试租户")

    # 查询现有文档 ID
    doc_result = await db.execute(select(Document.id).limit(20))
    document_ids = [row[0] for row in doc_result.fetchall()]

    # 如果文档表为空，创建测试文档
    if not document_ids:
        logger.info("[seed] 文档表为空，创建测试文档...")
        from app.models.db_models import DocumentStatus
        test_docs = []
        for i in range(20):  # 创建 20 个测试文档
            d = Document(
                id=gen_id(),
                tenant_id=random.choice(tenant_ids),
                filename=f"票据样本_{i+1:03d}.pdf",
                file_path=f"./data/uploads/test_{i+1:03d}.pdf",
                file_type="pdf",
                file_size=random.randint(50000, 500000),
                md5_hash=uuid.uuid4().hex,  # 随机 MD5（32位十六进制）
                status=DocumentStatus.COMPLETED,
                chunk_count=random.randint(5, 30),
                page_count=random.randint(1, 10),
            )
            test_docs.append(d)
            db.add(d)
        await db.flush()
        document_ids = [d.id for d in test_docs]
        logger.info(f"[seed] 已创建 {len(document_ids)} 个测试文档")

    # 查询现有票据主档 ID
    bill_result = await db.execute(select(BillRecord.id).limit(20))
    bill_record_ids = [row[0] for row in bill_result.fetchall()]

    # 如果票据表为空，创建测试票据
    if not bill_record_ids:
        logger.info("[seed] 票据主档表为空，创建测试票据...")
        test_bills = []
        for i in range(20):  # 创建 20 张测试票据
            b = BillRecord(
                id=gen_id(),
                tenant_id=random.choice(tenant_ids),
                ticket_number=rand_ticket_number(),
                ticket_type=random.choice(["银行承兑汇票", "商业承兑汇票"]),
                issue_date=(datetime.now() - timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d"),
                due_date=(datetime.now() + timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
                amount_numeric=rand_amount(),
                amount_text=random.choice(["壹佰万元整", "贰百万元整", "伍佰万元整", "壹仟万元整"]),
                currency="人民币",
                drawer=rand_company(),
                acceptor=rand_bank(),
                payee=rand_company(),
                drawee_bank=rand_bank(),
                risk_flags=[],
                latest_version=1,
            )
            test_bills.append(b)
            db.add(b)
        await db.flush()
        bill_record_ids = [b.id for b in test_bills]
        logger.info(f"[seed] 已创建 {len(bill_record_ids)} 张测试票据")

    return tenant_ids, document_ids, bill_record_ids


# ──────────────────────────────────────────────────────────────────────────────
# 各表种子数据生成函数
# ──────────────────────────────────────────────────────────────────────────────

async def seed_audit_tasks(db: AsyncSession, tenant_ids: List[str], document_ids: List[str]) -> List[str]:
    """
    插入 100 条审核主任务种子数据
    分布：20 pending / 20 running / 40 completed / 20 failed；覆盖 3 种 task_type
    Returns:
        audit_task_ids: 供后续表作外键引用
    """
    logger.info("[seed] 开始插入 audit_tasks 种子数据...")
    task_ids = []  # 收集生成的任务 ID，供其他表作外键

    # 定义状态分布：(状态, 数量) 列表，合计 100 条
    status_distribution = [
        (AuditTaskStatus.PENDING,   20),  # 20 条待处理
        (AuditTaskStatus.RUNNING,   20),  # 20 条执行中
        (AuditTaskStatus.COMPLETED, 40),  # 40 条已完成（最多，代表历史审核记录）
        (AuditTaskStatus.FAILED,    20),  # 20 条失败（约 20% 失败率符合真实场景）
    ]

    task_types = [  # 三种常用业务类型
        AuditTaskType.FULL_AUDIT,
        AuditTaskType.DISCOUNT_APPLY,
        AuditTaskType.ACCEPTANCE_PROMPT,
    ]

    for status, count in status_distribution:
        for i in range(count):
            task_id = gen_id()
            task_ids.append(task_id)
            task = AuditTask(
                id=task_id,
                tenant_id=random.choice(tenant_ids),
                document_id=random.choice(document_ids),
                task_type=random.choice(task_types),
                status=status,
                priority=random.randint(1, 10),   # 随机优先级
                progress_pct=(
                    0.0 if status == AuditTaskStatus.PENDING
                    else random.uniform(10.0, 90.0) if status == AuditTaskStatus.RUNNING
                    else 100.0 if status == AuditTaskStatus.COMPLETED
                    else random.uniform(0.0, 80.0)  # FAILED 时进度不确定
                ),
                current_agent=(
                    None if status in (AuditTaskStatus.PENDING, AuditTaskStatus.COMPLETED)
                    else random.choice(["document_parser_agent", "element_extraction_agent",
                                        "compliance_retrieval_agent", "risk_assessment_agent"])
                ),
                error_code=("SYS_AGENT_EXCEPTION" if status == AuditTaskStatus.FAILED else None),
                error_msg=("Agent 执行超时" if status == AuditTaskStatus.FAILED else None),
                started_at=(None if status == AuditTaskStatus.PENDING else rand_date(30)),
                completed_at=(rand_date(30) if status in (AuditTaskStatus.COMPLETED, AuditTaskStatus.FAILED) else None),
                dag_plan={"nodes": ["document_parser", "element_extraction", "compliance_retrieval"],
                          "edges": [["document_parser", "element_extraction"]]},  # 示例 DAG
            )
            db.add(task)

    await db.flush()  # 刷新以获取数据库生成的时间戳
    logger.info(f"[seed] audit_tasks 插入完成，共 {len(task_ids)} 条")
    return task_ids


async def seed_bill_elements(db: AsyncSession, task_ids: List[str]) -> None:
    """
    插入 100 条票据要素抽取结果种子数据
    分布：30 条高置信(>0.9) / 50 条中置信(0.7~0.9) / 20 条低置信(<0.7)
    """
    logger.info("[seed] 开始插入 bill_elements 种子数据...")

    # 高/中/低置信度的分布数量
    confidence_distribution = [
        (0.91, 0.99, 30),   # 高置信：30 条
        (0.70, 0.90, 50),   # 中置信：50 条
        (0.40, 0.69, 20),   # 低置信：20 条
    ]

    used_tasks = random.sample(task_ids, min(100, len(task_ids)))  # 取 100 个任务 ID（如不足则全取）
    task_iter = iter(used_tasks)   # 迭代器，依次为每条记录分配任务 ID

    for low_conf, high_conf, count in confidence_distribution:
        for _ in range(count):
            try:
                task_id = next(task_iter)  # 获取下一个任务 ID
            except StopIteration:
                break  # 任务 ID 用尽则停止

            confidence = round(random.uniform(low_conf, high_conf), 3)  # 随机置信度
            amount = rand_amount()

            elem = BillElement(
                id=gen_id(),
                audit_task_id=task_id,
                ticket_number=rand_ticket_number(),
                ticket_type=random.choice(["银行承兑汇票", "商业承兑汇票"]),
                issue_date=(datetime.now() - timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d"),
                due_date=rand_date_str(),
                amount_numeric=amount,
                amount_text=random.choice(["壹佰万元整", "贰百万元整", "伍佰万元整", "壹仟万元整"]),
                currency="人民币",
                drawer=rand_company(),
                drawer_account=f"62{random.randint(10**17, 10**18-1)}",  # 模拟 19 位银行卡号
                drawer_bank=rand_bank(),
                acceptor=rand_bank(),
                payee=rand_company(),
                drawee_bank=rand_bank(),
                endorsers=[rand_company() for _ in range(random.randint(0, 4))],  # 0~4 个背书人
                maturity_days=random.randint(30, 365),
                trade_purpose=random.choice(["货款结算", "工程款支付", "设备采购款", "原材料采购", "服务费结算"]),
                acceptance_clause=random.choice([None, "利率按 LPR+50BP 浮动", "到期自动兑付", None]),
                special_remarks=random.choice([None, None, "限于背书转让", "不可撤销"]),
                confidence_score=confidence,
                field_confidences={
                    "ticket_number": round(random.uniform(confidence - 0.1, min(confidence + 0.1, 1.0)), 3),
                    "amount_numeric": round(random.uniform(confidence - 0.1, min(confidence + 0.1, 1.0)), 3),
                },
                extraction_method=random.choice(["ocr_only", "vision_llm", "hybrid"]),
            )
            db.add(elem)

    await db.flush()
    logger.info("[seed] bill_elements 插入完成，共 100 条")


async def seed_compliance_checks(db: AsyncSession, task_ids: List[str]) -> None:
    """
    插入 100 条合规检查项种子数据
    分布：80% is_compliant=True / 20% False；severe/warning/info 各比例 5:10:5
    """
    logger.info("[seed] 开始插入 compliance_checks 种子数据...")

    element_fields = [  # 18 个要素字段，随机选取
        "ticket_number", "ticket_type", "issue_date", "due_date",
        "amount_numeric", "amount_text", "currency", "drawer",
        "drawer_account", "drawer_bank", "acceptor", "payee",
        "drawee_bank", "endorsers", "maturity_days", "trade_purpose",
        "acceptance_clause", "special_remarks",
    ]

    regulations = [  # 随机法规引用
        "票据法第22条", "票据法第26条", "票据法第44条",
        "中国人民银行令[2016]第3号第15条", "商业汇票承兑、贴现与再贴现管理暂行办法第8条",
        "上海票据交易所票据业务规则第3.2条", "电子商业汇票系统管理办法第18条",
    ]

    used_tasks = (task_ids * 3)[:100]  # 复用任务 ID（允许一个任务有多条合规检查）

    for i in range(100):
        is_compliant = random.random() < 0.80   # 80% 概率合规
        if is_compliant:
            violation_level = ViolationLevel.INFO  # 合规时只有提示级别
        else:
            violation_level = random.choice([     # 违规时按 5:10:5 比例分配严重程度
                ViolationLevel.SEVERE,
                ViolationLevel.WARNING, ViolationLevel.WARNING,
                ViolationLevel.INFO,
            ])

        check = ComplianceCheck(
            id=gen_id(),
            audit_task_id=used_tasks[i],
            element_field=random.choice(element_fields),
            element_value=str(random.choice(["2024-01-15", "壹佰万元整", "银行承兑汇票", rand_company()])),
            regulation_ref=random.choice(regulations),
            rag_query=f"票据{random.choice(['出票日期', '金额', '背书', '承兑'])}格式要求是什么？",
            rag_answer=f"根据{random.choice(regulations)}，{random.choice(['要求格式正确', '金额需与大写一致', '背书须连续'])}",
            rag_score=round(random.uniform(0.6, 0.99), 3),
            is_compliant=is_compliant,
            violation_level=violation_level,
            violation_desc=(None if is_compliant else f"字段值不符合{random.choice(regulations)}要求"),
            suggestion=(None if is_compliant else "请修改后重新提交"),
        )
        db.add(check)

    await db.flush()
    logger.info("[seed] compliance_checks 插入完成，共 100 条")


async def seed_endorsement_chains(db: AsyncSession, task_ids: List[str]) -> None:
    """
    插入 100 条背书链路分析结果种子数据
    分布：40 连续有效 / 30 断链 / 20 重复被背书 / 10 空白背书
    """
    logger.info("[seed] 开始插入 endorsement_chains 种子数据...")

    used_tasks = random.sample(task_ids, min(100, len(task_ids)))

    distribution = [
        ("continuous",    40, True,  False, 0),   # (类型, 数量, is_continuous, has_cycle, blank_count)
        ("broken",        30, False, False, 0),
        ("duplicate",     20, True,  True,  0),   # 重复被背书 = 有闭环
        ("blank",         10, True,  False, 2),   # 空白背书
    ]

    idx = 0  # 任务 ID 索引
    for chain_type, count, is_continuous, has_cycle, blank_count in distribution:
        for _ in range(count):
            if idx >= len(used_tasks):
                break
            endorser_count = random.randint(1, 5)  # 随机背书层数
            violation_codes = []
            if not is_continuous:
                violation_codes.append("E001")  # 背书链断裂
            if has_cycle:
                violation_codes.append("E003")  # 背书闭环
            if blank_count > 0:
                violation_codes.append("E005")  # 空白背书

            # 生成背书有向图 JSON（模拟格式）
            nodes = [{"id": f"n{j}", "name": rand_company(), "type": "endorser"}
                     for j in range(endorser_count + 1)]
            edges = [{"from": f"n{j}", "to": f"n{j+1}", "date": rand_date(60).strftime("%Y-%m-%d")}
                     for j in range(endorser_count)]

            chain = EndorsementChain(
                id=gen_id(),
                audit_task_id=used_tasks[idx],
                chain_graph={"nodes": nodes, "edges": edges},
                endorser_count=endorser_count,
                is_continuous=is_continuous,
                violation_codes=violation_codes,
                violation_details=[{"code": c, "desc": f"违规说明-{c}"} for c in violation_codes],
                max_chain_depth=endorser_count,
                has_cycle=has_cycle,
                blank_endorsement_count=blank_count,
            )
            db.add(chain)
            idx += 1

    await db.flush()
    logger.info("[seed] endorsement_chains 插入完成，共 100 条")


async def seed_contract_reviews(db: AsyncSession, task_ids: List[str], document_ids: List[str]) -> None:
    """
    插入 100 条合同审核结果种子数据
    分布：50 全匹配 / 30 金额不匹配 / 20 交易方不匹配
    """
    logger.info("[seed] 开始插入 contract_reviews 种子数据...")

    used_tasks = random.sample(task_ids, min(100, len(task_ids)))

    distribution = [
        ("all_match",      50, True,  True),    # (类型, 数量, amount_match, party_match)
        ("amount_mismatch", 30, False, True),
        ("party_mismatch",  20, True,  False),
    ]

    idx = 0
    for review_type, count, amount_match, party_match in distribution:
        for _ in range(count):
            if idx >= len(used_tasks):
                break

            # 匹配度：全匹配时高分，有不匹配时低分
            match_score = (
                random.uniform(80, 100) if amount_match and party_match
                else random.uniform(40, 65) if not amount_match
                else random.uniform(50, 70)
            )

            mismatch_details = []
            if not amount_match:
                mismatch_details.append({
                    "field": "amount", "severity": "warning",
                    "bill_value": rand_amount(), "contract_value": rand_amount(),
                })
            if not party_match:
                mismatch_details.append({
                    "field": "party", "severity": "severe",
                    "bill_value": rand_company(), "contract_value": rand_company(),
                })

            review = ContractReview(
                id=gen_id(),
                audit_task_id=used_tasks[idx],
                contract_document_id=random.choice(document_ids) if random.random() > 0.1 else None,
                match_score=round(match_score, 2),
                trade_background_score=round(random.uniform(60, 95), 2),
                amount_match=amount_match,
                party_match=party_match,
                date_match=(random.random() > 0.05),   # 95% 日期匹配
                purpose_match=(random.random() > 0.1), # 90% 用途匹配
                mismatch_details=mismatch_details,
                review_notes=random.choice([
                    "票据要素与合同条款一致", "金额差异在允许范围内",
                    "需核查贸易背景真实性", "建议补充合同附件",
                ]),
            )
            db.add(review)
            idx += 1

    await db.flush()
    logger.info("[seed] contract_reviews 插入完成，共 100 条")


async def seed_risk_assessments(db: AsyncSession, task_ids: List[str]) -> None:
    """
    插入 100 条风险评估结果种子数据
    分布：5 级风险各 20 条；composite_score 均匀分布 0~100
    """
    logger.info("[seed] 开始插入 risk_assessments 种子数据...")

    # 各风险等级对应的综合评分范围
    risk_distribution = [
        (RiskLevel.LOW,         85.0, 100.0, 20),  # (等级, 分数下限, 分数上限, 数量)
        (RiskLevel.MEDIUM_LOW,  70.0,  84.9, 20),
        (RiskLevel.MEDIUM_HIGH, 50.0,  69.9, 20),
        (RiskLevel.HIGH,        30.0,  49.9, 20),
        (RiskLevel.CRITICAL,     0.0,  29.9, 20),
    ]

    used_tasks = random.sample(task_ids, min(100, len(task_ids)))
    idx = 0

    for risk_level, score_min, score_max, count in risk_distribution:
        for _ in range(count):
            if idx >= len(used_tasks):
                break

            composite = round(random.uniform(score_min, score_max), 2)

            # 四维评分（各维度在综合分附近随机波动）
            compliance_score  = round(min(100, max(0, composite + random.uniform(-15, 15))), 2)
            endorsement_score = round(min(100, max(0, composite + random.uniform(-10, 10))), 2)
            contract_score    = round(min(100, max(0, composite + random.uniform(-20, 20))), 2)
            fraud_score       = round(min(100, max(0, composite + random.uniform(-5, 5))), 2)

            assessment = RiskAssessment(
                id=gen_id(),
                audit_task_id=used_tasks[idx],
                compliance_score=compliance_score,
                endorsement_score=endorsement_score,
                contract_score=contract_score,
                fraud_score=fraud_score,
                composite_score=composite,
                risk_level=risk_level,
                dimension_details={
                    "compliance":  {"score": compliance_score,  "weight": 0.30},
                    "endorsement": {"score": endorsement_score, "weight": 0.30},
                    "contract":    {"score": contract_score,    "weight": 0.20},
                    "fraud":       {"score": fraud_score,       "weight": 0.20},
                },
                missing_dimensions=(["contract"] if random.random() < 0.1 else []),  # 10% 缺合同维度
                assessor_notes=random.choice([
                    "综合评分良好，建议直接通过", "存在背书链断裂，建议人工核查",
                    "金额较大需重点关注贸易背景", "检测到可疑印章，建议拒绝",
                    "历史无异常，快速通道审批", "多项指标异常，建议退件",
                ]),
            )
            db.add(assessment)
            idx += 1

    await db.flush()
    logger.info("[seed] risk_assessments 插入完成，共 100 条")


async def seed_audit_reports(db: AsyncSession, task_ids: List[str]) -> None:
    """
    插入 100 条审核报告种子数据
    分布：70 条含 PDF / 30 条仅 JSON；is_final 各半
    """
    logger.info("[seed] 开始插入 audit_reports 种子数据...")

    used_tasks = random.sample(task_ids, min(100, len(task_ids)))

    for i, task_id in enumerate(used_tasks):
        has_pdf = i < 70  # 前 70 条有 PDF
        is_final = (i % 2 == 0)  # 交替 is_final

        report_json = {  # 9 节结构的报告 JSON（简化版，供前端展示）
            "summary": {"task_id": task_id, "status": "completed"},
            "elements": {"ticket_number": rand_ticket_number()},
            "compliance": {"overall": True, "violations": []},
            "endorsement": {"is_continuous": True, "depth": random.randint(1, 5)},
            "contract": {"match_score": round(random.uniform(60, 99), 2)},
            "risk": {"composite_score": round(random.uniform(50, 99), 2), "level": "MEDIUM_LOW"},
            "fraud": {"overall_fraud_score": round(random.random() * 0.3, 3)},
            "flow": {"completed_steps": 7, "total_steps": 7},
            "conclusion": {"recommendation": random.choice(["通过", "退件", "人工复核"])},
        }

        report = AuditReport(
            id=gen_id(),
            audit_task_id=task_id,
            report_json=report_json,
            pdf_path=(f"./data/processed/report_{task_id[:8]}.pdf" if has_pdf else None),
            pdf_size_bytes=(random.randint(50000, 500000) if has_pdf else None),
            is_final=is_final,
            training_mode=(random.random() < 0.1),  # 10% 为培训模式报告
            version=random.randint(1, 3),
            generated_by="agent",
        )
        db.add(report)

    await db.flush()
    logger.info("[seed] audit_reports 插入完成，共 100 条")


async def seed_flow_tracking(db: AsyncSession, task_ids: List[str]) -> List[str]:
    """
    插入 100 条流转追踪主表种子数据 + 对应的 flow_messages 记录
    分布：7 步各有不同 completed_steps 的进度；95% 正常节点 / 5% 异常节点
    Returns:
        flow_task_ids: 供 flow_messages 作外键引用
    """
    logger.info("[seed] 开始插入 flow_tracking_tasks 种子数据...")

    used_tasks = random.sample(task_ids, min(100, len(task_ids)))
    flow_task_ids = []

    business_types = [
        AuditTaskType.ACCEPTANCE_PROMPT, AuditTaskType.DISCOUNT_APPLY,
        AuditTaskType.ENDORSEMENT, AuditTaskType.PAYMENT_PROMPT,
    ]

    node_names = [  # 七步流转的节点名称列表
        "发起行-发送", "前置机-接收", "票交所-处理", "对手行-接收",
        "对手行-应答", "票交所-转发", "发起行-接收",
    ]

    for i, task_id in enumerate(used_tasks):
        completed_steps = (i % 8)  # 0~7 步，均匀覆盖各阶段进度
        is_completed = (completed_steps == 7)

        flow_task_id = gen_id()
        flow_task_ids.append(flow_task_id)

        flow_task = FlowTrackingTask(
            id=flow_task_id,
            audit_task_id=task_id,
            business_type=random.choice(business_types),
            ticket_number=rand_ticket_number(),
            total_steps=7,
            completed_steps=completed_steps,
            current_node=(node_names[min(completed_steps, 6)] if completed_steps < 7 else "已完成"),
            status=(FlowTrackingStatus.COMPLETED if is_completed
                    else FlowTrackingStatus.INITIATED if completed_steps == 0
                    else FlowTrackingStatus.IN_FLIGHT),
            timeout_level=random.choice(["NORMAL", "NORMAL", "NORMAL", "WATCH", "WARNING"]),  # 大多数正常
            summary_text=f"当前票据已完成 {completed_steps}/7 步流转",
            initiated_at=rand_date(30),
            completed_at=(rand_date(30) if is_completed else None),
        )
        db.add(flow_task)

        # 为已完成的步骤生成 flow_messages 记录
        for step in range(1, completed_steps + 1):
            is_anomaly = (random.random() < 0.05)  # 5% 概率为异常节点
            msg = FlowMessage(
                id=gen_id(),
                flow_task_id=flow_task_id,
                step_index=step,
                node_name=node_names[step - 1],
                msg_type=random.choice(["send", "ack"]),
                msg_content={"step": step, "timestamp": datetime.now().isoformat()},
                status=(FlowMessageStatus.ERROR if is_anomaly else FlowMessageStatus.RESPONDED),
                arrived_at=rand_date(30),
                processing_ms=random.randint(50, 3000),
                is_anomaly=is_anomaly,
                anomaly_reason=("处理超时" if is_anomaly else None),
            )
            db.add(msg)

    await db.flush()
    logger.info(f"[seed] flow_tracking_tasks + flow_messages 插入完成，共 {len(flow_task_ids)} 个追踪任务")
    return flow_task_ids


async def seed_fraud_detections(db: AsyncSession, task_ids: List[str]) -> None:
    """
    插入 100 条欺诈检测结果种子数据
    分布：70 条无嫌疑 / 20 条印章异常 / 10 条重复票据命中
    """
    logger.info("[seed] 开始插入 fraud_detections 种子数据...")

    used_tasks = random.sample(task_ids, min(100, len(task_ids)))

    distribution = [
        ("clean",     70, 0.0, 0.5,  0.0, 0.3,  0.0, 0.1,  0.0, 0.1,  0.0, 0.2),
        # (类型, 数量, seal_min, seal_max, dup_min, dup_max, tamper_min, tamper_max,
        #   network_min, network_max, blacklist_min, blacklist_max)
        ("seal_issue",    20, 0.6, 0.95, 0.0, 0.2,  0.0, 0.3,  0.0, 0.2,  0.0, 0.1),
        ("duplicate",     10, 0.0, 0.3,  1.0, 1.0,  0.0, 0.2,  0.3, 0.7,  0.0, 0.3),
    ]

    idx = 0
    for det_type, count, s_lo, s_hi, d_lo, d_hi, t_lo, t_hi, n_lo, n_hi, b_lo, b_hi in distribution:
        for _ in range(count):
            if idx >= len(used_tasks):
                break

            seal_score      = round(random.uniform(s_lo, s_hi), 3)
            duplicate_score = round(random.uniform(d_lo, d_hi), 3)
            tamper_score    = round(random.uniform(t_lo, t_hi), 3)
            network_score   = round(random.uniform(n_lo, n_hi), 3)
            blacklist_score = round(random.uniform(b_lo, b_hi), 3)

            # 综合欺诈评分：加权求和（权重来自需求文档）
            overall = round(
                seal_score      * 0.30 +
                duplicate_score * 0.25 +
                tamper_score    * 0.25 +
                network_score   * 0.15 +
                blacklist_score * 0.05,
                3
            )

            fraud_level = (
                "fraud"     if overall >= 0.8 else
                "high_risk" if overall >= 0.6 else
                "suspicious" if overall >= 0.3 else
                "clean"
            )

            detection = FraudDetection(
                id=gen_id(),
                audit_task_id=used_tasks[idx],
                seal_score=seal_score,
                seal_check_result={"similarity": 1 - seal_score, "blacklist_hit": seal_score > 0.5},
                duplicate_score=duplicate_score,
                duplicate_check_result={"is_duplicate": duplicate_score >= 1.0, "ticket_number_match": duplicate_score >= 1.0},
                tamper_score=tamper_score,
                tamper_check_result={"suspicious_regions": [], "method": "ocr_confidence_analysis"},
                network_score=network_score,
                network_check_result={"has_cycle": network_score > 0.5, "cycle_path": []},
                blacklist_score=blacklist_score,
                blacklist_check_result={"hit_entities": [], "match_confidence": blacklist_score},
                overall_fraud_score=overall,
                fraud_level=fraud_level,
            )
            db.add(detection)
            idx += 1

    await db.flush()
    logger.info("[seed] fraud_detections 插入完成，共 100 条")


async def seed_batch_tasks(db: AsyncSession, tenant_ids: List[str]) -> List[str]:
    """
    插入 10 个批量任务主表记录（不是 100 条，符合实际业务场景）
    配合 100 条子任务记录，每个批次 10 条
    Returns:
        batch_task_ids: 供 batch_task_items 作外键引用
    """
    logger.info("[seed] 开始插入 batch_tasks 种子数据...")

    batch_task_ids = []
    concurrency_options = [10, 20, 50]  # 不同并发配置

    for i in range(10):  # 10 个批次
        status = random.choice([
            BatchTaskStatus.PENDING, BatchTaskStatus.RUNNING,
            BatchTaskStatus.COMPLETED, BatchTaskStatus.COMPLETED,  # COMPLETED 比例更高
        ])
        batch_task_id = gen_id()
        batch_task_ids.append(batch_task_id)

        batch = BatchTask(
            id=batch_task_id,
            tenant_id=random.choice(tenant_ids),
            task_type=random.choice([AuditTaskType.FULL_AUDIT, AuditTaskType.DISCOUNT_APPLY]),
            total_count=10,                         # 每批 10 个子任务
            completed_count=(10 if status == BatchTaskStatus.COMPLETED else random.randint(0, 9)),
            failed_count=random.randint(0, 2),      # 失败率约 0~20%
            concurrency=random.choice(concurrency_options),
            status=status,
            result_summary=(
                {"total": 10, "completed": 10, "failed": random.randint(0, 2),
                 "risk_distribution": {"LOW": 4, "MEDIUM_LOW": 3, "MEDIUM_HIGH": 2, "HIGH": 1}}
                if status == BatchTaskStatus.COMPLETED else None
            ),
            submitted_by=gen_id(),   # 随机用户 ID（测试数据，不要求真实存在）
            started_at=(rand_date(30) if status != BatchTaskStatus.PENDING else None),
            completed_at=(rand_date(30) if status == BatchTaskStatus.COMPLETED else None),
        )
        db.add(batch)

    await db.flush()
    logger.info(f"[seed] batch_tasks 插入完成，共 {len(batch_task_ids)} 条")
    return batch_task_ids


async def seed_batch_task_items(
    db: AsyncSession,
    batch_task_ids: List[str],
    audit_task_ids: List[str],
    document_ids: List[str]
) -> None:
    """
    插入 100 条批量子任务种子数据（每个批量任务 10 条子任务）
    分布：失败率约 15%
    """
    logger.info("[seed] 开始插入 batch_task_items 种子数据...")

    item_idx = 0
    for batch_id in batch_task_ids:  # 每个批量任务 10 条子任务
        for i in range(10):
            is_failed = (random.random() < 0.15)   # 15% 失败率
            status = (AuditTaskStatus.FAILED if is_failed
                      else random.choice([AuditTaskStatus.PENDING, AuditTaskStatus.COMPLETED, AuditTaskStatus.COMPLETED]))

            item = BatchTaskItem(
                id=gen_id(),
                batch_task_id=batch_id,
                audit_task_id=(random.choice(audit_task_ids) if not is_failed else None),
                document_id=random.choice(document_ids),
                item_index=i,
                status=status,
                error_msg=("文件格式不支持" if is_failed else None),
            )
            db.add(item)
            item_idx += 1

    await db.flush()
    logger.info(f"[seed] batch_task_items 插入完成，共 {item_idx} 条")


async def seed_error_code_mappings(db: AsyncSession) -> None:
    """
    插入 100 条错误码字典种子数据
    覆盖 Agent 可能产生的所有错误码，分布在 5 个类别中
    """
    logger.info("[seed] 开始插入 error_code_mappings 种子数据...")

    # 预定义的完整错误码列表（100 条，覆盖各 Agent 的错误场景）
    error_codes = []

    # 要素类错误 E001~E025（25 条）
    element_errors = [
        ("E001", "票据号码格式错误", "票据号码不符合票交所规定的格式要求"),
        ("E002", "出票日期缺失", "票据上未印制出票日期"),
        ("E003", "到期日早于出票日", "到期日期在出票日期之前，属于无效票据"),
        ("E004", "金额大小写不一致", "票面金额数字与大写不匹配"),
        ("E005", "金额为零或负数", "票面金额必须为正整数"),
        ("E006", "出票人名称缺失", "票据上未填写出票人全称"),
        ("E007", "承兑人签章缺失", "银行承兑汇票缺少承兑行公章"),
        ("E008", "收款人信息错误", "收款人名称与账号不对应"),
        ("E009", "开户行信息不全", "出票人开户行名称不完整"),
        ("E010", "票据类型错误", "票据类型标注不符合分类规范"),
        ("E011", "账号格式错误", "银行账号位数不符合要求"),
        ("E012", "到期日超限", "票据期限超过最长期限（6个月）"),
        ("E013", "贸易背景说明缺失", "贴现申请时必须填写贸易背景"),
        ("E014", "承兑条款格式错误", "承兑条款格式不符合规范"),
        ("E015", "特殊记载无效", "特殊记载事项包含禁止条款"),
        ("E016", "OCR识别置信度低", "图像质量不佳，关键字段识别不确定"),
        ("E017", "币种不支持", "当前系统仅支持人民币票据"),
        ("E018", "出票人与承兑人相同", "同一主体不能既是出票人又是承兑人"),
        ("E019", "金额大写格式错误", "大写金额不符合财务规范格式"),
        ("E020", "到期日为非工作日", "到期日应顺延至下一工作日"),
        ("E021", "出票日期为未来日期", "出票日期不能晚于当前日期"),
        ("E022", "票据已过期", "到期日已过，票据失效"),
        ("E023", "承兑行不在白名单", "承兑行未在票交所注册或已暂停业务"),
        ("E024", "多份重复要素", "同一字段存在多处填写且内容不一致"),
        ("E025", "票据号码重复", "系统已存在相同号码的票据"),
    ]

    # 合规类错误 C001~C020（20 条）
    compliance_errors = [
        ("C001", "违反票据法第22条", "出票要素不完整，不符合票据法规定"),
        ("C002", "违反反洗钱规定", "单笔金额超过50万需额外申报"),
        ("C003", "违反外汇管理规定", "跨境票据须符合外汇管理相关规定"),
        ("C004", "违反行业准入规定", "特定行业的票据业务受限"),
        ("C005", "违反贴现利率规定", "贴现利率不得低于人民银行规定下限"),
        ("C006", "背书转让次数超限", "电子票据背书次数超过系统规定上限"),
        ("C007", "承兑行资质不足", "承兑行评级低于贴现银行的最低要求"),
        ("C008", "票据期限不合规", "本次票据期限超出申请业务允许的最大期限"),
        ("C009", "缺少背书连续证明", "存在空白背书，无法证明背书连续性"),
        ("C010", "贸易合同缺失", "申请贴现时必须提交贸易合同"),
        ("C011", "发票与票据不匹配", "增值税发票金额与票据金额差异超过5%"),
        ("C012", "质押品不足值", "票据质押时，质押品价值低于票面金额"),
        ("C013", "出票人信用不足", "出票人在本行的信用评分低于准入标准"),
        ("C014", "超出业务额度", "申请业务超过租户当前可用额度"),
        ("C015", "合同有效期过期", "提供的贸易合同已超出有效期"),
        ("C016", "关联交易未披露", "存在疑似关联交易但未按规定披露"),
        ("C017", "抵押登记缺失", "质押背书业务需完成抵押登记"),
        ("C018", "税务异常", "出票人存在欠税或税务违规记录"),
        ("C019", "工商异常", "出票人存在工商吊销或注销风险"),
        ("C020", "评级下调", "承兑行评级在票据存续期间被下调"),
    ]

    # 背书类错误（15 条）
    endorsement_errors = [
        ("EN01", "背书链断裂", "背书人与前手收款人不一致，背书链不连续"),
        ("EN02", "重复背书", "同一主体在背书链中出现两次"),
        ("EN03", "背书闭环", "背书图中存在环路，涉嫌虚假背书"),
        ("EN04", "空白背书", "背书人签章但未填写被背书人"),
        ("EN05", "背书日期异常", "背书日期早于前手背书日期"),
        ("EN06", "背书人已注销", "背书人企业已被吊销营业执照"),
        ("EN07", "背书格式错误", "背书栏填写内容不符合规范"),
        ("EN08", "超出最大背书层数", "背书次数超过系统允许的最大层数"),
        ("EN09", "被背书人与背书人相同", "自我背书属于无效背书"),
        ("EN10", "背书撤销未记录", "存在已撤销的背书但未在系统更新"),
        ("EN11", "背书人在黑名单", "背书链中有主体列入监管黑名单"),
        ("EN12", "背书超期", "背书日期与票据到期日的间隔超过规定"),
        ("EN13", "背书印章不清晰", "背书印章图像质量低，无法核验真实性"),
        ("EN14", "委托收款背书缺失", "托收委托业务需要完整的委托收款背书"),
        ("EN15", "质押背书格式错误", "质押背书必须注明「质押」字样"),
    ]

    # 欺诈类错误（20 条）
    fraud_errors = [
        ("F001", "印章疑似伪造", "印章特征与真实样本相似度低于阈值"),
        ("F002", "图像疑似篡改", "票据图像存在局部修改痕迹"),
        ("F003", "重复票据", "系统中已存在相同号码和要素的票据"),
        ("F004", "关联网络异常", "背书链呈现典型的虚假贸易网络特征"),
        ("F005", "主体在黑名单", "出票人/背书人/承兑人命中监管黑名单"),
        ("F006", "印章边界异常", "印章边界不规则，疑似数字合成"),
        ("F007", "字迹笔压异常", "手写字迹的笔压分布不符合自然书写规律"),
        ("F008", "重复使用号码", "票据号码格式与已知伪票号码格式匹配"),
        ("F009", "快速流转异常", "票据在极短时间内连续背书多次"),
        ("F010", "金额修改痕迹", "金额字段存在覆盖修改的像素痕迹"),
        ("F011", "日期修改痕迹", "日期字段存在覆盖修改的像素痕迹"),
        ("F012", "承兑行印章与官方不符", "承兑行印章与官方备案印章存在差异"),
        ("F013", "出票人疑似空壳公司", "出票人注册资本极低且无实际业务证明"),
        ("F014", "历史欺诈关联", "出票人/背书人与历史欺诈案件存在关联"),
        ("F015", "跨区域异常", "票据签发地与出票人注册地不一致且无合理说明"),
        ("F016", "号码序列异常", "批量伪造票据通常具有连续号码特征"),
        ("F017", "承兑金额异常", "承兑金额远超承兑人历史业务规模"),
        ("F018", "多次提示拒付", "历史上该出票人票据有被拒付记录"),
        ("F019", "账户冻结", "出票人或背书人账户已被司法冻结"),
        ("F020", "企业异常经营", "出票人企业存在严重的经营异常记录"),
    ]

    # 系统类错误（20 条）
    system_errors = [
        ("SYS001", "Agent执行超时", "Agent 在规定时间内未完成执行"),
        ("SYS002", "Agent未注册", "请求的 Agent 名称未在注册表中找到"),
        ("SYS003", "Agent执行异常", "Agent 内部发生未预期的系统异常"),
        ("SYS004", "数据库连接失败", "无法连接到 PostgreSQL 数据库"),
        ("SYS005", "向量库连接失败", "无法连接到 Milvus 向量数据库"),
        ("SYS006", "LLM服务不可用", "大语言模型 API 请求失败"),
        ("SYS007", "文件解析失败", "PDF/图片解析过程发生错误"),
        ("SYS008", "OCR服务异常", "PaddleOCR 识别服务异常"),
        ("SYS009", "RAG检索失败", "向量检索或精排过程发生错误"),
        ("SYS010", "并发超限", "当前并发任务数已达到系统上限"),
        ("SYS011", "存储空间不足", "服务器磁盘空间不足，无法保存文件"),
        ("SYS012", "内存溢出", "处理大文件时内存使用超限"),
        ("SYS013", "网络超时", "外部服务请求网络超时"),
        ("SYS014", "PDF生成失败", "reportlab 生成 PDF 时发生错误"),
        ("SYS015", "配置项缺失", "必要的配置项未设置（如 API Key）"),
        ("SYS016", "租户配额耗尽", "当前租户的文档配额已用尽"),
        ("SYS017", "跳过节点", "前置关键节点失败，当前节点已跳过"),
        ("SYS018", "DAG执行失败", "任务编排图执行过程发生错误"),
        ("SYS019", "版本不兼容", "数据格式与当前 Agent 版本不兼容"),
        ("SYS020", "外部接口异常", "票交所/银行系统接口返回错误"),
    ]

    # 合并所有错误码（共 100 条）
    all_errors = (
        [(code, title, desc, "element",     ViolationLevel.WARNING) for code, title, desc in element_errors] +
        [(code, title, desc, "compliance",  ViolationLevel.SEVERE)  for code, title, desc in compliance_errors] +
        [(code, title, desc, "endorsement", ViolationLevel.WARNING) for code, title, desc in endorsement_errors] +
        [(code, title, desc, "fraud",       ViolationLevel.SEVERE)  for code, title, desc in fraud_errors] +
        [(code, title, desc, "system",      ViolationLevel.INFO)    for code, title, desc in system_errors]
    )

    for code, title, desc, category, severity in all_errors:
        mapping = ErrorCodeMapping(
            id=gen_id(),
            error_code=code,
            category=category,
            severity=severity,
            zh_title=title,
            zh_desc=desc,
            remediation=(f"请修正{title}相关内容后重新提交" if category != "system" else None),
            is_active=True,
        )
        db.add(mapping)

    await db.flush()
    logger.info(f"[seed] error_code_mappings 插入完成，共 {len(all_errors)} 条")


async def seed_blacklist_entities(db: AsyncSession, tenant_ids: List[str]) -> None:
    """
    插入 100 条黑名单实体种子数据
    分布：50 company / 30 person / 20 account；20 条全局黑名单
    """
    logger.info("[seed] 开始插入 blacklist_entities 种子数据...")

    sources = [
        "中国人民银行失信名单", "司法冻结账户", "银行内部风控",
        "最高法失信被执行人名单", "金融黑名单共享平台", "外汇局违规名单",
    ]

    distributions = [
        (EntityType.COMPANY, 50),  # 50 家企业
        (EntityType.PERSON,  30),  # 30 个自然人
        (EntityType.ACCOUNT, 20),  # 20 个账户
    ]

    idx = 0
    for entity_type, count in distributions:
        for i in range(count):
            is_global = (idx < 20)  # 前 20 条为全局黑名单

            if entity_type == EntityType.COMPANY:
                name = f"违规企业{i+1}号{rand_company()[:4]}"  # 截短公司名加编号区分
                entity_id = f"{random.randint(10**17, 10**18-1)}"  # 统一社会信用代码
            elif entity_type == EntityType.PERSON:
                names = ["张三", "李四", "王五", "赵六", "陈七", "刘八", "周九", "吴十"]
                name = f"违规{random.choice(names)}{random.randint(100, 999)}"
                entity_id = f"{random.randint(100000000000000000, 999999999999999999)}"  # 18位身份证
            else:
                name = f"违规账户{i+1}号"
                entity_id = f"62{random.randint(10**17, 10**18-1)}"  # 模拟银行账号

            entity = BlacklistEntity(
                id=gen_id(),
                entity_type=entity_type,
                entity_name=name,
                entity_id=entity_id,
                source=random.choice(sources),
                reason=random.choice([
                    "票据欺诈", "背书链伪造", "拒绝付款", "虚构贸易背景",
                    "洗钱嫌疑", "账户被冻结", "信用评级极差", "拒绝配合核查",
                ]),
                risk_score=(1.0 if is_global else round(random.uniform(0.5, 1.0), 2)),
                is_active=(random.random() > 0.05),  # 95% 有效
                is_global=is_global,
                tenant_id=(None if is_global else random.choice(tenant_ids)),
                expired_at=(None if is_global else
                           (datetime.now() + timedelta(days=random.randint(90, 365)) if random.random() > 0.5 else None)),
            )
            db.add(entity)
            idx += 1

    await db.flush()
    logger.info("[seed] blacklist_entities 插入完成，共 100 条")


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────

async def main():
    """
    种子数据主函数：按依赖顺序依次插入各表数据
    执行顺序：基础表 → audit_tasks → 各结果表 → 字典表 → 黑名单表
    """
    # 创建数据库引擎（不复用应用引擎，脚本独立运行）
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,           # 脚本模式不打印 SQL，减少控制台噪音
        pool_pre_ping=True,   # 自动检测连接健康状态
    )

    # 创建所有表（如已存在则跳过）
    logger.info("[seed] 开始初始化数据库表结构...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)  # 创建 db_models.py 中的基础表
        # 注意：agent_models 的表也通过同一个 Base 注册，会一并创建
    logger.info("[seed] 数据库表结构初始化完成")

    # 创建 Session 工厂
    AsyncSessionLocal = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )

    async with AsyncSessionLocal() as db:
        try:
            # 1. 准备基础数据（租户/文档/票据）
            tenant_ids, document_ids, bill_record_ids = await get_base_ids(db)

            # 2. 插入审核主任务（其他表的外键来源）
            task_ids = await seed_audit_tasks(db, tenant_ids, document_ids)

            # 3. 串行插入各结果表（AsyncSession 不允许同一会话并发写入）
            await seed_bill_elements(db, task_ids)
            await seed_compliance_checks(db, task_ids)
            await seed_endorsement_chains(db, task_ids)
            await seed_contract_reviews(db, task_ids, document_ids)
            await seed_risk_assessments(db, task_ids)
            await seed_audit_reports(db, task_ids)
            await seed_fraud_detections(db, task_ids)

            # 4. 流转追踪（包含子表 flow_messages，需要先创建主表）
            await seed_flow_tracking(db, task_ids)

            # 5. 批量任务（包含子表 batch_task_items）
            batch_task_ids = await seed_batch_tasks(db, tenant_ids)
            await seed_batch_task_items(db, batch_task_ids, task_ids, document_ids)

            # 6. 插入字典数据（独立，无外键依赖，串行执行）
            await seed_error_code_mappings(db)
            await seed_blacklist_entities(db, tenant_ids)

            # 7. 提交所有数据到数据库
            await db.commit()
            logger.info("=" * 60)
            logger.info("[seed] 所有种子数据插入成功！")
            logger.info("=" * 60)

        except Exception as e:
            await db.rollback()  # 出错时回滚所有未提交数据
            logger.error(f"[seed] 种子数据插入失败，已回滚: {e}", exc_info=True)
            raise

    await engine.dispose()  # 释放数据库连接池


if __name__ == "__main__":
    asyncio.run(main())  # 运行异步主函数
