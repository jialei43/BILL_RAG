# app/agents/utils/flow_channel_mock.py
# 流转追踪 Mock 适配层
# 职责：
#   - 模拟票据业务七步报文流转中的三个外部系统接口：
#       1. 发起行前置机（负责报文组装和发送）
#       2. 票交所（负责报文转发和合规审核）
#       3. 对手行（负责接收报文并回执）
#   - 生产环境通过配置 FLOW_TRACKING_MOCK=false 切换为真实接口适配器
#   - Mock 层按照真实协议格式返回数据，保证测试可重复运行
# 使用方式：
#   from app.agents.utils.flow_channel_mock import FlowChannelMock
#   channel = FlowChannelMock()
#   result = await channel.send_step(step_index=1, ticket_number="...", business_type="...")

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from typing import Dict, Optional

from loguru import logger


# 七步流转协议的节点名称（每步有发起节点和应答节点）
FLOW_STEP_NODES = {
    1: {"send": "发起行-报文组装",      "ack": "发起行-前置机发送"},
    2: {"send": "前置机-票交所发送",     "ack": "票交所-接收确认"},
    3: {"send": "票交所-报文处理",       "ack": "票交所-转发对手行"},
    4: {"send": "对手行-接收",          "ack": "对手行-处理中"},
    5: {"send": "对手行-应答生成",       "ack": "对手行-票交所应答"},
    6: {"send": "票交所-应答接收",       "ack": "票交所-发起行转发"},
    7: {"send": "发起行前置机-应答接收", "ack": "发起行-业务完成"},
}

# 各步骤的模拟延迟范围（毫秒）：模拟真实网络和系统处理时间
STEP_DELAY_RANGES_MS = {
    1: (50, 200),      # 本地前置机组装：快速
    2: (100, 500),     # 前置机→票交所网络传输
    3: (200, 800),     # 票交所处理和合规检查
    4: (100, 400),     # 票交所→对手行传输
    5: (500, 2000),    # 对手行处理：最耗时（人工或系统审核）
    6: (100, 400),     # 对手行→票交所应答传输
    7: (50, 200),      # 票交所→发起行应答传输
}

# 各步骤的模拟失败概率（0.0~1.0）：用于触发超时和异常场景测试
STEP_FAILURE_RATE = {
    1: 0.02,   # 发起行前置机：偶发故障
    2: 0.03,   # 前置机→票交所：网络偶发超时
    3: 0.05,   # 票交所处理：合规检查导致延时
    4: 0.03,   # 跨行传输：网络质量不稳定
    5: 0.08,   # 对手行处理：最高失败率（系统不稳定）
    6: 0.03,   # 对手行→票交所应答
    7: 0.02,   # 发起行应答接收
}


class FlowChannelMock:
    """
    流转报文通道 Mock 实现
    生产环境替换为实际的前置机/票交所/对手行 HTTP 客户端
    Mock 层的接口签名与生产接口完全一致，切换时无需修改 FlowTrackingAgent
    """

    def __init__(self, failure_rate_multiplier: float = 1.0):
        # failure_rate_multiplier：测试时可调高失败率，1.0=正常，0=全部成功
        self._failure_rate_multiplier = failure_rate_multiplier

    async def send_step(
        self,
        step_index: int,
        ticket_number: str,
        business_type: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        模拟发送第 step_index 步骤的报文，等待节点应答

        Args:
            step_index:     步骤序号（1~7）
            ticket_number:  票据号码（报文关键字段）
            business_type:  业务类型（决定报文格式）
            metadata:       附加元数据（如金额、期限等）

        Returns:
            dict: 包含以下字段的模拟应答
              - step_index: 步骤序号
              - send_node: 发送节点名称
              - ack_node: 应答节点名称
              - status: "success" / "timeout" / "error"
              - processing_ms: 本步骤处理耗时
              - arrived_at: 到达时间戳（ISO 格式）
              - msg_content: 模拟报文摘要
              - is_anomaly: 是否异常
              - anomaly_reason: 异常原因（is_anomaly=True 时有值）
        """
        if step_index not in FLOW_STEP_NODES:
            raise ValueError(f"无效步骤序号: {step_index}，有效范围 1~7")

        # 模拟网络传输延迟（异步等待，不阻塞）
        delay_min, delay_max = STEP_DELAY_RANGES_MS[step_index]
        delay_ms = random.randint(delay_min, delay_max)
        await asyncio.sleep(delay_ms / 1000.0)   # 转换为秒

        # 判断本步骤是否模拟失败
        failure_rate = STEP_FAILURE_RATE[step_index] * self._failure_rate_multiplier
        is_anomaly = random.random() < failure_rate

        step_nodes = FLOW_STEP_NODES[step_index]

        if is_anomaly:
            # 模拟超时或错误
            anomaly_types = ["timeout", "error"]
            anomaly_status = random.choice(anomaly_types)
            anomaly_reason = (
                f"Step {step_index} {anomaly_status}: "
                f"{'等待节点应答超时（300s）' if anomaly_status == 'timeout' else '节点处理异常，错误码 E9001'}"
            )
            logger.warning(f"[FlowChannelMock] Step {step_index} anomaly: {anomaly_reason}")
            return {
                "step_index":   step_index,
                "send_node":    step_nodes["send"],
                "ack_node":     step_nodes["ack"],
                "status":       anomaly_status,
                "processing_ms": delay_ms,
                "arrived_at":   datetime.now(timezone.utc).isoformat(),
                "msg_content": self._gen_msg_content(step_index, ticket_number, business_type, metadata),
                "is_anomaly":   True,
                "anomaly_reason": anomaly_reason,
            }
        else:
            # 模拟成功响应
            return {
                "step_index":   step_index,
                "send_node":    step_nodes["send"],
                "ack_node":     step_nodes["ack"],
                "status":       "success",
                "processing_ms": delay_ms,
                "arrived_at":   datetime.now(timezone.utc).isoformat(),
                "msg_content": self._gen_msg_content(step_index, ticket_number, business_type, metadata),
                "is_anomaly":   False,
                "anomaly_reason": None,
            }

    def _gen_msg_content(
        self, step_index: int, ticket_number: str, business_type: str, metadata: Optional[dict]
    ) -> dict:
        """
        生成脱敏的模拟报文摘要
        生产环境替换为真实报文协议解析结果
        """
        base = {
            "msg_id":        f"MSG{ticket_number[:8]}-S{step_index:02d}",   # 报文流水号
            "ticket_number": ticket_number[:4] + "****",   # 票据号码脱敏
            "business_type": business_type,
            "step":          step_index,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            # 只保留安全字段，敏感字段不记录
            safe_keys = ["amount_range", "currency", "maturity_days"]
            for k in safe_keys:
                if k in metadata:
                    base[k] = metadata[k]
        return base
