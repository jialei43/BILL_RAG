#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/init_admin.py
# 初始化脚本：首次部署系统时运行，创建管理员账号和示例租户
# 使用方式：python scripts/init_admin.py
# 注意：只需要运行一次，重复运行会报错（机构代码 SYS_ADMIN 已存在）

import asyncio   # Python 异步运行时（运行异步函数需要）
import uuid      # 生成唯一 ID
import sys       # 系统模块（用于修改 Python 路径）
from pathlib import Path   # 路径处理

# 把项目根目录加入 Python 模块搜索路径
# 因为这个脚本在 scripts/ 子目录下，直接 import app.xxx 会找不到
# Path(__file__).parent.parent：scripts 的上一级 = 项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
# 异步 SQLAlchemy（让数据库操作不阻塞）
from app.models.db_models import Base, Tenant, User, TenantStatus, UserRole  # 数据库模型
from app.core.auth import get_password_hash                          # 密码哈希函数
from config.settings import settings                                 # 配置（数据库连接地址）


async def init_admin():
    """
    初始化管理员数据
    1. 创建所有数据库表
    2. 创建系统管理员租户和账号
    3. 创建演示用的示例租户和账号
    """
    # 创建数据库引擎（连接到 PostgreSQL）
    engine = create_async_engine(settings.DATABASE_URL)
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # 创建所有数据库表（如果已存在则跳过）
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # create_all：扫描所有继承自 Base 的模型类，自动建表

    async with SessionLocal() as session:   # 开启数据库会话
        # ── 创建系统管理员租户 ──────────────────────────────────────────────
        tenant_id = str(uuid.uuid4())   # 生成租户唯一 ID
        tenant = Tenant(
            id=tenant_id,
            name="系统管理机构",           # 机构名称
            code="SYS_ADMIN",             # 机构代码（唯一标识）
            license_no="SYS-000001",      # 牌照编号（管理系统专用）
            status=TenantStatus.ACTIVE,   # 状态：正常激活
            doc_quota=999999,             # 配额：近乎无限（管理员不受限）
            qps_limit=1000,              # QPS：1000/秒（管理员高配额）
            milvus_partition="tenant_sys_admin",  # Milvus 分区名
        )
        session.add(tenant)   # 加入数据库会话

        # ── 创建超级管理员用户 ────────────────────────────────────────────
        admin = User(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,           # 关联到系统管理机构
            username="admin",              # 登录用户名
            email="admin@billrag.com",     # 邮箱
            hashed_password=get_password_hash("Admin@123456"),
            # 密码 "Admin@123456" 经 bcrypt 哈希后存储（绝不存明文）
            role=UserRole.SUPER_ADMIN,     # 超级管理员，系统唯一
            is_active=True,               # 账号激活
        )
        session.add(admin)

        # ── 创建示例票据机构租户（演示用） ────────────────────────────────
        demo_tenant_id = str(uuid.uuid4())
        demo_tenant = Tenant(
            id=demo_tenant_id,
            name="示例票据经纪公司",       # 模拟一个真实的票据机构
            code="DEMO_BROKER",
            license_no="DEMO-000001",
            status=TenantStatus.TRIAL,    # 状态：试用期（有功能限制）
            doc_quota=1000,              # 文档配额：1000个（试用版）
            qps_limit=20,               # QPS：20/秒
            milvus_partition="tenant_demo_broker",
        )
        session.add(demo_tenant)

        # 创建示例机构的普通用户
        demo_user = User(
            id=str(uuid.uuid4()),
            tenant_id=demo_tenant_id,
            username="demo",
            email="demo@billrag.com",
            hashed_password=get_password_hash("Demo@123456"),
            role=UserRole.USER,  # 普通用户
            is_active=True,
        )
        session.add(demo_user)

        await session.commit()   # 提交所有更改到数据库

    # 打印初始化结果（友好提示用户如何使用）
    print("✅ 初始化完成！")
    print(f"\n管理员账号:")
    print(f"  用户名: admin")
    print(f"  密码:   Admin@123456")
    print(f"\n演示账号:")
    print(f"  用户名: demo")
    print(f"  密码:   Demo@123456")
    print(f"\nAPI 文档: http://localhost:8000/api/docs")

    await engine.dispose()   # 释放数据库连接池（脚本结束时清理资源）


# 程序入口：当直接运行此脚本时执行
if __name__ == "__main__":
    asyncio.run(init_admin())   # asyncio.run：启动事件循环并运行异步函数
