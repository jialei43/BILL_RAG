# app/core/auth.py
# JWT 认证与租户鉴权模块
# 解决"用户是谁？他有什么权限？"这两个核心安全问题
# JWT（JSON Web Token）：一种无状态的身份验证令牌，登录后颁发，每次请求携带即可证明身份

import uuid                            # 生成唯一 ID（UUID 格式，如：550e8400-e29b-41d4-a716-446655440000）
from datetime import datetime, timedelta  # 用于计算 Token 过期时间
from typing import Optional            # Optional 表示某个值可以为 None

from fastapi import Depends, HTTPException, status
# Depends：依赖注入，让 FastAPI 自动调用指定函数并传入结果
# HTTPException：HTTP 错误（如 401 未授权、403 禁止访问）
# status：HTTP 状态码常量（status.HTTP_401_UNAUTHORIZED = 401）

from fastapi.security import OAuth2PasswordBearer
# OAuth2PasswordBearer：从请求的 Authorization 头提取 Bearer Token
# 告诉 FastAPI：从 "Authorization: Bearer <token>" 头中提取 token

from jose import JWTError, jwt         # jose 库：JWT 的编码、解码、验证
from passlib.context import CryptContext  # passlib：密码哈希库，安全地存储密码
from pydantic import BaseModel         # BaseModel：数据模型基类，自动验证数据格式

from config.settings import settings   # 导入配置（SECRET_KEY、ALGORITHM 等）
from loguru import logger              # 日志


# 创建密码哈希上下文：使用 bcrypt 算法对密码进行哈希（不可逆加密）
# bcrypt 特点：即使数据库被盗，黑客也无法从哈希值反推出原始密码
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 定义从哪里提取 Token：从 /api/v1/auth/login 接口获取 Token
# 这会让 API 文档自动显示"需要登录"的锁图标
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


class TokenData(BaseModel):
    """JWT Token 解码后的数据结构（Token 携带的用户信息）"""
    user_id: str           # 用户唯一 ID
    tenant_id: str         # 用户所属租户 ID（用于数据隔离，防止 A 公司看到 B 公司的数据）
    is_admin: bool = False # 是否是管理员（默认不是）


class Token(BaseModel):
    """登录成功后返回给客户端的 Token 数据结构"""
    access_token: str      # JWT Token 字符串（客户端存储，每次请求都要带上）
    token_type: str = "bearer"  # Token 类型（固定为 "bearer"）
    tenant_id: str         # 用户所属租户 ID
    user_id: str           # 用户 ID


def verify_password(plain: str, hashed: str) -> bool:
    """
    验证密码是否正确
    plain：用户登录时输入的明文密码
    hashed：数据库中存储的哈希密码
    返回 True 表示密码正确，False 表示错误
    """
    return pwd_context.verify(plain, hashed)  # bcrypt 内部会提取盐值并重新哈希后比较


def get_password_hash(password: str) -> str:
    """
    对明文密码进行哈希处理，用于注册时存储密码
    每次调用结果都不同（因为 bcrypt 会自动生成随机盐值）
    """
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """
    生成 JWT Token
    data：要编码到 Token 中的数据（如 user_id、tenant_id、is_admin）
    expires_delta：Token 有效期（默认从配置读取，一般是 24 小时）
    返回编码后的 JWT 字符串
    """
    to_encode = data.copy()                    # 复制数据（避免修改原始 data）
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        # 计算过期时间：当前 UTC 时间 + 有效期
    )
    to_encode["exp"] = expire                  # 把过期时间写入 Token 载荷
    token = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    logger.info(  # Token 颁发日志：记录 user_id/tenant_id，方便追查"谁在什么时候登录"
        f"[auth] Token 已颁发 user_id={data.get('user_id')} "
        f"tenant_id={data.get('tenant_id')} expires={expire.strftime('%Y-%m-%d %H:%M:%S')}UTC"
    )
    return token  # 用 SECRET_KEY 签名，生成 JWT Token 字符串


def decode_token(token: str) -> TokenData:
    """
    解码并验证 JWT Token
    token：客户端携带的 Token 字符串
    返回 TokenData 对象（包含用户信息）
    如果 Token 无效或过期，抛出 401 错误
    """
    # 预先定义"认证失败"时要抛出的错误
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,     # HTTP 401 = 未授权
        detail="无效的认证凭证",                       # 错误信息
        headers={"WWW-Authenticate": "Bearer"},       # 告诉客户端应该用 Bearer Token 认证
    )
    try:
        # 用 SECRET_KEY 解码 Token，同时验证签名和过期时间
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])

        user_id: str = payload.get("user_id")         # 从 Token 载荷中取出用户 ID
        tenant_id: str = payload.get("tenant_id")     # 取出租户 ID

        if user_id is None or tenant_id is None:      # 必填字段缺失，Token 格式不对
            raise credentials_exception

        return TokenData(
            user_id=user_id,
            tenant_id=tenant_id,
            is_admin=payload.get("is_admin", False),  # 取出管理员标志，默认 False
        )
    except JWTError as e:                              # Token 签名错误或已过期
        logger.warning(f"[auth] Token 解码失败: {e}")  # 记录具体原因（过期/签名错/格式错）
        raise credentials_exception                   # 统一返回"认证失败"错误


async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    """
    FastAPI 依赖函数：从请求头提取并验证 Token，返回当前登录用户信息
    在路由函数参数中写 current_user: TokenData = Depends(get_current_user) 即可使用
    FastAPI 会自动：
      1. 从 Authorization 头提取 Token
      2. 调用本函数验证
      3. 把结果注入到路由函数的 current_user 参数中
    """
    return decode_token(token)   # 直接解码并验证 Token


async def get_admin_user(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    """
    FastAPI 依赖函数：验证当前用户是管理员
    在需要管理员权限的路由上使用 Depends(get_admin_user)
    如果不是管理员，返回 403 禁止访问
    """
    if not current_user.is_admin:                     # 检查管理员标志
        logger.warning(  # 非管理员访问管理接口，记录 user_id 方便安全审计
            f"[auth] 非管理员尝试访问管理接口 user_id={current_user.user_id} "
            f"tenant_id={current_user.tenant_id}"
        )
        raise HTTPException(status_code=403, detail="需要管理员权限")  # 403 = 禁止访问
    return current_user                               # 是管理员，放行并返回用户信息
