/**
 * auth.js — 认证模块
 * 负责 JWT 的存取、解码、过期检测、登录/登出，以及页面级路由守卫
 * 依赖：config.js（必须先加载）
 */

const Auth = (() => {
  // ── JWT 解码（仅解 payload，不验签；验签在后端完成）──────────────────────
  /**
   * 将 base64url 编码的字符串解码为对象
   * JWT 格式：header.payload.signature，我们只需 payload 部分
   * @param {string} token - JWT 字符串
   * @returns {object|null} - 解码后的 payload，失败返回 null
   */
  function _decodeToken(token) {
    try {
      const parts = token.split('.');            // 按 '.' 分割三段
      if (parts.length !== 3) return null;       // 不是合法 JWT
      const payload = parts[1];                  // 取第二段 payload
      // base64url → base64：替换 '-' 和 '_'，补齐 '=' 填充
      const base64 = payload.replace(/-/g, '+').replace(/_/g, '/')
        + '=='.slice(0, (4 - payload.length % 4) % 4);
      return JSON.parse(atob(base64));           // 解码并 JSON 反序列化
    } catch {
      return null;                               // 任何异常均视为无效 token
    }
  }

  // ── Token 持久化 ────────────────────────────────────────────────────────────
  /**
   * 存储 token 到 localStorage
   * @param {string} token - JWT 字符串
   */
  function saveToken(token) {
    localStorage.setItem(CONFIG.TOKEN_KEY, token);  // 写入 localStorage
  }

  /**
   * 读取 localStorage 中的 token
   * @returns {string|null}
   */
  function getToken() {
    return localStorage.getItem(CONFIG.TOKEN_KEY);  // 不存在时返回 null
  }

  /**
   * 清除 localStorage 中的所有认证数据
   */
  function clearAuth() {
    localStorage.removeItem(CONFIG.TOKEN_KEY);   // 删除 token
    localStorage.removeItem(CONFIG.USER_KEY);    // 删除用户信息缓存
  }

  // ── 用户信息 ────────────────────────────────────────────────────────────────
  /**
   * 持久化用户信息（登录成功后调用）
   * @param {object} user - { user_id, tenant_id, is_admin, username }
   */
  function saveUser(user) {
    localStorage.setItem(CONFIG.USER_KEY, JSON.stringify(user));
  }

  /**
   * 读取缓存的用户信息
   * @returns {object|null}
   */
  function getUser() {
    try {
      const raw = localStorage.getItem(CONFIG.USER_KEY);
      return raw ? JSON.parse(raw) : null;       // 解析失败返回 null
    } catch {
      return null;
    }
  }

  // ── Token 有效性 ────────────────────────────────────────────────────────────
  /**
   * 检查当前 token 是否存在且未过期
   * @returns {boolean}
   */
  function isAuthenticated() {
    const token = getToken();
    if (!token) return false;                    // 无 token
    const payload = _decodeToken(token);
    if (!payload || !payload.exp) return false;  // 无效 token 或无过期字段
    // exp 是 Unix 时间戳（秒），转为毫秒后与当前时间比较
    const expiresAt = payload.exp * 1000 - CONFIG.TOKEN_EXPIRE_BUFFER;
    return Date.now() < expiresAt;               // 未到过期时间视为有效
  }

  /**
   * 从 token 中提取角色
   * @returns {'admin'|'user'|null}
   */
  function getRole() {
    const user = getUser();
    if (!user) return null;
    // 兼容新旧格式：优先读 role 字段，降级读 is_admin
    if (user.role) return user.role;
    return user.is_admin ? CONFIG.ROLE.ADMIN : CONFIG.ROLE.USER;
  }

  /**
   * 判断当前用户是否为管理员
   * @returns {boolean}
   */
  function isAdmin() {
    return getRole() === CONFIG.ROLE.ADMIN;
  }

  // ── 登录 ───────────────────────────────────────────────────────────────────
  /**
   * 执行登录请求
   * @param {string} username
   * @param {string} password
   * @param {'user'|'admin'} expectedRole - 用户在登录页选择的角色类型
   * @returns {Promise<{success: boolean, error?: string}>}
   */
  async function login(username, password, expectedRole) {
    // 构造 x-www-form-urlencoded 格式（FastAPI OAuth2PasswordRequestForm 要求）
    const body = new URLSearchParams();
    body.append('username', username);           // 用户名字段
    body.append('password', password);           // 密码字段

    const resp = await fetch(`${CONFIG.API_BASE}${CONFIG.API_PREFIX}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: body.toString(),
    });

    if (!resp.ok) {
      // HTTP 4xx/5xx 错误，解析错误信息
      const err = await resp.json().catch(() => ({ detail: '登录失败，请重试' }));
      return { success: false, error: err.detail || '账号或密码错误' };
    }

    const data = await resp.json();              // { access_token, token_type, tenant_id, user_id }

    // 解码 token，获取 role 字段（super_admin / tenant_admin / user）
    const payload = _decodeToken(data.access_token);
    const backendRole = payload?.role ?? 'user';  // 后端 JWT 中的实际角色
    // super_admin 和 tenant_admin 均可进入管理后台
    const isAdminRole = backendRole === 'super_admin' || backendRole === 'tenant_admin';
    const actualRole = isAdminRole ? CONFIG.ROLE.ADMIN : CONFIG.ROLE.USER;

    // 角色不匹配：用户账号不能登录管理员入口，反之亦然
    if (expectedRole === CONFIG.ROLE.ADMIN && !isAdminRole) {
      return { success: false, error: '该账号不具有管理员权限，请使用用户账号登录' };
    }
    if (expectedRole === CONFIG.ROLE.USER && isAdminRole) {
      return { success: false, error: '请使用管理员入口登录管理员账号' };
    }

    // 存储认证信息
    saveToken(data.access_token);
    saveUser({
      user_id:    data.user_id,
      tenant_id:  data.tenant_id,
      role:       actualRole,
      backendRole: backendRole,  // 保留原始角色（super_admin/tenant_admin/user）供精细权限判断
      username:   username,
    });

    return { success: true, role: actualRole };  // 返回成功 + 角色
  }

  // ── 登出 ───────────────────────────────────────────────────────────────────
  /**
   * 登出：清除本地认证状态并跳转到登录页
   */
  function logout() {
    clearAuth();                                         // 清除 token 和用户信息
    window.location.href = CONFIG.ROUTES.LOGIN;          // 跳转到登录页
  }

  // ── 路由守卫 ────────────────────────────────────────────────────────────────
  /**
   * 页面级路由守卫：在每个受保护页面的顶部调用
   * @param {'user'|'admin'} requiredRole - 该页面要求的角色
   */
  function guardRoute(requiredRole) {
    if (!isAuthenticated()) {
      // 未登录，跳回登录页
      window.location.href = CONFIG.ROUTES.LOGIN;
      return false;
    }
    const role = getRole();
    if (requiredRole === CONFIG.ROLE.ADMIN && role !== CONFIG.ROLE.ADMIN) {
      // 普通用户试图访问管理页，重定向到聊天页
      window.location.href = CONFIG.ROUTES.CHAT;
      return false;
    }
    if (requiredRole === CONFIG.ROLE.USER && role !== CONFIG.ROLE.USER) {
      // 管理员试图访问用户聊天页，重定向到管理后台
      window.location.href = CONFIG.ROUTES.ADMIN_DASHBOARD;
      return false;
    }
    return true;  // 验证通过
  }

  // ── 公开 API ────────────────────────────────────────────────────────────────
  return {
    login,
    logout,
    getToken,
    getUser,
    isAuthenticated,
    isAdmin,
    getRole,
    guardRoute,
  };
})();  // IIFE：立即执行，避免污染全局命名空间
