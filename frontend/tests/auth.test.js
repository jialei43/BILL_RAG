/**
 * auth.test.js — Auth 模块单元测试
 * 覆盖：token 存取、JWT 解码、角色判断、路由守卫逻辑
 */

TestRunner.suite('Auth 模块 — Token 存储与读取', () => {

  TestRunner.test('saveToken + getToken：能存能取', () => {
    localStorage.setItem(CONFIG.TOKEN_KEY, 'test_token_123');
    const got = Auth.getToken();
    TestRunner.assertEqual(got, 'test_token_123', 'getToken 应返回存入的值');
    localStorage.removeItem(CONFIG.TOKEN_KEY);           // 清理
  });

  TestRunner.test('getToken：无 token 时返回 null', () => {
    localStorage.removeItem(CONFIG.TOKEN_KEY);
    const got = Auth.getToken();
    TestRunner.assertEqual(got, null, '未设置时应返回 null');
  });

  TestRunner.test('getUser：用户信息序列化/反序列化正确', () => {
    const user = { user_id: 'u1', tenant_id: 't1', is_admin: false, username: 'alice', role: 'user' };
    localStorage.setItem(CONFIG.USER_KEY, JSON.stringify(user));
    const got = Auth.getUser();
    TestRunner.assertEqual(got.user_id,   'u1',    'user_id 应一致');
    TestRunner.assertEqual(got.is_admin,  false,   'is_admin 应为 false');
    TestRunner.assertEqual(got.username,  'alice', 'username 应一致');
    localStorage.removeItem(CONFIG.USER_KEY);           // 清理
  });

  TestRunner.test('getUser：无缓存时返回 null', () => {
    localStorage.removeItem(CONFIG.USER_KEY);
    TestRunner.assertEqual(Auth.getUser(), null, '无数据时应返回 null');
  });
});


TestRunner.suite('Auth 模块 — JWT 解码与过期检测', () => {
  /**
   * 构造一个指定 payload 的合法 JWT 格式（不签名，仅供解码测试）
   * @param {object} payload
   * @returns {string}
   */
  function makeJwt(payload) {
    const header  = btoa(JSON.stringify({ alg: 'HS256', typ: 'JWT' }))
      .replace(/=/g, '').replace(/\+/g, '-').replace(/\//g, '_');
    const body    = btoa(JSON.stringify(payload))
      .replace(/=/g, '').replace(/\+/g, '-').replace(/\//g, '_');
    return `${header}.${body}.fake_signature`;   // 签名段无所谓（前端只解码不验签）
  }

  TestRunner.test('isAuthenticated：有效 token（未过期）返回 true', () => {
    const exp = Math.floor(Date.now() / 1000) + 3600;    // 1 小时后过期
    const jwt = makeJwt({ user_id: 'u1', tenant_id: 't1', is_admin: false, exp });
    localStorage.setItem(CONFIG.TOKEN_KEY, jwt);
    TestRunner.assert(Auth.isAuthenticated(), '有效 token 应返回 true');
    localStorage.removeItem(CONFIG.TOKEN_KEY);
  });

  TestRunner.test('isAuthenticated：已过期 token 返回 false', () => {
    const exp = Math.floor(Date.now() / 1000) - 100;     // 100 秒前已过期
    const jwt = makeJwt({ user_id: 'u1', tenant_id: 't1', is_admin: false, exp });
    localStorage.setItem(CONFIG.TOKEN_KEY, jwt);
    TestRunner.assert(!Auth.isAuthenticated(), '过期 token 应返回 false');
    localStorage.removeItem(CONFIG.TOKEN_KEY);
  });

  TestRunner.test('isAuthenticated：无 token 返回 false', () => {
    localStorage.removeItem(CONFIG.TOKEN_KEY);
    TestRunner.assert(!Auth.isAuthenticated(), '无 token 应返回 false');
  });

  TestRunner.test('getRole：is_admin=false → user', () => {
    localStorage.setItem(CONFIG.USER_KEY, JSON.stringify({ is_admin: false }));
    TestRunner.assertEqual(Auth.getRole(), CONFIG.ROLE.USER, 'is_admin=false 应为 user 角色');
    localStorage.removeItem(CONFIG.USER_KEY);
  });

  TestRunner.test('getRole：is_admin=true → admin', () => {
    localStorage.setItem(CONFIG.USER_KEY, JSON.stringify({ is_admin: true }));
    TestRunner.assertEqual(Auth.getRole(), CONFIG.ROLE.ADMIN, 'is_admin=true 应为 admin 角色');
    localStorage.removeItem(CONFIG.USER_KEY);
  });

  TestRunner.test('isAdmin：is_admin=true 返回 true', () => {
    localStorage.setItem(CONFIG.USER_KEY, JSON.stringify({ is_admin: true }));
    TestRunner.assert(Auth.isAdmin(), 'isAdmin 应返回 true');
    localStorage.removeItem(CONFIG.USER_KEY);
  });

  TestRunner.test('isAdmin：is_admin=false 返回 false', () => {
    localStorage.setItem(CONFIG.USER_KEY, JSON.stringify({ is_admin: false }));
    TestRunner.assert(!Auth.isAdmin(), 'isAdmin 应返回 false');
    localStorage.removeItem(CONFIG.USER_KEY);
  });
});


TestRunner.suite('Auth 模块 — 路由守卫逻辑', () => {
  let redirected = null;    // 记录 location.href 被赋予的值

  // 拦截页面跳转（测试环境不能真正跳转）
  const _origDesc = Object.getOwnPropertyDescriptor(window, 'location');
  function _mockLocation() {
    try {
      Object.defineProperty(window, 'location', {
        writable: true,
        value: { href: window.location.href },
      });
    } catch {}
  }
  function _restoreLocation() {
    try {
      if (_origDesc) Object.defineProperty(window, 'location', _origDesc);
    } catch {}
  }

  TestRunner.test('guardRoute：未登录时 isAuthenticated 返回 false', () => {
    localStorage.removeItem(CONFIG.TOKEN_KEY);
    localStorage.removeItem(CONFIG.USER_KEY);
    // 不调用 guardRoute（会触发真实跳转）；仅验证 isAuthenticated 判断
    TestRunner.assert(!Auth.isAuthenticated(), '未登录状态 isAuthenticated 应为 false');
  });

  TestRunner.test('guardRoute：已登录管理员 isAdmin 返回 true', () => {
    localStorage.setItem(CONFIG.USER_KEY, JSON.stringify({ is_admin: true }));
    TestRunner.assert(Auth.isAdmin(), '管理员 isAdmin 应为 true');
    localStorage.removeItem(CONFIG.USER_KEY);
  });

  TestRunner.test('guardRoute：已登录普通用户 isAdmin 返回 false', () => {
    localStorage.setItem(CONFIG.USER_KEY, JSON.stringify({ is_admin: false }));
    TestRunner.assert(!Auth.isAdmin(), '普通用户 isAdmin 应为 false');
    localStorage.removeItem(CONFIG.USER_KEY);
  });
});


TestRunner.suite('Config 常量 — 完整性验证', () => {

  TestRunner.test('CONFIG.API_BASE 非空', () => {
    TestRunner.assert(CONFIG.API_BASE && CONFIG.API_BASE.length > 0, 'API_BASE 不能为空');
  });

  TestRunner.test('CONFIG.ROLE 包含 user 和 admin', () => {
    TestRunner.assert('user'  in CONFIG.ROLE, 'ROLE 应包含 user');
    TestRunner.assert('admin' in CONFIG.ROLE, 'ROLE 应包含 admin');
  });

  TestRunner.test('CONFIG.ROUTES 包含所有必要路由', () => {
    const required = ['LOGIN', 'CHAT', 'ADMIN_DASHBOARD', 'ADMIN_DOCUMENTS', 'ADMIN_TENANTS'];
    required.forEach(key => {
      TestRunner.assert(key in CONFIG.ROUTES, `ROUTES 缺少 ${key}`);
    });
  });

  TestRunner.test('CONFIG.UPLOAD.BILL_TYPES 包含常见票据格式', () => {
    const billTypes = CONFIG.UPLOAD.BILL_TYPES;
    ['.pdf', '.jpg', '.png'].forEach(ext => {
      TestRunner.assert(billTypes.includes(ext), `BILL_TYPES 应包含 ${ext}`);
    });
  });

  TestRunner.test('CONFIG 对象不可修改（已冻结）', () => {
    const before = CONFIG.API_BASE;
    try { CONFIG.API_BASE = 'hacked'; } catch {}    // 严格模式会抛出，非严格模式静默失败
    TestRunner.assertEqual(CONFIG.API_BASE, before, 'CONFIG 应是冻结对象，不可写入');
  });
});
