/**
 * config.js — 全局配置中心
 * 所有与环境相关的常量在这里统一管理，修改部署地址只需改此文件
 */

const CONFIG = {
  // ── API 基础地址 ────────────────────────────────────────────────────────────
  API_BASE: 'http://localhost:8002',           // 后端服务地址（生产环境修改此项）
  API_PREFIX: '/api/v1',                        // RESTful 版本前缀

  // ── 认证 ───────────────────────────────────────────────────────────────────
  TOKEN_KEY: 'bill_rag_token',                  // localStorage 中存储 JWT 的键名
  USER_KEY:  'bill_rag_user',                   // localStorage 中存储用户信息的键名
  TOKEN_EXPIRE_BUFFER: 5 * 60 * 1000,           // token 提前 5 分钟视为过期（毫秒）

  // ── 角色常量 ────────────────────────────────────────────────────────────────
  ROLE: {
    USER:  'user',                              // 普通用户：聊天 + 票据上传识别
    ADMIN: 'admin',                             // 管理员：文档管理 + 租户管理 + 监控
  },

  // ── 路由映射（相对于域名根路径）────────────────────────────────────────────
  ROUTES: {
    LOGIN:            '/frontend/index.html',
    CHAT:             '/frontend/chat.html',
    ADMIN_DASHBOARD:  '/frontend/admin/index.html',
    ADMIN_DOCUMENTS:  '/frontend/admin/documents.html',
    ADMIN_TENANTS:    '/frontend/admin/tenants.html',
    ADMIN_MONITOR:    '/frontend/admin/monitor.html',
  },

  // ── 文件上传 ────────────────────────────────────────────────────────────────
  UPLOAD: {
    MAX_SIZE: 50 * 1024 * 1024,                // 单文件最大 50MB
    DOC_TYPES: [                               // 文档管理支持的格式（管理员上传知识库）
      '.pdf', '.doc', '.docx',
      '.xlsx', '.xls',
      '.png', '.jpg', '.jpeg', '.tiff',
    ],
    BILL_TYPES: [                              // 票据识别支持的格式（用户端 + 按钮）
      '.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif',
    ],
  },

  // ── 分页 ───────────────────────────────────────────────────────────────────
  PAGE_SIZE: 20,                               // 默认每页条数

  // ── 请求超时 ────────────────────────────────────────────────────────────────
  REQUEST_TIMEOUT: 30 * 1000,                  // 普通请求超时 30s
  UPLOAD_TIMEOUT:  5 * 60 * 1000,             // 文件上传超时 5 分钟

  // ── 监控外部服务 ────────────────────────────────────────────────────────────
  GRAFANA_URL:    'http://localhost:3000',      // Grafana 地址（admin/monitor.html 嵌入）
  PROMETHEUS_URL: 'http://localhost:9090',      // Prometheus 地址
};

Object.freeze(CONFIG);        // 冻结，防止运行时意外修改
Object.freeze(CONFIG.ROLE);
Object.freeze(CONFIG.ROUTES);
Object.freeze(CONFIG.UPLOAD);
