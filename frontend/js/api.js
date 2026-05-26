/**
 * api.js — HTTP API 客户端
 * 封装所有与后端的交互，统一处理认证头、错误、超时
 * 依赖：config.js, auth.js
 */

const API = (() => {
  // ── 内部工具 ────────────────────────────────────────────────────────────────
  /**
   * 构造带认证头的公共 headers
   * @returns {object}
   */
  function _authHeaders() {
    const token = Auth.getToken();
    return {
      'Authorization': `Bearer ${token}`,        // JWT Bearer 认证
      'Content-Type': 'application/json',         // 默认 JSON 请求体
    };
  }

  /**
   * 通用 fetch 包装：统一处理错误响应
   * @param {string} url
   * @param {RequestInit} options
   * @returns {Promise<any>} - 解析后的 JSON 数据
   * @throws {Error} - HTTP 错误或网络错误
   */
  async function _request(url, options = {}) {
    let resp;
    try {
      resp = await fetch(url, options);          // 发起请求
    } catch (e) {
      throw new Error('网络连接失败，请检查网络或服务状态');  // 网络层错误
    }

    if (resp.status === 401) {
      Auth.logout();                             // token 失效，强制重新登录
      throw new Error('登录已过期，请重新登录');
    }

    if (resp.status === 403) {
      throw new Error('权限不足，无法执行此操作');
    }

    if (resp.status === 429) {
      throw new Error('请求过于频繁，请稍后再试');
    }

    if (!resp.ok) {
      // 尝试解析后端返回的错误 JSON
      const errBody = await resp.json().catch(() => null);
      const detail = errBody?.detail || errBody?.error || `请求失败 (${resp.status})`;
      throw new Error(detail);
    }

    // 204 No Content 无响应体
    if (resp.status === 204) return null;

    return resp.json();                          // 解析 JSON 响应体
  }

  // ── 认证接口 ────────────────────────────────────────────────────────────────
  const auth = {
    /**
     * 注册新用户（管理员调用）
     * @param {{ username, email, password, is_admin }} data
     */
    register: (data) => _request(
      `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/auth/register`,
      { method: 'POST', headers: _authHeaders(), body: JSON.stringify(data) }
    ),
  };

  // ── 文档接口 ────────────────────────────────────────────────────────────────
  const docs = {
    /**
     * 获取文档列表（分页）
     * @param {{ page, page_size, status_filter }} params
     */
    list: ({ page = 1, page_size = CONFIG.PAGE_SIZE, status_filter } = {}) => {
      const qs = new URLSearchParams({ page, page_size });
      if (status_filter) qs.set('status_filter', status_filter);  // 可选状态过滤
      return _request(
        `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/documents/?${qs}`,
        { headers: _authHeaders() }
      );
    },

    /**
     * 上传单个文档（支持进度回调）
     * @param {File} file
     * @param {function} onProgress - (percent: number) => void
     */
    upload: (file, onProgress) => {
      return new Promise((resolve, reject) => {
        const formData = new FormData();
        formData.append('file', file);           // 文件字段名必须为 'file'

        const xhr = new XMLHttpRequest();

        // 进度事件：计算上传百分比
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable && onProgress) {
            onProgress(Math.round(e.loaded / e.total * 100));
          }
        };

        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(JSON.parse(xhr.responseText));
          } else {
            const err = JSON.parse(xhr.responseText || '{}');
            reject(new Error(err.detail || `上传失败 (${xhr.status})`));
          }
        };

        xhr.onerror = () => reject(new Error('网络错误，上传失败'));
        xhr.ontimeout = () => reject(new Error('上传超时'));
        xhr.timeout = CONFIG.UPLOAD_TIMEOUT;

        xhr.open('POST', `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/documents/upload`);
        xhr.setRequestHeader('Authorization', `Bearer ${Auth.getToken()}`);
        // 注意：不手动设置 Content-Type，让浏览器自动填写 multipart boundary
        xhr.send(formData);
      });
    },

    /**
     * 删除文档
     * @param {string} docId
     */
    delete: (docId) => _request(
      `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/documents/${docId}`,
      { method: 'DELETE', headers: _authHeaders() }
    ),

    /**
     * 重试解析失败的文档
     * @param {string} docId
     */
    retry: (docId) => _request(
      `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/documents/${docId}/retry`,
      { method: 'POST', headers: _authHeaders() }
    ),

    /**
     * 获取单个文档详情
     * @param {string} docId
     */
    get: (docId) => _request(
      `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/documents/${docId}`,
      { headers: _authHeaders() }
    ),

    /**
     * 获取文档统计（总分块数、今日查询数）
     */
    stats: () => _request(
      `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/documents/stats`,
      { headers: _authHeaders() }
    ),

    /**
     * 查询批次内所有文件的处理状态（轮询用）
     * @param {string} batchId
     * @returns {Promise<{batch_id, total, overall_status, is_cancelled, documents}>}
     */
    batchStatus: (batchId) => _request(
      `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/documents/batch/${encodeURIComponent(batchId)}`,
      { headers: _authHeaders() }
    ),

    /**
     * 重试批次内所有失败文件（返回新批次信息）
     * @param {string} batchId
     * @returns {Promise<BatchUploadResponse>}
     */
    batchRetry: (batchId) => _request(
      `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/documents/batch/${encodeURIComponent(batchId)}/retry`,
      { method: 'POST', headers: _authHeaders() }
    ),
  };

  // ── 查询（问答）接口 ─────────────────────────────────────────────────────────
  const query = {
    /**
     * 非流式问答
     * @param {{ query: string, top_k?: number }} data
     */
    ask: (data) => _request(
      `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/query/`,
      { method: 'POST', headers: _authHeaders(), body: JSON.stringify(data) }
    ),

    /**
     * 流式问答：返回 ReadableStream，调用方逐行处理 SSE 数据
     * @param {{ query: string, top_k?: number }} data
     * @returns {Promise<Response>} - 原始 fetch Response，供外层处理流
     */
    stream: async (data) => {
      const resp = await fetch(
        `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/query/stream`,
        {
          method: 'POST',
          headers: _authHeaders(),               // 带认证头
          body: JSON.stringify(data),
        }
      );
      if (resp.status === 401) { Auth.logout(); throw new Error('登录已过期'); }
      if (!resp.ok) {
        const e = await resp.json().catch(() => ({}));
        throw new Error(e.detail || '流式请求失败');
      }
      return resp;                               // 返回原始 Response 供 ReadableStream 读取
    },

    /**
     * 提交问答反馈
     * @param {{ query_id: string, score: number, comment?: string }} data
     */
    feedback: (data) => _request(
      `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/query/feedback`,
      { method: 'POST', headers: _authHeaders(), body: JSON.stringify(data) }
    ),

    /**
     * 票据咨询（multipart：票据文件 + 用户问题 → 意图路由 → Agent报告或RAG回答）
     * @param {File|null} billFile - 票据图片/PDF（可为 null，无文件时降级为 RAG）
     * @param {string} query - 用户问题
     * @param {string|null} billRecordId - 已入库票据ID（可选，与 billFile 二选一）
     * @returns {Promise<ConsultResponse>}
     */
    consult: (billFile, query, billRecordId = null) => {
      return new Promise((resolve, reject) => {
        const formData = new FormData();
        formData.append('query', query);                           // 必填：用户问题
        if (billFile) formData.append('bill_file', billFile);     // 可选：票据文件
        if (billRecordId) formData.append('bill_record_id', billRecordId);  // 可选：已入库票据

        const xhr = new XMLHttpRequest();
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(JSON.parse(xhr.responseText));
          } else {
            if (xhr.status === 401) { Auth.logout(); reject(new Error('登录已过期')); return; }
            const err = JSON.parse(xhr.responseText || '{}');
            reject(new Error(err.detail || `咨询请求失败 (${xhr.status})`));
          }
        };
        xhr.onerror = () => reject(new Error('网络错误，咨询请求失败'));
        xhr.timeout = 200000;                                      // 审核流程最长 180s，留余量
        xhr.ontimeout = () => reject(new Error('审核超时（超过200秒），请稍后重试'));
        xhr.open('POST', `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/query/consult`);
        xhr.setRequestHeader('Authorization', `Bearer ${Auth.getToken()}`);
        // 不设 Content-Type，由浏览器自动填写 multipart boundary
        xhr.send(formData);
      });
    },
  };

  // ── 票据接口 ────────────────────────────────────────────────────────────────
  const bills = {
    /**
     * 票据要素识别（用户端 + 按钮触发）
     * @param {File} file - 票据图片或 PDF
     * @returns {Promise<BillRecognitionResponse>}
     */
    recognize: (file) => {
      const formData = new FormData();
      formData.append('file', file);             // 字段名 'file'
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(JSON.parse(xhr.responseText));
          } else {
            const err = JSON.parse(xhr.responseText || '{}');
            reject(new Error(err.detail || '识别失败'));
          }
        };
        xhr.onerror = () => reject(new Error('网络错误'));
        xhr.timeout = CONFIG.UPLOAD_TIMEOUT;
        xhr.ontimeout = () => reject(new Error('识别超时'));
        xhr.open('POST', `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/bill-recognition/recognize`);
        xhr.setRequestHeader('Authorization', `Bearer ${Auth.getToken()}`);
        xhr.send(formData);
      });
    },

    /**
     * 上传票据并入库（触发生命周期管理）
     * @param {File} file
     */
    upload: (file) => {
      const formData = new FormData();
      formData.append('file', file);
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
          else {
            const err = JSON.parse(xhr.responseText || '{}');
            reject(new Error(err.detail || '入库失败'));
          }
        };
        xhr.onerror = () => reject(new Error('网络错误'));
        xhr.timeout = CONFIG.UPLOAD_TIMEOUT;
        xhr.open('POST', `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/bills/upload`);
        xhr.setRequestHeader('Authorization', `Bearer ${Auth.getToken()}`);
        xhr.send(formData);
      });
    },

    /**
     * 获取票据列表
     */
    list: ({ page = 1, page_size = CONFIG.PAGE_SIZE } = {}) => {
      const qs = new URLSearchParams({ page, page_size });
      return _request(
        `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/bills/?${qs}`,
        { headers: _authHeaders() }
      );
    },

    /**
     * 查询单张票据详情
     * @param {string} ticketNumber
     */
    get: (ticketNumber) => _request(
      `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/bills/${encodeURIComponent(ticketNumber)}`,
      { headers: _authHeaders() }
    ),
  };

  // ── 租户接口（仅管理员可用）──────────────────────────────────────────────────
  const tenants = {
    /**
     * 创建新租户
     * @param {{ name, code, license_no?, doc_quota?, qps_limit? }} data
     */
    create: (data) => _request(
      `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/tenants/`,
      { method: 'POST', headers: _authHeaders(), body: JSON.stringify(data) }
    ),

    /**
     * 获取所有租户列表
     */
    list: () => _request(
      `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/tenants/`,
      { headers: _authHeaders() }
    ),

    /**
     * 获取租户统计数据
     * @param {string} tenantId
     */
    stats: (tenantId) => _request(
      `${CONFIG.API_BASE}${CONFIG.API_PREFIX}/tenants/${tenantId}/stats`,
      { headers: _authHeaders() }
    ),
  };

  // ── 系统健康接口 ────────────────────────────────────────────────────────────
  const system = {
    /** 健康检查：{ status, version, services } */
    health: () => _request(`${CONFIG.API_BASE}/health`),
  };

  // ── 公开接口 ────────────────────────────────────────────────────────────────
  return { auth, docs, query, bills, tenants, system };
})();
