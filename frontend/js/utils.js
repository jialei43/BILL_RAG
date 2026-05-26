/**
 * utils.js — 通用工具函数库
 * 不依赖任何第三方库，提供日期格式化、文件处理、UI 辅助等工具
 */

const Utils = (() => {

  // ── 日期时间 ────────────────────────────────────────────────────────────────
  /**
   * 将 ISO 日期字符串格式化为易读格式
   * @param {string} isoStr - 如 "2026-05-20T09:33:14.363Z"
   * @param {boolean} withTime - 是否包含时间部分
   * @returns {string} - 如 "2026-05-20 09:33"
   */
  function formatDate(isoStr, withTime = true) {
    if (!isoStr) return '-';                     // 空值处理
    const d = new Date(isoStr);
    if (isNaN(d)) return isoStr;                 // 无法解析则原样返回
    const pad = (n) => String(n).padStart(2, '0');  // 补零函数
    const date = `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}`;
    if (!withTime) return date;
    const time = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
    return `${date} ${time}`;
  }

  /**
   * 将时间戳转为"X 分钟前"等相对时间描述
   * @param {string} isoStr
   * @returns {string}
   */
  function timeAgo(isoStr) {
    if (!isoStr) return '';
    const diff = (Date.now() - new Date(isoStr)) / 1000;  // 差值秒数
    if (diff < 60)    return '刚刚';
    if (diff < 3600)  return `${Math.floor(diff/60)} 分钟前`;
    if (diff < 86400) return `${Math.floor(diff/3600)} 小时前`;
    return `${Math.floor(diff/86400)} 天前`;
  }

  // ── 文件处理 ────────────────────────────────────────────────────────────────
  /**
   * 将字节数格式化为人类可读大小
   * @param {number} bytes
   * @returns {string} - 如 "1.5 MB"
   */
  function formatFileSize(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log2(bytes) / 10);  // 计算单位级别
    return `${(bytes / (1024 ** i)).toFixed(1)} ${units[i]}`;
  }

  /**
   * 根据文件名获取扩展名（小写）
   * @param {string} filename
   * @returns {string} - 如 ".pdf"
   */
  function getFileExt(filename) {
    const idx = filename.lastIndexOf('.');
    return idx >= 0 ? filename.slice(idx).toLowerCase() : '';
  }

  /**
   * 根据文件类型返回对应的 Emoji 图标
   * @param {string} filename
   * @returns {string}
   */
  function fileIcon(filename) {
    const ext = getFileExt(filename);
    const map = {
      '.pdf':  '📄', '.doc': '📝', '.docx': '📝',
      '.xlsx': '📊', '.xls': '📊',
      '.png':  '🖼️', '.jpg': '🖼️', '.jpeg': '🖼️',
      '.tiff': '🖼️', '.tif': '🖼️',
    };
    return map[ext] || '📎';                     // 默认回形针图标
  }

  /**
   * 验证文件类型是否在允许列表内
   * @param {File} file
   * @param {string[]} allowedExts - 如 ['.pdf', '.jpg']
   * @returns {boolean}
   */
  function isFileTypeAllowed(file, allowedExts) {
    const ext = getFileExt(file.name);
    return allowedExts.includes(ext);
  }

  // ── 状态映射 ────────────────────────────────────────────────────────────────
  /**
   * 文档处理状态 → 中文标签 + CSS 颜色类
   * @param {string} status - 'pending'|'processing'|'completed'|'failed'
   * @returns {{ label: string, colorClass: string }}
   */
  function docStatusBadge(status) {
    const map = {
      pending:    { label: '排队中', colorClass: 'badge-warning' },
      processing: { label: '处理中', colorClass: 'badge-info'    },
      completed:  { label: '已完成', colorClass: 'badge-success' },
      failed:     { label: '已失败', colorClass: 'badge-danger'  },
    };
    return map[status] || { label: status, colorClass: 'badge-default' };
  }

  /**
   * 租户状态 → 样式
   * @param {string} status - 'active'|'suspended'|'trial'
   */
  function tenantStatusBadge(status) {
    const map = {
      active:    { label: '运营中', colorClass: 'badge-success' },
      suspended: { label: '已暂停', colorClass: 'badge-danger'  },
      trial:     { label: '试用期', colorClass: 'badge-warning' },
    };
    return map[status] || { label: status, colorClass: 'badge-default' };
  }

  // ── Toast 通知 ───────────────────────────────────────────────────────────────
  /**
   * 弹出 toast 通知（依赖页面中存在 #toast-container 元素）
   * @param {string} message
   * @param {'success'|'error'|'info'|'warning'} type
   * @param {number} duration - 显示时长毫秒
   */
  function toast(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toast-container');
    if (!container) return;                      // 容器不存在则跳过

    const el = document.createElement('div');
    el.className = `toast toast-${type}`;        // CSS 类控制颜色
    el.innerHTML = `
      <span class="toast-icon">${{ success:'✓', error:'✕', info:'ℹ', warning:'⚠' }[type]}</span>
      <span class="toast-msg">${escapeHtml(message)}</span>
    `;
    container.appendChild(el);                  // 插入容器

    // 触发进入动画（下一帧）
    requestAnimationFrame(() => el.classList.add('toast-show'));

    // 指定时间后自动消失
    setTimeout(() => {
      el.classList.remove('toast-show');
      el.classList.add('toast-hide');
      el.addEventListener('transitionend', () => el.remove());  // 动画结束后移除 DOM
    }, duration);
  }

  // ── 安全工具 ────────────────────────────────────────────────────────────────
  /**
   * HTML 转义，防止 XSS
   * @param {string} str
   * @returns {string}
   */
  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // ── 防抖 ───────────────────────────────────────────────────────────────────
  /**
   * 函数防抖：延迟执行，在等待期间如再次调用则重新计时
   * 常用于搜索框输入事件
   * @param {function} fn
   * @param {number} delay - 毫秒
   * @returns {function}
   */
  function debounce(fn, delay) {
    let timer = null;
    return function (...args) {
      clearTimeout(timer);                       // 取消上一次定时
      timer = setTimeout(() => fn.apply(this, args), delay);
    };
  }

  // ── 数字格式化 ──────────────────────────────────────────────────────────────
  /**
   * 大数字添加千分位分隔符
   * @param {number} n
   * @returns {string}
   */
  function formatNumber(n) {
    if (n == null) return '-';
    return n.toLocaleString('zh-CN');
  }

  /**
   * 耗时格式化
   * @param {number} ms - 毫秒
   * @returns {string}
   */
  function formatDuration(ms) {
    if (ms == null) return '-';
    if (ms < 1000) return `${ms.toFixed(0)}ms`;      // 小于 1 秒显示毫秒
    return `${(ms/1000).toFixed(1)}s`;                // 大于 1 秒显示秒
  }

  /**
   * 复制文本到剪贴板
   * @param {string} text
   */
  async function copyToClipboard(text) {
    try {
      await navigator.clipboard.writeText(text);
      toast('已复制到剪贴板', 'success', 1500);
    } catch {
      toast('复制失败，请手动复制', 'error');
    }
  }

  // ── 公开接口 ────────────────────────────────────────────────────────────────
  return {
    formatDate, timeAgo,
    formatFileSize, getFileExt, fileIcon, isFileTypeAllowed,
    docStatusBadge, tenantStatusBadge,
    toast, escapeHtml, debounce,
    formatNumber, formatDuration, copyToClipboard,
  };
})();
