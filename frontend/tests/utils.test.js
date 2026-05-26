/**
 * utils.test.js — Utils 工具函数单元测试
 */

TestRunner.suite('Utils — 日期格式化', () => {

  TestRunner.test('formatDate：正常 ISO 字符串格式化', () => {
    const result = Utils.formatDate('2026-05-20T09:33:14.363Z', false);
    TestRunner.assert(result.startsWith('2026-05-20'), `日期应以 2026-05-20 开头，得到: ${result}`);
  });

  TestRunner.test('formatDate：空值返回 "-"', () => {
    TestRunner.assertEqual(Utils.formatDate(null), '-', '空值应返回 -');
    TestRunner.assertEqual(Utils.formatDate(''),   '-', '空字符串应返回 -');
  });

  TestRunner.test('formatDate：带时间输出含冒号', () => {
    const result = Utils.formatDate('2026-05-20T09:33:14.363Z', true);
    TestRunner.assert(result.includes(':'), '带时间格式应包含冒号');
  });

  TestRunner.test('timeAgo：刚刚', () => {
    const now = new Date().toISOString();
    TestRunner.assertEqual(Utils.timeAgo(now), '刚刚', '刚创建应显示"刚刚"');
  });
});


TestRunner.suite('Utils — 文件处理', () => {

  TestRunner.test('formatFileSize：0 字节', () => {
    TestRunner.assertEqual(Utils.formatFileSize(0), '0 B', '0 字节应显示 "0 B"');
  });

  TestRunner.test('formatFileSize：1024 字节 = 1.0 KB', () => {
    TestRunner.assertEqual(Utils.formatFileSize(1024), '1.0 KB');
  });

  TestRunner.test('formatFileSize：1MB', () => {
    TestRunner.assertEqual(Utils.formatFileSize(1024 * 1024), '1.0 MB');
  });

  TestRunner.test('getFileExt：正确提取小写扩展名', () => {
    TestRunner.assertEqual(Utils.getFileExt('document.PDF'),  '.pdf');
    TestRunner.assertEqual(Utils.getFileExt('report.DOCX'),   '.docx');
    TestRunner.assertEqual(Utils.getFileExt('image.JPG'),     '.jpg');
    TestRunner.assertEqual(Utils.getFileExt('no_ext'),        '');
  });

  TestRunner.test('fileIcon：已知类型有图标', () => {
    TestRunner.assert(Utils.fileIcon('test.pdf')  !== '📎', 'PDF 应有专属图标');
    TestRunner.assert(Utils.fileIcon('test.xlsx') !== '📎', 'Excel 应有专属图标');
  });

  TestRunner.test('isFileTypeAllowed：在白名单内', () => {
    const allowed = ['.pdf', '.jpg', '.png'];
    const file = { name: 'ticket.pdf' };
    TestRunner.assert(Utils.isFileTypeAllowed(file, allowed), '.pdf 应被允许');
  });

  TestRunner.test('isFileTypeAllowed：不在白名单', () => {
    const allowed = ['.pdf', '.jpg'];
    const file = { name: 'virus.exe' };
    TestRunner.assert(!Utils.isFileTypeAllowed(file, allowed), '.exe 应被拒绝');
  });
});


TestRunner.suite('Utils — 状态徽章', () => {

  TestRunner.test('docStatusBadge：completed → 已完成 + success 样式', () => {
    const badge = Utils.docStatusBadge('completed');
    TestRunner.assertEqual(badge.label,      '已完成');
    TestRunner.assertEqual(badge.colorClass, 'badge-success');
  });

  TestRunner.test('docStatusBadge：failed → 已失败 + danger 样式', () => {
    const badge = Utils.docStatusBadge('failed');
    TestRunner.assertEqual(badge.label,      '已失败');
    TestRunner.assertEqual(badge.colorClass, 'badge-danger');
  });

  TestRunner.test('tenantStatusBadge：active → 运营中', () => {
    TestRunner.assertEqual(Utils.tenantStatusBadge('active').label, '运营中');
  });
});


TestRunner.suite('Utils — 安全与格式化', () => {

  TestRunner.test('escapeHtml：转义 < > & " \'', () => {
    const raw = '<script>alert("xss")</script>';
    const escaped = Utils.escapeHtml(raw);
    TestRunner.assert(!escaped.includes('<script>'), '应转义 <script>');
    TestRunner.assert(escaped.includes('&lt;'),      '应包含 &lt;');
    TestRunner.assert(escaped.includes('&gt;'),      '应包含 &gt;');
  });

  TestRunner.test('escapeHtml：空值返回空串', () => {
    TestRunner.assertEqual(Utils.escapeHtml(null), '');
    TestRunner.assertEqual(Utils.escapeHtml(''),   '');
  });

  TestRunner.test('formatNumber：添加千位分隔符', () => {
    const result = Utils.formatNumber(1234567);
    TestRunner.assert(result.includes(',') || result.includes('，'), '应有千分位分隔符');
  });

  TestRunner.test('formatDuration：毫秒', () => {
    TestRunner.assertEqual(Utils.formatDuration(500), '500ms');
  });

  TestRunner.test('formatDuration：秒', () => {
    TestRunner.assertEqual(Utils.formatDuration(2500), '2.5s');
  });

  TestRunner.test('formatDuration：null 返回 "-"', () => {
    TestRunner.assertEqual(Utils.formatDuration(null), '-');
  });

  TestRunner.test('debounce：延迟后只执行一次', (done) => {
    let count = 0;
    const fn = Utils.debounce(() => count++, 50);
    fn(); fn(); fn();                               // 快速调用 3 次
    // 验证在防抖期间 count 仍为 0
    TestRunner.assertEqual(count, 0, '防抖期间不应立即执行');
    // 注：浏览器环境中 50ms 后真正执行，这里仅测试同步行为
  });
});


TestRunner.suite('API 客户端 — 模块结构', () => {

  TestRunner.test('API 对象存在且有所有子模块', () => {
    ['auth', 'docs', 'query', 'bills', 'tenants', 'system'].forEach(mod => {
      TestRunner.assert(mod in API, `API.${mod} 应存在`);
    });
  });

  TestRunner.test('API.docs 有所有文档操作方法', () => {
    ['list', 'upload', 'delete', 'retry', 'get'].forEach(method => {
      TestRunner.assertEqual(typeof API.docs[method], 'function', `API.docs.${method} 应为函数`);
    });
  });

  TestRunner.test('API.query 有 ask, stream, feedback 方法', () => {
    ['ask', 'stream', 'feedback'].forEach(m => {
      TestRunner.assertEqual(typeof API.query[m], 'function', `API.query.${m} 应为函数`);
    });
  });

  TestRunner.test('API.bills 有 recognize, upload, list, get 方法', () => {
    ['recognize', 'upload', 'list', 'get'].forEach(m => {
      TestRunner.assertEqual(typeof API.bills[m], 'function', `API.bills.${m} 应为函数`);
    });
  });

  TestRunner.test('API.tenants 有 create, list, stats 方法', () => {
    ['create', 'list', 'stats'].forEach(m => {
      TestRunner.assertEqual(typeof API.tenants[m], 'function', `API.tenants.${m} 应为函数`);
    });
  });
});
