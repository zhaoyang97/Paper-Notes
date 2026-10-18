---
title: 搜索结果
search:
  exclude: true
hide:
  - toc
  - navigation
---

<script>
if (!new URLSearchParams(location.search).get("q")) location.replace("../");
</script>

# 搜索结果

<div id="pn-search-root">
<div id="pn-search-box">
<input id="pn-search-input" type="search" autocomplete="off" autocapitalize="off" spellcheck="false" placeholder="搜索论文笔记…" aria-label="搜索论文笔记">
<label class="pn-search-mode pn-search-mode-open">搜索模式：<select id="pn-search-mode" aria-label="搜索模式"><option value="rough">粗略搜索</option><option value="exact">精细搜索</option></select></label>
<fieldset id="pn-search-conference-filter" aria-label="按会议筛选" disabled><legend>会议</legend><div id="pn-search-conferences"></div></fieldset>
</div>
<div id="pn-search-status"></div>
<ol id="pn-search-results"></ol>
</div>
