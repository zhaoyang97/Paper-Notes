/* Anonymous, browser-local paper library.
 * No account, login, avatar or backend is involved. Data is kept per origin
 * in localStorage and can be exported by the user at any time.
 */
(function () {
  "use strict";

  var STORAGE_KEY = "papernotes-library-v1";
  var state = null;

  function load() {
    if (state) return state;
    try {
      state = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    } catch (e) {
      state = null;
    }
    if (!state || !Array.isArray(state.folders)) {
      state = { version: 1, folders: [] };
    }
    return state;
  }

  function save() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (e) {
      window.alert("本机收藏保存失败，可能是浏览器存储空间不足。");
    }
  }

  function id() {
    return "f-" + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  }

  function esc(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function folders() {
    return load().folders;
  }

  function ensureDefault() {
    var fs = folders();
    if (!fs.length) {
      fs.push({ id: id(), name: "我的收藏", createdAt: new Date().toISOString(), papers: [] });
      save();
    }
    return fs;
  }

  function currentPaper() {
    var main = document.querySelector("main, article");
    var h1 = document.querySelector("main h1, article h1, h1");
    var canonical = document.querySelector('link[rel="canonical"]');
    return {
      url: canonical && canonical.href ? canonical.href : location.href.split("#")[0],
      title: h1 ? h1.textContent.trim() : document.title.replace(/\s*[|｜].*$/, "").trim(),
      text: main ? main.textContent.replace(/\s+/g, " ").trim().slice(0, 500) : ""
    };
  }

  function addToFolder(folderId, paper) {
    var folder = folders().filter(function (f) { return f.id === folderId; })[0];
    if (!folder) return false;
    folder.papers = folder.papers || [];
    if (!folder.papers.some(function (p) { return p.url === paper.url; })) {
      folder.papers.push({ url: paper.url, title: paper.title, addedAt: new Date().toISOString() });
      save();
      return true;
    }
    return false;
  }

  function addFolder(name) {
    name = String(name || "").trim();
    if (!name) return null;
    var folder = { id: id(), name: name, createdAt: new Date().toISOString(), papers: [] };
    folders().push(folder);
    save();
    return folder;
  }

  function removeFolder(folderId) {
    state.folders = folders().filter(function (f) { return f.id !== folderId; });
    save();
  }

  function download(filename, content, type) {
    var blob = new Blob([content], { type: type || "text/plain;charset=utf-8" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }

  function arxivPdfUrl(root) {
    var links = (root || document).querySelectorAll('a[href*="arxiv.org/"]');
    for (var i = 0; i < links.length; i++) {
      try {
        var url = new URL(links[i].href, location.href);
        if (!/^(www\.)?arxiv\.org$/i.test(url.hostname)) continue;
        var match = url.pathname.match(/^\/(?:abs|pdf)\/([^/]+?)(?:\.pdf)?\/?$/i);
        if (match) return "https://arxiv.org/pdf/" + match[1] + ".pdf";
      } catch (e) {}
    }
    return null;
  }

  function downloadArxivPdf(paper, button, sourceDoc) {
    var pdfUrl = arxivPdfUrl(sourceDoc || document);
    if (!pdfUrl) {
      if (button) window.alert("当前论文没有找到 arXiv PDF 地址。");
      return Promise.reject(new Error("arXiv link not found"));
    }
    var originalText = button ? button.textContent : "";
    if (button) {
      button.disabled = true;
      button.textContent = "准备下载…";
    }
    return fetch(pdfUrl, { mode: "cors" })
      .then(function (response) {
        if (!response.ok) throw new Error("PDF request failed");
        return response.blob();
      })
      .then(function (blob) {
        var safeTitle = (paper.title || "paper").replace(/[\\/:*?"<>|]+/g, " ").trim().slice(0, 120) || "paper";
        var objectUrl = URL.createObjectURL(blob);
        var link = document.createElement("a");
        link.href = objectUrl;
        link.download = safeTitle + ".pdf";
        document.body.appendChild(link);
        link.click();
        setTimeout(function () { URL.revokeObjectURL(objectUrl); link.remove(); }, 1000);
      })
      .catch(function () {
        window.open(pdfUrl, "_blank", "noopener");
      })
      .finally(function () {
        if (button) {
          button.disabled = false;
          button.textContent = originalText;
        }
      });
  }

  function downloadStoredPaperPdf(paper, button) {
    return fetch(paper.url, { credentials: "same-origin" })
      .then(function (response) { return response.ok ? response.text() : ""; })
      .then(function (html) {
        var sourceDoc = new DOMParser().parseFromString(html, "text/html");
        return downloadArxivPdf(paper, button, sourceDoc);
      })
      .catch(function (error) {
        if (button && error.message !== "arXiv link not found") window.alert("无法读取论文页面，请稍后重试。");
        throw error;
      });
  }

  function pagePart(doc) {
    return doc.querySelector(".md-content, main, article") || doc.body;
  }

  function cleanPaperPart(doc, paper) {
    var part = pagePart(doc).cloneNode(true);
    var remove = part.querySelectorAll(
      "script, style, nav, header, footer, button, svg, form, " +
      ".headerlink, .md-content__button, .md-source-file, .md-meta, " +
      ".md-tags, .md-annotation, .md-search, .md-sidebar, .md-header, " +
      ".md-footer, .md-tabs, .md-jump-to"
    );
    for (var i = 0; i < remove.length; i++) remove[i].remove();
    var links = part.querySelectorAll("a[href]");
    for (var j = 0; j < links.length; j++) {
      try { links[j].href = new URL(links[j].getAttribute("href"), paper.url).href; } catch (e) {}
    }
    var images = part.querySelectorAll("img[src]");
    for (var k = 0; k < images.length; k++) {
      try { images[k].src = new URL(images[k].getAttribute("src"), paper.url).href; } catch (e) {}
      images[k].removeAttribute("loading");
      images[k].removeAttribute("decoding");
    }
    return part.outerHTML;
  }

  function fetchPaper(paper) {
    return fetch(paper.url, { credentials: "same-origin" })
      .then(function (r) { return r.ok ? r.text() : ""; })
      .then(function (html) {
        var doc = new DOMParser().parseFromString(html, "text/html");
        var part = pagePart(doc);
        var title = (doc.querySelector("main h1, article h1, h1") || {}).textContent || paper.title;
        return { url: paper.url, title: title.trim(), html: cleanPaperPart(doc, paper) };
      });
  }

  function htmlExport(papers, name) {
    var shell = "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><title>" +
      esc(name) + "</title><style>body{max-width:900px;margin:2rem auto;padding:0 1rem;font:16px/1.7 system-ui,sans-serif;color:#222}article{border-bottom:1px solid #ddd;padding:1rem 0 2rem}.pn-export-toc{margin:1.5rem 0 2rem;padding:1rem 1.5rem;background:#f7f5fb;border-radius:.5rem}.pn-export-toc li{margin:.25rem 0}a{color:#6750a4}img{max-width:100%;height:auto}code{background:#f3f3f3;padding:.1em .3em}</style></head><body><h1>" +
      esc(name) + "</h1><p>导出时间：" + new Date().toLocaleString() + "</p>";
    return Promise.all(papers.map(fetchPaper)).then(function (items) {
      var toc = "<nav class=\"pn-export-toc\"><h2>论文目录</h2><ol>" + items.map(function (item, index) {
        return "<li><a href=\"#paper-" + (index + 1) + "\">" + esc(item.title) + "</a></li>";
      }).join("") + "</ol></nav>";
      var body = toc + items.map(function (item, index) {
        return "<article id=\"paper-" + (index + 1) + "\"><h2><a href=\"" + esc(item.url) + "\">" + esc(item.title) + "</a></h2>" + item.html + "</article>";
      }).join("\n");
      download(name + ".html", shell + body + "</body></html>", "text/html;charset=utf-8");
    });
  }

  function injectStyle() {
    if (document.getElementById("pn-library-style")) return;
    var style = document.createElement("style");
    style.id = "pn-library-style";
    style.textContent =
      ".pn-library-link{display:inline-flex;align-items:center;align-self:center;height:2rem;margin:0 .45rem;font-size:.78rem;line-height:1;white-space:nowrap}" +
      ".pn-library-panel{position:fixed;z-index:100;right:1rem;top:4rem;width:min(22rem,calc(100vw - 2rem));padding:1rem;background:var(--md-default-bg-color);color:var(--md-default-fg-color);border:1px solid var(--md-default-fg-color--lightest);border-radius:.5rem;box-shadow:0 .4rem 1.2rem #0003}" +
      ".pn-library-panel h3{margin:0 0 .7rem;font-size:1rem}.pn-library-panel button,.pn-library-panel input,.pn-library-toolbar select{font:inherit}" +
      ".pn-library-folder{display:flex;align-items:center;gap:.4rem;width:100%;margin:.25rem 0;padding:.45rem .55rem;border:1px solid var(--md-default-fg-color--lightest);border-radius:.35rem;background:transparent;color:inherit;text-align:left;cursor:pointer}" +
      ".pn-library-folder:hover,.pn-library-action:hover{border-color:var(--md-accent-fg-color)}" +
      ".pn-library-new{display:flex;gap:.35rem;margin-top:.7rem}.pn-library-new input{min-width:0;flex:1;padding:.4rem}" +
      ".pn-library-action{padding:.35rem .6rem;border:1px solid var(--md-default-fg-color--lightest);border-radius:.35rem;background:transparent;color:inherit;cursor:pointer}" +
      ".pn-library-toolbar{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;margin:.7rem 0;padding:.6rem;border:1px solid var(--md-default-fg-color--lightest);border-radius:.4rem}" +
      ".pn-library-toolbar select{padding:.3rem;max-width:15rem}.pn-library-check{margin-right:.45rem}.pn-library-paper{display:flex;align-items:center;gap:.45rem;margin:.45rem 0;padding:.45rem 0;border-bottom:1px solid var(--md-default-fg-color--lightest)}.pn-library-paper>a{flex:1}.pn-library-pdf-row-button{font-size:.75rem;padding:.25rem .45rem}" +
      ".pn-category-paper-action{margin-left:.4rem;padding:.12rem .35rem;font-size:.72rem;vertical-align:middle}";
    document.head.appendChild(style);
  }

  function closePanel() {
    var p = document.querySelector(".pn-library-panel");
    if (p) p.remove();
  }

  function openPicker(papers) {
    closePanel();
    ensureDefault();
    var panel = document.createElement("div");
    panel.className = "pn-library-panel";
    panel.innerHTML = "<h3>添加到收藏夹</h3><div class=\"pn-library-folder-list\"></div><div class=\"pn-library-new\"><input placeholder=\"新建收藏夹\" aria-label=\"新建收藏夹\"><button class=\"pn-library-action\">新建</button></div>";
    var list = panel.querySelector(".pn-library-folder-list");
    folders().forEach(function (folder) {
      var b = document.createElement("button");
      b.className = "pn-library-folder";
      b.textContent = folder.name + "（" + (folder.papers || []).length + "）";
      b.addEventListener("click", function () {
        papers.forEach(function (paper) { addToFolder(folder.id, paper); });
        closePanel();
        window.alert("已添加 " + papers.length + " 篇论文到“" + folder.name + "”。");
        renderBatchState();
      });
      list.appendChild(b);
    });
    var input = panel.querySelector("input");
    panel.querySelector("button.pn-library-action").addEventListener("click", function () {
      var folder = addFolder(input.value);
      if (!folder) return;
      openPicker(papers);
    });
    document.body.appendChild(panel);
  }

  function paperFromLink(link) {
    return { url: new URL(link.href, location.href).href.split("#")[0], title: link.textContent.trim() };
  }

  function renderBatchState() {
    var results = document.getElementById("pn-search-results");
    if (!results) return;
    var items = results.querySelectorAll(".pn-hit");
    for (var i = 0; i < items.length; i++) {
      if (items[i].querySelector(".pn-library-check")) continue;
      var link = items[i].querySelector(".pn-hit-link");
      if (!link) continue;
      var check = document.createElement("input");
      check.type = "checkbox";
      check.className = "pn-library-check";
      check.setAttribute("aria-label", "选择论文");
      check.dataset.paper = JSON.stringify(paperFromLink(link));
      link.parentNode.insertBefore(check, link);
    }
    var root = results.parentNode;
    var toolbar = document.getElementById("pn-library-batch-toolbar");
    if (!toolbar) {
      toolbar = document.createElement("div");
      toolbar.id = "pn-library-batch-toolbar";
      toolbar.className = "pn-library-toolbar";
      toolbar.innerHTML = "<label><input type=\"checkbox\" class=\"pn-library-select-all\"> 全选当前结果</label><span>批量收藏：</span><button class=\"pn-library-action\">添加选中论文</button><span class=\"pn-library-batch-count\"></span>";
      root.insertBefore(toolbar, results);
      toolbar.querySelector("button").addEventListener("click", function () {
        var selected = Array.prototype.slice.call(results.querySelectorAll(".pn-library-check:checked")).map(function (c) { return JSON.parse(c.dataset.paper); });
        if (selected.length) openPicker(selected); else window.alert("请先选择论文。");
      });
      toolbar.querySelector(".pn-library-select-all").addEventListener("change", function (e) {
        var checks = results.querySelectorAll(".pn-library-check");
        for (var i = 0; i < checks.length; i++) checks[i].checked = e.target.checked;
        toolbar.querySelector(".pn-library-batch-count").textContent = e.target.checked && checks.length ? "已选 " + checks.length + " 篇" : "";
      });
      results.addEventListener("change", function () {
        var n = results.querySelectorAll(".pn-library-check:checked").length;
        toolbar.querySelector(".pn-library-batch-count").textContent = n ? "已选 " + n + " 篇" : "";
        var all = toolbar.querySelector(".pn-library-select-all");
        var total = results.querySelectorAll(".pn-library-check").length;
        all.checked = total > 0 && n === total;
        all.indeterminate = n > 0 && n < total;
      });
    }
    var selectAll = toolbar.querySelector(".pn-library-select-all");
    if (selectAll) {
      var total = results.querySelectorAll(".pn-library-check").length;
      var selected = results.querySelectorAll(".pn-library-check:checked").length;
      selectAll.checked = total > 0 && selected === total;
      selectAll.indeterminate = selected > 0 && selected < total;
    }
  }

  function isPaperLink(link) {
    try {
      var url = new URL(link.href, location.href);
      if (url.origin !== location.origin) return false;
      var parts = url.pathname.replace(/\/+$/, "").split("/").filter(Boolean);
      return parts.length >= 3 && parts[parts.length - 1].toLowerCase() !== "index" && /^[a-z]+\d{4}$/i.test(parts[0]);
    } catch (e) {
      return false;
    }
  }

  function renderCategoryBatch() {
    var content = document.querySelector(".md-content");
    var pathParts = location.pathname.replace(/\/+$/, "").split("/").filter(Boolean);
    if (!content || pathParts.length > 2 || document.getElementById("pn-search-results")) return;
    var seen = Object.create(null);
    var links = Array.prototype.slice.call(content.querySelectorAll("a[href]")).filter(function (link) {
      if (!isPaperLink(link)) return false;
      var url = new URL(link.href, location.href).href.split("#")[0];
      if (seen[url]) return false;
      seen[url] = true;
      return true;
    });
    if (!links.length || document.getElementById("pn-category-batch-toolbar")) return;

    links.forEach(function (link) {
      var paper = paperFromLink(link);
      var check = document.createElement("input");
      check.type = "checkbox";
      check.className = "pn-library-check pn-category-check";
      check.dataset.paper = JSON.stringify(paper);
      check.setAttribute("aria-label", "选择论文");
      link.parentNode.insertBefore(check, link);

      var add = document.createElement("button");
      add.type = "button";
      add.className = "pn-library-action pn-category-paper-action";
      add.textContent = "加入收藏";
      add.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        openPicker([paper]);
      });
      link.insertAdjacentElement("afterend", add);
    });

    var toolbar = document.createElement("div");
    toolbar.id = "pn-category-batch-toolbar";
    toolbar.className = "pn-library-toolbar";
    toolbar.innerHTML = "<label><input type=\"checkbox\" class=\"pn-category-select-all\"> 全选当前分类</label><button class=\"pn-library-action\">添加选中论文</button><span class=\"pn-library-batch-count\"></span>";
    var firstHeading = content.querySelector("h1, h2");
    if (firstHeading) firstHeading.insertAdjacentElement("afterend", toolbar);
    else content.insertBefore(toolbar, content.firstChild);

    var checks = function () { return content.querySelectorAll(".pn-category-check"); };
    var updateState = function () {
      var all = toolbar.querySelector(".pn-category-select-all");
      var total = checks().length;
      var selected = content.querySelectorAll(".pn-category-check:checked").length;
      all.checked = total > 0 && selected === total;
      all.indeterminate = selected > 0 && selected < total;
      toolbar.querySelector(".pn-library-batch-count").textContent = selected ? "已选 " + selected + " 篇" : "";
    };
    toolbar.querySelector(".pn-category-select-all").addEventListener("change", function (e) {
      var list = checks();
      for (var i = 0; i < list.length; i++) list[i].checked = e.target.checked;
      updateState();
    });
    content.addEventListener("change", function (e) {
      if (e.target.classList.contains("pn-category-check")) updateState();
    });
    toolbar.querySelector("button").addEventListener("click", function () {
      var selected = Array.prototype.slice.call(content.querySelectorAll(".pn-category-check:checked")).map(function (c) { return JSON.parse(c.dataset.paper); });
      if (selected.length) openPicker(selected); else window.alert("请先选择论文。");
    });
  }

  function injectPaperButton() {
    var pathParts = location.pathname.replace(/\/+$/, "").split("/").filter(Boolean);
    // Paper pages are /conference/area/paper-slug/; don't add paper actions
    // to the homepage, conference indexes or area indexes.
    if (pathParts.length < 3 || pathParts[pathParts.length - 1].toLowerCase() === "index") return;
    var paper = currentPaper();
    if (!paper.title || document.querySelector(".pn-library-paper-button")) return;
    var h1 = document.querySelector("main h1, article h1, h1");
    if (!h1) return;
    var bar = document.createElement("div");
    bar.className = "pn-library-toolbar pn-library-paper-toolbar";
    bar.innerHTML = "<button class=\"pn-library-action pn-library-paper-button\">☆ 加入收藏夹</button><button class=\"pn-library-action pn-library-export-button\">导出本文</button>";
    h1.insertAdjacentElement("afterend", bar);
    bar.querySelector(".pn-library-paper-button").addEventListener("click", function () { openPicker([paper]); });
    bar.querySelector(".pn-library-export-button").addEventListener("click", function () { htmlExport([paper], paper.title || "paper"); });
    if (arxivPdfUrl()) {
      var pdfButton = document.createElement("button");
      pdfButton.className = "pn-library-action pn-library-pdf-button";
      pdfButton.textContent = "下载 PDF";
      pdfButton.addEventListener("click", function () { downloadArxivPdf(paper, pdfButton); });
      bar.appendChild(pdfButton);
    }
  }

  function renderLibraryPage() {
    var root = document.getElementById("pn-library-root");
    if (!root) return;
    injectStyle();
    ensureDefault();
    function render() {
      var fs = folders();
      root.innerHTML = "<div class=\"pn-library-toolbar\"><button class=\"pn-library-action\" data-new>新建收藏夹</button><span>数据仅保存在当前浏览器</span></div><div class=\"pn-library-folders\"></div>";
      var container = root.querySelector(".pn-library-folders");
      fs.forEach(function (folder) {
        var section = document.createElement("section");
        section.innerHTML = "<h2>" + esc(folder.name) + " <small>（" + (folder.papers || []).length + "）</small></h2><div class=\"pn-library-toolbar\"><label><input type=\"checkbox\" data-select-all> 全选</label><button class=\"pn-library-action\" data-download>下载选中 PDF</button><button class=\"pn-library-action\" data-remove-selected>删除选中论文</button><button class=\"pn-library-action\" data-html>导出 HTML</button><button class=\"pn-library-action\" data-delete>删除收藏夹</button><span data-download-status></span></div><div class=\"pn-library-paper-list\"></div>";
        var list = section.querySelector(".pn-library-paper-list");
        (folder.papers || []).forEach(function (paper) {
          var row = document.createElement("div");
          row.className = "pn-library-paper";
          row.innerHTML = "<input type=\"checkbox\" data-paper-check aria-label=\"选择论文\"><a href=\"" + esc(paper.url) + "\">" + esc(paper.title || paper.url) + "</a><button type=\"button\" class=\"pn-library-action pn-library-pdf-row-button\">下载 PDF</button>";
          row.querySelector("[data-paper-check]").dataset.paper = JSON.stringify(paper);
          row.querySelector(".pn-library-pdf-row-button").addEventListener("click", function () {
            downloadStoredPaperPdf(paper, row.querySelector(".pn-library-pdf-row-button"));
          });
          list.appendChild(row);
        });
        var selectAll = section.querySelector("[data-select-all]");
        var status = section.querySelector("[data-download-status]");
        selectAll.addEventListener("change", function () {
          var checks = section.querySelectorAll("[data-paper-check]");
          for (var i = 0; i < checks.length; i++) checks[i].checked = selectAll.checked;
        });
        section.querySelector("[data-download]").addEventListener("click", function () {
          var selected = Array.prototype.slice.call(section.querySelectorAll("[data-paper-check]:checked")).map(function (check) {
            return JSON.parse(check.dataset.paper);
          });
          if (!selected.length) {
            window.alert("请先选择论文。");
            return;
          }
          var index = 0;
          var next = function () {
            if (index >= selected.length) {
              status.textContent = "已完成";
              return;
            }
            status.textContent = "正在下载 " + (index + 1) + "/" + selected.length;
            downloadStoredPaperPdf(selected[index], null)
              .catch(function () {})
              .finally(function () {
                index += 1;
                setTimeout(next, 900);
              });
          };
          next();
        });
        section.querySelector("[data-remove-selected]").addEventListener("click", function () {
          var selected = Array.prototype.slice.call(section.querySelectorAll("[data-paper-check]:checked")).map(function (check) {
            return JSON.parse(check.dataset.paper).url;
          });
          if (!selected.length) {
            window.alert("请先选择论文。");
            return;
          }
          if (!window.confirm("确定从“" + folder.name + "”删除选中的 " + selected.length + " 篇论文吗？")) return;
          folder.papers = (folder.papers || []).filter(function (paper) { return selected.indexOf(paper.url) < 0; });
          save();
          render();
        });
        section.querySelector("[data-html]").addEventListener("click", function () { htmlExport(folder.papers || [], folder.name); });
        section.querySelector("[data-delete]").addEventListener("click", function () { if (window.confirm("删除收藏夹“" + folder.name + "”？")) { removeFolder(folder.id); render(); } });
        container.appendChild(section);
      });
      root.querySelector("[data-new]").addEventListener("click", function () { var name = window.prompt("收藏夹名称"); if (addFolder(name)) render(); });
    }
    render();
  }

  function init() {
    injectStyle();
    if (!document.querySelector(".pn-library-link")) {
      var target = document.querySelector(".md-header__option") || document.querySelector(".md-header__inner");
      if (target) {
        var link = document.createElement("a");
        link.className = "pn-library-link";
        link.href = (location.pathname.indexOf("/search/") >= 0 ? "../" : "/") + "library/";
        link.textContent = "我的收藏";
        target.appendChild(link);
      }
    }
    injectPaperButton();
    renderBatchState();
    renderCategoryBatch();
    renderLibraryPage();
    var results = document.getElementById("pn-search-results");
    if (results && window.MutationObserver) new MutationObserver(renderBatchState).observe(results, { childList: true });
  }

  if (window.document$ && typeof window.document$.subscribe === "function") window.document$.subscribe(init);
  else document.addEventListener("DOMContentLoaded", init);
})();
