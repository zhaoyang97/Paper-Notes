/* Global search for paper-notes.
 * The header search is the primary search UI. It renders results in Material's
 * existing search panel, so users never need to open an empty /search/ page.
 */
(function () {
  "use strict";

  var input = document.querySelector(".md-search__input");
  if (!input) return;

  var form = input.closest(".md-search__form");
  var output = document.querySelector(".md-search-result");
  var list = output && output.querySelector(".md-search-result__list");
  var meta = output && output.querySelector(".md-search-result__meta");
  if (!form || !output || !list || !meta) return;

  var PAGE_SIZE = 12;
  var docs = null;
  var aliases = [];
  var loading = null;
  var timer = null;

  function esc(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function normalize(s) {
    return String(s || "")
      .toLowerCase()
      .replace(/[‐‑]/g, "-")
      .replace(/[^a-z0-9\u4e00-\u9fff]+/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function tokenize(s) {
    if (!s) return [];
    var parts = String(s).toLowerCase().split(/[^a-z0-9\u4e00-\u9fff]+/);
    var out = [];
    for (var i = 0; i < parts.length; i++) {
      var p = parts[i];
      if (!p) continue;
      if (/[\u4e00-\u9fff]/.test(p)) {
        for (var j = 0; j < p.length; j++) {
          var ch = p.charAt(j);
          if (ch >= "\u4e00" && ch <= "\u9fff") {
            out.push(ch);
            if (j + 1 < p.length) out.push(ch + p.charAt(j + 1));
          }
        }
      } else if (p.length >= 2) {
        out.push(p);
      }
    }
    return out;
  }

  function addTokens(map, text, weight) {
    var ts = tokenize(text);
    for (var i = 0; i < ts.length; i++) {
      if (!map[ts[i]] || map[ts[i]] < weight) map[ts[i]] = weight;
    }
  }

  function variants(text, mode) {
    var q = normalize(text);
    var out = [];
    var seen = Object.create(null);
    for (var i = 0; i < aliases.length; i++) {
      var group = aliases[i];
      if (group.modes && group.modes.indexOf(mode) < 0) continue;
      var terms = [group.canonical].concat(group.aliases || []);
      for (var j = 0; j < terms.length; j++) {
        if (q !== normalize(terms[j])) continue;
        for (var k = 0; k < terms.length; k++) {
          var key = normalize(terms[k]);
          if (!seen[key]) {
            seen[key] = true;
            out.push(terms[k]);
          }
        }
        break;
      }
    }
    return out;
  }

  function expand(text) {
    var expanded = typeof window.__pnExpand === "function" ? window.__pnExpand(text) : text;
    // Rough search expands translations and harmless surface variants only.
    // Abbreviation/full-name relations (JEPA -> Joint Embedding...) are exact-only.
    var terms = variants(text, "rough");
    if (terms.length) expanded += " " + terms.join(" ");
    return expanded;
  }

  function phraseInText(text, phrase) {
    var haystack = " " + normalize(text) + " ";
    var needle = normalize(phrase);
    if (!needle) return false;
    if (haystack.indexOf(" " + needle + " ") >= 0) return true;
    return /[\u4e00-\u9fff]/.test(needle) && haystack.indexOf(needle) >= 0;
  }

  function buildDocs(raw) {
    var freq = Object.create(null);
    for (var i = 0; i < raw.length; i++) {
      var tags = raw[i].tags || [];
      for (var j = 0; j < tags.length; j++) freq[tags[j]] = (freq[tags[j]] || 0) + 1;
    }
    var out = [];
    for (var r = 0; r < raw.length; r++) {
      var d = raw[r];
      var loc = d.location || "";
      var parts = loc.replace(/\/+$/, "").split("/");
      if (!loc || loc.indexOf("#") >= 0 || parts.length < 3 || !d.title) continue;
      var tags = d.tags || [];
      var title = String(d.title).trim();
      var kw = d.text || "";
      var roughMap = Object.create(null);
      addTokens(roughMap, title, 3);
      addTokens(roughMap, kw, 2);
      addTokens(roughMap, tags.filter(function (tag) { return freq[tag] >= 5; }).join(" "), 2);
      out.push({
        loc: loc,
        title: title,
        venue: parts[0],
        kw: kw,
        roughMap: roughMap,
        searchText: title + " " + kw + " " + tags.join(" "),
        titleNorm: normalize(title),
        kwNorm: normalize(kw),
        tlen: title.length
      });
    }
    return out;
  }

  function loadIndex(done) {
    if (docs) return done(null);
    if (loading) return loading.push(done);
    loading = [done];
    Promise.all([
      fetch("/tag_search_rules.json").then(function (r) {
        return r.ok ? r.json() : fetch("/tag_aliases.json").then(function (fallback) {
          return fallback.ok ? fallback.json() : { rules: [] };
        });
      }),
      fetch("/search/search_index.json").then(function (r) { return r.json(); })
    ]).then(function (data) {
      aliases = data[0].rules || [];
      docs = buildDocs(data[1].docs || []);
      var callbacks = loading;
      loading = null;
      callbacks.forEach(function (cb) { cb(null); });
    }).catch(function (err) {
      var callbacks = loading;
      loading = null;
      callbacks.forEach(function (cb) { cb(err); });
    });
  }

  function search(query, mode) {
    var groups = query.split("#").map(function (part) {
      var text = part.trim();
      var seen = Object.create(null);
      var toks = tokenize(expand(text)).filter(function (t) {
        if (seen[t]) return false;
        seen[t] = true;
        return true;
      });
      return { text: text, aliases: variants(text, "exact"), tokens: toks };
    }).filter(function (g) { return g.text && g.tokens.length; });
    if (!groups.length) return [];

    var hits = [];
    for (var i = 0; i < docs.length; i++) {
      var doc = docs[i];
      var score = 0;
      var ok = true;
      for (var g = 0; g < groups.length; g++) {
        var matched = 0;
        var groupScore = 0;
        if (mode === "exact") {
          var choices = groups[g].aliases.length ? groups[g].aliases : [groups[g].text];
          for (var v = 0; v < choices.length; v++) {
            if (phraseInText(doc.searchText, choices[v])) {
              matched = 1;
              groupScore = 10 + tokenize(choices[v]).length;
              break;
            }
          }
        } else {
          for (var t = 0; t < groups[g].tokens.length; t++) {
            var weight = doc.roughMap[groups[g].tokens[t]];
            if (weight) { matched++; groupScore += weight; }
          }
        }
        if (!matched) { ok = false; break; }
        score += groupScore * (0.7 + 0.3 * matched / groups[g].tokens.length);
      }
      if (!ok) continue;
      for (var b = 0; b < groups.length; b++) {
        var gn = normalize(groups[b].text);
        if (gn.length >= 2 && doc.titleNorm.indexOf(gn) >= 0) score += 6;
        else if (gn.length >= 2 && doc.kwNorm.indexOf(gn) >= 0) score += 2;
      }
      hits.push({ doc: doc, score: score });
    }
    hits.sort(function (a, b) { return b.score - a.score || a.doc.tlen - b.doc.tlen; });
    return hits;
  }

  function highlight(text, query) {
    var terms = query.split("#").map(function (p) { return normalize(p); }).filter(Boolean);
    var result = esc(text);
    terms.sort(function (a, b) { return b.length - a.length; });
    terms.forEach(function (term) {
      if (!term) return;
      var re = new RegExp("(" + term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig");
      result = result.replace(re, "<mark>$1</mark>");
    });
    return result;
  }

  function render() {
    var query = input.value.trim();
    var mode = modeSelect.value;
    if (!query) {
      meta.textContent = "选择搜索模式后输入关键词；多个词条用 # 分隔";
      list.innerHTML = "";
      return;
    }
    meta.textContent = "正在搜索…";
    list.innerHTML = "";
    loadIndex(function (err) {
      if (err) {
        meta.textContent = "搜索索引加载失败，请稍后重试";
        return;
      }
      var hits = search(query, mode);
      meta.textContent = hits.length ? "找到 " + hits.length + " 条结果" : "没有找到匹配的笔记";
      list.innerHTML = hits.slice(0, PAGE_SIZE).map(function (hit) {
        return '<li class="md-search-result__item pn-global-hit"><a href="/' +
          hit.doc.loc + '" class="md-search-result__link">' + highlight(hit.doc.title, query) +
          '</a><div class="pn-global-meta">' + esc(hit.doc.venue) +
          (hit.doc.kw ? " · " + esc(hit.doc.kw) : "") + "</div></li>";
      }).join("");
    });
  }

  var modeLabel = document.createElement("label");
  modeLabel.className = "pn-global-mode";
  modeLabel.textContent = "搜索模式";
  var modeSelect = document.createElement("select");
  modeSelect.setAttribute("aria-label", "搜索模式");
  modeSelect.innerHTML = '<option value="rough">粗略搜索</option><option value="exact">精细搜索</option>';
  modeLabel.appendChild(modeSelect);
  form.appendChild(modeLabel);

  var style = document.createElement("style");
  style.textContent =
    ".md-search__form{position:relative}" +
    ".pn-global-mode{display:none;position:absolute;left:1rem;top:3.1rem;z-index:2;align-items:center;gap:.35rem;padding:.3rem .45rem;border-radius:.3rem;background:var(--md-default-bg-color);box-shadow:0 2px 8px rgba(0,0,0,.16);font-size:.75rem;color:var(--md-default-fg-color--light)}" +
    ".md-search:focus-within .pn-global-mode{display:inline-flex}" +
    ".pn-global-mode select{font:inherit;color:inherit;background:var(--md-default-bg-color);border:1px solid var(--md-default-fg-color--lighter);border-radius:.25rem;padding:.15rem .25rem}" +
    ".pn-global-hit{padding:.4rem 1rem!important}" +
    ".pn-global-meta{margin-top:.15rem;font-size:.72rem;color:var(--md-default-fg-color--light);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}" +
    ".pn-global-hit mark{color:#e8590c;background:rgba(232,89,12,.14);font-weight:700}";
  document.head.appendChild(style);

  input.addEventListener("focus", function () { render(); }, true);
  input.addEventListener("input", function (e) {
    e.stopImmediatePropagation();
    clearTimeout(timer);
    timer = setTimeout(render, 120);
  }, true);
  document.addEventListener("keydown", function (e) {
    if (e.key !== "Enter" || e.target !== input) return;
    e.preventDefault();
    e.stopPropagation();
    var query = input.value.trim();
    if (!query) return;
    location.href = "/search/?q=" + encodeURIComponent(query) +
      (modeSelect.value === "exact" ? "&mode=exact" : "");
  }, true);
  form.addEventListener("submit", function (e) { e.preventDefault(); render(); });
  modeSelect.addEventListener("change", render);
})();
