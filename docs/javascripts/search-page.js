/* Full-page search results for paper-notes.
 *
 * Loaded on every page via `extra_javascript`, but only activates on the
 * dedicated /search/ page (detected by the #pn-search-root container). It
 * reuses the slimmed `search/search_index.json` we already ship (title +
 * keywords + tags after hooks/slim_search_index.py), so no lunr.js and no
 * Material search worker are involved — the scoring is a small, predictable,
 * CJK-aware token overlap that we fully control.
 *
 * Why a custom scorer instead of lunr: the corpus fields are short (title +
 * ~5 keywords), so weighted token coverage + a title-substring bonus ranks
 * better and more predictably than BM25's length-normalisation (which the
 * dropdown UI already has to fight with a client-side re-rank).
 */
(function () {
  "use strict";

  var LANG =
    (document.documentElement.lang || "zh").toLowerCase().indexOf("en") === 0
      ? "en"
      : "zh";

  var T = {
    zh: {
      placeholder: "搜索论文笔记…",
      loading: "正在加载搜索索引…",
      empty: "输入关键词开始搜索（匹配标题 + 关键词）",
      none: "没有找到匹配的笔记。",
      count: function (n, shown) {
        return shown < n
          ? "找到 " + n + " 条结果（显示前 " + shown + " 条）"
          : "找到 " + n + " 条结果";
      },
      deep: function (q) {
        return "在 Bing 上全文深搜「" + q + "」";
      },
      failed: "搜索索引加载失败，请稍后重试。"
    },
    en: {
      placeholder: "Search paper notes…",
      loading: "Loading search index…",
      empty: "Type a keyword to search (matches title + keywords)",
      none: "No matching notes found.",
      count: function (n, shown) {
        return shown < n
          ? n + " results (showing top " + shown + ")"
          : n + " results";
      },
      deep: function (q) {
        return "Full-text search \u201C" + q + "\u201D on Bing";
      },
      failed: "Failed to load the search index. Please try again later."
    }
  }[LANG];

  var MAX_RENDER = 100;
  var docsCache = null; // pre-tokenised docs, built once per page load
  var building = false;
  var waiters = [];

  function base() {
    var i = location.pathname.indexOf("/search/");
    return i >= 0 ? location.pathname.slice(0, i + 1) : "/";
  }

  function expand(q) {
    // Reuse the bilingual (中↔英) expansion dictionary from overrides/main.html
    // when available, so the full page matches the dropdown's recall.
    try {
      return typeof window.__pnExpand === "function" ? window.__pnExpand(q) : q;
    } catch (e) {
      return q;
    }
  }

  // CJK-aware tokenizer: Latin tokens (len >= 2) + CJK unigrams & bigrams.
  // Applied identically at index time and query time so the two are consistent.
  function tokenize(s) {
    if (!s) return [];
    s = s.toLowerCase().replace(/\$[^$]*\$/g, " ").replace(/\u200b/g, " ");
    var parts = s.split(/[^a-z0-9\u4e00-\u9fff]+/);
    var out = [];
    for (var i = 0; i < parts.length; i++) {
      var p = parts[i];
      if (!p) continue;
      if (/[\u4e00-\u9fff]/.test(p)) {
        var buf = "";
        for (var j = 0; j < p.length; j++) {
          var ch = p.charAt(j);
          if (ch >= "\u4e00" && ch <= "\u9fff") {
            if (buf) {
              if (buf.length >= 2) out.push(buf);
              buf = "";
            }
            out.push(ch); // unigram
            var nx = p.charAt(j + 1);
            if (nx >= "\u4e00" && nx <= "\u9fff") out.push(ch + nx); // bigram
          } else {
            buf += ch;
          }
        }
        if (buf && buf.length >= 2) out.push(buf);
      } else if (p.length >= 2) {
        out.push(p);
      }
    }
    return out;
  }

  function addTokens(map, text, weight) {
    var toks = tokenize(text);
    for (var i = 0; i < toks.length; i++) {
      var t = toks[i];
      if (!map[t] || map[t] < weight) map[t] = weight;
    }
  }

  function prettyArea(a) {
    return a.replace(/[_-]+/g, " ").replace(/\b\w/g, function (c) {
      return c.toUpperCase();
    });
  }

  function buildDocs(raw) {
    var docs = [];
    for (var i = 0; i < raw.length; i++) {
      var d = raw[i];
      var loc = d.location || "";
      if (!loc || loc.indexOf("#") >= 0) continue; // section anchors
      var segs = loc.replace(/\/+$/, "").split("/");
      if (segs.length < 3) continue; // skip conf-root / area-index pages
      var title = (d.title || "").trim();
      if (!title) continue;
      var kw = d.text || "";
      var kwList = kw ? kw.split(/[,\u3001]\s*/).filter(Boolean) : [];
      var tags = (d.tags || []).join(" ");
      var map = Object.create(null);
      addTokens(map, title, 3);
      addTokens(map, kw, 2);
      addTokens(map, tags, 2);
      docs.push({
        loc: loc,
        title: title,
        venue: segs[0],
        area: prettyArea(segs[1]),
        kw: kw,
        kwList: kwList,
        map: map,
        titleNorm: title.toLowerCase(),
        kwNorm: kw.toLowerCase(),
        tlen: title.length
      });
    }
    return docs;
  }

  function ensureIndex(cb) {
    if (docsCache) {
      cb(null);
      return;
    }
    waiters.push(cb);
    if (building) return;
    building = true;
    fetch(base() + "search/search_index.json")
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        docsCache = buildDocs(data.docs || []);
        flush(null);
      })
      .catch(function (err) {
        flush(err);
      });
    function flush(err) {
      building = false;
      var ws = waiters;
      waiters = [];
      for (var i = 0; i < ws.length; i++) ws[i](err);
    }
  }

  function runSearch(q) {
    var qn = q.toLowerCase().replace(/\s+/g, " ").trim();
    var qToks = [];
    var seen = Object.create(null);
    var raw = tokenize(expand(q));
    for (var i = 0; i < raw.length; i++) {
      if (!seen[raw[i]]) {
        seen[raw[i]] = 1;
        qToks.push(raw[i]);
      }
    }
    if (!qToks.length) return [];
    var hits = [];
    for (var d = 0; d < docsCache.length; d++) {
      var doc = docsCache[d];
      var s = 0,
        matched = 0;
      for (var k = 0; k < qToks.length; k++) {
        var w = doc.map[qToks[k]];
        if (w) {
          s += w;
          matched++;
        }
      }
      if (s <= 0) continue;
      s *= 0.4 + 0.6 * (matched / qToks.length); // favour matching all terms
      if (qn.length >= 2) {
        if (doc.titleNorm.indexOf(qn) >= 0) s += 12; // near-exact title match
        else if (doc.kwNorm.indexOf(qn) >= 0) s += 4;
      }
      hits.push({ doc: doc, s: s });
    }
    hits.sort(function (a, b) {
      return b.s - a.s || a.doc.tlen - b.doc.tlen;
    });
    return hits;
  }

  function esc(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  // Build the set of strings to highlight from a query: whole whitespace-
  // separated phrases (exact user input + bilingual expansions) plus the
  // search tokens (CJK bigrams + Latin words; single CJK chars are dropped to
  // avoid highlighting one stray character everywhere). Longest first so
  // "\u56fe\u50cf\u62fc\u8d34" wins over its "\u56fe\u50cf" sub-bigram during range marking.
  function highlightTerms(q) {
    var ex = expand(q);
    var set = Object.create(null);
    var phrases = ex.toLowerCase().split(/\s+/);
    for (var i = 0; i < phrases.length; i++) {
      var p = phrases[i].replace(/[^a-z0-9\u4e00-\u9fff]+/g, "");
      if (p.length >= 2) set[p] = 1;
    }
    var toks = tokenize(ex);
    for (var j = 0; j < toks.length; j++) {
      if (toks[j].length >= 2) set[toks[j]] = 1;
    }
    var out = [];
    for (var k in set) out.push(k);
    out.sort(function (a, b) {
      return b.length - a.length;
    });
    return out;
  }

  // HTML-escape `text` while wrapping any span that matches a highlight term
  // in <mark>. Matching is done on the raw (case-folded) string and the output
  // is escaped per-character, so HTML entities can never be corrupted.
  function hl(text, terms) {
    var n = text.length;
    if (!n || !terms.length) return esc(text);
    var low = text.toLowerCase();
    var mark = new Array(n);
    for (var i = 0; i < n; i++) mark[i] = false;
    for (var t = 0; t < terms.length; t++) {
      var term = terms[t];
      var tl = term.length;
      if (!tl) continue;
      var from = 0,
        idx;
      while ((idx = low.indexOf(term, from)) !== -1) {
        for (var m = idx; m < idx + tl; m++) mark[m] = true;
        from = idx + tl;
      }
    }
    var out = "",
      open = false;
    for (var c = 0; c < n; c++) {
      if (mark[c] && !open) {
        out += '<mark class="pn-mark">';
        open = true;
      } else if (!mark[c] && open) {
        out += "</mark>";
        open = false;
      }
      out += esc(text.charAt(c));
    }
    if (open) out += "</mark>";
    return out;
  }

  function deepHost() {
    if (/papernotes\.org$/.test(location.hostname)) return location.hostname;
    return LANG === "en" ? "en.papernotes.org" : "papernotes.org";
  }

  function deepLink(q) {
    var href =
      "https://www.bing.com/search?q=site:" +
      deepHost() +
      "+" +
      encodeURIComponent(q);
    return (
      '<a class="pn-deep" href="' +
      href +
      '" target="_blank" rel="noopener">' +
      esc(T.deep(q)) +
      "</a>"
    );
  }

  function getQ() {
    var m = /[?&]q=([^&]*)/.exec(location.search);
    return m ? decodeURIComponent(m[1].replace(/\+/g, " ")) : "";
  }

  function render(status, list, q) {
    q = (q || "").trim();
    if (!q) {
      status.textContent = T.empty;
      list.innerHTML = "";
      return;
    }
    status.textContent = T.loading;
    list.innerHTML = "";
    ensureIndex(function (err) {
      if (err) {
        status.textContent = T.failed;
        return;
      }
      var hits = runSearch(q);
      if (!hits.length) {
        status.innerHTML = esc(T.none) + " · " + deepLink(q);
        return;
      }
      var shown = Math.min(hits.length, MAX_RENDER);
      status.innerHTML = T.count(hits.length, shown) + " · " + deepLink(q);
      var terms = highlightTerms(q);
      var b = base();
      var html = "";
      for (var i = 0; i < shown; i++) {
        var doc = hits[i].doc;
        html +=
          '<li class="pn-hit">' +
          '<a class="pn-hit-link" href="' +
          b +
          doc.loc +
          '">' +
          hl(doc.title, terms) +
          "</a>" +
          '<div class="pn-hit-meta">' +
          esc(doc.venue) +
          " · " +
          esc(doc.area) +
          "</div>" +
          (doc.kwList && doc.kwList.length
            ? '<div class="pn-hit-kw">' +
              doc.kwList
                .map(function (k) {
                  return '<span class="pn-kw">' + hl(k, terms) + "</span>";
                })
                .join("") +
              "</div>"
            : "") +
          "</li>";
      }
      list.innerHTML = html;
    });
  }

  var styleInjected = false;
  function injectStyle() {
    if (styleInjected) return;
    styleInjected = true;
    var css =
      "#pn-search-root{max-width:46rem;margin:0 auto}" +
      "#pn-search-box{display:flex;gap:.5rem;margin:.4rem 0 1rem}" +
      "#pn-search-input{flex:1;font-size:1rem;padding:.6rem .8rem;border:1px solid var(--md-default-fg-color--lighter);border-radius:.4rem;background:var(--md-default-bg-color);color:var(--md-default-fg-color)}" +
      "#pn-search-input:focus{outline:none;border-color:var(--md-accent-fg-color)}" +
      "#pn-search-status{color:var(--md-default-fg-color--light);font-size:.8rem;margin-bottom:.4rem}" +
      "#pn-search-results{list-style:none;margin:0;padding:0}" +
      ".pn-hit{padding:.7rem 0;border-bottom:1px solid var(--md-default-fg-color--lightest)}" +
      ".pn-hit-link{font-weight:600;font-size:1rem;line-height:1.4}" +
      ".pn-hit-meta{font-size:.72rem;color:var(--md-default-fg-color--light);margin-top:.15rem}" +
      ".pn-hit-kw{margin-top:.35rem;display:flex;flex-wrap:wrap;gap:.3rem}" +
      ".pn-kw{font-size:.72rem;color:var(--md-default-fg-color--light);background:var(--md-default-fg-color--lightest);border-radius:.6rem;padding:.08rem .5rem;line-height:1.7;white-space:nowrap}" +
      // Matched-term highlight, mainstream-search style: a warm bold colour that
      // pops on both the purple title links and the grey keyword chips, plus a
      // faint tinted pill. All rules are scoped under `.md-typeset` so they beat
      // Material's own `.md-typeset mark` default (blue text / yellow box) which
      // otherwise wins on specificity. `.pn-kw .pn-mark` drops the pill inside
      // chips (the chip already has its own background) and just recolours text.
      ".md-typeset .pn-mark{color:#e8590c;background:rgba(232,89,12,.14);font-weight:700;border-radius:.2rem;padding:0 .12em}" +
      ".md-typeset .pn-kw .pn-mark{background:none;padding:0}" +
      '[data-md-color-scheme="slate"] .md-typeset .pn-mark{color:#ffa94d;background:rgba(255,169,77,.16)}' +
      '[data-md-color-scheme="slate"] .md-typeset .pn-kw .pn-mark{background:none}' +
      ".pn-deep{font-size:.8rem;white-space:nowrap}" +
      "body.pn-on-search .md-search{display:none!important}" +
      'body.pn-on-search label[for="__search"]{display:none!important}' +
      // Hiding the header search leaves Material thinking search is "open"
      // (#__search stays checked), which fires its
      // `[data-md-toggle=search]:checked ~ .md-header .md-header__option{max-width:0;opacity:0}`
      // rule and collapses the dark/light palette toggle. Force it back (our
      // !important beats Material's non-important collapse rule).
      "body.pn-on-search .md-header__option{max-width:none!important;opacity:1!important}";
    var st = document.createElement("style");
    st.textContent = css;
    document.head.appendChild(st);
  }

  function init() {
    var root = document.getElementById("pn-search-root");
    document.body.classList.toggle("pn-on-search", !!root);
    if (!root) return; // not the search page
    injectStyle();
    var input = document.getElementById("pn-search-input");
    var status = document.getElementById("pn-search-status");
    var list = document.getElementById("pn-search-results");
    if (!input || !status || !list) return;

    input.placeholder = T.placeholder;

    var initialQ = getQ();
    if (initialQ) input.value = initialQ;

    var timer = null;
    function update(immediate) {
      var q = input.value;
      var nq = q.trim();
      var url = base() + "search/" + (nq ? "?q=" + encodeURIComponent(nq) : "");
      try {
        history.replaceState(null, "", url);
      } catch (e) {}
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      if (immediate) {
        render(status, list, q);
      } else {
        timer = setTimeout(function () {
          render(status, list, q);
        }, 180);
      }
    }

    input.addEventListener("input", function () {
      update(false);
    });
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        update(true);
      }
    });

    render(status, list, initialQ);
    try {
      input.focus();
    } catch (e) {}
  }

  // Material instant navigation replaces <main> without a full reload, so hook
  // its document$ observable (emits on first load and after each instant nav).
  // Fall back to DOMContentLoaded when Material's bundle is unavailable.
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(init);
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
